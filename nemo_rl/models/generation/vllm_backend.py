# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any, Iterable, Optional

import torch

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
    )


def _patch_gemma3_mm():
    """Patch gemma3_mm.py to support new HF multimodal format (post transformers v4.52).

    Patch taken from:https://github.com/vllm-project/vllm/pull/19151/files#diff-5890909300e4e6c3160444e4587ec3fd80498bb83f598b22ce81337f75992b06
    """
    from packaging.version import Version as PkgVersion

    assert PkgVersion(vllm.__version__) < PkgVersion("0.9.2"), (
        f"You are using vllm version {vllm.__version__}. "
        "Please remove this patch (_patch_gemma3_mm in nemo_rl/models/generation/vllm_backend.py) "
        "since it is included in vllm>=0.9.2."
    )

    from vllm.logger import init_logger
    from vllm.model_executor.models import gemma3_mm
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        WeightsMapper,
    )

    logger = init_logger("gemma3_mm_patch")

    gemma3_mm.Gemma3ForConditionalGeneration.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    gemma3_mm.Gemma3ForConditionalGeneration.load_weights = load_weights
    logger.info("Successfully patched gemma3_mm.py in vllm_backend.")


_patch_gemma3_mm()


class VllmInternalWorkerExtension:
    def init_collective(
        self, rank_prefix: int, ip: str, port: int, world_size: int
    ) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        rank = rank_prefix + local_rank + 1  # 1 is the head node of the train cluster

        # Temporary fix for vllm==0.9.0 which overrides the NCCL_CUMEM_ENABLE to 0 and causes
        # https://github.com/NVIDIA-NeMo/RL/issues/564. This can be removed after it is upgraded to vllm>=0.9.1rc1.
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group = PyNcclCommunicator(pg, device=self.device)

    def report_device_id(self) -> str:
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def prepare_refit_info(
        self, state_dict_info: Optional[dict[str, Any]] = None
    ) -> None:
        """Prepare the info for refit.

        DtensorPolicyWorker:
            colocated inference: state_dict_info is None
            non-colocated inference: state_dict_info is a dict of {tensor_name: (shape, dtype)}

        MegatronPolicyWorker:
            colocated inference: state_dict_info is a dict of {tensor_name: (shape, dtype, numel)}
            non-colocated inference: not implemented yet
        """
        self.state_dict_info = state_dict_info

    def update_weights_from_global_ipc_handles(self, global_device_ipc_handles):
        """Update weights from global IPC handles.

        Args:
            global_device_ipc_handles (dict): Dictionary mapping device UUIDs to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated.
        """
        device_uuid = self.report_device_id()
        local_device_ipc_handles = global_device_ipc_handles[device_uuid]
        return self.update_weights_from_local_ipc_handles(local_device_ipc_handles)

    def update_weights_from_local_ipc_handles(self, local_device_ipc_handles):
        """Update weights from local IPC handles.

        Args:
            local_device_ipc_handles (dict): parameter IPC handles for local device.

        Returns:
            bool: True if weights were successfully updated.
        """
        try:
            is_tensor_packed = local_device_ipc_handles[0]
            if is_tensor_packed:
                _, all_handles, tensor_metadata = local_device_ipc_handles
            else:
                _, name_and_handle_list = local_device_ipc_handles

            device_id = self.device.index
            weights = []

            if is_tensor_packed:
                assert self.state_dict_info is not None, (
                    "state_dict_info is not prepared. "
                    "Please call prepare_refit_info when initializing the worker."
                )

                # Extract packed tensor from IPC handle
                dtype_to_packed_tensor = {}
                for dtype, tensor_handle in all_handles:
                    func, args = tensor_handle
                    list_args = list(args)
                    list_args[6] = device_id
                    tensor = func(*list_args)
                    dtype_to_packed_tensor[dtype] = tensor

                # Unpack tensor to weights. Here we only return a view of the tensor to avoid
                # using extra memory.
                for key, metadata in tensor_metadata.items():
                    # dtype for the 1st and 2nd steps may be different (e.g. e_score_correction_bias)
                    if isinstance(metadata, tuple):
                        # use dtype of current step
                        offset, dtype = metadata
                        shape, _, size = self.state_dict_info[key]
                        # update record
                        self.state_dict_info[key] = (shape, dtype, size)
                    else:
                        offset = metadata
                        shape, dtype, size = self.state_dict_info[key]
                    tensor = dtype_to_packed_tensor[dtype][offset : offset + size].view(
                        *shape
                    )
                    weights.append((key, tensor))
            else:
                # Process each handle to get the tensor
                for name, handle in name_and_handle_list:
                    func, args = handle
                    list_args = list(args)
                    list_args[6] = device_id
                    tensor = func(*list_args)
                    weights.append((name, tensor))

            # Load weights into the model
            self.model_runner.model.load_weights(weights=weights)
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_ipc_handles: {e}"
            )
            return False

    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        try:
            for name, (shape, dtype) in self.state_dict_info.items():
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                self.model_update_group.broadcast(weight, src=0)
                self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True
