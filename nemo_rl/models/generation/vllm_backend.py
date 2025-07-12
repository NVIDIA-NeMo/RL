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
from typing import Any

import torch

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
    )


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

    def update_weights_from_ipc_handles(self, device_ipc_handles):
        """Update weights from IPC handles.

        Args:
            device_ipc_handles (dict): parameter IPC handles for local device.

        Returns:
            bool: True if weights were successfully updated.
        """
        try:
            is_tensor_packed = device_ipc_handles[0]
            if is_tensor_packed:
                _, all_handles, tensor_metadata = device_ipc_handles
            else:
                _, name_and_handle_list = device_ipc_handles

            device_id = self.device.index
            weights = []

            if is_tensor_packed:
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
                for key, (shape, dtype, offset, size) in tensor_metadata.items():
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

    def update_weights_from_collective(self, info: dict[str, Any]) -> bool:
        """Update the model weights from collective communication."""
        try:
            for name, (shape, dtype) in info.items():
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                self.model_update_group.broadcast(weight, src=0)
                self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True
