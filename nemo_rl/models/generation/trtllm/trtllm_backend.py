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

"""TRT-LLM WorkerExtension for NCCL-based weight synchronisation.

This extension is injected into TRT-LLM's RayGPUWorker via the
``ray_worker_extension_cls`` parameter on ``tensorrt_llm.LLM``.  It follows
the same pattern as ``VllmInternalWorkerExtension`` (NCCL broadcast via
nemo_rl's ``packed_broadcast_consumer``) but targets the TRT-LLM internal
model / model_loader API.

TRT-LLM's built-in ``WorkerExtension.update_weights()`` uses CUDA IPC
handles.  This custom extension uses NCCL instead, matching the nemo-rl
non-colocated weight-update path.
"""

from typing import Any

import torch

from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension

from nemo_rl.utils.packed_tensor import packed_broadcast_consumer


class NcclExtension(WorkerExtension):
    """NCCL-based weight update extension for TRT-LLM Ray workers.

    Attributes set by TRT-LLM's mixin injection (from ``RayGPUWorker``):
        self.engine    – ``PyExecutor`` instance
        self.device_id – int GPU ordinal
    """

    # ------------------------------------------------------------------ #
    #  Collective initialisation (called once during setup)
    # ------------------------------------------------------------------ #

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        # Use nemo-rl's own StatelessProcessGroup (backed by nccl4py) instead
        # of vllm.distributed.* — TrtllmGenerationWorker runs under the
        # container's system python where vllm isn't installed, but nccl4py
        # is in nemo-rl core deps. The exposed `.broadcast(tensor, src=...)`
        # method matches what packed_broadcast_consumer calls.
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rank = train_world_size + rank_prefix + local_rank

        pg = StatelessProcessGroup(
            master_address=ip, port=port, rank=rank, world_size=world_size,
        )
        pg.init_nccl_communicator(device=self.device_id)
        self.model_update_group = pg

    # ------------------------------------------------------------------ #
    #  Refit metadata (weight name → (shape, dtype) mapping)
    # ------------------------------------------------------------------ #

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        self.state_dict_info = state_dict_info

    # ------------------------------------------------------------------ #
    #  NCCL weight receive + reload
    # ------------------------------------------------------------------ #

    @control_action_decorator
    def update_weights_from_collective(self) -> bool:
        """Receive weights via NCCL broadcast, then update model parameters.

        packed_broadcast_consumer uses double-buffered NCCL transfer across
        multiple CUDA streams. A full device sync is required before reading
        the received tensors on the default stream to avoid a data race.

        post_unpack_func is called with List[Tuple[str, Tensor]] per chunk;
        TRT-LLM's model_loader.reload expects a dict-like (weight_mapper
        does weights.items() / weights[name]), so we convert per chunk.
        """
        assert hasattr(self, "state_dict_info") and self.state_dict_info is not None, (
            "state_dict_info not set — call prepare_refit_info first"
        )
        model_engine = self.engine.model_engine
        model = model_engine.model

        def load_model_weight_func(weight_list):
            # packed_broadcast_consumer runs NCCL broadcast on a non-default
            # stream (streams[buffer_idx]); model_loader.reload internally
            # spawns a ThreadPoolExecutor whose worker threads each use their
            # own per-thread default stream and won't see the broadcast's
            # writes without an explicit sync. Without this sync, worker
            # threads read partially-written tensors → corrupted refit →
            # KL divergence blow-up between generation and policy.
            torch.cuda.current_stream().synchronize()
            model_engine.model_loader.reload(
                model, dict(weight_list), allow_partial_loading=True,
            )

        try:
            for module in model.modules():
                if hasattr(module, "pre_reload_weights") and not getattr(module, "_weights_removed", False):
                    module.pre_reload_weights()
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )
            for module in model.modules():
                if hasattr(module, "process_weights_after_loading") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.process_weights_after_loading()
                if hasattr(module, "post_load_weights") and not getattr(module, "_weights_removed", False):
                    module.post_load_weights()
            torch.cuda.current_stream().synchronize()
            self.engine.reset_prefix_cache()
        except Exception as e:
            print(
                f"Error in NcclExtension.update_weights_from_collective: {e}"
            )
            import traceback
            traceback.print_exc()
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def report_device_id(self) -> str:
        from tensorrt_llm._torch.utils import get_device_uuid
        return get_device_uuid(self.device_id)
