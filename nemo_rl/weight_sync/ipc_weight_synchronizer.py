# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""IPC (ZMQ) weight synchronizer for colocated vLLM generation.

Handles weight transfer between a colocated policy and vLLM generation
backend using ZMQ IPC sockets and CUDA IPC handles. This is the primary
transport for colocated vLLM deployments.

Lifecycle per sync:
  1. policy.offload_before_refit()       -- free GPU for weight staging
  2. generation.prepare_for_generation(tags=["weights"])  -- allocate buffers
  3. policy.stream_weights_via_ipc_zmq() -- send weights via ZMQ
     generation.update_weights_via_ipc_zmq() -- receive weights
  4. policy.offload_after_refit()        -- restore optimizer state
  5. generation.prepare_for_generation(tags=["kv_cache"]) -- rebuild KV cache
"""

import os
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import (
    DraftApplyRequest,
    WeightSyncSelection,
    WeightSynchronizer,
)


class IPCWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using ZMQ IPC for colocated vLLM deployments.

    Both the policy and generation workers run on the same GPUs. Weights
    are transferred via CUDA IPC handles over ZMQ sockets, avoiding
    any network overhead.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface.
        generation: Generation object implementing GenerationInterface
            (concretely a VllmGeneration instance).
        refit_buffer_size_gb: Fixed buffer size in GB for weight staging.
            If None, buffer size is computed dynamically from free GPU memory.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        refit_buffer_size_gb: Optional[float | int] = None,
    ):
        self._policy = policy
        self._generation = generation
        self._refit_buffer_size_gb = refit_buffer_size_gb
        self._stale = True

    def sync_weights(
        self,
        *,
        selection: WeightSyncSelection = WeightSyncSelection(),
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
        draft_apply_request: DraftApplyRequest | None = None,
    ) -> dict[str, object]:
        self.validate_selection(selection)
        if draft_apply_request is not None and not selection.draft:
            raise ValueError("target-only weight sync cannot produce a draft receipt")
        self._stale = True
        if draft_apply_request is not None:
            draft_apply_request.receipt()
        self._policy.offload_before_refit()
        self._generation.prepare_for_generation(tags=["weights"])

        sync_succeeded = False
        receipt: dict[str, object] = {"successful": False}
        try:
            timer_context = (
                timer.time("prepare_for_generation/transfer_and_update_weights")
                if timer is not None
                else nullcontext()
            )
            with timer_context:
                buffer_size_bytes = self._compute_buffer_size()

                policy_kwargs: dict[str, Any] = {
                    "buffer_size_bytes": buffer_size_bytes,
                    "kv_scales": kv_scales,
                }
                generation_kwargs: dict[str, Any] = {}
                if not selection.draft:
                    # Policy serialization gains this keyword in the policy
                    # integration task. Preserve the deployed full-sync call
                    # shape until then.
                    policy_kwargs["selection"] = selection
                    generation_kwargs["selection"] = selection
                futures_train = self._policy.stream_weights_via_ipc_zmq(**policy_kwargs)
                futures_inference = self._generation.update_weights_via_ipc_zmq(
                    **generation_kwargs
                )

                ray.get(futures_train)
                results = ray.get(futures_inference)
                receiver_results = [result for result in results if result is not None]
                update_success = bool(receiver_results) and all(
                    result is True for result in receiver_results
                )

                if not update_success:
                    raise RuntimeError(
                        "Weight transfer failed during IPC/ZMQ sync. "
                        "This often indicates an issue with cuda-ipc or the vLLM worker."
                    )
                receipt = {"successful": True}
                if draft_apply_request is not None:
                    receipt["draft_apply_receipt"] = draft_apply_request.receipt()
            sync_succeeded = True
        finally:
            self._policy.offload_after_refit()
            self._generation.prepare_for_generation(tags=["kv_cache"])
            self._stale = not sync_succeeded

        return receipt

    @property
    def supports_component_selection(self) -> bool:
        return self._generation.cfg.get("backend") == "vllm"

    @property
    def supports_draft_apply_receipts(self) -> bool:
        return self.supports_component_selection

    @property
    def is_stale(self) -> bool:
        return self._stale

    def init_communicator(self) -> None:
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        pass

    def _compute_buffer_size(self) -> int:
        if self._refit_buffer_size_gb is not None:
            if self._refit_buffer_size_gb <= 0:
                raise ValueError("refit_buffer_size_gb must be > 0")
            return int(self._refit_buffer_size_gb * (1024**3))

        memory_ratio_raw = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.3")
        try:
            memory_ratio = float(memory_ratio_raw)
        except ValueError as exc:
            raise ValueError(
                f"NRL_REFIT_BUFFER_MEMORY_RATIO must be a valid float, got {memory_ratio_raw!r}"
            ) from exc
        if memory_ratio <= 0:
            raise ValueError(
                f"NRL_REFIT_BUFFER_MEMORY_RATIO must be > 0, got {memory_ratio}"
            )
        return int(self._policy.get_free_memory_bytes() * memory_ratio)
