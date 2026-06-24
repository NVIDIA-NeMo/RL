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

"""HTTP weight synchronizer for colocated SGLang generation.

Handles weight transfer between a colocated policy and SGLang generation
backend using HTTP streaming. SGLang exposes an HTTP endpoint for weight
updates, so the policy streams weights directly to SGLang servers.

Lifecycle per sync:
  1. policy.offload_before_refit()       -- free GPU for weight staging
  2. generation.prepare_for_generation(tags=["weights"])  -- allocate buffers
  3. policy.stream_weights_via_http()    -- push weights via HTTP
  4. policy.offload_after_refit()        -- restore optimizer state
  5. generation.prepare_for_generation(tags=["kv_cache"]) -- rebuild KV cache
"""

import os
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class HTTPWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using HTTP for colocated SGLang deployments.

    Both the policy and generation workers run on the same GPUs. Weights
    are streamed to SGLang servers via their HTTP weight-update API.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface.
        generation: SGLangGeneration instance exposing get_rollout_engine_urls().
        refit_buffer_size_gb: Fixed buffer size in GB for weight staging.
            If None, buffer size is computed dynamically from free GPU memory.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        refit_buffer_size_gb: Optional[int] = None,
    ):
        self._policy = policy
        self._generation = generation
        self._refit_buffer_size_gb = refit_buffer_size_gb
        self._stale = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._policy.offload_before_refit()
        self._generation.prepare_for_generation(tags=["weights"])

        sync_succeeded = False
        try:
            timer_context = (
                timer.time("prepare_for_generation/transfer_and_update_weights")
                if timer is not None
                else nullcontext()
            )
            with timer_context:
                buffer_size_bytes = self._compute_buffer_size()
                futures_train = self._policy.stream_weights_via_http(
                    rollout_engine_urls=self._generation.get_rollout_engine_urls(),
                    buffer_size_bytes=buffer_size_bytes,
                )
                ray.get(futures_train)
            sync_succeeded = True
        finally:
            self._policy.offload_after_refit()
            self._generation.prepare_for_generation(tags=["kv_cache"])

        self._stale = not sync_succeeded

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        pass

    def _compute_buffer_size(self) -> int:
        if self._refit_buffer_size_gb is not None:
            if self._refit_buffer_size_gb <= 0:
                raise ValueError("refit_buffer_size_gb must be > 0")
            return self._refit_buffer_size_gb * (1024**3)

        memory_ratio_raw = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.3")
        try:
            memory_ratio = float(memory_ratio_raw)
        except ValueError as exc:
            raise ValueError(
                f"NRL_REFIT_BUFFER_MEMORY_RATIO must be a valid float, got {memory_ratio_raw!r}"
            ) from exc
        if memory_ratio <= 0:
            raise ValueError("NRL_REFIT_BUFFER_MEMORY_RATIO must be > 0")

        return int(self._policy.get_free_memory_bytes() * memory_ratio)
