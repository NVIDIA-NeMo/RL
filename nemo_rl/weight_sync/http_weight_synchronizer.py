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
  1. policy.finish_training(OPTIMIZER_ONLY)  -- free optimizer for weight staging
  2. generation.prepare_for_generation()     -- allocate weight buffers
  3. generation.invalidate_kv_cache()        -- clear stale KV cache
  4. policy.stream_weights_via_http()        -- push weights via HTTP
  5. policy.finish_training(FULL)            -- exit training phase (offload model)
  6. generation.prepare_for_generation()     -- rebuild KV cache
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.models.policy.interfaces import OffloadMode
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class HTTPWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using HTTP for colocated SGLang deployments.

    Both the policy and generation workers run on the same GPUs. Weights
    are streamed to SGLang servers via their HTTP weight-update API.

    Args:
        policy: Policy object implementing PolicyTrainerInterface.
        generation: SGLangGeneration instance exposing get_sglang_url_to_gpu_uuids().
    """

    def __init__(self, policy: Any, generation: Any):
        self._policy = policy
        self._generation = generation
        self._stale = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._policy.finish_training(offload_mode=OffloadMode.OPTIMIZER_ONLY)
        self._generation.prepare_for_generation(tags=["weights"])

        sync_succeeded = False
        try:
            timer_context = (
                timer.time("prepare_for_generation/transfer_and_update_weights")
                if timer is not None
                else nullcontext()
            )
            with timer_context:
                sglang_url_to_gpu_uuids = self._generation.get_sglang_url_to_gpu_uuids()

                flush_success = self._generation.invalidate_kv_cache()
                if not flush_success:
                    print("SGLang KV cache invalidation failed before weight update. ")

                futures_train = self._policy.stream_weights_via_http(
                    sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
                )
                ray.get(futures_train)
            sync_succeeded = True
        finally:
            self._policy.finish_training(offload_mode=OffloadMode.FULL)
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
