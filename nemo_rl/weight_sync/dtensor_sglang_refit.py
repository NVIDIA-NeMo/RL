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

"""Driver-side SGLang refit drivers for the DTensor policy backend.

These run in the driver process, which is synced without a training-backend
extra, so they cannot live beside code that imports ``megatron.bridge`` or
``nemo_automodel`` at module scope. They only drive the ``policy`` and
``policy_generation`` facades.
"""

from typing import Any

import ray


def refit_sglang_colocated(
    *,
    policy: Any,
    policy_generation: Any,
    buffer_size_bytes: int,
) -> bool:
    """Refit colocated SGLang engines from the FSDP/DTensor policy.

    Lifecycle: connect (when new engines appeared), pause + KV
    invalidation, send HF tensor buckets via
    Ray IPC, post-process, continue. Mirrors the Megatron colocated driver;
    the FSDP path is BF16-only.
    """

    (
        rollout_engines,
        _rollout_engine_lock,
        num_new_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
    ) = policy_generation.get_updatable_engines_and_lock()

    if num_new_engines > 0:
        policy.connect_sglang_rollout_engines(
            engine_gpu_counts=engine_gpu_counts,
            engine_gpu_offsets=engine_gpu_offsets,
        )
        policy_generation.clear_updatable_num_new_engines()
        assert policy_generation.num_new_engines == 0, (
            "clear_updatable_num_new_engines did not zero num_new_engines"
        )

    # Pause with the configured mode, but only invalidate the KV cache when
    # the mode actually drops generation state. "in_place" leaves the engine
    # paused without dropping its KV cache, so flushing would clobber the
    # still-valid in-place state.
    pause_mode = policy_generation.pause_generation_mode
    policy_generation.pause_generation(mode=pause_mode)
    if pause_mode != "in_place":
        policy_generation.invalidate_kv_cache()

    policy_generation.begin_weight_update()
    try:
        futures = policy.update_weights_to_sglang_colocated(
            rollout_engines=rollout_engines,
            buffer_size_bytes=buffer_size_bytes,
        )
        ray.get(futures)
    finally:
        # Close the session and resume on every path, so a failed refit
        # leaves the engine usable instead of wedged in the update state.
        policy_generation.end_weight_update()
        policy_generation.continue_generation()
    return True


def refit_sglang_distributed(
    *,
    policy: Any,  # noqa: ARG001 — accepted for dispatch parity
    policy_generation: Any,  # noqa: ARG001
    buffer_size_bytes: int,  # noqa: ARG001
) -> bool:
    """SGLang disaggregate broadcast is not currently supported for FSDP.

    Per the design, only the Megatron backend implements the distributed
    refit path (it depends on AutoBridge restoring full HF tensors on
    trainer rank 0). FSDP non-colocated refits should keep using the
    legacy ``broadcast_weights_for_collective`` flow with a non-SGLang
    generation backend.
    """
    raise NotImplementedError(
        "SGLang weight_transfer_mode='broadcast' is currently only supported "
        "for the Megatron policy backend."
    )
