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
"""Driver-side SGLang weight-update dispatch for the FSDP/DTensor backend.

These helpers are pure orchestration: they run in the driver process and only
call facade methods on ``policy`` / ``policy_generation``. They live here,
apart from ``dtensor_policy_worker_v2``, because that module imports heavy
training-backend internals (``nemo_automodel``, torch distributed APIs) at
module scope which the driver venv is not guaranteed to provide -- and because
the Megatron sibling module must stay importable without the ``mcore`` extra
(``mcore`` and ``sglang`` are declared as conflicting extras in
``pyproject.toml``). Keep this module free of worker-module imports at module
scope; use lazy in-function imports for anything heavier.
"""

import sys
from typing import Any

import ray


# ---------------------------------------------------------------------------
# Driver-side SGLang weight-update dispatch (FSDP backend)
# ---------------------------------------------------------------------------
def refit_sglang_colocated(
    *,
    policy: Any,
    policy_generation: Any,
    buffer_size_bytes: int,
) -> bool:
    """Refit colocated SGLang engines from the FSDP/DTensor policy.

    Lifecycle: optional fault-tolerance recover, connect (when new /
    recovered engines), pause + KV invalidation, send HF tensor buckets via
    Ray IPC, post-process, continue. Mirrors the Megatron colocated driver;
    the FSDP path is BF16-only.
    """
    from nemo_rl.models.policy.utils import (
        cancel_ray_refs,
        fetch_updatable_engines_with_recover,
        invalidate_sglang_kv_cache_for_refit,
    )

    monitor_refit_lease = None
    generation_pause_attempted = False
    generation_continue_succeeded = False
    transfer_futures = []
    transfer_started = False
    unsafe_engine_state = False
    try:
        (
            rollout_engines,
            _rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            monitor_refit_lease,
        ) = fetch_updatable_engines_with_recover(policy_generation)

        if num_new_engines > 0:
            policy.connect_sglang_rollout_engines(
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            policy_generation.clear_updatable_num_new_engines()
            assert policy_generation.num_new_engines == 0, (
                "clear_updatable_num_new_engines did not zero num_new_engines"
            )

        pause_mode = policy_generation.pause_generation_mode
        generation_pause_attempted = True
        policy_generation.pause_generation(mode=pause_mode)
        invalidate_sglang_kv_cache_for_refit(policy_generation, pause_mode)
        transfer_started = True
        transfer_futures = policy.update_weights_to_sglang_colocated(
            rollout_engines=rollout_engines,
            buffer_size_bytes=buffer_size_bytes,
        )
        ray.get(transfer_futures)
        policy_generation.post_process_weights()
    except BaseException as exc:
        unsafe_engine_state = transfer_started
        cancel_ray_refs(transfer_futures)
        if unsafe_engine_state:
            try:
                quarantine_confirmed = policy_generation.quarantine_all_engines()
            except BaseException as quarantine_exc:
                quarantine_confirmed = False
                exc.add_note(
                    "SGLang engine quarantine raised while handling the partial "
                    f"colocated weight update: {quarantine_exc!r}."
                )
            if not quarantine_confirmed:
                exc.add_note(
                    "Clean termination of every SGLang engine process could not "
                    "be confirmed. All actor slots remain quarantined."
                )
            exc.add_note(
                "The colocated SGLang weight stream may have been applied only "
                "partially. Generation remains paused, the refit health-monitor "
                "lease is retained, and this run must fail."
            )
        raise
    finally:
        active_error = sys.exception()
        if generation_pause_attempted and not unsafe_engine_state:
            try:
                policy_generation.continue_generation()
            except BaseException as cleanup_exc:
                if active_error is None:
                    raise
                active_error.add_note(
                    "Generation resume after SGLang refit failure also failed: "
                    f"{cleanup_exc!r}. Health monitoring remains suspended."
                )
            else:
                generation_continue_succeeded = True
        if (
            not unsafe_engine_state
            and monitor_refit_lease is not None
            and (not generation_pause_attempted or generation_continue_succeeded)
        ):
            policy_generation.health_monitoring_release_refit(monitor_refit_lease)
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
