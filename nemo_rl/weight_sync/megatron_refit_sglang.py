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
"""Driver-side SGLang weight-update dispatch for the Megatron backend.

These helpers are pure orchestration: they run in the driver process and only
call facade methods on ``policy`` / ``policy_generation``. They live here,
apart from ``megatron_policy_worker``, because that module imports
``megatron.bridge`` at module scope and the driver venv is built without the
``mcore`` extra. ``mcore`` and ``sglang`` are declared as conflicting extras
(see ``pyproject.toml``), so no single venv can ever provide both -- importing
the worker module in the driver to reach these functions raises
``ModuleNotFoundError``. Keep this module free of megatron / transformer_engine
/ worker-module imports at module scope; use lazy in-function imports for
anything heavier.
"""

import sys
from typing import Any

import ray

from nemo_rl.models.generation.sglang.utils.refit_deadline import (
    SGLangRefitDeadline,
    SGLangRefitTimeoutError,
)

_BEST_EFFORT_SGLANG_REFIT_CLEANUP_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Driver-side SGLang weight-update dispatch (Megatron backend)
# ---------------------------------------------------------------------------
def refit_sglang_colocated(
    *,
    policy: Any,
    policy_generation: Any,
    buffer_size_bytes: int,
) -> bool:
    """Refit colocated SGLang engines from the Megatron policy.

    Lifecycle: optional fault-tolerance recover, connect (when new /
    recovered engines), pause + flush, send HF tensor buckets via Ray
    IPC, post-process, continue.
    """
    from nemo_rl.models.policy.utils import (
        cancel_ray_refs,
        fetch_updatable_engines_with_recover,
        get_sglang_quantization_cfg,
        invalidate_sglang_kv_cache_for_refit,
    )

    sglang_quant = get_sglang_quantization_cfg(policy_generation)
    target_precision = sglang_quant["scheme"]
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
        # Per-worker actor method is now synchronous (per-chunk ray.get +
        # lifetime-safe IPC handled inside send_hf_buckets_via_ipc_actor_impl),
        # but the policy-group dispatch still returns one Ray future per
        # worker; we await those here to wait for all trainer ranks.
        transfer_started = True
        transfer_futures = policy.update_weights_to_sglang_colocated(
            rollout_engines=rollout_engines,
            buffer_size_bytes=buffer_size_bytes,
            target_precision=target_precision,
            sglang_quantization_cfg=sglang_quant,
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
    policy: Any,
    policy_generation: Any,
    buffer_size_bytes: int,
) -> bool:
    """Broadcast Megatron-restored HF tensors to disaggregate SGLang via NCCL.

    Trainer rank 0 owns the SGLang weight-update group; non-rank-0 ranks still
    walk the AutoBridge collective inside ``update_weights_to_sglang_distributed``
    but do not broadcast. Includes optional fault-tolerance recover prelude.
    """
    from nemo_rl.models.policy.utils import (
        cancel_ray_refs,
        fetch_updatable_engines_with_recover,
        get_sglang_quantization_cfg,
    )

    sglang_quant = get_sglang_quantization_cfg(policy_generation)
    target_precision = sglang_quant["scheme"]
    deadline = SGLangRefitDeadline(policy_generation.refit_timeout_s)
    monitor_refit_lease = None
    generation_pause_attempted = False
    generation_continue_succeeded = False
    transfer_futures = []
    communicator_touched = False
    transfer_started = False
    unsafe_engine_state = False
    try:
        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            _engine_gpu_offsets,
            monitor_refit_lease,
        ) = fetch_updatable_engines_with_recover(
            policy_generation,
            deadline=deadline,
        )

        # Every refit verifies the communicator state. The worker reuses the
        # existing group when the same engines are still healthy and rebuilds
        # it only when absent, changed, or marked poisoned.
        communicator_touched = True
        policy.connect_sglang_rollout_engines_distributed(
            rollout_engines=rollout_engines,
            engine_gpu_counts=engine_gpu_counts,
            timeout_s=deadline.remaining(
                "connecting the SGLang weight-update communicator"
            ),
        )

        if num_new_engines > 0:
            policy_generation.clear_updatable_num_new_engines()
            assert policy_generation.num_new_engines == 0, (
                "clear_updatable_num_new_engines did not zero num_new_engines"
            )

        # ``in_place`` preserves valid generation state, so cache invalidation
        # is intentionally skipped for that mode.
        pause_mode = policy_generation.pause_generation_mode
        generation_pause_attempted = True
        policy_generation.pause_generation(mode=pause_mode, deadline=deadline)
        if pause_mode != "in_place" and not policy_generation.invalidate_kv_cache(
            deadline=deadline
        ):
            raise RuntimeError("SGLang KV-cache invalidation failed before refit")

        # From this point onward, any error can leave a prefix of the new
        # weights installed in-place on one or more engines. The pinned SGLang
        # server cannot cancel an update after the HTTP client times out.
        transfer_started = True
        transfer_futures = policy.update_weights_to_sglang_distributed(
            rollout_engines=rollout_engines,
            rollout_engine_lock=rollout_engine_lock,
            buffer_size_bytes=buffer_size_bytes,
            timeout_s=deadline.remaining("dispatching SGLang weight transfer"),
            target_precision=target_precision,
            sglang_quantization_cfg=sglang_quant,
        )
        deadline.ray_get(
            transfer_futures,
            stage="waiting for SGLang weight transfer",
            cancel_on_error=True,
        )
        policy_generation.post_process_weights(deadline=deadline)
    except BaseException as exc:
        cancel_ray_refs(transfer_futures)
        cleanup_timeout_s = max(
            deadline.remaining_or_zero(),
            _BEST_EFFORT_SGLANG_REFIT_CLEANUP_TIMEOUT_S,
        )
        communicator_cleanup_confirmed = not communicator_touched
        if communicator_touched:
            cleanup_deadline = SGLangRefitDeadline(cleanup_timeout_s)
            try:
                cleanup_refs = policy.abort_sglang_rollout_engines_distributed(
                    timeout_s=cleanup_deadline.remaining(
                        "dispatching SGLang communicator cleanup"
                    )
                )
                cleanup_deadline.ray_get(
                    cleanup_refs,
                    stage="waiting for SGLang communicator cleanup",
                    cancel_on_error=False,
                )
            except BaseException as cleanup_exc:
                exc.add_note(
                    "SGLang communicator cleanup did not complete before "
                    f"returning the refit failure: {cleanup_exc!r}. A later "
                    "refit is fail-closed until cleanup is confirmed."
                )
            else:
                communicator_cleanup_confirmed = True

        unsafe_engine_state = transfer_started or not communicator_cleanup_confirmed
        if unsafe_engine_state:
            try:
                quarantine_confirmed = policy_generation.quarantine_all_engines(
                    timeout_s=cleanup_timeout_s
                )
            except BaseException as quarantine_exc:
                quarantine_confirmed = False
                exc.add_note(
                    "SGLang engine quarantine raised while handling an unsafe "
                    f"refit state: {quarantine_exc!r}."
                )
            if not quarantine_confirmed:
                exc.add_note(
                    "Clean termination of every SGLang engine process could not "
                    "be confirmed. All actor slots remain quarantined."
                )
            if transfer_started:
                exc.add_note(
                    "The SGLang weight stream may have been applied only partially. "
                    "Generation remains paused, the refit health-monitor lease is "
                    "retained, and this run must fail rather than resume those "
                    "engines."
                )
            else:
                exc.add_note(
                    "SGLang communicator bootstrap cleanup was not confirmed. "
                    "The engines are quarantined and this run must fail rather "
                    "than reuse uncertain engine control state."
                )
        raise
    finally:
        active_error = sys.exception()
        completion_error = None
        if generation_pause_attempted and not unsafe_engine_state:
            try:
                policy_generation.continue_generation(
                    deadline=deadline,
                    best_effort=True,
                )
            except BaseException as cleanup_exc:
                if active_error is None:
                    raise
                active_error.add_note(
                    "Generation resume after SGLang refit failure also failed: "
                    f"{cleanup_exc!r}. Health monitoring remains suspended."
                )
            else:
                generation_continue_succeeded = True
                if active_error is None and deadline.remaining_or_zero() <= 0:
                    completion_error = SGLangRefitTimeoutError(
                        "SGLang refit deadline expired while safely resuming generation"
                    )

        if (
            not unsafe_engine_state
            and monitor_refit_lease is not None
            and (not generation_pause_attempted or generation_continue_succeeded)
        ):
            try:
                policy_generation.health_monitoring_release_refit(monitor_refit_lease)
            except BaseException as cleanup_exc:
                if active_error is None:
                    raise
                active_error.add_note(
                    f"Restoring SGLang health monitoring also failed: {cleanup_exc!r}"
                )
        if completion_error is not None:
            raise completion_error
    return True
