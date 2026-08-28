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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import ray

from nemo_rl.models.generation.constants import SGLANG_BACKEND
from nemo_rl.models.generation.interfaces import CheckpointEngineConfig
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

_MEBIBYTE = 1024 * 1024


def _flatten_metadata(results: list[Any]) -> list[Any]:
    return [
        item
        for result in results
        for item in (result if isinstance(result, list) else [result])
    ]


def _sort_ranked_metadata(metadata: list[Any]) -> list[Any]:
    if all(isinstance(item, dict) and "rank" in item for item in metadata):
        return sorted(metadata, key=lambda item: item["rank"])
    return metadata


def _ordered_generation_metadata(generation_results: list[Any]) -> list[Any]:
    """Order generation metadata by global rollout rank.

    Each result belongs to one generation engine or data-parallel group.
    Engine-local ranks may be unique only within a group, so sort each group
    before concatenating them in engine order.
    """
    metadata: list[Any] = []
    for group_result in generation_results:
        group_metadata = (
            group_result if isinstance(group_result, list) else [group_result]
        )
        metadata.extend(_sort_ranked_metadata(group_metadata))
    return metadata


@dataclass
class CheckpointEngineWeightSynchronizer(WeightSynchronizer):
    """Coordinate checkpoint-engine setup and policy-to-rollout transfers."""

    _policy: Any
    _generation: Any
    _checkpoint_engine_config: CheckpointEngineConfig
    _stale: bool = True
    _checkpoint_engine_ready: bool = False
    _bucket_size_bytes: int | None = None
    _terminal_error: BaseException | None = None

    def init_communicator(self) -> None:
        self._raise_if_terminal()
        # SGLang's checkpoint-engine path builds its topology below and does not
        # consume the legacy refit metadata.  Gathering it would needlessly
        # materialize every sharded policy tensor on every training rank.
        if not self._is_sglang():
            self._generation.prepare_refit_info(self._policy.prepare_refit_info())
        self._ensure_ready_and_consume_count()

    def _set_terminal(self, exc: BaseException) -> None:
        if self._terminal_error is None:
            self._terminal_error = exc

    def _raise_if_terminal(self) -> None:
        # A failed rebind or a failure after a transfer started leaves NIXL
        # state (and possibly the served model) in an unknown condition; the
        # latch guarantees no later sync can issue RPCs over it.
        if self._terminal_error is not None:
            raise RuntimeError(
                "Checkpoint-engine synchronizer is in a terminal error state "
                "from a previous refit; restart the job. Original error: "
                f"{self._terminal_error!r}"
            ) from self._terminal_error

    def _use_fault_tolerance(self) -> bool:
        return bool(
            self._generation.sglang_cfg["sglang_cfg"].get("use_fault_tolerance")
        )

    def _ensure_ready_and_consume_count(self) -> None:
        """(Re)initialize the communicator; consume SGLang's new-engine count.

        ``_start_engines`` reports both the startup fleet and every recovered
        cohort through ``num_new_engines`` — consuming it only after a
        successful setup keeps a failed setup retryable and stops the first
        ordinary refit from being misclassified as crash recovery.
        """
        needs_init = not self._checkpoint_engine_ready
        try:
            self._ensure_checkpoint_engine_ready()
        except BaseException as exc:
            # NIXL prepare()/add_remote_agent() are not transactional: a retry
            # over their partial state silently skips or double-registers, so a
            # failed (re)bind is terminal until that is fixed upstream.
            self._set_terminal(exc)
            raise
        if needs_init and self._is_sglang():
            self._generation.clear_updatable_num_new_engines()

    def _sglang_recover_and_rebind(self) -> None:
        """Restart dead engines and rebind the paired NIXL fabric to them."""
        if self._use_fault_tolerance():
            from nemo_rl.models.generation.sglang.fault_tolerance import (
                RecoveryRollbackError,
            )

            try:
                # Always probe (it pauses the monitor and finds dead slots);
                # a plain failure here rolled the cohort back inside
                # ``_recover`` and is retryable on the next sync.
                self._generation.recover_updatable_engines()
            except RecoveryRollbackError as exc:
                self._set_terminal(exc)
                raise
            (_, _, num_new_engines, _, _) = (
                self._generation.get_updatable_engines_and_lock()
            )
            if num_new_engines > 0:
                # Replacement actors have no receivers and their paired policy
                # senders still bind the dead agents; force a full rebind.
                self._checkpoint_engine_ready = False
        self._ensure_ready_and_consume_count()

    @property
    def is_stale(self) -> bool:
        return self._stale

    def _release_after_refit(self) -> bool:
        cfg = self._checkpoint_engine_config
        return bool(cfg["engine_kwargs"][cfg["backend"]]["release_after_refit"])

    def _run_policy(
        self, checkpoint_method: str, **method_kwargs: Any
    ) -> list[ray.ObjectRef]:
        return self._policy.worker_group.run_all_workers_single_data(
            "checkpoint_engine_rpc",
            checkpoint_method=checkpoint_method,
            method_kwargs=method_kwargs,
        )

    def _is_sglang(self) -> bool:
        return self._generation.cfg["backend"] == SGLANG_BACKEND

    def _generation_rpc(self) -> str:
        return (
            "checkpoint_engine_rpc_async"
            if self._generation.cfg["vllm_cfg"]["async_engine"]
            else "checkpoint_engine_rpc"
        )

    def _run_generation(
        self, checkpoint_method: str, method_args: tuple[Any, ...] = ()
    ) -> list[ray.ObjectRef]:
        if self._is_sglang():
            return self._generation.run_checkpoint_engine_method(
                checkpoint_method, method_args
            )
        return self._generation.worker_group.run_all_workers_single_data(
            self._generation_rpc(),
            checkpoint_method=checkpoint_method,
            method_args=method_args,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

    def _resolve_bucket_size_bytes(self) -> int:
        if self._bucket_size_bytes is not None:
            return self._bucket_size_bytes

        memory_ratio_raw = self._checkpoint_engine_config[
            "update_weights_bucket_memory_ratio"
        ]
        try:
            memory_ratio = float(memory_ratio_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "update_weights_bucket_memory_ratio must be a valid float, got "
                f"{memory_ratio_raw!r}."
            ) from exc
        if not 0 < memory_ratio < 1:
            raise ValueError(
                "update_weights_bucket_memory_ratio must be between 0 and 1, got "
                f"{memory_ratio_raw!r}."
            )

        total_memory = _flatten_metadata(
            ray.get(
                self._run_policy("checkpoint_engine_total_memory_bytes")
                + self._run_generation("checkpoint_engine_total_memory_bytes")
            )
        )
        minimum_total_bytes = min(int(value) for value in total_memory)
        bucket_size_bytes = int(minimum_total_bytes * memory_ratio)
        bucket_size_bytes = bucket_size_bytes // _MEBIBYTE * _MEBIBYTE
        if bucket_size_bytes < _MEBIBYTE:
            raise ValueError(
                "Checkpoint-engine bucket sizing produced less than 1 MiB per buffer."
            )

        self._bucket_size_bytes = bucket_size_bytes
        print(
            "[checkpoint engine] Bucket size: "
            f"{bucket_size_bytes // _MEBIBYTE} MiB per buffer "
            f"({memory_ratio:.1%} of {minimum_total_bytes / 1024**3:.2f} GiB "
            "minimum total GPU memory)."
        )
        return bucket_size_bytes

    def _ensure_checkpoint_engine_ready(self) -> None:
        if self._checkpoint_engine_ready:
            return

        cfg = self._checkpoint_engine_config
        backend = cfg["backend"]
        bucket_size_bytes = self._resolve_bucket_size_bytes()
        engine_kwargs = cfg["engine_kwargs"][backend]

        ray.get(
            self._run_policy(
                "init_checkpoint_engine",
                backend=backend,
                bucket_size_bytes=bucket_size_bytes,
                engine_kwargs=engine_kwargs,
            )
            + self._run_generation(
                "init_checkpoint_engine",
                (backend, bucket_size_bytes, engine_kwargs),
            )
        )

        policy_prepare_refs = self._run_policy("prepare_checkpoint_engine")
        generation_prepare_refs = self._run_generation("prepare_checkpoint_engine")
        prepare_results = ray.get(policy_prepare_refs + generation_prepare_refs)
        policy_metadata = _sort_ranked_metadata(
            _flatten_metadata(prepare_results[: len(policy_prepare_refs)])
        )
        generation_metadata = _ordered_generation_metadata(
            prepare_results[len(policy_prepare_refs) :]
        )
        topology = {
            "metadata": policy_metadata + generation_metadata,
            "train_world_size": len(policy_metadata),
            "rollout_world_size": len(generation_metadata),
        }
        if self._is_sglang():
            generation_init_refs = (
                self._generation.init_checkpoint_engine_process_groups(**topology)
            )
        else:
            worker_count = len(self._generation.worker_group.workers)
            workers_per_group = worker_count // self._generation.dp_size
            generation_init_refs = (
                self._generation.worker_group.run_all_workers_multiple_data(
                    self._generation_rpc(),
                    method_args=[
                        (
                            rank_prefix,
                            topology["train_world_size"],
                            topology["rollout_world_size"],
                            topology["metadata"],
                        )
                        for rank_prefix in range(0, worker_count, workers_per_group)
                    ],
                    run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                    common_kwargs={
                        "checkpoint_method": "init_checkpoint_engine_process_group"
                    },
                )
            )
        ray.get(
            self._run_policy("init_checkpoint_engine_process_group", **topology)
            + generation_init_refs
        )
        self._checkpoint_engine_ready = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._raise_if_terminal()
        self._stale = True
        if self._is_sglang():
            self._sglang_recover_and_rebind()
        else:
            self._ensure_checkpoint_engine_ready()
        context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )

        try:
            with context:
                if self._is_sglang():
                    self._sglang_transfer(kv_scales)
                else:
                    self._transfer(kv_scales)
                self._stale = False
        finally:
            # Never finalize on the terminal path: NIXL may still have
            # in-flight work, and finalizing would mask the primary error.
            if self._terminal_error is None and self._release_after_refit():
                self.shutdown()

    def _transfer(self, kv_scales: Optional[dict[str, float]]) -> None:
        """Run one policy->rollout checkpoint-engine transfer."""
        policy_refs = self._run_policy(
            "send_weights_via_checkpoint_engine", kv_scales=kv_scales
        )
        results = ray.get(
            policy_refs + self._run_generation("update_weights_from_checkpoint_engine")
        )
        if not all(
            result for result in results[len(policy_refs) :] if result is not None
        ):
            raise RuntimeError(
                "Weight transfer failed during "
                f"{self._checkpoint_engine_config['backend']} "
                "checkpoint-engine sync."
            )

    def _sglang_transfer(self, kv_scales: Optional[dict[str, float]]) -> None:
        """Transfer inside the engine-side weight-update session.

        SGLang gates ``update_weights_from_tensor`` on a session opened by
        ``begin_weight_update``, and only ``end_weight_update`` rebuilds the
        quantized kernel layouts afterwards, so the transfer has to sit inside
        that envelope however the bytes arrive. The pause matters for the same
        reason it does on the sibling path: the buckets land as several
        ``update_weights_from_tensor`` calls, each taking the server's model
        update lock on its own, so a request admitted between two buckets would
        run against a half-updated model.

        The success-path envelope is the one ``_SGLangWeightSynchronizer._refit``
        uses; the sibling additionally rejects ``kv_scales``, which this
        transport forwards to the policy. The failure path deliberately
        diverges from the sibling: once the transfer has started, a failure is
        terminal and serving is NOT resumed (a NCCL broadcast can be safely
        redone next refit; interrupted one-sided NIXL work cannot).
        """
        self._generation.prepare_for_generation(tags=["weights"])
        # Cleanup is two-phase, split at "did the transfer start":
        # - Before any bucket moved, nothing changed the served weights, so a
        #   failure releases every acquired state (KV pool re-acquired, monitor
        #   resumed, serving readmitted) and stays retryable.
        # - Once the transfer starts, a failure of the transfer OR of
        #   end_weight_update leaves a half-updated model and possibly
        #   in-flight NIXL work: latch terminal, keep serving and the health
        #   monitor paused, and let the primary exception propagate. Job abort
        #   is the recovery mechanism.
        transfer_started = False
        try:
            self._generation.pause_generation(
                mode=self._generation.pause_generation_mode
            )
            if not self._generation.invalidate_kv_cache():
                raise RuntimeError("SGLang KV cache invalidation failed before refit.")

            self._generation.begin_weight_update()
            try:
                transfer_started = True
                self._transfer(kv_scales)
            except BaseException:
                # Close the session without masking the transfer error.
                try:
                    self._generation.end_weight_update()
                except Exception as end_exc:
                    print(
                        "[SGLang refit] end_weight_update also failed after a "
                        f"failed transfer (suppressed): {end_exc!r}"
                    )
                raise
            # A transfer that moved bytes but never finalized kernel layouts is
            # as unusable as a failed transfer: end must succeed too.
            self._generation.end_weight_update()
        except BaseException as exc:
            if transfer_started:
                self._set_terminal(exc)
                raise
            # Pre-transfer failure: safe cleanup, then propagate the original.
            try:
                self._generation.prepare_for_generation(tags=["kv_cache"])
                self._generation.continue_generation()
            except Exception as cleanup_exc:
                print(
                    "[SGLang refit] cleanup after a pre-transfer failure also "
                    f"failed (suppressed): {cleanup_exc!r}"
                )
            raise
        # Success: re-acquire the KV pool (this also resumes the #3613 health
        # monitor) before readmitting requests.
        self._generation.prepare_for_generation(tags=["kv_cache"])
        self._generation.continue_generation()

    def shutdown(self) -> None:
        if not self._checkpoint_engine_ready:
            return
        ray.get(
            self._run_policy("finalize_checkpoint_engine")
            + self._run_generation("finalize_checkpoint_engine")
        )
        self._checkpoint_engine_ready = False
