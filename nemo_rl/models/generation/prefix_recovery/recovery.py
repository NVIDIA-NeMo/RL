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

"""Cut live model calls at a checkpoint and resume them after a restore.

The worker builds one ``PrefixRecovery`` in ``setup_token_capture`` when
``target_level`` is ``prefix``, or a ``NoPrefixRecovery`` otherwise, and calls
it at fixed points: when a call begins and gets its sampling settings, when its
final output arrives and its terminal row is written, at every checkpoint cut
and snapshot flush. Only the ``ProgressReader`` it is given knows the engine.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import threading
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from nemo_rl.models.generation.prefix_recovery.parts import PrefixRecoveryParts
from nemo_rl.models.generation.prefix_recovery.restore import (
    _generation_cut_blockers,
    restore_reply_text,
)
from nemo_rl.models.generation.prefix_recovery.state import (
    _CompletedCaptureState,
    _RequestCaptureState,
)

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0

# Sampling settings already warned about, so each warns once per process.
_LOGGED_UNCUTTABLE_REASONS: set[str] = set()

# One character spans at most four UTF-8 bytes, hence at most four byte-level
# tokens; decoding a slightly longer tail is enough to see whether it is whole.
_PREFIX_BOUNDARY_TAIL_TOKENS = 8


def initial_generation_cut_counts() -> dict[str, int]:
    """Zeroed progress-read counters, so each is reported from the first step.

    How many reads of the engine's decoding progress, for how many calls, and
    the total microseconds spent waiting for and holding its output lock.
    """
    return {
        "progress_reads": 0,
        "progress_read_calls": 0,
        "progress_read_wait_us": 0,
        "progress_read_hold_us": 0,
    }


def _require_inventory_identity(
    prefix: Any, worker_capture_key: str, *, call_kind: str
) -> None:
    """Refuse a cut whose inventory entry names another episode than the call."""
    # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
    from nemo_gym.episode_types import EpisodeId

    expected = EpisodeId(rollout_id=prefix.rollout_id, attempt=prefix.attempt)
    if worker_capture_key != expected.capture_key:
        raise RuntimeError(
            "generation-prefix inventory identity does not match the "
            f"{call_kind} call: model_call_id={prefix.model_call_id!r}, "
            f"inventory_rollout_id={prefix.rollout_id!r}, "
            f"worker_capture_key={worker_capture_key!r}"
        )


def _ends_inside_character(tokenizer: Any, generated_token_ids: list[int]) -> bool:
    """Whether the generated text stops partway through a multi-byte character.

    The engine detokenizes a resumed call's tail with a stream primed on the
    prompt, which ends with the restored prefix. A prefix ending inside a
    character makes that stream emit the whole decoded prompt as the tail's
    first text.
    """
    tail = generated_token_ids[-_PREFIX_BOUNDARY_TAIL_TOKENS:]
    return tokenizer.decode(tail, skip_special_tokens=False).endswith("\ufffd")


class PrefixRecovery:
    """Generation-prefix cuts and restores for one worker's captured calls."""

    def __init__(self, parts: PrefixRecoveryParts) -> None:
        self._parts = parts
        self._registry = parts.registry
        # Guards this object's own state below; never held across TQ I/O.
        self._lock = threading.Lock()
        self._completed: dict[str, _CompletedCaptureState] = {}
        self._receipts: dict[tuple[str, str], Any] = {}
        # Calls this process resumed from a restored generation prefix, and
        # their prefix tokens; the driver logs the sums each step.
        self._restored_calls = 0
        self._restored_tokens = 0
        self._counts = initial_generation_cut_counts()
        # Chunk rows of ended calls, deleted when the next snapshot starts, and
        # the latest snapshot that deleted them.
        self._obsolete_keys: list[str] = []
        self._flushed_epoch = 0
        # A dedicated thread keeps completions' TQ writes from consuming the
        # threads needed to create the checkpoint cut.
        self._control_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-generation-cut-control"
        )
        # Obsolete chunk rows are deleted here, off the serving loop.
        self._cleanup_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-generation-cut-cleanup"
        )

    @property
    def control_token(self) -> str | None:
        """The bearer token the cut route requires; None refuses every cut."""
        return self._parts.control_token

    # ---- A call's life --------------------------------------------------

    def on_begin(self, admission: Any) -> None:
        """Count a call Gym's ``begin_call`` accepted with a restored cut."""
        if admission.generation_cut is None:
            return
        with self._lock:
            self._restored_calls += 1
            self._restored_tokens += admission.generation_cut.prefix_token_count

    def on_sampling_params(self, request: Any, sampling_params: Any) -> None:
        """Record the engine's resolved output budget, and whether a cut can resume."""
        state = self._registry.get(request)
        if state is None:
            return
        effective_output_limit = int(sampling_params.max_tokens)
        if effective_output_limit <= 0:
            raise ValueError("effective output limit must be positive")
        state.set_effective_output_limit(effective_output_limit)
        # Such a call regenerates from its previous boundary after a restore,
        # which costs time but not correctness; see _generation_cut_blockers.
        blockers = _generation_cut_blockers(sampling_params, self._parts.tokenizer)
        if not blockers:
            return
        reason = f"{', '.join(blockers)} do not carry across a generation cut"
        state.mark_uncuttable(reason)
        if reason not in _LOGGED_UNCUTTABLE_REASONS:
            _LOGGED_UNCUTTABLE_REASONS.add(reason)
            LOGGER.warning(
                "model calls using %s are not cut; they regenerate after a restore",
                ", ".join(blockers),
            )

    def observe(self, request: Any, request_output: Any) -> None:
        """Record the tokens a finished output adds past the last cut."""
        state = self._registry.get(request)
        if state is None:
            return
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return
        observed = state.recorded_token_count
        try:
            output = outputs[0]
            # Finished, so the engine appends to these lists no more.
            generation_token_ids = list(getattr(output, "token_ids", None) or ())[
                observed:
            ]
            generation_logprobs = self._parts.progress_reader.logprobs(
                generation_token_ids, getattr(output, "logprobs", None), start=observed
            )
            finish_reason = getattr(output, "finish_reason", None)
            stop_reason = getattr(output, "stop_reason", None)
        except (RuntimeError, TypeError, ValueError) as error:
            state.record_error(f"{type(error).__name__}: {error}")
            return
        state.record(
            generation_token_ids,
            generation_logprobs,
            start=observed,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )

    def restore_reply_text(
        self, request: Any, request_output: Any, *, tokenizer: Any
    ) -> None:
        """Prepend a resumed call's restored text to its final output."""
        state = self._registry.get(request)
        if state is None or not state.resumed_generation_token_ids:
            return
        restore_reply_text(
            request,
            request_output,
            prefix_token_ids=state.resumed_generation_token_ids,
            engine_prompt_token_ids=state.prompt_token_ids,
            tokenizer=tokenizer,
        )

    def retire(self, staging_keys: Sequence[str]) -> None:
        """Queue an ended call's chunk rows for the next snapshot to delete.

        A snapshot in progress may still name them, so they wait until the
        next one starts, after this one is saved.
        """
        if not staging_keys:
            return
        with self._lock:
            self._obsolete_keys.extend(staging_keys)

    async def flush(self, epoch: int) -> None:
        """Delete the chunk rows retired before snapshot ``epoch`` started.

        Returns once the delete is done: TQ must not clear rows while the
        snapshot is saved.
        """
        await asyncio.get_running_loop().run_in_executor(
            self._cleanup_executor, self._flush_retired, epoch
        )

    def _flush_retired(self, epoch: int) -> None:
        # They belong to calls that ended before Gym froze this snapshot's
        # inventory, which names a finished call by its terminal row only, and
        # a saved snapshot keeps its own copy of TransferQueue.
        with self._lock:
            # A late flush from an older snapshot must not delete rows retired
            # since a newer one started; that snapshot may still name them.
            if epoch <= self._flushed_epoch:
                return
            self._flushed_epoch = epoch
            staging_keys = self._obsolete_keys
            self._obsolete_keys = []
        if not staging_keys:
            return
        try:
            self._parts.sink.clear(staging_keys)
        except Exception:  # noqa: BLE001 - the calls have already ended
            # A failed clear only leaves rows the next restore clears.
            LOGGER.exception(
                "failed to clear %d obsolete generation chunk rows",
                len(staging_keys),
            )

    def counts(self) -> dict[str, int]:
        """Restore and progress-read counts since this process started.

        Returns:
            ``restored_calls`` and their total ``restored_prefix_tokens``, plus
            ``progress_reads``, ``progress_read_calls`` and the microseconds
            they waited for and held the engine's output lock.
        """
        with self._lock:
            return {
                "restored_calls": self._restored_calls,
                "restored_prefix_tokens": self._restored_tokens,
                **self._counts,
            }

    def shutdown(self) -> None:
        # Waiting for a wedged TQ write here would make actor shutdown hang.
        for executor in (self._control_executor, self._cleanup_executor):
            executor.shutdown(wait=False, cancel_futures=True)

    # ---- A checkpoint cut -----------------------------------------------

    async def run_cut(self, inventory: Any) -> Any:
        """Cut every call Gym's inventory names, off the serving loop."""
        # Measured from arrival so time queued behind a stale cut counts.
        timeout_s = self._parts.control_timeout_s
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        return await asyncio.get_running_loop().run_in_executor(
            self._control_executor,
            functools.partial(self.checkpoint_cut, inventory, deadline=deadline),
        )

    def _record_progress(
        self, states: list[_RequestCaptureState], *, deadline: float | None
    ) -> None:
        """Record the tokens each call decoded since its last record.

        One read at an engine output boundary copies every call's new token
        IDs, so all calls are read at the same point; their logprobs are then
        read outside it, from positions that read saw published. A call the
        engine has finished but whose consumer has not yet taken the output is
        read whole, with its finish reason. One the engine does not hold (not
        yet submitted, or aborted) keeps what it has recorded.
        """
        pending = [
            (state, state.call.model_call_id, state.recorded_token_count)
            for state in states
        ]
        if not pending:
            return
        reader = self._parts.progress_reader
        try:
            read = reader.read(
                [(model_call_id, start) for _, model_call_id, start in pending],
                timeout_s=None if deadline is None else deadline - time.monotonic(),
            )
        except TimeoutError:
            LOGGER.warning(
                "generation cut could not read the engine's progress before its "
                "deadline; %d calls are cut at the tokens they have recorded",
                len(pending),
            )
            return
        with self._lock:
            self._counts["progress_reads"] += 1
            self._counts["progress_read_calls"] += len(pending)
            self._counts["progress_read_wait_us"] += round(read.wait_seconds * 1e6)
            self._counts["progress_read_hold_us"] += round(read.hold_seconds * 1e6)
        LOGGER.debug(
            "generation cut read %d calls: waited %.2f ms, held %.2f ms",
            len(pending),
            read.wait_seconds * 1e3,
            read.hold_seconds * 1e3,
        )
        for state, model_call_id, start in pending:
            progress = read.progress.get(model_call_id)
            if progress is None:
                continue
            try:
                generated_logprobs = reader.logprobs(
                    progress.token_ids, progress.logprobs, start=start
                )
            except (RuntimeError, TypeError, ValueError) as error:
                state.record_error(f"{type(error).__name__}: {error}")
                continue
            state.record(
                progress.token_ids,
                generated_logprobs,
                start=start,
                finish_reason=progress.finish_reason,
                stop_reason=progress.stop_reason,
            )

    @staticmethod
    def _failed_ack(prefix: Any, *, reason: str, uncut: Counter[str]) -> Any:
        """Acknowledge one uncut call and tally it under ``reason``."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

        uncut[reason] += 1
        return GenerationCutPrefixAck.failure(prefix)

    def _completed_ack(
        self, prefix: Any, checkpoint_id: str, uncut: Counter[str]
    ) -> Any:
        """Describe a call whose terminal path won the cut/abort race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

        with self._lock:
            completed = self._completed.get(prefix.model_call_id)
        if completed is not None:
            _require_inventory_identity(
                prefix, completed.coords.capture_key, call_kind="completed"
            )
        if completed is None:
            # Gym can name a call before its request reaches this worker, so
            # absence does not mean the call's finished result aged out.
            reason = "no_capture_or_completed_state"
            unusable_reason = "this worker has no record of it"
        elif completed.coords.disposition != "staged":
            reason = "completed_not_staged"
            unusable_reason = (
                f"its terminal write ended as {completed.coords.disposition!r}"
            )
        elif completed.effective_output_limit is None:
            reason = "completed_missing_output_limit"
            unusable_reason = "vLLM did not resolve its effective output limit"
        elif completed.terminal_finish_reason is None:
            reason = "completed_missing_finish_reason"
            unusable_reason = "it finished without a stop or length finish reason"
        else:
            return GenerationCutPrefixAck(
                **prefix.model_dump(mode="json"),
                disposition="durable_prefix",
                cut_kind="terminal_completion",
                frozen_buffer_id=f"terminal/{checkpoint_id}",
                staging_keys=(completed.coords.staging_key,),
                prefix_token_count=completed.generation_token_count,
                prefix_digest=completed.coords.digest,
                effective_output_limit=completed.effective_output_limit,
                terminal_finish_reason=completed.terminal_finish_reason,
                terminal_stop_reason=completed.terminal_stop_reason,
            )
        # The call regenerates from its previous boundary after a restore.
        LOGGER.warning(
            "cannot cut completed model call %s: %s",
            prefix.model_call_id,
            unusable_reason,
        )
        return self._failed_ack(prefix, reason=reason, uncut=uncut)

    def _cut_active(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
        uncut: Counter[str],
    ) -> Any | None:
        """Cut one live call, or return ``None`` if abort won the race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

        # _checkpoint_generation_cut checked both before cutting any call.
        capture = self._parts.capture
        sink = self._parts.sink

        # Abort and terminal completion own the same lifecycle. Once the cut
        # owns it, the call cannot be removed until its frozen tokens are
        # staged or rolled back (leaving the block rolls back an unsealed cut).
        # If abort won first, use its terminal evidence.
        with state.lifecycle() as owned:
            current = self._registry.get_by_model_call_id(prefix.model_call_id)
            if current is not state or owned.terminal_started:
                return None

            _require_inventory_identity(
                prefix, state.call.capture_key, call_kind="active"
            )

            progress, frozen = owned.freeze(checkpoint_id)
            uncuttable_reason = progress.uncuttable_reason
            observation_error = progress.observation_error
            effective_output_limit = progress.effective_output_limit
            if uncuttable_reason is not None:
                # Expected for this request; the worker warned when it was set.
                LOGGER.debug(
                    "not cutting model call %s: %s",
                    prefix.model_call_id,
                    uncuttable_reason,
                )
                return self._failed_ack(
                    prefix, reason="uncuttable_request", uncut=uncut
                )
            if observation_error is not None:
                # Its recorded tokens or logprobs are inconsistent with vLLM's.
                # Only this call regenerates; the rest of the inventory is cut.
                LOGGER.warning(
                    "cannot cut model call %s: %s",
                    prefix.model_call_id,
                    observation_error,
                )
                return self._failed_ack(prefix, reason="observation_error", uncut=uncut)
            if effective_output_limit is None:
                # Checked before freezing, so the call's buffer is untouched.
                LOGGER.warning(
                    "cannot cut model call %s: vLLM did not resolve its "
                    "effective output limit",
                    prefix.model_call_id,
                )
                return self._failed_ack(
                    prefix, reason="missing_output_limit", uncut=uncut
                )
            assert frozen is not None  # freeze refuses only the cases above
            generated_token_ids = frozen.generated_token_ids
            inherited_prefix_token_count = (
                state.call.admission.generation_cut.prefix_token_count
                if state.call.admission.generation_cut is not None
                else 0
            )
            total_generation_token_count = inherited_prefix_token_count + len(
                generated_token_ids
            )
            unusable_prefix = True
            if total_generation_token_count == 0:
                # Expected for a call still prefilling; nothing to keep yet.
                LOGGER.debug(
                    "not cutting model call %s: it has no generated tokens yet",
                    prefix.model_call_id,
                )
            elif generated_token_ids and _ends_inside_character(
                self._parts.tokenizer,
                state.resumed_generation_token_ids + generated_token_ids,
            ):
                LOGGER.warning(
                    "cannot cut model call %s: its %d generated tokens end inside "
                    "a multi-byte character",
                    prefix.model_call_id,
                    total_generation_token_count,
                )
            else:
                unusable_prefix = False
            if unusable_prefix:
                owned.rollback(checkpoint_id)
                return self._failed_ack(
                    prefix,
                    reason="zero_generated_tokens"
                    if total_generation_token_count == 0
                    else "incomplete_utf8_boundary",
                    uncut=uncut,
                )
            staged_key = None
            try:
                cumulative_record = capture.build_prefix_record(
                    state.call,
                    prompt_token_ids=state.prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=frozen.generated_logprobs,
                )
                if frozen.chunk_token_ids:
                    chunk_record = (
                        cumulative_record
                        if not progress.staging_keys
                        else capture.build_generation_chunk_record(
                            state.call,
                            generated_token_ids=frozen.chunk_token_ids,
                            generated_logprobs=frozen.chunk_logprobs,
                        )
                    )
                    result = sink.stage_generation_prefix(
                        chunk_record,
                        checkpoint_id=checkpoint_id,
                        chunk_sequence=frozen.chunk_sequence,
                    )
                    staged_key = result.staging_key
                    if not result.ok:
                        raise RuntimeError(
                            "generation-prefix staging failed for "
                            f"{prefix.model_call_id!r}: {result.error}"
                        )
                    if staged_key is None:
                        raise RuntimeError(
                            "generation-prefix staging returned no staging key for "
                            f"{prefix.model_call_id!r}"
                        )
                candidate_staging_keys = progress.staging_keys
                if staged_key is not None:
                    candidate_staging_keys += (staged_key,)
                acknowledgement = GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_prefix",
                    cut_kind="active_prefix",
                    frozen_buffer_id=frozen.chunk_key,
                    staging_keys=candidate_staging_keys,
                    prefix_token_count=total_generation_token_count,
                    prefix_digest=cumulative_record.digest,
                    effective_output_limit=effective_output_limit,
                    terminal_finish_reason=progress.terminal_finish_reason,
                    terminal_stop_reason=progress.terminal_stop_reason,
                )
                # Validate the complete acknowledgement before adopting the
                # staged chunk; a failed one leaves the cut to roll back.
                owned.seal(checkpoint_id, staged_key)
            except Exception:
                # Leaving the lifecycle rolls the unsealed cut back.
                if staged_key is not None:
                    try:
                        sink.clear([staged_key])
                    except Exception:  # noqa: BLE001 — preserve the cut failure
                        LOGGER.exception(
                            "Failed to clear rejected generation-prefix row %s",
                            staged_key,
                        )
                raise
            return acknowledgement

    def checkpoint_cut(self, inventory: Any, *, deadline: float | None = None) -> Any:
        """Stage a stable prefix for every call named by Gym's frozen inventory.

        ``deadline`` is a ``time.monotonic()`` bound. Gym stops waiting for the
        receipt at its own control deadline, so calls not reached by then are
        acknowledged as failures (they regenerate after restore) instead of
        holding the single cut-control thread for the next checkpoint.
        """
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutReceipt,
        )

        receipt_key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._lock:
            cached_receipt = self._receipts.get(receipt_key)
        if cached_receipt is not None:
            return cached_receipt
        active_states = [
            state
            for prefix in inventory.active_prefixes
            if (state := self._registry.get_by_model_call_id(prefix.model_call_id))
            is not None
        ]
        self._record_progress(active_states, deadline=deadline)
        acknowledgements = []
        # Why each uncut call was left uncut; Gym counts them per checkpoint.
        uncut: Counter[str] = Counter()
        for index, prefix in enumerate(inventory.active_prefixes):
            if deadline is not None and time.monotonic() >= deadline:
                remaining = inventory.active_prefixes[index:]
                LOGGER.warning(
                    "generation cut passed its control deadline: checkpoint_id=%s "
                    "uncut=%d of %d model_call_ids=%s",
                    inventory.checkpoint_id,
                    len(remaining),
                    len(inventory.active_prefixes),
                    [call.model_call_id for call in remaining],
                )
                acknowledgements.extend(
                    self._failed_ack(call, reason="control_deadline", uncut=uncut)
                    for call in remaining
                )
                break
            state = self._registry.get_by_model_call_id(prefix.model_call_id)
            try:
                acknowledgement = (
                    self._cut_active(prefix, state, inventory.checkpoint_id, uncut)
                    if state is not None
                    else None
                )
                if acknowledgement is None:
                    acknowledgement = self._completed_ack(
                        prefix, inventory.checkpoint_id, uncut
                    )
            # Gym acknowledges cuts per call, so one call's failure (a TQ write,
            # an identity mismatch) regenerates that call alone after a restore
            # instead of failing the receipt for every call on this worker.
            except Exception:  # noqa: BLE001 - see above
                LOGGER.exception(
                    "cannot cut model call %s; it regenerates after a restore",
                    prefix.model_call_id,
                )
                acknowledgement = self._failed_ack(
                    prefix, reason="cut_error", uncut=uncut
                )
            acknowledgements.append(acknowledgement)
        if uncut:
            LOGGER.info(
                "generation cut %s left %d of %d model calls uncut: %s",
                inventory.checkpoint_id,
                uncut.total(),
                len(inventory.active_prefixes),
                dict(sorted(uncut.items())),
            )
        receipt = GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"worker-{inventory.inventory_digest}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"tq-{inventory.inventory_digest}",
            prefixes=tuple(acknowledgements),
        )
        with self._lock:
            self._receipts[receipt_key] = receipt
            if len(self._receipts) > 256:
                self._receipts.pop(next(iter(self._receipts)))
        return receipt

    def _rebuild_prefix(self, admission: Any, prefix_token_ids: list[int]) -> Any:
        """Fetch an admission's staged prefix chunks and rebuild the cumulative prefix."""
        continuation = admission.generation_cut
        # Gym requires at least one staging key per continuation.
        staging_keys = list(continuation.staging_keys)
        snapshots = self._parts.staging_source.fetch(staging_keys)
        if len(snapshots) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        weight_versions = [snapshot.weight_version for snapshot in snapshots]
        if weight_versions != sorted(weight_versions):
            raise RuntimeError(
                "generation-cut chunk policy versions are not monotonically non-decreasing"
            )
        current_weight_version = self._parts.weight_version()
        if weight_versions[-1] > current_weight_version:
            raise RuntimeError(
                "generation-cut prefix contains policy version "
                f"{weight_versions[-1]}, newer than the current rollout version "
                f"{current_weight_version}"
            )
        token_ids_delta = [
            token_id for snapshot in snapshots for token_id in snapshot.token_ids_delta
        ]
        token_mask_delta = [
            mask for snapshot in snapshots for mask in snapshot.token_mask_delta
        ]
        generation_logprobs_delta = [
            logprob
            for snapshot in snapshots
            for logprob in snapshot.generation_log_probs_delta
        ]
        if any(
            mask == 0.0
            for snapshot in snapshots[1:]
            for mask in snapshot.token_mask_delta
        ):
            raise RuntimeError(
                "only the first generation-cut chunk may contain prompt tokens"
            )

        # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
        from nemo_gym.token_id_capture.staging.digest import (
            compute_chain_hash,
            compute_extras_digest,
            hash_token_ids,
        )
        from nemo_gym.token_id_capture.staging.records import StagedCallRecord

        delta_len = len(token_ids_delta)
        # A cut chain may span retries performed after later weight updates.
        # Its token-level behavior logprobs remain attached to their original
        # chunks. The cumulative snapshot uses the oldest contributing version
        # so downstream replay-buffer staleness checks remain conservative.
        # Gym's begin_call checks this snapshot's digest and generated-token
        # count against the admission before the call is served.
        snapshot = StagedCallRecord.from_components(
            capture_key=continuation.source_capture_key,
            model_call_id=continuation.source_model_call_id,
            parent_call_id=admission.parent_call_id,
            mode=admission.mode,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=admission.prev_len + delta_len,
            weight_version=weight_versions[0],
            token_ids_delta=token_ids_delta,
            token_mask_delta=token_mask_delta,
            generation_log_probs_delta=generation_logprobs_delta,
            extras=None,
            extras_digest=compute_extras_digest(None),
            chain_hash=compute_chain_hash(admission.parent_chain_hash, token_ids_delta),
            cumulative_hash=hash_token_ids(prefix_token_ids + token_ids_delta),
        )
        # Lets a consumer confirm the exact generated tokens it later receives
        # (e.g. a trained row) without access to the staging store.
        generated_prefix_ids = [
            token_id
            for token_id, mask in zip(token_ids_delta, token_mask_delta, strict=True)
            if mask == 1.0
        ]
        prefix_ids_sha256 = hashlib.sha256(
            ",".join(map(str, generated_prefix_ids)).encode()
        ).hexdigest()
        LOGGER.info(
            "generation prefix restored: rollout_id=%s model_call_id=%s "
            "source_model_call_id=%s prefix_tokens=%d prefix_digest=%s "
            "weight_version_span=[%d,%d] prefix_ids_sha256=%s",
            admission.capture_key,
            admission.model_call_id,
            continuation.source_model_call_id,
            continuation.prefix_token_count,
            continuation.prefix_digest,
            weight_versions[0],
            weight_versions[-1],
            prefix_ids_sha256,
        )
        return snapshot

    def resolve_restore(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> tuple[Any, Any | None]:
        """Resolve an admission's restored prefix, or drop it so the call regenerates.

        Gym has already consumed the restored cut when it sends this call, and
        generating the whole response from the turn boundary is always valid,
        so a prefix that cannot be fetched or does not match its admission
        costs time, not the rollout. Returns the admission to serve and the
        verified prefix snapshot, or the admission without its cut and None.
        """
        continuation = admission.generation_cut
        if continuation is None:
            return admission, None
        try:
            snapshot = self._rebuild_prefix(admission, prefix_token_ids)
            # Checked here, before the prefix is spliced into the prompt; Gym's
            # begin_call repeats it and would otherwise fail the call.
            if snapshot.digest != continuation.prefix_digest:
                raise RuntimeError("rebuilt prefix digest does not match the admission")
            generated = sum(mask == 1.0 for mask in snapshot.token_mask_delta)
            if generated != continuation.prefix_token_count:
                raise RuntimeError(
                    f"rebuilt prefix has {generated} generated tokens, "
                    f"admission expects {continuation.prefix_token_count}"
                )
            return admission, snapshot
        except Exception:  # noqa: BLE001 - any unusable prefix regenerates
            LOGGER.warning(
                "cannot restore the generation prefix of model call %s; "
                "regenerating it from its turn boundary",
                admission.model_call_id,
                exc_info=True,
            )
        # The regenerated call will not extend these rows, so they are retired
        # here instead of by its terminal write.
        self.retire(continuation.staging_keys)
        return admission.model_copy(update={"generation_cut": None}), None

    def remember_completed(
        self,
        model_call_id: str,
        coords: Any,
        generation_token_count: int,
        *,
        effective_output_limit: int | None = None,
        terminal_finish_reason: Literal["stop", "length"] | None = None,
        terminal_stop_reason: str | int | None = None,
    ) -> None:
        """Keep a finished call's terminal row for a cut that still names it."""
        now = time.monotonic()
        with self._lock:
            # Dict insertion order is completion order. Remove only entries old
            # enough that the response/inventory race has certainly elapsed;
            # never discard fresh evidence because the deployment crossed an
            # arbitrary call-count threshold.
            while self._completed:
                oldest_id = next(iter(self._completed))
                oldest = self._completed[oldest_id]
                if (
                    now - oldest.completed_at_monotonic
                    <= _COMPLETED_CAPTURE_RETENTION_S
                ):
                    break
                self._completed.pop(oldest_id)
            self._completed.pop(model_call_id, None)
            self._completed[model_call_id] = _CompletedCaptureState(
                coords=coords,
                generation_token_count=generation_token_count,
                effective_output_limit=effective_output_limit,
                terminal_finish_reason=terminal_finish_reason,
                terminal_stop_reason=terminal_stop_reason,
                completed_at_monotonic=now,
            )


class NoPrefixRecovery:
    """What a worker without generation-prefix cuts does at each recovery point.

    Every hook is a no-op, so the worker never asks whether cuts are on.
    """

    control_token: str | None = None

    def resolve_restore(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> tuple[Any, Any | None]:
        if admission.generation_cut is None:
            return admission, None
        # Gym only sends a restored cut to a worker that acknowledged one, so
        # this is a misconfiguration; regenerating the call is always valid.
        LOGGER.warning(
            "generation-prefix cuts are off; regenerating model call %s from "
            "its turn boundary",
            admission.model_call_id,
        )
        return admission.model_copy(update={"generation_cut": None}), None

    def on_begin(self, admission: Any) -> None:
        pass

    def on_sampling_params(self, request: Any, sampling_params: Any) -> None:
        pass

    def observe(self, request: Any, request_output: Any) -> None:
        pass

    def restore_reply_text(
        self, request: Any, request_output: Any, *, tokenizer: Any
    ) -> None:
        pass

    def remember_completed(
        self,
        model_call_id: str,
        coords: Any,
        generation_token_count: int,
        *,
        effective_output_limit: int | None = None,
        terminal_finish_reason: Literal["stop", "length"] | None = None,
        terminal_stop_reason: str | int | None = None,
    ) -> None:
        # Only a generation cut reads completed calls.
        pass

    def retire(self, staging_keys: Sequence[str]) -> None:
        pass

    async def flush(self, epoch: int) -> None:
        pass

    def counts(self) -> dict[str, int]:
        return {"restored_calls": 0, "restored_prefix_tokens": 0}

    def shutdown(self) -> None:
        pass

    async def run_cut(self, inventory: Any) -> Any:
        raise RuntimeError("generation-prefix cuts are disabled on this worker")
