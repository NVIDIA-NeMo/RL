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

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0


def initial_generation_cut_counts() -> dict[str, int]:
    """Zeroed progress-read counters, so each is reported from the first step.

    How many reads of vLLM's decoding progress, for how many calls, and the total
    microseconds spent waiting for and holding vLLM's output lock.
    """
    return {
        "progress_reads": 0,
        "progress_read_calls": 0,
        "progress_read_wait_us": 0,
        "progress_read_hold_us": 0,
    }


@dataclass(frozen=True)
class _CallProgress:
    """A consistent copy of what one call has recorded, taken under its lock."""

    generated_token_ids: list[int]
    generated_logprobs: list[float]
    # Tokens already written to TQ by sealed cuts.
    staged_token_count: int
    staging_keys: tuple[str, ...]
    cut_in_progress: bool
    terminal_started: bool
    effective_output_limit: int | None
    terminal_finish_reason: Literal["stop", "length"] | None
    terminal_stop_reason: str | int | None
    observation_error: str | None
    # Set when the request's sampling settings would not carry across a cut,
    # so the call regenerates after a restore instead of resuming its prefix.
    uncuttable_reason: str | None


@dataclass(frozen=True)
class _FrozenCut:
    """The tokens one cut froze: the chunk to stage and the whole prefix."""

    chunk_key: str
    chunk_sequence: int
    chunk_token_ids: list[int]
    chunk_logprobs: list[float]
    generated_token_ids: list[int]
    generated_logprobs: list[float]


class _RequestCaptureState:
    """Append-only generated tokens for one in-flight captured request.

    Tokens are recorded lazily: a checkpoint cut reads the tokens vLLM decoded
    since the last record, and the finished output records the rest. ``record``
    keeps each position once whichever source reaches it first.

    Two locks guard the rest, and callers never take either directly. Every
    method here takes ``_lock`` itself. Lifecycle changes (freezing, sealing
    or rolling back a cut, starting the terminal write) exist only on the
    ``_CallLifecycle`` that ``lifecycle()`` yields while it holds
    ``_lifecycle_lock``, which a cut keeps across its TQ write. ``call``,
    ``prompt_token_ids``, ``media`` and ``resumed_generation_token_ids`` never
    change after ``begin_call``.

    Each successful cut stages only the tokens past the last sealed one. The
    ordered TQ keys remain in Gym lineage; the full list is kept locally to
    compute the cumulative digest advertised by the latest cut.
    """

    __slots__ = (
        "_call",
        "_prompt_token_ids",
        "_media",
        "_resumed_generation_token_ids",
        "_lock",
        "_lifecycle_lock",
        "_generated_token_ids",
        "_generated_logprobs",
        "_staged_token_count",
        "_staging_keys",
        "_frozen_cut",
        "_next_chunk_sequence",
        "_terminal_started",
        "_effective_output_limit",
        "_terminal_finish_reason",
        "_terminal_stop_reason",
        "_observation_error",
        "_uncuttable_reason",
    )

    def __init__(
        self,
        call: Any,
        prompt_token_ids: Sequence[int],
        *,
        media: Any | None = None,
        resumed_generation_token_ids: Sequence[int] = (),
        effective_output_limit: int | None = None,
        staging_keys: Sequence[str] = (),
    ) -> None:
        self._call = call
        self._prompt_token_ids = list(prompt_token_ids)
        # Backend-owned opaque state that must survive until the terminal row
        # is staged. vLLM uses this for its processed-media snapshot; prefix
        # recovery is setup-guarded to text-only, but ordinary terminal capture
        # still shares this lifecycle container.
        self._media = media
        self._resumed_generation_token_ids = list(resumed_generation_token_ids)
        self._lock = threading.Lock()
        # Serializes lifecycle-changing TQ operations for this call. Token
        # recording uses ``_lock`` instead, so a finished output can be
        # recorded while a cut stages its chunk.
        self._lifecycle_lock = threading.Lock()
        self._generated_token_ids: list[int] = []
        self._generated_logprobs: list[float] = []
        self._staged_token_count = 0
        self._staging_keys = list(staging_keys)
        # The cut in progress: its checkpoint and the token count it froze.
        self._frozen_cut: tuple[str, int] | None = None
        # Names each frozen chunk, so a rolled-back cut never reuses a TQ key.
        self._next_chunk_sequence = 0
        self._terminal_started = False
        self._effective_output_limit = effective_output_limit
        self._terminal_finish_reason: Literal["stop", "length"] | None = None
        self._terminal_stop_reason: str | int | None = None
        self._observation_error: str | None = None
        self._uncuttable_reason: str | None = None

    @property
    def call(self) -> Any:
        return self._call

    @property
    def prompt_token_ids(self) -> list[int]:
        return self._prompt_token_ids

    @property
    def media(self) -> Any | None:
        return self._media

    @property
    def resumed_generation_token_ids(self) -> list[int]:
        return self._resumed_generation_token_ids

    @property
    def recorded_token_count(self) -> int:
        with self._lock:
            return len(self._generated_token_ids)

    def record(
        self,
        generated_token_ids: Sequence[int],
        generated_logprobs: Sequence[float],
        *,
        start: int | None = None,
        finish_reason: Any = None,
        stop_reason: Any = None,
    ) -> None:
        """Record tokens read at position ``start`` (default: the end).

        A cut and the finished output both read from vLLM, so positions already
        recorded by the other are skipped rather than appended twice. Callers
        read ``start`` from ``recorded_token_count``, and tokens are never
        removed, so it never lies past the recorded end. ``finish_reason`` is
        how vLLM finished the call, if it has.
        """
        with self._lock:
            skip = 0 if start is None else len(self._generated_token_ids) - start
            self._generated_token_ids.extend(generated_token_ids[skip:])
            self._generated_logprobs.extend(generated_logprobs[skip:])
            if finish_reason is None:
                return
            if finish_reason not in ("stop", "length"):
                self._observation_error = (
                    f"unsupported terminal finish_reason {finish_reason!r}"
                )
            elif (
                self._terminal_finish_reason is not None
                and self._terminal_finish_reason != finish_reason
            ):
                self._observation_error = (
                    "conflicting terminal finish reasons: "
                    f"{self._terminal_finish_reason!r} and {finish_reason!r}"
                )
            elif not isinstance(stop_reason, (str, int, type(None))):
                self._observation_error = (
                    "terminal stop_reason must be a string, integer, or None"
                )
            else:
                self._terminal_finish_reason = finish_reason
                self._terminal_stop_reason = stop_reason

    def record_error(self, error: str) -> None:
        """Keep the call out of cuts: its recorded tokens disagree with vLLM's."""
        with self._lock:
            self._observation_error = error

    def mark_uncuttable(self, reason: str) -> None:
        with self._lock:
            self._uncuttable_reason = reason

    def set_effective_output_limit(self, limit: int) -> None:
        """Keep vLLM's resolved output budget; a restored cut's own one wins."""
        with self._lock:
            if self._effective_output_limit is None:
                self._effective_output_limit = limit

    def snapshot(self) -> _CallProgress:
        with self._lock:
            return self._progress()

    @contextlib.contextmanager
    def lifecycle(self) -> Iterator[_CallLifecycle]:
        """Own this call's lifecycle until the block ends."""
        with self._lifecycle_lock:
            yield from self._owned()

    @contextlib.contextmanager
    def try_lifecycle(self) -> Iterator[_CallLifecycle | None]:
        """Like ``lifecycle``, but yield None at once if a cut or write owns it."""
        if not self._lifecycle_lock.acquire(blocking=False):
            yield None
            return
        try:
            yield from self._owned()
        finally:
            self._lifecycle_lock.release()

    def _owned(self) -> Iterator[_CallLifecycle]:
        owned = _CallLifecycle(self)
        try:
            yield owned
        finally:
            owned._invalidate()
            # A cut never outlives the hold that froze it: one the block did
            # not seal (staging failed, or it raised) leaves its tokens unstaged.
            with self._lock:
                self._frozen_cut = None

    def _progress(self) -> _CallProgress:
        return _CallProgress(
            generated_token_ids=list(self._generated_token_ids),
            generated_logprobs=list(self._generated_logprobs),
            staged_token_count=self._staged_token_count,
            staging_keys=tuple(self._staging_keys),
            cut_in_progress=self._frozen_cut is not None,
            terminal_started=self._terminal_started,
            effective_output_limit=self._effective_output_limit,
            terminal_finish_reason=self._terminal_finish_reason,
            terminal_stop_reason=self._terminal_stop_reason,
            observation_error=self._observation_error,
            uncuttable_reason=self._uncuttable_reason,
        )


class _CallLifecycle:
    """Lifecycle changes to one call, usable only while its lifecycle is held.

    ``_RequestCaptureState.lifecycle`` is the only way to get one, and it stops
    working when that block ends, like ``DataPlaneMutationCut``.
    """

    __slots__ = ("_state", "_live")

    def __init__(self, state: _RequestCaptureState) -> None:
        self._state = state
        self._live = True

    def _invalidate(self) -> None:
        self._live = False

    def _require_live(self) -> _RequestCaptureState:
        if not self._live:
            raise RuntimeError("call lifecycle is no longer held")
        return self._state

    @property
    def terminal_started(self) -> bool:
        state = self._require_live()
        with state._lock:
            return state._terminal_started

    @property
    def staging_keys(self) -> tuple[str, ...]:
        state = self._require_live()
        with state._lock:
            return tuple(state._staging_keys)

    def begin_terminal(self) -> tuple[str, ...]:
        """Start the call's terminal write; return the chunk rows it retires."""
        state = self._require_live()
        with state._lock:
            if state._terminal_started:
                raise RuntimeError(
                    f"model call {state._call.model_call_id!r} already started "
                    "terminal token capture"
                )
            state._terminal_started = True
            return tuple(state._staging_keys)

    def freeze(self, checkpoint_id: str) -> tuple[_CallProgress, _FrozenCut | None]:
        """Freeze the recorded length for a cut, if the call can be cut.

        Returns what the call had recorded, and the frozen chunk and prefix,
        or None when the call is uncuttable, its record is inconsistent, or
        vLLM has not resolved its output limit.
        """
        state = self._require_live()
        with state._lock:
            progress = state._progress()
            if (
                progress.uncuttable_reason is not None
                or progress.observation_error is not None
                or progress.effective_output_limit is None
            ):
                return progress, None
            if state._frozen_cut is not None:
                raise RuntimeError(
                    "cannot start a generation cut while another cut is in progress"
                )
            frozen = len(state._generated_token_ids)
            staged = state._staged_token_count
            sequence = state._next_chunk_sequence
            state._next_chunk_sequence += 1
            state._frozen_cut = (checkpoint_id, frozen)
            return progress, _FrozenCut(
                chunk_key=f"active/{checkpoint_id}/{sequence}",
                chunk_sequence=sequence,
                chunk_token_ids=state._generated_token_ids[staged:frozen],
                chunk_logprobs=state._generated_logprobs[staged:frozen],
                generated_token_ids=state._generated_token_ids[:frozen],
                generated_logprobs=state._generated_logprobs[:frozen],
            )

    def seal(self, checkpoint_id: str, staged_key: str | None) -> None:
        """Count a staged cut's tokens as written to TQ and keep its row key."""
        state = self._require_live()
        with state._lock:
            state._staged_token_count = self._frozen_count(checkpoint_id)
            state._frozen_cut = None
            if staged_key is not None:
                state._staging_keys.append(staged_key)

    def rollback(self, checkpoint_id: str) -> None:
        """Abandon a cut; its tokens stay unstaged for the next one."""
        state = self._require_live()
        with state._lock:
            self._frozen_count(checkpoint_id)
            state._frozen_cut = None

    def _frozen_count(self, checkpoint_id: str) -> int:
        frozen_cut = self._state._frozen_cut
        if frozen_cut is None or frozen_cut[0] != checkpoint_id:
            raise RuntimeError(
                f"generation cut {checkpoint_id!r} does not own the frozen cut"
            )
        return frozen_cut[1]


@dataclass(frozen=True)
class _CompletedCaptureState:
    """Time-bounded terminal evidence retained across the response/cut race."""

    coords: Any
    generation_token_count: int
    effective_output_limit: int | None
    terminal_finish_reason: Literal["stop", "length"] | None
    terminal_stop_reason: str | int | None
    completed_at_monotonic: float


def _remaining_generation_limits_after_prefix(
    *,
    max_tokens: int,
    min_tokens: int | None,
    prefix_token_count: int,
) -> tuple[int, int | None]:
    """Return output limits for the suffix after restoring generated tokens."""
    return (
        max_tokens - prefix_token_count,
        None if min_tokens is None else max(0, min_tokens - prefix_token_count),
    )


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


# One character spans at most four UTF-8 bytes, hence at most four byte-level
# tokens; decoding a slightly longer tail is enough to see whether it is whole.
_PREFIX_BOUNDARY_TAIL_TOKENS = 8


def _ends_inside_character(tokenizer: Any, generated_token_ids: list[int]) -> bool:
    """Whether the generated text stops partway through a multi-byte character.

    vLLM detokenizes a resumed call's tail with a stream primed on the prompt,
    which ends with the restored prefix. A prefix ending inside a character
    makes that stream emit the whole decoded prompt as the tail's first text.
    """
    tail = generated_token_ids[-_PREFIX_BOUNDARY_TAIL_TOKENS:]
    return tokenizer.decode(tail, skip_special_tokens=False).endswith("\ufffd")


class GenerationCutCaptureMixin:
    """Registry, checkpoint-cut, and resume logic shared by generation backends.

    Methods use the capture-host state initialized by the vLLM worker and
    ``_RequestCaptureState``; nothing here touches a generation engine.
    """

    # Initialized by the vLLM worker; declared so type checkers can see the
    # insertion-ordered dicts this mixin evicts from.
    _completed_capture_calls: dict[str, _CompletedCaptureState]
    _generation_cut_receipts: dict[tuple[str, str], Any]
    _generation_cut_tokenizer: Any
    _generation_prefix_cuts_enabled: bool
    _restored_generation_prefix_calls: int
    _restored_generation_prefix_tokens: int
    _generation_cut_counts: dict[str, int]
    _obsolete_generation_cut_keys: list[str]
    _flushed_generation_cut_epoch: int

    def _pop_request_capture(self, request: Any) -> _RequestCaptureState | None:
        with self._capture_registry_lock:
            state = self._capture_calls.pop(id(request), None)
            if state is not None:
                self._capture_calls_by_model_call_id.pop(state.call.model_call_id, None)
            return state

    def _get_request_capture(self, request: Any) -> _RequestCaptureState | None:
        with self._capture_registry_lock:
            return self._capture_calls.get(id(request))

    def _remember_completed_capture(
        self,
        model_call_id: str,
        coords: Any,
        generation_token_count: int,
        *,
        effective_output_limit: int | None = None,
        terminal_finish_reason: Literal["stop", "length"] | None = None,
        terminal_stop_reason: str | int | None = None,
    ) -> None:
        # Only a generation cut reads this map; without cuts it would only grow.
        if not self._generation_prefix_cuts_enabled:
            return
        now = time.monotonic()
        with self._capture_registry_lock:
            # Dict insertion order is completion order. Remove only entries old
            # enough that the response/inventory race has certainly elapsed;
            # never discard fresh evidence because the deployment crossed an
            # arbitrary call-count threshold.
            while self._completed_capture_calls:
                oldest_id = next(iter(self._completed_capture_calls))
                oldest = self._completed_capture_calls[oldest_id]
                if (
                    now - oldest.completed_at_monotonic
                    <= _COMPLETED_CAPTURE_RETENTION_S
                ):
                    break
                self._completed_capture_calls.pop(oldest_id)
            self._completed_capture_calls.pop(model_call_id, None)
            self._completed_capture_calls[model_call_id] = _CompletedCaptureState(
                coords=coords,
                generation_token_count=generation_token_count,
                effective_output_limit=effective_output_limit,
                terminal_finish_reason=terminal_finish_reason,
                terminal_stop_reason=terminal_stop_reason,
                completed_at_monotonic=now,
            )

    def _record_generation_progress(
        self, states: list[_RequestCaptureState], *, deadline: float | None
    ) -> None:
        """Record the tokens each call decoded since its last record.

        Backend-specific: the vLLM worker reads them from its engine.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot read generation progress for a cut"
        )

    def _resolve_generation_cut(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> Any:
        """Fetch an admission's staged prefix chunks and rebuild the cumulative prefix."""
        continuation = admission.generation_cut
        # Gym requires at least one staging key per continuation.
        staging_keys = list(continuation.staging_keys)
        snapshots = self._staging_source.fetch(staging_keys)
        if len(snapshots) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        weight_versions = [snapshot.weight_version for snapshot in snapshots]
        if weight_versions != sorted(weight_versions):
            raise RuntimeError(
                "generation-cut chunk policy versions are not monotonically non-decreasing"
            )
        current_weight_version = self._rollout_weight_version
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

    def _resolve_usable_generation_cut(
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
            snapshot = self._resolve_generation_cut(admission, prefix_token_ids)
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
        self._retire_generation_cut_rows(continuation.staging_keys)
        return admission.model_copy(update={"generation_cut": None}), None

    def get_generation_prefix_counts(self) -> dict[str, int]:
        """Generation-prefix restore and cut counts since this process started.

        Returns:
            ``restored_calls`` and their total ``restored_prefix_tokens``, plus
            ``progress_reads``, ``progress_read_calls`` and the microseconds
            they waited for and held vLLM's output lock.
        """
        with self._capture_registry_lock:
            return {
                "restored_calls": self._restored_generation_prefix_calls,
                "restored_prefix_tokens": self._restored_generation_prefix_tokens,
                **self._generation_cut_counts,
            }

    @staticmethod
    def _failed_generation_cut_ack(
        prefix: Any, *, reason: str, uncut: Counter[str]
    ) -> Any:
        """Acknowledge one uncut call and tally it under ``reason``."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

        uncut[reason] += 1
        return GenerationCutPrefixAck.failure(prefix)

    def _abort_request_capture(
        self, request: Any, *, reason: str, blocking: bool = True
    ) -> bool:
        """Drop the in-flight capture state for a request that errored.

        A generation cut or a terminal write holds the call's lifecycle lock
        across a TQ round trip. With ``blocking=False`` the abort does not wait
        for it and returns False instead; it returns True once the call needs
        no abort.
        """
        state = self._get_request_capture(request)
        if state is None:
            return True
        obsolete_staging_keys: tuple[str, ...] = ()
        with state.lifecycle() if blocking else state.try_lifecycle() as owned:
            if owned is None:
                return False
            if not owned.terminal_started:
                staging_keys = owned.begin_terminal()
                popped = self._pop_request_capture(request)
                if popped is state and self.token_capture is not None:
                    coords = self.token_capture.fail_call(state.call, reason=reason)
                    self._remember_completed_capture(
                        state.call.model_call_id, coords, 0
                    )
                    obsolete_staging_keys = staging_keys
        self._retire_generation_cut_rows(obsolete_staging_keys)
        return True

    def _retire_generation_cut_rows(self, staging_keys: Sequence[str]) -> None:
        """Queue an ended call's chunk rows for the next snapshot to delete.

        A snapshot in progress may still name them, so they wait until the
        next one starts, after this one is saved.
        """
        if not staging_keys:
            return
        with self._capture_registry_lock:
            self._obsolete_generation_cut_keys.extend(staging_keys)

    def _flush_obsolete_generation_cut_rows(self, epoch: int) -> None:
        """Delete the chunk rows retired before snapshot ``epoch`` started.

        They belong to calls that ended before Gym froze this snapshot's
        inventory, which names a finished call by its terminal row only, and a
        saved snapshot keeps its own copy of TransferQueue.
        """
        with self._capture_registry_lock:
            # A late flush from an older snapshot must not delete rows retired
            # since a newer one started; that snapshot may still name them.
            if epoch <= self._flushed_generation_cut_epoch:
                return
            self._flushed_generation_cut_epoch = epoch
            staging_keys = self._obsolete_generation_cut_keys
            self._obsolete_generation_cut_keys = []
        sink = self._capture_sink
        if not staging_keys or sink is None:
            return
        try:
            sink.clear(staging_keys)
        except Exception:  # noqa: BLE001 - the calls have already ended
            # A failed clear only leaves rows the next restore clears.
            LOGGER.exception(
                "failed to clear %d obsolete generation chunk rows",
                len(staging_keys),
            )

    async def _run_generation_cut_control(
        self,
        operation: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """Run cut control independently of blocked terminal capture writes."""
        return await asyncio.get_running_loop().run_in_executor(
            self._generation_cut_control_executor,
            operation,
            *args,
        )

    def _completed_generation_cut_ack(
        self, prefix: Any, checkpoint_id: str, uncut: Counter[str]
    ) -> Any:
        """Describe a call whose terminal path won the cut/abort race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

        with self._capture_registry_lock:
            completed = self._completed_capture_calls.get(prefix.model_call_id)
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
        return self._failed_generation_cut_ack(prefix, reason=reason, uncut=uncut)

    def _checkpoint_active_generation_cut(
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
        capture = self.token_capture
        sink = self._capture_sink

        # Abort and terminal completion own the same lifecycle. Once the cut
        # owns it, the call cannot be removed until its frozen tokens are
        # staged or rolled back (leaving the block rolls back an unsealed cut).
        # If abort won first, use its terminal evidence.
        with state.lifecycle() as owned:
            with self._capture_registry_lock:
                current = self._capture_calls_by_model_call_id.get(prefix.model_call_id)
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
                return self._failed_generation_cut_ack(
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
                return self._failed_generation_cut_ack(
                    prefix, reason="observation_error", uncut=uncut
                )
            if effective_output_limit is None:
                # Checked before freezing, so the call's buffer is untouched.
                LOGGER.warning(
                    "cannot cut model call %s: vLLM did not resolve its "
                    "effective output limit",
                    prefix.model_call_id,
                )
                return self._failed_generation_cut_ack(
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
                self._generation_cut_tokenizer,
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
                return self._failed_generation_cut_ack(
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

    def _checkpoint_generation_cut(
        self, inventory: Any, *, deadline: float | None = None
    ) -> Any:
        """Stage a stable prefix for every call named by Gym's frozen inventory.

        ``deadline`` is a ``time.monotonic()`` bound. Gym stops waiting for the
        receipt at its own control deadline, so calls not reached by then are
        acknowledged as failures (they regenerate after restore) instead of
        holding the single cut-control thread for the next checkpoint.
        """
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutReceipt,
        )

        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")
        receipt_key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._capture_registry_lock:
            cached_receipt = self._generation_cut_receipts.get(receipt_key)
        if cached_receipt is not None:
            return cached_receipt
        with self._capture_registry_lock:
            active_states = [
                state
                for prefix in inventory.active_prefixes
                if (
                    state := self._capture_calls_by_model_call_id.get(
                        prefix.model_call_id
                    )
                )
                is not None
            ]
        self._record_generation_progress(active_states, deadline=deadline)
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
                    self._failed_generation_cut_ack(
                        call, reason="control_deadline", uncut=uncut
                    )
                    for call in remaining
                )
                break
            with self._capture_registry_lock:
                state = self._capture_calls_by_model_call_id.get(prefix.model_call_id)
            try:
                acknowledgement = (
                    self._checkpoint_active_generation_cut(
                        prefix, state, inventory.checkpoint_id, uncut
                    )
                    if state is not None
                    else None
                )
                if acknowledgement is None:
                    acknowledgement = self._completed_generation_cut_ack(
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
                acknowledgement = self._failed_generation_cut_ack(
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
        with self._capture_registry_lock:
            self._generation_cut_receipts[receipt_key] = receipt
            if len(self._generation_cut_receipts) > 256:
                self._generation_cut_receipts.pop(
                    next(iter(self._generation_cut_receipts))
                )
        return receipt
