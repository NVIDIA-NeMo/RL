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

"""Per-call capture state and the registry of a worker's live captured calls."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal


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


class CaptureRegistry:
    """In-flight captured calls, indexed by request object and by model call ID.

    The worker's request path uses the first index; a checkpoint cut uses the
    second, because Gym's inventory names calls by model call ID.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_request: dict[int, _RequestCaptureState] = {}
        self._by_model_call_id: dict[str, _RequestCaptureState] = {}

    def add(self, request: Any, state: _RequestCaptureState) -> None:
        with self._lock:
            if state.call.model_call_id in self._by_model_call_id:
                raise RuntimeError(
                    f"model call {state.call.model_call_id!r} is already active "
                    "in token capture"
                )
            self._by_request[id(request)] = state
            self._by_model_call_id[state.call.model_call_id] = state

    def get(self, request: Any) -> _RequestCaptureState | None:
        with self._lock:
            return self._by_request.get(id(request))

    def __contains__(self, request: Any) -> bool:
        with self._lock:
            return id(request) in self._by_request

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_request)

    def pop(self, request: Any) -> _RequestCaptureState | None:
        with self._lock:
            state = self._by_request.pop(id(request), None)
            if state is not None:
                self._by_model_call_id.pop(state.call.model_call_id, None)
            return state

    def get_by_model_call_id(self, model_call_id: str) -> _RequestCaptureState | None:
        with self._lock:
            return self._by_model_call_id.get(model_call_id)
