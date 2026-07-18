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

import asyncio
import time
from collections import OrderedDict, deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any


class StreamingToolCallError(RuntimeError):
    """Base error for streaming tool-call prefill sessions."""


class StreamingToolCallSessionNotFoundError(StreamingToolCallError):
    """Raised when a streaming tool-call session does not exist."""


class StreamingToolCallPrefixMismatchError(StreamingToolCallError):
    """Raised when a prompt no longer extends the committed token prefix."""


class StreamingToolCallPromptTooLongError(StreamingToolCallError):
    """Raised before speculative prefill would exceed the model context."""


class StreamingToolCallSessionClosedError(StreamingToolCallError):
    """Raised when work is submitted to a closed or failed session."""


class StreamingToolCallEngineStateMismatchError(StreamingToolCallSessionClosedError):
    """Raised when local and engine streaming-prompt accounting diverge."""


class StreamingToolCallFinalizationUnavailableError(StreamingToolCallError):
    """Raised before final decode when a session cannot be reused safely."""


def validate_prompt_token_count(
    *, prompt_token_count: int, max_prompt_tokens: int
) -> None:
    """Reject an engine prompt before it can exceed its fixed token buffer."""
    if prompt_token_count > max_prompt_tokens:
        raise StreamingToolCallPromptTooLongError(
            "streaming prefill prompt would exceed its token limit: "
            f"{prompt_token_count} > {max_prompt_tokens} "
            "(maximum context length)"
        )


@dataclass(frozen=True)
class StreamingToolCallAppendResult:
    """Result returned after one prefill chunk has completed."""

    sequence_no: int
    committed_tokens: int
    chunk_tokens: int
    dummy_tokens: int


@dataclass(frozen=True)
class StreamingToolCallBackgroundStartResult:
    """Result returned once background prefill has been safely enqueued."""

    sequence_no: int
    scheduled_chunks: int
    scheduled_tokens: int
    enqueue_seconds: float


@dataclass(frozen=True)
class StreamingToolCallCloseResult:
    """Result returned when a prefill session is closed."""

    prefix_matched: bool
    committed_tokens: int
    completed_chunks: int
    dummy_tokens: int
    background_scheduled_chunks: int
    background_scheduled_tokens: int
    background_completed_chunks: int
    background_completed_tokens: int
    background_completed_dummy_tokens: int
    background_effective_chunks: int
    background_dynamic_tokens: int
    background_cancelled_chunks: int
    background_cancelled_tokens: int
    background_failed_chunks: int
    background_failed_tokens: int
    background_enqueue_seconds: float
    background_completion_seconds: float


@dataclass(frozen=True)
class StreamingToolCallPrimeResult:
    """Result returned after a one-shot prefill session completes."""

    committed_tokens: int
    chunk_tokens: int
    completed_chunks: int
    dummy_tokens: int
    prefix_matched: bool


@dataclass
class _PendingChunk:
    token_ids: list[int]
    acknowledgement: asyncio.Future[int]


@dataclass
class _PendingFinal:
    streaming_input: Any
    token_count: int


@dataclass(frozen=True)
class _FinalOutputError:
    error: BaseException


_FINAL_OUTPUT_DONE = object()


@dataclass
class _StreamingToolCallSession:
    session_id: str
    request_id: str
    input_queue: asyncio.Queue[_PendingChunk | _PendingFinal | None]
    append_lock: asyncio.Lock
    task: asyncio.Task[None] | None = None
    committed_token_ids: list[int] = field(default_factory=list)
    last_candidate_token_ids: list[int] | None = None
    last_result: StreamingToolCallAppendResult | None = None
    pending_acknowledgements: deque[asyncio.Future[int]] = field(default_factory=deque)
    next_sequence_no: int = 0
    completed_chunks: int = 0
    dummy_tokens: int = 0
    last_activity_monotonic: float = field(default_factory=time.monotonic)
    terminal_error: BaseException | None = None
    closed: bool = False
    sealed: bool = False
    sealed_prompt_token_ids: list[int] | None = None
    finalizing: bool = False
    final_output_queue: asyncio.Queue[Any] | None = None
    final_output_closed: bool = False
    engine_submitted_prompt_tokens: int = 0
    engine_observed_prompt_tokens: int | None = None
    background_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    background_start_prompt_token_ids: list[int] | None = None
    background_start_result: StreamingToolCallBackgroundStartResult | None = None
    background_scheduled_chunks: int = 0
    background_scheduled_tokens: int = 0
    background_completed_chunks: int = 0
    background_completed_tokens: int = 0
    background_completed_dummy_tokens: int = 0
    background_effective_chunks: int = 0
    background_dynamic_tokens: int = 0
    background_failed_chunks: int = 0
    background_failed_tokens: int = 0
    background_enqueue_seconds: float = 0.0
    background_completion_seconds: float = 0.0
    cache_page_baseline: int | None = None
    cache_page_gate_active: bool = False


GenerateCallback = Callable[[AsyncIterator[Any], str], AsyncGenerator[Any, None]]
MakeStreamingInputCallback = Callable[[list[int]], Any]
MakeFinalStreamingInputCallback = Callable[[list[int], Any], Any]
CountOutputTokensCallback = Callable[[Any], int]
GetPromptTokenCountCallback = Callable[[Any], int | None]


class StreamingToolCallPrefillManager:
    """Manage resumable vLLM prefill sessions for running tool calls.

    The manager is intentionally independent of vLLM types. The HTTP worker
    supplies factories for streaming inputs and engine generation, which keeps
    lifecycle and prefix-invariant tests lightweight.
    """

    def __init__(
        self,
        *,
        generate: GenerateCallback,
        make_streaming_input: MakeStreamingInputCallback,
        make_final_streaming_input: MakeFinalStreamingInputCallback | None = None,
        count_output_tokens: CountOutputTokensCallback,
        get_prompt_token_count: GetPromptTokenCountCallback | None = None,
        max_sessions: int,
        session_ttl_seconds: float,
        stability_margin_tokens: int,
        max_prompt_tokens: int,
        cache_page_size_tokens: int | None = None,
        require_cache_page_crossing: bool = True,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if stability_margin_tokens < 0:
            raise ValueError("stability_margin_tokens must be non-negative")
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be at least 1")
        if cache_page_size_tokens is not None and cache_page_size_tokens < 1:
            raise ValueError("cache_page_size_tokens must be positive when set")

        self._generate = generate
        self._make_streaming_input = make_streaming_input
        self._make_final_streaming_input = make_final_streaming_input
        self._count_output_tokens = count_output_tokens
        self._get_prompt_token_count = get_prompt_token_count
        self._max_sessions = max_sessions
        self._session_ttl_seconds = session_ttl_seconds
        self._stability_margin_tokens = stability_margin_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._cache_page_size_tokens = cache_page_size_tokens
        self._require_cache_page_crossing = require_cache_page_crossing
        self._sessions: dict[str, _StreamingToolCallSession] = {}
        self._aborted_sessions: OrderedDict[str, float] = OrderedDict()
        self._lock = asyncio.Lock()
        self._accepting_sessions = True
        self._total_dummy_tokens = 0
        self._total_prefill_tokens = 0
        self._total_prompt_too_long_rejections = 0
        self._event_loop: asyncio.AbstractEventLoop | None = None

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop | None:
        """Return the event loop that owns session tasks and synchronization."""
        return self._event_loop

    def bind_to_current_loop(self) -> None:
        """Bind manager state to the current event loop exactly once."""
        current_loop = asyncio.get_running_loop()
        if self._event_loop is None:
            self._event_loop = current_loop
        elif self._event_loop is not current_loop:
            raise RuntimeError(
                "streaming tool-call manager used from a non-owner event loop"
            )

    def _assert_owner_loop(self) -> None:
        self.bind_to_current_loop()

    @property
    def active_session_count(self) -> int:
        """Return the number of currently registered sessions."""
        return len(self._sessions)

    @property
    def total_dummy_tokens(self) -> int:
        """Return dummy decode tokens produced by streaming prefill sessions."""
        return self._total_dummy_tokens

    @property
    def total_prefill_tokens(self) -> int:
        """Return stable prompt tokens submitted by streaming sessions."""
        return self._total_prefill_tokens

    @property
    def total_prompt_too_long_rejections(self) -> int:
        """Return chunks rejected before exceeding the model context."""
        return self._total_prompt_too_long_rejections

    @property
    def cache_page_size_tokens(self) -> int | None:
        """Return the engine's effective APC page size, when available."""
        return self._cache_page_size_tokens

    def set_cache_page_size_tokens(self, cache_page_size_tokens: int) -> None:
        """Set the effective APC page size reported by the loaded vLLM worker."""
        if cache_page_size_tokens < 1:
            raise ValueError("cache_page_size_tokens must be positive")
        self._cache_page_size_tokens = cache_page_size_tokens

    def _validate_prompt_token_count(self, prompt_token_count: int) -> None:
        try:
            validate_prompt_token_count(
                prompt_token_count=prompt_token_count,
                max_prompt_tokens=self._max_prompt_tokens,
            )
        except StreamingToolCallPromptTooLongError:
            self._total_prompt_too_long_rejections += 1
            raise

    @staticmethod
    def _validate_engine_prompt_accounting(
        session: _StreamingToolCallSession,
    ) -> None:
        """Require the engine session to match the acknowledged token prefix.

        A streaming input is visible to vLLM before its dummy decode output
        acknowledges the append. If the append task is cancelled in that
        interval, the engine can retain tokens that were never committed by
        the manager. Reusing that session would append the final suffix at the
        wrong offset and can overrun vLLM's fixed prompt buffer.
        """
        committed_tokens = len(session.committed_token_ids)
        submitted_tokens = session.engine_submitted_prompt_tokens
        observed_tokens = session.engine_observed_prompt_tokens
        if submitted_tokens != committed_tokens or observed_tokens not in (
            None,
            committed_tokens,
        ):
            raise StreamingToolCallEngineStateMismatchError(
                "streaming tool-call engine prompt accounting diverged: "
                f"committed={committed_tokens}, submitted={submitted_tokens}, "
                f"observed={observed_tokens}"
            )

    @staticmethod
    def _validate_initial_candidate(
        prompt_token_ids: list[int],
        initial_candidate_token_ids: list[int] | None,
    ) -> None:
        if initial_candidate_token_ids is None:
            return
        if not initial_candidate_token_ids:
            raise StreamingToolCallPrefixMismatchError(
                "the initial candidate token prefix is empty"
            )
        if not _has_prefix(prompt_token_ids, initial_candidate_token_ids):
            raise StreamingToolCallPrefixMismatchError(
                "the prompt does not extend the initial candidate token prefix"
            )

    async def _get_or_create_session(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
        initial_candidate_token_ids: list[int] | None,
    ) -> tuple[_StreamingToolCallSession, bool]:
        self._assert_owner_loop()
        self._validate_initial_candidate(prompt_token_ids, initial_candidate_token_ids)
        await self.expire_stale_sessions()
        async with self._lock:
            if not self._accepting_sessions:
                raise StreamingToolCallError("streaming tool-call sessions are paused")
            if session_id in self._aborted_sessions:
                raise StreamingToolCallSessionClosedError(
                    f"streaming tool-call session was aborted: {session_id}"
                )
            existing_session = self._sessions.get(session_id)
            if existing_session is not None:
                return existing_session, False
            if len(self._sessions) >= self._max_sessions:
                raise StreamingToolCallError(
                    "streaming tool-call session capacity has been reached"
                )

            session = _StreamingToolCallSession(
                session_id=session_id,
                request_id=f"streaming-tool-call-{session_id}",
                input_queue=asyncio.Queue(maxsize=1),
                append_lock=asyncio.Lock(),
                last_candidate_token_ids=(
                    list(initial_candidate_token_ids)
                    if initial_candidate_token_ids is not None
                    else None
                ),
            )
            self._sessions[session_id] = session
            session.task = asyncio.create_task(
                self._run_session(session),
                name=f"streaming-tool-call-{session_id}",
            )
            return session, True

    async def start(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
        sequence_no: int = 0,
        initial_candidate_token_ids: list[int] | None = None,
    ) -> StreamingToolCallAppendResult:
        """Start a session and record its first tokenization candidate.

        A partial chat-template rendering ends with unstable closing delimiters.
        The first candidate therefore cannot prove any stable prefix by itself.
        Prefill starts after a later candidate establishes a common prefix. An
        authoritative model prefix can be supplied as that earlier candidate so
        the first streamed tool snapshot admits useful prefill immediately.
        """
        session, created = await self._get_or_create_session(
            session_id=session_id,
            prompt_token_ids=prompt_token_ids,
            initial_candidate_token_ids=initial_candidate_token_ids,
        )
        if not created:
            if (
                sequence_no == session.next_sequence_no - 1
                and session.last_candidate_token_ids == prompt_token_ids
                and session.last_result is not None
            ):
                return session.last_result
            raise StreamingToolCallError(
                f"streaming tool-call session already exists: {session_id}"
            )

        try:
            return await self.append(
                session_id=session_id,
                prompt_token_ids=prompt_token_ids,
                sequence_no=sequence_no,
            )
        except asyncio.CancelledError:
            await self.abort(session_id=session_id)
            raise
        except Exception:
            await self.abort(session_id=session_id)
            raise

    async def start_background(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
        sequence_no: int = 0,
        initial_candidate_token_ids: list[int] | None = None,
        dynamic_token_baseline: int = 0,
    ) -> StreamingToolCallBackgroundStartResult:
        """Start a session and return after its first chunk is safely enqueued.

        Engine completion remains owned by the manager. A later close reports
        whether the queued chunk completed before the authoritative tool output
        arrived, so the HTTP request does not have to race the command runtime.
        """
        if dynamic_token_baseline < 0:
            raise ValueError("dynamic_token_baseline must be non-negative")
        started_at = time.perf_counter()
        session, created = await self._get_or_create_session(
            session_id=session_id,
            prompt_token_ids=prompt_token_ids,
            initial_candidate_token_ids=initial_candidate_token_ids,
        )
        if not created:
            if (
                session.background_start_prompt_token_ids == prompt_token_ids
                and session.background_start_result is not None
            ):
                return session.background_start_result
            raise StreamingToolCallError(
                f"streaming tool-call session already exists: {session_id}"
            )

        session.background_start_prompt_token_ids = list(prompt_token_ids)
        if (
            self._require_cache_page_crossing
            and self._cache_page_size_tokens is not None
        ):
            session.cache_page_baseline = dynamic_token_baseline
            session.cache_page_gate_active = True
        loop = asyncio.get_running_loop()
        enqueued: asyncio.Future[int] = loop.create_future()
        task = asyncio.create_task(
            self._run_background_append(
                session=session,
                prompt_token_ids=prompt_token_ids,
                sequence_no=sequence_no,
                dynamic_token_baseline=dynamic_token_baseline,
                enqueued=enqueued,
            ),
            name=f"streaming-tool-call-background-{session_id}-{sequence_no}",
        )
        session.background_tasks.add(task)
        task.add_done_callback(
            lambda completed_task: self._consume_background_task(
                session, completed_task
            )
        )

        try:
            scheduled_tokens = await enqueued
        except asyncio.CancelledError:
            await self.abort(session_id=session_id)
            raise
        except Exception:
            await self.abort(session_id=session_id)
            raise

        enqueue_seconds = time.perf_counter() - started_at
        scheduled_chunks = int(scheduled_tokens > 0)
        session.background_scheduled_chunks += scheduled_chunks
        session.background_scheduled_tokens += scheduled_tokens
        session.background_enqueue_seconds += enqueue_seconds
        result = StreamingToolCallBackgroundStartResult(
            sequence_no=sequence_no,
            scheduled_chunks=scheduled_chunks,
            scheduled_tokens=scheduled_tokens,
            enqueue_seconds=enqueue_seconds,
        )
        session.background_start_result = result
        return result

    async def append(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
        sequence_no: int,
    ) -> StreamingToolCallAppendResult:
        """Observe a tokenization and prefill only its proven stable prefix."""
        return await self._append(
            session_id=session_id,
            prompt_token_ids=prompt_token_ids,
            sequence_no=sequence_no,
            enqueued=None,
        )

    async def _append(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
        sequence_no: int,
        enqueued: asyncio.Future[int] | None,
        cache_page_baseline: int | None = None,
    ) -> StreamingToolCallAppendResult:
        self._assert_owner_loop()
        session = self._get_session(session_id)
        async with session.append_lock:
            self._raise_if_unavailable(session)
            self._validate_engine_prompt_accounting(session)
            if session.sealed:
                raise StreamingToolCallSessionClosedError(
                    f"streaming tool-call session is sealed: {session_id}"
                )
            if sequence_no == session.next_sequence_no - 1:
                if (
                    session.last_candidate_token_ids == prompt_token_ids
                    and session.last_result is not None
                ):
                    return session.last_result
                raise StreamingToolCallError(
                    f"sequence {sequence_no} was already used with different tokens"
                )
            if sequence_no != session.next_sequence_no:
                raise StreamingToolCallError(
                    f"expected sequence {session.next_sequence_no}, got {sequence_no}"
                )
            if not _has_prefix(prompt_token_ids, session.committed_token_ids):
                raise StreamingToolCallPrefixMismatchError(
                    "prompt token IDs do not extend the committed session prefix"
                )

            if session.last_candidate_token_ids is None:
                stable_token_count = 0
            else:
                common_prefix_tokens = _common_prefix_length(
                    session.last_candidate_token_ids, prompt_token_ids
                )
                if common_prefix_tokens < len(session.committed_token_ids):
                    raise StreamingToolCallPrefixMismatchError(
                        "new tokenization changed the committed session prefix"
                    )
                stable_token_count = max(
                    len(session.committed_token_ids),
                    common_prefix_tokens - self._stability_margin_tokens,
                )

            self._validate_prompt_token_count(stable_token_count)

            # Avoid starting GPU work until the proven stable prefix extends
            # the cacheable prefix by at least one full engine page. If the
            # first background snapshot is too short, the session retains this
            # gate across later observations. Once a chunk crosses the page,
            # later partial pages can accumulate in the same engine request.
            effective_cache_page_baseline = cache_page_baseline
            if effective_cache_page_baseline is None and session.cache_page_gate_active:
                effective_cache_page_baseline = session.cache_page_baseline
            if (
                effective_cache_page_baseline is not None
                and self._cache_page_size_tokens is not None
                and stable_token_count // self._cache_page_size_tokens
                <= effective_cache_page_baseline // self._cache_page_size_tokens
            ):
                if enqueued is not None and not enqueued.done():
                    enqueued.set_result(0)
                result = StreamingToolCallAppendResult(
                    sequence_no=sequence_no,
                    committed_tokens=len(session.committed_token_ids),
                    chunk_tokens=0,
                    dummy_tokens=session.dummy_tokens,
                )
                self._complete_observation(
                    session,
                    prompt_token_ids=prompt_token_ids,
                    result=result,
                )
                return result

            chunk_token_ids = prompt_token_ids[
                len(session.committed_token_ids) : stable_token_count
            ]
            if not chunk_token_ids:
                if enqueued is not None and not enqueued.done():
                    enqueued.set_result(0)
                result = StreamingToolCallAppendResult(
                    sequence_no=sequence_no,
                    committed_tokens=len(session.committed_token_ids),
                    chunk_tokens=0,
                    dummy_tokens=session.dummy_tokens,
                )
                self._complete_observation(
                    session,
                    prompt_token_ids=prompt_token_ids,
                    result=result,
                )
                return result

            loop = asyncio.get_running_loop()
            acknowledgement: asyncio.Future[int] = loop.create_future()
            chunk = _PendingChunk(
                token_ids=chunk_token_ids,
                acknowledgement=acknowledgement,
            )
            await session.input_queue.put(chunk)
            # The first admitted chunk has crossed a full APC page. Later
            # continuation chunks may now accumulate partial pages inside this
            # same streaming engine request without paying a new request cost.
            session.cache_page_gate_active = False
            if enqueued is not None and not enqueued.done():
                enqueued.set_result(len(chunk_token_ids))
            session.last_activity_monotonic = time.monotonic()

            try:
                dummy_tokens = await acknowledgement
            except asyncio.CancelledError:
                raise
            except Exception:
                self._raise_if_unavailable(session)
                raise
            self._raise_if_unavailable(session)
            session.committed_token_ids.extend(chunk_token_ids)
            session.completed_chunks += 1
            session.dummy_tokens += dummy_tokens
            self._total_dummy_tokens += dummy_tokens
            self._total_prefill_tokens += len(chunk_token_ids)
            session.last_activity_monotonic = time.monotonic()

            result = StreamingToolCallAppendResult(
                sequence_no=sequence_no,
                committed_tokens=len(session.committed_token_ids),
                chunk_tokens=len(chunk_token_ids),
                dummy_tokens=session.dummy_tokens,
            )
            self._complete_observation(
                session,
                prompt_token_ids=prompt_token_ids,
                result=result,
            )
            return result

    async def prime(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
    ) -> StreamingToolCallPrimeResult:
        """Prefill one exact prompt without exposing session round trips."""
        await self.start(
            session_id=session_id,
            prompt_token_ids=prompt_token_ids,
            sequence_no=0,
        )
        try:
            appended = await self.append(
                session_id=session_id,
                prompt_token_ids=prompt_token_ids,
                sequence_no=1,
            )
            closed = await self.close(
                session_id=session_id,
                final_prompt_token_ids=prompt_token_ids,
            )
        except asyncio.CancelledError:
            await self.abort(session_id=session_id)
            raise
        except StreamingToolCallError:
            await self.abort(session_id=session_id)
            raise
        return StreamingToolCallPrimeResult(
            committed_tokens=closed.committed_tokens,
            chunk_tokens=appended.chunk_tokens,
            completed_chunks=closed.completed_chunks,
            dummy_tokens=closed.dummy_tokens,
            prefix_matched=closed.prefix_matched,
        )

    @staticmethod
    def _make_close_result(
        session: _StreamingToolCallSession,
        *,
        final_prompt_token_ids: list[int],
    ) -> StreamingToolCallCloseResult:
        prefix_matched = _has_prefix(
            final_prompt_token_ids, session.committed_token_ids
        )
        background_cancelled_chunks = max(
            0,
            session.background_scheduled_chunks
            - session.background_completed_chunks
            - session.background_failed_chunks,
        )
        background_cancelled_tokens = max(
            0,
            session.background_scheduled_tokens
            - session.background_completed_tokens
            - session.background_failed_tokens,
        )
        return StreamingToolCallCloseResult(
            prefix_matched=prefix_matched,
            committed_tokens=len(session.committed_token_ids),
            completed_chunks=session.completed_chunks,
            dummy_tokens=session.dummy_tokens,
            background_scheduled_chunks=session.background_scheduled_chunks,
            background_scheduled_tokens=session.background_scheduled_tokens,
            background_completed_chunks=session.background_completed_chunks,
            background_completed_tokens=session.background_completed_tokens,
            background_completed_dummy_tokens=(
                session.background_completed_dummy_tokens
            ),
            background_effective_chunks=session.background_effective_chunks,
            background_dynamic_tokens=session.background_dynamic_tokens,
            background_cancelled_chunks=background_cancelled_chunks,
            background_cancelled_tokens=background_cancelled_tokens,
            background_failed_chunks=session.background_failed_chunks,
            background_failed_tokens=session.background_failed_tokens,
            background_enqueue_seconds=session.background_enqueue_seconds,
            background_completion_seconds=session.background_completion_seconds,
        )

    async def seal(
        self, *, session_id: str, final_prompt_token_ids: list[int]
    ) -> StreamingToolCallCloseResult:
        """Seal a settled session for one authoritative same-request decode.

        This method never waits for speculative GPU work. Callers can therefore
        fail open to an ordinary request without adding background completion
        latency to the model-call critical path.
        """
        self._assert_owner_loop()
        await asyncio.sleep(0)
        session = self._get_session(session_id)
        self._raise_if_unavailable(session)
        self._validate_prompt_token_count(len(final_prompt_token_ids))
        if session.sealed or session.finalizing:
            raise StreamingToolCallFinalizationUnavailableError(
                f"streaming tool-call session is already finalizing: {session_id}"
            )
        if (
            session.append_lock.locked()
            or session.pending_acknowledgements
            or not session.input_queue.empty()
            or any(not task.done() for task in session.background_tasks)
        ):
            raise StreamingToolCallFinalizationUnavailableError(
                f"streaming tool-call session still has in-flight work: {session_id}"
            )
        self._validate_engine_prompt_accounting(session)
        if session.completed_chunks <= 0:
            raise StreamingToolCallFinalizationUnavailableError(
                f"streaming tool-call session completed no prefill: {session_id}"
            )
        if not _has_prefix(final_prompt_token_ids, session.committed_token_ids):
            raise StreamingToolCallPrefixMismatchError(
                "final prompt token IDs do not extend the committed session prefix"
            )
        if len(final_prompt_token_ids) == len(session.committed_token_ids):
            raise StreamingToolCallFinalizationUnavailableError(
                f"streaming tool-call final suffix is empty: {session_id}"
            )

        session.sealed = True
        session.sealed_prompt_token_ids = list(final_prompt_token_ids)
        session.last_activity_monotonic = time.monotonic()
        return self._make_close_result(
            session,
            final_prompt_token_ids=final_prompt_token_ids,
        )

    async def finalize(
        self,
        *,
        session_id: str,
        final_prompt_token_ids: list[int],
        final_sampling_params: Any,
    ) -> AsyncGenerator[Any, None]:
        """Append the final prompt suffix and return its engine output stream."""
        self._assert_owner_loop()
        if self._make_final_streaming_input is None:
            raise StreamingToolCallFinalizationUnavailableError(
                "same-request final streaming input is not configured"
            )
        self._validate_prompt_token_count(len(final_prompt_token_ids))

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise StreamingToolCallSessionNotFoundError(session_id)
            self._raise_if_unavailable(session)
            if not session.sealed or session.sealed_prompt_token_ids is None:
                raise StreamingToolCallFinalizationUnavailableError(
                    f"streaming tool-call session is not sealed: {session_id}"
                )
            if session.finalizing:
                raise StreamingToolCallFinalizationUnavailableError(
                    f"streaming tool-call session is already finalizing: {session_id}"
                )
            if session.sealed_prompt_token_ids != final_prompt_token_ids:
                raise StreamingToolCallPrefixMismatchError(
                    "final model-call tokens differ from the sealed prompt tokens"
                )
            if (
                session.append_lock.locked()
                or session.pending_acknowledgements
                or not session.input_queue.empty()
                or any(not task.done() for task in session.background_tasks)
            ):
                raise StreamingToolCallFinalizationUnavailableError(
                    f"streaming tool-call session has late in-flight work: {session_id}"
                )
            self._validate_engine_prompt_accounting(session)

            final_suffix_token_ids = final_prompt_token_ids[
                len(session.committed_token_ids) :
            ]
            if not final_suffix_token_ids:
                raise StreamingToolCallFinalizationUnavailableError(
                    f"streaming tool-call final suffix is empty: {session_id}"
                )
            session.finalizing = True
            session.final_output_queue = asyncio.Queue()
            final_input = self._make_final_streaming_input(
                final_suffix_token_ids,
                final_sampling_params,
            )
            session.input_queue.put_nowait(
                _PendingFinal(
                    streaming_input=final_input,
                    token_count=len(final_suffix_token_ids),
                )
            )
            session.last_activity_monotonic = time.monotonic()

        return self._iterate_final_outputs(session)

    async def _iterate_final_outputs(
        self, session: _StreamingToolCallSession
    ) -> AsyncGenerator[Any, None]:
        output_queue = session.final_output_queue
        assert output_queue is not None
        try:
            while True:
                output = await output_queue.get()
                if output is _FINAL_OUTPUT_DONE:
                    return
                if isinstance(output, _FinalOutputError):
                    raise StreamingToolCallSessionClosedError(
                        f"streaming tool-call final decode failed: {session.session_id}"
                    ) from output.error
                yield output
        finally:
            async with self._lock:
                if self._sessions.get(session.session_id) is session:
                    self._sessions.pop(session.session_id)
            self._cancel_session(session)

    async def close(
        self, *, session_id: str, final_prompt_token_ids: list[int]
    ) -> StreamingToolCallCloseResult:
        """Validate the final prefix and cancel speculative work without waiting."""
        self._assert_owner_loop()
        # Let an acknowledgement that is already runnable commit before the
        # zero-grace close snapshots counters. This yields once but never waits
        # for new engine work.
        await asyncio.sleep(0)
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise StreamingToolCallSessionNotFoundError(session_id)

        result = self._make_close_result(
            session,
            final_prompt_token_ids=final_prompt_token_ids,
        )
        self._cancel_session(session)
        return result

    async def abort(self, *, session_id: str) -> bool:
        """Cancel present or late-arriving work for one session ID."""
        self._assert_owner_loop()
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            self._aborted_sessions.pop(session_id, None)
            self._aborted_sessions[session_id] = (
                time.monotonic() + self._session_ttl_seconds
            )
            while len(self._aborted_sessions) > self._max_sessions:
                self._aborted_sessions.popitem(last=False)
        if session is None:
            return False
        self._cancel_session(session)
        return True

    async def invalidate_all(self) -> int:
        """Cancel every session, for example before a model weight update."""
        self._assert_owner_loop()
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            self._cancel_session(session)
        return len(sessions)

    async def pause_and_invalidate(self) -> int:
        """Atomically stop admission and cancel every active session."""
        self._assert_owner_loop()
        async with self._lock:
            self._accepting_sessions = False
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            self._cancel_session(session)
        tasks = [session.task for session in sessions if session.task is not None]
        tasks.extend(task for session in sessions for task in session.background_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(sessions)

    async def resume(self) -> None:
        """Allow new sessions after a model lifecycle transition completes."""
        self._assert_owner_loop()
        async with self._lock:
            self._accepting_sessions = True

    async def expire_stale_sessions(self) -> int:
        """Cancel sessions that have not made progress within the configured TTL."""
        self._assert_owner_loop()
        cutoff = time.monotonic() - self._session_ttl_seconds
        now = time.monotonic()
        async with self._lock:
            stale_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.last_activity_monotonic < cutoff
            ]
            stale_sessions = [
                self._sessions.pop(session_id) for session_id in stale_ids
            ]
            while self._aborted_sessions:
                session_id, expires_at = next(iter(self._aborted_sessions.items()))
                if expires_at > now:
                    break
                self._aborted_sessions.pop(session_id)
        for session in stale_sessions:
            self._cancel_session(session)
        return len(stale_sessions)

    async def wait_for_cleanup(self) -> None:
        """Yield once so cancellation reaches session tasks in tests and shutdown."""
        self._assert_owner_loop()
        await asyncio.sleep(0)

    def _get_session(self, session_id: str) -> _StreamingToolCallSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise StreamingToolCallSessionNotFoundError(session_id)
        return session

    @staticmethod
    def _complete_observation(
        session: _StreamingToolCallSession,
        *,
        prompt_token_ids: list[int],
        result: StreamingToolCallAppendResult,
    ) -> None:
        session.last_candidate_token_ids = list(prompt_token_ids)
        session.last_result = result
        session.next_sequence_no += 1
        session.last_activity_monotonic = time.monotonic()

    @staticmethod
    def _raise_if_unavailable(session: _StreamingToolCallSession) -> None:
        if session.terminal_error is not None:
            raise StreamingToolCallSessionClosedError(
                f"streaming tool-call session failed: {session.session_id}"
            ) from session.terminal_error
        if session.closed:
            raise StreamingToolCallSessionClosedError(session.session_id)

    async def _run_background_append(
        self,
        *,
        session: _StreamingToolCallSession,
        prompt_token_ids: list[int],
        sequence_no: int,
        dynamic_token_baseline: int,
        enqueued: asyncio.Future[int],
    ) -> None:
        completion_started_at = time.perf_counter()
        dummy_tokens_before = session.dummy_tokens
        try:
            result = await self._append(
                session_id=session.session_id,
                prompt_token_ids=prompt_token_ids,
                sequence_no=sequence_no,
                enqueued=enqueued,
                cache_page_baseline=(
                    dynamic_token_baseline
                    if self._require_cache_page_crossing
                    else None
                ),
            )
        except asyncio.CancelledError:
            if not enqueued.done():
                enqueued.cancel()
            raise
        except Exception as error:
            if not enqueued.done():
                enqueued.set_exception(error)
            else:
                scheduled_tokens = enqueued.result()
                session.background_failed_chunks += int(scheduled_tokens > 0)
                session.background_failed_tokens += scheduled_tokens
            raise

        if result.chunk_tokens <= 0:
            return
        committed_tokens_before = result.committed_tokens - result.chunk_tokens
        dynamic_tokens_before = max(0, committed_tokens_before - dynamic_token_baseline)
        dynamic_tokens_after = max(0, result.committed_tokens - dynamic_token_baseline)
        dynamic_tokens = dynamic_tokens_after - dynamic_tokens_before
        completed_dummy_tokens = result.dummy_tokens - dummy_tokens_before
        session.background_completed_chunks += 1
        session.background_completed_tokens += result.chunk_tokens
        session.background_completed_dummy_tokens += completed_dummy_tokens
        session.background_effective_chunks += int(dynamic_tokens > 0)
        session.background_dynamic_tokens += dynamic_tokens
        session.background_completion_seconds += (
            time.perf_counter() - completion_started_at
        )

    @staticmethod
    def _consume_background_task(
        session: _StreamingToolCallSession,
        task: asyncio.Task[None],
    ) -> None:
        session.background_tasks.discard(task)
        if task.cancelled():
            return
        # Retrieving the exception prevents an unobserved task warning. The
        # session and close-result counters retain the actionable failure data.
        task.exception()

    async def _run_session(self, session: _StreamingToolCallSession) -> None:
        async def input_stream() -> AsyncGenerator[Any, None]:
            while True:
                chunk = await session.input_queue.get()
                if chunk is None:
                    return
                if isinstance(chunk, _PendingFinal):
                    submitted_tokens = (
                        session.engine_submitted_prompt_tokens + chunk.token_count
                    )
                    self._validate_prompt_token_count(submitted_tokens)
                    session.engine_submitted_prompt_tokens = submitted_tokens
                    yield chunk.streaming_input
                    return
                submitted_tokens = session.engine_submitted_prompt_tokens + len(
                    chunk.token_ids
                )
                try:
                    self._validate_prompt_token_count(submitted_tokens)
                except Exception as error:
                    if not chunk.acknowledgement.done():
                        chunk.acknowledgement.set_exception(error)
                    raise
                session.engine_submitted_prompt_tokens = submitted_tokens
                session.pending_acknowledgements.append(chunk.acknowledgement)
                yield self._make_streaming_input(chunk.token_ids)

        try:
            async for output in self._generate(input_stream(), session.request_id):
                if self._get_prompt_token_count is not None:
                    observed_prompt_tokens = self._get_prompt_token_count(output)
                    if observed_prompt_tokens is not None:
                        session.engine_observed_prompt_tokens = observed_prompt_tokens
                        if (
                            observed_prompt_tokens
                            != session.engine_submitted_prompt_tokens
                        ):
                            raise StreamingToolCallEngineStateMismatchError(
                                "vLLM reported an unexpected streaming prompt length: "
                                f"submitted={session.engine_submitted_prompt_tokens}, "
                                f"observed={observed_prompt_tokens}"
                            )
                if session.finalizing:
                    assert session.final_output_queue is not None
                    session.final_output_queue.put_nowait(output)
                    continue
                output_tokens = self._count_output_tokens(output)
                for _ in range(output_tokens):
                    if not session.pending_acknowledgements:
                        break
                    acknowledgement = session.pending_acknowledgements.popleft()
                    if not acknowledgement.done():
                        acknowledgement.set_result(1)
        except asyncio.CancelledError as error:
            self._close_final_output(session, error=error)
            raise
        except Exception as error:
            session.terminal_error = error
            self._fail_pending(session, error)
            self._close_final_output(session, error=error)
        else:
            if session.finalizing:
                self._close_final_output(session)
            elif not session.closed:
                error = StreamingToolCallSessionClosedError(
                    f"streaming tool-call engine request ended early: {session.session_id}"
                )
                session.terminal_error = error
                self._fail_pending(session, error)

    def _cancel_session(self, session: _StreamingToolCallSession) -> None:
        session.closed = True
        error = StreamingToolCallSessionClosedError(session.session_id)
        self._fail_pending(session, error)
        self._close_final_output(session, error=error)
        while not session.input_queue.empty():
            queued_chunk = session.input_queue.get_nowait()
            if (
                isinstance(queued_chunk, _PendingChunk)
                and not queued_chunk.acknowledgement.done()
            ):
                queued_chunk.acknowledgement.set_exception(error)
        if session.task is not None and not session.task.done():
            session.task.cancel()
        for task in tuple(session.background_tasks):
            if not task.done():
                task.cancel()

    @staticmethod
    def _fail_pending(session: _StreamingToolCallSession, error: BaseException) -> None:
        while session.pending_acknowledgements:
            acknowledgement = session.pending_acknowledgements.popleft()
            if not acknowledgement.done():
                acknowledgement.set_exception(error)

    @staticmethod
    def _close_final_output(
        session: _StreamingToolCallSession,
        error: BaseException | None = None,
    ) -> None:
        if session.final_output_queue is None or session.final_output_closed:
            return
        if error is not None:
            session.final_output_queue.put_nowait(_FinalOutputError(error))
        session.final_output_queue.put_nowait(_FINAL_OUTPUT_DONE)
        session.final_output_closed = True


def _has_prefix(token_ids: list[int], prefix_token_ids: list[int]) -> bool:
    return token_ids[: len(prefix_token_ids)] == prefix_token_ids


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))
