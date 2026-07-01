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
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any


class StreamingToolCallError(RuntimeError):
    """Base error for streaming tool-call prefill sessions."""


class StreamingToolCallSessionNotFoundError(StreamingToolCallError):
    """Raised when a streaming tool-call session does not exist."""


class StreamingToolCallPrefixMismatchError(StreamingToolCallError):
    """Raised when a prompt no longer extends the committed token prefix."""


class StreamingToolCallSessionClosedError(StreamingToolCallError):
    """Raised when work is submitted to a closed or failed session."""


@dataclass(frozen=True)
class StreamingToolCallAppendResult:
    """Result returned after one prefill chunk has completed."""

    sequence_no: int
    committed_tokens: int
    chunk_tokens: int
    dummy_tokens: int


@dataclass(frozen=True)
class StreamingToolCallCloseResult:
    """Result returned when a prefill session is closed."""

    prefix_matched: bool
    committed_tokens: int
    completed_chunks: int
    dummy_tokens: int


@dataclass
class _PendingChunk:
    token_ids: list[int]
    acknowledgement: asyncio.Future[int]


@dataclass
class _StreamingToolCallSession:
    session_id: str
    request_id: str
    input_queue: asyncio.Queue[_PendingChunk | None]
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


GenerateCallback = Callable[[AsyncIterator[Any], str], AsyncGenerator[Any, None]]
MakeStreamingInputCallback = Callable[[list[int]], Any]
CountOutputTokensCallback = Callable[[Any], int]


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
        count_output_tokens: CountOutputTokensCallback,
        max_sessions: int,
        session_ttl_seconds: float,
        stability_margin_tokens: int,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if stability_margin_tokens < 0:
            raise ValueError("stability_margin_tokens must be non-negative")

        self._generate = generate
        self._make_streaming_input = make_streaming_input
        self._count_output_tokens = count_output_tokens
        self._max_sessions = max_sessions
        self._session_ttl_seconds = session_ttl_seconds
        self._stability_margin_tokens = stability_margin_tokens
        self._sessions: dict[str, _StreamingToolCallSession] = {}
        self._lock = asyncio.Lock()
        self._accepting_sessions = True
        self._total_dummy_tokens = 0
        self._total_prefill_tokens = 0
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

    async def start(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
        sequence_no: int = 0,
    ) -> StreamingToolCallAppendResult:
        """Start a session and record its first tokenization candidate.

        A partial chat-template rendering ends with unstable closing delimiters.
        The first candidate therefore cannot prove any stable prefix by itself.
        Prefill starts after a later candidate establishes a common prefix.
        """
        self._assert_owner_loop()
        await self.expire_stale_sessions()
        async with self._lock:
            if not self._accepting_sessions:
                raise StreamingToolCallError("streaming tool-call sessions are paused")
            existing_session = self._sessions.get(session_id)
            if existing_session is not None:
                if (
                    sequence_no == existing_session.next_sequence_no - 1
                    and existing_session.last_candidate_token_ids == prompt_token_ids
                    and existing_session.last_result is not None
                ):
                    return existing_session.last_result
                raise StreamingToolCallError(
                    f"streaming tool-call session already exists: {session_id}"
                )
            if len(self._sessions) >= self._max_sessions:
                raise StreamingToolCallError(
                    "streaming tool-call session capacity has been reached"
                )

            session = _StreamingToolCallSession(
                session_id=session_id,
                request_id=f"streaming-tool-call-{session_id}",
                input_queue=asyncio.Queue(maxsize=1),
                append_lock=asyncio.Lock(),
            )
            self._sessions[session_id] = session
            session.task = asyncio.create_task(
                self._run_session(session),
                name=f"streaming-tool-call-{session_id}",
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

    async def append(
        self,
        *,
        session_id: str,
        prompt_token_ids: list[int],
        sequence_no: int,
    ) -> StreamingToolCallAppendResult:
        """Observe a tokenization and prefill only its proven stable prefix."""
        self._assert_owner_loop()
        session = self._get_session(session_id)
        async with session.append_lock:
            self._raise_if_unavailable(session)
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

            chunk_token_ids = prompt_token_ids[
                len(session.committed_token_ids) : stable_token_count
            ]
            if not chunk_token_ids:
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

    async def close(
        self, *, session_id: str, final_prompt_token_ids: list[int]
    ) -> StreamingToolCallCloseResult:
        """Validate the final prefix and cancel speculative work without waiting."""
        self._assert_owner_loop()
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise StreamingToolCallSessionNotFoundError(session_id)

        prefix_matched = _has_prefix(
            final_prompt_token_ids, session.committed_token_ids
        )
        result = StreamingToolCallCloseResult(
            prefix_matched=prefix_matched,
            committed_tokens=len(session.committed_token_ids),
            completed_chunks=session.completed_chunks,
            dummy_tokens=session.dummy_tokens,
        )
        self._cancel_session(session)
        return result

    async def abort(self, *, session_id: str) -> bool:
        """Cancel a session if present and return whether it existed."""
        self._assert_owner_loop()
        async with self._lock:
            session = self._sessions.pop(session_id, None)
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
        async with self._lock:
            stale_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.last_activity_monotonic < cutoff
            ]
            stale_sessions = [
                self._sessions.pop(session_id) for session_id in stale_ids
            ]
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

    async def _run_session(self, session: _StreamingToolCallSession) -> None:
        async def input_stream() -> AsyncGenerator[Any, None]:
            while True:
                chunk = await session.input_queue.get()
                if chunk is None:
                    return
                session.pending_acknowledgements.append(chunk.acknowledgement)
                yield self._make_streaming_input(chunk.token_ids)

        try:
            async for output in self._generate(input_stream(), session.request_id):
                output_tokens = self._count_output_tokens(output)
                for _ in range(output_tokens):
                    if not session.pending_acknowledgements:
                        break
                    acknowledgement = session.pending_acknowledgements.popleft()
                    if not acknowledgement.done():
                        acknowledgement.set_result(1)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            session.terminal_error = error
            self._fail_pending(session, error)
        else:
            if not session.closed:
                error = StreamingToolCallSessionClosedError(
                    f"streaming tool-call engine request ended early: {session.session_id}"
                )
                session.terminal_error = error
                self._fail_pending(session, error)

    def _cancel_session(self, session: _StreamingToolCallSession) -> None:
        session.closed = True
        error = StreamingToolCallSessionClosedError(session.session_id)
        self._fail_pending(session, error)
        while not session.input_queue.empty():
            queued_chunk = session.input_queue.get_nowait()
            if queued_chunk is not None and not queued_chunk.acknowledgement.done():
                queued_chunk.acknowledgement.set_exception(error)
        if session.task is not None and not session.task.done():
            session.task.cancel()

    @staticmethod
    def _fail_pending(session: _StreamingToolCallSession, error: BaseException) -> None:
        while session.pending_acknowledgements:
            acknowledgement = session.pending_acknowledgements.popleft()
            if not acknowledgement.done():
                acknowledgement.set_exception(error)


def _has_prefix(token_ids: list[int], prefix_token_ids: list[int]) -> bool:
    return token_ids[: len(prefix_token_ids)] == prefix_token_ids


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))
