import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from nemo_rl.models.generation.vllm.streaming_tool_call import (
    StreamingToolCallError,
    StreamingToolCallPrefillManager,
    StreamingToolCallPrefixMismatchError,
    StreamingToolCallSessionClosedError,
    StreamingToolCallSessionNotFoundError,
)


@dataclass(frozen=True)
class _FakeStreamingInput:
    token_ids: list[int]


@dataclass(frozen=True)
class _FakeOutput:
    token_count: int


class _FakeEngine:
    def __init__(self) -> None:
        self.received_chunks: list[list[int]] = []
        self.request_ids: list[str] = []
        self.block_outputs = False
        self.release_output = asyncio.Event()

    async def generate(
        self, inputs: AsyncIterator[_FakeStreamingInput], request_id: str
    ) -> AsyncGenerator[_FakeOutput, None]:
        self.request_ids.append(request_id)
        async for streaming_input in inputs:
            self.received_chunks.append(streaming_input.token_ids)
            if self.block_outputs:
                await self.release_output.wait()
            yield _FakeOutput(token_count=1)


def _make_manager(
    engine: _FakeEngine,
    *,
    max_sessions: int = 2,
    session_ttl_seconds: float = 60,
    stability_margin_tokens: int = 0,
) -> StreamingToolCallPrefillManager:
    return StreamingToolCallPrefillManager(
        generate=engine.generate,
        make_streaming_input=lambda token_ids: _FakeStreamingInput(token_ids),
        count_output_tokens=lambda output: output.token_count,
        max_sessions=max_sessions,
        session_ttl_seconds=session_ttl_seconds,
        stability_margin_tokens=stability_margin_tokens,
    )


@pytest.mark.asyncio
async def test_prefills_only_new_monotonic_suffix() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine)

    initial = await manager.start(
        session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=0
    )
    retried_initial = await manager.start(
        session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=0
    )
    appended = await manager.append(
        session_id="session", prompt_token_ids=[1, 2, 3, 4, 5], sequence_no=1
    )
    appended_again = await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
        sequence_no=2,
    )

    assert engine.received_chunks == [[1, 2, 3], [4, 5]]
    assert initial.committed_tokens == 0
    assert retried_initial == initial
    assert initial.chunk_tokens == 0
    assert appended.committed_tokens == 3
    assert appended.chunk_tokens == 3
    assert appended_again.committed_tokens == 5
    assert appended_again.chunk_tokens == 2
    assert appended_again.dummy_tokens == 2
    assert manager.total_prefill_tokens == 5
    assert manager.total_dummy_tokens == 2

    closed = await manager.close(
        session_id="session", final_prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    assert closed.prefix_matched
    assert closed.completed_chunks == 2
    assert closed.dummy_tokens == 2


@pytest.mark.asyncio
async def test_unchanged_candidate_is_idempotent_and_advances_sequence() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine)
    await manager.start(session_id="session", prompt_token_ids=[1, 2])

    result = await manager.append(
        session_id="session", prompt_token_ids=[1, 2], sequence_no=1
    )
    retried_result = await manager.append(
        session_id="session", prompt_token_ids=[1, 2], sequence_no=1
    )
    result_after_unchanged = await manager.append(
        session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=2
    )

    assert retried_result == result
    assert result.chunk_tokens == 2
    assert result_after_unchanged.chunk_tokens == 0
    assert engine.received_chunks == [[1, 2]]


@pytest.mark.asyncio
async def test_rejects_prefix_and_sequence_mismatches() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine)
    await manager.start(session_id="session", prompt_token_ids=[1, 2])

    with pytest.raises(StreamingToolCallError, match="expected sequence"):
        await manager.append(
            session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=3
        )

    await manager.append(
        session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=1
    )
    with pytest.raises(StreamingToolCallPrefixMismatchError, match="committed"):
        await manager.append(
            session_id="session", prompt_token_ids=[1, 9, 3], sequence_no=2
        )


@pytest.mark.asyncio
async def test_close_reports_authoritative_prefix_mismatch() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine)
    await manager.start(session_id="session", prompt_token_ids=[1, 2, 3])
    await manager.append(
        session_id="session", prompt_token_ids=[1, 2, 3, 4], sequence_no=1
    )

    result = await manager.close(
        session_id="session", final_prompt_token_ids=[1, 9, 3, 4]
    )

    assert not result.prefix_matched
    assert result.committed_tokens == 3


@pytest.mark.asyncio
async def test_close_cancels_inflight_append_without_waiting() -> None:
    engine = _FakeEngine()
    engine.block_outputs = True
    manager = _make_manager(engine)

    await manager.start(session_id="session", prompt_token_ids=[1, 2, 3])
    append_task = asyncio.create_task(
        manager.append(
            session_id="session",
            prompt_token_ids=[1, 2, 3, 4],
            sequence_no=1,
        )
    )
    while not engine.received_chunks:
        await asyncio.sleep(0)

    result = await asyncio.wait_for(
        manager.close(session_id="session", final_prompt_token_ids=[1, 2, 3, 4]),
        timeout=0.1,
    )
    assert result.committed_tokens == 0
    with pytest.raises(StreamingToolCallSessionClosedError):
        await append_task


@pytest.mark.asyncio
async def test_capacity_abort_and_missing_session() -> None:
    first_engine = _FakeEngine()
    manager = _make_manager(first_engine, max_sessions=1)
    await manager.start(session_id="first", prompt_token_ids=[1])

    with pytest.raises(StreamingToolCallError, match="capacity"):
        await manager.start(session_id="second", prompt_token_ids=[2])

    assert await manager.abort(session_id="first")
    assert not await manager.abort(session_id="first")
    with pytest.raises(StreamingToolCallSessionNotFoundError):
        await manager.append(session_id="missing", prompt_token_ids=[1], sequence_no=0)


@pytest.mark.asyncio
async def test_invalidate_all_cancels_every_session() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine)
    await manager.start(session_id="one", prompt_token_ids=[1])
    await manager.start(session_id="two", prompt_token_ids=[2])

    assert manager.active_session_count == 2
    assert await manager.invalidate_all() == 2
    assert manager.active_session_count == 0


@pytest.mark.asyncio
async def test_pause_invalidates_and_blocks_admission_until_resume() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine)
    await manager.start(session_id="one", prompt_token_ids=[1])

    assert await manager.pause_and_invalidate() == 1
    with pytest.raises(StreamingToolCallError, match="paused"):
        await manager.start(session_id="two", prompt_token_ids=[2])

    await manager.resume()
    await manager.start(session_id="two", prompt_token_ids=[2])
    assert manager.active_session_count == 1


@pytest.mark.asyncio
async def test_expires_stale_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, session_ttl_seconds=1)
    now = 100.0
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.streaming_tool_call.time.monotonic",
        lambda: now,
    )
    await manager.start(session_id="session", prompt_token_ids=[1])

    now = 102.0
    assert await manager.expire_stale_sessions() == 1
    assert manager.active_session_count == 0


@pytest.mark.asyncio
async def test_engine_failure_is_forwarded_to_append() -> None:
    async def failed_generate(
        inputs: AsyncIterator[Any], request_id: str
    ) -> AsyncGenerator[Any, None]:
        async for _ in inputs:
            raise ValueError("engine failed")
        if False:
            yield None

    manager = StreamingToolCallPrefillManager(
        generate=failed_generate,
        make_streaming_input=lambda token_ids: token_ids,
        count_output_tokens=lambda output: 1,
        max_sessions=1,
        session_ttl_seconds=60,
        stability_margin_tokens=0,
    )

    await manager.start(session_id="session", prompt_token_ids=[1])
    with pytest.raises(StreamingToolCallSessionClosedError) as error:
        await manager.append(
            session_id="session", prompt_token_ids=[1, 2], sequence_no=1
        )
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_stability_margin_holds_back_last_tokens() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, stability_margin_tokens=2)
    await manager.start(session_id="session", prompt_token_ids=[1, 2, 3, 90, 91])

    result = await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 90, 91],
        sequence_no=1,
    )

    assert result.committed_tokens == 1
    assert engine.received_chunks == [[1]]
