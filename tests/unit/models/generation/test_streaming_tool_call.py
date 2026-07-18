import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from threading import Lock, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.generation.vllm.streaming_tool_call import (
    StreamingToolCallError,
    StreamingToolCallFinalizationUnavailableError,
    StreamingToolCallPrefillManager,
    StreamingToolCallPrefixMismatchError,
    StreamingToolCallPromptTooLongError,
    StreamingToolCallSessionClosedError,
    StreamingToolCallSessionNotFoundError,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


@dataclass(frozen=True)
class _FakeStreamingInput:
    token_ids: list[int]
    sampling_params: Any = None
    final: bool = False


@dataclass(frozen=True)
class _FakeOutput:
    token_count: int
    prompt_token_count: int


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
        prompt_token_count = 0
        async for streaming_input in inputs:
            self.received_chunks.append(streaming_input.token_ids)
            prompt_token_count += len(streaming_input.token_ids)
            if self.block_outputs:
                await self.release_output.wait()
            yield _FakeOutput(
                token_count=1,
                prompt_token_count=prompt_token_count,
            )


def _make_manager(
    engine: _FakeEngine,
    *,
    max_sessions: int = 2,
    session_ttl_seconds: float = 60,
    stability_margin_tokens: int = 0,
    max_prompt_tokens: int = 1024,
    cache_page_size_tokens: int | None = None,
    require_cache_page_crossing: bool = True,
) -> StreamingToolCallPrefillManager:
    return StreamingToolCallPrefillManager(
        generate=engine.generate,
        make_streaming_input=lambda token_ids: _FakeStreamingInput(token_ids),
        make_final_streaming_input=lambda token_ids, sampling_params: (
            _FakeStreamingInput(
                token_ids,
                sampling_params=sampling_params,
                final=True,
            )
        ),
        count_output_tokens=lambda output: output.token_count,
        get_prompt_token_count=lambda output: output.prompt_token_count,
        max_sessions=max_sessions,
        session_ttl_seconds=session_ttl_seconds,
        stability_margin_tokens=stability_margin_tokens,
        max_prompt_tokens=max_prompt_tokens,
        cache_page_size_tokens=cache_page_size_tokens,
        require_cache_page_crossing=require_cache_page_crossing,
    )


async def _wait_for_prefill_completion(
    manager: StreamingToolCallPrefillManager,
) -> None:
    async def wait() -> None:
        while manager.total_prefill_tokens == 0:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


def test_worker_records_streaming_and_prefix_cache_metrics() -> None:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"enable_vllm_metrics_logger": True}}
    worker._vllm_metrics_lock = Lock()
    worker.streaming_tool_call_manager = SimpleNamespace(
        total_dummy_tokens=3,
        total_prefill_tokens=40,
        total_prompt_too_long_rejections=2,
    )
    worker._reset_vllm_logger_metrics()

    worker._record_vllm_gauge_metric("vllm:num_requests_running", 2)
    worker._record_vllm_gauge_metric("vllm:num_requests_waiting", 1)
    worker._record_vllm_gauge_metric("vllm:kv_cache_usage_perc", 0.25)
    worker._record_vllm_counter_metric("vllm:generation_tokens", 100)
    worker._record_vllm_counter_metric("vllm:prefix_cache_queries", 80)
    worker._record_vllm_counter_metric("vllm:prefix_cache_hits", 48)

    assert worker.get_vllm_logger_metrics() == {
        "inflight_batch_sizes": [2],
        "num_pending_samples": [1],
        "kv_cache_usage_perc": [0.25],
        "generation_tokens": [97],
        "streaming_tool_call_dummy_tokens": [3],
        "streaming_tool_call_prefill_tokens": [40],
        "streaming_tool_call_prompt_too_long_rejections": [2],
        "prefix_cache_queries": [80],
        "prefix_cache_hits": [48],
    }

    worker.clear_vllm_logger_metrics()
    assert all(
        not metric_values for metric_values in worker.get_vllm_logger_metrics().values()
    )


@pytest.mark.asyncio
async def test_worker_configures_page_size_reported_by_vllm_workers() -> None:
    class _FakeLlm:
        async def collective_rpc(self, method: str, args: tuple[Any, ...]):
            if method == "bind_numa":
                return [True, True]
            if method == "report_kv_cache_block_size":
                return [1152, 1152]
            if method == "report_device_id":
                return ["GPU-0", "GPU-1"]
            raise AssertionError(method)

    manager = _make_manager(_FakeEngine())
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.llm = _FakeLlm()
    worker.streaming_tool_call_manager = manager
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker._mtp_load_from_disk = False

    await worker.post_init_async()

    assert manager.cache_page_size_tokens == 1152
    assert worker.vllm_device_ids == ["GPU-0", "GPU-1"]


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
async def test_sealed_session_streams_authoritative_final_suffix() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine)

    await manager.start(session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=0)
    await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4],
        sequence_no=1,
    )
    sealed = await manager.seal(
        session_id="session",
        final_prompt_token_ids=[1, 2, 3, 4, 5],
    )
    final_outputs = await manager.finalize(
        session_id="session",
        final_prompt_token_ids=[1, 2, 3, 4, 5],
        final_sampling_params="final-params",
    )

    outputs = [output async for output in final_outputs]

    assert sealed.prefix_matched
    assert sealed.committed_tokens == 3
    assert engine.received_chunks == [[1, 2, 3], [4, 5]]
    assert len(outputs) == 1
    assert manager.active_session_count == 0


@pytest.mark.asyncio
async def test_seal_rejects_oversized_final_suffix_before_engine_submission() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, max_prompt_tokens=4)

    await manager.start(session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=0)
    await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4],
        sequence_no=1,
    )

    with pytest.raises(
        StreamingToolCallPromptTooLongError,
        match="streaming prefill prompt would exceed its token limit: 5 > 4",
    ):
        await manager.seal(
            session_id="session",
            final_prompt_token_ids=[1, 2, 3, 4, 5],
        )

    assert engine.received_chunks == [[1, 2, 3]]
    assert manager.total_prompt_too_long_rejections == 1
    assert await manager.abort(session_id="session")


@pytest.mark.asyncio
async def test_seal_rejects_engine_tokens_without_append_acknowledgement() -> None:
    engine = _FakeEngine()
    engine.block_outputs = True
    manager = _make_manager(engine)

    await manager.start_background(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4],
        initial_candidate_token_ids=[1, 2, 3],
    )
    while not engine.received_chunks:
        await asyncio.sleep(0)

    session = manager._sessions["session"]
    background_task = next(iter(session.background_tasks))
    background_task.cancel()
    await asyncio.gather(background_task, return_exceptions=True)
    engine.release_output.set()
    while session.pending_acknowledgements:
        await asyncio.sleep(0)

    with pytest.raises(
        StreamingToolCallSessionClosedError,
        match=(
            "engine prompt accounting diverged: committed=0, submitted=3, observed=3"
        ),
    ):
        await manager.seal(
            session_id="session",
            final_prompt_token_ids=[1, 2, 3, 4, 5],
        )

    assert engine.received_chunks == [[1, 2, 3]]
    assert await manager.abort(session_id="session")


@pytest.mark.asyncio
async def test_seal_fails_open_without_waiting_for_inflight_prefill() -> None:
    engine = _FakeEngine()
    engine.block_outputs = True
    manager = _make_manager(engine)

    await manager.start_background(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4],
        initial_candidate_token_ids=[1, 2, 3],
    )

    with pytest.raises(
        StreamingToolCallFinalizationUnavailableError,
        match="in-flight work",
    ):
        await asyncio.wait_for(
            manager.seal(
                session_id="session",
                final_prompt_token_ids=[1, 2, 3, 4, 5],
            ),
            timeout=0.1,
        )

    assert await manager.abort(session_id="session")


@pytest.mark.asyncio
async def test_first_snapshot_prefills_authoritative_model_prefix() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, stability_margin_tokens=1)

    admitted = await manager.start(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        sequence_no=0,
        initial_candidate_token_ids=[1, 2, 3, 4],
    )
    appended = await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
        sequence_no=1,
    )

    assert engine.received_chunks == [[1, 2, 3], [4, 5]]
    assert admitted.committed_tokens == 3
    assert admitted.chunk_tokens == 3
    assert appended.committed_tokens == 5
    assert appended.chunk_tokens == 2


@pytest.mark.asyncio
async def test_background_start_returns_before_engine_completion() -> None:
    engine = _FakeEngine()
    engine.block_outputs = True
    manager = _make_manager(engine)

    started = await asyncio.wait_for(
        manager.start_background(
            session_id="session",
            prompt_token_ids=[1, 2, 3, 4],
            initial_candidate_token_ids=[1, 2, 3],
            dynamic_token_baseline=2,
        ),
        timeout=0.1,
    )

    assert started.scheduled_chunks == 1
    assert started.scheduled_tokens == 3
    assert manager.total_prefill_tokens == 0
    closed = await asyncio.wait_for(
        manager.close(
            session_id="session",
            final_prompt_token_ids=[1, 2, 3, 4, 5],
        ),
        timeout=0.1,
    )
    assert closed.committed_tokens == 0
    assert closed.background_completed_chunks == 0
    assert closed.background_cancelled_chunks == 1
    assert closed.background_cancelled_tokens == 3


@pytest.mark.asyncio
async def test_background_completion_is_settled_once_at_close() -> None:
    engine = _FakeEngine()
    engine.block_outputs = True
    manager = _make_manager(engine)

    started = await manager.start_background(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4],
        initial_candidate_token_ids=[1, 2, 3],
        dynamic_token_baseline=2,
    )
    engine.release_output.set()
    await _wait_for_prefill_completion(manager)

    closed = await manager.close(
        session_id="session",
        final_prompt_token_ids=[1, 2, 3, 4, 5],
    )

    assert started.scheduled_chunks == 1
    assert closed.prefix_matched
    assert closed.committed_tokens == 3
    assert closed.background_scheduled_tokens == 3
    assert closed.background_completed_chunks == 1
    assert closed.background_completed_tokens == 3
    assert closed.background_completed_dummy_tokens == 1
    assert closed.background_effective_chunks == 1
    assert closed.background_dynamic_tokens == 1
    assert closed.background_cancelled_chunks == 0
    assert closed.background_failed_chunks == 0
    assert closed.background_completion_seconds > 0


@pytest.mark.asyncio
async def test_background_prefill_defers_until_stable_prefix_crosses_cache_page() -> (
    None
):
    engine = _FakeEngine()
    manager = _make_manager(engine, cache_page_size_tokens=4)

    started = await manager.start_background(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
        initial_candidate_token_ids=[1, 2, 3, 4, 5, 6, 7],
        dynamic_token_baseline=6,
    )
    still_deferred = await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        sequence_no=1,
    )
    admitted = await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        sequence_no=2,
    )
    closed = await manager.close(
        session_id="session",
        final_prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )

    assert started.scheduled_chunks == 0
    assert started.scheduled_tokens == 0
    assert still_deferred.chunk_tokens == 0
    assert admitted.chunk_tokens == 8
    assert engine.received_chunks == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert closed.prefix_matched
    assert closed.committed_tokens == 8
    assert closed.background_completed_chunks == 0
    assert closed.background_cancelled_chunks == 0


@pytest.mark.asyncio
async def test_background_prefill_admits_exact_cache_page_boundary() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, cache_page_size_tokens=4)

    started = await manager.start_background(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        initial_candidate_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        dynamic_token_baseline=6,
    )
    await _wait_for_prefill_completion(manager)
    closed = await manager.close(
        session_id="session",
        final_prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    assert started.scheduled_chunks == 1
    assert started.scheduled_tokens == 8
    assert engine.received_chunks == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert closed.prefix_matched
    assert closed.committed_tokens == 8
    assert closed.background_completed_chunks == 1
    assert closed.background_cancelled_chunks == 0


@pytest.mark.asyncio
async def test_same_request_prefill_admits_without_crossing_cache_page() -> None:
    engine = _FakeEngine()
    manager = _make_manager(
        engine,
        cache_page_size_tokens=4,
        require_cache_page_crossing=False,
    )

    started = await manager.start_background(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4],
        initial_candidate_token_ids=[1, 2, 3],
        dynamic_token_baseline=3,
    )
    await _wait_for_prefill_completion(manager)
    closed = await manager.close(
        session_id="session",
        final_prompt_token_ids=[1, 2, 3, 4, 5],
    )

    assert started.scheduled_chunks == 1
    assert started.scheduled_tokens == 3
    assert engine.received_chunks == [[1, 2, 3]]
    assert closed.committed_tokens == 3
    assert closed.prefix_matched


@pytest.mark.asyncio
async def test_first_snapshot_rejects_nonmatching_model_prefix() -> None:
    manager = _make_manager(_FakeEngine())

    with pytest.raises(
        StreamingToolCallPrefixMismatchError,
        match="does not extend the initial candidate",
    ):
        await manager.start(
            session_id="session",
            prompt_token_ids=[1, 2, 3],
            sequence_no=0,
            initial_candidate_token_ids=[1, 9],
        )


@pytest.mark.asyncio
async def test_prime_prefills_exact_prompt_and_closes_session() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, stability_margin_tokens=2)

    result = await manager.prime(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5],
    )

    assert result.committed_tokens == 3
    assert result.chunk_tokens == 3
    assert result.completed_chunks == 1
    assert result.dummy_tokens == 1
    assert result.prefix_matched
    assert engine.received_chunks == [[1, 2, 3]]
    assert manager.active_session_count == 0
    assert manager.total_prefill_tokens == 3
    assert manager.total_dummy_tokens == 1


@pytest.mark.asyncio
async def test_continuation_proves_prefix_before_prefilling_mutable_output() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, stability_margin_tokens=0)

    initial = await manager.start(
        session_id="session",
        prompt_token_ids=[1, 2, 9, 10],
        sequence_no=0,
    )
    first_candidate = await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 9, 10],
        sequence_no=1,
    )
    appended = await manager.append(
        session_id="session",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 9, 10],
        sequence_no=2,
    )
    closed = await manager.close(
        session_id="session",
        final_prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 9, 10],
    )

    assert initial.committed_tokens == 0
    assert initial.chunk_tokens == 0
    assert first_candidate.committed_tokens == 2
    assert first_candidate.chunk_tokens == 2
    assert appended.committed_tokens == 4
    assert appended.chunk_tokens == 2
    assert engine.received_chunks == [[1, 2], [3, 4]]
    assert closed.prefix_matched
    assert closed.completed_chunks == 2
    assert closed.dummy_tokens == 2
    assert manager.active_session_count == 0


@pytest.mark.asyncio
async def test_rejects_chunk_before_cumulative_prompt_exceeds_limit() -> None:
    engine = _FakeEngine()
    manager = _make_manager(engine, max_prompt_tokens=3)

    await manager.start(session_id="session", prompt_token_ids=[1, 2, 3], sequence_no=0)
    accepted = await manager.append(
        session_id="session", prompt_token_ids=[1, 2, 3, 4], sequence_no=1
    )

    with pytest.raises(
        StreamingToolCallError,
        match="streaming prefill prompt would exceed its token limit: 4 > 3",
    ):
        await manager.append(
            session_id="session",
            prompt_token_ids=[1, 2, 3, 4, 5],
            sequence_no=2,
        )

    assert accepted.committed_tokens == 3
    assert engine.received_chunks == [[1, 2, 3]]
    assert manager.total_prompt_too_long_rejections == 1
    assert await manager.abort(session_id="session")


@pytest.mark.asyncio
async def test_prime_aborts_session_after_engine_failure() -> None:
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
        max_prompt_tokens=1024,
    )

    with pytest.raises(StreamingToolCallSessionClosedError) as error:
        await manager.prime(session_id="session", prompt_token_ids=[1, 2, 3])

    assert isinstance(error.value.__cause__, ValueError)
    assert manager.active_session_count == 0


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
async def test_abort_blocks_late_prefill_start() -> None:
    manager = _make_manager(_FakeEngine())

    assert not await manager.abort(session_id="session")
    with pytest.raises(StreamingToolCallSessionClosedError, match="was aborted"):
        await manager.start(session_id="session", prompt_token_ids=[1])


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
async def test_worker_pauses_sessions_on_manager_owner_loop() -> None:
    owner_loop = asyncio.new_event_loop()

    def run_owner_loop() -> None:
        asyncio.set_event_loop(owner_loop)
        owner_loop.run_forever()

    owner_thread = Thread(target=run_owner_loop)
    owner_thread.start()

    async def create_active_manager() -> StreamingToolCallPrefillManager:
        manager = _make_manager(_FakeEngine())
        manager.bind_to_current_loop()
        await manager.start(session_id="session", prompt_token_ids=[1])
        return manager

    try:
        manager_future = asyncio.run_coroutine_threadsafe(
            create_active_manager(), owner_loop
        )
        manager = manager_future.result(timeout=1)
        worker = object.__new__(VllmAsyncGenerationWorkerImpl)
        worker.streaming_tool_call_manager = manager

        assert await worker._pause_streaming_tool_call_sessions() == 1
        assert manager.active_session_count == 0
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=1)
        assert not owner_thread.is_alive()
        owner_loop.close()


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
        max_prompt_tokens=1024,
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
