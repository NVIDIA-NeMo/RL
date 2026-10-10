import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm.engine_loop import EngineOwnerLoop, on_engine_loop
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


class _OwnerCheckedEngine:
    """Fake AsyncLLM that fails if any method runs off the owner loop."""

    def __init__(self, owner_loop: EngineOwnerLoop) -> None:
        self.owner_loop = owner_loop
        self.calls: list[str] = []
        self.aborted: list[str] = []
        self.model_config = MagicMock()
        self.renderer = MagicMock()
        self.input_processor = MagicMock()
        self.vllm_config = MagicMock()

    def _record(self, name: str) -> None:
        self.owner_loop.assert_owner()
        self.calls.append(name)

    async def collective_rpc(self, method: str, args: tuple[Any, ...] = ()) -> Any:
        self._record(f"rpc:{method}")
        return [f"device-{method}"]

    async def reset_prefix_cache(self) -> None:
        self._record("reset_prefix_cache")

    async def pause_generation(self, **kwargs: Any) -> None:
        self._record("pause_generation")

    async def resume_generation(self) -> None:
        self._record("resume_generation")

    async def generate(self, prompt: Any, sampling_params: Any, request_id: str):
        for step in range(3):
            self._record(f"generate:{step}")
            yield step

    async def abort(self, request_id: str) -> None:
        self._record("abort")
        self.aborted.append(request_id)

    def shutdown(self) -> None:
        self._record("shutdown")


def _worker(
    owner_loop: EngineOwnerLoop | None, engine: Any
) -> VllmAsyncGenerationWorkerImpl:
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._engine_owner_loop = owner_loop
    worker.llm = engine
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker.server_thread = None
    worker.http_server = None
    worker.http_server_task = None
    worker._sparse_refit_receiver = None
    worker.tokenizer = MagicMock()
    return worker


def test_owner_loop_runs_coroutines_on_its_thread() -> None:
    owner_loop = EngineOwnerLoop()

    async def identity() -> tuple[int, asyncio.AbstractEventLoop]:
        owner_loop.assert_owner()
        return threading.get_ident(), asyncio.get_running_loop()

    try:
        thread_id, loop = owner_loop.run(identity())
        assert thread_id == owner_loop.thread.ident
        assert loop is owner_loop.loop
        with pytest.raises(RuntimeError, match="outside its owner loop"):
            owner_loop.assert_owner()
    finally:
        owner_loop.shutdown()


def test_owner_loop_rejects_blocking_reentry_and_work_after_shutdown() -> None:
    owner_loop = EngineOwnerLoop()

    async def reenter() -> None:
        with pytest.raises(RuntimeError, match="Cannot block"):
            owner_loop.run(asyncio.sleep(0))

    owner_loop.run(reenter())
    owner_loop.shutdown()
    assert not owner_loop.thread.is_alive()
    with pytest.raises(RuntimeError, match="not running"):
        owner_loop.run(asyncio.sleep(0))


@pytest.mark.asyncio
async def test_run_async_propagates_cancellation_to_the_owner_loop() -> None:
    owner_loop = EngineOwnerLoop()
    started = threading.Event()
    cancelled = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(owner_loop.run_async(blocked()))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(cancelled.wait, 5)
    finally:
        owner_loop.shutdown()


@pytest.mark.asyncio
async def test_on_engine_loop_runs_inline_without_an_owner_loop() -> None:
    caller = asyncio.get_running_loop()

    class Host:
        _engine_owner_loop = None

        @on_engine_loop
        async def where(self) -> asyncio.AbstractEventLoop:
            return asyncio.get_running_loop()

    assert await Host().where() is caller


@pytest.mark.asyncio
async def test_engine_entry_points_run_on_the_owner_loop() -> None:
    owner_loop = EngineOwnerLoop()
    engine = _OwnerCheckedEngine(owner_loop)
    worker = _worker(owner_loop, engine)
    try:
        assert await worker.report_device_id_async() == ["device-report_device_id"]
        await worker.reset_prefix_cache_async()
        assert await worker.pause_generation_async(clear_cache=False)
        assert await worker.resume_generation_async()
        assert engine.calls == [
            "rpc:report_device_id",
            "reset_prefix_cache",
            "pause_generation",
            "resume_generation",
        ]
    finally:
        owner_loop.shutdown()


@pytest.mark.asyncio
async def test_engine_generate_steps_and_aborts_on_the_owner_loop() -> None:
    owner_loop = EngineOwnerLoop()
    engine = _OwnerCheckedEngine(owner_loop)
    worker = _worker(owner_loop, engine)
    try:
        outputs = [
            step
            async for step in worker._engine_generate(
                prompt="p", sampling_params=None, request_id="complete"
            )
        ]
        assert outputs == [0, 1, 2]

        stream = worker._engine_generate(
            prompt="p", sampling_params=None, request_id="early-close"
        )
        assert await anext(stream) == 0
        await stream.aclose()
        assert engine.aborted == ["early-close"]
    finally:
        owner_loop.shutdown()


@pytest.mark.asyncio
async def test_http_server_runs_on_owner_loop_and_stops_before_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_loop = EngineOwnerLoop()
    engine = _OwnerCheckedEngine(owner_loop)
    worker = _worker(owner_loop, engine)
    events: list[str] = []

    class Server:
        should_exit = False

        async def serve(self, sockets: Any = None) -> None:
            owner_loop.assert_owner()
            events.append("serve")
            while not self.should_exit:
                await asyncio.sleep(0.01)
            events.append("server-stopped")

    monkeypatch.setattr(
        worker, "_build_vllm_server", lambda: (Server(), "http://node:1/v1", None)
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.shutdown_telemetry",
        lambda: None,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.torch.cuda.empty_cache",
        lambda: None,
    )

    await worker._start_vllm_server_on_engine_loop()
    assert worker.base_url == "http://node:1/v1"
    for _ in range(100):
        if events:
            break
        await asyncio.sleep(0.01)
    assert events == ["serve"]

    assert await worker.shutdown()
    assert events == ["serve", "server-stopped"]
    assert engine.calls == ["rpc:cleanup", "shutdown"]
    assert worker._engine_owner_loop is None
    assert not owner_loop.thread.is_alive()
