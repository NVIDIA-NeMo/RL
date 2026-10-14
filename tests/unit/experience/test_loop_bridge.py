# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU tests using real, separate engine and HTTP event loops."""

import asyncio
import threading
from collections.abc import AsyncGenerator, Iterator
from types import SimpleNamespace as NS
from typing import Literal

import pytest
from pydantic import ValidationError

from nemo_rl.experience.direct_capture import (
    DirectCapture,
    DirectCaptureConfig,
    current_capture,
)
from nemo_rl.experience.loop_bridge import HTTPBridgeConfig, queued_iterator
from nemo_rl.experience.vllm_http_client import _AsyncLLMHTTPClient


@pytest.fixture
def engine_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run)
    thread.start()
    assert ready.wait(5)
    try:
        yield loop
    finally:

        async def check_shutdown() -> None:
            remaining = asyncio.all_tasks() - {asyncio.current_task()}
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
            assert not remaining, "bridge leaked producer tasks"

        try:
            asyncio.run_coroutine_threadsafe(check_shutdown(), loop).result(5)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            assert not thread.is_alive()
            loop.close()


@pytest.mark.parametrize("capacity", [1, 16])
@pytest.mark.parametrize("concurrent", [1, 16])
def test_order_identity_and_no_lost_wakeups(
    engine_loop: asyncio.AbstractEventLoop, capacity: int, concurrent: int
) -> None:
    async def consume() -> None:
        items = [object() for _ in range(256)]

        async def source() -> AsyncGenerator[object, None]:
            assert asyncio.get_running_loop() is engine_loop
            for index, item in enumerate(items):
                if index % 3 == 0:
                    await asyncio.sleep(0)
                yield item

        received = []
        async for item in queued_iterator(source(), engine_loop, capacity=capacity):
            received.append(item)
            if len(received) % 5 == 0:
                await asyncio.sleep(0)
        assert all(a is b for a, b in zip(items, received, strict=True))

    async def run() -> None:
        await asyncio.wait_for(
            asyncio.gather(*(consume() for _ in range(concurrent))), 10
        )

    asyncio.run(run())


@pytest.mark.parametrize("capacity", [1, 16])
def test_bounded_read_ahead_and_close_waits_for_cleanup(
    engine_loop: asyncio.AbstractEventLoop, capacity: int
) -> None:
    produced = []
    blocked = threading.Event()
    closed = threading.Event()

    async def source() -> AsyncGenerator[int, None]:
        try:
            for i in range(1000):
                produced.append(i)
                if len(produced) == capacity + 2:
                    blocked.set()
                yield i
        finally:
            assert asyncio.get_running_loop() is engine_loop
            await asyncio.sleep(0)
            closed.set()

    async def run() -> None:
        stream = queued_iterator(source(), engine_loop, capacity=capacity)
        assert await asyncio.wait_for(anext(stream), 5) == 0
        assert await asyncio.to_thread(blocked.wait, 5)
        # One consumed, capacity queued, one held by the blocked producer.
        assert len(produced) == capacity + 2
        await asyncio.wait_for(stream.aclose(), 5)
        assert closed.is_set()

    asyncio.run(run())


def test_engine_error_follows_buffered_outputs(
    engine_loop: asyncio.AbstractEventLoop,
) -> None:
    error = RuntimeError("engine failed")

    async def source() -> AsyncGenerator[int, None]:
        yield 1
        yield 2
        raise error

    async def run() -> None:
        stream = queued_iterator(source(), engine_loop, capacity=16)
        assert await anext(stream) == 1
        assert await anext(stream) == 2
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)
        assert caught.value is error

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("mode", ["per_output", "queued"])
@pytest.mark.parametrize("failure", ["close_full", "cancel_wait", "engine_error"])
def test_capture_abort_drains_engine_before_releasing_request(
    engine_loop: asyncio.AbstractEventLoop,
    mode: Literal["per_output", "queued"],
    failure: str,
) -> None:
    closed = threading.Event()
    started = threading.Event()
    aborted = []
    staged = []
    capture = DirectCapture(
        DirectCaptureConfig(logprob_capture="python"),
        lambda call: staged.append(call) or {"ordinal": call.ordinal},
        lambda: 0,
        clear=lambda keys: None,
    )
    capture.register("key", "rollout", temperature=1, top_p=1)

    class Engine:
        model_config = renderer = input_processor = vllm_config = None

        async def generate(
            self, *args: object, **kwargs: object
        ) -> AsyncGenerator[object, None]:
            try:
                for _ in range(100):
                    started.set()
                    if failure == "cancel_wait":
                        await asyncio.Event().wait()
                    if failure == "engine_error":
                        raise RuntimeError("engine failed")
                    yield NS(finished=False, outputs=[NS(routed_experts=None)])
            finally:
                assert asyncio.get_running_loop() is engine_loop
                await asyncio.sleep(0)
                closed.set()

        async def abort(self, request_id: str) -> None:
            assert asyncio.get_running_loop() is engine_loop
            aborted.append(request_id)

    bridge = _AsyncLLMHTTPClient(
        Engine(),
        engine_loop,
        bridge_config=HTTPBridgeConfig(mode=mode, max_pending_outputs=1),
    )

    async def run() -> None:
        context = capture.enter("key")
        token = current_capture.set(context)
        stream = bridge.generate(
            None,
            NS(num_logprobs=None, logprobs=None, output_kind=NS(name="DELTA")),
            "call",
        )
        try:
            if failure == "close_full":
                await anext(stream)
            elif failure == "cancel_wait":
                pending = asyncio.create_task(anext(stream))
                assert await asyncio.to_thread(started.wait, 5)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(RuntimeError, match="engine failed"):
                    await anext(stream)
        finally:
            await stream.aclose()
            current_capture.reset(token)
            capture.leave(context)
        assert closed.is_set()
        assert aborted == ["call"]
        receipt = await asyncio.to_thread(capture.finalize, "key", timeout=5)
        assert receipt["failure"] is not None
        assert not receipt["manifest"]
        assert not staged

    try:
        asyncio.run(asyncio.wait_for(run(), 10))
    finally:
        capture.shutdown()


def test_cumulative_outputs_do_not_read_ahead(
    engine_loop: asyncio.AbstractEventLoop,
) -> None:
    progress = []

    class Engine:
        model_config = renderer = input_processor = vllm_config = None

        async def generate(
            self, *args: object, **kwargs: object
        ) -> AsyncGenerator[object, None]:
            for index in range(3):
                progress.append(index)
                yield index

    bridge = _AsyncLLMHTTPClient(
        Engine(), engine_loop, bridge_config=HTTPBridgeConfig(mode="queued")
    )

    async def run() -> None:
        stream = bridge.generate(None, NS(output_kind=NS(name="CUMULATIVE")), "call")
        assert await anext(stream) == 0
        await asyncio.sleep(0.01)
        assert progress == [0]
        assert [i async for i in stream] == [1, 2]

    asyncio.run(run())


def test_bridge_config_validation() -> None:
    assert HTTPBridgeConfig().mode == "per_output"
    for options in (
        {"mode": "typo"},
        {"max_pending_outputs": 0},
        {"max_pending_outputs": -1},
    ):
        with pytest.raises(ValidationError):
            HTTPBridgeConfig.model_validate(options)


def test_first_output_does_not_wait_for_a_batch(
    engine_loop: asyncio.AbstractEventLoop,
) -> None:
    closed = threading.Event()

    async def source() -> AsyncGenerator[int, None]:
        try:
            yield 1
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def run() -> None:
        stream = queued_iterator(source(), engine_loop, capacity=16)
        assert await asyncio.wait_for(anext(stream), 5) == 1
        await stream.aclose()
        assert closed.is_set()

    asyncio.run(run())


def test_cancel_before_engine_registration_leaves_no_producer(
    engine_loop: asyncio.AbstractEventLoop,
) -> None:
    blocked = threading.Event()
    release = threading.Event()

    def hold_engine() -> None:
        blocked.set()
        assert release.wait(5)

    engine_loop.call_soon_threadsafe(hold_engine)
    assert blocked.wait(5)

    async def source() -> AsyncGenerator[int, None]:
        yield 1
        await asyncio.Event().wait()

    async def run() -> None:
        stream = queued_iterator(source(), engine_loop, capacity=1)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        # Let cancellation enqueue its cleanup before the engine can start.
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 5)
        await stream.aclose()

    try:
        asyncio.run(run())
    finally:
        release.set()
