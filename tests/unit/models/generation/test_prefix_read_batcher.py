# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only concurrency tests for cross-rollout prefix reads."""

import asyncio
import threading

import pytest

from nemo_rl.models.generation.prefix_read_batcher import PrefixReadBatcher

# The row limit is required: production passes rollout_recovery's batch size.
LIMITS = {"max_rows": 256}

pytestmark = pytest.mark.nemo_gym


class _Fetch:
    """Echo fetch that records each GET; ``block`` holds it until released."""

    def __init__(self, *, block: bool = False, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.started, self.release = threading.Event(), threading.Event()
        if not block:
            self.release.set()
        self.fail_on = fail_on

    def __call__(self, keys: list[str]) -> list[str]:
        self.calls.append(list(keys))
        self.started.set()
        assert self.release.wait(5)
        if self.fail_on in keys:
            raise ValueError("corrupt row")
        return keys


@pytest.mark.asyncio
async def test_burst_deduplicates_keys_and_preserves_chain_order() -> None:
    fetch = _Fetch()
    reader = PrefixReadBatcher(fetch, **LIMITS, wait_seconds=0.01)
    try:
        results = await asyncio.gather(
            *(reader.fetch([str(i), "shared"]) for i in range(64))
        )
        assert results == [[str(i), "shared"] for i in range(64)]
        assert len(fetch.calls) == 1
        assert len(fetch.calls[0]) == 65
    finally:
        await reader.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("rows,expected", [(2, 3), (5, 1)])
async def test_row_limit_alone_splits_long_chains(rows: int, expected: int) -> None:
    fetch = _Fetch()
    reader = PrefixReadBatcher(fetch, max_rows=rows)
    try:
        # A lone request must flush without waiting for more arrivals.
        assert await asyncio.wait_for(reader.fetch(list("abcde")), 1) == list("abcde")
        assert len(fetch.calls) == expected
        assert max(map(len, fetch.calls)) <= rows
    finally:
        await reader.aclose()


@pytest.mark.asyncio
async def test_bad_row_error_isolated_to_its_caller() -> None:
    fetch = _Fetch(fail_on="bad")
    reader = PrefixReadBatcher(fetch, **LIMITS)
    try:
        results = await asyncio.gather(
            reader.fetch(["good"]),
            reader.fetch(["bad"]),
            return_exceptions=True,
        )
        assert results[0] == ["good"]
        assert isinstance(results[1], ValueError)
        # One shared GET, then each key alone once.
        assert sorted(map(tuple, fetch.calls)) == [
            ("bad",),
            ("good",),
            ("good", "bad"),
        ]
        assert await reader.fetch(["next"]) == ["next"]
    finally:
        await reader.aclose()


@pytest.mark.asyncio
async def test_cancel_does_not_cancel_shared_read() -> None:
    fetch = _Fetch(block=True)
    reader = PrefixReadBatcher(fetch, **LIMITS)
    first = asyncio.create_task(reader.fetch(["same"]))
    second = asyncio.create_task(reader.fetch(["same"]))
    try:
        assert await asyncio.to_thread(fetch.started.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        fetch.release.set()
        assert await second == ["same"]
    finally:
        fetch.release.set()
        await reader.aclose()


@pytest.mark.asyncio
async def test_shutdown_wakes_inflight_request() -> None:
    fetch = _Fetch(block=True)
    reader = PrefixReadBatcher(fetch, **LIMITS)
    request = asyncio.create_task(reader.fetch(["one"]))
    try:
        assert await asyncio.to_thread(fetch.started.wait, 5)
        await reader.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.wait_for(request, 1)
    finally:
        fetch.release.set()
        await reader.aclose()


@pytest.mark.asyncio
async def test_shutdown_before_runner_starts() -> None:
    reader = PrefixReadBatcher(_Fetch(), **LIMITS)
    request = asyncio.create_task(reader.fetch(["one"]))
    await asyncio.sleep(0)
    await reader.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(request, 1)
    with pytest.raises(RuntimeError, match="closed"):
        await reader.fetch(["two"])


def test_shutdown_from_actor_loop_closes_serving_loop_reader() -> None:
    fetch = _Fetch(block=True)
    errors = []
    reader = PrefixReadBatcher(fetch, **LIMITS)

    async def serve() -> None:
        try:
            await reader.fetch(["one"])
        except RuntimeError as error:
            errors.append(error)

    server = threading.Thread(target=lambda: asyncio.run(serve()))
    server.start()
    try:
        assert fetch.started.wait(5)
        asyncio.run(reader.aclose())
    finally:
        fetch.release.set()
        server.join(5)
    assert not server.is_alive()
    assert len(errors) == 1


def test_shutdown_returns_when_serving_loop_stopped_but_open() -> None:
    reader = PrefixReadBatcher(_Fetch(), **LIMITS)
    # Like the reserved-socket server thread: the loop stops but is never closed.
    serving = asyncio.new_event_loop()
    try:
        assert serving.run_until_complete(reader.fetch(["one"])) == ["one"]
        asyncio.run(asyncio.wait_for(reader.aclose(), 1))
    finally:
        tasks = asyncio.all_tasks(serving)
        for task in tasks:
            task.cancel()
        serving.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        serving.close()
