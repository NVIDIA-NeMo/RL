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


def test_burst_deduplicates_keys_and_preserves_chain_order() -> None:
    async def run() -> None:
        calls = []

        def fetch(keys: list[str]) -> list[str]:
            calls.append(keys)
            return keys

        reader = PrefixReadBatcher(fetch, **LIMITS, wait_seconds=0.01)
        try:
            results = await asyncio.gather(
                *(reader.fetch([str(i), "shared"]) for i in range(64))
            )
            assert results == [[str(i), "shared"] for i in range(64)]
            assert len(calls) == 1
            assert len(calls[0]) == 65
        finally:
            await reader.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("rows,expected", [(2, 3), (5, 1)])
def test_row_limit_alone_splits_long_chains(rows: int, expected: int) -> None:
    async def run() -> None:
        calls = []

        def fetch(keys: list[str]) -> list[str]:
            calls.append(keys)
            return keys

        reader = PrefixReadBatcher(fetch, max_rows=rows, max_pending=10)
        try:
            assert await reader.fetch(list("abcde")) == list("abcde")
            assert len(calls) == expected
            assert max(map(len, calls)) <= rows
        finally:
            await reader.aclose()

    asyncio.run(run())


def test_missing_key_isolated_and_reader_remains_usable() -> None:
    async def run() -> None:
        def fetch(keys: list[str]) -> list[str]:
            if "missing" in keys:
                raise KeyError("missing")
            return keys

        reader = PrefixReadBatcher(fetch, **LIMITS)
        try:
            results = await asyncio.gather(
                reader.fetch(["good"]),
                reader.fetch(["missing"]),
                return_exceptions=True,
            )
            assert results[0] == ["good"]
            assert isinstance(results[1], KeyError)
            assert await reader.fetch(["next"]) == ["next"]
        finally:
            await reader.aclose()

    asyncio.run(run())


def test_bad_row_error_isolated_to_its_caller() -> None:
    async def run() -> None:
        calls: list[list[str]] = []

        def fetch(keys: list[str]) -> list[str]:
            calls.append(list(keys))
            if "bad" in keys:
                raise ValueError("corrupt row")
            return keys

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
            assert sorted(map(tuple, calls)) == [("bad",), ("good",), ("good", "bad")]
        finally:
            await reader.aclose()

    asyncio.run(run())


def test_failed_batch_retries_each_key_once() -> None:
    async def run() -> None:
        calls: list[list[str]] = []

        def fetch(keys: list[str]) -> list[str]:
            calls.append(list(keys))
            raise KeyError("store unavailable")

        reader = PrefixReadBatcher(fetch, **LIMITS)
        keys = [f"k{index}" for index in range(32)]
        try:
            with pytest.raises(KeyError):
                await reader.fetch(keys)
            assert len(calls) == len(keys) + 1
        finally:
            await reader.aclose()

    asyncio.run(run())


def test_cancel_does_not_cancel_shared_read() -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()

        def fetch(keys: list[str]) -> list[str]:
            started.set()
            assert release.wait(5)
            return keys

        reader = PrefixReadBatcher(fetch, **LIMITS)
        first = asyncio.create_task(reader.fetch(["same"]))
        second = asyncio.create_task(reader.fetch(["same"]))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            assert await second == ["same"]
        finally:
            release.set()
            await reader.aclose()

    asyncio.run(run())


def test_shutdown_wakes_inflight_and_backpressured_requests() -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()

        def fetch(keys: list[str]) -> list[str]:
            started.set()
            assert release.wait(5)
            return keys

        reader = PrefixReadBatcher(fetch, **LIMITS, max_pending=1)
        first = asyncio.create_task(reader.fetch(["one"]))
        second = asyncio.create_task(reader.fetch(["two"]))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await reader.aclose()
            results = await asyncio.wait_for(
                asyncio.gather(first, second, return_exceptions=True), 1
            )
            assert all(isinstance(result, RuntimeError) for result in results)
        finally:
            release.set()
            await reader.aclose()

    asyncio.run(run())


def test_single_request_flushes_without_more_arrivals() -> None:
    async def run() -> None:
        reader = PrefixReadBatcher(lambda keys: keys, **LIMITS)
        try:
            assert await asyncio.wait_for(reader.fetch(["one"]), 1) == ["one"]
        finally:
            await reader.aclose()

    asyncio.run(run())


def test_chain_larger_than_queue_does_not_deadlock() -> None:
    async def run() -> None:
        reader = PrefixReadBatcher(lambda keys: keys, **LIMITS, max_pending=2)
        try:
            assert await asyncio.wait_for(reader.fetch(list("abcdef")), 1) == list(
                "abcdef"
            )
        finally:
            await reader.aclose()

    asyncio.run(run())


def test_shutdown_before_runner_starts() -> None:
    async def run() -> None:
        reader = PrefixReadBatcher(lambda keys: keys, **LIMITS)
        request = asyncio.create_task(reader.fetch(["one"]))
        await asyncio.sleep(0)
        await reader.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.wait_for(request, 1)
        with pytest.raises(RuntimeError, match="closed"):
            await reader.fetch(["two"])

    asyncio.run(run())


def test_shutdown_from_actor_loop_closes_serving_loop_reader() -> None:
    started = threading.Event()
    release = threading.Event()
    errors = []

    def fetch(keys: list[str]) -> list[str]:
        started.set()
        assert release.wait(5)
        return keys

    reader = PrefixReadBatcher(fetch, **LIMITS)

    async def serve() -> None:
        try:
            await reader.fetch(["one"])
        except RuntimeError as error:
            errors.append(error)

    server = threading.Thread(target=lambda: asyncio.run(serve()))
    server.start()
    try:
        assert started.wait(5)
        asyncio.run(reader.aclose())
    finally:
        release.set()
        server.join(5)
    assert not server.is_alive()
    assert len(errors) == 1
