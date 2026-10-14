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

"""Tests for local and Ray-style async data-plane dispatch."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from nemo_rl.data_plane.async_utils import (
    call_data_plane,
    complete_before_cancellation,
)


class _LocalClient:
    def thread_id(self) -> int:
        return threading.get_ident()

    async def async_value(self, *, value: int) -> int:
        return value


class _RemoteMethod:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def remote(self, *, value: int) -> int:
        self.calls.append(value)
        return value


class _RemoteClient:
    def __init__(self) -> None:
        self.value = _RemoteMethod()


def test_sync_call_stays_inline_by_default() -> None:
    caller_thread_id = threading.get_ident()

    result = asyncio.run(call_data_plane(_LocalClient(), "thread_id"))

    assert result == caller_thread_id


def test_sync_call_can_be_offloaded() -> None:
    caller_thread_id = threading.get_ident()

    result = asyncio.run(
        call_data_plane(_LocalClient(), "thread_id", offload_sync=True)
    )

    assert result != caller_thread_id


def test_local_coroutine_result_is_awaited() -> None:
    result = asyncio.run(call_data_plane(_LocalClient(), "async_value", value=7))

    assert result == 7


def test_ray_style_remote_result_is_awaited() -> None:
    client = _RemoteClient()

    result = asyncio.run(call_data_plane(client, "value", value=11))

    assert result == 11
    assert client.value.calls == [11]


@pytest.mark.parametrize(
    "method_name",
    ["get_data", "put_samples", "get_samples", "list_sample_ids", "clear_samples"],
)
@pytest.mark.parametrize("offload_sync", [False, True])
def test_enabled_async_io_uses_native_method_on_calling_loop(
    method_name: str, offload_sync: bool
) -> None:
    async def exercise() -> None:
        caller_thread_id = threading.get_ident()
        caller_loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = asyncio.Event()

        def sync_method(*, value: int) -> int:
            pytest.fail("enabled async I/O must not call the synchronous method")

        async def async_method(*, value: int) -> int:
            assert threading.get_ident() == caller_thread_id
            assert asyncio.get_running_loop() is caller_loop
            entered.set()
            await release.wait()
            return value

        client = SimpleNamespace(
            async_io_enabled=True,
            **{method_name: sync_method, f"async_{method_name}": async_method},
        )
        task = asyncio.create_task(
            call_data_plane(client, method_name, offload_sync=offload_sync, value=7)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not task.done()
        release.set()
        assert await task == 7

    asyncio.run(exercise())


@pytest.mark.parametrize("offload_sync", [False, True])
def test_disabled_async_io_preserves_sync_dispatch(offload_sync: bool) -> None:
    caller_thread_id = threading.get_ident()

    def sync_method() -> int:
        return threading.get_ident()

    async def async_method() -> int:
        pytest.fail("disabled async I/O must preserve the synchronous path")

    client = SimpleNamespace(
        async_io_enabled=False,
        get_samples=sync_method,
        async_get_samples=async_method,
    )
    result = asyncio.run(
        call_data_plane(client, "get_samples", offload_sync=offload_sync)
    )
    assert (result != caller_thread_id) == offload_sync


def test_enabled_async_io_keeps_checkpoint_dispatch() -> None:
    caller_thread_id = threading.get_ident()
    client = SimpleNamespace(
        async_io_enabled=True,
        save_checkpoint=threading.get_ident,
    )

    result = asyncio.run(call_data_plane(client, "save_checkpoint", offload_sync=True))

    assert result != caller_thread_id


def test_remote_io_does_not_inspect_local_async_configuration() -> None:
    class RemoteClient:
        get_samples = _RemoteMethod()

        @property
        def async_io_enabled(self) -> bool:
            pytest.fail("Ray dispatch must not inspect local client properties")

    result = asyncio.run(call_data_plane(RemoteClient(), "get_samples", value=11))

    assert result == 11


def test_enabled_async_io_does_not_retry_sync_after_failure() -> None:
    def sync_method() -> None:
        pytest.fail("async I/O failures must not trigger duplicate sync writes")

    async def async_method() -> None:
        raise RuntimeError("write outcome is unknown")

    client = SimpleNamespace(
        async_io_enabled=True,
        put_samples=sync_method,
        async_put_samples=async_method,
    )

    with pytest.raises(RuntimeError, match="write outcome is unknown"):
        asyncio.run(call_data_plane(client, "put_samples"))


def test_enabled_async_io_requires_the_async_method() -> None:
    client = SimpleNamespace(async_io_enabled=True, get_samples=lambda: None)

    with pytest.raises(AttributeError, match="async_get_samples"):
        asyncio.run(call_data_plane(client, "get_samples"))


def test_complete_before_cancellation_preserves_result_and_failure() -> None:
    async def exercise() -> None:
        async def value() -> int:
            return 13

        async def failure() -> None:
            raise ValueError("failed write")

        assert await complete_before_cancellation(value()) == 13
        with pytest.raises(ValueError, match="failed write"):
            await complete_before_cancellation(failure())

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_count", [1, 3])
def test_cancellation_keeps_ownership_until_late_write_finishes(
    cancel_count: int,
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        rows: list[str] = []
        actions: list[str] = []

        async def write() -> None:
            entered.set()
            await release.wait()
            rows.append("sample")
            actions.append("publish")

        async def owner() -> None:
            try:
                await complete_before_cancellation(write())
            finally:
                rows.clear()
                actions.append("cleanup")

        task = asyncio.create_task(owner())
        await asyncio.wait_for(entered.wait(), timeout=1)
        for index in range(cancel_count):
            task.cancel(f"cancel {index}")
            await asyncio.sleep(0)
            assert not task.done()
            assert not actions
        release.set()
        with pytest.raises(asyncio.CancelledError, match="cancel 0"):
            await task
        assert actions == ["publish", "cleanup"]
        assert rows == []

    asyncio.run(exercise())


def test_cancellation_consumes_late_failure_and_preserves_original_cancel() -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def write() -> None:
            entered.set()
            await release.wait()
            raise RuntimeError("publication failed after cancellation")

        operation = asyncio.create_task(write())
        owner = asyncio.create_task(complete_before_cancellation(operation))
        await asyncio.wait_for(entered.wait(), timeout=1)
        owner.cancel("shutdown")
        await asyncio.sleep(0)
        assert not operation.cancelled()
        assert not owner.done()
        release.set()
        with pytest.raises(asyncio.CancelledError, match="shutdown"):
            await owner
        assert operation.done()
        assert isinstance(operation.exception(), RuntimeError)

    asyncio.run(exercise())


def test_cancelled_operation_propagates_without_hanging() -> None:
    async def exercise() -> None:
        operation = asyncio.get_running_loop().create_future()
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await complete_before_cancellation(operation)

    asyncio.run(exercise())
