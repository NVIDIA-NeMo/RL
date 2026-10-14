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

"""Async adapter contracts with real normalization and mocked TQ transport."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError
from tensordict import NonTensorStack, TensorDict

pytest.importorskip("transfer_queue")

from nemo_rl.data_plane.adapters import transfer_queue as adapter
from nemo_rl.data_plane.interfaces import KVBatchMeta


@pytest.fixture
def make_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., adapter.TQDataPlaneClient]:
    """Exercise the constructor without connecting to Ray or a storage server."""
    monkeypatch.setattr(adapter, "_connect_existing", lambda: None)

    def make(*, enabled: bool = True) -> adapter.TQDataPlaneClient:
        return adapter.TQDataPlaneClient(
            {
                "enabled": True,
                "impl": "transfer_queue",
                "backend": "simple",
                "claim_meta_poll_interval_s": 0.5,
                "async_io": {"enabled": enabled},
            },
            bootstrap=False,
        )

    return make


def test_native_put_matches_sync_metadata_and_preserves_non_tensor_leaves(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    calls: list[dict[str, Any]] = []

    def sync_put(**kwargs: Any) -> None:
        calls.append(kwargs)

    async def native_put(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(adapter.tq, "kv_batch_put", sync_put)
    monkeypatch.setattr(adapter.tq, "async_kv_batch_put", native_put)
    fields = TensorDict(
        {
            "tokens": torch.tensor([[2**40, 7], [8, 9]], dtype=torch.int64),
            "scores": torch.tensor([[-0.0, -0.1], [1.0, 2.0]], requires_grad=True),
            "labels": NonTensorStack({"text": "雪"}, {"text": "λ"}),
        },
        batch_size=[2],
    )
    ids = ["sample-b", "sample-a"]
    tags = [{"version": 3}, {"version": 4}]
    client = make_client()
    expected = client.put_samples(ids, "p", fields, tags)
    actual = asyncio.run(client.async_put_samples(ids, "p", fields, tags))

    assert actual == expected
    assert actual.sample_ids == ids
    assert actual.fields == ["tokens", "scores", "labels"]
    assert actual.tags == tags
    for call in calls:
        assert call["keys"] == ids
        assert call["partition_id"] == "p"
        assert call["tags"] == tags
        assert call["tags"][0] is not tags[0]
        wire = call["fields"]
        assert not wire["scores"].requires_grad
        assert torch.equal(
            wire["scores"].view(torch.uint8), fields["scores"].view(torch.uint8)
        )
        assert torch.equal(wire["tokens"], fields["tokens"])
        assert wire.get("labels").tolist() == [{"text": "雪"}, {"text": "λ"}]
    assert fields["scores"].requires_grad
    assert client._data_operations_started


def test_native_selective_get_matches_sync_densification_and_ragged_fields(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    calls: list[dict[str, Any]] = []
    wire = TensorDict(
        {
            "uniform": torch.nested.nested_tensor(
                [torch.tensor([1, 2]), torch.tensor([3, 4])]
            ),
            "ragged": torch.nested.nested_tensor(
                [torch.tensor([5]), torch.tensor([6, 7])]
            ),
            "labels": NonTensorStack("first", "second"),
        },
        batch_size=[2],
    )

    def sync_get(**kwargs: Any) -> TensorDict:
        calls.append(kwargs)
        return wire

    async def native_get(**kwargs: Any) -> TensorDict:
        return sync_get(**kwargs)

    monkeypatch.setattr(adapter.tq, "kv_batch_get", sync_get)
    monkeypatch.setattr(adapter.tq, "async_kv_batch_get", native_get)
    client = make_client()
    fields = ["uniform", "ragged", "labels"]
    expected = client.get_samples(["b", "a"], "p", fields)
    actual = asyncio.run(client.async_get_samples(["b", "a"], "p", fields))

    assert calls == [dict(keys=["b", "a"], partition_id="p", select_fields=fields)] * 2
    assert torch.equal(actual["uniform"], expected["uniform"])
    assert not actual["uniform"].is_nested
    assert actual["ragged"].is_nested
    for left, right in zip(actual["ragged"].unbind(), expected["ragged"].unbind()):
        assert torch.equal(left, right)
    assert actual.get("labels").tolist() == ["first", "second"]


def test_get_data_requires_explicit_fields_and_honors_override(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    selected: list[list[str]] = []

    async def native_get(**kwargs: Any) -> TensorDict:
        selected.append(kwargs["select_fields"])
        return TensorDict({}, batch_size=[1])

    monkeypatch.setattr(adapter.tq, "async_kv_batch_get", native_get)
    client = make_client()

    async def exercise() -> None:
        meta = KVBatchMeta("p", None, ["a"])
        with pytest.raises(ValueError, match="select_fields or meta.fields"):
            await client.async_get_data(meta)
        meta.fields = ["from_meta"]
        await client.async_get_data(meta)
        await client.async_get_data(meta, ["override"])

    asyncio.run(exercise())
    assert selected == [["from_meta"], ["override"]]


@pytest.mark.parametrize("backend,enabled", [("simple", False), ("mooncake_cpu", True)])
@pytest.mark.parametrize(
    "method", ["get_samples", "put_samples", "list_sample_ids", "clear_samples"]
)
def test_baseline_and_mooncake_use_sync_fallback_off_caller_thread(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
    backend: str,
    enabled: bool,
    method: str,
) -> None:
    client = make_client(enabled=enabled)
    # Only transport selection is under test; no Mooncake engine is initialized.
    client._backend = backend
    caller_thread = threading.get_ident()
    observed: list[int] = []
    result = object()

    def sync_operation(*args: Any) -> Any:
        observed.append(threading.get_ident())
        return result

    monkeypatch.setattr(client, method, sync_operation)

    async def exercise() -> None:
        if method == "get_samples":
            assert await client.async_get_samples(["a"], "p", ["x"]) is result
        elif method == "put_samples":
            assert await client.async_put_samples(["a"], "p") is result
        elif method == "list_sample_ids":
            assert await client.async_list_sample_ids("p") is result
        else:
            await client.async_clear_samples(["a"], "p")

    asyncio.run(exercise())
    assert len(observed) == 1
    assert observed[0] != caller_thread


def test_mixed_operations_add_no_concurrency_limit(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def exercise() -> None:
        client = make_client()
        entered: asyncio.Queue[str] = asyncio.Queue()
        release = asyncio.Event()
        active = 0
        peak = 0
        seen: list[str] = []

        async def wait(name: str) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            seen.append(name)
            entered.put_nowait(name)
            await release.wait()
            active -= 1

        async def put(**kwargs: Any) -> None:
            await wait("put")

        async def get(**kwargs: Any) -> TensorDict:
            await wait("get")
            return TensorDict({}, batch_size=[1])

        async def listing(**kwargs: Any) -> dict:
            await wait("list")
            return {"p": {"a": {}}}

        async def clear(**kwargs: Any) -> None:
            await wait("clear")

        monkeypatch.setattr(adapter.tq, "async_kv_batch_put", put)
        monkeypatch.setattr(adapter.tq, "async_kv_batch_get", get)
        monkeypatch.setattr(adapter.tq, "async_kv_list", listing)
        monkeypatch.setattr(adapter.tq, "async_kv_clear", clear)
        tasks = []
        for _ in range(4):
            tasks.extend(
                [
                    asyncio.create_task(client.async_put_samples(["a"], "p")),
                    asyncio.create_task(client.async_get_samples(["a"], "p", ["x"])),
                    asyncio.create_task(client.async_list_sample_ids("p")),
                    asyncio.create_task(client.async_clear_samples(["a"], "p")),
                ]
            )
        try:
            for _ in tasks:
                await asyncio.wait_for(entered.get(), timeout=1)
            assert len(seen) == active == 16
        finally:
            release.set()
            await asyncio.gather(*tasks)
        assert peak == 16
        assert sorted(seen) == sorted(["clear", "get", "list", "put"] * 4)

    asyncio.run(exercise())


def test_cancelled_submitted_write_settles_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def exercise() -> None:
        client = make_client()
        entered = asyncio.Event()
        release = asyncio.Event()
        rows: set[str] = set()
        actions: list[str] = []

        async def put(**kwargs: Any) -> None:
            entered.set()
            await release.wait()
            rows.update(kwargs["keys"])
            actions.append("put")

        async def clear(**kwargs: Any) -> None:
            rows.difference_update(kwargs["keys"])
            actions.append("clear")

        monkeypatch.setattr(adapter.tq, "async_kv_batch_put", put)
        monkeypatch.setattr(adapter.tq, "async_kv_clear", clear)

        async def owner() -> None:
            try:
                await client.async_put_samples(["a"], "p")
            finally:
                await client.async_clear_samples(["a"], "p")

        task = asyncio.create_task(owner())
        await asyncio.wait_for(entered.wait(), timeout=1)
        for message in ("first cancellation", "second cancellation"):
            task.cancel(message)
            await asyncio.sleep(0)
            assert not task.done()
            assert not actions
        release.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await task
        assert actions == ["put", "clear"]
        assert not rows

    asyncio.run(exercise())


@pytest.mark.parametrize("method", ["get_samples", "list_sample_ids"])
def test_reads_and_listings_cancel_without_waiting_for_transport(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
    method: str,
) -> None:
    async def exercise() -> None:
        client = make_client()
        entered = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()

        async def operation(**kwargs: Any) -> None:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(adapter.tq, "async_kv_batch_get", operation)
        monkeypatch.setattr(adapter.tq, "async_kv_list", operation)
        awaitable = (
            client.async_get_samples(["a"], "p", ["x"])
            if method == "get_samples"
            else client.async_list_sample_ids("p")
        )
        task = asyncio.create_task(awaitable)
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel("stop reading")
        with pytest.raises(asyncio.CancelledError, match="stop reading"):
            await asyncio.wait_for(task, timeout=1)
        assert cancelled.is_set()
        assert not release.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend,enabled", [("simple", False), ("mooncake_cpu", True)])
@pytest.mark.parametrize("method", ["get_samples", "list_sample_ids"])
def test_fallback_read_cancellation_does_not_wait_for_sync_thread(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
    backend: str,
    enabled: bool,
    method: str,
) -> None:
    async def exercise() -> None:
        client = make_client(enabled=enabled)
        client._backend = backend
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()

        def operation(*args: Any) -> None:
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(timeout=5)
            finally:
                finished.set()

        monkeypatch.setattr(client, method, operation)
        awaitable = (
            client.async_get_samples(["a"], "p", ["x"])
            if method == "get_samples"
            else client.async_list_sample_ids("p")
        )
        task = asyncio.create_task(awaitable)
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel("stop reading")
            done, _ = await asyncio.wait([task], timeout=1)
            assert task in done
            assert not finished.is_set()
            with pytest.raises(asyncio.CancelledError, match="stop reading"):
                await task
        finally:
            # Cancelling to_thread cannot stop its already-running thread.
            release.set()
            assert await asyncio.to_thread(finished.wait, 1)

    asyncio.run(exercise())


def test_submitted_clear_settles_before_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        rows = {"a"}

        async def clear(**kwargs: Any) -> None:
            entered.set()
            await release.wait()
            rows.difference_update(kwargs["keys"])

        monkeypatch.setattr(adapter.tq, "async_kv_clear", clear)
        task = asyncio.create_task(make_client().async_clear_samples(["a"], "p"))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel("stop clearing")
        await asyncio.sleep(0)
        assert not task.done()
        assert rows == {"a"}
        release.set()
        with pytest.raises(asyncio.CancelledError, match="stop clearing"):
            await task
        assert not rows

    asyncio.run(exercise())


def test_clear_all_can_cancel_during_listing_before_deletion(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def listing(**kwargs: Any) -> dict:
            entered.set()
            await release.wait()
            return {"p": {"a": {}}}

        clear = Mock(side_effect=AssertionError("cancelled listing must not clear"))
        monkeypatch.setattr(adapter.tq, "async_kv_list", listing)
        monkeypatch.setattr(adapter.tq, "async_kv_clear", clear)
        task = asyncio.create_task(make_client().async_clear_samples(None, "p"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel("stop listing")
            done, _ = await asyncio.wait([task], timeout=1)
            assert task in done
            with pytest.raises(asyncio.CancelledError, match="stop listing"):
                await task
            clear.assert_not_called()
        finally:
            release.set()

    asyncio.run(exercise())


def test_listing_clear_all_and_empty_operations_preserve_contracts(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    clear_calls: list[dict[str, Any]] = []
    listed = {"p": {"z": {}, "a": {}}}

    async def listing(**kwargs: Any) -> dict:
        assert kwargs == {"partition_id": "p"}
        return listed

    async def clear(**kwargs: Any) -> None:
        clear_calls.append(kwargs)

    monkeypatch.setattr(adapter.tq, "async_kv_list", listing)
    monkeypatch.setattr(adapter.tq, "async_kv_clear", clear)
    monkeypatch.setattr(
        adapter.tq, "async_kv_batch_put", Mock(side_effect=AssertionError("empty put"))
    )
    monkeypatch.setattr(
        adapter.tq, "async_kv_batch_get", Mock(side_effect=AssertionError("empty get"))
    )
    client = make_client()

    async def exercise() -> None:
        assert await client.async_list_sample_ids("p") == ["a", "z"]
        await client.async_clear_samples(None, "p")
        assert clear_calls == [{"keys": ["z", "a"], "partition_id": "p"}]
        await client.async_clear_samples([], "p")
        assert (await client.async_put_samples([], "p")).sample_ids == []
        assert (await client.async_get_samples([], "p", ["x"])).batch_size == (0,)
        listed.clear()
        with pytest.warns(RuntimeWarning, match="found nothing to clear"):
            await client.async_clear_samples(None, "p")
        assert len(clear_calls) == 1

    asyncio.run(exercise())


def test_failed_operation_does_not_retry_sync(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    attempts = 0

    async def native_put(**kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("unknown publication")

    monkeypatch.setattr(adapter.tq, "async_kv_batch_put", native_put)
    sync_put = Mock(side_effect=AssertionError("duplicate sync write"))
    monkeypatch.setattr(adapter.tq, "kv_batch_put", sync_put)
    client = make_client()

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="unknown publication"):
            await client.async_put_samples(["a"], "p")
        await client.async_put_samples(["b"], "p")

    asyncio.run(exercise())
    assert attempts == 2
    sync_put.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"max_inflight": 8},
        {"enabled": "maybe"},
        {"unknown_option": 1},
    ],
)
def test_invalid_async_config_fails_before_connect(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    connect = Mock()
    monkeypatch.setattr(adapter, "_connect_existing", connect)
    with pytest.raises(ValidationError):
        adapter.TQDataPlaneClient(
            {
                "enabled": True,
                "impl": "transfer_queue",
                "backend": "simple",
                "claim_meta_poll_interval_s": 0.5,
                "async_io": options,
            },
            bootstrap=False,
        )
    connect.assert_not_called()


def test_adapter_adds_no_live_event_loop_ownership_restriction(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def listing(**kwargs: Any) -> dict:
        return {"p": {"a": {}}}

    monkeypatch.setattr(adapter.tq, "async_kv_list", listing)
    client = make_client()
    first_loop = asyncio.new_event_loop()
    second_loop = asyncio.new_event_loop()
    try:
        assert first_loop.run_until_complete(client.async_list_sample_ids("p")) == ["a"]
        assert second_loop.run_until_complete(client.async_list_sample_ids("p")) == [
            "a"
        ]
    finally:
        first_loop.close()
        second_loop.close()
