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

    def make(
        *, enabled: bool = True, max_inflight: int = 2
    ) -> adapter.TQDataPlaneClient:
        return adapter.TQDataPlaneClient(
            {
                "enabled": True,
                "impl": "transfer_queue",
                "backend": "simple",
                "claim_meta_poll_interval_s": 0.5,
                "async_io": {"enabled": enabled, "max_inflight": max_inflight},
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


def test_mixed_operations_share_one_inflight_limit(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def exercise() -> None:
        client = make_client(max_inflight=2)
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
        tasks = [
            asyncio.create_task(client.async_put_samples(["a"], "p")),
            asyncio.create_task(client.async_get_samples(["a"], "p", ["x"])),
            asyncio.create_task(client.async_list_sample_ids("p")),
            asyncio.create_task(client.async_clear_samples(["a"], "p")),
        ]
        await asyncio.wait_for(entered.get(), timeout=1)
        await asyncio.wait_for(entered.get(), timeout=1)
        await asyncio.sleep(0)
        assert len(seen) == active == 2
        release.set()
        await asyncio.gather(*tasks)
        assert peak == 2
        assert sorted(seen) == ["clear", "get", "list", "put"]
        assert client._async_operations == 0

    asyncio.run(exercise())


def test_cancelled_submitted_write_settles_before_cleanup_and_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def exercise() -> None:
        client = make_client(max_inflight=1)
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
        close = Mock()
        monkeypatch.setattr(adapter.tq, "close", close)

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
        for operation in (
            client.close,
            lambda: client.save_checkpoint("unused"),
            lambda: client.load_checkpoint("unused"),
        ):
            with pytest.raises(RuntimeError, match="Drain asynchronous"):
                operation()
        close.assert_not_called()
        release.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await task
        assert actions == ["put", "clear"]
        assert not rows
        assert client._async_operations == 0
        client.close()
        close.assert_called_once()
        with pytest.raises(RuntimeError, match="closed"):
            await client.async_put_samples(["late"], "p")

    asyncio.run(exercise())


def test_cancelled_queued_write_is_not_submitted(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Callable[..., adapter.TQDataPlaneClient],
) -> None:
    async def exercise() -> None:
        client = make_client(max_inflight=1)
        entered = asyncio.Event()
        release = asyncio.Event()
        submitted: list[str] = []

        async def put(**kwargs: Any) -> None:
            submitted.extend(kwargs["keys"])
            entered.set()
            await release.wait()

        monkeypatch.setattr(adapter.tq, "async_kv_batch_put", put)
        first = asyncio.create_task(client.async_put_samples(["first"], "p"))
        await asyncio.wait_for(entered.wait(), timeout=1)
        queued = asyncio.create_task(client.async_put_samples(["queued"], "p"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        await first
        assert submitted == ["first"]
        assert client._async_operations == 0

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


def test_failed_operation_releases_capacity_and_does_not_retry_sync(
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
    client = make_client(max_inflight=1)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="unknown publication"):
            await client.async_put_samples(["a"], "p")
        assert client._async_operations == 0
        await client.async_put_samples(["b"], "p")

    asyncio.run(exercise())
    assert attempts == 2
    sync_put.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"max_inflight": 0},
        {"max_inflight": -1},
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


def test_client_rejects_another_live_loop_but_allows_closed_loop_replacement(
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
        with pytest.raises(RuntimeError, match="one event loop"):
            second_loop.run_until_complete(client.async_list_sample_ids("p"))
        first_loop.close()
        assert second_loop.run_until_complete(client.async_list_sample_ids("p")) == [
            "a"
        ]
    finally:
        first_loop.close()
        second_loop.close()
