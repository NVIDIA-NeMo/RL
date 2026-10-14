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
"""Accounting and ownership guarantees of asynchronous observed operations."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import observability
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import MetricsDataPlaneClient
from nemo_rl.telemetry.span_groups import RLSpanGroup


def _fields(rows=1):
    return TensorDict({"x": torch.arange(rows * 4).reshape(rows, 4)}, [rows])


class _AsyncClient(NoOpDataPlaneClient):
    """Real in-memory storage, with synchronous data calls forbidden."""

    def __init__(self):
        super().__init__()
        self.register_partition("p", ["x"], 100, ["trainer"])
        self.read_columns = []

    @property
    def async_io_enabled(self):
        return True

    def _unexpected_sync(self, *args, **kwargs):
        raise AssertionError("async observability called a synchronous data method")

    put_samples = _unexpected_sync
    get_samples = _unexpected_sync
    get_data = _unexpected_sync
    list_sample_ids = _unexpected_sync
    clear_samples = _unexpected_sync

    async def async_put_samples(self, sample_ids, partition_id, fields=None, tags=None):
        await asyncio.sleep(0)
        return NoOpDataPlaneClient.put_samples(
            self, sample_ids, partition_id, fields, tags
        )

    async def async_get_samples(self, sample_ids, partition_id, select_fields):
        await asyncio.sleep(0)
        self.read_columns.append(list(select_fields))
        return NoOpDataPlaneClient.get_samples(
            self, sample_ids, partition_id, select_fields
        )

    async def async_get_data(self, meta, select_fields=None):
        return await self.async_get_samples(
            meta.sample_ids,
            meta.partition_id,
            select_fields if select_fields is not None else meta.fields,
        )

    async def async_list_sample_ids(self, partition_id):
        await asyncio.sleep(0)
        return NoOpDataPlaneClient.list_sample_ids(self, partition_id)

    async def async_clear_samples(self, sample_ids, partition_id):
        await asyncio.sleep(0)
        NoOpDataPlaneClient.clear_samples(self, sample_ids, partition_id)


def test_async_round_trip_preserves_hashes_events_and_live_bytes():
    async def scenario():
        inner = _AsyncClient()
        events = []
        client = MetricsDataPlaneClient(
            inner, on_event=events.append, verify_tensor_hash=True
        )
        assert client.async_io_enabled
        fields = _fields(2)
        meta = await client.async_put_samples(
            ["a", "b"], "p", fields, tags=[{"stage": 1}, {"stage": 2}]
        )
        assert meta.tags == [{"stage": 1}, {"stage": 2}]
        assert "x_hash" not in fields.keys()
        assert client.snapshot()["bytes_outstanding"] == 64
        out = await client.async_get_samples(["b", "a"], "p", ["x"])
        assert list(out.keys()) == ["x"]
        torch.testing.assert_close(out["x"], fields["x"].flip(0))
        out = await client.async_get_data(meta, ["x"])
        assert list(out.keys()) == ["x"]
        torch.testing.assert_close(out["x"], fields["x"])
        assert await client.async_list_sample_ids("p") == ["a", "b"]
        await client.async_clear_samples(["a", "missing"], "p")
        assert client.snapshot()["bytes_outstanding"] == 32
        await client.async_clear_samples(None, "p")
        snap = client.snapshot()
        assert snap["bytes_outstanding"] == 0
        assert snap["peak_bytes_outstanding"] == 64
        assert not client._partition_versions
        assert snap["hash_verify"]["rows_recorded"] == 2
        assert snap["hash_verify"]["rows_checked"] == 4
        assert snap["hash_verify"]["mismatches"] == 0
        assert [e["op"] for e in events] == [
            "put",
            "get",
            "get_data",
            "list_sample_ids",
            "clear",
            "clear",
        ]
        assert all(e["status"] == "ok" for e in events)
        assert events[0]["n_bytes"] == 64
        # Read-side accounting includes the hash mirrors transported with data.
        assert events[1]["n_bytes"] == 80
        assert snap["by_op"]["get"]["n_keys"] == 2
        client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [TimeoutError("deadline"), ValueError("bad put")])
def test_async_failed_put_has_status_but_no_live_bytes(failure):
    class FailingClient(_AsyncClient):
        async def async_put_samples(self, *args, **kwargs):
            await asyncio.sleep(0)
            raise failure

    async def scenario():
        events = []
        client = MetricsDataPlaneClient(FailingClient(), on_event=events.append)
        with pytest.raises(type(failure), match=str(failure)):
            await client.async_put_samples(["a"], "p", _fields())
        snap = client.snapshot()
        assert snap["bytes_outstanding"] == 0
        assert snap["total_ops"] == 0
        assert snap["by_op"]["put"]["errors"] == 1
        assert events[0]["status"] == (
            "timeout" if isinstance(failure, TimeoutError) else "error"
        )
        assert client._async_operations == 0
        client.close()

    asyncio.run(scenario())


def test_async_missing_hash_mirrors_retries_without_changing_read_result():
    async def scenario():
        inner = _AsyncClient()
        NoOpDataPlaneClient.put_samples(inner, ["a"], "p", _fields())
        events = []
        client = MetricsDataPlaneClient(
            inner, on_event=events.append, verify_tensor_hash=True
        )
        out = await client.async_get_samples(["a"], "p", ["x"])
        torch.testing.assert_close(out["x"], _fields()["x"])
        assert inner.read_columns == [["x", "x_hash"], ["x"]]
        assert [e["status"] for e in events] == ["error", "ok"]
        stats = client.snapshot()["hash_verify"]
        assert stats["guard_failures"] == 1
        assert stats["rows_unverified"] == 1
        client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("clear_before_rewrite", [False, True])
def test_async_reconcile_ignores_stale_snapshot_and_keeps_per_operation_timing(
    monkeypatch, clear_before_rewrite
):
    async def scenario():
        clock = [0.0]
        monkeypatch.setattr(observability, "monotonic", lambda: clock[0])
        listing_started = asyncio.Event()
        release_listing = asyncio.Event()

        class ListingClient(_AsyncClient):
            async def async_put_samples(self, *args, **kwargs):
                clock[0] += 1.0 if args[0] == ["a"] else 2.0
                return await super().async_put_samples(*args, **kwargs)

            async def async_list_sample_ids(self, partition_id):
                # The snapshot predates the second put; treating it as current
                # would wrongly remove that new key from live-byte accounting.
                listing_started.set()
                await release_listing.wait()
                return []

        client = MetricsDataPlaneClient(ListingClient())
        monkeypatch.setattr(observability, "_RECONCILE_ROWS", 1)
        first = asyncio.create_task(client.async_put_samples(["a"], "p", _fields()))
        await listing_started.wait()
        monkeypatch.setattr(observability, "_RECONCILE_ROWS", 100)
        if clear_before_rewrite:
            await client.async_clear_samples(["a"], "p")
        await client.async_put_samples(["b"], "p", _fields())
        clock[0] = 4.0
        release_listing.set()
        await first
        snap = client.snapshot()
        assert snap["bytes_outstanding"] == (32 if clear_before_rewrite else 64)
        assert client._keys_by_partition["p"] == (
            {"b"} if clear_before_rewrite else {"a", "b"}
        )
        # First call: 4s elapsed - its own 1s RPC. Second: 2s - 2s.
        assert snap["self_ms"] == pytest.approx(3000)
        assert snap["by_op"]["put"]["wall_ms"] == pytest.approx(3000)
        client.close()

    asyncio.run(scenario())


def test_async_reconciliation_releases_external_clears_without_sync_calls(monkeypatch):
    async def scenario():
        inner = _AsyncClient()
        client = MetricsDataPlaneClient(inner)
        await client.async_put_samples(["a"], "p", _fields())
        NoOpDataPlaneClient.clear_samples(inner, ["a"], "p")
        monkeypatch.setattr(observability, "_RECONCILE_ROWS", 2)
        await client.async_put_samples(["b"], "p", _fields())
        assert client.snapshot()["bytes_outstanding"] == 32
        assert client._keys_by_partition["p"] == {"b"}
        client.close()

    asyncio.run(scenario())


def test_async_reconciliation_failure_keeps_successful_put_and_retries_next_window(
    monkeypatch,
):
    class ListingClient(_AsyncClient):
        listing_calls = 0

        async def async_list_sample_ids(self, partition_id):
            self.listing_calls += 1
            if self.listing_calls == 1:
                raise TimeoutError("metadata unavailable")
            return await super().async_list_sample_ids(partition_id)

    async def scenario():
        inner = ListingClient()
        client = MetricsDataPlaneClient(inner)
        monkeypatch.setattr(observability, "_RECONCILE_ROWS", 2)
        await client.async_put_samples(["a", "b"], "p", _fields(2))
        assert client.snapshot()["bytes_outstanding"] == 64
        NoOpDataPlaneClient.clear_samples(inner, ["a", "b"], "p")
        await client.async_put_samples(["c"], "p", _fields())
        assert inner.listing_calls == 1
        await client.async_put_samples(["d"], "p", _fields())
        assert inner.listing_calls == 2
        assert client.snapshot()["bytes_outstanding"] == 64
        assert client._keys_by_partition["p"] == {"c", "d"}
        client.close()

    asyncio.run(scenario())


def test_async_failed_clear_does_not_release_live_bytes():
    class FailingClient(_AsyncClient):
        async def async_clear_samples(self, *args, **kwargs):
            await asyncio.sleep(0)
            raise TimeoutError("clear unavailable")

    async def scenario():
        client = MetricsDataPlaneClient(FailingClient())
        await client.async_put_samples(["a"], "p", _fields())
        with pytest.raises(TimeoutError, match="clear unavailable"):
            await client.async_clear_samples(None, "p")
        snap = client.snapshot()
        assert snap["bytes_outstanding"] == 32
        assert snap["by_op"]["clear"]["errors"] == 1
        assert await client.async_list_sample_ids("p") == ["a"]
        client.close()

    asyncio.run(scenario())


def test_async_clear_all_keeps_accounting_for_a_concurrent_surviving_put():
    async def scenario():
        listed = asyncio.Event()
        release_clear = asyncio.Event()

        class SnapshotClearClient(_AsyncClient):
            async def async_clear_samples(self, sample_ids, partition_id):
                assert sample_ids is None
                # Match TQ's clear(None): clear exactly the keys that its
                # preceding metadata listing returned, allowing writes between.
                keys = NoOpDataPlaneClient.list_sample_ids(self, partition_id)
                listed.set()
                await release_clear.wait()
                NoOpDataPlaneClient.clear_samples(self, keys, partition_id)

        client = MetricsDataPlaneClient(SnapshotClearClient())
        await client.async_put_samples(["old"], "p", _fields())
        clearing = asyncio.create_task(client.async_clear_samples(None, "p"))
        await listed.wait()
        await client.async_put_samples(["new"], "p", _fields())
        release_clear.set()
        await clearing
        assert await client.async_list_sample_ids("p") == ["new"]
        assert client.snapshot()["bytes_outstanding"] == 32
        assert client._keys_by_partition["p"] == {"new"}
        client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_phase", ["put", "reconcile"])
def test_cancelled_put_finishes_accounting_before_lifecycle_is_allowed(
    cancel_phase, tmp_path, monkeypatch
):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class GatedClient(_AsyncClient):
            async def async_put_samples(self, *args, **kwargs):
                if cancel_phase == "put":
                    started.set()
                    await release.wait()
                return await super().async_put_samples(*args, **kwargs)

            async def async_list_sample_ids(self, partition_id):
                if cancel_phase == "reconcile":
                    started.set()
                    await release.wait()
                return await super().async_list_sample_ids(partition_id)

        inner = GatedClient()
        events = []
        client = MetricsDataPlaneClient(inner, on_event=events.append)
        monkeypatch.setattr(observability, "_RECONCILE_ROWS", 1)
        task = asyncio.create_task(client.async_put_samples(["a"], "p", _fields()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        for lifecycle in (
            lambda: client.save_checkpoint(tmp_path / "blocked"),
            lambda: client.load_checkpoint(tmp_path / "blocked"),
            client.close,
        ):
            with pytest.raises(RuntimeError, match="async data-plane operations"):
                lifecycle()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert NoOpDataPlaneClient.list_sample_ids(inner, "p") == ["a"]
        assert client.snapshot()["bytes_outstanding"] == 32
        assert [e["status"] for e in events if e["op"] == "put"] == ["ok"]
        assert client._async_operations == 0
        client.close()

    asyncio.run(scenario())


def test_async_per_prompt_span_and_observability_disabled_byte_walk(monkeypatch):
    spans = []
    monkeypatch.setattr(observability, "in_per_prompt_scope", lambda: True)
    monkeypatch.setattr(observability, "is_span_group_enabled", lambda group: True)

    @contextmanager
    def span(group, name, **attrs):
        spans.append((group, name, attrs))
        yield None

    def unexpected_walk(fields):
        raise AssertionError("byte walk should be disabled")

    monkeypatch.setattr(observability, "umbrella_span", span)
    monkeypatch.setattr(observability, "_td_bytes", unexpected_walk)

    async def scenario():
        client = MetricsDataPlaneClient(_AsyncClient(), observability_enabled=False)
        await client.async_put_samples(["a"], "p", _fields())
        await client.async_get_samples(["a"], "p", ["x"])
        assert [s[0] for s in spans] == [RLSpanGroup.U_PER_PROMPT] * 2
        assert all(s[2][observability._OP_ATTR] in ("put", "get") for s in spans)
        assert client.snapshot()["bytes_outstanding"] == 0

    asyncio.run(scenario())
