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
"""Durability, overlap, encoding parity and recovery tests for prefix writes."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import Any

import pytest
import torch

from nemo_rl.data_plane.prefix_pipeline import (
    CutSteps,
    BatchReleased,
    PrefixWrite,
    PrefixWritePipeline,
    PrefixWritePipelineConfig,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from nemo_rl.models.generation.prefix_recovery import PrefixRecovery
from nemo_rl.utils.routed_experts_codec import routed_experts_tensor_metadata
from tests.unit.models.generation.test_vllm_token_capture_hosting import (
    _assert_rolled_back,
    _batch_worker_fixture,
    _next_inventory,
)


def _pipeline(recovery: PrefixRecovery, **kwargs: Any) -> PrefixRecovery:
    config = PrefixWritePipelineConfig(
        enabled=True,
        direct_column_encoding=True,
        **kwargs,
    )
    return PrefixRecovery(recovery._parts, pipeline_config=config)


@pytest.mark.parametrize("routed", [False, True])
def test_direct_columns_match_general_sink_and_preserve_owned_buffers(
    routed: bool,
) -> None:
    recovery, memory, _, inventory = _batch_worker_fixture([2, 4, 3])
    recovery.checkpoint_cut(inventory)
    records = [record for _, record in memory.generation_prefix_records]
    recovery.shutdown()
    attached = [None] * len(records)
    if routed:
        attached = []
        routed_records = []
        for record in records:
            tensor = torch.arange(record.delta_len * 4, dtype=torch.int16).reshape(
                record.delta_len, 2, 2
            )
            routed_records.append(
                record.model_copy(
                    update={
                        "extras": {
                            "routed_experts": routed_experts_tensor_metadata(tensor)
                        }
                    }
                )
            )
            attached.append({"routed_experts": tensor})
        records = routed_records
    clients = [
        SimpleNamespace(put_samples=MagicMock(spec=lambda **kwargs: None))
        for _ in range(2)
    ]
    sinks = [TQTokenSink(client, staging_partition="staging") for client in clients]
    sinks[1].enable_direct_prefix_columns()
    for sink in sinks:
        sink.stage_generation_prefix_batch(
            records,
            checkpoint_id="cut",
            chunk_sequences=[0] * len(records),
            attachments=attached,
        )
    old, new = [client.put_samples.call_args.kwargs for client in clients]
    assert old["sample_ids"] == new["sample_ids"]
    assert old["tags"] == new["tags"]
    assert set(old["fields"].keys()) == set(new["fields"].keys())
    for name in old["fields"].keys():
        for index in range(len(records)):
            torch.testing.assert_close(
                old["fields"][name][index], new["fields"][name][index]
            )
    if routed:
        saved = new["fields"]["routed_experts"][0].clone()
        attached[0]["routed_experts"].zero_()
        torch.testing.assert_close(new["fields"]["routed_experts"][0], saved)


def _inventory(identity: str, calls: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_id="cut",
        inventory_digest=identity,
        active_prefixes=[SimpleNamespace(model_call_id=call) for call in calls],
    )


def test_preparation_overlaps_put_but_receipts_and_locks_wait_for_durability() -> None:
    recovery, sink, states, inventory = _batch_worker_fixture([2, 3, 4], batch_size=1)
    entered, release = threading.Event(), threading.Event()
    sink.before_write = lambda: (entered.set(), release.wait(5))
    pipeline = _pipeline(recovery, max_batch_rows=1)
    try:
        # Separate inventories expose preparation overlap across independent calls.
        from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

        inventories = [
            GenerationCutInventory.build(
                checkpoint_id=inventory.checkpoint_id,
                server_name=inventory.server_name,
                active_prefixes=[prefix],
            )
            for prefix in inventory.active_prefixes
        ]
        futures = [
            pipeline._pipeline.submit(item, deadline=None) for item in inventories
        ]
        assert entered.wait(5)
        deadline = time.monotonic() + 5
        while not all(state.snapshot().cut_in_progress for state in states):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert not any(future.done() for future in futures)
        for state in states:
            with state.try_lifecycle() as owned:
                assert owned is None
        release.set()
        for item, future in zip(inventories, futures, strict=True):
            future.result(5).validate_for(item)
        assert all(state.snapshot().staging_keys for state in states)
    finally:
        release.set()
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_cross_inventory_merge_duplicates_and_overlap_barrier() -> None:
    started, release = threading.Event(), threading.Event()
    events = []
    sink = MagicMock()

    def write(records: list[Any], **kwargs: Any) -> list[str]:
        started.set()
        assert release.wait(5)
        return [record.key for record in records]

    sink.stage_generation_prefix_batch.side_effect = write

    def prepare(inventory: Any, *, deadline: float | None) -> CutSteps:
        events.append(("prepare", inventory.inventory_digest, threading.get_ident()))
        record = SimpleNamespace(key=inventory.inventory_digest, delta_len=1)
        keys = yield PrefixWrite(
            "cut", (record,), (0,), (None,), 100, lambda: [record.key]
        )
        events.append(("seal", inventory.inventory_digest, threading.get_ident()))
        yield BatchReleased()
        return keys

    pipeline = PrefixWritePipeline(
        config=PrefixWritePipelineConfig(enabled=True, coalesce_wait_seconds=0.05),
        sink=sink,
        prepare=prepare,
    )
    try:
        first = pipeline.submit(_inventory("one", ["a"]), deadline=None)
        assert first is pipeline.submit(_inventory("one", ["a"]), deadline=None)
        second = pipeline.submit(_inventory("two", ["b"]), deadline=None)
        overlapping = pipeline.submit(_inventory("three", ["a"]), deadline=None)
        assert started.wait(5)
        assert not first.done() and not second.done()
        assert not any(event[1] == "three" for event in events)
        release.set()
        assert first.result(5) == ["one"]
        assert second.result(5) == ["two"]
        assert overlapping.result(5) == ["three"]
        assert [event[:2] for event in events].index(("seal", "one")) < [
            event[:2] for event in events
        ].index(("prepare", "three"))
        assert len({event[2] for event in events}) == 1
        assert pipeline.stats()["max_rows_per_write"] == 2
    finally:
        release.set()
        pipeline.close(wait=True)


@pytest.mark.parametrize("failure", ["batch", "all", "deadline", "oversize"])
def test_native_pipeline_failure_rollback_and_deadlines(failure: str) -> None:
    recovery, sink, states, inventory = _batch_worker_fixture([2, 3])
    sink.fail_multi_row = failure == "batch"
    sink.fail_after_write = failure == "all"
    pipeline = _pipeline(
        recovery,
        **(
            {"max_batch_payload_bytes": 1, "max_buffered_payload_bytes": 2}
            if failure == "oversize"
            else {}
        ),
    )
    try:
        receipt = pipeline._pipeline.submit(
            inventory,
            deadline=time.monotonic() - 1 if failure == "deadline" else None,
        ).result(5)
        receipt.validate_for(inventory)
        if failure == "batch":
            assert all(ack.disposition == "durable_prefix" for ack in receipt.prefixes)
            assert len(sink.batches) > 1
        else:
            assert all(ack.disposition == "durable_failure" for ack in receipt.prefixes)
            for state, count in zip(states, [2, 3], strict=True):
                _assert_rolled_back(state, count)
        assert pipeline._pipeline.stats()["buffered_payload_bytes"] == 0
    finally:
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_native_pipeline_repeated_cuts_keep_the_saved_chunk_chain() -> None:
    recovery, sink, states, inventory = _batch_worker_fixture([2, 3])
    pipeline = _pipeline(recovery)
    try:
        first = pipeline._pipeline.submit(inventory, deadline=None).result(5)
        for state in states:
            state.record([31, 32], [-0.2, -0.3])
        second_inventory = _next_inventory(inventory, "cut-2")
        second = pipeline._pipeline.submit(second_inventory, deadline=None).result(5)
        second.validate_for(second_inventory)
        assert all(len(ack.staging_keys) == 2 for ack in second.prefixes)
        assert [ack.prefix_token_count for ack in second.prefixes] == [4, 5]
        assert [ack.prefix_token_count for ack in first.prefixes] == [2, 3]
    finally:
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_shutdown_keeps_active_write_ownership_until_it_finishes() -> None:
    recovery, sink, states, inventory = _batch_worker_fixture([2])
    entered, release = threading.Event(), threading.Event()
    sink.before_write = lambda: (entered.set(), release.wait(5))
    pipeline = _pipeline(recovery)
    try:
        future = pipeline._pipeline.submit(inventory, deadline=None)
        assert entered.wait(5)
        pipeline._pipeline.close(wait=False)
        with pytest.raises(RuntimeError, match="closed"):
            pipeline._pipeline.submit(inventory, deadline=None)
        assert not future.done()
        with states[0].try_lifecycle() as owned:
            assert owned is None
        release.set()
        with pytest.raises(RuntimeError, match="closed"):
            future.result(5)
        assert not states[0].snapshot().cut_in_progress
        with states[0].try_lifecycle() as owned:
            assert owned is not None
    finally:
        release.set()
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_cancelled_http_waiter_does_not_cancel_the_durable_write_or_duplicate_it() -> (
    None
):
    recovery, sink, _, inventory = _batch_worker_fixture([2])
    entered, release = threading.Event(), threading.Event()
    sink.before_write = lambda: (entered.set(), release.wait(5))
    pipeline = _pipeline(recovery)

    async def scenario() -> Any:
        waiter = asyncio.create_task(pipeline.run_cut(inventory))
        assert await asyncio.to_thread(entered.wait, 5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        retry = asyncio.create_task(pipeline.run_cut(inventory))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.wait_for(retry, 5)

    try:
        asyncio.run(scenario()).validate_for(inventory)
        assert len(sink.batches) == 1
    finally:
        release.set()
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()
