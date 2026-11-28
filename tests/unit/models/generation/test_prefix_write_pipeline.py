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
import logging
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data.captured_media import MEDIA_TENSOR_DIGESTS_FIELD, CapturedMediaItem
from nemo_rl.data_plane.prefix_writer_actor import (
    PrefixWriterService,
    RayPrefixWriterClient,
)
from nemo_rl.data_plane.tq_token_sink import MEDIA_IMGS_FIELD, TQTokenSink
from nemo_rl.models.generation.prefix_recovery import PrefixRecovery
from nemo_rl.models.generation.prefix_recovery import recovery as recovery_module
from nemo_rl.models.generation.prefix_recovery import (
    write_pipeline as write_pipeline_module,
)
from nemo_rl.models.generation.prefix_recovery.write_pipeline import (
    PREFIX_WRITE_LIMITS,
    CutWrites,
    PrefixWrite,
    PrefixWriteLimits,
    PrefixWritePipeline,
    WriteSettled,
)
from tests.unit.models.generation.test_vllm_token_capture_hosting import (
    _assert_rolled_back,
    _batch_worker_fixture,
    _next_inventory,
)

pytestmark = pytest.mark.nemo_gym


def _pipeline(recovery: PrefixRecovery, **kwargs: Any) -> PrefixRecovery:
    limits = replace(PREFIX_WRITE_LIMITS, **kwargs)
    return PrefixRecovery(replace(recovery._parts, write_limits=limits))


def test_prefix_write_buffer_must_hold_two_batches() -> None:
    with pytest.raises(ValueError, match=r">= 2 \* max_batch_payload_bytes"):
        PrefixWriteLimits(
            max_pending_inventories=1,
            max_buffered_payload_bytes=3,
            max_batch_payload_bytes=2,
            coalesce_wait_seconds=0.0,
        )


def _media_cut_rows(
    monkeypatch: pytest.MonkeyPatch, writer_actor: bool
) -> dict[str, tuple[dict[str, Any], dict[str, torch.Tensor]]]:
    """Cut two media calls on a media partition; return every written row by key."""
    from nemo_gym.token_id_capture.adapters.vllm import MEDIA_SPANS_FIELD

    item = CapturedMediaItem(
        modality="image",
        placeholder_offset=0,
        placeholder_length=2,
        placeholder_digest="a" * 64,
        token_id=7,
        embedding_spans=((0, 2),),
        imgs_sizes=((2, 2),),
    )
    extras = {
        MEDIA_SPANS_FIELD: [item.to_dict()],
        MEDIA_TENSOR_DIGESTS_FIELD: {
            "imgs": {"dtype": "bfloat16", "shape": [4, 8], "sha256": "0" * 64}
        },
    }
    monkeypatch.setattr("ray.get", lambda reference: reference)
    recovery, _, _, inventory = _batch_worker_fixture([2, 3])

    def client() -> SimpleNamespace:
        return SimpleNamespace(
            put_samples=MagicMock(spec=lambda **kwargs: None),
            clear_samples=MagicMock(spec=lambda **kwargs: None),
        )

    local = client()
    sink = TQTokenSink(
        local,
        staging_partition="staging",
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
    written = local
    if writer_actor:
        written = client()
        service = PrefixWriterService(written)
        actor = SimpleNamespace(
            put_samples=SimpleNamespace(remote=service.put_samples),
            clear_samples=SimpleNamespace(remote=service.clear_samples),
        )
        sink.set_prefix_writer(RayPrefixWriterClient(actor, writer_pid=1))
    recovery = PrefixRecovery(
        replace(
            recovery._parts,
            sink=sink,
            prefix_row_extras=lambda state: extras,
        )
    )
    try:
        receipt = asyncio.run(recovery.run_cut(inventory))
        receipt.validate_for(inventory)
        assert all(ack.disposition == "durable_prefix" for ack in receipt.prefixes)
    finally:
        asyncio.run(recovery.aclose())
        recovery.shutdown()
    if written is not local:
        local.put_samples.assert_not_called()
    rows = {}
    for call in written.put_samples.call_args_list:
        kwargs = call.kwargs
        for index, key in enumerate(kwargs["sample_ids"]):
            rows[key] = (
                kwargs["tags"][index],
                {
                    name: kwargs["fields"][name][index]
                    for name in kwargs["fields"].keys()
                },
            )
    return rows


def test_media_cut_rows_match_with_the_writer_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media capture writes the same cut rows through the writer actor as without."""
    expected = _media_cut_rows(monkeypatch, writer_actor=False)
    actual = _media_cut_rows(monkeypatch, writer_actor=True)
    assert actual.keys() == expected.keys() and len(expected) == 2
    for key, (tags, fields) in expected.items():
        assert actual[key][0] == tags
        assert actual[key][1].keys() == fields.keys()
        assert MEDIA_IMGS_FIELD in fields
        for name, value in fields.items():
            torch.testing.assert_close(actual[key][1][name], value, rtol=0, atol=0)


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
    pipeline = _pipeline(recovery)
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
        while sum(state.snapshot().cut_in_progress for state in states) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        # The write in flight plus one prepared ahead; the third call waits
        # unfrozen until the first write is durable.
        frozen = [state for state in states if state.snapshot().cut_in_progress]
        assert len(frozen) == 2
        assert not any(future.done() for future in futures)
        for state in frozen:
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

    def prepare(
        inventory: Any, *, deadline: float | None, held_only: bool
    ) -> CutWrites:
        events.append(("prepare", inventory.inventory_digest, threading.get_ident()))
        record = SimpleNamespace(key=inventory.inventory_digest, delta_len=1)
        keys = yield PrefixWrite(
            "cut", (record,), (0,), (None,), 100, lambda error: [record.key]
        )
        events.append(("seal", inventory.inventory_digest, threading.get_ident()))
        yield WriteSettled()
        return keys

    pipeline = PrefixWritePipeline(
        limits=replace(PREFIX_WRITE_LIMITS, coalesce_wait_seconds=0.05),
        max_batch_rows=256,
        max_batch_tokens=4_194_304,
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
        # "one" and "two" share the first write; "three" waits for its own.
        counts = pipeline.counts()
        assert counts["write_batches"] == 2
        assert counts["merged_inventories"] == 3
        assert counts["failed_inventories"] == 0
        assert counts["peak_buffered_payload_bytes"] > 0
    finally:
        release.set()
        pipeline.close(wait=True)


@pytest.mark.parametrize("failure", ["batch", "all", "deadline"])
def test_native_pipeline_failure_rollback_and_deadlines(failure: str) -> None:
    recovery, sink, states, inventory = _batch_worker_fixture([2, 3])
    sink.fail_multi_row = failure == "batch"
    sink.fail_after_write = failure == "all"
    pipeline = _pipeline(recovery)
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
        assert pipeline._pipeline._buffered == 0
        assert pipeline.counts()["pipeline_failed_inventories"] == 0
    finally:
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_oversize_rows_are_written_alone_even_past_the_buffer() -> None:
    """A row above the batch limit is written alone, not refused."""
    recovery, sink, states, inventory = _batch_worker_fixture([2, 3])
    # Each row's estimate is far above both limits.
    pipeline = _pipeline(
        recovery, max_batch_payload_bytes=1, max_buffered_payload_bytes=2
    )
    try:
        receipt = pipeline._pipeline.submit(inventory, deadline=None).result(5)
        receipt.validate_for(inventory)
        assert all(ack.disposition == "durable_prefix" for ack in receipt.prefixes)
        assert sink.batches == [[3], [4]]  # First chunks include the prompt.
        assert not any(state.snapshot().cut_in_progress for state in states)
        assert pipeline._pipeline._buffered == 0
        assert pipeline.counts()["pipeline_peak_buffered_payload_bytes"] > 2
    finally:
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_oversize_row_between_small_rows_is_written_alone() -> None:
    recovery, sink, states, inventory = _batch_worker_fixture([2, 100, 3])
    # Small rows estimate about 600 bytes, the middle one about 2,100.
    pipeline = _pipeline(
        recovery, max_batch_payload_bytes=1300, max_buffered_payload_bytes=2600
    )
    try:
        receipt = pipeline._pipeline.submit(inventory, deadline=None).result(5)
        receipt.validate_for(inventory)
        assert all(ack.disposition == "durable_prefix" for ack in receipt.prefixes)
        assert sink.batches == [[3], [101], [4]]
        assert not any(state.snapshot().cut_in_progress for state in states)
    finally:
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_an_oversize_reservation_blocks_growth_but_not_release() -> None:
    pipeline = PrefixWritePipeline(
        limits=replace(
            PREFIX_WRITE_LIMITS, max_batch_payload_bytes=1, max_buffered_payload_bytes=2
        ),
        max_batch_rows=256,
        max_batch_tokens=4_194_304,
        sink=MagicMock(),
        prepare=MagicMock(),
    )
    try:
        small, big = SimpleNamespace(reservation=0), SimpleNamespace(reservation=0)
        pipeline._reserve(small, 1)
        pipeline._reserve(big, 10, allow_overflow=True)
        pipeline._reserve(small, 0)
        assert pipeline._buffered == 10
        with pytest.raises(RuntimeError, match="exceeded its byte reservation"):
            pipeline._reserve(small, 1)
    finally:
        pipeline.close(wait=True)


def test_only_a_single_row_may_exceed_the_batch_byte_limit() -> None:
    sink = MagicMock()

    def prepare(
        inventory: Any, *, deadline: float | None, held_only: bool
    ) -> CutWrites:
        records = tuple(
            SimpleNamespace(key=f"{inventory.inventory_digest}-{index}", delta_len=1)
            for index in range(2)
        )
        yield PrefixWrite("cut", records, (0, 0), (None, None), 2, lambda error: [None])

    pipeline = PrefixWritePipeline(
        limits=replace(
            PREFIX_WRITE_LIMITS, max_batch_payload_bytes=1, max_buffered_payload_bytes=2
        ),
        max_batch_rows=256,
        max_batch_tokens=4_194_304,
        sink=sink,
        prepare=prepare,
    )
    try:
        future = pipeline.submit(_inventory("one", ["a"]), deadline=None)
        with pytest.raises(ValueError, match="Only a single row"):
            future.result(5)
        sink.stage_generation_prefix_batch.assert_not_called()
    finally:
        pipeline.close(wait=True)


def test_byte_limit_splits_a_batch_and_seals_every_row() -> None:
    recovery, sink, states, inventory = _batch_worker_fixture([2, 3, 4])
    # Each row's estimate is about 600 bytes: two fit, a third does not.
    pipeline = _pipeline(
        recovery, max_batch_payload_bytes=1300, max_buffered_payload_bytes=2600
    )
    try:
        receipt = pipeline._pipeline.submit(inventory, deadline=None).result(5)
        receipt.validate_for(inventory)
        assert sink.batches == [[3, 4], [5]]  # First chunks include the prompt.
        assert all(ack.disposition == "durable_prefix" for ack in receipt.prefixes)
        assert not any(state.snapshot().cut_in_progress for state in states)
    finally:
        pipeline._pipeline.close(wait=True)
        pipeline.shutdown()
        recovery.shutdown()


def test_admission_limit_rejects_new_inventories_but_shares_duplicates() -> None:
    release = threading.Event()
    sink = MagicMock()
    sink.stage_generation_prefix_batch.side_effect = lambda records, **kwargs: (
        release.wait(5),
        [record.key for record in records],
    )[1]

    def prepare(
        inventory: Any, *, deadline: float | None, held_only: bool
    ) -> CutWrites:
        record = SimpleNamespace(key=inventory.inventory_digest, delta_len=1)
        keys = yield PrefixWrite(
            "cut", (record,), (0,), (None,), 1, lambda error: [None]
        )
        yield WriteSettled()
        return keys

    pipeline = PrefixWritePipeline(
        limits=replace(PREFIX_WRITE_LIMITS, max_pending_inventories=1),
        max_batch_rows=256,
        max_batch_tokens=4_194_304,
        sink=sink,
        prepare=prepare,
    )
    try:
        first = pipeline.submit(_inventory("one", ["a"]), deadline=None)
        assert pipeline.submit(_inventory("one", ["a"]), deadline=None) is first
        with pytest.raises(RuntimeError, match="admission limit"):
            pipeline.submit(_inventory("two", ["b"]), deadline=None)
        release.set()
        assert first.result(5) == ["one"]
    finally:
        release.set()
        pipeline.close(wait=True)


def test_prepared_batch_that_expires_before_put_fails_without_a_write(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A batch prepared in time but written after its deadline is not written."""
    from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

    recovery, sink, states, inventory = _batch_worker_fixture([2, 3], batch_size=1)
    clock = {"now": 0.0}
    fake_time = SimpleNamespace(monotonic=lambda: clock["now"])
    monkeypatch.setattr(recovery_module, "time", fake_time)
    monkeypatch.setattr(write_pipeline_module, "time", fake_time)
    entered, release = threading.Event(), threading.Event()
    sink.before_write = lambda: (entered.set(), release.wait(5))
    pipeline = _pipeline(recovery)
    first, second = (
        GenerationCutInventory.build(
            checkpoint_id=inventory.checkpoint_id,
            server_name=inventory.server_name,
            active_prefixes=[prefix],
        )
        for prefix in inventory.active_prefixes
    )
    try:
        with caplog.at_level(logging.INFO):
            written = pipeline._pipeline.submit(first, deadline=5.0)
            assert entered.wait(5)
            expired = pipeline._pipeline.submit(second, deadline=5.0)
            # The second call is prepared while the first PUT is blocked.
            deadline = time.monotonic() + 5
            while not states[1].snapshot().cut_in_progress:
                assert time.monotonic() < deadline
                time.sleep(0.001)
            clock["now"] = 10.0
            release.set()
            assert written.result(5).prefixes[0].disposition == "durable_prefix"
            late = expired.result(5)
        late.validate_for(second)
        assert late.prefixes[0].disposition == "durable_failure"
        assert sink.batches == [[3]]
        _assert_rolled_back(states[1], 3)
        assert {"control_deadline": 1} in [
            record.args[-1]
            for record in caplog.records
            if record.getMessage().startswith("generation cut ")
        ]
    finally:
        release.set()
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
