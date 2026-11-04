# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Concurrency, byte bounds, thread ownership and failure tests for cut streaming."""

import asyncio
import contextvars
import threading
import time
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.generation_cut_control import GenerationCutControl
from nemo_rl.models.generation.generation_cut_pipeline import (
    GenerationCutPipeline,
    PrefixCutBatchReleased,
    PrefixCutWrite,
)
from nemo_rl.models.generation.vllm.config import GenerationCutControlConfig


class Sink:
    def __init__(self):
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.release.set()
        self.calls = []
        self.rows = {}
        self.active = 0
        self.peak_active = 0
        self.fail = False

    def stage_generation_prefix_batch(
        self, records, *, checkpoint_id, chunk_sequences, attachments=None
    ):
        with self.lock:
            self.calls.append([r.identifier for r in records])
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            assert self.release.wait(5), "Test did not release native write"
            if self.fail:
                raise ValueError("injected put failure")
            keys = [f"{checkpoint_id}/{r.identifier}" for r in records]
            with self.lock:
                self.rows.update(zip(keys, records, strict=True))
            return keys
        finally:
            with self.lock:
                self.active -= 1

    def clear(self, keys):
        with self.lock:
            for key in keys:
                self.rows.pop(key, None)


def inventory(index, *, checkpoint="checkpoint", size=100, chunks=1):
    return SimpleNamespace(
        checkpoint_id=checkpoint,
        inventory_digest=str(index),
        index=index,
        size=size,
        chunks=chunks,
    )


def factory(sink, events):
    def prepare(item, *, deadline):
        result = []
        for index in range(item.chunks):
            lock = threading.RLock()
            identifier = f"{item.index}-{index}"
            with lock:
                events.append(("prepare", identifier, threading.get_ident()))
                record = SimpleNamespace(identifier=identifier, token_ids_delta=[1])

                def fallback(record=record, checkpoint=item.checkpoint_id):
                    try:
                        return sink.stage_generation_prefix_batch(
                            [record], checkpoint_id=checkpoint, chunk_sequences=[0]
                        )
                    except ValueError:
                        return [None]

                try:
                    keys = yield PrefixCutWrite(
                        item.checkpoint_id,
                        (record,),
                        (0,),
                        (None,),
                        item.size,
                        deadline,
                        fallback,
                        False,
                    )
                    events.append(("ack", identifier, threading.get_ident()))
                    result.extend(keys)
                except BaseException:
                    events.append(("rollback", identifier, threading.get_ident()))
                    raise
            yield PrefixCutBatchReleased()
        return result

    return prepare


def pipeline(sink, events, *, buffer=1200, batch=300, writers=4, wait=0):
    return GenerationCutPipeline(
        sink,
        prepare=factory(sink, events),
        max_buffered_bytes=buffer,
        max_batch_bytes=batch,
        max_rows=256,
        max_tokens=10000,
        write_concurrency=writers,
        coalesce_wait_seconds=wait,
    )


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for test condition")
        time.sleep(0.002)


def test_prepares_across_inventories_while_four_writes_are_running():
    sink, events = Sink(), []
    sink.release.clear()
    p = pipeline(sink, events, buffer=1800, batch=300)
    futures = []
    try:
        for i in range(10):
            futures.append(p.submit(inventory(i, size=300), deadline=None))
        until(lambda: sink.active == 4)
        until(lambda: len([e for e in events if e[0] == "prepare"]) == 6)
        assert not any(f.done() for f in futures)
        assert p.stats()["buffered_payload_bytes"] == 1800
        assert p.stats()["peak_buffered_payload_bytes"] <= 1800
        sink.release.set()
        assert [f.result(timeout=5) for f in futures] == [
            [f"checkpoint/{i}-0"] for i in range(10)
        ]
        assert sink.peak_active == 4
        assert len({e[2] for e in events}) == 1
        until(lambda: p.stats()["pending_inventories"] == 0)
        assert p.stats()["buffered_payload_bytes"] == 0
    finally:
        sink.release.set()
        p.close(wait=True)


def test_merges_prepared_rows_across_inventory_boundaries_and_keeps_key_order():
    sink, events = Sink(), []
    p = pipeline(sink, events, wait=0.02)
    try:
        # Stop the owner from dispatching before the complete metadata burst.
        with p._condition:
            futures = [p.submit(inventory(i), deadline=None) for i in range(8)]
        assert [f.result(timeout=5) for f in futures] == [
            [f"checkpoint/{i}-0"] for i in range(8)
        ]
        assert any(len(call) > 1 for call in sink.calls)
        assert all(len(call) <= 3 for call in sink.calls)
        assert sorted(key for call in sink.calls for key in call) == [
            f"{i}-0" for i in range(8)
        ]
    finally:
        p.close(wait=True)


def test_multi_batch_inventory_resumes_only_on_its_original_owner_thread():
    sink, events = Sink(), []
    p = pipeline(sink, events)
    try:
        futures = [p.submit(inventory(i, chunks=3), deadline=None) for i in range(6)]
        for i, future in enumerate(futures):
            assert future.result(timeout=5) == [f"checkpoint/{i}-{j}" for j in range(3)]
        assert len({e[2] for e in events}) == 1
        assert [e[0] for e in events].count("ack") == 18
        assert p.stats()["peak_buffered_payload_bytes"] <= 1200
    finally:
        p.close(wait=True)


def test_write_failure_uses_each_inventory_fallback_without_false_success():
    sink, events = Sink(), []
    sink.fail = True
    p = pipeline(sink, events, wait=0.02)
    try:
        futures = [p.submit(inventory(i), deadline=None) for i in range(5)]
        assert all(f.result(timeout=5) == [None] for f in futures)
        assert not sink.rows
        assert p.stats()["failed_merged_writes"] > 0
    finally:
        p.close(wait=True)


def test_one_failed_fallback_does_not_orphan_other_inventories_successful_rows():
    class PartialSink(Sink):
        def stage_generation_prefix_batch(self, records, **kwargs):
            if len(records) > 1:
                raise ValueError("reject merged put")
            if records[0].identifier == "1-0":
                raise RuntimeError("reject one inventory")
            return super().stage_generation_prefix_batch(records, **kwargs)

    sink, events = PartialSink(), []
    p = pipeline(sink, events, wait=0.05)
    try:
        with p._condition:
            futures = [p.submit(inventory(i), deadline=None) for i in range(3)]
        assert futures[0].result(timeout=5) == ["checkpoint/0-0"]
        with pytest.raises(RuntimeError, match="reject one inventory"):
            futures[1].result(timeout=5)
        assert futures[2].result(timeout=5) == ["checkpoint/2-0"]
        assert set(sink.rows) == {"checkpoint/0-0", "checkpoint/2-0"}
        assert any(e[:2] == ("rollback", "1-0") for e in events)
    finally:
        p.close(wait=True)


def test_expired_member_does_not_poison_another_inventory_in_a_merged_write():
    sink, events = Sink(), []
    p = pipeline(sink, events, wait=0.1)
    try:
        with p._condition:
            expired = p.submit(inventory(1), deadline=time.monotonic() + 0.03)
            valid = p.submit(inventory(2), deadline=None)
        with pytest.raises(TimeoutError):
            expired.result(timeout=5)
        assert valid.result(timeout=5) == ["checkpoint/2-0"]
        assert set(sink.rows) == {"checkpoint/2-0"}
    finally:
        p.close(wait=True)


def test_expired_unstarted_inventory_never_freezes_or_writes():
    sink, events = Sink(), []
    p = pipeline(sink, events)
    try:
        future = p.submit(inventory(1), deadline=time.monotonic() - 1)
        with pytest.raises(TimeoutError):
            future.result(timeout=5)
        assert not events and not sink.calls
    finally:
        p.close(wait=True)


def test_deadline_expiring_after_preparation_never_starts_a_put_or_retry():
    sink, events = Sink(), []
    p = pipeline(sink, events, wait=0.1)
    try:
        future = p.submit(inventory(1), deadline=time.monotonic() + 0.03)
        until(lambda: bool(events))
        with pytest.raises(TimeoutError):
            future.result(timeout=5)
        assert not sink.calls
        assert any(e[0] == "rollback" for e in events)
        assert not p.stats()["buffered_payload_bytes"]
    finally:
        p.close(wait=True)


def test_native_put_keeps_ownership_after_deadline_and_rejects_cancellation():
    sink, events = Sink(), []
    sink.release.clear()
    p = pipeline(sink, events)
    try:
        future = p.submit(inventory(1), deadline=time.monotonic() + 0.05)
        until(lambda: sink.active == 1)
        time.sleep(0.06)
        assert not future.cancel()
        assert not future.done()
        assert not any(e[0] == "rollback" for e in events)
        sink.release.set()
        assert future.result(timeout=5) == ["checkpoint/1-0"]
    finally:
        sink.release.set()
        p.close(wait=True)


@pytest.mark.asyncio
async def test_cancelled_http_waiter_keeps_call_reservation_until_pipeline_finishes():
    sink, events = Sink(), []
    sink.release.clear()
    p = pipeline(sink, events)
    control = GenerationCutControl(max_workers=1, max_active_inventories=8)

    async def execute(item):
        return await asyncio.shield(asyncio.wrap_future(p.submit(item, deadline=None)))

    async def run(item, call):
        return await control.run(
            checkpoint_id=item.checkpoint_id,
            inventory_digest=item.inventory_digest,
            call_ids=frozenset({call}),
            deadline=None,
            operation=lambda: execute(item),
        )

    first = asyncio.create_task(run(inventory(1), "same-call"))
    try:
        await asyncio.to_thread(until, lambda: sink.active == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert len(control._active) == 1
        other = asyncio.create_task(run(inventory(2), "same-call"))
        await asyncio.sleep(0)
        assert p.stats()["submitted_inventories"] == 1
        assert len(control._pending) == 1
        sink.release.set()
        assert await asyncio.wait_for(other, 5) == ["checkpoint/2-0"]
        assert not control._active
        assert not control._pending
    finally:
        sink.release.set()
        control.close()
        await asyncio.to_thread(p.close, wait=True)


@pytest.mark.asyncio
async def test_admission_joins_duplicates_and_keeps_checkpoint_epoch_barrier():
    sink, events = Sink(), []
    sink.release.clear()
    p = pipeline(sink, events)
    control = GenerationCutControl(max_workers=1, max_active_inventories=8)

    async def run(item):
        async def execute():
            return await asyncio.shield(
                asyncio.wrap_future(p.submit(item, deadline=None))
            )

        return await control.run(
            checkpoint_id=item.checkpoint_id,
            inventory_digest=item.inventory_digest,
            call_ids=frozenset({str(item.index)}),
            deadline=None,
            operation=execute,
        )

    first = asyncio.create_task(run(inventory(1)))
    duplicate = asyncio.create_task(run(inventory(1)))
    try:
        await asyncio.to_thread(until, lambda: sink.active == 1)
        next_epoch = asyncio.create_task(
            run(inventory(2, checkpoint="next-checkpoint"))
        )
        await asyncio.sleep(0)
        assert p.stats()["submitted_inventories"] == 1
        sink.release.set()
        a, b, c = await asyncio.wait_for(
            asyncio.gather(first, duplicate, next_epoch), 5
        )
        assert a == b == ["checkpoint/1-0"]
        assert c == ["next-checkpoint/2-0"]
        assert p.stats()["submitted_inventories"] == 2
    finally:
        sink.release.set()
        control.close()
        await asyncio.to_thread(p.close, wait=True)


def test_close_retains_native_write_and_lifecycle_ownership_until_completion():
    sink, events = Sink(), []
    sink.release.clear()
    p = pipeline(sink, events, buffer=300, batch=300, writers=1)
    try:
        first = p.submit(inventory(1, size=300), deadline=None)
        second = p.submit(inventory(2, size=300), deadline=None)
        until(lambda: sink.active == 1)
        p.close(wait=False)
        assert not first.done()
        assert not any(e[0] in ("ack", "rollback") for e in events)
        with pytest.raises(RuntimeError, match="closed"):
            p.submit(inventory(3), deadline=None)
        sink.release.set()
        p.close(wait=True)
        # A successful native write is acknowledged on its owner before close
        # rolls back/ends the rest of the stream. No write is abandoned.
        assert ("ack", "1-0", events[0][2]) in events
        assert first.done() and second.done()
        assert not sink.active and not p.stats()["buffered_payload_bytes"]
    finally:
        sink.release.set()
        p.close(wait=True)


def test_oversized_prepared_payload_rolls_back_and_does_not_block_other_jobs():
    sink, events = Sink(), []
    p = pipeline(sink, events)
    try:
        bad = p.submit(inventory(1, size=301), deadline=None)
        good = p.submit(inventory(2), deadline=None)
        with pytest.raises(ValueError, match="byte reservation"):
            bad.result(timeout=5)
        assert good.result(timeout=5) == ["checkpoint/2-0"]
        assert any(e[:2] == ("rollback", "1-0") for e in events)
        assert p.stats()["buffered_payload_bytes"] == 0
    finally:
        p.close(wait=True)


def test_per_inventory_context_is_retained_across_yields_and_writers():
    sink, events = Sink(), []
    variable = contextvars.ContextVar("pipeline_test", default="missing")
    observed = []
    underlying = factory(sink, events)

    def prepare(item, *, deadline):
        observed.append((item.index, variable.get()))
        result = yield from underlying(item, deadline=deadline)
        observed.append((item.index, variable.get()))
        return result

    p = GenerationCutPipeline(
        sink,
        prepare=prepare,
        max_buffered_bytes=1200,
        max_batch_bytes=300,
        max_rows=256,
        max_tokens=10000,
        write_concurrency=4,
        coalesce_wait_seconds=0,
    )
    try:
        futures = []
        for i in range(8):
            token = variable.set(str(i))
            futures.append(p.submit(inventory(i, chunks=2), deadline=None))
            variable.reset(token)
        for future in futures:
            future.result(timeout=5)
        assert all(value == str(index) for index, value in observed)
    finally:
        p.close(wait=True)


@pytest.mark.parametrize(
    "updates",
    [
        {"group_pending_inventories": True},
        {"batch_inventory_writes": True},
        {"pipeline": {"enabled": True, "max_buffered_payload_bytes": 1}},
    ],
)
def test_pipeline_config_rejects_competing_modes_and_insufficient_buffer(updates):
    options = dict(pipeline={"enabled": True})
    options.update(updates)
    with pytest.raises(ValueError):
        GenerationCutControlConfig(**options)
