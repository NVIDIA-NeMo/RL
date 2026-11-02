# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Concurrency, ownership and failure regressions for shared prefix writes."""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pytest

from nemo_rl.data_plane.generation_prefix_write_batcher import (
    GenerationPrefixWriteBatcher,
)
from nemo_rl.utils.prefix_cut_diagnostics import prefix_cut_probe, prefix_cut_timing


@dataclass
class Record:
    name: str
    token_ids_delta: list[int] = field(default_factory=lambda: [1, 2])


class Sink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.bad: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.incomplete = False
        self.cleared: list[str] = []

    def stage_generation_prefix_batch(
        self,
        records: Any,
        *,
        checkpoint_id: str,
        chunk_sequences: Any,
        attachments: Any,
    ) -> list[str]:
        self.calls.append((checkpoint_id, [record.name for record in records]))
        assert len(records) == len(chunk_sequences) == len(attachments)
        self.entered.set()
        assert self.release.wait(5), "test write was not released"
        if self.bad is not None and any(record.name == self.bad for record in records):
            raise RuntimeError("bad row")
        with prefix_cut_timing("fake_tq_put"):
            keys = [
                f"{checkpoint_id}/{record.name}/{sequence}"
                for record, sequence in zip(records, chunk_sequences, strict=True)
            ]
        return keys[:-1] if self.incomplete else keys

    def clear(self, keys: list[str]) -> None:
        self.cleared.extend(keys)


def submit(
    writer: GenerationPrefixWriteBatcher,
    name: str,
    *,
    checkpoint: str = "cp",
    deadline: float | None = None,
    coalesce: bool = True,
    tokens: int = 2,
    payload_bytes: int = 16,
) -> list[str]:
    return writer.stage(
        [Record(name, [1] * tokens)],
        checkpoint_id=checkpoint,
        chunk_sequences=[7],
        attachments=[None],
        payload_bytes=payload_bytes,
        deadline=deadline,
        coalesce=coalesce,
    )


def test_combines_admitted_requests_and_demultiplexes_keys() -> None:
    sink = Sink()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0.1
    )
    start = threading.Barrier(4)

    def call(index: int) -> list[str]:
        start.wait()
        return submit(writer, f"r{index}")

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(call, range(4)))
        assert results == [[f"cp/r{index}/7"] for index in range(4)]
        assert len(sink.calls) == 1
        assert set(sink.calls[0][1]) == {f"r{index}" for index in range(4)}
        assert writer.stats()["max_requests_per_batch"] == 4
        assert writer.stats()["pending_requests"] == 0
    finally:
        writer.close()


@pytest.mark.parametrize(
    "constraint", ["checkpoint", "rows", "tokens", "bytes", "isolated_retry"]
)
def test_does_not_merge_incompatible_or_over_limit_requests(constraint: str) -> None:
    sink = Sink()
    writer = GenerationPrefixWriteBatcher(
        sink,
        max_rows=1 if constraint == "rows" else 256,
        max_tokens=3 if constraint == "tokens" else 1000,
        max_payload_bytes=20 if constraint == "bytes" else 1000,
        wait_seconds=0.05,
    )
    barrier = threading.Barrier(2)

    def call(index: int) -> list[str]:
        barrier.wait()
        return submit(
            writer,
            f"r{index}",
            checkpoint=f"cp{index}" if constraint == "checkpoint" else "cp",
            coalesce=constraint != "isolated_retry",
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(call, range(2)))
        assert len(sink.calls) == 2
        assert all(len(names) == 1 for _, names in sink.calls)
    finally:
        writer.close()


def test_bad_shared_member_does_not_fail_another_inventory() -> None:
    sink = Sink()
    sink.bad = "bad"
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0.1
    )
    start = threading.Barrier(2)

    def call(name: str) -> list[str]:
        start.wait()
        return submit(writer, name)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            good = pool.submit(call, "good")
            bad = pool.submit(call, "bad")
            assert good.result(3) == ["cp/good/7"]
            with pytest.raises(RuntimeError, match="bad row"):
                bad.result(3)
        assert len(sink.calls[0][1]) == 2
        assert sorted(names[0] for _, names in sink.calls[1:]) == ["bad", "good"]
    finally:
        writer.close()


def test_expired_queued_write_never_starts_tq() -> None:
    sink = Sink()
    sink.release.clear()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(submit, writer, "active")
            assert sink.entered.wait(1)
            expired = pool.submit(
                submit, writer, "expired", deadline=time.monotonic() + 0.04
            )
            with pytest.raises(TimeoutError):
                expired.result(1)
            sink.release.set()
            assert first.result(1) == ["cp/active/7"]
        writer.close()
        assert sink.calls == [("cp", ["active"])]
    finally:
        sink.release.set()
        writer.close()


def test_running_put_retains_ownership_past_deadline() -> None:
    sink = Sink()
    sink.release.clear()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                submit, writer, "active", deadline=time.monotonic() + 0.04
            )
            assert sink.entered.wait(1)
            time.sleep(0.06)
            assert not result.done(), (
                "A running write lost its caller before TQ completed"
            )
            sink.release.set()
            assert result.result(1) == ["cp/active/7"]
    finally:
        sink.release.set()
        writer.close()


def test_capacity_wait_obeys_deadline() -> None:
    sink = Sink()
    sink.release.clear()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, max_pending=1, wait_seconds=0
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(submit, writer, "active")
            assert sink.entered.wait(1)
            with pytest.raises(TimeoutError, match="capacity"):
                submit(writer, "full", deadline=time.monotonic() + 0.03)
            sink.release.set()
            first.result(1)
        assert sink.calls == [("cp", ["active"])]
    finally:
        sink.release.set()
        writer.close()


def test_close_drains_accepted_work_and_rejects_new_submissions() -> None:
    sink = Sink()
    sink.release.clear()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(submit, writer, "first")
            assert sink.entered.wait(1)
            second = pool.submit(submit, writer, "second")
            with writer._condition:
                assert writer._condition.wait_for(lambda: writer._pending == 2, 1)
            with pytest.raises(TimeoutError, match="shutdown"):
                writer.close(timeout_s=0.01)
            with pytest.raises(RuntimeError, match="closed"):
                submit(writer, "after-close")
            sink.release.set()
            assert first.result(1) == ["cp/first/7"]
            assert second.result(1) == ["cp/second/7"]
        writer.close()
        writer.close()
    finally:
        sink.release.set()
        writer.close()


def test_incomplete_sink_result_is_rejected_and_returned_keys_cleared() -> None:
    sink = Sink()
    sink.incomplete = True
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0
    )
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            writer.stage(
                [Record("a"), Record("b")],
                checkpoint_id="cp",
                chunk_sequences=[0, 1],
                attachments=[None, None],
                payload_bytes=16,
                deadline=None,
            )
        assert sink.cleared == ["cp/a/0"]
    finally:
        writer.close()


def test_shared_put_telemetry_is_separate_from_inventory_waits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    sink = Sink()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0.1
    )
    start = threading.Barrier(2)

    def call(index: int) -> list[str]:
        with prefix_cut_probe(
            enabled=True,
            checkpoint_id="cp",
            inventory_digest=f"inv-{index}",
            calls=1,
            deadline=None,
        ):
            start.wait()
            with prefix_cut_timing("inventory_write_batch_wait"):
                return submit(writer, str(index))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(call, range(2)))
    finally:
        writer.close()
    inventories = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("prefix_cut_probe ")
    ]
    shared = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("prefix_write_batch_probe ")
    ]
    assert len(inventories) == 2 and len(shared) == 1
    assert all("fake_tq_put" not in item["phases"] for item in inventories)
    assert shared[0]["inventories"] == 2 and shared[0]["calls"] == 2
    assert shared[0]["phases"]["fake_tq_put"]["calls"] == 1


def test_idle_writer_does_not_retain_previously_written_records() -> None:
    sink = Sink()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=1000, wait_seconds=0
    )
    try:
        record = Record("temporary")
        reference = weakref.ref(record)
        writer.stage(
            [record],
            checkpoint_id="cp",
            chunk_sequences=[0],
            attachments=[None],
            payload_bytes=16,
            deadline=None,
        )
        del record
        until = time.monotonic() + 1
        while reference() is not None and time.monotonic() < until:
            time.sleep(0.005)
            gc.collect()
        assert reference() is None
    finally:
        writer.close()
