# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Coalesce independent inventories' prepared prefix writes on a dedicated thread."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from nemo_rl.utils.prefix_cut_diagnostics import (
    prefix_cut_probe_active,
    prefix_write_batch_probe,
)

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import StagedCallRecord

LOGGER = logging.getLogger(__name__)


class PrefixWriteSink(Protocol):
    """The existing TQTokenSink batch-write and rollback contract."""

    def stage_generation_prefix_batch(
        self,
        records: Sequence[StagedCallRecord],
        *,
        checkpoint_id: str,
        chunk_sequences: Sequence[int],
        attachments: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[str]: ...

    def clear(self, keys: list[str]) -> None: ...


@dataclass(eq=False)
class _Write:
    checkpoint_id: str
    records: tuple[StagedCallRecord, ...]
    chunk_sequences: tuple[int, ...]
    attachments: tuple[Mapping[str, Any] | None, ...]
    tokens: int
    payload_bytes: int
    deadline: float | None
    coalesce: bool
    measured: bool
    queued_at: float = field(default_factory=time.monotonic)
    future: Future[list[str]] = field(default_factory=Future)


class GenerationPrefixWriteBatcher:
    """Share puts without moving transaction ownership out of cut executors.

    Only already-admitted inventories contribute rows. The caller keeps its
    lifecycle locks until its own ordered keys are returned. A queued request
    may time out without writing; once a put starts, the caller waits for its
    real outcome even beyond the deadline, so no live write loses ownership.

    Limits govern merging whole requests. An individual request exceeding a
    merge limit is written alone, retaining the existing inventory batching
    bounds. ``coalesce=False`` isolates existing per-row failure retries.
    """

    def __init__(
        self,
        sink: PrefixWriteSink,
        *,
        max_rows: int,
        max_tokens: int,
        max_payload_bytes: int = 256 * 1024 * 1024,
        max_pending: int = 16,
        wait_seconds: float = 0.01,
    ) -> None:
        for name, value in (
            ("max_rows", max_rows),
            ("max_tokens", max_tokens),
            ("max_payload_bytes", max_payload_bytes),
            ("max_pending", max_pending),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("wait_seconds must be finite and nonnegative")
        self._sink = sink
        self._max_rows = max_rows
        self._max_tokens = max_tokens
        self._max_bytes = max_payload_bytes
        self._max_pending = max_pending
        self._wait_seconds = wait_seconds
        self._condition = threading.Condition()
        self._queue: deque[_Write] = deque()
        self._pending = 0
        self._active = False
        self._closed = False
        self._stats: dict[str, int | float] = {
            "batches": 0,
            "failed_batches": 0,
            "requests": 0,
            "rows": 0,
            "max_requests_per_batch": 0,
            "max_rows_per_batch": 0,
            "put_wall_s": 0.0,
            "put_thread_cpu_s": 0.0,
        }
        self._thread = threading.Thread(
            target=self._run, name="nrl-prefix-write-batcher", daemon=True
        )
        self._thread.start()

    def stage(
        self,
        records: Sequence[StagedCallRecord],
        *,
        checkpoint_id: str,
        chunk_sequences: Sequence[int],
        attachments: Sequence[Mapping[str, Any] | None],
        payload_bytes: int,
        deadline: float | None,
        coalesce: bool = True,
    ) -> list[str]:
        """Write one caller's rows and return only that caller's ordered keys."""
        if not (len(records) == len(chunk_sequences) == len(attachments)):
            raise ValueError(
                "records, sequences and attachments must have equal lengths"
            )
        if type(payload_bytes) is not int or payload_bytes < 0:
            raise ValueError("payload_bytes must be a nonnegative integer")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite or None")
        if not records:
            return []
        request = _Write(
            checkpoint_id,
            tuple(records),
            tuple(chunk_sequences),
            tuple(attachments),
            sum(len(record.token_ids_delta) for record in records),
            payload_bytes,
            deadline,
            coalesce,
            prefix_cut_probe_active(),
        )
        with self._condition:
            while self._pending >= self._max_pending and not self._closed:
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "prefix write deadline expired waiting for capacity"
                    )
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("prefix write batcher is closed")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("prefix write deadline expired before enqueue")
            self._pending += 1
            self._queue.append(request)
            self._condition.notify_all()
        try:
            return request.future.result(timeout=self._remaining(deadline))
        except TimeoutError:
            # Serialize cancellation with the writer claiming a put. A running
            # future cannot be cancelled and must retain its caller's ownership.
            with self._condition:
                cancelled = request.future.cancel()
                self._condition.notify_all()
            if cancelled:
                raise TimeoutError("prefix write deadline expired before put") from None
            return request.future.result()

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    def _collect(self) -> list[_Write] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._queue or self._closed)
            if not self._queue:
                return None
            first = self._queue.popleft()
            batch = [first]
            rows, tokens, size = len(first.records), first.tokens, first.payload_bytes
            until = first.queued_at + self._wait_seconds
            while (
                first.coalesce
                and rows < self._max_rows
                and tokens < self._max_tokens
                and size < self._max_bytes
            ):
                if self._queue:
                    candidate = self._queue[0]
                    if (
                        not candidate.coalesce
                        or candidate.checkpoint_id != first.checkpoint_id
                        or rows + len(candidate.records) > self._max_rows
                        or tokens + candidate.tokens > self._max_tokens
                        or size + candidate.payload_bytes > self._max_bytes
                    ):
                        break
                    batch.append(self._queue.popleft())
                    rows += len(candidate.records)
                    tokens += candidate.tokens
                    size += candidate.payload_bytes
                    continue
                deadlines = [
                    item.deadline for item in batch if item.deadline is not None
                ]
                remaining = min([until, *deadlines]) - time.monotonic()
                if remaining <= 0 or self._closed:
                    break
                self._condition.wait(remaining)
            ready = []
            for item in batch:
                if item.future.cancelled():
                    self._pending -= 1
                elif item.deadline is not None and time.monotonic() >= item.deadline:
                    item.future.set_exception(
                        TimeoutError("prefix write expired before put")
                    )
                    self._pending -= 1
                elif item.future.set_running_or_notify_cancel():
                    ready.append(item)
                else:
                    self._pending -= 1
            self._active = bool(ready)
            self._condition.notify_all()
            return ready

    def _put(self, batch: list[_Write]) -> list[str]:
        records = [record for item in batch for record in item.records]
        sequences = [sequence for item in batch for sequence in item.chunk_sequences]
        attachments = [attachment for item in batch for attachment in item.attachments]
        started, cpu_started = time.monotonic(), time.thread_time()
        failure: Exception | None = None
        keys: list[str] = []
        with self._condition:
            batch_id = int(self._stats["batches"]) + 1
        try:
            with prefix_write_batch_probe(
                enabled=any(item.measured for item in batch),
                checkpoint_id=batch[0].checkpoint_id,
                batch_id=f"writer-{batch_id}",
                inventories=len(batch),
                rows=len(records),
            ):
                keys = self._sink.stage_generation_prefix_batch(
                    records,
                    checkpoint_id=batch[0].checkpoint_id,
                    chunk_sequences=sequences,
                    attachments=attachments,
                )
                if len(keys) != len(records):
                    self._sink.clear(keys)
                    raise RuntimeError("prefix sink returned an incomplete batch")
        except Exception as error:
            # The sink discards all partially written keys on failure. Original
            # cut callers keep their independent per-row fallback/rollback path.
            failure = error
        finally:
            elapsed, cpu = time.monotonic() - started, time.thread_time() - cpu_started
            with self._condition:
                self._stats["batches"] += 1
                self._stats["failed_batches"] += int(failure is not None)
                self._stats["requests"] += len(batch)
                self._stats["rows"] += len(records)
                self._stats["max_requests_per_batch"] = max(
                    self._stats["max_requests_per_batch"], len(batch)
                )
                self._stats["max_rows_per_batch"] = max(
                    self._stats["max_rows_per_batch"], len(records)
                )
                self._stats["put_wall_s"] += elapsed
                self._stats["put_thread_cpu_s"] += cpu
                self._condition.notify_all()
        LOGGER.info(
            "generation_prefix_write_batch %s",
            json.dumps(
                {
                    "checkpoint_id": batch[0].checkpoint_id,
                    "batch_id": batch_id,
                    "inventories": len(batch),
                    "rows": len(records),
                    "tokens": sum(item.tokens for item in batch),
                    "payload_bytes": sum(item.payload_bytes for item in batch),
                    "max_queue_wait_s": max(started - item.queued_at for item in batch),
                    "put_wall_s": elapsed,
                    "put_thread_cpu_s": cpu,
                    "failed": failure is not None,
                },
                sort_keys=True,
            ),
        )
        if failure is not None:
            raise failure
        return keys

    def _write(self, batch: list[_Write]) -> None:
        results: list[tuple[_Write, list[str] | Exception]] = []
        try:
            keys = self._put(batch)
        except Exception as error:
            if len(batch) == 1:
                results.append((batch[0], error))
            else:
                # A bad member of a shared put must not fail another inventory.
                # Retry each request separately; its original caller still owns
                # per-row fallback if that independent request also fails.
                for item in batch:
                    if item.deadline is not None and time.monotonic() >= item.deadline:
                        results.append(
                            (item, TimeoutError("prefix write expired before retry"))
                        )
                        continue
                    try:
                        result = self._put([item])
                    except Exception as request_error:
                        results.append((item, request_error))
                    else:
                        results.append((item, result))
        else:
            offset = 0
            for item in batch:
                count = len(item.records)
                results.append((item, keys[offset : offset + count]))
                offset += count
        with self._condition:
            self._pending -= len(batch)
            self._active = False
            self._condition.notify_all()
        for item, result in results:
            if isinstance(result, Exception):
                item.future.set_exception(result)
            else:
                item.future.set_result(result)

    def _run(self) -> None:
        try:
            while True:
                batch = self._collect()
                if batch is None:
                    return
                if batch:
                    self._write(batch)
                del batch  # Do not retain the previous cohort while idle.
        except Exception as error:
            # A failed daemon must wake callers rather than strand native cuts.
            with self._condition:
                self._closed = True
                while self._queue:
                    item = self._queue.popleft()
                    if not item.future.done():
                        item.future.set_exception(error)
                    self._pending -= 1
                self._condition.notify_all()
            LOGGER.exception("generation prefix writer stopped unexpectedly")

    def stats(self) -> dict[str, int | float]:
        """Return bounded counters; queued requests still own their cut slots."""
        with self._condition:
            return {
                **self._stats,
                "pending_requests": self._pending,
                "active_puts": int(self._active),
            }

    def close(self, *, timeout_s: float = 30.0) -> None:
        """Reject new submissions and drain accepted work with a bounded join."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("generation prefix writer did not drain before shutdown")
