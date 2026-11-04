# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Stream inventory preparations into bounded, concurrent prefix writes.

HTTP admission still owns identities, overlap barriers and checkpoint epochs.
One preparation thread owns every transaction generator and its lifecycle locks.
Only immutable prepared records cross into the write pool. A byte reservation
is held until a successful write is acknowledged and sealed, or rolled back.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

from nemo_rl.data_plane.generation_prefix_write_batcher import PrefixWriteSink
from nemo_rl.utils.checkpoint_latency import CheckpointLatencyConfig, emit_latency
from nemo_rl.utils.prefix_cut_diagnostics import (
    prefix_cut_timing,
    prefix_cut_wait,
    prefix_write_batch_probe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PrefixCutWrite:
    """One generator's detached rows, with its existing per-row retry policy."""

    checkpoint_id: str
    records: tuple[Any, ...]
    chunk_sequences: tuple[int, ...]
    attachments: tuple[Mapping[str, Any] | None, ...]
    payload_bytes: int
    deadline: float | None
    fallback: Callable[[], list[str | None]]
    measured: bool

    @property
    def tokens(self) -> int:
        return sum(len(record.token_ids_delta) for record in self.records)


@dataclass(frozen=True, slots=True)
class PrefixCutBatchReleased:
    """A write has been sealed/rolled back; preparation may reserve more bytes."""


CutSteps = Generator[
    PrefixCutWrite | PrefixCutBatchReleased, list[str | None] | None, Any
]


@dataclass(eq=False, slots=True)
class _Job:
    inventory: Any
    deadline: float | None
    context: contextvars.Context
    future: Future[Any]
    queued_at: float
    iterator: CutSteps | None
    request: PrefixCutWrite | None
    reserved_bytes: int


@dataclass(slots=True)
class _Flight:
    jobs: list[_Job]
    future: Future[list[list[str | None] | BaseException]]


class GenerationCutPipeline:
    """Checkpoint-scoped FIFO stream, independent of inventory execution slots.

    Limits cover estimated numeric tensor payload, including queued and active
    writes plus a worst-case batch reservation while preparing. They are not a
    total RSS limit: live engine histories and preparation temporaries remain
    owned by the frontend. Admission bounds inventory metadata separately.
    """

    def __init__(
        self,
        sink: PrefixWriteSink,
        *,
        prepare: Callable[..., CutSteps],
        max_buffered_bytes: int,
        max_batch_bytes: int,
        max_rows: int,
        max_tokens: int,
        write_concurrency: int,
        coalesce_wait_seconds: float,
        latency_diagnostics: CheckpointLatencyConfig | None = None,
    ) -> None:
        for value in (
            max_buffered_bytes,
            max_batch_bytes,
            max_rows,
            max_tokens,
            write_concurrency,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Prefix pipeline limits must be positive integers")
        if max_batch_bytes > max_buffered_bytes:
            raise ValueError(
                "Prefix pipeline buffer must hold at least one maximum batch"
            )
        if not 0 <= coalesce_wait_seconds < float("inf"):
            raise ValueError(
                "Prefix pipeline coalesce wait must be finite and nonnegative"
            )
        self._sink = sink
        self._prepare = prepare
        self._max_buffered = max_buffered_bytes
        self._max_batch = max_batch_bytes
        self._max_rows = max_rows
        self._max_tokens = max_tokens
        self._concurrency = write_concurrency
        self._wait = coalesce_wait_seconds
        self._latency_config = latency_diagnostics
        self._latency_checkpoint: str | None = None
        self._latency_started = 0.0
        self._latency_last_report = 0.0
        self._latency_phases: dict[str, dict[str, float]] = {}
        self._condition = threading.Condition()
        self._incoming: deque[_Job] = deque()
        self._ready: deque[_Job] = deque()
        self._prepared: deque[_Job] = deque()
        self._jobs: set[_Job] = set()
        self._flights: list[_Flight] = []
        self._closed = False
        self._failure: BaseException | None = None
        self._buffered = 0
        self._prepared_since: float | None = None
        self._batch_id = 0
        self._stats: dict[str, int | float] = {
            "submitted_inventories": 0,
            "completed_inventories": 0,
            "failed_inventories": 0,
            "peak_buffered_payload_bytes": 0,
            "write_batches": 0,
            "failed_merged_writes": 0,
            "write_wall_s": 0.0,
            "write_thread_cpu_s": 0.0,
            "max_inventories_per_write": 0,
            "max_rows_per_write": 0,
            "peak_active_writes": 0,
            "prepare_wall_s": 0.0,
            "prepare_thread_cpu_s": 0.0,
            "buffer_wait_s": 0.0,
        }
        self._writers = ThreadPoolExecutor(
            max_workers=write_concurrency,
            thread_name_prefix="nrl-prefix-pipeline-write",
        )
        self._thread = threading.Thread(
            target=self._run, name="nrl-prefix-pipeline-prepare", daemon=True
        )
        self._thread.start()

    def _activity(self, phase: str, wall: float, cpu: float) -> None:
        if self._latency_config is None:
            return
        totals = self._latency_phases.setdefault(
            phase, dict(wall_s=0.0, thread_cpu_s=0.0)
        )
        totals["wall_s"] += wall
        totals["thread_cpu_s"] += cpu

    def _report_latency(self) -> None:
        config = self._latency_config
        if config is None:
            return
        if self._latency_checkpoint is None and self._ready:
            self._latency_checkpoint = self._ready[0].inventory.checkpoint_id
            self._latency_started = time.time()
            self._latency_phases = {}
            self._latency_last_report = 0.0
        if self._latency_checkpoint is None:
            return
        now = time.time()
        with self._condition:
            final = not self._jobs and not self._flights
            pending = len(self._jobs)
            buffered = self._buffered
        if not final and now - self._latency_last_report < config.progress_interval_s:
            return
        emit_latency(
            "generation_cut_pipeline_progress",
            checkpoint_id=self._latency_checkpoint,
            started_unix_s=self._latency_started,
            final=final,
            scheduler_phases={
                name: dict(totals) for name, totals in self._latency_phases.items()
            },
            pending_inventories=pending,
            ready_inventories=len(self._ready),
            prepared_inventories=len(self._prepared),
            active_writes=len(self._flights),
            buffered_payload_bytes=buffered,
            max_buffered_payload_bytes=self._max_buffered,
            preparation_reservation_bytes=self._max_batch,
            write_concurrency=self._concurrency,
            preparation_thread_id=threading.get_ident(),
        )
        self._latency_last_report = now
        if final:
            self._latency_checkpoint = None

    def submit(self, inventory: Any, *, deadline: float | None) -> Future[Any]:
        """Admission must retain checkpoint and call reservations until completion."""
        with self._condition:
            if self._closed:
                if self._failure is not None:
                    raise RuntimeError(
                        "generation cut pipeline is closed"
                    ) from self._failure
                raise RuntimeError("generation cut pipeline is closed")
            future: Future[Any] = Future()
            job = _Job(
                inventory,
                deadline,
                contextvars.copy_context(),
                future,
                time.monotonic(),
                None,
                None,
                0,
            )
            self._jobs.add(job)
            self._incoming.append(job)
            self._stats["submitted_inventories"] += 1
            self._condition.notify_all()
            return future

    def stats(self) -> dict[str, int | float]:
        with self._condition:
            return dict(
                self._stats,
                pending_inventories=len(self._jobs),
                buffered_payload_bytes=self._buffered,
                active_writes=len(self._flights),
                prepared_inventories=len(self._prepared),
                closed=int(self._closed),
                failed=int(self._failure is not None),
            )

    def _reserve(self, job: _Job, amount: int) -> None:
        with self._condition:
            self._buffered += amount - job.reserved_bytes
            job.reserved_bytes = amount
            if not 0 <= self._buffered <= self._max_buffered:
                raise RuntimeError(
                    "generation cut pipeline exceeded its payload reservation"
                )
            self._stats["peak_buffered_payload_bytes"] = max(
                self._stats["peak_buffered_payload_bytes"], self._buffered
            )

    def _finish(
        self, job: _Job, *, result: Any = None, error: BaseException | None = None
    ) -> None:
        if job.iterator is not None:
            try:
                job.context.run(job.iterator.close)
            except BaseException as close_error:
                if error is None:
                    error = close_error
        self._reserve(job, 0)
        job.request = None
        with self._condition:
            self._jobs.discard(job)
            self._stats[
                "completed_inventories" if error is None else "failed_inventories"
            ] += 1
            self._condition.notify_all()
        if not job.future.done():
            if error is None:
                job.future.set_result(result)
            else:
                job.future.set_exception(error)

    def _advance(self, job: _Job) -> None:
        if job.iterator is None:
            if not job.future.set_running_or_notify_cancel():
                self._finish(
                    job, error=RuntimeError("prefix preparation cancelled before start")
                )
                return
            if job.deadline is not None and time.monotonic() >= job.deadline:
                self._finish(
                    job,
                    error=TimeoutError(
                        "prefix pipeline deadline expired before preparation"
                    ),
                )
                return
            job.context.run(
                prefix_cut_wait,
                "pipeline_prepare_queue_wait",
                time.monotonic() - job.queued_at,
            )
        self._reserve(job, self._max_batch)
        started, cpu = time.monotonic(), time.thread_time()
        if self._latency_config is not None:
            emit_latency(
                "generation_cut_pipeline_prepare",
                event="begin",
                checkpoint_id=job.inventory.checkpoint_id,
                inventory_digest=job.inventory.inventory_digest,
                first_prepare=job.iterator is None,
                queued_s=started - job.queued_at,
                active_writes=len(self._flights),
                buffered_payload_bytes=self._buffered,
            )
        try:
            if job.iterator is None:
                job.iterator = job.context.run(
                    self._prepare, job.inventory, deadline=job.deadline
                )
            request = job.context.run(next, job.iterator)
            if not isinstance(request, PrefixCutWrite):
                raise TypeError("prefix preparation must yield a write request")
            if request.checkpoint_id != job.inventory.checkpoint_id:
                raise ValueError("prefix preparation changed checkpoint identity")
            if not 0 < request.payload_bytes <= self._max_batch:
                raise ValueError("prepared prefix batch exceeds its byte reservation")
            if not (
                len(request.records)
                == len(request.chunk_sequences)
                == len(request.attachments)
            ):
                raise ValueError("prepared prefix batch metadata lengths differ")
            job.request = request
            self._reserve(job, request.payload_bytes)
            self._prepared.append(job)
            if self._prepared_since is None:
                self._prepared_since = time.monotonic()
        except StopIteration as finished:
            self._finish(job, result=finished.value)
        except BaseException as error:
            self._finish(job, error=error)
        finally:
            elapsed = time.monotonic() - started
            cpu_elapsed = time.thread_time() - cpu
            with self._condition:
                self._stats["prepare_wall_s"] += elapsed
                self._stats["prepare_thread_cpu_s"] += cpu_elapsed
            self._activity("prepare", elapsed, cpu_elapsed)
            if self._latency_config is not None:
                emit_latency(
                    "generation_cut_pipeline_prepare",
                    event="end",
                    checkpoint_id=job.inventory.checkpoint_id,
                    inventory_digest=job.inventory.inventory_digest,
                    wall_s=elapsed,
                    thread_cpu_s=cpu_elapsed,
                )

    def _write(
        self, jobs: list[_Job], batch_id: int
    ) -> list[list[str | None] | BaseException]:
        requests = [cast(PrefixCutWrite, job.request) for job in jobs]
        records = [record for request in requests for record in request.records]
        sequences = [seq for request in requests for seq in request.chunk_sequences]
        attachments = [value for request in requests for value in request.attachments]
        started, cpu = time.monotonic(), time.thread_time()
        failed_merge = False
        results: list[list[str | None] | BaseException] = []
        try:
            # Per-inventory contexts stay on the preparation thread. Each writer
            # gets a copy plus its own numeric diagnostic probe.
            with (
                prefix_write_batch_probe(
                    enabled=any(request.measured for request in requests)
                    or self._latency_config is not None,
                    checkpoint_id=requests[0].checkpoint_id,
                    batch_id=f"pipeline-{batch_id}",
                    inventories=len(jobs),
                    rows=len(records),
                ),
                prefix_cut_timing("batch_stage_total"),
            ):
                try:
                    if any(
                        request.deadline is not None
                        and time.monotonic() >= request.deadline
                        for request in requests
                    ):
                        raise TimeoutError(
                            "prefix pipeline batch deadline expired before write"
                        )
                    keys = self._sink.stage_generation_prefix_batch(
                        records,
                        checkpoint_id=requests[0].checkpoint_id,
                        chunk_sequences=sequences,
                        attachments=attachments,
                    )
                    if len(keys) != len(records):
                        self._sink.clear(keys)
                        raise ValueError(
                            "prefix sink returned the wrong number of keys"
                        )
                except Exception:
                    failed_merge = True
                    # The sink's native failure path clears partial keys. Each
                    # transaction keeps the existing per-row retry/deadline policy.
                    for request in requests:
                        if (
                            request.deadline is not None
                            and time.monotonic() >= request.deadline
                        ):
                            results.append(
                                TimeoutError(
                                    "prefix pipeline deadline expired before write retry"
                                )
                            )
                            continue
                        try:
                            results.append(request.fallback())
                        except BaseException as error:
                            results.append(error)
                else:
                    offset = 0
                    for request in requests:
                        stop = offset + len(request.records)
                        own_keys: list[str | None] = []
                        own_keys.extend(keys[offset:stop])
                        results.append(own_keys)
                        offset = stop
        finally:
            elapsed = time.monotonic() - started
            cpu_elapsed = time.thread_time() - cpu
            with self._condition:
                self._stats["write_batches"] += 1
                self._stats["failed_merged_writes"] += int(failed_merge)
                self._stats["write_wall_s"] += elapsed
                self._stats["write_thread_cpu_s"] += cpu_elapsed
                self._condition.notify_all()
            LOGGER.info(
                "generation_cut_pipeline_write %s",
                json.dumps(
                    {
                        "checkpoint_id": requests[0].checkpoint_id,
                        "batch_id": f"pipeline-{batch_id}",
                        "pid": os.getpid(),
                        "started_unix_s": time.time() - elapsed,
                        "finished_unix_s": time.time(),
                        "inventories": [job.inventory.inventory_digest for job in jobs],
                        "rows": len(records),
                        "tensor_payload_bytes": sum(
                            request.payload_bytes for request in requests
                        ),
                        "wall_s": elapsed,
                        "thread_cpu_s": cpu_elapsed,
                        "used_fallback": failed_merge,
                    },
                    sort_keys=True,
                ),
            )
        if len(results) != len(jobs):
            raise RuntimeError("prefix pipeline write lost a batch result")
        return results

    def _collect(self) -> list[_Job]:
        """Merge compatible prepared rows without waiting for an inventory end marker."""
        first = self._prepared.popleft()
        assert first.request is not None
        group = [first]
        rows, tokens, size = (
            len(first.request.records),
            first.request.tokens,
            first.request.payload_bytes,
        )
        while self._prepared:
            candidate = self._prepared[0]
            request = candidate.request
            assert request is not None
            if (
                request.checkpoint_id != first.request.checkpoint_id
                or rows + len(request.records) > self._max_rows
                or tokens + request.tokens > self._max_tokens
                or size + request.payload_bytes > self._max_batch
            ):
                break
            group.append(self._prepared.popleft())
            rows += len(request.records)
            tokens += request.tokens
            size += request.payload_bytes
        self._prepared_since = time.monotonic() if self._prepared else None
        return group

    def _completed(self, flight: _Flight) -> None:
        try:
            results = flight.future.result()
        except BaseException as error:
            for job in flight.jobs:
                self._finish(job, error=error)
            return
        for job, keys in zip(flight.jobs, results, strict=True):
            assert job.iterator is not None
            if isinstance(keys, BaseException):
                self._finish(job, error=keys)
                continue
            try:
                marker = job.context.run(job.iterator.send, keys)
                if not isinstance(marker, PrefixCutBatchReleased):
                    raise TypeError(
                        "prefix generator must release a completed batch before preparing again"
                    )
            except StopIteration as finished:
                self._finish(job, result=finished.value)
                continue
            except BaseException as error:
                self._finish(job, error=error)
                continue
            job.request = None
            self._reserve(job, 0)
            self._ready.append(job)

    def _notify(self, future: Future[Any]) -> None:
        with self._condition:
            self._condition.notify_all()

    def _execute_write(
        self, context: contextvars.Context, jobs: list[_Job], batch_id: int
    ) -> list[list[str | None] | BaseException]:
        return context.run(self._write, jobs, batch_id)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._incoming:
                        self._ready.append(self._incoming.popleft())
                    closed = self._closed
                self._report_latency()
                finished = [flight for flight in self._flights if flight.future.done()]
                if finished:
                    completion_started, completion_cpu = (
                        time.monotonic(),
                        time.thread_time(),
                    )
                    for flight in finished:
                        self._flights.remove(flight)
                        self._completed(flight)
                    self._activity(
                        "completion_processing",
                        time.monotonic() - completion_started,
                        time.thread_time() - completion_cpu,
                    )
                    continue
                if closed:
                    while self._ready:
                        self._finish(
                            self._ready.popleft(),
                            error=RuntimeError("prefix pipeline closed"),
                        )
                    while self._prepared:
                        self._finish(
                            self._prepared.popleft(),
                            error=RuntimeError("prefix pipeline closed before write"),
                        )
                    self._prepared_since = None
                    if not self._flights:
                        break
                can_prepare = (
                    bool(self._ready)
                    and self._buffered + self._max_batch <= self._max_buffered
                )
                old_enough = (
                    self._prepared_since is not None
                    and time.monotonic() - self._prepared_since >= self._wait
                )
                needs_space = bool(self._ready) and not can_prepare
                batch_full = (
                    sum(
                        cast(PrefixCutWrite, job.request).payload_bytes
                        for job in self._prepared
                    )
                    >= self._max_batch
                    or sum(
                        len(cast(PrefixCutWrite, job.request).records)
                        for job in self._prepared
                    )
                    >= self._max_rows
                    or sum(
                        cast(PrefixCutWrite, job.request).tokens
                        for job in self._prepared
                    )
                    >= self._max_tokens
                )
                if (
                    not closed
                    and self._prepared
                    and len(self._flights) < self._concurrency
                    and (needs_space or old_enough or batch_full)
                ):
                    group = self._collect()
                    self._batch_id += 1
                    context = group[0].context.copy()
                    future = self._writers.submit(
                        self._execute_write, context, group, self._batch_id
                    )
                    self._flights.append(_Flight(group, future))
                    with self._condition:
                        self._stats["peak_active_writes"] = max(
                            self._stats["peak_active_writes"], len(self._flights)
                        )
                        self._stats["max_inventories_per_write"] = max(
                            self._stats["max_inventories_per_write"], len(group)
                        )
                        self._stats["max_rows_per_write"] = max(
                            self._stats["max_rows_per_write"],
                            sum(len(job.request.records) for job in group),
                        )
                    future.add_done_callback(self._notify)
                    continue
                if not closed and can_prepare:
                    self._advance(self._ready.popleft())
                    continue
                with self._condition:
                    # Recheck under the notifier lock to avoid missing a submit or
                    # native completion between the predicate and wait.
                    if self._incoming or any(
                        flight.future.done() for flight in self._flights
                    ):
                        continue
                    if self._closed != closed:
                        continue
                    wait_reason = (
                        "buffer_capacity_wait"
                        if needs_space
                        else "writer_capacity_wait"
                        if self._prepared and len(self._flights) >= self._concurrency
                        else "coalesce_wait"
                        if self._prepared
                        else "write_completion_wait"
                    )
                    wait_started, wait_cpu = time.monotonic(), time.thread_time()
                    timeout = (
                        max(0.0, self._wait - (time.monotonic() - self._prepared_since))
                        if not closed
                        and self._prepared_since is not None
                        and len(self._flights) < self._concurrency
                        else None
                    )
                    if self._latency_config is not None:
                        timeout = (
                            min(
                                float(timeout), self._latency_config.progress_interval_s
                            )
                            if timeout is not None
                            else self._latency_config.progress_interval_s
                        )
                    self._condition.wait(timeout)
                    self._activity(
                        wait_reason,
                        time.monotonic() - wait_started,
                        time.thread_time() - wait_cpu,
                    )
                    if self._ready:
                        self._stats["buffer_wait_s"] += time.monotonic() - wait_started
        except BaseException as error:
            LOGGER.exception("generation cut pipeline scheduler failed")
            self._failure = error
            # Native writes retain ownership even after an internal scheduler
            # failure. Drain and discard their unadvertised rows before rollback.
            for flight in self._flights:
                try:
                    results = flight.future.result()
                    keys = [
                        key
                        for result in results
                        if not isinstance(result, BaseException)
                        for key in result
                        if key is not None
                    ]
                    if keys:
                        self._sink.clear(keys)
                except BaseException:
                    LOGGER.exception(
                        "Failed draining prefix pipeline after scheduler failure"
                    )
            for job in tuple(self._jobs):
                self._finish(job, error=error)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            self._writers.shutdown(wait=True)

    def close(self, *, wait: bool) -> None:
        """Reject new work; only the owner thread rolls back frozen transactions.

        Active native writes finish before their transaction ownership is released.
        ``wait=False`` starts that cleanup without blocking actor shutdown.
        """
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if wait:
            self._thread.join()
