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
"""Bounded prefix preparation overlapped with one durable write stream."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from nemo_rl.data_plane.generation_cut_rows import (
    GenerationCutRow,
    StagedGenerationCut,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefixWriteLimits:
    """Bounds of the write pipeline every generation-prefix cut goes through."""

    # Admission fails beyond this many pending inventories.
    max_pending_inventories: int
    # Estimated queued and in-flight payload, plus one preparation reservation.
    max_buffered_payload_bytes: int
    # One merged write; a larger single row is written alone.
    max_batch_payload_bytes: int
    # How long a prepared batch waits for compatible rows to merge with.
    coalesce_wait_seconds: float

    def __post_init__(self) -> None:
        # Room for one write in flight plus one preparation reservation.
        if self.max_buffered_payload_bytes < 2 * self.max_batch_payload_bytes:
            raise ValueError(
                "max_buffered_payload_bytes must be >= 2 * max_batch_payload_bytes"
            )


PREFIX_WRITE_LIMITS = PrefixWriteLimits(
    max_pending_inventories=128,
    max_buffered_payload_bytes=512 * 1024 * 1024,
    max_batch_payload_bytes=128 * 1024 * 1024,
    coalesce_wait_seconds=0.01,
)


@dataclass(frozen=True)
class PrefixWrite:
    """Frozen calls' rows; only their owner thread may resume their transactions."""

    checkpoint_id: str
    rows: tuple[GenerationCutRow, ...]
    payload_bytes: int
    # Called with the failed batch write's error; writes what it can row by row.
    fallback: Callable[[BaseException], list[StagedGenerationCut | None]]

    @property
    def tokens(self) -> int:
        return sum(row.delta_len for row in self.rows)


@dataclass(frozen=True)
class WriteSettled:
    """All transactions of the preceding write have sealed or rolled back."""


CutWrites = Generator[PrefixWrite | WriteSettled, list[StagedGenerationCut | None], Any]


@dataclass(eq=False)
class _Job:
    inventory: Any
    deadline: float | None
    future: Future[Any]
    held_only: bool
    iterator: CutWrites | None = None
    request: PrefixWrite | None = None
    reservation: int = 0


class PrefixWritePipeline:
    """Keep lifecycle ownership on one thread while a separate thread writes.

    Inventories sharing a model call never prepare concurrently. Duplicate
    in-flight inventory requests share their result. Byte reservations cover
    prepared/in-flight numeric columns, plus one maximum preparation batch;
    they do not bound live generation histories or total process RSS.
    """

    def __init__(
        self,
        *,
        limits: PrefixWriteLimits,
        max_batch_rows: int,
        max_batch_tokens: int,
        sink: Any,
        prepare: Callable[..., CutWrites],
    ) -> None:
        self._limits = limits
        self._max_batch_rows = max_batch_rows
        self._max_batch_tokens = max_batch_tokens
        self._sink = sink
        self._prepare = prepare
        self._condition = threading.Condition()
        self._incoming: deque[_Job] = deque()
        self._ready: deque[_Job] = deque()
        self._prepared: deque[_Job] = deque()
        self._jobs: dict[tuple[str, str], _Job] = {}
        self._busy: set[str] = set()
        self._buffered = 0
        self._closed = False
        self._counts = {
            "write_batches": 0,
            "merged_inventories": 0,
            "failed_inventories": 0,
            "peak_buffered_payload_bytes": 0,
        }
        # Nanoseconds each thread spent on cuts: wall time, and the CPU time
        # that thread itself used, which excludes waiting for the GIL or a PUT.
        self._stage_ns = {
            "prepare": 0,
            "prepare_cpu": 0,
            "write": 0,
            "write_cpu": 0,
        }
        self._writers = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-prefix-write"
        )
        self._thread = threading.Thread(
            target=self._run, name="nrl-prefix-prepare", daemon=True
        )
        self._thread.start()

    def submit(
        self, inventory: Any, *, deadline: float | None, held_only: bool = False
    ) -> Future[Any]:
        key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._condition:
            if self._closed:
                raise RuntimeError("Prefix write pipeline is closed")
            if key in self._jobs:
                return self._jobs[key].future
            if len(self._jobs) >= self._limits.max_pending_inventories:
                raise RuntimeError("Prefix pipeline inventory admission limit reached")
            future: Future[Any] = Future()
            future.set_running_or_notify_cancel()
            job = _Job(inventory, deadline, future, held_only)
            self._jobs[key] = job
            self._incoming.append(job)
            self._condition.notify_all()
            return job.future

    def counts(self) -> dict[str, int]:
        """Counts since this pipeline started.

        Returns:
            ``write_batches``, the ``merged_inventories`` they carried,
            ``failed_inventories`` and ``peak_buffered_payload_bytes``. Then
            microseconds of wall and thread CPU time: ``prepare_us`` and
            ``prepare_cpu_us`` on the preparation thread (reading progress,
            freezing calls, sealing or rolling back), and ``write_us`` and
            ``write_cpu_us`` on the writer thread (building, encoding and
            writing rows; with a writer actor, its work is wall time only).
        """
        with self._condition:
            return {
                **self._counts,
                **{f"{stage}_us": ns // 1000 for stage, ns in self._stage_ns.items()},
            }

    @contextlib.contextmanager
    def _timed(self, stage: Literal["prepare", "write"]) -> Iterator[None]:
        """Add the wall and thread CPU time of the block to ``stage``'s totals."""
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            yield
        finally:
            wall = time.perf_counter_ns() - wall
            cpu = time.thread_time_ns() - cpu
            with self._condition:
                self._stage_ns[stage] += wall
                self._stage_ns[f"{stage}_cpu"] += cpu

    def _reserve(self, job: _Job, amount: int, *, allow_overflow: bool = False) -> None:
        with self._condition:
            growth = amount - job.reservation
            self._buffered += growth
            job.reservation = amount
            # Releasing never fails, even while an oversized row holds the
            # buffer above its bound.
            if self._buffered < 0 or (
                growth > 0
                and not allow_overflow
                and self._buffered > self._limits.max_buffered_payload_bytes
            ):
                raise RuntimeError("Prefix pipeline exceeded its byte reservation")
            self._counts["peak_buffered_payload_bytes"] = max(
                self._counts["peak_buffered_payload_bytes"], self._buffered
            )

    def _finish(
        self, job: _Job, *, result: Any = None, error: BaseException | None = None
    ) -> None:
        try:
            if job.iterator is not None:
                job.iterator.close()
        except BaseException as close_error:
            error = error or close_error
        self._reserve(job, 0)
        if job.iterator is not None:
            self._busy.difference_update(
                prefix.model_call_id for prefix in job.inventory.active_prefixes
            )
        with self._condition:
            self._jobs.pop(
                (job.inventory.checkpoint_id, job.inventory.inventory_digest)
            )
            if error is not None:
                self._counts["failed_inventories"] += 1
            self._condition.notify_all()
        if error is None:
            job.future.set_result(result)
        else:
            job.future.set_exception(error)

    def _advance(self, job: _Job) -> None:
        self._reserve(job, self._limits.max_batch_payload_bytes)
        try:
            if job.iterator is None:
                job.iterator = self._prepare(
                    job.inventory, deadline=job.deadline, held_only=job.held_only
                )
                self._busy.update(
                    prefix.model_call_id for prefix in job.inventory.active_prefixes
                )
            request = next(job.iterator)
            if not isinstance(request, PrefixWrite):
                raise TypeError("Prefix preparation must yield a write request")
            if request.checkpoint_id != job.inventory.checkpoint_id:
                raise ValueError("Prefix write changed checkpoint identity")
            oversized = request.payload_bytes > self._limits.max_batch_payload_bytes
            if request.payload_bytes <= 0:
                raise ValueError("Prefix write must carry a positive payload")
            if oversized and len(request.rows) != 1:
                raise ValueError("Only a single row may exceed the batch byte limit")
            job.request = request
            # A row above the batch limit is written alone (_collect never adds to
            # it). Its reservation may exceed the buffer; preparation then waits
            # until it is written.
            self._reserve(job, request.payload_bytes, allow_overflow=oversized)
            self._prepared.append(job)
        except StopIteration as completed:
            self._finish(job, result=completed.value)
        except BaseException as error:
            self._finish(job, error=error)

    def _write(
        self, jobs: list[_Job]
    ) -> list[list[StagedGenerationCut | None] | BaseException]:
        requests = [job.request for job in jobs]
        assert all(request is not None for request in requests)
        rows = [row for request in requests for row in request.rows]
        with self._timed("write"):
            try:
                if any(
                    job.deadline is not None and time.monotonic() >= job.deadline
                    for job in jobs
                ):
                    raise TimeoutError("Prefix batch expired before PUT")
                staged = self._sink.stage_generation_cut_rows(
                    rows, checkpoint_id=requests[0].checkpoint_id
                )
                if len(staged) != len(rows):
                    self._sink.clear([row.staging_key for row in staged])
                    raise ValueError("Prefix sink returned the wrong number of keys")
            except Exception as error:
                results = []
                for request in requests:
                    try:
                        results.append(request.fallback(error))
                    except BaseException as error:
                        results.append(error)
            else:
                results, offset = [], 0
                for request in requests:
                    stop = offset + len(request.rows)
                    results.append(staged[offset:stop])
                    offset = stop
            finally:
                with self._condition:
                    self._counts["write_batches"] += 1
                    self._counts["merged_inventories"] += len(jobs)
        return results

    def _collect(self) -> list[_Job]:
        first = self._prepared.popleft()
        group = [first]
        assert first.request is not None
        rows, tokens, size = (
            len(first.request.rows),
            first.request.tokens,
            first.request.payload_bytes,
        )
        while self._prepared:
            request = self._prepared[0].request
            assert request is not None
            if (
                request.checkpoint_id != first.request.checkpoint_id
                or rows + len(request.rows) > self._max_batch_rows
                or tokens + request.tokens > self._max_batch_tokens
                or size + request.payload_bytes > self._limits.max_batch_payload_bytes
            ):
                break
            group.append(self._prepared.popleft())
            rows += len(request.rows)
            tokens += request.tokens
            size += request.payload_bytes
        return group

    def _complete(self, jobs: list[_Job], future: Future[Any]) -> None:
        try:
            results = future.result()
        except BaseException as error:
            for job in jobs:
                self._finish(job, error=error)
            return
        for job, staged in zip(jobs, results, strict=True):
            if isinstance(staged, BaseException):
                self._finish(job, error=staged)
                continue
            try:
                assert job.iterator is not None
                marker = job.iterator.send(staged)
                if not isinstance(marker, WriteSettled):
                    raise TypeError("Prefix owner must settle a completed write")
            except StopIteration as completed:
                self._finish(job, result=completed.value)
            except BaseException as error:
                self._finish(job, error=error)
            else:
                job.request = None
                self._reserve(job, 0)
                self._ready.append(job)

    def _notify(self, future: Future[Any]) -> None:
        with self._condition:
            self._condition.notify_all()

    def _run(self) -> None:
        flight: tuple[list[_Job], Future[Any]] | None = None
        prepared_at: float | None = None
        try:
            while True:
                with self._condition:
                    while self._incoming:
                        self._ready.append(self._incoming.popleft())
                    closed = self._closed
                if flight is not None and flight[1].done():
                    with self._timed("prepare"):
                        self._complete(*flight)
                    flight = None
                    continue
                if closed:
                    while self._ready:
                        self._finish(
                            self._ready.popleft(),
                            error=RuntimeError("Prefix pipeline closed"),
                        )
                    while self._prepared:
                        self._finish(
                            self._prepared.popleft(),
                            error=RuntimeError("Prefix pipeline closed before PUT"),
                        )
                    if flight is None:
                        break
                candidate = next(
                    (
                        job
                        for job in self._ready
                        if job.iterator is not None
                        or not self._busy.intersection(
                            prefix.model_call_id
                            for prefix in job.inventory.active_prefixes
                        )
                    ),
                    None,
                )
                prepared_rows = sum(len(job.request.rows) for job in self._prepared)
                can_prepare = (
                    candidate is not None
                    and self._buffered + self._limits.max_batch_payload_bytes
                    <= self._limits.max_buffered_payload_bytes
                    # Text rows rarely reach the byte bound. Preparing at most one
                    # write ahead of the write in flight keeps calls frozen for
                    # about two writes, not the whole queue.
                    and prepared_rows < self._max_batch_rows
                )
                should_write = self._prepared and (
                    not can_prepare
                    or prepared_at is None
                    or time.monotonic() - prepared_at
                    >= self._limits.coalesce_wait_seconds
                    or prepared_rows >= self._max_batch_rows
                )
                if not closed and flight is None and should_write:
                    jobs = self._collect()
                    future = self._writers.submit(self._write, jobs)
                    flight = (jobs, future)
                    future.add_done_callback(self._notify)
                    prepared_at = time.monotonic() if self._prepared else None
                    continue
                if not closed and can_prepare:
                    assert candidate is not None
                    self._ready.remove(candidate)
                    with self._timed("prepare"):
                        self._advance(candidate)
                    if prepared_at is None and self._prepared:
                        prepared_at = time.monotonic()
                    continue
                with self._condition:
                    if (
                        self._incoming
                        or self._closed != closed
                        or (flight is not None and flight[1].done())
                    ):
                        continue
                    timeout = (
                        max(
                            0.0,
                            self._limits.coalesce_wait_seconds
                            - (time.monotonic() - prepared_at),
                        )
                        if flight is None and prepared_at is not None
                        else None
                    )
                    self._condition.wait(timeout)
        except BaseException as error:
            LOGGER.exception("Prefix pipeline scheduler failed")
            if flight is not None:
                try:
                    results = flight[1].result()
                    keys = [
                        row.staging_key
                        for result in results
                        if not isinstance(result, BaseException)
                        for row in result
                        if row is not None
                    ]
                    if keys:
                        self._sink.clear(keys)
                except BaseException:
                    LOGGER.exception("Failed to clear unadvertised pipeline writes")
            with self._condition:
                self._closed = True
                remaining = list(self._jobs.values())
            for job in remaining:
                self._finish(job, error=error)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            self._writers.shutdown(wait=True)

    def close(self, *, wait: bool) -> None:
        """Reject new work; drain started writes before releasing transactions."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if wait:
            self._thread.join()
