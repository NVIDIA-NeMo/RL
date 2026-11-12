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

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class PrefixWritePipelineConfig(BaseModel, extra="forbid"):
    """Opt-in text/router encoder and byte-bounded preparation/write overlap."""

    enabled: bool = False
    direct_column_encoding: bool = False
    max_pending_inventories: Annotated[int, Field(gt=0, strict=True)] = 128
    max_buffered_payload_bytes: Annotated[int, Field(gt=0, strict=True)] = 536870912
    max_batch_payload_bytes: Annotated[int, Field(gt=0, strict=True)] = 134217728
    max_batch_rows: Annotated[int, Field(gt=0, strict=True)] = 256
    max_batch_tokens: Annotated[int, Field(gt=0, strict=True)] = 4194304
    coalesce_wait_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.01


@dataclass(frozen=True)
class PrefixWrite:
    """Immutable rows; only their owner thread may resume their transactions."""

    checkpoint_id: str
    records: tuple[Any, ...]
    sequences: tuple[int, ...]
    attachments: tuple[Any, ...]
    payload_bytes: int
    fallback: Callable[[], list[str | None]]

    @property
    def tokens(self) -> int:
        return sum(record.delta_len for record in self.records)


@dataclass(frozen=True)
class BatchReleased:
    """All transactions of the preceding write have sealed or rolled back."""


CutSteps = Generator[PrefixWrite | BatchReleased, list[str | None], Any]


@dataclass(eq=False)
class _Job:
    inventory: Any
    deadline: float | None
    future: Future[Any]
    iterator: CutSteps | None = None
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
        config: PrefixWritePipelineConfig,
        sink: Any,
        prepare: Callable[..., CutSteps],
    ) -> None:
        if config.max_buffered_payload_bytes < 2 * config.max_batch_payload_bytes:
            raise ValueError(
                "Prefix pipeline buffer must hold a write and one preparation reservation"
            )
        self._config = config
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
        self._stats: dict[str, int | float] = {
            "write_batches": 0,
            "merged_inventories": 0,
            "max_rows_per_write": 0,
            "prepare_wall_s": 0.0,
            "prepare_thread_cpu_s": 0.0,
            "write_wall_s": 0.0,
            "peak_buffered_payload_bytes": 0,
            "completed_inventories": 0,
            "failed_inventories": 0,
        }
        self._writers = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-prefix-write"
        )
        self._thread = threading.Thread(
            target=self._run, name="nrl-prefix-prepare", daemon=True
        )
        self._thread.start()

    def submit(self, inventory: Any, *, deadline: float | None) -> Future[Any]:
        key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._condition:
            if self._closed:
                raise RuntimeError("Prefix write pipeline is closed")
            if key in self._jobs:
                return self._jobs[key].future
            if len(self._jobs) >= self._config.max_pending_inventories:
                raise RuntimeError("Prefix pipeline inventory admission limit reached")
            future: Future[Any] = Future()
            future.set_running_or_notify_cancel()
            job = _Job(inventory, deadline, future)
            self._jobs[key] = job
            self._incoming.append(job)
            self._condition.notify_all()
            return job.future

    def stats(self) -> dict[str, int | float]:
        with self._condition:
            return dict(
                self._stats,
                pending_inventories=len(self._jobs),
                buffered_payload_bytes=self._buffered,
            )

    def _reserve(self, job: _Job, amount: int) -> None:
        with self._condition:
            self._buffered += amount - job.reservation
            job.reservation = amount
            if not 0 <= self._buffered <= self._config.max_buffered_payload_bytes:
                raise RuntimeError("Prefix pipeline exceeded its byte reservation")
            self._stats["peak_buffered_payload_bytes"] = max(
                self._stats["peak_buffered_payload_bytes"], self._buffered
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
            self._stats[
                "completed_inventories" if error is None else "failed_inventories"
            ] += 1
            self._condition.notify_all()
        if error is None:
            job.future.set_result(result)
        else:
            job.future.set_exception(error)

    def _advance(self, job: _Job) -> None:
        started, cpu = time.perf_counter(), time.thread_time()
        self._reserve(job, self._config.max_batch_payload_bytes)
        try:
            if job.iterator is None:
                job.iterator = self._prepare(job.inventory, deadline=job.deadline)
                self._busy.update(
                    prefix.model_call_id for prefix in job.inventory.active_prefixes
                )
            request = next(job.iterator)
            if not isinstance(request, PrefixWrite):
                raise TypeError("Prefix preparation must yield a write request")
            if request.checkpoint_id != job.inventory.checkpoint_id:
                raise ValueError("Prefix write changed checkpoint identity")
            if not 0 < request.payload_bytes <= self._config.max_batch_payload_bytes:
                raise ValueError("Prefix write exceeds its byte reservation")
            if (
                not len(request.records)
                == len(request.sequences)
                == len(request.attachments)
            ):
                raise ValueError("Prefix batch lengths differ")
            job.request = request
            self._reserve(job, request.payload_bytes)
            self._prepared.append(job)
        except StopIteration as completed:
            self._finish(job, result=completed.value)
        except BaseException as error:
            self._finish(job, error=error)
        finally:
            with self._condition:
                self._stats["prepare_wall_s"] += time.perf_counter() - started
                self._stats["prepare_thread_cpu_s"] += time.thread_time() - cpu

    def _write(self, jobs: list[_Job]) -> list[list[str | None] | BaseException]:
        requests = [job.request for job in jobs]
        assert all(request is not None for request in requests)
        records = [record for request in requests for record in request.records]
        started = time.perf_counter()
        try:
            if any(
                job.deadline is not None and time.monotonic() >= job.deadline
                for job in jobs
            ):
                raise TimeoutError("Prefix batch expired before PUT")
            keys = self._sink.stage_generation_prefix_batch(
                records,
                checkpoint_id=requests[0].checkpoint_id,
                chunk_sequences=[
                    seq for request in requests for seq in request.sequences
                ],
                attachments=[
                    item for request in requests for item in request.attachments
                ],
            )
            if len(keys) != len(records):
                self._sink.clear(keys)
                raise ValueError("Prefix sink returned the wrong number of keys")
        except Exception:
            results = []
            for request in requests:
                try:
                    results.append(request.fallback())
                except BaseException as error:
                    results.append(error)
        else:
            results, offset = [], 0
            for request in requests:
                stop = offset + len(request.records)
                results.append(keys[offset:stop])
                offset = stop
        finally:
            with self._condition:
                self._stats["write_batches"] += 1
                self._stats["merged_inventories"] += len(jobs)
                self._stats["max_rows_per_write"] = max(
                    self._stats["max_rows_per_write"], len(records)
                )
                self._stats["write_wall_s"] += time.perf_counter() - started
        return results

    def _collect(self) -> list[_Job]:
        first = self._prepared.popleft()
        group = [first]
        assert first.request is not None
        rows, tokens, size = (
            len(first.request.records),
            first.request.tokens,
            first.request.payload_bytes,
        )
        while self._prepared:
            request = self._prepared[0].request
            assert request is not None
            if (
                request.checkpoint_id != first.request.checkpoint_id
                or rows + len(request.records) > self._config.max_batch_rows
                or tokens + request.tokens > self._config.max_batch_tokens
                or size + request.payload_bytes > self._config.max_batch_payload_bytes
            ):
                break
            group.append(self._prepared.popleft())
            rows += len(request.records)
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
        for job, keys in zip(jobs, results, strict=True):
            if isinstance(keys, BaseException):
                self._finish(job, error=keys)
                continue
            try:
                assert job.iterator is not None
                marker = job.iterator.send(keys)
                if not isinstance(marker, BatchReleased):
                    raise TypeError("Prefix owner must release a completed batch")
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
                can_prepare = (
                    candidate is not None
                    and self._buffered + self._config.max_batch_payload_bytes
                    <= self._config.max_buffered_payload_bytes
                )
                should_write = self._prepared and (
                    not can_prepare
                    or prepared_at is None
                    or time.monotonic() - prepared_at
                    >= self._config.coalesce_wait_seconds
                    or sum(len(job.request.records) for job in self._prepared)
                    >= self._config.max_batch_rows
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
                            self._config.coalesce_wait_seconds
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
                        key
                        for result in results
                        if not isinstance(result, BaseException)
                        for key in result
                        if key is not None
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
        LOGGER.info("prefix_pipeline event=shutdown_requested metrics=%s", self.stats())
