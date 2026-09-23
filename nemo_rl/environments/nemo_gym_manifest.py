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
"""Direct ledger reads for token-capture receipts (``manifest_transport: local_file``).

The NemoGym actor and Gym's policy-model proxy (the ledger writer) share one
node, so a completed rollout's ``<capture_dir>/lineage/<rollout_id>.lineage.jsonl``
can be read straight from tmpfs instead of through the ``/manifest`` HTTP route.
That removes every manifest request from the rollout path, where a 2 ms GET
used to queue behind rollout-long ``/run`` requests on Gym's shared aiohttp
pool. Rollout HTTP traffic and its event-loop implementation are unchanged.

:class:`ManifestReceiptService` owns, per actor:

* one bounded ``asyncio.Queue`` of admitted jobs (``queue_size``);
* one ``ThreadPoolExecutor`` of ``workers`` threads, fed by exactly ``workers``
  consumer tasks so the executor's own unbounded queue never grows;
* one monotonic deadline per job covering admission, lock wait, snapshot,
  parse and receipt assembly;
* cooperative cancellation: a cancelled waiter cannot stop a thread, so the
  consumer keeps its slot until the underlying job actually returns and the
  job checks ``cancel_event`` between stages;
* bounded histograms (fixed-size reservoirs) of the read pipeline, reported
  periodically rather than per read.

The reader itself (:class:`nemo_gym.token_id_capture.lineage.FileManifestReader`)
is Gym-owned: file layout, lock convention and manifest schema stay in one
place. This module only schedules it.
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_RESERVOIR_SIZE = 4096


class _Reservoir:
    """Fixed-memory sample of a stream of floats (Vitter's algorithm R)."""

    __slots__ = ("_samples", "_seen", "_limit", "_sum", "_max")

    def __init__(self, limit: int = _RESERVOIR_SIZE) -> None:
        self._samples: list[float] = []
        self._seen = 0
        self._limit = limit
        self._sum = 0.0
        self._max = 0.0

    def add(self, value: float) -> None:
        self._seen += 1
        self._sum += value
        if value > self._max:
            self._max = value
        if len(self._samples) < self._limit:
            self._samples.append(value)
        else:
            slot = random.randrange(self._seen)
            if slot < self._limit:
                self._samples[slot] = value

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {}
        ordered = sorted(self._samples)

        def pct(p: float) -> float:
            index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
            return ordered[index]

        return {
            "count": float(self._seen),
            "mean": self._sum / self._seen,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": self._max,
        }

    def reset(self) -> None:
        self._samples.clear()
        self._seen = 0
        self._sum = 0.0
        self._max = 0.0


@dataclass(frozen=True)
class ManifestReadJob:
    """Inputs a receipt needs. Treated as immutable while the job runs."""

    rollout_id: str
    terminal_response_id: Optional[str]
    scored_response: Optional[dict]
    reward: float


@dataclass
class ManifestReadResult:
    rollout_id: str
    receipt: Optional[dict]
    error: Optional[str] = None
    error_kind: Optional[str] = (
        None  # missing_root | corrupt | timeout | cancelled | setup | error
    )
    total_s: float = 0.0
    admission_wait_s: float = 0.0


@dataclass(eq=False)  # identity-hashed: tickets live in a set
class _Ticket:
    job: ManifestReadJob
    future: asyncio.Future[ManifestReadResult]
    deadline: float
    submitted_at: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_at: Optional[float] = None
    deadline_handle: Optional[asyncio.TimerHandle] = None
    reported: bool = False


class ManifestReceiptService:
    """Bounded, actor-wide pipeline: ledger snapshot -> manifest -> receipt.

    Construct once per actor inside a running event loop (``start`` creates
    consumer tasks on ``asyncio.get_running_loop()``).
    """

    def __init__(
        self,
        reader: Any,
        assemble_receipt: Callable[..., dict],
        *,
        workers: int,
        queue_size: int,
        timeout_s: float,
        thread_name_prefix: str = "manifest-reader",
    ) -> None:
        if workers <= 0:
            raise ValueError("manifest_read_workers must be positive")
        if queue_size <= 0:
            raise ValueError("manifest_queue_size must be positive")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("control_timeout_s must be positive")
        self._reader = reader
        self._assemble_receipt = assemble_receipt
        self._workers = int(workers)
        self._queue_size = int(queue_size)
        self._timeout_s = float(timeout_s)
        self._thread_name_prefix = thread_name_prefix
        self._queue: Optional[asyncio.Queue[_Ticket]] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._consumers: list[asyncio.Task[None]] = []
        self._tickets: set[_Ticket] = set()
        self._running: set[_Ticket] = set()
        self._closed = False
        self._started = False
        # Metrics (event-loop thread only, except the thread-side stats which
        # are attached to the ticket and folded in on the loop).
        self._admission_wait = _Reservoir()
        self._lock_wait = _Reservoir()
        self._snapshot = _Reservoir()
        self._parse = _Reservoir()
        self._assemble = _Reservoir()
        self._total = _Reservoir()
        self._bytes = _Reservoir()
        self._rows = _Reservoir()
        self._queue_high_water = 0
        self._active_high_water = 0
        self._counts: dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "receipt_ok": 0,
            "receipt_none": 0,
            "timeout": 0,
            "cancelled": 0,
            "corrupt": 0,
            "missing_root": 0,
            "error": 0,
            "lock_retries": 0,
        }

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def queue_size(self) -> int:
        return self._queue_size

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self) -> None:
        """Create the queue, executor and consumer tasks on the running loop."""
        if self._closed:
            raise RuntimeError("ManifestReceiptService is closed")
        if self._started:
            return
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._queue_size)
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix=self._thread_name_prefix
        )
        self._consumers = [
            loop.create_task(self._consume(index), name=f"manifest-consumer-{index}")
            for index in range(self._workers)
        ]
        self._started = True

    def close_nowait(self) -> None:
        """Resolve all waiters and stop consumers without waiting for OS threads.

        Running threads observe cancellation between stages. Queue shutdown also
        wakes submitters blocked on admission, including those not yet enqueued.
        """
        if self._closed:
            return
        self._closed = True
        for ticket in list(self._tickets):
            self._fail(ticket, "cancelled", "manifest service shut down")
        for ticket in self._running:
            ticket.cancel_event.set()
        if self._queue is not None:
            self._queue.shutdown(immediate=True)
        for consumer in self._consumers:
            consumer.cancel()
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def close(self) -> None:
        """Stop the service and await consumer cleanup; never block on a thread."""
        self.close_nowait()
        if self._consumers:
            await asyncio.gather(*self._consumers, return_exceptions=True)
            self._consumers.clear()

    async def submit(self, job: ManifestReadJob) -> asyncio.Future[ManifestReadResult]:
        """Admit a read and return its deadline-bounded result future.

        One absolute budget covers admission and result delivery. Expiry resolves
        the future even if a worker is still inside a blocking filesystem call.
        The consumer remains occupied until the actual thread exits.
        """
        queue = self._queue
        if self._closed or queue is None:
            raise RuntimeError("ManifestReceiptService is not running")
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        ticket = _Ticket(
            job=job,
            future=loop.create_future(),
            deadline=now + self._timeout_s,
            submitted_at=now,
        )
        self._tickets.add(ticket)
        self._counts["submitted"] += 1
        ticket.future.add_done_callback(
            lambda future: self._on_waiter_done(ticket, future)
        )
        # Use loop.time() for scheduling; the synchronous Gym reader uses monotonic().
        loop_deadline = loop.time() + max(0.0, ticket.deadline - time.monotonic())
        ticket.deadline_handle = loop.call_at(loop_deadline, self._expire, ticket)
        try:
            async with asyncio.timeout_at(loop_deadline):
                await queue.put(ticket)
        except TimeoutError:
            self._expire(ticket)
        except asyncio.QueueShutDown:
            self._fail(
                ticket, "cancelled", "manifest service shut down during admission"
            )
        except asyncio.CancelledError:
            ticket.future.cancel()
            self._fail(ticket, "cancelled", "manifest admission cancelled")
            raise
        else:
            self._queue_high_water = max(self._queue_high_water, queue.qsize())
        return ticket.future

    async def receipt(self, job: ManifestReadJob) -> ManifestReadResult:
        """Admit and await a read using the same deadline as streaming submit()."""
        future = await self.submit(job)
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _on_waiter_done(
        self, ticket: _Ticket, future: asyncio.Future[ManifestReadResult]
    ) -> None:
        if future.cancelled():
            self._fail(ticket, "cancelled", "manifest waiter cancelled")

    def _expire(self, ticket: _Ticket) -> None:
        self._fail(
            ticket,
            "timeout",
            f"manifest read for {ticket.job.rollout_id} exceeded its {self._timeout_s}s deadline",
        )

    async def _consume(self, index: int) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            try:
                ticket = await queue.get()
            except asyncio.QueueShutDown:
                return
            try:
                if ticket.future.cancelled():
                    self._fail(ticket, "cancelled", "manifest waiter cancelled")
                if ticket.reported:
                    continue
                if time.monotonic() >= ticket.deadline:
                    self._expire(ticket)
                    continue
                executor = self._executor
                if executor is None:
                    self._fail(ticket, "cancelled", "manifest executor is closed")
                    continue
                ticket.started_at = time.monotonic()
                self._running.add(ticket)
                self._active_high_water = max(
                    self._active_high_water, len(self._running)
                )
                try:
                    outcome = await asyncio.shield(
                        asyncio.get_running_loop().run_in_executor(
                            executor, self._run_job, ticket
                        )
                    )
                except asyncio.CancelledError:
                    self._fail(ticket, "cancelled", "manifest consumer stopped")
                    raise
                except Exception as error:
                    # This is the per-job isolation boundary: a bad ledger or
                    # reader must resolve its waiter and never kill a consumer.
                    self._fail(ticket, "error", f"manifest worker failed: {error!r}")
                else:
                    self._finish(ticket, outcome)
                finally:
                    self._running.discard(ticket)
            finally:
                queue.task_done()

    def _run_job(
        self, ticket: _Ticket
    ) -> tuple[Optional[dict], Optional[str], Optional[str], Any, float]:
        """Thread body: snapshot -> manifest -> receipt. Returns (receipt, error, kind, stats, assemble_s)."""
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.lineage import (
            LedgerRootMismatch,
            ManifestReadCancelled,
            ManifestReadStats,
            ManifestReadTimeout,
        )

        job = ticket.job
        stats = ManifestReadStats()
        try:
            manifest = self._reader.read_manifest(
                job.rollout_id,
                deadline=ticket.deadline,
                cancel_event=ticket.cancel_event,
                stats=stats,
            )
        except ManifestReadTimeout as error:
            return None, str(error), "timeout", stats, 0.0
        except ManifestReadCancelled as error:
            return None, str(error), "cancelled", stats, 0.0
        except LedgerRootMismatch as error:
            return None, str(error), "missing_root", stats, 0.0
        except ValueError as error:
            # Malformed JSON / non-object row / invalid rollout id: deterministic,
            # never retried.
            return None, str(error), "corrupt", stats, 0.0
        except OSError as error:
            return None, f"ledger read failed: {error}", "error", stats, 0.0
        if ticket.cancel_event.is_set():
            return (
                None,
                f"manifest read for {job.rollout_id} cancelled before receipt assembly",
                "cancelled",
                stats,
                0.0,
            )
        if time.monotonic() >= ticket.deadline:
            return (
                None,
                f"manifest read for {job.rollout_id} exceeded its deadline before receipt assembly",
                "timeout",
                stats,
                0.0,
            )
        assemble_started = time.perf_counter()
        try:
            receipt = self._assemble_receipt(
                job.rollout_id,
                manifest,
                terminal_response_id=job.terminal_response_id,
                scored_response=job.scored_response,
                reward=job.reward,
            )
        except (
            Exception
        ) as error:  # receipt assembly is pure; surface, don't crash the worker
            return (
                None,
                f"receipt assembly failed: {error!r}",
                "error",
                stats,
                time.perf_counter() - assemble_started,
            )
        return receipt, None, None, stats, time.perf_counter() - assemble_started

    def _finish(
        self,
        ticket: _Ticket,
        outcome: tuple[Optional[dict], Optional[str], Optional[str], Any, float],
    ) -> None:
        if ticket.reported:
            return
        if ticket.future.cancelled():
            self._fail(ticket, "cancelled", "manifest waiter cancelled")
            return
        if time.monotonic() >= ticket.deadline:
            self._expire(ticket)
            return
        receipt, error, kind, stats, assemble_s = outcome
        self._lock_wait.add(stats.lock_wait_s)
        self._snapshot.add(stats.snapshot_s)
        self._parse.add(stats.parse_s)
        self._bytes.add(float(stats.bytes_read))
        self._rows.add(float(stats.rows))
        self._counts["lock_retries"] += int(stats.lock_retries)
        self._assemble.add(assemble_s)
        self._resolve(ticket, receipt, error, kind)

    def _resolve(
        self,
        ticket: _Ticket,
        receipt: Optional[dict],
        error: Optional[str],
        kind: Optional[str],
    ) -> None:
        if ticket.reported:
            return
        ticket.reported = True
        self._tickets.discard(ticket)
        if ticket.deadline_handle is not None:
            ticket.deadline_handle.cancel()
        now = time.monotonic()
        total = now - ticket.submitted_at
        admission_wait = (ticket.started_at or now) - ticket.submitted_at
        self._counts["completed"] += 1
        self._counts["receipt_ok" if error is None else "receipt_none"] += 1
        if error is not None:
            self._counts[kind or "error"] += 1
        self._total.add(total)
        self._admission_wait.add(admission_wait)
        if not ticket.future.done():
            ticket.future.set_result(
                ManifestReadResult(
                    rollout_id=ticket.job.rollout_id,
                    receipt=receipt,
                    error=error,
                    error_kind=kind,
                    total_s=total,
                    admission_wait_s=admission_wait,
                )
            )

    def _fail(self, ticket: _Ticket, kind: str, message: str) -> None:
        ticket.cancel_event.set()
        self._resolve(ticket, None, message, kind)

    # ── metrics ───────────────────────────────────────────────────────────

    def queue_depth(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    def active_jobs(self) -> int:
        return len(self._running)

    def metrics_snapshot(self, *, reset: bool = True) -> dict[str, float]:
        """Flat metrics for the batch since the last snapshot (seconds unless named)."""
        out: dict[str, float] = {}
        for name, reservoir in (
            ("admission_wait_s", self._admission_wait),
            ("lock_wait_s", self._lock_wait),
            ("snapshot_s", self._snapshot),
            ("parse_s", self._parse),
            ("assemble_s", self._assemble),
            ("total_s", self._total),
            ("bytes", self._bytes),
            ("rows", self._rows),
        ):
            for stat, value in reservoir.summary().items():
                if stat == "count" and name != "total_s":
                    continue
                out[f"{name}/{stat}"] = value
        out["queue_high_water"] = float(self._queue_high_water)
        out["active_high_water"] = float(self._active_high_water)
        out["queue_capacity"] = float(self._queue_size)
        out["workers"] = float(self._workers)
        for key, value in self._counts.items():
            out[f"count/{key}"] = float(value)
        if reset:
            for reservoir in (
                self._admission_wait,
                self._lock_wait,
                self._snapshot,
                self._parse,
                self._assemble,
                self._total,
                self._bytes,
                self._rows,
            ):
                reservoir.reset()
            self._queue_high_water = 0
            self._active_high_water = 0
            for key in self._counts:
                self._counts[key] = 0
        return out


def format_metrics_line(prefix: str, metrics: dict[str, float]) -> str:
    """One compact log line per batch (never per read)."""
    count = int(metrics.get("count/completed", 0))
    ok = int(metrics.get("count/receipt_ok", 0))
    timeouts = int(metrics.get("count/timeout", 0))
    corrupt = int(metrics.get("count/corrupt", 0))
    errors = int(metrics.get("count/error", 0)) + int(
        metrics.get("count/missing_root", 0)
    )
    cancelled = int(metrics.get("count/cancelled", 0))

    def ms(key: str) -> str:
        value = metrics.get(key)
        return "-" if value is None else f"{value * 1000:.1f}"

    return (
        f"{prefix} manifest_local_file: reads={count} ok={ok} timeout={timeouts} "
        f"corrupt={corrupt} error={errors} cancelled={cancelled} | "
        f"total_ms p50/p95/p99/max={ms('total_s/p50')}/{ms('total_s/p95')}/{ms('total_s/p99')}/{ms('total_s/max')} | "
        f"admission_ms p50/p95/max={ms('admission_wait_s/p50')}/{ms('admission_wait_s/p95')}/{ms('admission_wait_s/max')} | "
        f"lock_ms p95={ms('lock_wait_s/p95')} snapshot_ms p95={ms('snapshot_s/p95')} "
        f"parse_ms p95={ms('parse_s/p95')} assemble_ms p95={ms('assemble_s/p95')} | "
        f"bytes p50/max={metrics.get('bytes/p50', 0):.0f}/{metrics.get('bytes/max', 0):.0f} "
        f"rows p50/max={metrics.get('rows/p50', 0):.0f}/{metrics.get('rows/max', 0):.0f} | "
        f"queue_hw={int(metrics.get('queue_high_water', 0))}/{int(metrics.get('queue_capacity', 0))} "
        f"active_hw={int(metrics.get('active_high_water', 0))}/{int(metrics.get('workers', 0))} "
        f"lock_retries={int(metrics.get('count/lock_retries', 0))}"
    )
