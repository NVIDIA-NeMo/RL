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

"""Admission for bounded, independent generation-cut inventories.

Bookkeeping belongs to the HTTP event loop, not the executor. Waiting for an
overlapping inventory must never occupy a thread needed to finish its owner.
"""

from __future__ import annotations

import asyncio
import functools
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nemo_rl.utils.prefix_cut_diagnostics import (
    PrefixCutProbe,
    current_prefix_cut_probe,
    prefix_cut_count,
    prefix_cut_wait,
)

_ADMISSION_SAMPLE_SECONDS = 5.0


def generation_cut_deadline(
    timeout_s: float | None, caller_deadline_ts: str | None
) -> float | None:
    """Bound a cut by both its worker timeout and the caller's Unix deadline.

    The caller deadline stays fixed across HTTP retries and transit delays.
    Convert it to the worker's monotonic clock before admission.
    """
    now = time.monotonic()
    deadline = None if timeout_s is None else now + timeout_s
    if caller_deadline_ts is not None:
        try:
            caller_deadline = float(caller_deadline_ts)
        except ValueError as error:
            raise ValueError(
                "generation-cut deadline must be a finite Unix timestamp"
            ) from error
        if not math.isfinite(caller_deadline):
            raise ValueError("generation-cut deadline must be a finite Unix timestamp")
        caller_bound = now + (caller_deadline - time.time())
        deadline = caller_bound if deadline is None else min(deadline, caller_bound)
    if deadline is not None and deadline <= time.monotonic():
        raise TimeoutError("generation-cut caller deadline has expired")
    return deadline


async def run_generation_cut_request(
    operation: Callable[[], Awaitable[Any]],
    receive: Callable[[], Awaitable[dict[str, Any]]],
) -> Any:
    """Cancel a cut waiter when its already-parsed HTTP request disconnects.

    GenerationCutControl shields active executor work. Cancelling this waiter
    removes queued work, but never releases a running thread's reservation.
    All watcher tasks are also cleaned up if the HTTP handler is cancelled.
    """

    async def disconnected() -> None:
        while (await receive())["type"] != "http.disconnect":
            pass

    work = asyncio.create_task(operation())
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            (work, watcher), return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        watcher.result()
        prefix_cut_count("control_http_disconnected", 1)
        # Match vLLM's HTTP-disconnect behavior: finish the route normally
        # after cancelling its waiter, so middleware sees no missing response.
        return None
    finally:
        for task in (work, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


@dataclass(eq=False)
class _Inventory:
    key: tuple[str, str]
    call_ids: frozenset[str]
    task: asyncio.Task[Any] | None
    active: bool
    waiters: int
    probe: PrefixCutProbe | None
    queued_at: float


class GenerationCutControl:
    """Serialize overlapping calls and checkpoint epochs before executor entry.

    Independent inventories of the same checkpoint may run concurrently.
    A queued different checkpoint forms a barrier: later arrivals cannot
    overtake it. Caller cancellation never releases a running operation's
    reservations; its thread must actually finish first.

    ``run`` is confined to one event loop. ``close`` may be called from the
    actor's shutdown thread and refuses new work without cancelling active cuts.
    """

    def __init__(self, *, max_workers: int) -> None:
        if type(max_workers) is not int or not 1 <= max_workers <= 4:
            raise ValueError("generation-cut max_workers must be between 1 and 4")
        self._max_workers = max_workers
        self._loop: asyncio.AbstractEventLoop | None = None
        self._changed = asyncio.Event()
        self._jobs: dict[tuple[str, str], _Inventory] = {}
        self._pending: list[_Inventory] = []
        self._active: list[_Inventory] = []
        self._checkpoint_id: str | None = None
        self._retired_checkpoints: set[str] = set()
        self._closed = False

    async def run(
        self,
        *,
        checkpoint_id: str,
        inventory_digest: str,
        call_ids: frozenset[str],
        deadline: float | None,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Join a duplicate or admit an independent inventory before its deadline."""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("generation-cut admission belongs to another event loop")
        if self._closed:
            raise RuntimeError("generation-cut admission is closed")
        key = (checkpoint_id, inventory_digest)
        job = self._jobs.get(key)
        if job is not None and not job.active and job.task is not None:
            if job.task.cancelling():
                # The last caller cancelled before admission. A retry may
                # arrive before that task's done callback removes its entry.
                del self._jobs[key]
                job = None
        if job is None:
            probe = current_prefix_cut_probe()
            job = _Inventory(
                key,
                call_ids,
                task=None,
                active=False,
                waiters=0,
                probe=probe,
                queued_at=time.perf_counter() if probe is not None else 0.0,
            )
            self._jobs[key] = job
            self._pending.append(job)
            job.task = loop.create_task(self._execute(job, operation, deadline))
            job.task.add_done_callback(functools.partial(self._finished, job))
        else:
            if job.call_ids != call_ids:
                raise ValueError(
                    "duplicate inventory has different model-call identities"
                )
            prefix_cut_count("control_duplicate_join", 1)
        assert job.task is not None
        job.waiters += 1
        try:
            # HTTP cancellation must not cancel the task awaiting the executor:
            # cancelling that await would abandon a still-running thread.
            return await asyncio.shield(job.task)
        finally:
            job.waiters -= 1
            if job.waiters == 0 and not job.active and not job.task.done():
                prefix_cut_count("control_cancelled_before_start", 1)
                job.task.cancel()

    def _blocking_reason(self, job: _Inventory) -> str | None:
        """Report the first unsatisfied admission predicate, in admission order."""
        if len(self._active) >= self._max_workers:
            return "capacity"
        if any(active.key[0] != job.key[0] for active in self._active):
            return "active_checkpoint"
        if any(active.call_ids & job.call_ids for active in self._active):
            return "active_call_overlap"
        for earlier in self._pending:
            if earlier is job:
                return None
            if earlier.key[0] != job.key[0] or earlier.call_ids & job.call_ids:
                return "pending_barrier"
        return "not_pending"

    def _observe_queue(self) -> None:
        """Sample bounded gauges only for telemetry-enabled cut requests."""
        prefix_cut_count("admission_max_pending", len(self._pending), maximum=True)
        prefix_cut_count("admission_max_active", len(self._active), maximum=True)
        tq_ages = [
            age
            for active in self._active
            if active.probe is not None
            and (age := active.probe.active_phase_age("tq_put_ack")) is not None
        ]
        prefix_cut_count("admission_max_active_tq_puts", len(tq_ages), maximum=True)
        writer_ages = [
            age
            for active in self._active
            if active.probe is not None
            and (age := active.probe.active_phase_age("inventory_write_batch_wait"))
            is not None
        ]
        prefix_cut_count(
            "admission_max_active_write_batch_waiters", len(writer_ages), maximum=True
        )
        if writer_ages:
            prefix_cut_count(
                "admission_max_active_write_batch_wait_age_ms",
                int(1000 * max(writer_ages)),
                maximum=True,
            )
        if tq_ages:
            prefix_cut_count(
                "admission_max_active_tq_put_age_ms",
                int(1000 * max(tq_ages)),
                maximum=True,
            )

    async def _execute(
        self,
        job: _Inventory,
        operation: Callable[[], Awaitable[Any]],
        deadline: float | None,
    ) -> Any:
        admitted_at: float | None = None
        if job.probe is not None:
            prefix_cut_wait(
                "control_admission_task_dispatch", time.perf_counter() - job.queued_at
            )
        try:
            while True:
                if self._closed:
                    raise RuntimeError("generation-cut admission is closed")
                if job.key[0] in self._retired_checkpoints:
                    prefix_cut_count("control_stale_checkpoint_rejected", 1)
                    raise RuntimeError("generation-cut checkpoint has been superseded")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    prefix_cut_count("control_admission_deadline_expired", 1)
                    raise TimeoutError(
                        "generation-cut deadline expired before admission"
                    )
                reason = self._blocking_reason(job)
                if job.probe is not None:
                    self._observe_queue()
                if reason is None:
                    if self._checkpoint_id != job.key[0]:
                        if self._checkpoint_id is not None:
                            self._retired_checkpoints.add(self._checkpoint_id)
                        self._checkpoint_id = job.key[0]
                    self._pending.remove(job)
                    self._active.append(job)
                    job.active = True
                    if job.probe is not None:
                        admitted_at = time.perf_counter()
                    break
                # No await separates the predicate and clear: another operation
                # cannot release its reservation between them on this loop.
                self._changed.clear()
                # A bounded sampling tick observes active TQ writes and HTTP-loop
                # timer lag. Without telemetry retain the original deadline wait.
                wait_s = remaining
                if job.probe is not None:
                    wait_s = (
                        _ADMISSION_SAMPLE_SECONDS
                        if remaining is None
                        else min(remaining, _ADMISSION_SAMPLE_SECONDS)
                    )
                started = time.perf_counter() if job.probe is not None else 0.0
                cpu_started = time.thread_time() if job.probe is not None else 0.0
                process_started = time.process_time() if job.probe is not None else 0.0
                timed_out = False
                try:
                    async with asyncio.timeout(wait_s):
                        await self._changed.wait()
                except TimeoutError:
                    timed_out = True
                    if deadline is not None and time.monotonic() >= deadline:
                        prefix_cut_count("control_admission_deadline_expired", 1)
                        raise TimeoutError(
                            "generation-cut deadline expired before admission"
                        ) from None
                finally:
                    if job.probe is not None:
                        elapsed = time.perf_counter() - started
                        # Inclusive of HTTP-loop scheduling delays, not pure
                        # semaphore wait. Reason is sampled before each wait;
                        # CPU covers all work on this loop/thread while waiting.
                        prefix_cut_wait(
                            f"admission_blocked_{reason}",
                            elapsed,
                            thread_cpu_s=time.thread_time() - cpu_started,
                            process_cpu_s=time.process_time() - process_started,
                        )
                        if timed_out and wait_s is not None:
                            prefix_cut_wait(
                                "admission_http_loop_timer_lag",
                                max(0.0, elapsed - wait_s),
                            )
            return await operation()
        finally:
            if job in self._pending:
                self._pending.remove(job)
            if job in self._active:
                self._active.remove(job)
            if admitted_at is not None:
                prefix_cut_wait(
                    "control_admission_slot_hold", time.perf_counter() - admitted_at
                )
            self._changed.set()

    def _finished(self, job: _Inventory, task: asyncio.Task[Any]) -> None:
        # Also handles cancellation before _execute had its first turn.
        if self._jobs.get(job.key) is job:
            del self._jobs[job.key]
        if job in self._pending:
            self._pending.remove(job)
        self._changed.set()
        if not task.cancelled():
            # A disconnected caller might never retrieve a failed operation.
            task.exception()

    def close(self) -> None:
        """Refuse new admissions and cancel only operations not yet started."""
        self._closed = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(functools.partial(self._cancel_pending))

    def _cancel_pending(self) -> None:
        for job in tuple(self._pending):
            assert job.task is not None
            job.task.cancel()
        self._changed.set()
