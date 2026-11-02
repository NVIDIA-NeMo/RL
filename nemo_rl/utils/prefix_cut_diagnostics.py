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

"""Bounded, payload-free timing summaries for generation-prefix cut inventories.

Enabled by the caller's existing telemetry switch. Context propagation into the
cut executor lets the sink contribute to the same summary without changing its
protocol. Timings are inclusive: nested phases must not be added together.
Thread CPU measures the calling thread, not Torch's auxiliary CPU threads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal, TypeVar

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass
class _Timing:
    calls: int = 0
    wall_s: float = 0.0
    thread_cpu_s: float = 0.0
    max_wall_s: float = 0.0
    errors: int = 0
    process_cpu_s: float = 0.0


@dataclass
class PrefixCutProbe:
    """One inventory's bounded numeric observations; identities stay in logs."""

    checkpoint_id: str
    inventory_digest: str
    calls: int
    deadline: float | None
    started: float = field(default_factory=time.monotonic)
    started_unix_s: float = field(default_factory=time.time)
    timings: dict[str, _Timing] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    active_phases: dict[tuple[int, str], float] = field(default_factory=dict)

    def record(
        self,
        name: str,
        wall_s: float,
        cpu_s: float,
        error: bool,
        *,
        process_cpu_s: float = 0.0,
    ) -> None:
        """Accumulate one phase without keeping per-call records."""
        with self.lock:
            timing = self.timings.setdefault(name, _Timing())
            timing.calls += 1
            timing.wall_s += wall_s
            timing.thread_cpu_s += cpu_s
            timing.max_wall_s = max(timing.max_wall_s, wall_s)
            timing.errors += int(error)
            timing.process_cpu_s += process_cpu_s

    def active_phase_age(self, name: str) -> float | None:
        """Inspect an executing cut without retaining tensors or call identities."""
        now = time.perf_counter()
        with self.lock:
            starts = [
                start
                for (_, phase), start in self.active_phases.items()
                if phase == name
            ]
        return None if not starts else max(0.0, now - min(starts))

    def summary(self, outcome: str) -> dict:
        """Copy a JSON-safe snapshot, including deadline and CPU attribution."""
        now = time.monotonic()
        with self.lock:
            return {
                "checkpoint_id": self.checkpoint_id,
                "inventory_digest": self.inventory_digest,
                "pid": os.getpid(),
                "calls": self.calls,
                "outcome": outcome,
                "started_unix_s": self.started_unix_s,
                "finished_unix_s": time.time(),
                "elapsed_s": round(now - self.started, 6),
                "deadline_remaining_s": None
                if self.deadline is None
                else round(self.deadline - now, 6),
                "counts": dict(self.counts),
                "phases": {
                    name: {
                        "calls": timing.calls,
                        "wall_s": round(timing.wall_s, 6),
                        "thread_cpu_s": round(timing.thread_cpu_s, 6),
                        # All threads in this process; may exceed wall time.
                        # This is supporting evidence, not a GIL measurement.
                        "process_cpu_s": round(timing.process_cpu_s, 6),
                        "non_thread_cpu_wall_s": round(
                            max(0.0, timing.wall_s - timing.thread_cpu_s), 6
                        ),
                        "max_wall_s": round(timing.max_wall_s, 6),
                        "errors": timing.errors,
                    }
                    for name, timing in self.timings.items()
                },
            }


_CURRENT: ContextVar[PrefixCutProbe | None] = ContextVar(
    "prefix_cut_probe", default=None
)


@dataclass
class _LifecycleOwner:
    operation: str
    model_call_id: str
    checkpoint_id: str | None
    thread_id: int
    acquired_at: float
    active_phases: dict[tuple[int, str], float] = field(default_factory=dict)
    slowest_phase: str | None = None
    slowest_phase_s: float = 0.0
    finished_at: float | None = None
    guard: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        now = time.perf_counter()
        with self.guard:
            return {
                "operation": self.operation,
                "model_call_id": self.model_call_id,
                "checkpoint_id": self.checkpoint_id,
                "thread_id": self.thread_id,
                "held_s": (self.finished_at or now) - self.acquired_at,
                "active_phases": [
                    {"phase": phase, "thread_id": thread_id, "elapsed_s": now - start}
                    for (thread_id, phase), start in self.active_phases.items()
                ],
                "slowest_phase": self.slowest_phase,
                "slowest_phase_s": self.slowest_phase_s,
            }


_LIFECYCLE_OWNER: ContextVar[_LifecycleOwner | None] = ContextVar(
    "capture_lifecycle_owner", default=None
)
_LIFECYCLE_SLOW_SECONDS = 1.0
_LIFECYCLE_LOG_INTERVAL_SECONDS = 10.0
_LIFECYCLE_LAST_LOG: dict[tuple[str, str], float] = {}
_LIFECYCLE_LOG_GUARD = threading.Lock()


def _log_lifecycle(event: str, operation: str, values: dict) -> None:
    """Rate-limit payload-free slow samples; diagnostics never fail a call."""
    try:
        now = time.monotonic()
        with _LIFECYCLE_LOG_GUARD:
            key = (event, operation)
            if (
                now - _LIFECYCLE_LAST_LOG.get(key, float("-inf"))
                < _LIFECYCLE_LOG_INTERVAL_SECONDS
            ):
                return
            _LIFECYCLE_LAST_LOG[key] = now
        logger.info(
            "capture_lifecycle_lock %s",
            json.dumps(
                {"event": event, "pid": os.getpid(), "unix_s": time.time(), **values},
                sort_keys=True,
            ),
        )
    except Exception:
        pass


class CaptureLifecycleLock:
    """Keep native Lock semantics and observe labelled lifecycle operations.

    Direct acquire/release remains available, including cross-thread release.
    Only telemetry-enabled hold() scopes collect owner and phase information.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self._lock = threading.Lock()
        self._guard = threading.Lock()
        self._owner: _LifecycleOwner | None = None
        self.enabled = enabled

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *args: object) -> None:
        self.release()

    @contextmanager
    def hold(
        self,
        operation: Literal["cut", "terminal", "abort"],
        *,
        model_call_id: str,
        blocking: bool = True,
    ) -> Iterator[bool]:
        if not self.enabled:
            acquired = self.acquire(blocking)
            try:
                yield acquired
            finally:
                if acquired:
                    self.release()
            return
        with self._guard:
            previous_owner = self._owner
        started = time.perf_counter()
        acquired = self.acquire(blocking)
        waited = time.perf_counter() - started
        prefix_cut_wait(f"lifecycle_lock_wait_{operation}", waited)
        if waited >= _LIFECYCLE_SLOW_SECONDS:
            _log_lifecycle(
                "wait",
                operation,
                {
                    "operation": operation,
                    "model_call_id": model_call_id,
                    "wait_s": waited,
                    "acquired": acquired,
                    "owner_at_wait_start": None
                    if previous_owner is None
                    else previous_owner.snapshot(),
                },
            )
        if not acquired:
            yield False
            return
        probe = current_prefix_cut_probe()
        owner = _LifecycleOwner(
            operation,
            model_call_id,
            None if probe is None else probe.checkpoint_id,
            threading.get_ident(),
            time.perf_counter(),
        )
        with self._guard:
            self._owner = owner
        # Cut generators hold several locks across yields, so a ContextVar
        # scoped to a cut would leak into another suspended transaction.
        # Their phases already belong to the inventory's prefix-cut probe.
        token = _LIFECYCLE_OWNER.set(owner) if operation != "cut" else None
        try:
            yield True
        finally:
            finished = time.perf_counter()
            with owner.guard:
                owner.finished_at = finished
            with self._guard:
                self._owner = None
            if token is not None:
                _LIFECYCLE_OWNER.reset(token)
            self.release()
            held = finished - owner.acquired_at
            prefix_cut_wait(f"lifecycle_lock_hold_{operation}", held)
            if held >= _LIFECYCLE_SLOW_SECONDS:
                _log_lifecycle("hold", operation, owner.snapshot())


def prefix_cut_probe_active() -> bool:
    """Whether this context belongs to a measured cut inventory."""
    return _CURRENT.get() is not None


def current_prefix_cut_probe() -> PrefixCutProbe | None:
    """Return the shared probe for admission's bounded blocker observations."""
    return _CURRENT.get()


def prefix_cut_count(name: str, value: int, *, maximum: bool = False) -> None:
    """Record a numeric size/count, never a tensor, payload or staging key."""
    probe = _CURRENT.get()
    if probe is not None:
        with probe.lock:
            previous = probe.counts.get(name, 0)
            probe.counts[name] = max(previous, value) if maximum else previous + value


def prefix_cut_wait(
    name: str,
    elapsed_s: float,
    *,
    thread_cpu_s: float = 0.0,
    process_cpu_s: float = 0.0,
) -> None:
    """Record a wait; optional CPU covers the whole observed interval."""
    probe = _CURRENT.get()
    if probe is not None:
        probe.record(
            name, max(0.0, elapsed_s), thread_cpu_s, False, process_cpu_s=process_cpu_s
        )


def run_engine_snapshot(
    callback: Callable[[], _T],
    *,
    loop: asyncio.AbstractEventLoop,
    timeout_s: float | None,
) -> _T:
    """Run a route-reference lookup on its owner, splitting the thread round trip.

    Same submission, blocking wait and cancellation semantics as the unmeasured
    path. Callback timing runs on the owner thread; wakeup timing runs on the
    cut thread. Timed-out callbacks may still be completing, so incomplete
    waits are counted separately rather than reported as completed lookups.
    """
    probe = _CURRENT.get()
    submitted_at = time.perf_counter() if probe is not None else 0.0
    completed_at: float | None = None

    async def snapshot() -> _T:
        nonlocal completed_at
        if probe is None:
            return callback()
        started_at = time.perf_counter()
        probe.record(
            "engine_snapshot_dispatch_wait", started_at - submitted_at, 0.0, False
        )
        cpu_started = time.thread_time()
        error = True
        try:
            result = callback()
            error = False
            return result
        finally:
            completed_at = time.perf_counter()
            probe.record(
                "engine_snapshot_callback",
                completed_at - started_at,
                time.thread_time() - cpu_started,
                error,
            )

    future = asyncio.run_coroutine_threadsafe(snapshot(), loop)
    try:
        return future.result(timeout=timeout_s)
    except TimeoutError:
        future.cancel()
        prefix_cut_count("engine_snapshot_timeouts", 1)
        raise
    finally:
        if probe is not None:
            returned_at = time.perf_counter()
            if completed_at is None:
                probe.record(
                    "engine_snapshot_unresolved_wait",
                    returned_at - submitted_at,
                    0.0,
                    True,
                )
            else:
                probe.record(
                    "engine_snapshot_return_wakeup",
                    max(0.0, returned_at - completed_at),
                    0.0,
                    False,
                )


@contextmanager
def prefix_cut_timing(
    name: str, *, expected_stop_iteration: bool = False
) -> Iterator[None]:
    """Time one operation; do nothing outside a measured cut inventory."""
    probe = _CURRENT.get()
    owner = _LIFECYCLE_OWNER.get()
    if probe is None and owner is None:
        yield
        return
    wall_started, cpu_started = time.perf_counter(), time.thread_time()
    process_started = time.process_time()
    active_key = (threading.get_ident(), name)
    if probe is not None:
        with probe.lock:
            previous_start = probe.active_phases.get(active_key)
            probe.active_phases[active_key] = wall_started
    if owner is not None:
        with owner.guard:
            previous_owner_start = owner.active_phases.get(active_key)
            owner.active_phases[active_key] = wall_started
    error = True
    try:
        yield
        error = False
    except StopIteration:
        error = not expected_stop_iteration
        raise
    finally:
        elapsed = time.perf_counter() - wall_started
        if probe is not None:
            with probe.lock:
                if previous_start is None:
                    probe.active_phases.pop(active_key, None)
                else:
                    probe.active_phases[active_key] = previous_start
            probe.record(
                name,
                elapsed,
                time.thread_time() - cpu_started,
                error,
                process_cpu_s=time.process_time() - process_started,
            )
        if owner is not None:
            with owner.guard:
                if previous_owner_start is None:
                    owner.active_phases.pop(active_key, None)
                else:
                    owner.active_phases[active_key] = previous_owner_start
                if elapsed > owner.slowest_phase_s:
                    owner.slowest_phase, owner.slowest_phase_s = name, elapsed


@contextmanager
def prefix_write_batch_probe(
    *, enabled: bool, checkpoint_id: str, batch_id: str, inventories: int, rows: int
) -> Iterator[None]:
    """Measure a shared put separately from the inventories waiting for it."""
    if not enabled:
        yield
        return
    probe = PrefixCutProbe(checkpoint_id, batch_id, rows, None)
    token = _CURRENT.set(probe)
    outcome = "interrupted"
    try:
        yield
        outcome = "completed"
    finally:
        _CURRENT.reset(token)
        try:
            summary = probe.summary(outcome)
            summary["batch_id"] = summary.pop("inventory_digest")
            summary["inventories"] = inventories
            logger.info(
                "prefix_write_batch_probe %s", json.dumps(summary, sort_keys=True)
            )
        except Exception:
            logger.warning("Could not log shared prefix-write timing", exc_info=True)


@contextmanager
def prefix_cut_probe(
    *,
    enabled: bool,
    checkpoint_id: str,
    inventory_digest: str,
    calls: int,
    deadline: float | None,
) -> Iterator[None]:
    """Emit one inventory summary, also on failure or HTTP cancellation.

    Cancelled requests can leave executor work running; their summaries are
    explicitly labelled interrupted and may contain only completed phases.
    """
    if not enabled:
        yield
        return
    probe = PrefixCutProbe(checkpoint_id, inventory_digest, calls, deadline)
    context_token = _CURRENT.set(probe)
    outcome = "interrupted"
    try:
        yield
        outcome = "completed"
    finally:
        _CURRENT.reset(context_token)
        # Diagnostics must not replace a capture exception or prevent a receipt.
        try:
            logger.info(
                "prefix_cut_probe %s",
                json.dumps(probe.summary(outcome), sort_keys=True),
            )
        except Exception:
            logger.warning("Could not log prefix-cut timing summary", exc_info=True)
