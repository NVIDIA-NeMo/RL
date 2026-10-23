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
from typing import TypeVar

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass
class _Timing:
    calls: int = 0
    wall_s: float = 0.0
    thread_cpu_s: float = 0.0
    max_wall_s: float = 0.0
    errors: int = 0


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

    def record(self, name: str, wall_s: float, cpu_s: float, error: bool) -> None:
        """Accumulate one phase without keeping per-call records."""
        with self.lock:
            timing = self.timings.setdefault(name, _Timing())
            timing.calls += 1
            timing.wall_s += wall_s
            timing.thread_cpu_s += cpu_s
            timing.max_wall_s = max(timing.max_wall_s, wall_s)
            timing.errors += int(error)

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


def prefix_cut_probe_active() -> bool:
    """Whether this context belongs to a measured cut inventory."""
    return _CURRENT.get() is not None


def prefix_cut_count(name: str, value: int, *, maximum: bool = False) -> None:
    """Record a numeric size/count, never a tensor, payload or staging key."""
    probe = _CURRENT.get()
    if probe is not None:
        with probe.lock:
            previous = probe.counts.get(name, 0)
            probe.counts[name] = max(previous, value) if maximum else previous + value


def prefix_cut_wait(name: str, elapsed_s: float) -> None:
    """Record a cross-thread scheduling wait without attributing thread CPU."""
    probe = _CURRENT.get()
    if probe is not None:
        probe.record(name, max(0.0, elapsed_s), 0.0, False)


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
    if probe is None:
        yield
        return
    wall_started, cpu_started = time.perf_counter(), time.thread_time()
    error = True
    try:
        yield
        error = False
    except StopIteration:
        error = not expected_stop_iteration
        raise
    finally:
        probe.record(
            name,
            time.perf_counter() - wall_started,
            time.thread_time() - cpu_started,
            error,
        )


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
