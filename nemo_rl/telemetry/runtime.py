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

"""Best-effort diagnostics for selected asyncio thread offloads and event loops.

These helpers observe the existing executor; they do not change its capacity or
install a global asyncio patch. Names identify bounded call sites, never requests.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from nemo_rl.telemetry.metrics import record_rl_metrics, warn_once
from nemo_rl.telemetry.setup import telemetry_metrics_exporting

_T = TypeVar("_T")
_EVENT_LOOP_PROBE_INTERVAL_S = 1.0


def _emit(values: Mapping[str, float], attributes: Mapping[str, str]) -> None:
    # Deliberately best-effort: optional instrumentation must never change the
    # callable's result, exception, cancellation or cleanup behaviour.
    try:
        record_rl_metrics(values, attributes)
    except Exception:
        warn_once("runtime_metrics", "failed to record runtime diagnostics")


@dataclass
class _Submission:
    executor: str
    submitted_at: float
    state: str = "queued"
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> bool:
        started_at = time.perf_counter()
        with self.lock:
            if self.state not in {"queued", "retired"}:
                return False
            # Cancellation can retire the queued count after the executor has
            # started launching this wrapper. Count its eventual execution,
            # without decrementing the already-retired queued count again.
            if self.state == "queued":
                _emit(
                    {"executor_pending": -1.0},
                    {"rl.executor": self.executor, "rl.executor.state": "queued"},
                )
            self.state = "running"
            _emit(
                {"executor_pending": 1.0},
                {"rl.executor": self.executor, "rl.executor.state": "running"},
            )
            _emit(
                {"executor_queue_wait": (started_at - self.submitted_at) * 1000},
                {"rl.executor": self.executor},
            )
            return True

    def finish(self, *, status: str, started_at: float) -> None:
        with self.lock:
            self.state = "finished"
            self.finished_at = time.perf_counter()
            _emit(
                {"executor_execution": (self.finished_at - started_at) * 1000},
                {"rl.executor": self.executor, "rl.executor.status": status},
            )
            _emit(
                {"executor_pending": -1.0},
                {"rl.executor": self.executor, "rl.executor.state": "running"},
            )

    def retire_queued(self) -> None:
        with self.lock:
            if self.state != "queued":
                return
            self.state = "retired"
            _emit(
                {"executor_pending": -1.0},
                {"rl.executor": self.executor, "rl.executor.state": "queued"},
            )


async def run_in_thread(
    executor: str, function: Callable[..., _T], /, *args: Any, **kwargs: Any
) -> _T:
    """Observe one default-executor call without changing asyncio semantics.

    Args:
        executor: A bounded call-site label, such as ``generation_terminal``.
            It identifies work sharing an executor, not a dedicated thread pool.
        function: Synchronous function to offload.
        *args: Positional arguments forwarded to the function.
        **kwargs: Keyword arguments forwarded to the function.

    Records queue wait, execution time and queued/running counts. Successful
    calls also record completion-to-coroutine-resumption delay. Cancellation
    does not pretend to stop an already-running thread: it remains counted
    until the function exits. With metrics off, delegates to ``asyncio.to_thread``.
    """
    if not telemetry_metrics_exporting():
        return await asyncio.to_thread(function, *args, **kwargs)

    submission = _Submission(executor, 0.0)
    _emit(
        {"executor_pending": 1.0},
        {"rl.executor": executor, "rl.executor.state": "queued"},
    )
    call = functools.partial(function, *args, **kwargs)

    def run() -> _T:
        observed = submission.start()
        started_at = time.perf_counter()
        status = "error"
        try:
            result = call()
            status = "success"
        finally:
            if observed:
                submission.finish(status=status, started_at=started_at)
        return result

    context = contextvars.copy_context()

    def run_with_context() -> _T:
        return context.run(run)

    try:
        # Exclude registry creation and context preparation from queue wait.
        submission.submitted_at = time.perf_counter()
        future = asyncio.get_running_loop().run_in_executor(
            None, cast(Callable[..., _T], run_with_context)
        )
    except BaseException:
        submission.retire_queued()
        raise

    def cancelled(done: asyncio.Future[_T]) -> None:
        if done.cancelled():
            submission.retire_queued()

    future.add_done_callback(cancelled)
    result = await future
    if submission.finished_at is not None:
        _emit(
            {
                "executor_result_delivery": (
                    time.perf_counter() - submission.finished_at
                )
                * 1000
            },
            {"rl.executor": executor},
        )
    return result


async def _probe_event_loop(name: str) -> None:
    loop = asyncio.get_running_loop()
    while True:
        expected_at = loop.time() + _EVENT_LOOP_PROBE_INTERVAL_S
        await asyncio.sleep(_EVENT_LOOP_PROBE_INTERVAL_S)
        _emit(
            {"event_loop_lag": max(0.0, loop.time() - expected_at) * 1000},
            {"rl.event_loop": name},
        )


@asynccontextmanager
async def monitor_event_loop(name: str) -> AsyncIterator[None]:
    """Measure loop scheduling lag while a controller/server is alive.

    Creates one probe task only while metrics export is enabled. The task is
    cancelled and awaited on exit, including failure and cancellation paths.
    A blocked loop cannot emit until it runs again; this is not a watchdog.
    """
    if not telemetry_metrics_exporting():
        yield
        return
    probe = asyncio.create_task(_probe_event_loop(name), name=f"rl-loop-probe-{name}")
    try:
        yield
    finally:
        probe.cancel()
        with suppress(asyncio.CancelledError):
            await probe
