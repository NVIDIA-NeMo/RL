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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from nemo_rl.telemetry.metrics import (
    EVENT_LOOP_LAG_KEY,
    EXECUTOR_EXECUTION_KEY,
    EXECUTOR_PENDING_KEY,
    EXECUTOR_QUEUE_WAIT_KEY,
    EXECUTOR_RESULT_DELIVERY_KEY,
    record_rl_metrics,
    warn_once,
)
from nemo_rl.telemetry.setup import telemetry_metrics_exporting

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import Lifespan

_T = TypeVar("_T")
_EVENT_LOOP_PROBE_INTERVAL_S = 1.0

# Bounded call-site label carried by every ``rl.executor.*`` series.
RL_EXECUTOR_SITE_ATTR = "rl.executor.site"
# ``queued`` or ``running``; only on ``rl.executor.pending``.
RL_EXECUTOR_STATE_ATTR = "rl.executor.state"
# ``success`` or ``error``; only on ``rl.executor.execution``.
RL_EXECUTOR_STATUS_ATTR = "rl.executor.status"
# Bounded loop label carried by ``rl.event_loop.lag``.
RL_EVENT_LOOP_ATTR = "rl.event_loop"


def _emit(values: Mapping[str, float], attributes: Mapping[str, str]) -> None:
    # Deliberately best-effort: optional instrumentation must never change the
    # callable's result, exception, cancellation or cleanup behaviour.
    try:
        record_rl_metrics(values, attributes)
    except Exception:
        warn_once("runtime_metrics", "failed to record runtime diagnostics")


@dataclass
class _Submission:
    site: str
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
                    {EXECUTOR_PENDING_KEY: -1.0},
                    {
                        RL_EXECUTOR_SITE_ATTR: self.site,
                        RL_EXECUTOR_STATE_ATTR: "queued",
                    },
                )
            self.state = "running"
            _emit(
                {EXECUTOR_PENDING_KEY: 1.0},
                {RL_EXECUTOR_SITE_ATTR: self.site, RL_EXECUTOR_STATE_ATTR: "running"},
            )
            _emit(
                {EXECUTOR_QUEUE_WAIT_KEY: (started_at - self.submitted_at) * 1000},
                {RL_EXECUTOR_SITE_ATTR: self.site},
            )
            return True

    def finish(self, *, status: str, started_at: float) -> None:
        with self.lock:
            self.state = "finished"
            self.finished_at = time.perf_counter()
            _emit(
                {EXECUTOR_EXECUTION_KEY: (self.finished_at - started_at) * 1000},
                {RL_EXECUTOR_SITE_ATTR: self.site, RL_EXECUTOR_STATUS_ATTR: status},
            )
            _emit(
                {EXECUTOR_PENDING_KEY: -1.0},
                {RL_EXECUTOR_SITE_ATTR: self.site, RL_EXECUTOR_STATE_ATTR: "running"},
            )

    def retire_queued(self) -> None:
        with self.lock:
            if self.state != "queued":
                return
            self.state = "retired"
            _emit(
                {EXECUTOR_PENDING_KEY: -1.0},
                {RL_EXECUTOR_SITE_ATTR: self.site, RL_EXECUTOR_STATE_ATTR: "queued"},
            )


async def run_in_thread(
    site: str, function: Callable[..., _T], /, *args: Any, **kwargs: Any
) -> _T:
    """Observe one default-executor call without changing asyncio semantics.

    Records queue wait, execution time and queued/running counts. Successful
    calls also record completion-to-coroutine-resumption delay. Cancellation
    does not pretend to stop an already-running thread: it remains counted
    until the function exits. With metrics off, delegates to ``asyncio.to_thread``.

    Args:
        site: A bounded call-site label, such as ``generation_terminal``. It
            identifies work sharing the default executor, not a dedicated pool.
        function: Synchronous function to offload.
        *args: Positional arguments forwarded to the function.
        **kwargs: Keyword arguments forwarded to the function.

    Returns:
        The function's return value.
    """
    if not telemetry_metrics_exporting():
        return await asyncio.to_thread(function, *args, **kwargs)

    submission = _Submission(site, 0.0)
    _emit(
        {EXECUTOR_PENDING_KEY: 1.0},
        {RL_EXECUTOR_SITE_ATTR: site, RL_EXECUTOR_STATE_ATTR: "queued"},
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
                EXECUTOR_RESULT_DELIVERY_KEY: (
                    time.perf_counter() - submission.finished_at
                )
                * 1000
            },
            {RL_EXECUTOR_SITE_ATTR: site},
        )
    return result


async def _probe_event_loop(name: str) -> None:
    loop = asyncio.get_running_loop()
    while True:
        expected_at = loop.time() + _EVENT_LOOP_PROBE_INTERVAL_S
        await asyncio.sleep(_EVENT_LOOP_PROBE_INTERVAL_S)
        _emit(
            {EVENT_LOOP_LAG_KEY: max(0.0, loop.time() - expected_at) * 1000},
            {RL_EVENT_LOOP_ATTR: name},
        )


@asynccontextmanager
async def monitor_event_loop(name: str) -> AsyncIterator[None]:
    """Measure loop scheduling lag while a controller/server is alive.

    Creates one probe task only while metrics export is enabled. The task is
    cancelled and awaited on exit, including failure and cancellation paths.
    A blocked loop cannot emit until it runs again; this is not a watchdog.

    Args:
        name: Bounded loop label, such as ``controller`` or ``vllm_http``.
    """
    if not telemetry_metrics_exporting():
        yield
        return
    probe = asyncio.create_task(_probe_event_loop(name), name=f"rl-loop-probe-{name}")
    try:
        yield
    finally:
        probe.cancel()
        # wait() rather than awaiting the probe: a cancel aimed at the owner
        # while it waits here must propagate, not be absorbed as the probe's
        # own CancelledError.
        await asyncio.wait({probe})


def monitor_app_event_loop(app: FastAPI, name: str) -> None:
    """Run a loop probe for the lifetime of ``app``'s serving loop.

    Wraps the app's current lifespan, so call it after every other lifespan or
    startup/shutdown registration. Their lifecycle and any yielded state are
    preserved.

    Args:
        app: FastAPI app whose lifespan should own the probe.
        name: Bounded loop label, such as ``vllm_http``.
    """
    inner = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(server_app: FastAPI) -> AsyncIterator[Mapping[str, Any] | None]:
        async with monitor_event_loop(name):
            async with inner(server_app) as state:
                yield state

    app.router.lifespan_context = cast("Lifespan[FastAPI]", lifespan)
