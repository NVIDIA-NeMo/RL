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

"""Live, trace-only checkpoint phases and bounded progress diagnostics.

Gym owns participant and capture instrumentation. This module observes the RL
coordination boundaries around it. Nested phases carry no goodput bucket and
must not be summed. Completed save/outcome metrics remain the logger tee's job.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from nemo.lens.instruments import MetricSpec, record_metrics, register_metric_group

from nemo_rl.telemetry.instrumentation import (
    is_span_group_enabled,
    safe_set_span_attributes,
    span_cm,
)
from nemo_rl.telemetry.setup import get_telemetry_handle
from nemo_rl.telemetry.span_groups import RLSpanGroup

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import (
        DataPlaneCheckpointBarrier,
        DataPlaneMutationCut,
    )

logger = logging.getLogger(__name__)

Operation = Literal["save", "restore"]
Phase = Literal[
    "save_lock",
    "trainer_anchor",
    "snapshot_paths",
    "dispatch_pause",
    "generation_fence",
    "generation_release",
    "retire_dropped",
    "gym_prepare",
    "gym_renew",
    "gym_commit",
    "gym_resume",
    "gym_retire",
    "gym_forget",
    "gym_restore",
    "gym_restore_retire",
    "gym_control",
    "validate_staging",
    "drain_replies",
    "dp_barrier_wait",
    "dp_barrier_hold",
    "validate_cut",
    "capture_state",
    "tq_save",
    "tq_load",
    "sidecars",
    "publication",
    "snapshot_cleanup",
    "replay_restore",
    "recovery_prepare",
    "restore_inventory",
    "restore_prompts",
    "redispatch_schedule",
    "prefix_control",
    "prefix_cut",
    "prefix_batch_put",
    "prefix_restore_fetch",
    "prefix_restore_reconstruct",
]

_PROGRESS_INTERVAL_S = 30.0
_MAX_PENDING_NAMES = 16
_GROUP = "rl_checkpoint_live"
_REGISTERED = False
_REGISTRATION_LOCK = threading.Lock()
_WARNED = False
_SPECS = (
    MetricSpec(
        "phase_elapsed",
        "rl.rollout_checkpoint.phase_elapsed",
        "gauge",
        "s",
        "Elapsed time in an active RL checkpoint phase; zero on exit",
    ),
    MetricSpec(
        "pending_deployments",
        "rl.rollout_checkpoint.pending_deployments",
        "gauge",
        "{deployment}",
        "Gym deployment replies still awaited by RL",
    ),
    MetricSpec(
        "progress_age",
        "rl.rollout_checkpoint.progress_age",
        "gauge",
        "s",
        "Time since the last phase transition or deployment reply",
    ),
    MetricSpec(
        "phase_duration",
        "rl.rollout_checkpoint.live_phase.duration",
        "histogram",
        "ms",
        "RL coordination and prefix phase duration; nested phases overlap",
    ),
)
# Persistence, barrier and sidecar durations already belong to #4730's tee.
# These are the additional RL boundaries, not a second set of Gym timers.
_DURATION_PHASES = frozenset(
    {
        "dispatch_pause",
        "generation_fence",
        "generation_release",
        "gym_prepare",
        "gym_commit",
        "gym_resume",
        "gym_restore",
        "drain_replies",
        "prefix_control",
        "prefix_cut",
        "prefix_batch_put",
    }
)


def _enabled() -> bool:
    handle = get_telemetry_handle()
    return handle is not None and handle.is_exporting


def _record(values: dict[str, float], *, operation: str, phase: str) -> None:
    """Telemetry failure must not affect checkpoint state or cancellation."""
    global _REGISTERED, _WARNED
    handle = get_telemetry_handle()
    if handle is None or not handle.is_exporting:
        return
    try:
        if not _REGISTERED:
            with _REGISTRATION_LOCK:
                if not _REGISTERED:
                    register_metric_group(_GROUP, _SPECS)
                    _REGISTERED = True
        record_metrics(
            handle.meter,
            _GROUP,
            values,
            attributes={
                "rl.checkpoint.operation": operation,
                "rl.checkpoint.phase": phase,
            },
        )
    except Exception:
        if not _WARNED:
            _WARNED = True
            logger.warning(
                "Failed to record live checkpoint diagnostics", exc_info=True
            )


@contextmanager
def _span(name: str, **attributes: Any) -> Iterator[Any]:
    # span_cm is intentionally unbucketed: these phases nest and worker spans
    # overlap. CHECKPOINT's normal leaf helper would double-count goodput.
    if not is_span_group_enabled(RLSpanGroup.CHECKPOINT):
        yield None
        return
    with span_cm(name, record_exception=False, **attributes) as span:
        yield span


@dataclass(eq=False)
class CheckpointPhase:
    """Progress of one active phase; instance names are log/span attributes only."""

    name: str
    started: float = field(default_factory=time.monotonic)
    pending_count: int = 0
    pending_names: tuple[str, ...] = ()
    finished: bool = False
    last_progress: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def pending(self, instances: Iterable[str]) -> None:
        """Update after actual replies; observation alone is not progress."""
        names = tuple(instances)
        with self.lock:
            self.pending_count = len(names)
            self.pending_names = names[:_MAX_PENDING_NAMES]
            self.last_progress = time.monotonic()

    def snapshot(self, now: float) -> dict[str, Any]:
        """Return a bounded, JSON-safe diagnostic, without rollout identities."""
        with self.lock:
            return {
                "phase": self.name,
                "elapsed_s": max(0.0, now - self.started),
                "seconds_since_progress": max(0.0, now - self.last_progress),
                "pending_deployments": self.pending_count,
                "pending_instances": self.pending_names,
            }

    def record_live(self, operation: str) -> None:
        """Serialize observation with exit, so stale samples cannot revive gauges."""
        with self.lock:
            if self.finished:
                return
            now = time.monotonic()
            _record(
                {
                    "phase_elapsed": max(0.0, now - self.started),
                    "pending_deployments": float(self.pending_count),
                    "progress_age": max(0.0, now - self.last_progress),
                },
                operation=operation,
                phase=self.name,
            )


@dataclass
class CheckpointAttempt:
    """One RL attempt, shared with its observation task and copied thread context."""

    operation: Operation
    checkpoint_id: str
    outcome: str = "completed"
    reason: str = "completed"
    started: float = field(default_factory=time.monotonic)
    active: set[CheckpointPhase] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            active = tuple(self.active)
        return {
            "checkpoint_id": self.checkpoint_id,
            "operation": self.operation,
            "elapsed_s": max(0.0, now - self.started),
            "phases": [
                phase.snapshot(now) for phase in sorted(active, key=lambda p: p.started)
            ],
        }


_ATTEMPT: ContextVar[CheckpointAttempt | None] = ContextVar(
    "rl_checkpoint_attempt", default=None
)


def checkpoint_identity() -> str | None:
    """The current attempt's ID, also used for Gym's checkpoint protocol."""
    attempt = _ATTEMPT.get()
    return attempt.checkpoint_id if attempt is not None else None


async def _observe(attempt: CheckpointAttempt) -> None:
    while True:
        await asyncio.sleep(_PROGRESS_INTERVAL_S)
        with attempt.lock:
            active = tuple(attempt.active)
        for phase in active:
            phase.record_live(attempt.operation)
        snapshot = attempt.snapshot()
        logger.warning(
            "rollout_checkpoint_progress %s", json.dumps(snapshot, sort_keys=True)
        )


@asynccontextmanager
async def checkpoint_attempt(
    operation: Operation, *, enabled: bool = True
) -> AsyncIterator[CheckpointAttempt | None]:
    """Observe an attempt, including failures and skips, without changing deadlines.

    The observer shares the event loop; it is not an independent hang watchdog.
    A trace's attempt ID is deliberately absent from all metric attributes.
    """
    if not enabled or not _enabled():
        yield None
        return
    attempt = CheckpointAttempt(operation, f"rollout-{uuid.uuid4().hex}")
    token = _ATTEMPT.set(attempt)
    try:
        with _span(
            f"rl.rollout_checkpoint.{operation}",
            **{
                "rl.checkpoint.id": attempt.checkpoint_id,
                "rl.checkpoint.operation": operation,
            },
        ) as span:
            observer = asyncio.create_task(_observe(attempt))
            try:
                yield attempt
            except BaseException as error:
                attempt.outcome = (
                    "cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "failed"
                )
                attempt.reason = type(error).__name__
                logger.warning(
                    "rollout_checkpoint_interrupted %s",
                    json.dumps(attempt.snapshot(), sort_keys=True),
                )
                raise
            finally:
                task = asyncio.current_task()
                cancelling_before = task.cancelling() if task is not None else 0
                observer.cancel()
                try:
                    await observer
                except asyncio.CancelledError:
                    # Distinguish the observer's cancellation from a caller
                    # cancelling this attempt during observer teardown.
                    if task is not None and (
                        task.cancelling() > cancelling_before
                        or (task.cancelling() and attempt.outcome != "cancelled")
                    ):
                        attempt.outcome = "cancelled"
                        attempt.reason = "CancelledError"
                        raise
                except Exception:
                    # Diagnostics may fail, but must never turn a successful
                    # save into failure or replace the checkpoint exception.
                    logger.warning("Checkpoint progress observer failed", exc_info=True)
                finally:
                    safe_set_span_attributes(
                        span,
                        {
                            "rl.checkpoint.outcome": attempt.outcome,
                            "rl.checkpoint.reason": attempt.reason,
                        },
                    )
    finally:
        _ATTEMPT.reset(token)


@contextmanager
def checkpoint_phase(
    phase: Phase, *, detail: bool = False, attributes: Mapping[str, Any] | None = None
) -> Iterator[CheckpointPhase | None]:
    """Time a bounded RL phase; use Gym's existing instrumentation inside Gym."""
    if not _enabled() or (detail and not is_span_group_enabled(RLSpanGroup.PER_PROMPT)):
        yield None
        return
    attempt = _ATTEMPT.get()
    progress = CheckpointPhase(phase)
    if attempt is not None:
        with attempt.lock:
            attempt.active.add(progress)
    attrs = {"rl.checkpoint.phase": phase, **(attributes or {})}
    if attempt is not None:
        attrs["rl.checkpoint.id"] = attempt.checkpoint_id
    try:
        with _span(f"rl.rollout_checkpoint.{phase}", **attrs) as span:
            try:
                yield progress
            except BaseException as error:
                # Capture the active phase before stack unwinding removes it.
                safe_set_span_attributes(
                    span,
                    {
                        "rl.checkpoint.outcome": "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else "failed",
                        "error.type": type(error).__name__,
                        "rl.checkpoint.pending_deployments": progress.pending_count,
                        "rl.checkpoint.pending_instances": progress.pending_names,
                    },
                )
                diagnostic = progress.snapshot(time.monotonic())
                diagnostic["checkpoint_id"] = checkpoint_identity()
                logger.warning(
                    "rollout_checkpoint_phase_interrupted %s",
                    json.dumps(diagnostic, sort_keys=True),
                )
                raise
            else:
                safe_set_span_attributes(span, {"rl.checkpoint.outcome": "completed"})
    finally:
        if attempt is not None:
            with attempt.lock:
                attempt.active.discard(progress)
        with progress.lock:
            progress.finished = True
            if attempt is not None:
                _record(
                    {
                        "phase_elapsed": 0.0,
                        "pending_deployments": 0.0,
                        "progress_age": 0.0,
                    },
                    operation=attempt.operation,
                    phase=phase,
                )
        if phase in _DURATION_PHASES:
            _record(
                {"phase_duration": (time.monotonic() - progress.started) * 1000.0},
                operation=attempt.operation if attempt else "worker",
                phase=phase,
            )


_F = TypeVar("_F", bound=Callable[..., Any])


def checkpoint_phase_async(phase: Phase, *, detail: bool = False) -> Callable[[_F], _F]:
    """Decorate an async phase, preserving the method type Ray introspects."""

    def decorate(function: _F) -> _F:
        @functools.wraps(function)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            with checkpoint_phase(phase, detail=detail):
                return await function(*args, **kwargs)

        return cast(_F, wrapped)

    return decorate


def checkpoint_phase_sync(phase: Phase, *, detail: bool = False) -> Callable[[_F], _F]:
    """Decorate a blocking phase, preserving the original method's type."""

    def decorate(function: _F) -> _F:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with checkpoint_phase(phase, detail=detail):
                return function(*args, **kwargs)

        return cast(_F, wrapped)

    return decorate


class _SaveResult(Protocol):
    @property
    def saved(self) -> bool: ...

    @property
    def reason(self) -> str: ...


_SaveF = TypeVar("_SaveF", bound=Callable[..., Awaitable[_SaveResult]])


def trace_checkpoint_save(function: _SaveF) -> _SaveF:
    """Wrap the real save, so direct callers and periodic attempts are observed."""

    @functools.wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> _SaveResult:
        async with checkpoint_attempt("save") as attempt:
            result = await function(*args, **kwargs)
            if attempt is not None:
                attempt.outcome = "completed" if result.saved else "skipped"
                attempt.reason = result.reason
            return result

    return cast(_SaveF, wrapped)


@asynccontextmanager
async def checkpoint_barrier(
    barrier: DataPlaneCheckpointBarrier,
) -> AsyncIterator[DataPlaneMutationCut]:
    """Measure acquisition and exclusive hold separately; preserve error exits."""
    # AsyncExitStack forwards exception/cancellation to the original barrier;
    # manually calling __aexit__(None, None, None) here would change semantics.
    async with AsyncExitStack() as stack:
        with checkpoint_phase("dp_barrier_wait"):
            cut = await stack.enter_async_context(barrier.checkpoint())
        with checkpoint_phase("dp_barrier_hold"):
            yield cut


@asynccontextmanager
async def checkpoint_lock(lock: asyncio.Lock) -> AsyncIterator[None]:
    """Include serialization behind an earlier save in the attempt trace."""
    async with AsyncExitStack() as stack:
        with checkpoint_phase("save_lock"):
            await stack.enter_async_context(lock)
        yield
