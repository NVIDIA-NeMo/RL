# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bounded checkpoint diagnostics; never retain tensors or inspect the heap.

GC callbacks only append numeric observations. Logging runs outside the
collector and outside snapshot-gate locks. Thread CPU is an
interval observation, not a measurement of GIL ownership.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class CheckpointLatencyConfig(BaseModel, extra="forbid"):
    """Opt-in diagnostics shared by production and generation-only benchmarks."""

    enabled: bool = False
    progress_interval_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 5.0
    slow_call_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    max_events: Annotated[int, Field(ge=16, le=4096, strict=True)] = 256
    tq_internal_phases: bool = True
    gc_intervals: bool = True


@dataclass(frozen=True, slots=True)
class GateWork:
    operation: str
    model_call_id: str | None
    started_unix_s: float
    thread_id: int
    native_thread_id: int


G_GATE_WORK: ContextVar[GateWork | None] = ContextVar(
    "checkpoint_gate_work", default=None
)


def emit_latency(event_name: str, **values: Any) -> None:
    """Diagnostic handlers must not alter checkpoint or storage outcomes."""
    try:
        LOGGER.info(
            "%s %s",
            event_name,
            json.dumps(
                {"pid": os.getpid(), "unix_s": time.time(), **values}, sort_keys=True
            ),
        )
    except Exception:
        # Telemetry only: even an unavailable logging sink cannot fail generation.
        pass


class CheckpointLatencyDiagnostics:
    """Process-local GC intervals correlated with a checkpoint fence."""

    def __init__(self, config: CheckpointLatencyConfig, *, worker_ip: str) -> None:
        self.config = config
        self.worker_ip = worker_ip
        self.checkpoint_id: str | None = None
        self._window_started = 0.0
        self._gc_starts: dict[int, tuple[float, float, float, int]] = {}
        self._gc_events: deque[dict[str, Any]] = deque(maxlen=config.max_events)
        self._gc_count = 0
        self._gc_baseline = 0
        self._gc_wall_s = 0.0
        self._gc_cpu_s = 0.0
        self._gc_baseline_wall_s = 0.0
        self._gc_baseline_cpu_s = 0.0
        self._gc_longest: dict[str, Any] | None = None
        self._closed = False
        if config.enabled and config.gc_intervals:
            gc.callbacks.append(self._on_gc)

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        # Do not log, enumerate objects, acquire application locks, or start a
        # profiler in a GC callback. A collector can interrupt a lock owner.
        ident = threading.get_ident()
        if phase == "start":
            self._gc_starts[ident] = (
                time.time(),
                time.perf_counter(),
                time.thread_time(),
                info["generation"],
            )
        elif phase == "stop":
            start = self._gc_starts.pop(ident, None)
            if start is not None:
                self._gc_count += 1
                event = {
                    "started_unix_s": start[0],
                    "finished_unix_s": time.time(),
                    "wall_s": time.perf_counter() - start[1],
                    "thread_cpu_s": time.thread_time() - start[2],
                    "generation": start[3],
                    "thread_id": ident,
                    "native_thread_id": threading.get_native_id(),
                    "collected": info["collected"],
                    "uncollectable": info["uncollectable"],
                }
                self._gc_events.append(event)
                self._gc_wall_s += event["wall_s"]
                self._gc_cpu_s += event["thread_cpu_s"]
                if self.checkpoint_id is not None and (
                    self._gc_longest is None
                    or event["wall_s"] > self._gc_longest["wall_s"]
                ):
                    self._gc_longest = event

    def begin(self, checkpoint_id: str) -> None:
        """Begin one fenced window before scheduling the fence executor."""
        if self._closed or not self.config.enabled:
            return
        # A new window finishes the previous checkpoint's observations.
        if self.checkpoint_id is not None:
            self.end(self.checkpoint_id)
        self.checkpoint_id = checkpoint_id
        self._window_started = time.time()
        self._gc_baseline = self._gc_count
        self._gc_baseline_wall_s = self._gc_wall_s
        self._gc_baseline_cpu_s = self._gc_cpu_s
        self._gc_longest = None
        emit_latency(
            "checkpoint_latency_window",
            event="begin",
            checkpoint_id=checkpoint_id,
            worker_ip=self.worker_ip,
        )

    def end(self, checkpoint_id: str) -> None:
        """Report overlapping GC intervals after releasing the snapshot gate."""
        if self.checkpoint_id != checkpoint_id:
            return
        now = time.time()
        events = [
            event
            for event in self._gc_events
            if event["finished_unix_s"] >= self._window_started
        ]
        emit_latency(
            "checkpoint_gc_intervals",
            checkpoint_id=checkpoint_id,
            worker_ip=self.worker_ip,
            started_unix_s=self._window_started,
            finished_unix_s=now,
            events=events,
            dropped_events=max(0, self._gc_count - self._gc_baseline - len(events)),
            active_collections=len(self._gc_starts),
            completed_collection_wall_s=self._gc_wall_s - self._gc_baseline_wall_s,
            completed_collection_thread_cpu_s=self._gc_cpu_s - self._gc_baseline_cpu_s,
            longest_collection=self._gc_longest,
        )
        self.checkpoint_id = None

    def close(self) -> None:
        """Unregister the collector callback on shutdown."""
        if self._closed:
            return
        if self.checkpoint_id is not None:
            self.end(self.checkpoint_id)
        self._closed = True
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
