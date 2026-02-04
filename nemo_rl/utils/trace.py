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
"""Tracing infrastructure for NemoRL training with Perfetto/Chrome Trace Format support.

This module provides lightweight tracing for GRPO and other RL training algorithms,
generating Chrome Trace Event Format JSON files that can be visualized in Perfetto UI
(https://ui.perfetto.dev) or chrome://tracing.

Usage:
    # Enable tracing via environment variable
    export NEMORL_TRACE_ENABLED=1
    export NEMORL_TRACE_FILE=/path/to/trace.json

    # In your training code
    from nemo_rl.utils.trace import new_tracer, save_trace

    tracer = new_tracer("grpo_driver")

    for step in range(42):
        with tracer.span("step", metadata={"step": step}):
            with tracer.span("generation"):
                # generation code
                pass
            with tracer.span("training"):
                # training code
                pass

    save_trace(tracer.get_events(), actors=())
"""
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from nemo_rl.utils.timer import Timer

Event = dict[str, Any]


class Tracer:
    """Lightweight tracer for NemoRL training that outputs Chrome Trace Format.

    This tracer accumulates timing events during training and exports them to
    a JSON file compatible with Perfetto UI and chrome://tracing.
    """

    def __init__(
        self,
        enabled: bool = False,
        name: str = "",
    ):
        """Initialize the tracer.

        Args:
            enabled: Whether tracing is enabled. If False, all operations are no-ops.
        """
        self._enabled = enabled
        self._events: list[Event] = []
        self._events_lock = threading.Lock()
        self._span_stack: list[tuple[str, float, dict[str, Any]]] = []

        self._pid = os.getpid()
        self._name = name
        # We only intitialize tid upon the first span. This allows us to create the
        # tracer on a different thread from the one it'll be used on.
        self._tid = None

    def _ensure_tid(self):
        tid = threading.current_thread().native_id
        if self._tid is None:
            self._tid = tid
        else:
            assert tid == self._tid, f"Tracer used on different threads: {tid=} <> {self._tid=}"

    def start_span(self, name: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Start a traced span. Make sure to call end_span(name) when done.

        Args:
            name: Name of the span (e.g., "generation", "training", "step")
            metadata: Optional metadata to attach to the span (e.g., step numbers)

        Example:
            tracer.start_span("step", metadata={"step": step})
            # ...
            tracer.end_span("step")
        """
        if not self._enabled:
            return
        self._ensure_tid()

        start_ts = time.monotonic()
        if metadata is None:
            metadata = {}
        metadata["tracer_name"] = self._name
        self._span_stack.append((name, start_ts, metadata))

        begin_event = {
            "name": name,
            "ph": "B",  # Begin phase
            "ts": int(start_ts * 1_000_000),  # microseconds
            "pid": self._pid,
            "tid": self._tid,
            "args": metadata,
        }

        with self._events_lock:
            self._events.append(begin_event)

    def end_span(self, name: str) -> None:
        """End the most recently started span.

        Args:
            name: Optional name to verify we're ending the right span.

        Raises:
            ValueError: If name doesn't match current span or no span is active
        """
        if not self._enabled:
            return

        if not self._span_stack:
            raise ValueError(f"No active span to end (expected {name=})")

        span_name, _span_start, _span_metadata = self._span_stack.pop()

        if name != span_name:
            raise ValueError(f"Span name mismatch: expected '{name}', got '{span_name}'")

        end_event = {
            "name": span_name,
            "ph": "E",  # End phase
            "ts": int(time.monotonic() * 1_000_000),  # microseconds
            "pid": self._pid,
            "tid": self._tid,
        }

        with self._events_lock:
            self._events.append(end_event)

    @contextmanager
    def span(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[None, None, None]:
        """Create a traced span (timing block) with optional metadata.

        Args:
            name: Name of the span (e.g., "generation", "training", "step")
            metadata: Optional metadata to attach to the span (e.g., step numbers)

        Example:
            with tracer.span("step", metadata={"step": step}):
                # ...
        """
        if not self._enabled:
            yield
            return
        self._ensure_tid()

        self.start_span(name, metadata)
        try:
            yield
        finally:
            self.end_span(name)

    def add_instant_event(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        scope: str = "t",
    ) -> None:
        """Add an instant event (point-in-time marker) to the trace.

        Args:
            name: Name of the instant event
            metadata: Optional metadata to attach
            scope: Scope of the event ("t" = thread, "p" = process, "g" = global)
        """
        if not self._enabled:
            return
        self._ensure_tid()
        assert scope in ("t", "p", "g")

        if metadata is None:
            metadata = {}
        metadata["tracer_name"] = self._name

        event = {
            "name": name,
            "ph": "i",  # Instant event
            "ts": int(time.monotonic() * 1_000_000),
            "pid": self._pid,
            "tid": self._tid,
            "s": scope,
            "args": metadata,
        }

        with self._events_lock:
            self._events.append(event)

    def add_counter(
        self,
        name: str,
        value: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a counter event to the trace.

        Counter events are useful for tracking metrics over time (e.g., reward,
        loss, batch size) and appear as line graphs in trace viewers.

        Args:
            name: Name of the counter
            value: Counter value
            metadata: Optional additional metadata
        """
        if not self._enabled:
            return
        self._ensure_tid()

        if metadata is None:
            metadata = {}
        metadata["tracer_name"] = self._name
        metadata["value"] = value

        event = {
            "name": name,
            "ph": "C",  # Counter event
            "ts": int(time.monotonic() * 1_000_000),
            "pid": self._pid,
            "tid": self._tid,
            "args": metadata,
        }

        with self._events_lock:
            self._events.append(event)

    def get_events(self) -> list[Event]:
        """Get the accumulated trace events.

        Useful for programmatic analysis or custom export formats.

        Returns:
            List of trace event dictionaries
        """
        with self._events_lock:
            return list(self._events)

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled


def tracing_enabled():
    return os.environ.get("NEMORL_TRACE_ENABLED", "0").lower() in ("1", "true", "yes")


def new_tracer(name: str = "") -> Tracer:
    return Tracer(enabled=tracing_enabled(), name=name)


def define_collect_trace(get_tracer_events):
    def collect_trace(self, timing: bool):
        if timing:
            return time.monotonic()
        else:
            return get_tracer_events(self)
    return collect_trace


def save_trace(local_events: list[Event], actors: tuple[..., Any]):
    if not tracing_enabled():
        return

    events = local_events
    for actor in actors:
        # Poor man's clock synchronization to account for actors running on different
        # nodes.
        ts_local = time.monotonic()
        ts_actor = ray.get(actor.collect_trace.remote(timing=True))
        latency = (time.monotonic() - ts_local) / 2
        ts_delta = int((ts_actor - ts_local - latency) * 1_000_000)

        actor_events = ray.get(actor.collect_trace.remote(timing=False))
        for actor_event in actor_events:
            actor_event["ts"] -= ts_delta
        events.extend(actor_events)

    # Perfetto wants events to be sorted. Ensure that they are, even if we merged tracers.
    events.sort(key=lambda event: event["ts"])

    output_path = os.environ.get("NEMORL_TRACE_FILE", "nemorl_trace.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"Trace saved to: {output_path}")
    print("View in Perfetto UI: https://ui.perfetto.dev")
    print("Or open in Chrome: chrome://tracing")
    print(f"Total events: {len(events)}")


@contextmanager
def trace_and_time(
    tracer: Tracer,
    timer: Timer,
    span_name: str,
    time_label: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    time_label = time_label or span_name
    with tracer.span(span_name, metadata), timer.time(time_label):
        yield
