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
import itertools
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import ray

from nemo_rl.utils.timer import Timer

Event = dict[str, Any]

# Counter used to mint unique virtual TIDs per async span so Perfetto renders
# each span in its own thread lane (real-thread-shared Complete events overlap
# badly and get visually collapsed). Starts at a high base so virtual TIDs
# cannot collide with real OS thread IDs (typically < 2**22).
_virtual_tid_counter = itertools.count(0x70000000)

# Counter for minting virtual PIDs. A Tracer with ``virtual_process_name`` set
# uses a virtual PID so Perfetto renders it as its own process block (with the
# real tracer thread and every per-span virtual TID contained inside). High
# base keeps virtual PIDs clear of any real OS PIDs.
_virtual_pid_counter = itertools.count(0x60000000)

# Per-tracer sort-index block. Without thread_sort_index metadata, Perfetto
# orders threads within a process by TID, which places virtual TIDs (very
# large numbers) far from their real owning thread. Each tracer reserves a
# block of sort indices so the real thread and its virtual TIDs render
# adjacently. Block size (100) caps concurrent async spans per tracer in the
# sort-adjacent view; overflow still renders correctly, just not adjacent.
_sort_index_counter = itertools.count(0)
_SORT_BLOCK_SIZE = 100


class Tracer:
    """Lightweight tracer for NemoRL training that outputs Chrome Trace Format.

    This tracer accumulates timing events during training and exports them to
    a JSON file compatible with Perfetto UI and chrome://tracing.
    """

    def __init__(
        self,
        enabled: bool = False,
        name: str = "",
        virtual_process_name: Optional[str] = None,
    ):
        """Initialize the tracer.

        Args:
            enabled: Whether tracing is enabled. If False, all operations are no-ops.
            name: Label used for the tracer's real thread lane in Perfetto.
            virtual_process_name: If set, the tracer emits events under a minted
                virtual PID with this label, so Perfetto renders all of the
                tracer's events (real thread + per-span virtual threads) inside
                a dedicated process block. Use this for tracers whose activity
                is a logical unit that should be grouped together (e.g. a
                prompt-group worker and its concurrent samples).
        """
        self._enabled = enabled
        self._events: list[Event] = []
        self._events_lock = threading.Lock()
        self._span_stack: list[tuple[str, float, dict[str, Any]]] = []
        # Active async spans keyed by (name, async_id, category) ->
        # (start_ts, metadata, virtual_tid). We emit a single Chrome Trace "X"
        # Complete event on end so spans render on the thread (real or virtual)
        # rather than on a shared process-level async track.
        self._async_spans: dict[
            tuple[str, str, str], tuple[float, dict[str, Any], int]
        ] = {}
        # async_id -> virtual TID, stable within a tracer's lifetime.
        self._virtual_tids: dict[str, int] = {}
        # Sort-index block reserved for this tracer; real TID occupies offset 0
        # and virtual TIDs occupy offsets 1, 2, ... within this block so they
        # render adjacent to the real TID in Perfetto.
        self._sort_index_base = next(_sort_index_counter) * _SORT_BLOCK_SIZE

        if virtual_process_name is not None:
            self._pid = next(_virtual_pid_counter)
        else:
            self._pid = os.getpid()
        self._virtual_process_name = virtual_process_name
        self._name = name
        # We only intitialize tid upon the first span. This allows us to create the
        # tracer on a different thread from the one it'll be used on.
        self._tid = None

    def _ensure_tid(self):
        tid = threading.current_thread().native_id
        if self._tid is None:
            self._tid = tid
            ts = int(time.monotonic() * 1_000_000)
            with self._events_lock:
                # If this tracer owns a virtual process, label it so Perfetto
                # renders a named process block.
                if self._virtual_process_name:
                    self._events.append({
                        "name": "process_name",
                        "ph": "M",
                        "ts": ts,
                        "pid": self._pid,
                        "args": {"name": self._virtual_process_name},
                    })
                # Emit Chrome Trace metadata so Perfetto displays the thread
                # with the tracer's name instead of a raw TID, and sorts it
                # next to this tracer's virtual-TID sample lanes.
                if self._name:
                    self._events.append({
                        "name": "thread_name",
                        "ph": "M",
                        "ts": ts,
                        "pid": self._pid,
                        "tid": self._tid,
                        "args": {"name": self._name},
                    })
                    self._events.append({
                        "name": "thread_sort_index",
                        "ph": "M",
                        "ts": ts,
                        "pid": self._pid,
                        "tid": self._tid,
                        "args": {"sort_index": self._sort_index_base},
                    })
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

    def _get_virtual_tid(self, async_id: str) -> int:
        """Assign (once) a unique virtual TID for this async span.

        Chrome Trace Complete events on a shared real TID overlap in ways
        Perfetto collapses during rendering. Giving each concurrent span its
        own virtual TID makes every span appear as a distinct thread lane.
        The real owning thread is preserved via ``owner_tid`` in span args.
        """
        if async_id not in self._virtual_tids:
            vtid = next(_virtual_tid_counter)
            slot = len(self._virtual_tids) + 1  # real TID occupies slot 0
            self._virtual_tids[async_id] = vtid
            # If the tracer owns a virtual process, the process block already
            # names the group, so the per-span lane just needs the async_id.
            # Otherwise prefix with the tracer name so virtual lanes are still
            # identifiable among the many threads of a shared process.
            if self._virtual_process_name:
                thread_name = async_id
            elif self._name:
                thread_name = f"{self._name}/{async_id}"
            else:
                thread_name = async_id
            ts = int(time.monotonic() * 1_000_000)
            with self._events_lock:
                self._events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "ts": ts,
                    "pid": self._pid,
                    "tid": vtid,
                    "args": {"name": thread_name},
                })
                self._events.append({
                    "name": "thread_sort_index",
                    "ph": "M",
                    "ts": ts,
                    "pid": self._pid,
                    "tid": vtid,
                    "args": {"sort_index": self._sort_index_base + slot},
                })
        return self._virtual_tids[async_id]

    def start_async_span(
        self,
        name: str,
        async_id: str,
        category: str = "async",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Start an async span that may overlap with other async spans on the same thread.

        Unlike ``start_span`` these spans do not use the LIFO stack, so concurrent
        asyncio tasks on the same thread can each have their own live span. The
        actual event is emitted by ``end_async_span`` as a single Chrome Trace
        "X" (Complete) event on a per-span virtual TID so Perfetto renders each
        span in its own lane.
        """
        if not self._enabled:
            return
        self._ensure_tid()

        if metadata is None:
            metadata = {}
        metadata = {
            **metadata,
            "tracer_name": self._name,
            "async_id": async_id,
            "owner_tid": self._tid,
        }

        vtid = self._get_virtual_tid(async_id)
        self._async_spans[(name, async_id, category)] = (
            time.monotonic(),
            metadata,
            vtid,
        )

    def end_async_span(
        self,
        name: str,
        async_id: str,
        category: str = "async",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """End an async span and emit a single Chrome Trace "X" Complete event.

        Optional metadata is merged with the start metadata and attached as
        ``args`` on the emitted event, so callers can record values only known
        after the span body runs (e.g. output token counts, truncation flag).
        """
        if not self._enabled:
            return

        key = (name, async_id, category)
        state = self._async_spans.pop(key, None)
        if state is None:
            raise ValueError(
                f"No active async span to end (name={name!r}, async_id={async_id!r}, "
                f"category={category!r})"
            )
        start_ts, start_metadata, vtid = state
        end_ts = time.monotonic()

        args = dict(start_metadata)
        if metadata:
            args.update(metadata)

        event = {
            "name": name,
            "cat": category,
            "ph": "X",  # Complete event: rendered on the (virtual) thread lane
            "ts": int(start_ts * 1_000_000),
            "dur": int((end_ts - start_ts) * 1_000_000),
            "pid": self._pid,
            "tid": vtid,
            "args": args,
        }

        with self._events_lock:
            self._events.append(event)

    @contextmanager
    def async_span(
        self,
        name: str,
        async_id: str,
        category: str = "async",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager for an async (overlapping) span.

        Yields a mutable dict. Anything written into it before the context exits
        is attached as ``args`` to the end event, so callers can record values
        that are only known after the span body runs.

        Example:
            with tracer.async_span("sample_rollout", "p0_g3",
                                   category="sample",
                                   metadata={"sample_idx": 3}) as end_meta:
                result = await run_sample(...)
                end_meta["output_tokens"] = result.num_tokens
        """
        if not self._enabled:
            yield {}
            return

        self.start_async_span(name, async_id, category, metadata)
        end_metadata: dict[str, Any] = {}
        try:
            yield end_metadata
        finally:
            self.end_async_span(
                name, async_id, category,
                metadata=end_metadata or None,
            )

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


def new_tracer(
    name: str = "",
    virtual_process_name: Optional[str] = None,
) -> Tracer:
    return Tracer(
        enabled=tracing_enabled(),
        name=name,
        virtual_process_name=virtual_process_name,
    )


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
