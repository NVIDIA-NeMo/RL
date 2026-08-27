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

"""Opt-in Perfetto trace event collection for NeMo RL workloads.

The trace uses Chrome's JSON trace-event format, which can be opened directly at
https://ui.perfetto.dev. Tracing is disabled unless ``NEMORL_TRACE_ENABLED`` is
set to a truthy value. ``NEMORL_TRACE_FILE`` controls the output path.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

_TRACE_ENABLED_ENV = "NEMORL_TRACE_ENABLED"
_TRACE_FILE_ENV = "NEMORL_TRACE_FILE"
_DEFAULT_TRACE_FILE = "nemo_rl_perfetto_trace.json"
_TRUTHY = {"1", "true", "yes", "on"}


def trace_enabled() -> bool:
    """Return whether Perfetto tracing is enabled for this process."""
    return os.getenv(_TRACE_ENABLED_ENV, "0").lower() in _TRUTHY


def _timestamp_us() -> int:
    return time.monotonic_ns() // 1_000


def _stable_trace_id(value: str) -> int:
    """Map a process or thread identity to a positive signed 31-bit ID."""
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, "big") & 0x7FFF_FFFF) or 1


class Tracer:
    """Thread-safe in-memory trace-event collector.

    Regular spans are emitted on the calling OS-thread lane. Complete async
    spans are emitted on named virtual lanes, which makes independently
    progressing rollout samples easy to compare in Perfetto.

    Args:
        process_name: Display name for regular spans from this process.
        virtual_process_name: Optional display name for virtual async lanes.
        process_sort_index: Sort order for the regular process.
        virtual_process_sort_index: Sort order for the virtual process.
        enabled: Explicit enable override. By default, reads the environment.
    """

    def __init__(
        self,
        process_name: str,
        *,
        virtual_process_name: Optional[str] = None,
        process_sort_index: int = 0,
        virtual_process_sort_index: int = 1,
        enabled: Optional[bool] = None,
    ) -> None:
        self.enabled = trace_enabled() if enabled is None else enabled
        self.process_name = process_name
        self.virtual_process_name = virtual_process_name
        identity = f"{socket.gethostname()}:{os.getpid()}:{process_name}"
        self.process_id = _stable_trace_id(identity)
        self.virtual_process_id = _stable_trace_id(f"{identity}:virtual")
        self._process_sort_index = process_sort_index
        self._virtual_process_sort_index = virtual_process_sort_index
        self._events: list[dict[str, Any]] = []
        self._regular_tids: dict[int, int] = {}
        self._virtual_tids: dict[str, int] = {}
        self._open_async_spans: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        if self.enabled:
            self._add_process_metadata()

    def _append(self, event: dict[str, Any]) -> None:
        self._events.append(event)

    def _add_process_metadata(self) -> None:
        self._append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": self.process_id,
                "tid": 0,
                "args": {"name": self.process_name},
            }
        )
        self._append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": self.process_id,
                "tid": 0,
                "args": {"sort_index": self._process_sort_index},
            }
        )
        if self.virtual_process_name is not None:
            self._append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": self.virtual_process_id,
                    "tid": 0,
                    "args": {"name": self.virtual_process_name},
                }
            )
            self._append(
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": self.virtual_process_id,
                    "tid": 0,
                    "args": {"sort_index": self._virtual_process_sort_index},
                }
            )

    def _regular_tid(self) -> int:
        thread_ident = threading.get_ident()
        with self._lock:
            tid = self._regular_tids.get(thread_ident)
            if tid is not None:
                return tid
            tid = _stable_trace_id(f"{self.process_id}:thread:{thread_ident}")
            self._regular_tids[thread_ident] = tid
            self._append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": self.process_id,
                    "tid": tid,
                    "args": {"name": threading.current_thread().name},
                }
            )
            return tid

    def virtual_tid(self, track_name: str) -> int:
        """Return, creating if necessary, the virtual TID for ``track_name``."""
        with self._lock:
            tid = self._virtual_tids.get(track_name)
            if tid is not None:
                return tid
            tid = _stable_trace_id(f"{self.virtual_process_id}:track:{track_name}")
            self._virtual_tids[track_name] = tid
            pid = (
                self.virtual_process_id
                if self.virtual_process_name is not None
                else self.process_id
            )
            self._append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": track_name},
                }
            )
            self._append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": len(self._virtual_tids)},
                }
            )
            return tid

    def start_span(
        self,
        name: str,
        *,
        category: str = "driver",
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Start a synchronous span on the calling thread."""
        if not self.enabled:
            return
        event: dict[str, Any] = {
            "name": name,
            "cat": category,
            "ph": "B",
            "ts": _timestamp_us(),
            "pid": self.process_id,
            "tid": self._regular_tid(),
        }
        if args:
            event["args"] = args
        with self._lock:
            self._append(event)

    def end_span(self) -> None:
        """End the most recent synchronous span on the calling thread."""
        if not self.enabled:
            return
        with self._lock:
            self._append(
                {
                    "ph": "E",
                    "ts": _timestamp_us(),
                    "pid": self.process_id,
                    "tid": self._regular_tid(),
                }
            )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str = "driver",
        args: Optional[dict[str, Any]] = None,
    ) -> Iterator[None]:
        """Context manager for a synchronous span."""
        self.start_span(name, category=category, args=args)
        try:
            yield
        finally:
            self.end_span()

    def instant(
        self,
        name: str,
        *,
        category: str = "driver",
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit an instant event on the calling thread."""
        if not self.enabled:
            return
        event: dict[str, Any] = {
            "name": name,
            "cat": category,
            "ph": "i",
            "s": "t",
            "ts": _timestamp_us(),
            "pid": self.process_id,
            "tid": self._regular_tid(),
        }
        if args:
            event["args"] = args
        with self._lock:
            self._append(event)

    def counter(
        self,
        name: str,
        values: dict[str, int | float],
        *,
        category: str = "driver",
    ) -> None:
        """Emit a Perfetto counter event."""
        if not self.enabled:
            return
        with self._lock:
            self._append(
                {
                    "name": name,
                    "cat": category,
                    "ph": "C",
                    "ts": _timestamp_us(),
                    "pid": self.process_id,
                    "tid": self._regular_tid(),
                    "args": values,
                }
            )

    def start_async_span(
        self,
        name: str,
        span_id: str,
        *,
        track_name: str,
        category: str = "rollout",
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Start a complete-event span on a named virtual track."""
        if not self.enabled:
            return
        pid = (
            self.virtual_process_id
            if self.virtual_process_name is not None
            else self.process_id
        )
        with self._lock:
            if span_id in self._open_async_spans:
                raise ValueError(f"Trace span {span_id!r} is already open")
            self._open_async_spans[span_id] = {
                "name": name,
                "cat": category,
                "ts": _timestamp_us(),
                "pid": pid,
                "tid": self.virtual_tid(track_name),
                "args": dict(args or {}),
            }

    def end_async_span(
        self, span_id: str, *, args: Optional[dict[str, Any]] = None
    ) -> None:
        """End an async span and emit it as a complete event."""
        if not self.enabled:
            return
        end = _timestamp_us()
        with self._lock:
            start = self._open_async_spans.pop(span_id, None)
            if start is None:
                raise ValueError(f"Trace span {span_id!r} is not open")
            event_args = start.pop("args")
            event_args.update(args or {})
            self._append(
                {
                    **start,
                    "ph": "X",
                    "dur": max(0, end - start["ts"]),
                    "args": event_args,
                }
            )

    @contextmanager
    def async_span(
        self,
        name: str,
        span_id: str,
        *,
        track_name: str,
        category: str = "rollout",
        args: Optional[dict[str, Any]] = None,
    ) -> Iterator[None]:
        """Context manager for a span on a virtual async track."""
        self.start_async_span(
            name,
            span_id,
            track_name=track_name,
            category=category,
            args=args,
        )
        try:
            yield
        finally:
            self.end_async_span(span_id)

    def finalize_open_spans(self) -> None:
        """Close unfinished async spans so partial traces remain loadable."""
        if not self.enabled:
            return
        with self._lock:
            open_span_ids = list(self._open_async_spans)
        for span_id in open_span_ids:
            self.end_async_span(span_id, args={"incomplete": True})

    def events(self) -> list[dict[str, Any]]:
        """Return a shallow copy of all collected trace events."""
        with self._lock:
            return [dict(event) for event in self._events]

    def collect_trace(self, timing: bool = False) -> int | list[dict[str, Any]]:
        """Ray-friendly endpoint used by the driver during trace merge."""
        if timing:
            return _timestamp_us()
        self.finalize_open_spans()
        return self.events()


def _shift_timestamps(events: Sequence[dict[str, Any]], offset_us: int) -> None:
    for event in events:
        if "ts" in event:
            event["ts"] += offset_us


def save_trace(
    local_events: Sequence[dict[str, Any]],
    *,
    actors: Sequence[Any] = (),
    output_path: Optional[str | os.PathLike[str]] = None,
) -> Optional[Path]:
    """Merge local and Ray-actor events and write one Perfetto JSON file.

    A midpoint round-trip estimate aligns each actor's monotonic clock with the
    driver's. Actor collection is best effort so a failed actor cannot hide the
    useful driver-side portion of the trace.
    """
    if not trace_enabled():
        return None

    merged = [dict(event) for event in local_events]
    if actors:
        import ray

        for actor in actors:
            try:
                local_start = _timestamp_us()
                actor_time = int(ray.get(actor.collect_trace.remote(timing=True)))
                local_end = _timestamp_us()
                offset_us = ((local_start + local_end) // 2) - actor_time
                actor_events = [
                    dict(event)
                    for event in ray.get(actor.collect_trace.remote(timing=False))
                ]
                _shift_timestamps(actor_events, offset_us)
                merged.extend(actor_events)
            except Exception as error:
                warnings.warn(
                    f"Could not collect Perfetto events from Ray actor: {error}",
                    stacklevel=2,
                )

    merged.sort(key=lambda event: (event.get("ts", -1), event.get("ph", "")))
    path = Path(output_path or os.getenv(_TRACE_FILE_ENV, _DEFAULT_TRACE_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as trace_file:
        json.dump(merged, trace_file, separators=(",", ":"))
    temporary_path.replace(path)
    print(
        f"Perfetto trace written to {path.resolve()} ({len(merged)} events)",
        flush=True,
    )
    return path
