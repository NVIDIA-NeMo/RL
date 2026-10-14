# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Opt-in Chrome JSON traces, importable in Perfetto without a tracing daemon.

All processes share the host's monotonic clock. Each concurrent request gets a
logical track: overlapping requests must not become incorrectly nested slices
on a single event-loop thread. These slices measure wall time, not CPU time.
Only identifiers, sizes, and timings are recorded; no prompt or token contents.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextvars import ContextVar
from typing import Any

G_REQUEST_TRACK: ContextVar[int] = ContextVar("http_profile_track", default=0)


class Trace:
    def __init__(
        self,
        enabled: bool,
        role: str,
        *,
        track_offset: int = 0,
        process_metadata: bool = True,
    ) -> None:
        self.enabled = enabled
        self.role = role
        self.pid = os.getpid()
        self.track_offset = track_offset
        self.process_metadata = process_metadata
        self.events: list[dict[str, Any]] = []
        self.tracks: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self.events = []
        self.tracks = {}
        if self.enabled and self.process_metadata:
            self.events.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": self.pid,
                    "tid": 0,
                    "args": {"name": self.role},
                }
            )

    def track(self, name: str) -> int:
        if not self.enabled:
            return 0
        if name not in self.tracks:
            tid = self.track_offset + len(self.tracks) + 1
            self.tracks[name] = tid
            self.events.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": self.pid,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
        return self.tracks[name]

    def span(self, name: str, start: float, end: float, tid: int, **args: Any) -> None:
        if self.enabled:
            self.events.append(
                {
                    "ph": "X",
                    "cat": "http_profile",
                    "name": name,
                    "ts": start * 1e6,
                    "dur": (end - start) * 1e6,
                    "pid": self.pid,
                    "tid": tid,
                    "args": args,
                }
            )

    def flow(self, name: str, key: str, phase: str, when: float, tid: int) -> None:
        if self.enabled:
            flow_id = hashlib.blake2b(
                f"{name}:{key}".encode(), digest_size=8
            ).hexdigest()
            self.events.append(
                {
                    "ph": phase,
                    "cat": "http_profile",
                    "name": name,
                    "id": "0x" + flow_id,
                    # Bind flow ends to the enclosing slice, rather than the
                    # next slice. Move one nanosecond inside boundary slices.
                    "ts": when * 1e6 + 0.001,
                    "pid": self.pid,
                    "tid": tid,
                    **({"bp": "e"} if phase == "f" else {}),
                }
            )

    def counter(self, name: str, **values: int) -> None:
        if self.enabled:
            self.events.append(
                {
                    "ph": "C",
                    "name": name,
                    "ts": time.perf_counter() * 1e6,
                    "pid": self.pid,
                    "tid": 0,
                    "args": values,
                }
            )


class TraceHTTPMiddleware:
    """ASGI spans include body parsing and sending the final response chunk."""

    def __init__(self, app: Any, trace: Trace) -> None:
        self.app = app
        self.trace = trace
        self.sequence = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"].startswith("/bench/"):
            await self.app(scope, receive, send)
            return
        self.sequence += 1
        headers = dict(scope["headers"])
        key = headers.get(b"x-profile-call", b"").decode()
        tid = self.trace.track(key or f"request-{self.sequence}")
        context = G_REQUEST_TRACK.set(tid)
        started = time.perf_counter()
        first_byte = None
        received = sent = 0
        if key:
            self.trace.flow(
                "HTTP dispatch",
                key,
                "f" if self.trace.role.endswith("backend") else "t",
                started,
                tid,
            )

        async def traced_receive() -> Any:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            return message

        async def traced_send(message: Any) -> None:
            nonlocal first_byte, sent
            if message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
                if first_byte is None and message.get("body"):
                    first_byte = time.perf_counter()
            await send(message)

        try:
            await self.app(scope, traced_receive, traced_send)
        finally:
            ended = time.perf_counter()
            self.trace.span(
                "HTTP request",
                started,
                ended,
                tid,
                received_bytes=received,
                sent_bytes=sent,
                path=scope["path"],
            )
            if first_byte is not None:
                self.trace.span("until first response body", started, first_byte, tid)
            G_REQUEST_TRACK.reset(context)
