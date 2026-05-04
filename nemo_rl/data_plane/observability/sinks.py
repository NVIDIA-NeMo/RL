# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""MetricsSink ABC + built-in implementations.

A sink is the *output* side of the observability layer — the middleware
calls ``record(event)`` for each TQ op; the sink decides what to do with
it (accumulate in memory, emit a structured log line, push to wandb, …).
Sinks are pluggable so users can opt in without changing the middleware.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class MetricsSink(ABC):
    """Receives per-op events and exposes a cumulative snapshot."""

    @abstractmethod
    def record(self, event: dict[str, Any]) -> None:
        """Called once per data-plane operation.

        ``event`` keys:
          * ``op``: ``"put" | "get" | "register" | "clear" | "get_meta"``
          * ``partition_id``: str
          * ``n_keys``: int (0 if not applicable)
          * ``n_bytes``: int (0 if not applicable)
          * ``wall_ms``: float
          * ``status``: ``"ok" | "error" | "timeout"``
          * ``fields``: list[str] | None  (for inspection of what crossed)
        """

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Cumulative flat metrics dict, ready for wandb / TB logging.

        Keys are namespaced under ``data_plane/<op>/<metric>``.
        """

    def close(self) -> None:
        """Flush pending state. Default: no-op."""


class InMemorySink(MetricsSink):
    """Accumulates counters and timing in process memory.

    Use as the default — no external deps, cheap, lets the trainer
    snapshot once per step and emit through whatever logger it already
    uses (wandb, mlflow, tensorboard, plain-print).
    """

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "count": 0.0,
                "bytes": 0.0,
                "wall_ms": 0.0,
                "errors": 0.0,
            }
        )

    def record(self, event: dict[str, Any]) -> None:
        op = str(event.get("op", "unknown"))
        s = self._stats[op]
        s["count"] += 1
        s["bytes"] += float(event.get("n_bytes", 0))
        s["wall_ms"] += float(event.get("wall_ms", 0.0))
        if event.get("status") != "ok":
            s["errors"] += 1

    def snapshot(self) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for op, s in self._stats.items():
            for k, v in s.items():
                flat[f"data_plane/{op}/{k}"] = v
            wall_s = s["wall_ms"] / 1000.0
            if wall_s > 0:
                flat[f"data_plane/{op}/throughput_MB_s"] = (
                    s["bytes"] / 1e6 / wall_s
                )
        return flat


class LogSink(MetricsSink):
    """Emits one structured log line per event at DEBUG; INFO for errors.

    Use when you want a per-op trace in the run log without depending on
    wandb. Output goes through Python's stdlib logger; the calling
    framework controls log level and destination.
    """

    def __init__(self, logger_name: str = "nemo_rl.data_plane") -> None:
        self._log = logging.getLogger(logger_name)
        self._mem = InMemorySink()  # also accumulate so snapshot() works

    def record(self, event: dict[str, Any]) -> None:
        self._mem.record(event)
        if event.get("status") == "ok":
            self._log.debug("dp_op %s", event)
        else:
            self._log.info("dp_op_error %s", event)

    def snapshot(self) -> dict[str, Any]:
        return self._mem.snapshot()


def build_sink(name: str | None) -> MetricsSink:
    """Resolve a config-supplied sink name to a concrete sink."""
    if name in (None, "", "memory"):
        return InMemorySink()
    if name == "log":
        return LogSink()
    raise ValueError(
        f"unknown observability sink: {name!r}. "
        f"Supported: 'memory' (default), 'log'."
    )
