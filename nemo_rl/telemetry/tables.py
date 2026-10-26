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

"""Tables as NeMo Lens log records: one OTel log record per row.

``emit_table`` queues a table and returns; a daemon thread builds the records,
each named by the table and carrying ``rl.table``, ``rl.step`` and one
``rl.row.<column>`` attribute per non-null cell, under the span that was active
when ``emit_table`` was called. ``shutdown_telemetry`` drains the queue, within
its own deadline, before flushing the exporters.

Records go through the OTel logs API, so with no SDK ``LoggerProvider`` (Lens
logs off) every emit is a no-op.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry._logs import get_logger

from nemo_rl.telemetry.metrics import warn_once

TABLE_LOGGER_NAME = "nemo_rl.tables"

_queue: queue.Queue[tuple | None] = queue.Queue()
_thread: threading.Thread | None = None
_lock = threading.Lock()


def emit_table(columns: list[str], rows: list[list[Any]], step: int, name: str) -> None:
    """Queue a table for emission as Lens log records. Never raises or blocks."""
    global _thread
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_drain, name="lens-tables", daemon=True)
            _thread.start()
    _queue.put((columns, rows, step, name, otel_context.get_current()))


def stop_table_emitter(timeout_s: float) -> None:
    """Emit what is queued, waiting at most ``timeout_s``, then stop the thread."""
    global _thread
    with _lock:
        thread, _thread = _thread, None
    if thread is None:
        return
    _queue.put(None)
    thread.join(timeout_s)
    if thread.is_alive():
        warn_once("table_emitter", "Lens table emission did not finish before shutdown")


def _drain() -> None:
    while (item := _queue.get()) is not None:
        try:
            _emit(*item)
        except Exception:
            warn_once("table_emit", "failed to write a table to NeMo Lens logs")


def _emit(
    columns: list[str],
    rows: list[list[Any]],
    step: int,
    name: str,
    ctx: otel_context.Context,
) -> None:
    logger = get_logger(TABLE_LOGGER_NAME)
    keys = [f"rl.row.{column}" for column in columns]
    for row in rows:
        attributes: dict[str, Any] = {"rl.table": name, "rl.step": step}
        attributes.update({k: v for k, v in zip(keys, row) if v is not None})
        logger.emit(event_name=name, attributes=attributes, context=ctx)
