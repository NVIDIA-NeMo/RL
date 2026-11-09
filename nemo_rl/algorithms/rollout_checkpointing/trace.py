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

"""A per-rollout event log for checking what a stop and restore cost.

With ``rollout_checkpointing.trace``, the controller appends one JSON line per
event to ``rollout_trace.jsonl`` in the log directory: every rollout attempt's
dispatch, seal, refusal, abandonment (with its reason) and resume, each prompt
group's reservation, admission, seal and discard, each training step and
weight sync, and each checkpoint published or restored. Every line carries the
wall-clock time ``t`` and a ``run`` ID that is new for each process, so runs
that resume into the same log directory stay apart.

Events are queued as they happen and a background thread appends them, so
writing never blocks the controller's event loop; a run that is killed loses
only the events not yet written, typically the last few milliseconds' worth.
Rows match across runs by
``prompt_id`` (the dataset index) and ``generation_index``; group and attempt
IDs are new in every run.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import IO, Any, Optional

ROLLOUT_TRACE_FILENAME = "rollout_trace.jsonl"


class RolloutTrace:
    """Append rollout events to a JSONL file off the caller's thread, or drop them."""

    def __init__(self, path: Optional[Path]) -> None:
        """Open ``path`` for appending; ``None`` makes every call a no-op.

        Args:
            path: The JSONL file to append to, or ``None`` to disable tracing.
        """
        self._run_id = uuid.uuid4().hex
        self._file: Optional[IO[str]] = None
        # Created only when enabled: a disabled trace holds no lock, so the
        # controller class that defaults to one still pickles for Ray.
        self._lines: Optional[queue.Queue[Optional[str]]] = None
        self._writer: Optional[threading.Thread] = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8")
            self._lines = queue.Queue()
            self._writer = threading.Thread(
                target=self._write_lines, name="rollout-trace-writer", daemon=True
            )
            self._writer.start()
            self.emit("trace_opened", pid=os.getpid())

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def emit(self, event: str, **fields: Any) -> None:
        """Queue one event, stamped with the current wall-clock time."""
        if self._lines is None:
            return
        record = {"t": time.time(), "run": self._run_id, "event": event, **fields}
        self._lines.put(json.dumps(record, default=str) + "\n")

    def flush(self) -> None:
        """Wait until every queued event has been written."""
        if self._lines is not None:
            self._lines.join()

    def close(self) -> None:
        """Write every queued event, then stop the writer and close the file."""
        if self._file is None:
            return
        assert self._lines is not None and self._writer is not None
        self._lines.put(None)
        self._writer.join()
        self._file.close()
        self._file = None
        self._lines = None

    def _write_lines(self) -> None:
        """Append queued events in batches until ``close`` queues ``None``."""
        assert self._file is not None and self._lines is not None
        trace_file, lines = self._file, self._lines
        failed = False
        while True:
            batch = [lines.get()]
            while True:
                try:
                    batch.append(lines.get_nowait())
                except queue.Empty:
                    break
            done = batch[-1] is None
            if not failed:
                try:
                    trace_file.write("".join(line for line in batch if line))
                    trace_file.flush()
                except OSError as error:
                    # Tracing must never stop the run it observes.
                    failed = True
                    warnings.warn(
                        f"rollout trace stopped writing: {error}", stacklevel=1
                    )
            for _ in batch:
                lines.task_done()
            if done:
                return
