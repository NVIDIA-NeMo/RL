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
group's reservation, admission, seal and release, each training step and
weight sync, and each checkpoint published or restored. Every line carries the
wall-clock time ``t`` and a ``run`` ID that is new for each process, so runs
that resume into the same log directory stay apart.

Events are queued as they happen and a background thread appends them, so
writing never blocks the controller's event loop; a run that is killed loses
only the events not yet written, typically the last few milliseconds' worth.
Rows match across runs by
``prompt_id`` (the dataset index) and ``generation_index``; group and attempt
IDs are new in every run.

With token capture, each refit lists how many tokens every running call had
generated when it paused, and each published group counts its trained tokens
by the weight version each call began on, listing on its own only the calls a
refit paused.
``tokens_by_version`` splits each call at those refits, and
``batch_age_by_step`` turns that into each step's batch age: the
token-weighted number of weight updates between generating a token and
training on it. Run this module on a trace file to print it per step.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import uuid
import warnings
from collections.abc import Iterable, Iterator
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
        # Model calls a refit paused, until their group is published.
        self._cut_calls: set[str] = set()
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

    def emit_refit_cut(self, weight_version: int, calls: list[list[Any]]) -> None:
        """Record how far each running captured call got before a refit paused it."""
        if self._file is None:
            return
        self._cut_calls.update(call_id for call_id, _ in calls)
        self.emit("refit_cut", weight_version=weight_version, calls=calls)

    def emit_group_published(
        self, group_id: str, call_versions: list[tuple[str, int, int]]
    ) -> None:
        """Record a published group's trained tokens by the version each call began on.

        Only the calls a refit paused are listed on their own, since only they
        are split across versions; the rest are summed per version.
        """
        if self._file is None:
            return
        tokens_by_begin_version: dict[int, int] = {}
        cut_calls = []
        for call_id, version, tokens in call_versions:
            if call_id in self._cut_calls:
                self._cut_calls.discard(call_id)
                cut_calls.append([call_id, version, tokens])
            else:
                tokens_by_begin_version[version] = (
                    tokens_by_begin_version.get(version, 0) + tokens
                )
        self.emit(
            "group_published",
            group_id=group_id,
            tokens_by_version=tokens_by_begin_version,
            calls=cut_calls,
        )

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


def read_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the events of a trace file in the order they were written."""
    with open(path, encoding="utf-8") as trace_file:
        for line in trace_file:
            yield json.loads(line)


def tokens_by_version(events: Iterable[dict[str, Any]]) -> dict[str, dict[int, int]]:
    """Each published group's generated tokens, by the weight version that made them.

    A call's tokens count under the version it began on until a refit paused
    it, and under the version that refit synced from there on. ``refit_cut``
    events say how many tokens each running call had at the pause, and the next
    ``weights_synced`` says which version it continued on. A group's other calls
    arrive already summed by the version they began on.
    """
    # Call ID -> [(tokens generated before the refit, version after it), ...]
    cuts: dict[str, list[tuple[int, int]]] = {}
    pending: list[tuple[str, int]] = []
    groups: dict[str, dict[int, int]] = {}
    for event in events:
        kind = event["event"]
        if kind == "trace_opened":
            # A refit the previous process did not finish never synced.
            pending = []
        elif kind == "refit_cut":
            pending.extend((call_id, tokens) for call_id, tokens in event["calls"])
        elif kind == "weights_synced":
            for call_id, tokens in pending:
                cuts.setdefault(call_id, []).append((tokens, event["weight_version"]))
            pending = []
        elif kind == "group_published":
            # JSON keys are strings.
            counts = {
                int(version): tokens
                for version, tokens in event.get("tokens_by_version", {}).items()
            }
            for call_id, version, total in event["calls"]:
                done = 0
                for tokens, next_version in cuts.pop(call_id, []):
                    tokens = min(tokens, total)
                    if tokens > done:
                        counts[version] = counts.get(version, 0) + tokens - done
                        done = tokens
                    version = next_version
                if total > done:
                    counts[version] = counts.get(version, 0) + total - done
            groups[event["group_id"]] = counts
    return groups


def batch_age_by_step(events: Iterable[dict[str, Any]]) -> dict[int, float]:
    """Each step's token-weighted age: policy updates from generating to training.

    A step a resumed run trained again keeps the later run's value.
    """
    events = list(events)
    groups = tokens_by_version(events)
    ages: dict[int, float] = {}
    for event in events:
        if event["event"] != "step_trained":
            continue
        weighted = total = 0
        for group_id in event["group_ids"]:
            for version, tokens in groups.get(group_id, {}).items():
                weighted += tokens * (event["trained_version"] - version)
                total += tokens
        if total:
            ages[event["step"]] = weighted / total
    return ages


def main() -> None:
    parser = argparse.ArgumentParser(description="Print each step's batch age.")
    parser.add_argument("trace", type=Path, help=f"a {ROLLOUT_TRACE_FILENAME}")
    trace_path = parser.parse_args().trace
    for step, age in sorted(batch_age_by_step(read_events(trace_path)).items()):
        print(f"step {step}: batch age {age:.3f}")


if __name__ == "__main__":
    main()
