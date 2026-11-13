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

With token capture, each published group counts its trained tokens by the
weight version that generated them. vLLM keeps running calls across a refit,
so ``RefitCuts`` records how far each running call got when a refit paused it
and splits the call there. ``batch_age_by_step`` turns those counts into each
step's batch age: the token-weighted number of weight updates between
generating a token and training on it. The controller logs the same number as
``batch_age`` every step; run this module on a trace file to print it per step.
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
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import IO, Any, Optional

ROLLOUT_TRACE_FILENAME = "rollout_trace.jsonl"
# Per-group rollout metrics a step sums into its batch age.
BATCH_AGE_TOKENS_METRIC = "batch_age/trained_tokens"
BATCH_AGE_VERSION_SUM_METRIC = "batch_age/token_version_sum"


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


def read_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the events of a trace file in the order they were written."""
    with open(path, encoding="utf-8") as trace_file:
        for line in trace_file:
            yield json.loads(line)


class RefitCuts:
    """Where each running captured call switched weights, until its group publishes.

    vLLM keeps running calls across a refit, so a call's tokens after the refit
    come from the new weights.
    """

    def __init__(self) -> None:
        # Call ID -> (the version its first refit replaced,
        #             [(tokens before a refit, version after it), ...]).
        self._calls: dict[str, tuple[int, list[tuple[int, int]]]] = {}

    def record(
        self, replaced_version: int, new_version: int, calls: Iterable[Sequence[Any]]
    ) -> None:
        """Record how many tokens each call had when a refit paused it."""
        for call_id, tokens in calls:
            _, cuts = self._calls.setdefault(call_id, (replaced_version, []))
            cuts.append((tokens, new_version))

    def forget_before(self, oldest_trainable_version: int) -> None:
        """Drop calls whose group cannot be trained any more.

        A call began on or before the version its first refit replaced, and its
        group no later than the call, so the group is too old to train when
        that version is.
        """
        self._calls = {
            call_id: entry
            for call_id, entry in self._calls.items()
            if entry[0] >= oldest_trainable_version
        }

    def tokens_by_version(
        self, call_versions: Iterable[tuple[str, int, int]]
    ) -> dict[int, int]:
        """A published group's trained tokens by the weight version that made them.

        Args:
            call_versions: Each trained call's ID, the version it began on and
                the tokens it generated.

        Returns:
            Token counts by weight version. A call's tokens count under the
            version it began on until a refit paused it, and under the version
            that refit installed from there on.
        """
        counts: dict[int, int] = {}
        for call_id, version, total in call_versions:
            done = 0
            for tokens, next_version in self._calls.pop(call_id, (version, []))[1]:
                tokens = min(tokens, total)
                if tokens > done:
                    counts[version] = counts.get(version, 0) + tokens - done
                    done = tokens
                version = next_version
            if total > done:
                counts[version] = counts.get(version, 0) + total - done
        return counts


def tokens_by_version(events: Iterable[dict[str, Any]]) -> dict[str, dict[int, int]]:
    """Each published group's trained tokens, by the weight version that made them."""
    return {
        event["group_id"]: {
            # JSON keys are strings.
            int(version): tokens
            for version, tokens in event["tokens_by_version"].items()
        }
        for event in events
        if event["event"] == "group_published"
    }


def batch_age_group_metrics(counts: dict[int, int]) -> dict[str, int]:
    """A published group's share of its step's batch age, as two sums."""
    if not counts:
        return {}
    return {
        BATCH_AGE_TOKENS_METRIC: sum(counts.values()),
        BATCH_AGE_VERSION_SUM_METRIC: sum(
            version * tokens for version, tokens in counts.items()
        ),
    }


def pop_batch_age(
    per_group_metrics: dict[str, list[Any]], trained_version: int
) -> Optional[float]:
    """Remove the step's batch-age sums and return its batch age, if it has tokens.

    Args:
        per_group_metrics: The step's rollout metrics, one value per group.
        trained_version: The policy version the step updates.
    """
    tokens = sum(per_group_metrics.pop(BATCH_AGE_TOKENS_METRIC, []))
    version_sum = sum(per_group_metrics.pop(BATCH_AGE_VERSION_SUM_METRIC, []))
    if not tokens:
        return None
    return trained_version - version_sum / tokens


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
