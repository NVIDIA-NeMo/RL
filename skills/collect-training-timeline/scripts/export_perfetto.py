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
"""Export collected timelines as a Chrome/Perfetto trace for ui.perfetto.dev.

Reads the files NRL_COLLECT_TIMELINE_FIELDS=1 writes and lays them out as tracks:

  * Trainer / driver: trainer phases, with overlapping spans on parallel lanes.
  * Rollout lifecycle: buffered / consumed markers and a buffered-rollout counter.
  * Rollouts: one lane per rollout (SWE agent tasks only) plus a second lane of
    that rollout's model_call phases, cut at the engine milestones.
  * Generation replicas: in-flight counters (client, queued, running) per replica.

Timestamps from hosts listed in clock_calibration.jsonl are shifted onto the
driver's clock. A host with no calibration keeps its raw timestamps; the trace
metadata names them. Standard library only.
"""

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any, Optional

from timeline_common import (
    breakdown,
    inflight_series,
    load_jsonl,
    milestone,
    read_clock_offsets,
)

TRAINER_PID = 1
LIFECYCLE_PID = 2
ROLLOUT_PID = 3
REPLICA_PID_BASE = 10


def trainer_lanes(events: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Assign each span a lane so that spans on one lane nest or follow each other."""
    lanes: list[list[float]] = []
    placed = []
    for event in sorted(events, key=lambda e: (e["ts"], -e.get("dur", 0))):
        start, end = event["ts"], event["ts"] + event.get("dur", 0)
        lane = len(lanes)
        for index, stack in enumerate(lanes):
            while stack and stack[-1] <= start:
                stack.pop()
            if not stack or end <= stack[-1]:
                lane = index
                break
        if lane == len(lanes):
            lanes.append([])
        lanes[lane].append(end)
        placed.append((lane, event))
    return placed


class Trace:
    """Accumulates trace events and the track names that go with them."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._processes: set[int] = set()
        self._threads: set[tuple[int, int]] = set()

    def lane(self, pid: int, tid: int, process: str, thread: str, sort: int) -> None:
        if pid not in self._processes:
            self._processes.add(pid)
            self.events.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": process},
                }
            )
            self.events.append(
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": pid,
                    "tid": 0,
                    "args": {"sort_index": pid},
                }
            )
        if (pid, tid) not in self._threads:
            self._threads.add((pid, tid))
            self.events.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": thread},
                }
            )
            self.events.append(
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": sort},
                }
            )

    def add(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def clock_for(event: dict[str, Any]) -> Optional[str]:
    """The clock a span's own timestamp was taken on."""
    args = event.get("args", {})
    return args.get("clock_id") or args.get("hostname") or event.get("clock_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--training-timeline", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--rollout-lifecycle", type=Path)
    parser.add_argument("--clock-calibration", type=Path)
    parser.add_argument("--generation-topology", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="*.json or *.json.gz"
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=64,
        help="rollout lanes to draw, earliest first; 0 draws every rollout",
    )
    parser.add_argument(
        "--no-model-phases",
        action="store_true",
        help="leave out the per-call engine phase lane (much smaller output)",
    )
    parser.add_argument("--counter-bin-seconds", type=float, default=1.0)
    args = parser.parse_args()

    beside = args.training_timeline.parent
    lifecycle_path = args.rollout_lifecycle or beside / "rollout_lifecycle.jsonl"
    calibration_path = args.clock_calibration or beside / "clock_calibration.jsonl"
    topology_path = args.generation_topology or beside / "generation_topology.jsonl"

    training = load_jsonl(args.training_timeline)
    lifecycle = load_jsonl(lifecycle_path)
    topology = load_jsonl(topology_path)
    reference, offsets = read_clock_offsets(calibration_path)
    rollout_files = (
        sorted(args.rollout_root.rglob("timeline.jsonl")) if args.rollout_root else []
    )
    rollouts = {path: load_jsonl(path) for path in rollout_files}
    rollouts = {path: events for path, events in rollouts.items() if events}

    uncalibrated: collections.Counter[str] = collections.Counter()

    def corrected(ts: float, clock: Optional[str]) -> float:
        """Shift ts onto the reference clock, if its offset was measured."""
        if clock in offsets:
            return ts - offsets[clock]
        if offsets:
            uncalibrated[str(clock)] += 1
        return ts

    # Rollouts: the step each belongs to (from the lifecycle) and when it started.
    step_of = {
        str(r["rollout_id"]): r["target_display_step"]
        for r in lifecycle
        if "rollout_id" in r and r.get("target_display_step") is not None
    }
    ordered = []
    for path, events in rollouts.items():
        rollout_id = str(
            next(
                (
                    e["args"]["rollout_id"]
                    for e in events
                    if e.get("args", {}).get("rollout_id")
                ),
                path.parent.name,
            )
        )
        start = min(corrected(e["ts"], clock_for(e)) for e in events)
        ordered.append((step_of.get(rollout_id, 10**9), start, rollout_id, events))
    ordered.sort(key=lambda item: item[:3])
    total_rollouts = len(ordered)
    if args.max_rollouts:
        ordered = ordered[: args.max_rollouts]

    # Trace zero is the earliest corrected timestamp of anything drawn.
    stamps = [corrected(e["ts"], clock_for(e)) for e in training if "ts" in e]
    stamps += [corrected(r["ts"], r.get("clock_id")) for r in lifecycle if "ts" in r]
    stamps += [
        corrected(e["ts"], clock_for(e)) for *_, events in ordered for e in events
    ]
    if not stamps:
        raise SystemExit("no events to export")
    base = min(stamps)

    trace = Trace()
    dropped_phases = 0

    # Trainer / driver.
    spans = [
        {**e, "ts": corrected(e["ts"], clock_for(e)) - base}
        for e in training
        if e.get("ph") == "X"
    ]
    for lane, event in trainer_lanes(spans):
        tid = 1 + lane
        trace.lane(
            TRAINER_PID,
            tid,
            "Trainer / driver",
            "Training phases"
            if lane == 0
            else f"Training phases · parallel lane {lane}",
            tid,
        )
        trace.add({**event, "pid": TRAINER_PID, "tid": tid})

    # Rollout lifecycle.
    buffered = 0
    for record in sorted(lifecycle, key=lambda r: r.get("ts", 0)):
        if "ts" not in record:
            continue
        ts = corrected(record["ts"], record.get("clock_id")) - base
        is_buffered = record.get("event") == "rollout_buffered"
        buffered += 1 if is_buffered else -1
        trace.lane(LIFECYCLE_PID, 1, "Rollout lifecycle", "Buffered and consumed", 1)
        trace.add(
            {
                "ph": "i",
                "s": "t",
                "name": record.get("event", "rollout"),
                "cat": "rollout_lifecycle",
                "pid": LIFECYCLE_PID,
                "tid": 1,
                "ts": ts,
                "args": {k: v for k, v in record.items() if k not in ("ts", "event")},
            }
        )
        trace.add(
            {
                "ph": "C",
                "name": "buffered rollouts",
                "pid": LIFECYCLE_PID,
                "ts": ts,
                "args": {"value": buffered},
            }
        )

    # Rollouts: the spans as recorded, and each model_call cut at its milestones.
    for index, (step, _start, rollout_id, events) in enumerate(ordered):
        label = rollout_id if step == 10**9 else f"step {step} · {rollout_id}"
        tid = 2 * index + 1
        trace.lane(ROLLOUT_PID, tid, "Rollouts (SWE agent)", label, tid)
        for event in events:
            if event.get("ph") != "X":
                continue
            ts = corrected(event["ts"], clock_for(event)) - base
            trace.add({**event, "pid": ROLLOUT_PID, "tid": tid, "ts": ts})
            if args.no_model_phases or event.get("name") != "model_call":
                continue
            if event.get("args", {}).get("status") != "ok":
                continue
            cut = breakdown({**event, "ts": ts + base})
            if cut is None:
                continue
            trace.lane(
                ROLLOUT_PID,
                tid + 1,
                "Rollouts (SWE agent)",
                f"{label} · engine phases",
                tid + 1,
            )
            for name, start, end, klass in cut[1]:
                # Segments that span two hosts can come out negative (clock skew).
                if end < start:
                    dropped_phases += 1
                    continue
                trace.add(
                    {
                        "ph": "X",
                        "name": name,
                        "cat": "model_call_phase",
                        "pid": ROLLOUT_PID,
                        "tid": tid + 1,
                        "ts": start - base,
                        "dur": end - start,
                        "args": {"class": klass, "turn": event["args"].get("turn")},
                    }
                )

    # Generation replicas: in-flight counters from the drawn rollouts' calls.
    bin_us = args.counter_bin_seconds * 1e6
    per_replica: dict[Any, dict[str, list[tuple[float, float]]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for *_, events in ordered:
        for event in events:
            call = event.get("args", {})
            if event.get("name") != "model_call" or call.get("status") != "ok":
                continue
            replica = call.get("generation_replica_idx", 0)
            start = corrected(event["ts"], clock_for(event))
            per_replica[replica]["client in flight"].append(
                (start, start + event["dur"])
            )
            queued, scheduled, last = (
                milestone(call, n) for n in ("queued", "first_scheduled", "last_token")
            )
            if queued and scheduled:
                per_replica[replica]["engine queued"].append((queued, scheduled))
            if scheduled and last:
                per_replica[replica]["engine running"].append((scheduled, last))
    hosts: dict[Any, list[str]] = collections.defaultdict(list)
    for record in topology:
        hosts[record.get("replica_idx")].append(
            f"{record.get('role')}@{record.get('host')}"
        )
    for replica, series in sorted(per_replica.items(), key=lambda kv: str(kv[0])):
        pid = REPLICA_PID_BASE + (replica if isinstance(replica, int) else 0)
        process = f"Generation replica {replica}" + (
            f" · {', '.join(hosts[replica])}" if hosts.get(replica) else ""
        )
        trace.lane(pid, 1, process, "in flight", 1)
        origin = min(start for intervals in series.values() for start, _ in intervals)
        for name, intervals in series.items():
            for b, value in enumerate(inflight_series(intervals, bin_us, t0=origin)):
                trace.add(
                    {
                        "ph": "C",
                        "name": name,
                        "pid": pid,
                        "ts": origin + b * bin_us - base,
                        "args": {"value": round(value, 3)},
                    }
                )

    trace.events.sort(key=lambda e: (e["ph"] != "M", e.get("ts", 0)))
    document = {
        "traceEvents": trace.events,
        "displayTimeUnit": "ms",
        "metadata": {
            "base_timestamp_us": base,
            "reference_clock_id": reference,
            "calibrated_clocks": sorted(offsets),
            "uncalibrated_clocks": dict(uncalibrated),
            "rollouts_drawn": len(ordered),
            "rollouts_total": total_rollouts,
            "model_call_phases_dropped_negative": dropped_phases,
            "note": "Engine milestones inside model_call phases and the engine counters keep "
            "their raw timestamps: the request records no owning clock id.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", encoding="utf-8") as stream:  # type: ignore[operator]
        json.dump(document, stream, separators=(",", ":"), allow_nan=False)
    print(
        f"wrote {args.output}: {len(trace.events)} events, "
        f"{len(ordered)} of {total_rollouts} rollouts, "
        f"{len(offsets)} calibrated clocks, {len(uncalibrated)} uncalibrated"
    )


if __name__ == "__main__":
    main()
