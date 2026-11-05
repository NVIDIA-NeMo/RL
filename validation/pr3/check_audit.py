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


"""Validate real smoke receipts; report synchronized GPU regions separately."""

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def check(directory: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (directory / "controller.jsonl").read_text().splitlines()
    ]
    sessions = list(dict.fromkeys(row["session"] for row in rows))
    assert len(sessions) == 2, sessions
    initial, resumed = sessions
    init_rows = [row for row in rows if row["event"] == "init"]
    assert [row["step"] for row in init_rows] == [0, 2], init_rows
    cut = next(row for row in rows if row["event"] == "checkpoint_cut")
    assert cut["ready"] and cut["inflight"] > 0, cut
    cancelled = next(row for row in rows if row["event"] == "cancel_partial")
    abort = [row for row in rows if row["event"].startswith("abort_")]
    assert [row["event"] for row in abort] == ["abort_start", "abort_end"]
    assert all(
        row["step"] == cancelled["step"] and row["version"] == cancelled["version"]
        for row in abort
    )
    committed: dict[str, list[str]] = collections.defaultdict(list)
    dispatch_counts: collections.Counter[tuple[str, int]] = collections.Counter()
    resumed_ids = set()
    for row in rows:
        if row["event"] == "dispatch_start":
            dispatch_counts[row["session"], row["step"]] += 1
            assert all(
                tag["weight_version"] == row["version"] for tag in row["tags"]
            ), row
        if row["event"] == "consume":
            assert len(row["ids"]) == 32 and len(set(row["ids"])) == 32, row
            groups = collections.Counter(
                sample.rsplit("_g", 1)[0] for sample in row["ids"]
            )
            assert len(groups) == 8 and set(groups.values()) == {4}, groups
            committed[row["session"]].extend(row["ids"])
            if row["session"] == resumed:
                resumed_ids.update(groups)
    for session, ids in committed.items():
        assert len(ids) == len(set(ids)), session
    assert any(count > 1 for count in dispatch_counts.values()), dispatch_counts
    resumed_steps = [
        row["step"]
        for row in rows
        if row["event"] == "consume" and row["session"] == resumed
    ]
    assert resumed_steps == [2, 3, 4, 5], resumed_steps
    assert set(cut["ready"]) <= resumed_ids, set(cut["ready"]) - resumed_ids
    assert not set(committed[initial][:64]) & set(committed[resumed])
    worker_rows = [
        json.loads(line)
        for path in directory.glob("worker-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    train = [
        row
        for row in worker_rows
        if row["event"] == "training_gpu" and row["cuda_ms"] > 0
    ]
    generation = [
        row
        for row in worker_rows
        if row["event"] == "generation_gpu" and row["cuda_ms"] > 0
    ]
    intersections = [
        min(left["end_ns"], right["end_ns"]) - max(left["start_ns"], right["start_ns"])
        for left in train
        for right in generation
        if left["cuda_visible_devices"] != right["cuda_visible_devices"]
        and left["start_ns"] < right["end_ns"]
        and right["start_ns"] < left["end_ns"]
    ]
    assert intersections, "No synchronized GPU regions overlap"
    return {
        "sample_receipts_passed": True,
        "resume_steps": resumed_steps,
        "checkpoint_ready_groups": len(cut["ready"]),
        "checkpoint_inflight_groups": cut["inflight"],
        "dispatches_per_step": {
            f"{session}:{step}": count
            for (session, step), count in dispatch_counts.items()
        },
        "overlapping_synchronized_gpu_regions": len(intersections),
        "maximum_region_overlap_ms": max(intersections) / 1e6,
        "kernel_trace_verified": False,
        "overlap_limitation": "Regions synchronize CUDA boundaries and record nonzero event duration; exact concurrent kernels require a device trace.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = check(args.directory)
    (args.directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
