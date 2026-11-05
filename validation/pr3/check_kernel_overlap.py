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


"""Compare CUDA kernel timestamps, using Kineto's common absolute clock."""

import argparse
import bisect
import json
from pathlib import Path


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def check(directory: Path) -> dict[str, object]:
    intervals: dict[str, list[tuple[int, int]]] = {"training": [], "generation": []}
    sources: dict[str, dict[str, object]] = {}
    for role in intervals:
        for path in sorted(directory.glob(f"{role}_gpu-*.trace.json")):
            trace = json.loads(path.read_text())
            base = trace["baseTimeNanoseconds"]
            windows = [
                (event["ts"], event["ts"] + event["dur"])
                for event in trace["traceEvents"]
                if event.get("name") == f"pr3_{role}_gpu" and event.get("dur", 0) > 0
            ]
            windows.sort()
            starts = [start for start, end in windows]

            def in_window(timestamp: float) -> bool:
                index = bisect.bisect_right(starts, timestamp) - 1
                return index >= 0 and timestamp < windows[index][1]

            kernels = [
                (
                    base + round(event["ts"] * 1000),
                    base + round((event["ts"] + event["dur"]) * 1000),
                )
                for event in trace["traceEvents"]
                if event.get("cat") == "kernel"
                and event.get("dur", 0) > 0
                and in_window(event["ts"])
            ]
            sources[path.name] = {
                "kernel_count": len(kernels),
                "clock_base_ns": base,
                "first_kernel_ns": min((start for start, end in kernels), default=None),
                "last_kernel_ns": max((end for start, end in kernels), default=None),
            }
            intervals[role].extend(kernels)
            del trace
    assert all(intervals.values()), sources
    training, generation = (
        merge(intervals[role]) for role in ("training", "generation")
    )
    left = right = 0
    overlaps = []
    while left < len(training) and right < len(generation):
        start = max(training[left][0], generation[right][0])
        end = min(training[left][1], generation[right][1])
        if end > start:
            overlaps.append((start, end))
        if training[left][1] < generation[right][1]:
            left += 1
        else:
            right += 1
    assert overlaps, ("No actual training/generation CUDA kernel overlap", sources)
    return {
        "kernel_trace_verified": True,
        "sources": sources,
        "overlapping_kernel_intervals": len(overlaps),
        "total_kernel_overlap_ms": sum(end - start for start, end in overlaps) / 1e6,
        "largest_kernel_overlap_ms": max(end - start for start, end in overlaps) / 1e6,
        "training_kernel_union_ms": sum(end - start for start, end in training) / 1e6,
        "generation_kernel_union_ms": sum(end - start for start, end in generation)
        / 1e6,
        "note": "Private profiler/synchronization probes change timing; these numbers establish overlap, not throughput.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = check(args.directory)
    (args.directory / "kernel-overlap.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
