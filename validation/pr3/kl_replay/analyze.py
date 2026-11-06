# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check actual token workloads, native timing and ungated GPU kernel overlap."""

import argparse
import bisect
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from validation.pr3.check_kernel_overlap import merge


def capture_summary(directory: Path, version: int) -> dict[str, Any]:
    packet_directory = directory / f"step_{version}"
    if not packet_directory.exists():
        packet_directory = directory  # Early runner-smoke exports used one directory.
    packets = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(packet_directory.glob("chunk-*.pt"))
    ]
    events = [
        json.loads(line)
        for line in (directory / "controller.jsonl").read_text().splitlines()
    ]
    ids = [sample for packet in packets for sample in packet["ids"]]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("Expected exactly 32 unique real rollout rows")
    hashes = Counter()
    lengths = []
    group_sizes = Counter()
    group_max_tokens: dict[str, int] = {}
    for packet in packets:
        if packet["trainer_version"] != version:
            raise ValueError("Mixed weight versions")
        for row, length in zip(
            packet["data"]["input_ids"], packet["data"]["input_lengths"]
        ):
            hashes[
                hashlib.sha256(row[: int(length)].numpy().tobytes()).hexdigest()
            ] += 1
            lengths.append(int(length))
        for tag, mask in zip(packet["tags"], packet["data"]["token_mask"]):
            group = tag["group_id"]
            group_sizes[group] += 1
            group_max_tokens[group] = max(
                group_max_tokens.get(group, 0), int(mask.sum())
            )
    if len(group_sizes) != 8 or set(group_sizes.values()) != {4}:
        raise ValueError(f"Expected eight complete groups of four: {group_sizes}")
    first_train = min(
        event["time_ns"]
        for event in events
        if event["event"] == "training_start" and event["step"] == version
    )
    generation = [
        event["time_ns"]
        for event in events
        if event["event"] == "generation_end" and event["step"] == version
    ]
    if len(generation) != 8:
        raise ValueError(
            f"Expected eight completed prompt groups, got {len(generation)}"
        )
    metrics = {
        event["prefix"]: event["metrics"]
        for event in events
        if event["event"] == "metrics" and event["step"] == version + 1
    }
    return {
        "version": version,
        "sample_count": len(ids),
        "chunk_sizes": [len(packet["ids"]) for packet in packets],
        "token_hash_multiset": dict(hashes),
        "sorted_sequence_lengths": sorted(lengths),
        "group_max_generated_tokens": sorted(group_max_tokens.values()),
        "metrics": metrics,
        "post_backward_capture_seconds": sum(
            event["seconds"]
            for event in events
            if event["event"] == "capture_complete" and event["step"] == version
        ),
        "all_generation_finished_before_first_train_rpc": max(generation)
        <= first_train,
        "first_train_minus_last_generation_seconds": (first_train - max(generation))
        / 1e9,
    }


def kernel_summary(directory: Path) -> dict[str, Any]:
    intervals: dict[str, list[tuple[int, int]]] = {"training": [], "generation": []}
    sources = {}
    for role in intervals:
        for path in directory.glob(f"{role}_gpu-*.trace.json"):
            trace = json.loads(path.read_text())
            base = trace["baseTimeNanoseconds"]
            windows = sorted(
                (event["ts"], event["ts"] + event["dur"])
                for event in trace["traceEvents"]
                if event.get("name") == f"pr3_{role}_gpu" and event.get("dur", 0) > 0
            )
            starts = [start for start, _ in windows]
            kernels = []
            for event in trace["traceEvents"]:
                if event.get("cat") != "kernel" or event.get("dur", 0) <= 0:
                    continue
                index = bisect.bisect_right(starts, event["ts"]) - 1
                if index >= 0 and event["ts"] < windows[index][1]:
                    kernels.append(
                        (
                            base + round(event["ts"] * 1000),
                            base + round((event["ts"] + event["dur"]) * 1000),
                        )
                    )
            sources[path.name] = {"kernels": len(kernels), "clock_base_ns": base}
            intervals[role].extend(kernels)
    if not all(intervals.values()):
        raise ValueError(f"Missing actual training/generation kernels: {sources}")
    training, generation = (
        merge(intervals[role]) for role in ("training", "generation")
    )
    left = right = count = overlap_ns = 0
    while left < len(training) and right < len(generation):
        duration = min(training[left][1], generation[right][1]) - max(
            training[left][0], generation[right][0]
        )
        if duration > 0:
            overlap_ns += duration
            count += 1
        if training[left][1] < generation[right][1]:
            left += 1
        else:
            right += 1
    return {
        "sources": sources,
        "actual_overlap_observed": overlap_ns > 0,
        "overlap_intervals": count,
        "overlap_ms": overlap_ns / 1e6,
        "first_training_minus_last_generation_ms": (training[0][0] - generation[-1][1])
        / 1e6,
        "note": "Ungated profiling with rank0 synchronization changes timing. Cross-node timestamps use host wall clocks; sub-ms intersections need an independent clock-error bound. Not a throughput benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    stream = capture_summary(root / "stream-capture", 210)
    full = capture_summary(root / "full-capture", 210)
    profile = capture_summary(root / "profile-capture", 211)
    warm_stream = capture_summary(root / "stream-capture", 211)
    warm_full = capture_summary(root / "full-capture", 211)
    same = stream["token_hash_multiset"] == full["token_hash_multiset"]
    report = {
        "native_stream": stream,
        "native_full": full,
        "native_stream_after_warmup": warm_stream,
        "native_full_after_warmup": warm_full,
        "same_warm_native_token_workload": warm_stream["token_hash_multiset"]
        == warm_full["token_hash_multiset"],
        "same_native_token_workload": same,
        "native_timing_scope": "One cold resumed update per mode, with CPU capture overhead. Only compare as a paired token-workload measurement when hashes match; not steady end-to-end throughput.",
        "profiled_stream": profile,
        "kernels": kernel_summary(root / "trace"),
        "worker_replay": json.loads((root / "replay/diagnostic.json").read_text()),
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "same_native_token_workload": same,
                "kernels": report["kernels"],
                "worker_timing": report["worker_replay"]["timing_medians_seconds"],
            },
            indent=2,
        )
    )
