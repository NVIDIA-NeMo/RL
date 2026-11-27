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

"""Summarize complete paired runs; reject missing steps rather than comparing partial runs."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-period", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    runs = defaultdict(dict)
    with args.csv.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["method"], int(row["seed"]))
            step = int(row["step"])
            if step in runs[key]:
                raise ValueError(f"Duplicate row: {key}, step={step}")
            runs[key][step] = row
    methods = ("Dense AdamW", "LoGRA")
    summary = {"steps": args.steps, "seeds": args.seeds, "methods": {}}
    for method in methods:
        seeds = []
        for seed in args.seeds:
            rows = runs[method, seed]
            memory = {
                s: float(r["memory_gib"]) for s, r in rows.items() if r["memory_gib"]
            }
            accuracy = {
                s: 100 * float(r["accuracy"]) for s, r in rows.items() if r["accuracy"]
            }
            if set(memory) != set(range(1, args.steps + 1)):
                raise ValueError(
                    f"Incomplete memory measurements for {method}, seed={seed}"
                )
            expected_eval = set(range(0, args.steps + 1, args.eval_period)) | {
                args.steps
            }
            if set(accuracy) != expected_eval:
                raise ValueError(f"Incomplete evaluations for {method}, seed={seed}")
            values = list(memory.values()) + list(accuracy.values())
            if any(not math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite measurement for {method}, seed={seed}")
            seeds.append(
                {
                    "seed": seed,
                    "mean_update_peak_gib": statistics.mean(memory.values()),
                    "max_update_peak_gib": max(memory.values()),
                    "initial_accuracy_percent": accuracy[0],
                    "final_accuracy_percent": accuracy[args.steps],
                    "best_accuracy_percent": max(accuracy.values()),
                    "best_accuracy_step": max(sorted(accuracy), key=accuracy.get),
                }
            )
        summary["methods"][method] = {
            "per_seed": seeds,
            "across_seeds": {
                metric: mean_std([s[metric] for s in seeds])
                for metric in (
                    "mean_update_peak_gib",
                    "max_update_peak_gib",
                    "initial_accuracy_percent",
                    "final_accuracy_percent",
                )
            },
        }
    dense = summary["methods"]["Dense AdamW"]["across_seeds"]
    logra = summary["methods"]["LoGRA"]["across_seeds"]
    summary["memory_saving_percent"] = 100 * (
        1
        - logra["mean_update_peak_gib"]["mean"] / dense["mean_update_peak_gib"]["mean"]
    )
    summary["final_accuracy_difference_percentage_points"] = (
        logra["final_accuracy_percent"]["mean"]
        - dense["final_accuracy_percent"]["mean"]
    )
    summary["memory_definition"] = (
        "Mean across training GPUs of the allocated-memory peak within each update; "
        "then averaged across updates. Excludes rollout GPUs. Not time-averaged memory."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
