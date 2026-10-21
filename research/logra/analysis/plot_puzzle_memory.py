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

"""Plot paired sliding-puzzle training-memory profiles from exported scalar events."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    runs = defaultdict(dict)
    with args.csv.open() as stream:
        for row in csv.DictReader(stream):
            if row["memory_gib"]:
                key = (row["method"], int(row["seed"]))
                step = int(row["step"])
                value = float(row["memory_gib"])
                if step in runs[key] or not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"Invalid or duplicate memory sample: {key}, {step}"
                    )
                runs[key][step] = value
    methods = ("Dense AdamW", "LoGRA")
    seeds = (42, 43)
    expected = set(range(1, args.steps + 1))
    if set(runs) != {(method, seed) for method in methods for seed in seeds}:
        raise ValueError("Expected paired Dense/LoGRA runs for seeds 42 and 43")
    for key, values in runs.items():
        if set(values) != expected:
            raise ValueError(f"Incomplete run: {key}; steps={sorted(values)}")
    steps = sorted(expected)
    matrices = {
        method: np.array([[runs[method, seed][s] for s in steps] for seed in seeds])
        for method in methods
    }
    summary = {}
    for method, matrix in matrices.items():
        run_means = matrix.mean(axis=1)
        summary[method] = {
            "seeds": list(seeds),
            "updates_per_seed": args.steps,
            "mean_update_peak_gib": float(run_means.mean()),
            "seed_std_gib": float(run_means.std(ddof=1)),
            "per_seed_mean_gib": run_means.tolist(),
            "maximum_logged_mean_gpu_peak_gib": float(matrix.max()),
        }
    saving = 100 * (
        1
        - summary["LoGRA"]["mean_update_peak_gib"]
        / summary["Dense AdamW"]["mean_update_peak_gib"]
    )
    summary["saving_percent"] = saving
    summary["definition"] = (
        "Per-update peak PyTorch allocated memory, averaged across policy-training GPUs; includes all updates. Error bars show sample standard deviation across two seed means. Excludes rollout GPUs."
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"Dense AdamW": "#41658A", "LoGRA": "#B65D3C"}
    upper = max(float(matrix.max()) for matrix in matrices.values()) * 1.24
    fig, ax = plt.subplots(figsize=(5.3, 4.1), layout="constrained")
    means = [summary[m]["mean_update_peak_gib"] for m in methods]
    stds = [summary[m]["seed_std_gib"] for m in methods]
    ax.bar(
        range(2),
        means,
        yerr=stds,
        capsize=4,
        color=list(colors.values()),
        width=0.57,
        edgecolor="white",
    )
    ax.set(
        xticks=range(2),
        xticklabels=methods,
        ylabel="Mean update peak / training GPU (GiB)",
        title="Sliding puzzle · Qwen2.5-7B-Instruct",
        ylim=(0, upper),
    )
    for index, mean in enumerate(means):
        ax.text(
            index,
            mean + max(means) * 0.04,
            f"{mean:.2f} GiB",
            ha="center",
            fontweight="bold",
        )
    ax.text(
        0.97,
        0.96,
        f"{saving:.1f}% memory saving",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=colors["LoGRA"],
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E0E4E9", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#88929D")
        spine.set_linewidth(0.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=240, bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
