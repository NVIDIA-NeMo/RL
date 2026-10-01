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

"""Plot fresh native-GRPO measurements exported as method,seed,step,memory_gib,accuracy.

Accuracy is a fraction. Missing evaluations stay missing (never filled with zero).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurements = defaultdict(dict)
    with args.csv.open() as stream:
        for row in csv.DictReader(stream):
            for metric in ("memory_gib", "accuracy"):
                if row[metric]:
                    key = (row["method"], metric, int(row["step"]))
                    seed = int(row["seed"])
                    if seed in measurements[key]:
                        raise ValueError(f"Duplicate measurement: {key}, seed={seed}")
                    measurements[key][seed] = float(row[metric])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
    colors = {"Dense AdamW": "#41658A", "LoGRA": "#B65D3C"}
    for ax, metric, title, ylabel in zip(
        axes,
        ("memory_gib", "accuracy"),
        ("Training memory", "Held-out performance"),
        ("Mean per-GPU update peak (GiB)", "GSM8K accuracy (%)"),
    ):
        for method, color in colors.items():
            steps = sorted(k[2] for k in measurements if k[:2] == (method, metric))
            if not steps:
                continue
            values = [list(measurements[method, metric, s].values()) for s in steps]
            factor = 100 if metric == "accuracy" else 1
            means = np.array([np.mean(v) for v in values]) * factor
            std = (
                np.array([np.std(v, ddof=1) if len(v) > 1 else 0 for v in values])
                * factor
            )
            ax.plot(
                steps,
                means,
                color=color,
                lw=2,
                label=method,
                marker="o" if metric == "accuracy" else None,
                markersize=3.5,
            )
            ax.fill_between(
                steps, means - std, means + std, color=color, alpha=0.13, linewidth=0
            )
        ax.set(title=title, xlabel="GRPO training step", ylabel=ylabel)
        ax.grid(axis="y", color="#D8DDE2", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        if metric == "memory_gib":
            ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
