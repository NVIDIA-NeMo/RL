# /// script
# dependencies = ["matplotlib==3.8.0", "numpy<2"]
# ///
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
"""Overlay the two Inkling 2 Small reference experiments on the nine report panels and a step-timing figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Categorical slots 1 and 2; identity is also carried by the legend and line style.
STYLES = {
    "inkling2-small-sglang-bf16": {"color": "#2a78d6", "linestyle": "-", "marker": "o"},
    "inkling2-mini-sglang-bf16": {"color": "#1a9c6b", "linestyle": "--", "marker": "s"},
}
VAL_TIME = "timing/validation/total_validation_time"
PANELS = [
    (
        "validation",
        "validation/accuracy",
        "AIME 2024 evaluation",
        "Accuracy",
        "percent",
    ),
    (
        "validation",
        "validation/avg_length",
        "AIME mean response length",
        "Tokens",
        "linear",
    ),
    ("training", "train/gen_kl_error", "Generation KL", "KL", "linear"),
    ("training", "train/policy_kl_error", "Policy KL", "KL · log scale", "log"),
    ("training", "train/approx_entropy", "Approximate entropy", "Entropy", "linear"),
    (
        "training",
        "train/mean_gen_tokens_per_sample",
        "Training mean response length",
        "Tokens",
        "linear",
    ),
    ("training", "train/loss", "Training loss", "Loss", "linear"),
    ("training", "train/reward", "Training reward", "Mean reward", "linear"),
    (
        "training",
        "train/truncation_rate",
        "Training truncation rate",
        "Truncated responses",
        "percent",
    ),
]


def style_axis(ax, title: str, ylabel: str, xmax: int) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xmax)
    ax.grid(True, alpha=0.18, color="#64748b")
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#cbd5e1")


def save(fig, directory: Path, stem: str) -> None:
    for suffix in ["png", "svg"]:
        fig.savefig(
            directory / f"{stem}.{suffix}",
            dpi=180,
            facecolor=fig.get_facecolor(),
            metadata={"Creator": "NeMo RL Inkling 2 Mini vs Small results plot"},
        )
    svg = directory / f"{stem}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def plot_step_time(data: dict, directory: Path, xmax: int) -> None:
    """Per-step wall time, with validation steps marked, and AIME validation time."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    for name, exp in data.items():
        style = STYLES[name]
        rows = exp["training"]
        steps = [r["nemo_rl/step"] for r in rows]
        times = [r["timing/train/total_step_time"] for r in rows]
        left.plot(
            steps,
            times,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=exp["label"],
        )
        val = [(s, t) for s, t in zip(steps, times) if s % 10 == 0]
        left.plot(
            [s for s, _ in val],
            [t for _, t in val],
            color=style["color"],
            linestyle="none",
            marker=style["marker"],
            markersize=6,
        )
        vrows = [r for r in exp["validation"] if VAL_TIME in r]
        right.plot(
            [r["nemo_rl/step"] for r in vrows],
            [r[VAL_TIME] for r in vrows],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            marker=style["marker"],
            markersize=5,
            label=exp["label"],
        )
    style_axis(
        left, "Step time (markers: steps that include AIME validation)", "Seconds", xmax
    )
    style_axis(right, "AIME validation time", "Seconds", xmax)
    for ax in (left, right):
        ax.set_ylim(bottom=0)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.90),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Inkling 2 Mini vs Small GRPO · step timing",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.055,
        0.03,
        "Different model sizes and GPU counts (Small: 128 GPUs; Mini: 8 GPUs, both colocated): for reference, not a throughput comparison.",
        color="#475569",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.065, right=0.98, top=0.74, bottom=0.17, wspace=0.22)
    save(fig, directory, "inkling2-mini-vs-small-step-time")


def main() -> None:
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / "inkling2-mini-vs-small.json").read_text())[
        "experiments"
    ]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5))
    fig.patch.set_facecolor("#fcfcfb")
    last_step = max(
        row["nemo_rl/step"] for exp in data.values() for row in exp["training"]
    )
    xmax = 10 * ((last_step + 9) // 10)
    for ax, (group, key, title, ylabel, scale) in zip(axes.flat, PANELS):
        for name, exp in data.items():
            rows = [r for r in exp[group] if r.get(key) is not None]
            style = STYLES[name]
            ax.plot(
                [r["nemo_rl/step"] for r in rows],
                [r[key] for r in rows],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                marker=style["marker"] if group == "validation" else None,
                markersize=5,
                label=exp["label"],
            )
        style_axis(ax, title, ylabel, xmax)
        if scale == "percent":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 1)
        elif scale == "log":
            ax.set_yscale("log")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.935),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Inkling 2 Mini vs Small GRPO · DAPO-Math-17k",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.055,
        0.948,
        "3,072 response / 4,096 total tokens  ·  LR 1e−6  ·  32 prompts × 8 responses  ·  GB200",
        color="#475569",
        fontsize=11,
    )
    fig.text(
        0.055,
        0.018,
        "Raw logged values, no smoothing; each experiment is a chain of 2 h jobs resumed from checkpoints. Same recipe for both.\n"
        "AIME: 256 of the 960 rows, one response each.",
        color="#475569",
        fontsize=10,
        linespacing=1.6,
    )
    fig.subplots_adjust(
        left=0.065, right=0.98, top=0.86, bottom=0.10, wspace=0.30, hspace=0.55
    )
    save(fig, directory, "inkling2-mini-vs-small-nine-panels")
    plot_step_time(data, directory, xmax)


if __name__ == "__main__":
    main()
