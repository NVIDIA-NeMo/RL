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
"""Render all nine original report metrics for the single Flint reference run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main() -> None:
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / "9dd6d432e98d.json").read_text())
    panels = [
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
        (
            "training",
            "train/approx_entropy",
            "Approximate entropy",
            "Entropy",
            "linear",
        ),
        (
            "training",
            "train/mean_gen_tokens_per_sample",
            "Training mean response length",
            "Tokens",
            "linear",
        ),
        ("training", "train/loss", "Training loss", "Loss", "linear"),
        (
            "training",
            "train/reward",
            "Training reward",
            "Scaled + shaped reward",
            "linear",
        ),
        (
            "training",
            "train/truncation_rate",
            "Training truncation rate",
            "Truncated responses",
            "percent",
        ),
    ]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.patch.set_facecolor("#f8fafc")
    for ax, (group, key, title, ylabel, scale) in zip(axes.flat, panels):
        rows = data[group]
        ax.plot(
            [r["nemo_rl/step"] for r in rows],
            [r[key] for r in rows],
            color="#b45309",
            linewidth=1.8,
            marker="o" if group == "validation" else None,
            markersize=3.5,
        )
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, alpha=0.18, color="#64748b")
        ax.spines[["top", "right"]].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#cbd5e1")
        if scale == "percent":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 1 if group == "training" else 0.6)
        elif scale == "log":
            ax.set_yscale("log")
    fig.suptitle(
        "Flint GRPO · full DAPO-Math-17k",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.055,
        0.945,
        "3,072 response / 4,096 total tokens  ·  shaping ON  ·  LR 1e−7  ·  32 GB200 GPUs  ·  PR 4319",
        color="#475569",
        fontsize=11,
    )
    fig.text(
        0.055,
        0.018,
        "Run 9dd6d432e98d  •  100 training points + 21 evaluations  •  raw logged values, no smoothing\n"
        "AIME: 30 problems × 32 scored responses; step 0 precedes training. Lines connect observations.",
        color="#475569",
        fontsize=10,
        linespacing=1.6,
    )
    fig.subplots_adjust(
        left=0.065, right=0.98, top=0.885, bottom=0.10, wspace=0.30, hspace=0.55
    )
    for suffix in ["png", "svg"]:
        fig.savefig(
            directory / f"9dd6d432e98d-nine-panels.{suffix}",
            dpi=180,
            facecolor=fig.get_facecolor(),
            metadata={"Creator": "NeMo RL Flint results plot"},
        )
    svg = directory / "9dd6d432e98d-nine-panels.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
