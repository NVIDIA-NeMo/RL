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
"""Overlay the two Inkling 2 Small reference experiments on the nine report panels."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Categorical slots 1 and 2; identity is also carried by the legend and line style.
STYLES = {
    "sglang-bf16-sync": {"color": "#2a78d6", "linestyle": "-", "marker": "o"},
    "vllm-mxfp8-async": {"color": "#eb6834", "linestyle": "--", "marker": "s"},
}
PANELS = [
    ("validation", "validation/accuracy", "AIME 2024 evaluation", "Accuracy", "percent"),
    ("validation", "validation/avg_length", "AIME mean response length", "Tokens", "linear"),
    ("training", "train/gen_kl_error", "Generation KL", "KL", "linear"),
    ("training", "train/policy_kl_error", "Policy KL", "KL · log scale", "log"),
    ("training", "train/approx_entropy", "Approximate entropy", "Entropy", "linear"),
    ("training", "train/mean_gen_tokens_per_sample", "Training mean response length", "Tokens", "linear"),
    ("training", "train/loss", "Training loss", "Loss", "linear"),
    ("training", "train/reward", "Training reward", "Mean reward", "linear"),
    ("training", "train/truncation_rate", "Training truncation rate", "Truncated responses", "percent"),
]


def main() -> None:
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / "inkling2-small.json").read_text())["experiments"]
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
            rows = [r for r in exp[group] if key in r]
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
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, xmax)
        ax.grid(True, alpha=0.18, color="#64748b")
        ax.spines[["top", "right"]].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#cbd5e1")
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
        "Inkling 2 Small GRPO · DAPO-Math-17k",
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
        "Raw logged values, no smoothing; each experiment is a chain of 2 h jobs resumed from checkpoints.\n"
        "AIME: 256 of the 960 rows, one response each. Step 0 (base model) was evaluated with SGLang BF16 only.",
        color="#475569",
        fontsize=10,
        linespacing=1.6,
    )
    fig.subplots_adjust(
        left=0.065, right=0.98, top=0.86, bottom=0.10, wspace=0.30, hspace=0.55
    )
    for suffix in ["png", "svg"]:
        fig.savefig(
            directory / f"inkling2-small-nine-panels.{suffix}",
            dpi=180,
            facecolor=fig.get_facecolor(),
            metadata={"Creator": "NeMo RL Inkling 2 Small results plot"},
        )
    svg = directory / "inkling2-small-nine-panels.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
