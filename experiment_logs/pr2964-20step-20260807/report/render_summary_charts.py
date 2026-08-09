from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


class MetricRow(TypedDict):
    family: str
    model: str
    metric: str
    improvement_pct: float


ROWS: tuple[MetricRow, ...] = (
    {
        "family": "Step time",
        "model": "Qwen3-30B-A3B",
        "metric": "E2E",
        "improvement_pct": 3.55,
    },
    {
        "family": "Step time",
        "model": "Qwen3-30B-A3B",
        "metric": "Generation",
        "improvement_pct": 0.16,
    },
    {
        "family": "Step time",
        "model": "Qwen3-30B-A3B",
        "metric": "Policy",
        "improvement_pct": 9.42,
    },
    {
        "family": "Step time",
        "model": "Qwen3-30B-A3B",
        "metric": "LogProb",
        "improvement_pct": 13.61,
    },
    {
        "family": "Step time",
        "model": "Qwen3-235B-A22B",
        "metric": "E2E",
        "improvement_pct": 0.49,
    },
    {
        "family": "Step time",
        "model": "Qwen3-235B-A22B",
        "metric": "Generation",
        "improvement_pct": -0.83,
    },
    {
        "family": "Step time",
        "model": "Qwen3-235B-A22B",
        "metric": "Policy",
        "improvement_pct": 15.69,
    },
    {
        "family": "Step time",
        "model": "Qwen3-235B-A22B",
        "metric": "LogProb",
        "improvement_pct": 14.64,
    },
    {
        "family": "Step time",
        "model": "Nemotron3 Super",
        "metric": "E2E",
        "improvement_pct": 1.37,
    },
    {
        "family": "Step time",
        "model": "Nemotron3 Super",
        "metric": "Generation",
        "improvement_pct": 0.49,
    },
    {
        "family": "Step time",
        "model": "Nemotron3 Super",
        "metric": "Policy",
        "improvement_pct": 44.00,
    },
    {
        "family": "Step time",
        "model": "Nemotron3 Super",
        "metric": "LogProb",
        "improvement_pct": 9.50,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-30B-A3B",
        "metric": "E2E",
        "improvement_pct": 3.85,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-30B-A3B",
        "metric": "Generation",
        "improvement_pct": 0.13,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-30B-A3B",
        "metric": "Policy",
        "improvement_pct": 10.29,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-30B-A3B",
        "metric": "LogProb",
        "improvement_pct": 15.89,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-235B-A22B",
        "metric": "E2E",
        "improvement_pct": 0.86,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-235B-A22B",
        "metric": "Generation",
        "improvement_pct": -0.64,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-235B-A22B",
        "metric": "Policy",
        "improvement_pct": 18.66,
    },
    {
        "family": "Throughput",
        "model": "Qwen3-235B-A22B",
        "metric": "LogProb",
        "improvement_pct": 17.01,
    },
    {
        "family": "Throughput",
        "model": "Nemotron3 Super",
        "metric": "E2E",
        "improvement_pct": -1.82,
    },
    {
        "family": "Throughput",
        "model": "Nemotron3 Super",
        "metric": "Generation",
        "improvement_pct": -2.43,
    },
    {
        "family": "Throughput",
        "model": "Nemotron3 Super",
        "metric": "Policy",
        "improvement_pct": 69.91,
    },
    {
        "family": "Throughput",
        "model": "Nemotron3 Super",
        "metric": "LogProb",
        "improvement_pct": 8.90,
    },
)

MODEL_ORDER = ("Qwen3-30B-A3B", "Qwen3-235B-A22B", "Nemotron3 Super")
METRIC_ORDER = ("E2E", "Generation", "Policy", "LogProb")


def _render_metric_family(data: pd.DataFrame, family: str, output_base: Path) -> None:
    family_data = data[data["family"] == family]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    sns.barplot(
        data=family_data,
        x="model",
        y="improvement_pct",
        hue="metric",
        order=MODEL_ORDER,
        hue_order=METRIC_ORDER,
        palette=sns.color_palette("Paired", n_colors=len(METRIC_ORDER)),
        edgecolor="#192133",
        linewidth=2.0,
        errorbar=None,
        zorder=10,
        ax=axis,
    )
    axis.axhline(y=0, linestyle="--", linewidth=1.1, color="black", zorder=5)
    axis.set_xlabel("Model", fontsize=14)
    axis.set_ylabel("Improvement vs all-to-all baseline (%)", fontsize=14)
    axis.tick_params(axis="x", labelsize=12)
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(True, linestyle="--", dashes=(6, 6), linewidth=1.1, axis="y", zorder=0)
    for side in ("left", "right", "top", "bottom"):
        axis.spines[side].set_linewidth(2.0)
        axis.spines[side].set_color("black")

    handles, labels = axis.get_legend_handles_labels()
    axis.legend().remove()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        figure.savefig(
            output_base.with_suffix(f".{extension}"), bbox_inches="tight", dpi=300
        )
    plt.close(figure)


def render_charts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(ROWS)
    _render_metric_family(data, "Step time", output_dir / "step-time-improvement")
    _render_metric_family(data, "Throughput", output_dir / "throughput-improvement")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render strict HybridEP A/B summary charts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
    )
    args = parser.parse_args()
    render_charts(args.output_dir)


if __name__ == "__main__":
    main()
