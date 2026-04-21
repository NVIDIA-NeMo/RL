#!/usr/bin/env python3
"""Plot scalar GRPO ablations from TensorBoard and JSONL artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

SIZE_GUIDANCE = {
    event_accumulator.TENSORS: 0,
    event_accumulator.SCALARS: 0,
}

DEFAULT_METRICS = [
    "validation/accuracy",
    "train/loss",
    "train/advantages/mean",
    "train/dg_gate_mean",
    "train/dg_gate_positive_mean",
    "train/dg_gate_negative_mean",
    "train/dg_gate_spread",
    "train/dg_surprisal_mean",
]


@dataclass
class RunData:
    label: str
    run_dir: str
    metrics: dict[str, list[tuple[int, float]]]


def summarize_metrics(
    metrics: dict[str, list[tuple[int, float]]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name, points in metrics.items():
        if not points:
            continue
        final_step, final_value = points[-1]
        best_step, best_value = max(points, key=lambda item: item[1])
        worst_step, worst_value = min(points, key=lambda item: item[1])
        summary[name] = {
            "num_points": len(points),
            "final_step": int(final_step),
            "final_value": float(final_value),
            "best_step": int(best_step),
            "best_value": float(best_value),
            "worst_step": int(worst_step),
            "worst_value": float(worst_value),
        }
    return summary


def load_tensorboard_scalars(run_dir: str) -> dict[str, dict[int, float]]:
    event_files = sorted(
        glob.glob(
            os.path.join(run_dir, "**", "tensorboard", "**", "events*tfevents*"),
            recursive=True,
        )
    )
    merged: dict[str, dict[int, float]] = {}

    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            event_file, size_guidance=SIZE_GUIDANCE
        )
        accumulator.Reload()
        for metric_name in accumulator.scalars.Keys():
            metric_steps = merged.setdefault(metric_name, {})
            for scalar in accumulator.Scalars(metric_name):
                metric_steps[scalar.step] = scalar.value

    return merged


def load_reward_stats(run_dir: str) -> dict[str, dict[int, float]]:
    step_files = sorted(
        glob.glob(os.path.join(run_dir, "**", "train_data_step*.jsonl"), recursive=True)
    )
    reward_mean: dict[int, float] = {}
    reward_std: dict[int, float] = {}

    for path in step_files:
        step_str = os.path.splitext(os.path.basename(path))[0].replace("train_data_step", "")
        step = int(step_str)
        rewards: list[float] = []
        with open(path) as handle:
            for line in handle:
                sample = json.loads(line)
                value = sample.get("rewards")
                if isinstance(value, list):
                    rewards.extend(float(x) for x in value)
                elif value is not None:
                    rewards.append(float(value))
        if rewards:
            mean = sum(rewards) / len(rewards)
            var = sum((x - mean) ** 2 for x in rewards) / len(rewards)
            reward_mean[step] = mean
            reward_std[step] = var**0.5

    return {
        "train/reward_mean": reward_mean,
        "train/reward_std": reward_std,
    }


def normalize_metrics(metric_map: dict[str, dict[int, float]]) -> dict[str, list[tuple[int, float]]]:
    return {
        name: sorted((int(step), float(value)) for step, value in points.items())
        for name, points in metric_map.items()
    }


def parse_run_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    label, run_dir = value.split("=", 1)
    return label, run_dir


def plot_metric(metric_name: str, runs: list[RunData], out_dir: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for run in runs:
        points = run.metrics.get(metric_name, [])
        if not points:
            continue
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        plt.plot(steps, values, marker="o", linewidth=1.8, markersize=3, label=run.label)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(metric_name)
    plt.xlabel("step")
    plt.ylabel(metric_name.split("/")[-1])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    filename = metric_name.replace("/", "__") + ".png"
    plt.savefig(os.path.join(out_dir, filename), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run specification as LABEL=RUN_DIR",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric to plot. Can be passed multiple times.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for plots and merged JSON")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_to_plot = args.metric or DEFAULT_METRICS
    runs: list[RunData] = []
    merged_output: dict[str, dict[str, list[tuple[int, float]]]] = {}
    summary_output: dict[str, dict[str, dict[str, float | int]]] = {}

    for label, run_dir in args.run:
        metric_map = load_tensorboard_scalars(run_dir)
        metric_map.update(load_reward_stats(run_dir))
        normalized = normalize_metrics(metric_map)
        runs.append(RunData(label=label, run_dir=run_dir, metrics=normalized))
        merged_output[label] = normalized
        summary_output[label] = summarize_metrics(normalized)

    with open(os.path.join(args.out_dir, "merged_metrics.json"), "w") as handle:
        json.dump(merged_output, handle, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary_output, handle, indent=2)
    with open(os.path.join(args.out_dir, "summary.tsv"), "w") as handle:
        handle.write(
            "run\tmetric\tnum_points\tfinal_step\tfinal_value\tbest_step\tbest_value\tworst_step\tworst_value\n"
        )
        for label, metric_summary in summary_output.items():
            for metric_name, stats in sorted(metric_summary.items()):
                handle.write(
                    "\t".join(
                        [
                            label,
                            metric_name,
                            str(stats["num_points"]),
                            str(stats["final_step"]),
                            str(stats["final_value"]),
                            str(stats["best_step"]),
                            str(stats["best_value"]),
                            str(stats["worst_step"]),
                            str(stats["worst_value"]),
                        ]
                    )
                    + "\n"
                )

    for metric_name in metrics_to_plot:
        plot_metric(metric_name, runs, args.out_dir)


if __name__ == "__main__":
    main()
