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

"""Compare streaming LA0 with the existing V1 and full-batch LA0 runs.

Run on an allocated node. Default mode refreshes the two baseline exports through
the read-only W&B API; --offline uses the previous validation's local exports.
The streaming series comes from committed native log records, including resumes.
"""

import argparse
import ast
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
STEP = "nemo_rl/step"
METRICS = (
    ("train/reward", "Reward", "Mean reward"),
    ("train/mean_gen_tokens_per_sample", "Generation length", "Tokens / sample"),
    ("train/gen_kl_error", "Generation KL diagnostic", "gen_kl_error"),
    ("train/policy_kl_error", "Policy KL diagnostic", "policy_kl_error"),
    ("train/loss", "Policy loss", "Loss"),
    ("train/approx_entropy", "Approximate entropy", "Entropy"),
    ("train/grad_norm", "Gradient norm", "Preclip norm"),
    (
        "timing/train/total_step_time",
        "Step time (checkpoint included)",
        "Seconds / update",
    ),
)
EXTRA_KEYS = (
    "train/weight_version_lag",
    "train/policy_train_dispatches",
    "train/truncation_rate",
    "train/lr",
    "train/global_valid_seqs",
)
KEYS = (STEP, *(metric for metric, _, _ in METRICS), *EXTRA_KEYS)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUMPY_SCALAR = re.compile(r"np\.(?:float64|float32|int64|int32)\(([-+0-9.eE]+)\)")
Record = dict[str, int | float]


@dataclass(frozen=True)
class Baseline:
    label: str
    run_id: str
    timing_cache: str


BASELINES = (
    Baseline("V1 synchronous", "2a121400", "V1"),
    Baseline("SC full-batch LA0", "c926b37d", "SC_LA0"),
)
STREAM_LABEL = "SC streaming LA0 (exploratory)"


def merge_records(rows: list[Record]) -> list[Record]:
    """Merge metric/timing rows at the same one-based optimizer step."""
    merged: dict[int, Record] = {}
    for row in rows:
        step = row.get(STEP)
        if not isinstance(step, (int, float)) or not math.isfinite(step) or step < 1:
            continue
        if int(step) != step:
            raise ValueError(f"Nonintegral optimizer step: {step}")
        merged.setdefault(int(step), {STEP: int(step)}).update(
            {
                key: value
                for key, value in row.items()
                if key in KEYS and isinstance(value, (int, float))
            }
        )
    return [merged[step] for step in sorted(merged)]


def committed_log_rows(path: Path) -> list[Record]:
    """Only admit a metrics row once its following commit/step receipt appears."""
    rows = []
    timing = None
    pending = None
    for line in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", line)
        match = re.search(r"Total step time:\s*([\d.]+)s", line)
        if match:
            timing = float(match.group(1))
        if "step_metrics=" in line:
            expression = NUMPY_SCALAR.sub(r"\1", line.split("step_metrics=", 1)[1])
            pending = ast.literal_eval(expression)
            if timing is not None:
                pending["timing/train/total_step_time"] = timing
            timing = None
        match = re.search(r"train step (\d+)/", line)
        if match and pending is not None:
            rows.append(
                {
                    STEP: int(match.group(1)),
                    **{
                        (key if key.startswith("timing/") else f"train/{key}"): value
                        for key, value in pending.items()
                    },
                }
            )
            pending = None
    return rows


def load_baseline(
    spec: Baseline, *, offline: bool, baseline_repo: Path
) -> tuple[list[Record], dict]:
    """Read full history rather than the W&B sampled history endpoint."""
    if offline:
        paths = [
            baseline_repo / f"session/20261102_233142/{spec.run_id}-history.json",
            baseline_repo / f"validation/results/perf-{spec.timing_cache}-history.json",
        ]
        rows = [row for path in paths for row in json.loads(path.read_text())]
        return merge_records(rows), {
            "source": "cached W&B exports",
            "files": [
                {"path": str(path), "mtime": path.stat().st_mtime} for path in paths
            ],
            "run_id": spec.run_id,
        }

    # W&B is only needed for a live API refresh, not offline plot regeneration.
    import wandb

    key_path = Path("/lustre/fsw/portfolios/coreai/users/yuekaiz/.wandb_sharon")
    api = wandb.Api(
        api_key=key_path.read_text().strip(),
        overrides={"base_url": "https://api.wandb.ai"},
        timeout=60,
    )
    run = api.run(f"ys_fishcool-nvidia/nemo-rl-automodel-sc/{spec.run_id}")
    rows = [
        {key: value for key, value in row.items() if key in KEYS}
        for row in run.scan_history(page_size=1000)
    ]
    config = run.config
    return merge_records(rows), {
        "source": "W&B scan_history (unsampled)",
        "url": run.url,
        "state": run.state,
        "run_id": spec.run_id,
        "grpo": config.get("grpo"),
        "async_rl": config.get("async_rl"),
        "cluster": config.get("cluster"),
        "policy_model": config.get("policy", {}).get("model_name"),
        "checkpoint_save_period": config.get("checkpointing", {}).get("save_period"),
    }


def metric_values(
    rows: list[Record], metric: str, *, first: int, last: int
) -> dict[int, float]:
    return {
        int(row[STEP]): float(row[metric])
        for row in rows
        if first <= row[STEP] <= last and metric in row and math.isfinite(row[metric])
    }


def trailing_mean(points: dict[int, float], window: int) -> list[float]:
    """Use optimizer-step windows so a missing step cannot silently shift smoothing."""
    return [
        statistics.mean(
            value for step, value in points.items() if end - window < step <= end
        )
        for end in points
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, default=REPO.parent / "RL-pr4963")
    parser.add_argument(
        "--output", type=Path, default=REPO / "validation/pr3/results/curve-comparison"
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-step", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    series = {}
    sources = {}
    for spec in BASELINES:
        series[spec.label], sources[spec.label] = load_baseline(
            spec, offline=args.offline, baseline_repo=args.baseline_repo
        )

    paths = sorted((REPO / "validation/pr3/slurm").glob("*-logs/ray-driver.log"))
    paths = [
        path
        for path in paths
        if re.search(r"train step \d+/(?:300|600)\b", path.read_text(errors="replace"))
    ]
    records = [row for path in paths for row in committed_log_rows(path)]
    series[STREAM_LABEL] = merge_records(records)
    if not series[STREAM_LABEL]:
        raise ValueError("No committed streaming updates found")
    sources[STREAM_LABEL] = {
        "source": "committed native SC logs",
        "files": [str(path) for path in paths],
        "url": "https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090explorela0s42",
        "repeated_step_receipts": len(records) - len(series[STREAM_LABEL]),
        "qualification": "exploratory; original 1% BF16 gradient gate remains failed",
    }
    for label, rows in series.items():
        missing = sorted(
            set(range(1, int(rows[-1][STEP]) + 1)) - {int(row[STEP]) for row in rows}
        )
        sources[label]["missing_steps"] = missing
        if missing:
            raise ValueError(f"{label}: missing steps {missing}")
    last_stream_step = min(args.max_step, int(series[STREAM_LABEL][-1][STEP]))
    common_last = min(args.max_step, *(int(rows[-1][STEP]) for rows in series.values()))
    summary = {
        "generated_utc": generated,
        "max_plot_step": args.max_step,
        "streaming_last_step": last_stream_step,
        "matched_window": [1, common_last],
        "steady_timing_window": [11, common_last],
        "sources": sources,
        "metrics": {},
        "notes": [
            "Raw curves and trailing 10 optimizer-step means; one seed per run, no uncertainty estimate.",
            "Steps outside shared observed coverage are shaded; absent series are not extrapolated and summaries use only shared steps.",
            "Step time includes measured checkpoint/first-step overhead; excludes queue/setup time outside the step.",
            "Policy KL uses a logarithmic y-axis so baseline spikes do not flatten normal variation.",
            "Historical baselines: same model, data recipe, batch and resources; not a controlled same-image A/B.",
            "No online validation accuracy was recorded in these runs.",
            "Original 1% gradient gate failed; normalizer placement explains fixed-input BF16 rounding difference.",
        ],
    }
    colors = ["#4c78a8", "#ef8a23", "#119b83"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, METRICS):
        summary["metrics"][metric] = {}
        for color, (label, rows) in zip(colors, series.items()):
            points = metric_values(rows, metric, first=1, last=args.max_step)
            expected_points = min(args.max_step, int(rows[-1][STEP]))
            if len(points) != expected_points:
                raise ValueError(
                    f"{label}: {metric} has {len(points)}/{expected_points} finite plot observations"
                )
            ax.plot(
                list(points),
                list(points.values()),
                color=color,
                alpha=0.17,
                linewidth=0.7,
            )
            ax.plot(list(points), trailing_mean(points, 10), color=color, linewidth=1.9)
            common = metric_values(rows, metric, first=1, last=common_last)
            tail = metric_values(
                rows, metric, first=max(1, common_last - 19), last=common_last
            )
            steady = metric_values(rows, metric, first=11, last=common_last)
            nonfinite = [
                int(row[STEP])
                for row in rows
                if row[STEP] <= args.max_step
                and metric in row
                and not math.isfinite(row[metric])
            ]
            if len(common) != common_last or nonfinite:
                raise ValueError(
                    f"{label}: {metric} incomplete/nonfinite (count={len(common)}, bad={nonfinite})"
                )
            summary["metrics"][metric][label] = {
                "plotted_count": len(points),
                "matched_count": len(common),
                "matched_mean": statistics.mean(common.values()),
                "matched_last20_mean": statistics.mean(tail.values()),
                "steps11_to_common_mean": statistics.mean(steady.values())
                if steady
                else None,
                "matched_min": min(common.values()),
                "matched_max": max(common.values()),
            }
        ax.set(title=title, ylabel=ylabel, xlim=(1, args.max_step))
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if common_last < args.max_step:
            ax.axvspan(common_last, args.max_step, color="#808080", alpha=0.07)
            ax.axvline(common_last, color="#777777", linestyle=":", linewidth=0.9)
        if metric == "train/policy_kl_error":
            if any(
                min(metric_values(rows, metric, first=1, last=args.max_step).values())
                <= 0
                for rows in series.values()
            ):
                raise ValueError("Policy KL log axis requires positive observations")
            ax.set_yscale("log")
            ax.set_ylabel("policy_kl_error (log scale)")
        elif metric == "train/gen_kl_error":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer step")
    handles = [
        Line2D([], [], color=color, lw=2, label=label)
        for color, label in zip(colors, series)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        f"Qwen3.5-35B-A3B: synchronous / full-batch LA0 / streaming LA0\n"
        f"8 prompts x 4 responses | 24 H100 (16 training + 8 rollout) | streaming {last_stream_step}/{args.max_step}",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        f"Faint: raw values. Solid: trailing 10-step means. Shaded: outside shared coverage (1..{common_last}). Snapshot {generated}.\n"
        "Historical baselines; single seed. Step time excludes queue waits. Original 1% BF16 gradient gate remains FAILED.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.938), h_pad=2.1)
    for suffix in ("png", "svg"):
        image_path = args.output / f"comparison.{suffix}"
        fig.savefig(image_path, dpi=170, facecolor="white")
        if suffix == "svg":
            image_path.write_text(
                "\n".join(line.rstrip() for line in image_path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    for label, rows in series.items():
        run_id = sources[label].get("run_id", "yuekaiz5090explorela0s42")
        (args.output / f"{run_id}-history.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
    with (args.output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run", *KEYS], lineterminator="\n")
        writer.writeheader()
        for label, rows in series.items():
            writer.writerows(
                {"run": label, **row} for row in rows if row[STEP] <= args.max_step
            )
    (args.output / "comparison-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "streaming_last_step": last_stream_step,
                "common_last": common_last,
            }
        )
    )


if __name__ == "__main__":
    main()
