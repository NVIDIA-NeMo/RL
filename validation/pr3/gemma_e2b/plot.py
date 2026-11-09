# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot Gemma batch512 against the existing V1 and SC full-batch runs."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.lines import Line2D
from torch.distributed.checkpoint import FileSystemReader

from validation.pr3.plot_comparison import (
    KEYS,
    METRICS,
    STEP,
    committed_log_rows,
    merge_records,
    metric_values,
    trailing_mean,
)

HERE = Path(__file__).resolve().parent
LABEL = "SC streaming LA0 (exploratory)"


def audit_checkpoint(step: int) -> dict:
    """Read only saved scalar state before releasing the debug allocation."""
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    assert step >= 10
    checkpoint = HERE / f"results/streaming-la0/checkpoints/step_{step}"
    for name in (
        "policy/weights",
        "policy/optimizer",
        "train_dataloader.pt",
        "replay_buffer_metadata.pt",
        "rollout_recovery.pt",
        "data_plane",
    ):
        assert (checkpoint / name).exists(), name
    info = json.loads((checkpoint / "training_info.json").read_text())
    assert (
        info["total_steps"]
        == info["trainer_version"]
        == info["sampler_dispatch_index"]
        == step
    )
    assert info["consumed_samples"] == step * 32 and info["current_epoch"] == 0
    base = checkpoint / "policy/optimizer/optim"
    metadata = FileSystemReader(base).read_metadata()
    steps, rates, scheduler = [], [], {}
    for index, storage in metadata.storage_data.items():
        key = index.fqn
        selected = key.startswith("sched.") or (
            key.startswith("optim.param_groups.") and key.endswith((".step", ".lr"))
        )
        if (
            not selected
            or type(metadata.state_dict_metadata[key]).__name__
            != "BytesStorageMetadata"
        ):
            continue
        assert storage.length < 65536
        with (base / storage.relative_path).open("rb") as file:
            file.seek(storage.offset)
            value = torch.load(
                io.BytesIO(file.read(storage.length)),
                map_location="cpu",
                weights_only=False,
            )
        if key.startswith("sched."):
            scheduler[key] = value
        elif key.endswith(".step"):
            steps.append(value)
        else:
            rates.append(value)
    assert steps and all(value == step for value in steps)
    saved_rate = scheduler["sched._last_lr"][0]
    assert rates and all(value == saved_rate for value in rates)
    assert math.isclose(saved_rate, 2e-6, rel_tol=1e-12, abs_tol=0.0)
    assert scheduler["sched.last_epoch"] == step
    shards = len(list(base.glob("*.distcp")))
    assert shards == 4
    proof = {
        "step": step,
        "training_info": info,
        "optimizer_shards": shards,
        "optimizer_step_entries": len(steps),
        "all_saved_optimizer_steps_match": True,
        "optimizer_lr_entries": len(rates),
        "all_saved_lrs_match_scheduler": True,
        "saved_lr": saved_rate,
        "configured_peak_lr": 2e-6,
        "saved_lr_matches_peak_within_float_rounding": True,
        "scheduler": scheduler,
        "metadata_sha256": hashlib.sha256(
            (base / ".metadata").read_bytes()
        ).hexdigest(),
        "analyzed_inside_job": os.environ["SLURM_JOB_ID"],
        "scope": "Saved scalar state only; not live per-rank raw optimizer parity.",
    }
    (HERE / f"results/checkpoint{step}-state.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    return proof


def main() -> None:
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-step", type=int, choices=(150, 650, 900), default=150)
    args = parser.parse_args()
    target = args.max_step
    audit_checkpoint(target)
    output = HERE / "results/curve-comparison"
    output.mkdir(parents=True, exist_ok=True)
    baselines = json.loads((HERE / "results/baselines.json").read_text())
    api = wandb.Api(
        api_key=Path("/lustre/fsw/portfolios/coreai/users/yuekaiz/.wandb_sharon")
        .read_text()
        .strip(),
        overrides={"base_url": "https://api.wandb.ai"},
        timeout=60,
    )
    series, sources = {}, {}
    for label, spec in baselines.items():
        run = api.run(f"ys_fishcool-nvidia/nemo-rl-automodel-sc/{spec['run_id']}")
        rows = [
            {key: value for key, value in row.items() if key in KEYS}
            for row in run.scan_history(page_size=1000)
        ]
        series[label] = merge_records(rows)
        sources[label] = {"run_id": spec["run_id"], "url": run.url, "state": run.state}
        (output / f"{spec['run_id']}-history.json").write_text(
            json.dumps(series[label], indent=2) + "\n"
        )
    logs = (
        sorted((HERE / "slurm").glob("driver*.log"))
        + sorted((HERE / "slurm").glob("resume650-driver-*.log"))
        + sorted((HERE / "slurm").glob("resume900-driver-*.log"))
    )
    raw = [
        row for path in logs for row in committed_log_rows(path) if row[STEP] <= target
    ]
    stream = merge_records(raw)
    assert [row[STEP] for row in stream] == list(range(1, target + 1))
    assert len(raw) == len(stream), "Repeated optimizer-step receipts"
    assert all(row["train/weight_version_lag"] == 0 for row in stream)
    receipts = [
        match
        for path in logs
        for match in re.findall(
            r"train step (\d+)/(?:150|650|900)\s+trainer_v=(\d+)\s+lag=(\d+)",
            path.read_text(errors="replace"),
        )
        if int(match[0]) <= target
    ]
    assert len(receipts) == target
    assert all(step == version and lag == "0" for step, version, lag in receipts)
    series[LABEL] = stream
    common_last = min(target, *(int(rows[-1][STEP]) for rows in series.values()))
    sources[LABEL] = {
        "files": [str(path) for path in logs],
        "url": "https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090gemmala0s42",
        "source": "committed native SC logs",
    }
    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "generated_utc": snapshot,
        "analyzed_inside_job": os.environ["SLURM_JOB_ID"],
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "max_plot_step": target,
        "streaming_last_step": target,
        "matched_window": [1, common_last],
        "sources": sources,
        "metrics": {},
        "streaming_dispatch_counts": sorted(
            {row["train/policy_train_dispatches"] for row in stream}
        ),
        "streaming_dispatch_frequency": dict(
            Counter(int(row["train/policy_train_dispatches"]) for row in stream)
        ),
        "multiple_dispatch_updates": sum(
            row["train/policy_train_dispatches"] > 1 for row in stream
        ),
        "streaming_all_lag_zero": True,
        "streaming_integrity": {
            "all_nonzero_gradients": all(row["train/grad_norm"] > 0 for row in raw),
            "all_nonzero_advantages": all(
                row["train/advantages/max"] > 0 and row["train/advantages/min"] < 0
                for row in raw
            ),
            "valid_sample_frequency": dict(
                Counter(int(row["train/num_valid_samples"]) for row in raw)
            ),
            "scope": "Committed metrics only; not fixed-input or raw optimizer parity.",
        },
        "streaming_diagnostic_peaks": {
            key: [
                {STEP: row[STEP], "value": row[key]}
                for row in sorted(stream, key=lambda row: row[key], reverse=True)[:3]
            ]
            for key in (
                "train/gen_kl_error",
                "train/policy_kl_error",
                "train/grad_norm",
            )
        },
        "notes": [
            "Historical baselines; independent rollouts and one seed.",
            "Fixed-input streaming parity for this Gemma recipe has not been assessed.",
            "The earlier Qwen BF16 1% gate remains FAILED; this is a separate exploratory run.",
            "Step time includes measured checkpoint/cold-step overhead; workloads differ.",
            "No online validation accuracy is recorded.",
            "Readiness threshold16 does not force fixed chunks; some steps may use a full dispatch.",
            "Absent baselines are not extrapolated; common statistics stop at shared observed coverage.",
        ],
    }
    colors = ("#4c78a8", "#ef8a23", "#119b83")
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    for ax, (key, title, ylabel) in zip(axes.flat, METRICS):
        summary["metrics"][key] = {}
        positive = True
        for color, (label, rows) in zip(colors, series.items()):
            last = min(target, int(rows[-1][STEP]))
            points = metric_values(rows, key, first=1, last=last)
            assert list(points) == list(range(1, last + 1)), (label, key)
            positive &= min(points.values()) > 0
            ax.plot(
                list(points), list(points.values()), color=color, alpha=0.17, lw=0.7
            )
            ax.plot(list(points), trailing_mean(points, 10), color=color, lw=1.9)
            matched = metric_values(rows, key, first=1, last=common_last)
            summary["metrics"][key][label] = {
                "plotted_count": len(points),
                "matched_count": len(matched),
                "mean": statistics.mean(matched.values()),
                "first50_mean": statistics.mean(points[i] for i in range(1, 51)),
                "last50_mean": statistics.mean(
                    matched[i] for i in range(common_last - 49, common_last + 1)
                ),
                "last20_mean": statistics.mean(
                    matched[i] for i in range(common_last - 19, common_last + 1)
                ),
                "steps11_to_common_mean": statistics.mean(
                    matched[i] for i in range(11, common_last + 1)
                ),
                "min": min(matched.values()),
                "max": max(matched.values()),
                "run_last20_mean": statistics.mean(
                    points[i] for i in range(last - 19, last + 1)
                ),
                "run_mean": statistics.mean(points.values()),
                "run_last50_mean": statistics.mean(
                    points[i] for i in range(last - 49, last + 1)
                ),
                "run_min": min(points.values()),
                "run_max": max(points.values()),
            }
        ax.set(title=title, ylabel=ylabel, xlim=(1, target))
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if common_last < target:
            ax.axvspan(common_last, target, color="#808080", alpha=0.07)
            ax.axvline(common_last, color="#777777", linestyle=":", linewidth=0.9)
        if key in ("train/policy_kl_error", "train/grad_norm") and positive:
            ax.set_yscale("log")
        if key == "train/gen_kl_error":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer step")
    fig.suptitle(
        "Gemma4 E2B IT: synchronous / full-batch LA0 / streaming LA0\n"
        f"32 prompts x 16 responses | 8 H100 (4 training + 4 rollout) | {target} updates",
        fontsize=15,
        y=0.995,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=color, lw=2, label=label)
            for color, label in zip(colors, series)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.018,
        f"Faint: raw. Solid: trailing 10-step means. Shared coverage1..{common_last}. Snapshot {snapshot}.\n"
        "Exploratory, single seed; Gemma fixed-input streaming parity not assessed. Different rollout workloads.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.938), h_pad=2.1)
    for extension in ("png", "svg"):
        fig.savefig(output / f"comparison.{extension}", dpi=170, facecolor="white")
    plt.close(fig)
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run", *KEYS], lineterminator="\n")
        writer.writeheader()
        for label, rows in series.items():
            writer.writerows(
                {"run": label, **row} for row in rows if row[STEP] <= target
            )
    (output / "comparison-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"output": str(output), "steps": target, "common_last": common_last}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
