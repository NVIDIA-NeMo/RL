# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add local full-batch runs to the frozen PR3 exploratory curve snapshots."""

import csv
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from validation.pr3.plot_comparison import (
    KEYS,
    METRICS,
    REPO,
    STEP,
    committed_log_rows,
    merge_records,
    trailing_mean,
)

LOCAL = REPO.parent / "RL-pr4963"
FRESH = REPO.parent / "RL-pr4963-qwen35-la1-fresh600/validation/fresh600"
V1 = "v1 sync (shuangy)"
FULL = "full-batch LA0 (shuangy)"
STREAM = "streaming LA0 (yuekaiz)"
LABELS = {
    "V1 synchronous": V1,
    "SC full-batch LA0": FULL,
    "SC streaming LA0 (exploratory)": STREAM,
}
COLORS = {V1: "#4c78a8", FULL: "#ef8a23", STREAM: "#119b83"}


@dataclass
class Series:
    label: str
    rows: list[dict]
    source: dict
    color: str
    style: str = "-"


def file_proof(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def normalize(row: dict) -> dict:
    return {
        key
        if key == STEP or key.startswith(("train/", "timing/"))
        else f"train/{key}": value
        for key, value in row.items()
        if isinstance(value, (int, float))
    }


def existing(model: str) -> list[Series]:
    prefix = "validation/pr3/gemma_e2b" if model == "gemma" else "validation/pr3"
    base = REPO / prefix / "results/curve-comparison"
    summary = json.loads((base / "comparison-summary.json").read_text())
    grouped = {label: [] for label in LABELS}
    with (base / "comparison.csv").open() as file:
        for row in csv.DictReader(file):
            grouped[row["run"]].append(
                {
                    key: float(value)
                    for key, value in row.items()
                    if key in KEYS and value
                }
            )
    result = []
    for old, label in LABELS.items():
        source = {
            **summary["sources"][old],
            "snapshot_csv": file_proof(base / "comparison.csv"),
            "snapshot_generated_utc": summary["generated_utc"],
            "scope": "Existing observed records; no extrapolation or new training.",
        }
        result.append(
            Series(
                label,
                merge_records(grouped[old]),
                source,
                COLORS[label],
                "--" if label == V1 else "-",
            )
        )
    return result


def native_snapshot(
    base: Path, jobs: list[str], out: Path, *, after: int = 0
) -> tuple[list[dict], dict]:
    canonical, lineage, replaced = {}, [], []
    cache = out / ".native-snapshots"
    cache.mkdir(exist_ok=True)
    for job in jobs:
        path = base / f"slurm/{job}-logs/ray-driver.log"
        if not path.exists():
            continue
        # Parse and hash the same immutable bytes even while a driver appends.
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
        snapshot = cache / f"{job}.log"
        snapshot.write_bytes(data)
        rows = [row for row in committed_log_rows(snapshot) if row[STEP] > after]
        for row in rows:
            step = int(row[STEP])
            if step in canonical:
                replaced.append(step)
            canonical[step] = row
        lineage.append(
            {
                "path": str(path),
                "job": job,
                "sha256": hashlib.sha256(data).hexdigest(),
                "snapshot_bytes": len(data),
                "committed_rows_selected": len(rows),
            }
        )
    return [canonical[step] for step in sorted(canonical)], {
        "source": "Native committed optimizer-step receipts",
        "lineage": lineage,
        "replaced_steps": replaced,
        "resume_rule": "Later submitted segment replaces earlier replay of the same step.",
    }


def gemma_local() -> list[Series]:
    base = LOCAL / "validation/gemma/results/analysis-to900"
    result = []
    for lag, color in ((0, "#8b5db7"), (1, "#d65359")):
        path = base / f"la{lag}-canonical-native.jsonl"
        rows = [normalize(json.loads(line)) for line in path.read_text().splitlines()]
        assert [row[STEP] for row in rows] == list(range(1, 901))
        result.append(
            Series(
                f"full-batch LA{lag} (yuekaiz)",
                merge_records(rows),
                {
                    "canonical_jsonl": file_proof(path),
                    "analysis_summary": file_proof(base / "summary.json"),
                    "run_id": f"yuekaiz4963gemma4la{lag}s42w1",
                    "scope": "Completed local full-batch runs; existing canonical replay resolution.",
                },
                color,
                "--" if lag else "-",
            )
        )
    return result


def qwen_local(out: Path) -> list[Series]:
    manifest = FRESH / "submitted.tsv"
    jobs = [
        line.split()[1] for line in manifest.read_text().splitlines() if line.strip()
    ]
    fresh, source = native_snapshot(FRESH, jobs, out)
    source.update(
        {
            "submitted_manifest": file_proof(manifest),
            "run_id": "yuekaiz4963q35la1fixs42s600",
            "scope": "Fresh HF-base initialization; A_log FP32 refit and prefill-conv FP32-product fixes from start. May still be running at snapshot time.",
        }
    )
    old_base = LOCAL / "validation"
    cached = [
        old_base / f"results/{name}"
        for name in ("la1-history.json", "perf-SC_LA1-history.json")
    ]
    old = merge_records(
        [row for path in cached for row in json.loads(path.read_text())]
    )
    old = [row for row in old if row[STEP] <= 600]
    assert [row[STEP] for row in old] == list(range(1, 601))
    tail, tail_source = native_snapshot(old_base, ["21315214"], out, after=600)
    legacy = merge_records(old + tail)
    old_source = {
        "run_id": "yuekaiz4963la1s42",
        "cached_first600": [file_proof(path) for path in cached],
        "continuation": tail_source,
        "scope": "Legacy non-fresh LA1 with known backend/refit issues; resumed checkpoint600 with A_log-only fix, without the conv fix; user-cancelled. This is not a fresh fixed control.",
        "phase_boundary": {"checkpoint": 600, "first_A_log_only_fixed_update": 601},
    }
    return [
        Series(
            "LA1 fresh (yuekaiz; A_log+conv fixed)",
            merge_records(fresh),
            source,
            "#8b5db7",
        ),
        Series(
            "LA1 non-fresh (yuekaiz; legacy issues)",
            legacy,
            old_source,
            "#d65359",
            "--",
        ),
    ]


def plot(model: str) -> None:
    prefix = "validation/pr3/gemma_e2b" if model == "gemma" else "validation/pr3"
    out = REPO / prefix / "results/curve-comparison-expanded"
    out.mkdir(parents=True, exist_ok=True)
    base = existing(model)
    local = gemma_local() if model == "gemma" else qwen_local(out)
    series = [*base[:2], *local, base[2]]
    end = max(int(item.rows[-1][STEP]) for item in series)
    common = min(int(item.rows[-1][STEP]) for item in series)
    shuangy_end = int(base[1].rows[-1][STEP])
    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "generated_utc": snapshot,
        "analyzed_inside_job": os.environ["SLURM_JOB_ID"],
        "analysis_source": file_proof(Path(__file__)),
        "model": model,
        "all_series_common_window": [1, common],
        "sources": {},
        "metrics": {},
        "notes": [
            "Historical stochastic one-seed runs; not fixed-input parity or controlled speedup evidence.",
            "No online validation accuracy. Timing includes logged checkpoint/cold-update costs; excludes queue waits.",
            "Each series stops at its last observed committed update; no extrapolation.",
            "Original Qwen BF16 1% qualification remains FAILED; Gemma fixed-input parity NOT ASSESSED.",
            "Gemma local full-batch curves cover900; Shuangy short runs remain visible as separate sources.",
            "Qwen fresh and legacy LA1 have different fix/recovery histories, explicitly labelled; no silent combination.",
        ],
    }
    for item in series:
        last = int(item.rows[-1][STEP])
        assert [row[STEP] for row in item.rows] == list(range(1, last + 1)), item.label
        assert all(
            all(key in row and math.isfinite(row[key]) for key, _, _ in METRICS)
            for row in item.rows
        ), item.label
        summary["sources"][item.label] = {
            **item.source,
            "first_step": 1,
            "last_step": last,
            "count": len(item.rows),
        }
    fig, axes = plt.subplots(4, 2, figsize=(17, 18), sharex=True)
    for ax, (key, title, ylabel) in zip(axes.flat, METRICS):
        summary["metrics"][key] = {}
        all_positive = True
        for item in series:
            points = {int(row[STEP]): float(row[key]) for row in item.rows}
            all_positive &= min(points.values()) > 0
            ax.plot(
                list(points),
                list(points.values()),
                color=item.color,
                alpha=0.11,
                lw=0.65,
            )
            ax.plot(
                list(points),
                trailing_mean(points, 10),
                color=item.color,
                ls=item.style,
                lw=1.9,
            )
            matched = [value for step, value in points.items() if step <= common]
            summary["metrics"][key][item.label] = {
                "plotted_count": len(points),
                "common_mean": statistics.mean(matched),
                "own_last50_mean": statistics.mean(list(points.values())[-50:]),
                "own_min": min(points.values()),
                "own_max": max(points.values()),
            }
        ax.set(title=title, ylabel=ylabel, xlim=(1, end))
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if shuangy_end < end:
            ax.axvspan(shuangy_end, end, color="#808080", alpha=0.05)
            ax.axvline(shuangy_end, color="#888888", ls=":", lw=0.85)
        if key in ("train/policy_kl_error", "train/grad_norm"):
            if all_positive:
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=0.01 if "grad_norm" in key else 1e-4)
                ax.set_ylabel(ylabel + " (symlog near zero)")
                ax.set_ylim(bottom=0)
        elif key == "timing/train/total_step_time" and model == "qwen":
            ax.set_yscale("log")
            ax.set_ylabel("Seconds / update (log scale)")
        elif key == "train/gen_kl_error":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        if model == "qwen":
            ax.axvline(600, color="#d65359", alpha=0.45, ls=":", lw=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer step")
    title = (
        "Gemma4 E2B IT | batch512 | 8 H100 (4 training + 4 rollout)"
        if model == "gemma"
        else "Qwen3.5-35B-A3B | batch32 | 24 H100 (16 training + 8 rollout)"
    )
    fig.suptitle(
        title + "\nStreaming / local full-batch runs / Shuangy references",
        y=0.994,
        fontsize=16,
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=item.color,
                ls=item.style,
                lw=2,
                label=f"{item.label} [{int(item.rows[-1][STEP])} steps]",
            )
            for item in series
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    notes = f"Faint: raw. Bold: trailing 10-step mean. Shaded: after Shuangy LA0 ends at {shuangy_end}; no curve extrapolated."
    if model == "qwen":
        notes += "\nLegacy LA1: resumed 600, A_log-only fix from 601 (red dotted boundary), no conv fix. Fresh LA1: both fixes from start."
    notes += f"\nSnapshot {snapshot}. Training reward, not held-out accuracy. Different sampled workloads; no numerical/speedup qualification."
    fig.text(0.5, 0.014, notes, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(
        rect=(0.01, 0.065 if model == "qwen" else 0.05, 0.99, 0.885), h_pad=2.0
    )
    fig.savefig(out / "comparison.png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    with (out / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run", *KEYS], lineterminator="\n")
        writer.writeheader()
        for item in series:
            writer.writerows(
                {
                    "run": item.label,
                    **{key: value for key, value in row.items() if key in KEYS},
                }
                for row in item.rows
            )
    (out / "comparison-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": model,
                "output": str(out),
                "counts": {item.label: len(item.rows) for item in series},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    assert os.environ.get("USER") == "root" and os.environ.get("SLURM_JOB_ID"), (
        "Use an allocated container"
    )
    plot("gemma")
    plot("qwen")
