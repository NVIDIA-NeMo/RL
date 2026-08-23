from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from research.qwen3_8b_draft_cadence_200step.matrix import WINDOW, build_arms
from research.qwen3_8b_draft_cadence_200step.receipts import validate_arm_receipts


E2E_TPS = "performance/tokens_per_sec_per_gpu"
GEN_TPS = "performance/generation_tokens_per_sec_per_gpu"
STEP_TIME = "timing/train/total_step_time"
GEN_TIME = "timing/train/generation"
REFIT_TIME = "timing/train/prepare_for_generation/total"
ACCEPTED = "train/vllm/spec_num_accepted_tokens"
DRAFTED = "train/vllm/spec_num_draft_tokens"
UPDATE_REQUESTED = "train/draft_schedule/update_requested"
REFIT_REQUESTED = "train/draft_schedule/refit_requested"
REASONS = (
    "always",
    "fixed_interval",
    "none",
    "adaptive_degradation",
    "adaptive_burst",
    "max_interval",
)


def _finite_number(row: dict[str, Any], key: str, step: int) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing finite {key} at step {step}")
    return float(value)


def summarize_history(
    path: Path,
    *,
    start: int = WINDOW[0],
    end: int = WINDOW[1],
    speculative: bool = True,
) -> dict[str, Any]:
    by_step: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        step = row.get("_step")
        if type(step) is int and start <= step <= end:
            by_step[step] = row
    expected = list(range(start, end + 1))
    missing = [step for step in expected if step not in by_step]
    if missing:
        raise ValueError(f"history is missing closed-window steps: {missing}")
    rows = [by_step[step] for step in expected]
    for key in (E2E_TPS, GEN_TPS):
        if any(not isinstance(row.get(key), (int, float)) for row in rows):
            raise ValueError(f"logged throughput metric is required: {key}")
    accepted_total = 0.0
    drafted_total = 0.0
    refit_path_times = [
        _finite_number(row, REFIT_TIME, step) for step, row in zip(expected, rows)
    ]
    if speculative:
        accepted_total = sum(
            _finite_number(row, ACCEPTED, step) for step, row in zip(expected, rows)
        )
        drafted_total = sum(
            _finite_number(row, DRAFTED, step) for step, row in zip(expected, rows)
        )
        if drafted_total <= 0 or accepted_total < 0 or accepted_total > drafted_total:
            raise ValueError("window acceptance counts are invalid")
    window_requested_updates = (
        int(
            sum(
                _finite_number(row, UPDATE_REQUESTED, step)
                for step, row in zip(expected, rows)
            )
        )
        if speculative
        else 0
    )
    window_requested_refits = (
        int(
            sum(
                _finite_number(row, REFIT_REQUESTED, step)
                for step, row in zip(expected, rows)
            )
        )
        if speculative
        else 0
    )
    return {
        "window": {"start": start, "end": end, "count": len(rows)},
        "e2e_tps_per_gpu": fmean(
            _finite_number(row, E2E_TPS, step) for step, row in zip(expected, rows)
        ),
        "generation_tps_per_gpu": fmean(
            _finite_number(row, GEN_TPS, step) for step, row in zip(expected, rows)
        ),
        "mean_step_time_s": fmean(
            _finite_number(row, STEP_TIME, step) for step, row in zip(expected, rows)
        ),
        "mean_generation_time_s": fmean(
            _finite_number(row, GEN_TIME, step) for step, row in zip(expected, rows)
        ),
        "mean_total_refit_time_s": fmean(refit_path_times),
        "total_refit_path_time_s": sum(refit_path_times),
        "acceptance_rate": accepted_total / drafted_total if speculative else None,
        "accepted_tokens": accepted_total,
        "draft_tokens": drafted_total,
        "requested_updates": window_requested_updates,
        "requested_draft_refits": window_requested_refits,
        "window_requested_updates": window_requested_updates,
        "window_requested_draft_refits": window_requested_refits,
    }


def terminal_report_fields(terminal: dict[str, Any]) -> dict[str, int]:
    reason_counts = terminal.get("decision_reason_counts")
    if not isinstance(reason_counts, dict):
        raise ValueError("terminal decision reason counters are absent")
    source_fields = {
        "run_decision_count": "decision_count",
        "run_successful_updates": "successful_updates",
        "run_successful_draft_refits": "successful_draft_refits",
        "run_skipped_updates": "skipped_updates",
        "run_forced_updates": "forced_updates",
    }
    fields: dict[str, int] = {}
    for destination, source in source_fields.items():
        value = terminal.get(source)
        if type(value) is not int or value < 0:
            raise ValueError(f"terminal counter is absent or invalid: {source}")
        fields[destination] = value
    for reason in REASONS:
        value = reason_counts.get(reason)
        if type(value) is not int or value < 0:
            raise ValueError(f"terminal reason counter is absent or invalid: {reason}")
        fields[f"reason_{reason}"] = value
    return fields


def build_report(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in build_arms():
        arm_root = result_root / arm.name
        terminal = validate_arm_receipts(arm_root, arm)
        summary = summarize_history(
            arm_root / "wandb-history.jsonl", speculative=arm.drafter != "none"
        )
        rows.append(
            {
                "arm": arm.name,
                "drafter": arm.drafter,
                "cadence": arm.cadence,
                **summary,
                **terminal,
                **terminal_report_fields(terminal),
            }
        )
    baseline = next(row for row in rows if row["arm"] == "baseline")
    for row in rows:
        row["e2e_tps_speedup_vs_baseline"] = (
            row["e2e_tps_per_gpu"] / baseline["e2e_tps_per_gpu"]
        )
        row["generation_tps_speedup_vs_baseline"] = (
            row["generation_tps_per_gpu"] / baseline["generation_tps_per_gpu"]
        )
        row["step_time_reduction_vs_baseline"] = (
            1.0 - row["mean_step_time_s"] / baseline["mean_step_time_s"]
        )
        row["generation_time_reduction_vs_baseline"] = (
            1.0 - row["mean_generation_time_s"] / baseline["mean_generation_time_s"]
        )
    return rows


def write_report(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    fields = [
        "arm",
        "drafter",
        "cadence",
        "e2e_tps_per_gpu",
        "e2e_tps_speedup_vs_baseline",
        "generation_tps_per_gpu",
        "generation_tps_speedup_vs_baseline",
        "mean_step_time_s",
        "step_time_reduction_vs_baseline",
        "mean_generation_time_s",
        "generation_time_reduction_vs_baseline",
        "mean_total_refit_time_s",
        "total_refit_path_time_s",
        "acceptance_rate",
        "requested_updates",
        "requested_draft_refits",
        "run_decision_count",
        "run_successful_updates",
        "run_successful_draft_refits",
        "run_skipped_updates",
        "run_forced_updates",
        *(f"reason_{reason}" for reason in REASONS),
    ]
    with (output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Qwen3-8B cadence 200-step screening",
        "",
        "Closed analysis window: steps 21-200 (180 steps). Throughput uses canonical logged metrics.",
        "",
        "| Arm | E2E TPS/GPU | vs baseline | Gen TPS/GPU | vs baseline | Step time (s) | Acceptance | Updates | Mean refit path (s) | Total refit path (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acceptance = (
            "n/a" if row["acceptance_rate"] is None else f"{row['acceptance_rate']:.4f}"
        )
        lines.append(
            f"| {row['arm']} | {row['e2e_tps_per_gpu']:.3f} | {row['e2e_tps_speedup_vs_baseline']:.3f}x | "
            f"{row['generation_tps_per_gpu']:.3f} | {row['generation_tps_speedup_vs_baseline']:.3f}x | "
            f"{row['mean_step_time_s']:.3f} | {acceptance} | "
            f"{row['run_successful_updates']} | {row['mean_total_refit_time_s']:.3f} | "
            f"{row['total_refit_path_time_s']:.3f} |"
        )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    write_report(build_report(args.result_root), args.output_root)


if __name__ == "__main__":
    main()
