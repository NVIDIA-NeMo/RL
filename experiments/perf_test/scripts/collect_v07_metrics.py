"""Collect v0.7 Tier 1 perf_test metrics across runs into one CSV.

Usage (from worktree root):
    python experiments/perf_test/scripts/collect_v07_metrics.py \
        --logs-root logs/perf_test/v07_tier1\
        --output experiments/perf_test/scripts/v07_tier1_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

VARIANT_RE = re.compile(r"v07_(\d+)_([a-z0-9]+)")


@dataclass
class RunMetrics:
    model: str
    variant: str
    median_step_time: float | None
    median_logprob_time: float | None
    tokens_per_sec: float | None
    n_steps_seen: int


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _extract_steps(metrics: dict, key: str) -> list[float]:
    """Pull values for a tensorboard key dumped as {key: {"<step>": value}}."""
    series = metrics.get(key)
    if not isinstance(series, dict):
        return []
    items = sorted(series.items(), key=lambda kv: int(kv[0]))
    # Skip first 2 steps as warmup. Use whatever's left.
    items = items[2:]
    return [float(v) for _, v in items if v is not None]


def parse_run(metrics_json: Path) -> RunMetrics | None:
    parts = metrics_json.parts
    # logs/perf_test/v07_tier1/<model>/<variant>/metrics.json
    try:
        idx = parts.index("v07_tier1")
        model = parts[idx + 1]
        variant = parts[idx + 2]
    except (ValueError, IndexError):
        return None
    try:
        with open(metrics_json) as fh:
            data = json.load(fh)
    except Exception:
        return None

    step_times = _extract_steps(data, "train/global_step_time")
    logprob_times = _extract_steps(data, "train/logprob_step_time")

    tokens_per_sec: float | None = None
    if step_times:
        tokens_series = _extract_steps(data, "train/total_num_tokens")
        if tokens_series and len(tokens_series) == len(step_times):
            tokens_per_sec = statistics.median(
                t / s for t, s in zip(tokens_series, step_times) if s > 0
            )

    return RunMetrics(
        model=model,
        variant=variant,
        median_step_time=_median_or_none(step_times),
        median_logprob_time=_median_or_none(logprob_times),
        tokens_per_sec=tokens_per_sec,
        n_steps_seen=len(step_times) + 2,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[RunMetrics] = []
    for metrics_path in args.logs_root.glob("**/metrics.json"):
        rec = parse_run(metrics_path)
        if rec is not None:
            rows.append(rec)

    rows.sort(key=lambda r: (r.model, r.variant))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "variant",
                "median_step_s",
                "median_logprob_s",
                "tokens_per_sec",
                "steps_seen",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.model,
                    r.variant,
                    f"{r.median_step_time:.4f}" if r.median_step_time else "",
                    f"{r.median_logprob_time:.4f}" if r.median_logprob_time else "",
                    f"{r.tokens_per_sec:.1f}" if r.tokens_per_sec else "",
                    r.n_steps_seen,
                ]
            )

    print(f"[collect] wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
