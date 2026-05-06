"""Collect v0.7 Tier 1 perf_test metrics across runs into one CSV.

Parses the per-step "Performance Metrics" blocks emitted by NeMo-RL's GRPO driver
into ray-driver.log, takes the median of each metric over post-warmup steps,
and emits one row per (model, variant) run.

Usage (from worktree root):
    python experiments/perf_test/scripts/collect_v07_metrics.py \
        --logs-root logs/perf_test/v07_tier1 \
        --output experiments/perf_test/scripts/v07_tier1_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

WARMUP_STEPS = 2

# Each "🔍 Performance Metrics:" block has lines like
#   - E2E (Tokens/sec/gpu): 3081.01
# We pull a fixed set of keys from each block.
METRIC_PATTERNS = {
    "e2e_samples_per_sec_per_gpu": re.compile(r"E2E \(Samples/sec/gpu\):\s*([\d.]+)"),
    "e2e_tokens_per_sec_per_gpu": re.compile(r"E2E \(Tokens/sec/gpu\):\s*([\d.]+)"),
    "policy_train_tps_per_gpu": re.compile(
        r"Policy Training \(Tokens/sec/gpu\):\s*([\d.]+)"
    ),
    "logprob_tps_per_gpu": re.compile(
        r"Policy and Reference Logprobs \(Tokens/sec/gpu\):\s*([\d.]+)"
    ),
    "training_wg_tps_per_gpu": re.compile(
        r"Training Worker Group \(Tokens/sec/gpu\):\s*([\d.]+)"
    ),
    "generation_wg_tps_per_gpu": re.compile(
        r"Generation Worker Group \(Tokens/sec/gpu\):\s*([\d.]+)"
    ),
    "e2e_tokens_per_sec": re.compile(r"E2E \(Tokens/sec\):\s*([\d.]+)"),
    "training_tflops": re.compile(r"Training FLOPS:\s*([\d.]+)\s*TFLOPS"),
    "training_mfu_pct": re.compile(
        r"Training Model Floating Point Utilization:\s*([\d.]+)%"
    ),
    "mean_tokens_per_sample": re.compile(r"Mean Total Tokens per Sample:\s*([\d.]+)"),
}

BLOCK_HEADER = re.compile(r"Performance Metrics:")


@dataclass
class RunMetrics:
    model: str
    variant: str
    n_blocks: int
    n_used: int  # blocks after warmup
    medians: dict[str, float]


def parse_log(log_path: Path) -> tuple[int, int, dict[str, float]]:
    """Read a ray-driver.log, return (total_blocks, used_blocks, median_metrics)."""
    text = log_path.read_text(errors="replace")
    blocks = BLOCK_HEADER.split(text)[1:]  # drop preamble before first block
    used = blocks[WARMUP_STEPS:]
    series: dict[str, list[float]] = {k: [] for k in METRIC_PATTERNS}
    for block in used:
        for k, pat in METRIC_PATTERNS.items():
            m = pat.search(block)
            if m:
                try:
                    series[k].append(float(m.group(1)))
                except ValueError:
                    pass
    medians = {k: statistics.median(v) for k, v in series.items() if v}
    return len(blocks), len(used), medians


def find_runs(logs_root: Path):
    """Yield (model, variant, ray_driver_log_path) for each run directory.

    Layout: <logs_root>/<model>/<variant>/<jobid>-logs/ray-driver.log
    Pick the most recent jobid per (model, variant).
    """
    seen: dict[tuple[str, str], Path] = {}
    for log in logs_root.glob("*/*/*-logs/ray-driver.log"):
        model = log.parents[2].name
        variant = log.parents[1].name
        key = (model, variant)
        if key not in seen or log.stat().st_mtime > seen[key].stat().st_mtime:
            seen[key] = log
    for (model, variant), log in seen.items():
        yield model, variant, log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[RunMetrics] = []
    for model, variant, log in find_runs(args.logs_root):
        n_blocks, n_used, medians = parse_log(log)
        if n_blocks == 0:
            continue
        rows.append(RunMetrics(model, variant, n_blocks, n_used, medians))

    rows.sort(key=lambda r: (r.model, r.variant))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "variant", "n_blocks", "n_used"] + list(METRIC_PATTERNS)
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {
                "model": r.model,
                "variant": r.variant,
                "n_blocks": r.n_blocks,
                "n_used": r.n_used,
            }
            for k in METRIC_PATTERNS:
                v = r.medians.get(k)
                row[k] = f"{v:.2f}" if v is not None else ""
            writer.writerow(row)

    print(f"[collect] wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
