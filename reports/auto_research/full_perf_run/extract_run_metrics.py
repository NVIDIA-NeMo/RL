#!/usr/bin/env python3
"""Extract per-step and finalizer metrics from an SC full-perf driver log.

Usage: extract_run_metrics.py <ray-driver.log> [--json]

Parses the SingleControllerActor per-step block:
    • Total step time: <sec>s
    step_metrics={'loss': ..., 'grad_norm': ..., ...}
    train step N/M  trainer_v=V  lag=L
plus FinalizerActor rejection lines and gym retry-storm counters, into one
comparable record per run for the capture-vs-nocapture report.
"""

import ast
import json
import re
import sys
from collections import Counter

STEP_TIME_RE = re.compile(r"Total step time: ([0-9.]+)s")
STEP_METRICS_RE = re.compile(r"step_metrics=(\{.*\})\s*$")
STEP_HDR_RE = re.compile(r"train step (\d+)/(\d+)\s+trainer_v=(\d+)\s+lag=(\d+)")
FINALIZE_REJECT_RE = re.compile(r"finalize: rollout \S+ rejected \(([^)]+)\)")
RETRY_STORM_RE = re.compile(r"Hit (\d+) global `(\w+)`")
WANDB_RE = re.compile(r"https://wandb\.ai/[\w./-]+")

# step_metrics keys copied into the per-step record when present.
METRIC_KEYS = (
    "loss",
    "grad_norm",
    "token_mult_prob_error",
    "reward",
    "total_reward",
    "mean_reward",
    "probs_ratio",
    "num_valid_samples",
    "mean_length_reward_low",
    "mean_reward_low",
)


def parse(path: str) -> dict:
    steps: list[dict] = []
    pending: dict = {}
    reject_reasons: Counter = Counter()
    storm_peak: Counter = Counter()
    wandb_url = None

    with open(path, errors="replace") as fh:
        for line in fh:
            if wandb_url is None:
                m = WANDB_RE.search(line)
                if m:
                    wandb_url = m.group(0)
            m = STEP_TIME_RE.search(line)
            if m:
                pending["step_time_s"] = float(m.group(1))
                continue
            m = STEP_METRICS_RE.search(line)
            if m:
                try:
                    metrics = ast.literal_eval(m.group(1))
                    for key in METRIC_KEYS:
                        if key in metrics:
                            pending[key] = metrics[key]
                except (ValueError, SyntaxError):
                    pass
                continue
            m = STEP_HDR_RE.search(line)
            if m:
                pending.update(
                    step=int(m.group(1)),
                    max_steps=int(m.group(2)),
                    trainer_v=int(m.group(3)),
                    lag=int(m.group(4)),
                )
                steps.append(pending)
                pending = {}
                continue
            m = FINALIZE_REJECT_RE.search(line)
            if m:
                reject_reasons[m.group(1)] += 1
                continue
            m = RETRY_STORM_RE.search(line)
            if m:
                count, err = int(m.group(1)), m.group(2)
                storm_peak[err] = max(storm_peak[err], count)

    return {
        "log": path,
        "wandb": wandb_url,
        "steps": steps,
        "steady_state_step_time_s": (
            sum(s["step_time_s"] for s in steps[1:]) / (len(steps) - 1)
            if len(steps) > 1
            else None
        ),
        "finalizer_rejects": dict(reject_reasons),
        "gym_retry_storm_peaks": dict(storm_peak),
    }


def main() -> None:
    record = parse(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(record, indent=2, default=str))
        return
    print(f"log: {record['log']}")
    print(f"wandb: {record['wandb']}")
    for s in record["steps"]:
        print(
            f"  step {s.get('step')}/{s.get('max_steps')} "
            f"time={s.get('step_time_s')}s lag={s.get('lag')} "
            f"loss={s.get('loss'):.4f} grad_norm={s.get('grad_norm', float('nan')):.4f}"
            if "loss" in s
            else f"  step {s.get('step')} time={s.get('step_time_s')}s (no metrics line)"
        )
    print(f"steady-state step time (2+): {record['steady_state_step_time_s']}")
    print(f"finalizer rejects: {record['finalizer_rejects']}")
    print(f"gym retry storm peaks: {record['gym_retry_storm_peaks']}")


if __name__ == "__main__":
    main()
