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
"""Diagnose rollout performance of a NeMo-RL run, or compare two runs.

Reads the ``metrics.json`` written by ``tests/json_dump_tb_logs.py`` (metric ->
{step: value}) and, optionally, the Ray driver log. Standard library only.

Diagnose one run:

    uv run --no-project python rollout_perf_report.py RUN/metrics.json \
        --log RUN/<jobid>-logs/ray-driver.log

Compare a control and a treatment that ran the same test (paired by step):

    uv run --no-project python rollout_perf_report.py \
        --base BASE/metrics.json --treat TREAT/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path

GEN = "timing/train/generation"
STEP = "timing/train/total_step_time"
VAL = "timing/validation/total_validation_time"
SETUP = "timing/setup/total_setup_time_s"
MAX_GEN = "train/max_gen_tokens_per_sample"
MEAN_GEN = "train/mean_gen_tokens_per_sample"
PROMPT = "train/mean_prompt_length"
TMPE = "train/token_mult_prob_error"
GEN_KL = "train/gen_kl_error"
VAL_ACC = "validation/accuracy"

# Patterns read from the vLLM engine's own log lines. The driver also echoes the
# requested overrides ("Overrides: [...]") and the resolved config
# ("MasterConfig(...)"); those lines show what was asked for, not what ran.
ENGINE_PATTERNS = {
    "enforce_eager": r"enforce_eager=(True|False)",
    "cudagraph_mode": r"'cudagraph_mode': <CUDAGraphMode\.(\w+)",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
    "moe_backend": r"Using (\S+) Unquantized MoE backend",
    "attention_backend": r"Using (\S+) attention backend",
}
CONFIG_PATTERNS = {"max_new_tokens": r"'max_new_tokens': (\d+)"}


def series(metrics: dict, key: str) -> dict[int, float]:
    return {int(step): float(value) for step, value in metrics.get(key, {}).items()}


def med(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else math.nan


def rel_change(before: float, after: float) -> str:
    return "n/a" if before == 0 else f"{(after / before - 1) * 100:+.1f}%"


def scan_log(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    captured = False
    with path.open(errors="replace") as handle:
        for line in handle:
            for name, pattern in CONFIG_PATTERNS.items():
                if name not in found and (match := re.search(pattern, line)):
                    found[name] = match.group(1)
            if "Overrides:" in line or "MasterConfig(" in line:
                continue
            if "Capturing CUDA graphs" in line:
                captured = True
            for name, pattern in ENGINE_PATTERNS.items():
                if name not in found and (match := re.search(pattern, line)):
                    if name == "max_concurrency":
                        found[name] = (
                            f"{match.group(2)}x at {match.group(1)} tokens/request"
                        )
                    else:
                        found[name] = match.group(1)
    found["cuda_graph_capture_seen"] = str(captured)
    return found


def diagnose(metrics: dict, log: dict[str, str], max_new_tokens: int | None) -> None:
    gen, step = series(metrics, GEN), series(metrics, STEP)
    steps = sorted(s for s in set(gen) & set(step) if step[s] > 0)
    gen_vals = [gen[s] for s in steps]
    share = med(gen[s] / step[s] for s in steps)
    gen_med = med(gen_vals)
    print(f"steps with timing: {len(steps)}")
    if steps:
        quartiles = (
            statistics.quantiles(gen_vals, n=4) if len(gen_vals) >= 2 else [gen_med] * 3
        )
        iqr = (quartiles[2] - quartiles[0]) / gen_med if gen_med > 0 else math.nan
        print(f"total step time, median: {med(step[s] for s in steps):.1f} s")
        print(
            f"generation, median: {gen_med:.1f} s ({share:.0%} of step; interquartile spread {iqr:.0%})"
        )
    else:
        iqr = math.nan
    val = series(metrics, VAL)
    if val:
        print(f"validation: {len(val)} runs, {sum(val.values()):.1f} s total")
    setup = series(metrics, SETUP)
    if setup:
        print(f"setup: {sum(setup.values()):.1f} s")
    prompt, mean_gen, max_gen = (
        series(metrics, PROMPT),
        series(metrics, MEAN_GEN),
        series(metrics, MAX_GEN),
    )
    if prompt and mean_gen:
        print(
            f"tokens per sample: prompt {med(prompt.values()):.0f}, generated mean {med(mean_gen.values()):.0f}"
        )
    cap = max_new_tokens or (
        int(log["max_new_tokens"]) if "max_new_tokens" in log else None
    )
    hits = None
    if max_gen and cap:
        hits = sum(v >= cap for v in max_gen.values()) / len(max_gen)
        print(f"steps whose longest sample hit max_new_tokens={cap}: {hits:.0%}")
    for name, value in log.items():
        print(f"log: {name} = {value}")

    print("\ndiagnosis:")
    if math.isnan(share):
        print("- No generation timing in metrics.json; cannot diagnose the rollout.")
        return
    if share < 0.5:
        print(
            f"- Generation is {share:.0%} of step time; rollout tuning will not move wall time much."
        )
        return
    print(f"- Generation-bound ({share:.0%} of step time).")
    if hits is not None and hits >= 0.8 and not iqr >= 0.1:
        print(
            "- Decode-tail-bound: the longest response sets the step; per-step decode latency is the lever."
        )
    if prompt and mean_gen and med(prompt.values()) > 4 * med(mean_gen.values()):
        print(
            "- Prefill-heavy: check prefix caching, scheduler token budget and session affinity."
        )
    if log.get("enforce_eager") == "True" or log.get("cudagraph_mode") == "NONE":
        print(
            "- vLLM runs eager: enable CUDA graphs (PIECEWISE for hybrid Mamba models) and A/B it."
        )
    if "max_concurrency" in log:
        print(
            f"- KV capacity per replica: {log['max_concurrency']}; memory knobs only help if this is below the per-replica load."
        )


def compare(base: dict, treat: dict) -> None:
    gb, gt = series(base, GEN), series(treat, GEN)
    paired = sorted(s for s in set(gb) & set(gt) if gb[s] > 0)
    if paired:
        ratios = [gt[s] / gb[s] for s in paired]
        print(
            f"paired steps: {len(paired)}; generation ratio treat/base median {med(ratios):.3f} (min {min(ratios):.3f}, max {max(ratios):.3f})"
        )
    rows: list[tuple[str, str, Callable[[list[float]], float]]] = [
        ("generation per step, median (s)", GEN, med),
        ("generation, sum over paired steps (s)", GEN, sum),
        ("total step time, sum over paired steps (s)", STEP, sum),
        ("validation, sum (s)", VAL, sum),
        ("setup (s)", SETUP, sum),
        ("generated tokens per sample, median", MEAN_GEN, med),
        ("token_mult_prob_error, median", TMPE, med),
        ("gen_kl_error, mean", GEN_KL, statistics.mean),
    ]
    for label, key, fn in rows:
        b, t = series(base, key), series(treat, key)
        common = sorted(set(b) & set(t))
        if not common:
            continue
        vb, vt = fn([b[s] for s in common]), fn([t[s] for s in common])
        print(f"{label:<44} {vb:>12.5g} -> {vt:>12.5g} ({rel_change(vb, vt)})")
    for name, run in (("base", base), ("treat", treat)):
        tmpe = series(run, TMPE)
        acc = series(run, VAL_ACC)
        spikes = sum(v >= 1.04 for v in tmpe.values())
        accs = ", ".join(f"step {s}: {acc[s]:.3f}" for s in sorted(acc))
        print(
            f"[{name}] steps with token_mult_prob_error >= 1.04: {spikes}/{len(tmpe)}; validation accuracy {accs or 'n/a'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "metrics", nargs="?", type=Path, help="metrics.json of one run to diagnose"
    )
    parser.add_argument(
        "--log", type=Path, help="ray-driver.log of the same run (optional)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, help="generation cap, if not found in the log"
    )
    parser.add_argument(
        "--base", type=Path, help="control metrics.json for an A/B comparison"
    )
    parser.add_argument(
        "--treat", type=Path, help="treatment metrics.json for an A/B comparison"
    )
    args = parser.parse_args()
    if args.base and args.treat:
        compare(json.loads(args.base.read_text()), json.loads(args.treat.read_text()))
    elif args.metrics:
        log = scan_log(args.log) if args.log else {}
        diagnose(json.loads(args.metrics.read_text()), log, args.max_new_tokens)
    else:
        parser.error("pass one metrics.json, or --base and --treat")


if __name__ == "__main__":
    main()
