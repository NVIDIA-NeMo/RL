#!/usr/bin/env python3
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
"""Attach a per-prompt ``profile_band`` block to a profiled training JSONL.

Reads a JSONL produced by ``profile_run`` (rows must carry
``profiled_rewards``, ``profiled_output_lengths``,
``profiled_reasoning_lengths``, ``profiled_answer_lengths``, and
``pass_rate``) and writes a new JSONL where each row has an additional
``profile_band`` field consumed by Mechanism 6 in
``nemo_rl/utils/length_penalty.py``::

    profile_band:
      total:     {a, b, f}    # only present if data is non-degenerate
      reasoning: {a, b, f}
      answer:    {a, b, f}

For each channel:
  - reference set = passing profiled rollouts (reward > 0). If too few
    (< ``min_passing``), fall back to all profiled rollouts.
  - a = mean(reference)
  - b = mean(reference) + n_std * std(reference)
  - f = looked up in ``f_table`` by row's ``pass_rate``.
  - channel block is OMITTED if std == 0 (degenerate / cap-clamped),
    if reference set has fewer than 2 samples, or if pass_rate is not
    in the f_table.

Config yaml shape::

    n_std: 2.0
    min_passing: 2
    channels: [total, reasoning, answer]
    f_table:                       # pass_rate -> f; omitted pass_rates skip the row entirely
      - {pass_rate: 1.000, f: 0.6}
      - {pass_rate: 0.875, f: 0.7}
      - {pass_rate: 0.750, f: 0.8}
      - {pass_rate: 0.625, f: 0.9}

Usage::

    python scripts/build_profile_band_dataset.py \
        --input  /path/to/dapo17k_profiled_boxed_nanov3.jsonl \
        --config /path/to/profile_band.yaml \
        --output /path/to/dapo17k_profiled_band_boxed_nanov3.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# yaml is in stdlib via pyyaml on the cluster; fall back to a tiny parser if absent.
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

CHANNEL_TO_LENGTHS_KEY = {
    "total": "profiled_output_lengths",
    "reasoning": "profiled_reasoning_lengths",
    "answer": "profiled_answer_lengths",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Path to profiled JSONL (input).")
    p.add_argument("--config", required=True, help="Path to profile_band yaml config.")
    p.add_argument("--output", required=True, help="Path to write augmented JSONL.")
    p.add_argument(
        "--quiet", action="store_true", help="Suppress per-row diagnostics summary."
    )
    return p.parse_args()


def load_config(path: str) -> dict[str, Any]:
    if yaml is None:
        sys.exit("PyYAML not available; install pyyaml to use this script.")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"config at {path} did not parse as a dict")
    cfg.setdefault("n_std", 2.0)
    cfg.setdefault("min_passing", 2)
    cfg.setdefault("channels", ["total", "reasoning", "answer"])
    cfg.setdefault("f_table", [])
    # Normalize f_table to a {rounded_pass_rate: f} dict.
    f_table: dict[float, float] = {}
    for entry in cfg["f_table"]:
        pr = round(float(entry["pass_rate"]), 4)
        f_table[pr] = float(entry["f"])
    cfg["_f_table"] = f_table
    bad_channels = [c for c in cfg["channels"] if c not in CHANNEL_TO_LENGTHS_KEY]
    if bad_channels:
        sys.exit(
            f"unknown channels in config: {bad_channels}; valid: {sorted(CHANNEL_TO_LENGTHS_KEY)}"
        )
    return cfg


def lookup_f(pass_rate: float, f_table: dict[float, float]) -> float | None:
    return f_table.get(round(float(pass_rate), 4))


def channel_block(
    lengths: list[int],
    rewards: list[float],
    n_std: float,
    min_passing: int,
    f_value: float,
) -> dict[str, float] | None:
    """Compute {a, b, f} for one channel; return None if degenerate."""
    passing = [l for l, r in zip(lengths, rewards) if r is not None and r > 0]
    ref = passing if len(passing) >= min_passing else list(lengths)
    if len(ref) < 2:
        return None
    mean_l = statistics.mean(ref)
    std_l = statistics.stdev(ref)
    if std_l <= 0:
        return None
    a = mean_l
    b = mean_l + n_std * std_l
    return {"a": float(a), "b": float(b), "f": float(f_value)}


def build_band(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Construct the profile_band dict for this row, or None to skip."""
    f = lookup_f(row.get("pass_rate", -1.0), cfg["_f_table"])
    if f is None:
        return None
    rewards = row.get("profiled_rewards") or []
    band: dict[str, Any] = {}
    for ch_name in cfg["channels"]:
        lengths_key = CHANNEL_TO_LENGTHS_KEY[ch_name]
        lengths = row.get(lengths_key)
        if lengths is None:
            continue
        block = channel_block(
            lengths=lengths,
            rewards=rewards,
            n_std=cfg["n_std"],
            min_passing=cfg["min_passing"],
            f_value=f,
        )
        if block is not None:
            band[ch_name] = block
    return band or None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_with_band = 0
    n_pr_skipped = 0
    n_pr_skipped_by_passrate: Counter[float] = Counter()
    n_channels_emitted: Counter[str] = Counter()

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            row = json.loads(line)
            n_total += 1
            band = build_band(row, cfg)
            if band is None:
                pr = round(float(row.get("pass_rate", -1.0)), 4)
                if lookup_f(pr, cfg["_f_table"]) is None:
                    n_pr_skipped += 1
                    n_pr_skipped_by_passrate[pr] += 1
            else:
                row["profile_band"] = band
                n_with_band += 1
                for ch in band:
                    n_channels_emitted[ch] += 1
            fout.write(json.dumps(row) + "\n")

    if not args.quiet:
        print(f"input  : {in_path}")
        print(f"output : {out_path}")
        print(f"rows in: {n_total}")
        print(
            f"rows w/ profile_band: {n_with_band}  ({100 * n_with_band / n_total:.1f}%)"
        )
        print(f"rows skipped (pass_rate not in f_table): {n_pr_skipped}")
        if n_pr_skipped_by_passrate:
            print("  by pass_rate:")
            for pr in sorted(n_pr_skipped_by_passrate):
                print(f"    {pr:.3f}: {n_pr_skipped_by_passrate[pr]}")
        print("channels emitted (per row, summed):")
        for ch in cfg["channels"]:
            c = n_channels_emitted.get(ch, 0)
            print(f"  {ch:>10}: {c}  ({100 * c / n_total:.1f}% of rows)")
        print("f_table (rounded pass_rate -> f):")
        for pr in sorted(cfg["_f_table"]):
            print(f"  {pr:.3f} -> {cfg['_f_table'][pr]}")
        print(f"n_std={cfg['n_std']}  min_passing={cfg['min_passing']}")


if __name__ == "__main__":
    main()
