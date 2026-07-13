#!/usr/bin/env python3
"""Remap a length ordering into an interleaved short/long step schedule.

Takes an existing ordering (e.g. the gym-keyed absolute ordering) and re-ranks
prompts so consecutive training steps alternate between short-output and
long-output bins in a fixed pattern. Default pattern "2S2L": each 4-step
interval contains the next 2 shortest unconsumed bins followed by the next 2
longest unconsumed bins (working inward from both extremes), so every interval
carries a balanced short/long load while each individual step stays
length-homogeneous.

Bins are sized to the training run's prompts-per-step so bin boundaries align
with step boundaries (pass --prompts-per-step to match grpo.num_prompts_per_step).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ordering_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--prompts-per-step", type=int, default=64)
    parser.add_argument(
        "--pattern",
        default="2S2L",
        help="Steps per interval from the short (S) then long (L) end, e.g. 2S2L",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Apply the pattern within consecutive windows of sorted bins "
        "(local mixing) instead of converging from the global extremes.",
    )
    args = parser.parse_args()

    payload = json.loads(args.ordering_json.read_text())
    labels = payload["labels"]
    order = sorted(labels, key=lambda h: labels[h]["rank"])

    n = args.prompts_per_step
    bins = [order[i : i + n] for i in range(0, len(order), n)]
    # A partial trailing bin would shift every later step boundary off its
    # bin, silently de-homogenizing steps. Keep only full bins in the
    # schedule; the remainder rides at the very end (epoch drop-last tail).
    leftover: list[str] = []
    if bins and len(bins[-1]) < n:
        leftover = bins.pop()

    # Parse pattern like "2S2L" -> [("S", 2), ("L", 2)]
    import re

    pattern = [
        (kind.upper(), int(count))
        for count, kind in re.findall(r"(\d+)([SsLl])", args.pattern)
    ]
    if not pattern:
        raise SystemExit(f"unparseable pattern {args.pattern!r}")

    window = sum(count for _, count in pattern)
    schedule: list[int] = []
    if args.windowed:
        # Local mixing: chunk consecutive sorted bins into windows and apply
        # the pattern within each window (shorts from the window's front,
        # longs from its back). Preserves the global short->long progression
        # at window granularity.
        for start in range(0, len(bins), window):
            lo, hi = start, min(start + window, len(bins)) - 1
            for kind, count in pattern:
                for _ in range(count):
                    if lo > hi:
                        break
                    if kind == "S":
                        schedule.append(lo)
                        lo += 1
                    else:
                        schedule.append(hi)
                        hi -= 1
    else:
        # Global mixing: shorts ascend from the global front, longs descend
        # from the global back, converging.
        lo, hi = 0, len(bins) - 1
        while lo <= hi:
            for kind, count in pattern:
                for _ in range(count):
                    if lo > hi:
                        break
                    if kind == "S":
                        schedule.append(lo)
                        lo += 1
                    else:
                        schedule.append(hi)
                        hi -= 1

    new_order = [h for b in schedule for h in bins[b]] + leftover
    new_labels = {}
    for rank, h in enumerate(new_order):
        entry = dict(labels[h])
        entry["source_rank"] = labels[h]["rank"]
        entry["rank"] = rank
        new_labels[h] = entry

    payload_out = {
        "meta": {
            **payload.get("meta", {}),
            "interleaved": args.pattern,
            "prompts_per_step": n,
            "bin_schedule": schedule,
            "source_ordering": str(args.ordering_json),
        },
        "order": new_order,
        "labels": new_labels,
    }
    args.output_json.write_text(json.dumps(payload_out, indent=2))

    print(f"bin schedule ({len(bins)} bins of {n}): {schedule}")
    print("per new-step median absolute output length:")
    for step, b in enumerate(schedule):
        vals = [labels[h].get("value_absolute", 0.0) for h in bins[b]]
        print(
            f"  step {step:2d} <- bin {b:2d} ({'short' if b < len(bins) / 2 else 'long'}): "
            f"median={statistics.median(vals):7.1f}"
        )


if __name__ == "__main__":
    main()
