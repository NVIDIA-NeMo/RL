#!/usr/bin/env python3
"""Re-bin each vanilla-run step window into length-sorted steps.

Groups the vanilla (baseline) run's prompts by the training-step window they
originally appeared in (steps 1-4, 5-8, ... via the trace's ordinal step_id),
then sorts *within each window* by observed absolute output length and re-deals
into homogeneous bins of prompts-per-step, shortest bin first. The result keeps
every prompt inside its original 4-step curriculum window while making each
individual step length-homogeneous.

Prompts appearing in multiple vanilla steps (epoch-2 wrap) are assigned to
their first window. Windows whose unique-prompt count is not a multiple of
prompts-per-step contribute their remainder to the tail (epoch drop-last zone),
keeping later step boundaries bin-aligned.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gymkey_ordering_json", type=Path)
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--prompts-per-step", type=int, default=64)
    parser.add_argument("--window-steps", type=int, default=4)
    args = parser.parse_args()

    payload = json.loads(args.gymkey_ordering_json.read_text())
    labels = payload["labels"]
    # prompt_token_hash -> content key (the rekey tool stored the token hash)
    token_to_content = {
        entry["prompt_token_hash"]: key
        for key, entry in labels.items()
        if "prompt_token_hash" in entry
    }

    # First vanilla step ordinal per prompt (token hash), from turn-0 rows.
    first_step: dict[str, int] = {}
    with args.trace_jsonl.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("turn_idx") != 0:
                continue
            h = row.get("prompt_token_hash")
            if not h or h in first_step:
                continue
            m = re.search(r"-(\d+)$", str(row.get("step_id", "")))
            if m:
                first_step[h] = int(m.group(1))

    # Window index per content key (windows are 1-based vanilla steps).
    windows: dict[int, list[str]] = defaultdict(list)
    for token_hash, step in sorted(first_step.items(), key=lambda kv: kv[1]):
        key = token_to_content.get(token_hash)
        if key is None:
            continue
        windows[(step - 1) // args.window_steps].append(key)

    n = args.prompts_per_step
    new_order: list[str] = []
    tail: list[str] = []
    print(f"{len(first_step)} traced prompts across {len(windows)} windows")
    for w in sorted(windows):
        keys = sorted(windows[w], key=lambda k: labels[k]["value_absolute"])
        full = len(keys) // n * n
        bins = [keys[i : i + n] for i in range(0, full, n)]
        tail.extend(keys[full:])
        for b, chunk in enumerate(bins):
            med = statistics.median(labels[k]["value_absolute"] for k in chunk)
            lo_ = min(labels[k]["value_absolute"] for k in chunk)
            hi_ = max(labels[k]["value_absolute"] for k in chunk)
            print(
                f"  window {w} (vanilla steps {w*args.window_steps+1}-"
                f"{(w+1)*args.window_steps}) bin {b}: "
                f"median={med:7.1f} min={lo_:7.1f} max={hi_:7.1f}"
            )
            new_order.extend(chunk)
    new_order.extend(tail)
    print(f"{len(tail)} remainder prompts placed at tail")

    new_labels = {}
    for rank, key in enumerate(new_order):
        entry = dict(labels[key])
        entry["source_rank"] = labels[key]["rank"]
        entry["rank"] = rank
        new_labels[key] = entry

    out = {
        "meta": {
            **payload.get("meta", {}),
            "schedule": "vanilla_window_sorted",
            "window_steps": args.window_steps,
            "prompts_per_step": n,
            "source_ordering": str(args.gymkey_ordering_json),
            "source_trace": str(args.trace_jsonl),
        },
        "order": new_order,
        "labels": new_labels,
    }
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(new_labels)} labels to {args.output_json}")


if __name__ == "__main__":
    main()
