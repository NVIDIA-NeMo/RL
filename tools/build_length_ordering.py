#!/usr/bin/env python3
"""Derive a shortest-output-first prompt ordering from a NeMo-Gym rollout trace.

The NeMo-RL DirectRequest trace (``mocker_request_trace_jsonl``) records one row
per assistant turn, carrying the first-turn ``prompt_token_hash`` and the
observed ``output_length``. This script aggregates observed output length per
unique prompt and emits an ordering (shortest generation first) plus per-prompt
labels, keyed by ``prompt_token_hash`` so it can be joined back onto the training
dataset by ``nemo_rl.data.length_ordering``.

Two length metrics are supported:

* ``step_normalized`` (default): rank each prompt's total output *within the
  rollout step it appeared in*, then average. This removes the training-stage
  length drift (later steps generate more tokens regardless of prompt), leaving
  an intrinsic "this prompt generates more" signal.
* ``absolute``: raw mean total output tokens per prompt. Faithful to what the
  vLLM workers actually served, but confounded with training progression.

Because each prompt's generations all share one first-turn ``prompt_token_hash``
and appear in a single step, the per-step normalization is exact.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path, help="DirectRequest rollout trace")
    parser.add_argument("output_json", type=Path, help="Where to write the ordering")
    parser.add_argument(
        "--metric",
        choices=("step_normalized", "absolute"),
        default="step_normalized",
        help="Per-prompt length signal used for ordering (default: step_normalized)",
    )
    parser.add_argument(
        "--aggregate",
        choices=("sum", "max", "first"),
        default="sum",
        help="How to combine a rollout's multi-turn output lengths (default: sum)",
    )
    parser.add_argument(
        "--prompts-per-step",
        type=int,
        default=None,
        help="Prompts per training step for the suggested_step field. "
        "Defaults to round(num_unique_prompts / num_steps_in_trace).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"warning: skipped {bad} unparseable line(s) in {path}")
    return rows


def _gen_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("step_id"), row.get("sample_idx"))


def combine_turn_outputs(lengths: list[int], aggregate: str) -> int:
    if not lengths:
        return 0
    if aggregate == "sum":
        return sum(lengths)
    if aggregate == "max":
        return max(lengths)
    return lengths[0]  # "first"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.trace_jsonl)
    if not rows:
        raise SystemExit(f"no usable rows in {args.trace_jsonl}")

    # Per generation (one rollout = one (step_id, sample_idx)), collect its
    # per-turn output lengths and its first-turn prompt hash.
    turn_outputs: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    first_turn_hash: dict[tuple[Any, Any], str] = {}
    for row in rows:
        key = _gen_key(row)
        out_len = row.get("output_length")
        if isinstance(out_len, int):
            turn_outputs[key].append(out_len)
        if row.get("turn_idx") == 0 and row.get("prompt_token_hash"):
            first_turn_hash[key] = str(row["prompt_token_hash"])

    # Total output per generation.
    gen_total: dict[tuple[Any, Any], int] = {
        key: combine_turn_outputs(lengths, args.aggregate)
        for key, lengths in turn_outputs.items()
    }

    # Per-generation metric value (absolute or step-normalized rank in [0, 1]).
    gen_metric: dict[tuple[Any, Any], float] = {}
    if args.metric == "absolute":
        gen_metric = {key: float(total) for key, total in gen_total.items()}
    else:
        by_step: dict[Any, list[tuple[Any, Any]]] = defaultdict(list)
        for key in gen_total:
            by_step[key[0]].append(key)
        for step_id, keys in by_step.items():
            ordered = sorted(keys, key=lambda k: gen_total[k])
            n = len(ordered)
            # Average-rank percentile so ties share a value.
            i = 0
            while i < n:
                j = i
                while j + 1 < n and gen_total[ordered[j + 1]] == gen_total[ordered[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0
                pct = avg_rank / (n - 1) if n > 1 else 0.0
                for k in ordered[i : j + 1]:
                    gen_metric[k] = pct
                i = j + 1

    # Aggregate generations up to unique prompts (by first-turn hash).
    per_prompt_metric: dict[str, list[float]] = defaultdict(list)
    per_prompt_absolute: dict[str, list[int]] = defaultdict(list)
    for key, prompt_hash in first_turn_hash.items():
        per_prompt_metric[prompt_hash].append(gen_metric.get(key, 0.0))
        per_prompt_absolute[prompt_hash].append(gen_total.get(key, 0))

    prompt_value = {h: statistics.mean(v) for h, v in per_prompt_metric.items()}
    prompt_absolute = {h: statistics.mean(v) for h, v in per_prompt_absolute.items()}

    order = sorted(prompt_value, key=lambda h: (prompt_value[h], h))
    num_prompts = len(order)
    num_steps = len({key[0] for key in gen_total})
    prompts_per_step = args.prompts_per_step or max(
        1, round(num_prompts / num_steps) if num_steps else num_prompts
    )
    median_rank = (num_prompts - 1) / 2.0

    labels: dict[str, dict[str, Any]] = {}
    for rank, prompt_hash in enumerate(order):
        labels[prompt_hash] = {
            "rank": rank,
            "value_metric": round(prompt_value[prompt_hash], 6),
            "value_absolute": round(prompt_absolute[prompt_hash], 3),
            "bin": "short" if rank <= median_rank else "long",
            "suggested_step": rank // prompts_per_step,
        }

    payload = {
        "meta": {
            "trace_jsonl": str(args.trace_jsonl),
            "metric": args.metric,
            "aggregate": args.aggregate,
            "num_prompts": num_prompts,
            "num_steps_in_trace": num_steps,
            "prompts_per_step": prompts_per_step,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hash_field": "prompt_token_hash",
            "order_is": "shortest_output_first",
        },
        "order": order,
        "labels": labels,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _report(order, prompt_absolute, prompts_per_step, args)


def _report(
    order: list[str],
    prompt_absolute: dict[str, float],
    prompts_per_step: int,
    args: argparse.Namespace,
) -> None:
    num_prompts = len(order)
    print(
        f"wrote {num_prompts} prompt labels to {args.output_json} "
        f"(metric={args.metric}, aggregate={args.aggregate}, "
        f"prompts_per_step={prompts_per_step})"
    )
    short = sum(1 for h in order if prompt_absolute[h] <= statistics.median(prompt_absolute.values()))
    print(f"median split: {short} short / {num_prompts - short} long")
    print("new-step median absolute output length (should increase monotonically):")
    for step_idx in range((num_prompts + prompts_per_step - 1) // prompts_per_step):
        chunk = order[step_idx * prompts_per_step : (step_idx + 1) * prompts_per_step]
        if not chunk:
            continue
        med = statistics.median(prompt_absolute[h] for h in chunk)
        lo = min(prompt_absolute[h] for h in chunk)
        hi = max(prompt_absolute[h] for h in chunk)
        print(f"  new step {step_idx:2d}: median={med:7.1f}  min={lo:7.1f}  max={hi:7.1f}")


if __name__ == "__main__":
    main()
