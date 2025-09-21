#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _scalarize(v: Any):
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        if len(v) == 1:
            return _scalarize(v[0])
        # For lists >1, return tuple of scalarized values
        return tuple(_scalarize(x) for x in v)
    return v


def load_jsonl(path: Path):
    records = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # tolerate python-style dicts written by str(rec); eval safely
                r = eval(line, {"__builtins__": {}}, {})  # noqa: S307
            # Normalize expected fields to scalars
            for k in ("dataset", "pair_index", "reward_chosen", "reward_rejected", "reward_delta"):
                if k in r:
                    r[k] = _scalarize(r[k])
            # Coerce numeric types
            if isinstance(r.get("pair_index"), float):
                r["pair_index"] = int(r["pair_index"])  # indices are ints
            records.append(r)
    return records


def _normalize_key_value(v: Any):
    # Convert list/tuple to scalar if length 1; otherwise to tuple for hashing
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return _normalize_key_value(v[0])
        else:
            return tuple(_normalize_key_value(x) for x in v)
    # Leave basic types as-is (including None)
    return v


def index_by_key(records, key_fields):
    idx = {}
    for r in records:
        k = tuple(_normalize_key_value(r.get(kf, None)) for kf in key_fields)
        idx[k] = r
    return idx


def summarize(records):
    if not records:
        return {}
    deltas = [float(r["reward_delta"]) for r in records]
    chosen = [float(r["reward_chosen"]) for r in records]
    rejected = [float(r["reward_rejected"]) for r in records]
    pos_rate = sum(1 for d in deltas if d > 0.0) / len(deltas)
    return {
        "count": len(records),
        "delta.mean": sum(deltas) / len(deltas),
        "delta.min": min(deltas),
        "delta.max": max(deltas),
        "delta.positive_rate": pos_rate,
        "chosen.mean": sum(chosen) / len(chosen),
        "rejected.mean": sum(rejected) / len(rejected),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two RM validation JSONL logs and summarize churn")
    parser.add_argument("before", type=Path, help="Path to baseline JSONL (e.g., exp_A/validation/default_pairs_step_0.jsonl)")
    parser.add_argument("after", type=Path, help="Path to candidate JSONL (e.g., exp_B/validation/default_pairs_step_0.jsonl)")
    parser.add_argument("--key", nargs="*", default=["dataset", "pair_index"], help="Key fields to join on")
    parser.add_argument("--topk", type=int, default=20, help="Show top-K worst regressions")
    parser.add_argument("--dump_regressions", type=Path, default=None, help="Optional path to write worst regression keys as a JSON list")
    args = parser.parse_args()

    before = load_jsonl(args.before)
    after = load_jsonl(args.after)

    # Index by key
    idx_before = index_by_key(before, args.key)
    idx_after = index_by_key(after, args.key)

    # Join
    common_keys = sorted(set(idx_before.keys()) & set(idx_after.keys()))
    missing_in_after = sorted(set(idx_before.keys()) - set(idx_after.keys()))
    missing_in_before = sorted(set(idx_after.keys()) - set(idx_before.keys()))

    diffs = []
    for k in common_keys:
        b = idx_before[k]
        a = idx_after[k]
        bd = float(b["reward_delta"])
        ad = float(a["reward_delta"])
        diffs.append(
            {
                "key": k,
                "before_delta": bd,
                "after_delta": ad,
                "delta_change": ad - bd,
                "before_chosen": float(b["reward_chosen"]),
                "after_chosen": float(a["reward_chosen"]),
                "before_rejected": float(b["reward_rejected"]),
                "after_rejected": float(a["reward_rejected"]),
            }
        )

    # Summaries
    print("=== Coverage ===")
    print(f"Before count: {len(before)}")
    print(f"After  count: {len(after)}")
    print(f"Common      : {len(common_keys)}")
    print(f"Missing in after: {len(missing_in_after)}")
    print(f"Missing in before: {len(missing_in_before)}")
    print()

    print("=== Aggregate stats ===")
    s_before = summarize(before)
    s_after = summarize(after)
    for name, stats in (("before", s_before), ("after", s_after)):
        print(f"[{name}] count={stats.get('count', 0)} delta.mean={stats.get('delta.mean', 0):.6f}"
              f" delta.min={stats.get('delta.min', 0):.6f} delta.max={stats.get('delta.max', 0):.6f}"
              f" delta.positive_rate={stats.get('delta.positive_rate', 0):.6f}"
              f" chosen.mean={stats.get('chosen.mean', 0):.6f} rejected.mean={stats.get('rejected.mean', 0):.6f}")
    print()

    # Churn: fraction of sign flips in reward_delta
    sign = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    sign_flips = sum(1 for d in diffs if sign(d["before_delta"]) != sign(d["after_delta"]))
    print("=== Churn ===")
    print(f"Sign flips: {sign_flips}/{len(diffs)} = {sign_flips/len(diffs) if diffs else 0:.4f}")
    print()

    # Regressions sorted by delta_change ascending
    worst = sorted(diffs, key=lambda r: r["delta_change"])[: args.topk]
    print(f"=== Worst {args.topk} regressions (delta_change ascending) ===")
    for r in worst:
        print(
            f"key={r['key']} before={r['before_delta']:.6f} after={r['after_delta']:.6f} change={r['delta_change']:.6f}"
        )

    if args.dump_regressions:
        with open(args.dump_regressions, "w") as fo:
            json.dump([list(r["key"]) for r in worst], fo)
        print(f"Wrote worst regression keys to {args.dump_regressions}")


if __name__ == "__main__":
    main()
