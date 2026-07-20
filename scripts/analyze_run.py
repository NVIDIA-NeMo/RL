#!/usr/bin/env python3
"""Summarize a rollout-benchmark run: mean_reward, per-step, timeout count, run-time dist.
Usage: python3 analyze_run.py <path-to-nemo_gym_eval_results.jsonl>
"""
import json
import statistics
import sys
from collections import Counter, defaultdict

F = sys.argv[1]
rows = [json.loads(l) for l in open(F) if l.strip()]


def collect(d, key, acc):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                acc.append(v)
            collect(v, key, acc)
    elif isinstance(d, list):
        for x in d:
            collect(x, key, acc)


def first(r, key):
    acc = []
    collect(r.get("full_result", {}), key, acc)
    return acc[0] if acc else None


n = len(rows)
rewards = [r["reward"] for r in rows]
print(f"rollouts={n}  mean_reward={sum(rewards)/n:.4f}  solved={int(sum(rewards))}/{n}")
by = defaultdict(list)
for r in rows:
    by[r["eval_step"]].append(r["reward"])
for s in sorted(by):
    xs = by[s]
    print(f"  step {s}: mean={sum(xs)/len(xs):.4f} solved={int(sum(xs))}/{len(xs)}")

times = [first(r, "openhands_run_time") or 0 for r in rows]
timeouts = sum(1 for r in rows if first(r, "agent_timed_out") is True)
errs = Counter(first(r, "agent_error_kind") for r in rows if first(r, "agent_error_kind"))
p90 = sorted(times)[min(int(0.9 * n), n - 1)]
print(f"TIMEOUTS (agent_timed_out=True): {timeouts}/{n}")
print(
    f"run_time s: min={min(times):.0f} p50={statistics.median(times):.0f} "
    f"mean={statistics.mean(times):.0f} p90={p90:.0f} max={max(times):.0f}"
)
print(
    f"  >1200s:{sum(t>1200 for t in times)}  >1000s:{sum(t>1000 for t in times)}  "
    f">900s:{sum(t>900 for t in times)}  >600s:{sum(t>600 for t in times)}"
)
print(f"agent_error_kind: {dict(errs)}")
