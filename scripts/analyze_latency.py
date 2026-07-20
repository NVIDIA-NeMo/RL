#!/usr/bin/env python3
"""Distributions of REAL per-request LLM latency and per-action execution latency
(from extract_oh_latency.py). Usage: python3 analyze_latency.py <latency_dir>"""
import csv
import statistics as st
import sys
from collections import defaultdict


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)] if xs else 0


def summ(xs, label):
    print(f"  {label:24} n={len(xs):>6} min={min(xs):>5.1f} p50={pct(xs,.5):>5.1f} p90={pct(xs,.9):>5.1f} "
          f"p95={pct(xs,.95):>5.1f} p99={pct(xs,.99):>6.1f} max={max(xs):>6.1f} mean={st.mean(xs):>5.1f}")


def hist(xs, edges, label):
    print(f"  {label}:")
    for a, b in zip(edges, edges[1:]):
        c = sum(1 for x in xs if a <= x < b)
        print(f"    {a:>4}-{b:<4}s {100*c/len(xs):>4.1f}% {'█'*round(50*c/len(xs))}")
    c = sum(1 for x in xs if x >= edges[-1])
    print(f"    {edges[-1]:>4}+   s {100*c/len(xs):>4.1f}% {'█'*round(50*c/len(xs))}")


def main(dr):
    # ---- per-request ----
    rows = list(csv.DictReader(open(f"{dr}/per_request_latency.csv")))
    lat = [float(r["latency_s"]) for r in rows]
    print(f"=== PER-REQUEST LLM latency (s) — {len(lat)} requests across 5 runs ===")
    summ(lat, "overall")
    byrun = defaultdict(list)
    for r in rows:
        byrun[r["run"]].append(float(r["latency_s"]))
    for run in sorted(byrun):
        summ(byrun[run], run)
    hist(lat, [0, 2, 4, 6, 10, 15, 25, 40, 80], "per-request latency")

    # ---- per-action ----
    arows = list(csv.DictReader(open(f"{dr}/per_action_latency.csv")))
    al = [float(r["latency_s"]) for r in arows]
    print(f"\n=== PER-ACTION execution latency (s) — {len(al)} actions across 5 runs ===")
    summ(al, "overall")
    byt = defaultdict(list)
    for r in arows:
        byt[r["obs_type"]].append(float(r["latency_s"]))
    print("  by observation_type:")
    print(f"    {'obs_type':28} {'count':>7} {'%':>5} {'p50':>6} {'p90':>6} {'p99':>6} {'max':>7} {'mean':>6}")
    for t in sorted(byt, key=lambda k: -len(byt[k])):
        xs = byt[t]
        print(f"    {t:28} {len(xs):>7} {100*len(xs)/len(al):>4.1f}% {pct(xs,.5):>6.2f} {pct(xs,.9):>6.1f} "
              f"{pct(xs,.99):>6.1f} {max(xs):>7.1f} {st.mean(xs):>6.2f}")
    hist(al, [0, 0.5, 1, 2, 5, 10, 30, 60, 120], "per-action latency")


if __name__ == "__main__":
    main(sys.argv[1])
