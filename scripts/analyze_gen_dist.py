#!/usr/bin/env python3
"""Prompt-level distribution analysis: within the 16 generations of each prompt,
the distribution of turns and e2e latency — per prompt x per run.
Usage: python3 analyze_gen_dist.py gen_latency.csv [out_csv]
"""
import csv
import sys
from collections import defaultdict


def pctl(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)]


def stats(xs):
    n = len(xs)
    mean = sum(xs) / n if n else 0
    var = sum((x - mean) ** 2 for x in xs) / n if n else 0
    return dict(n=n, min=min(xs) if xs else 0, p10=pctl(xs, .1), p50=pctl(xs, .5),
                p90=pctl(xs, .9), max=max(xs) if xs else 0, mean=mean, std=var ** .5)


def short(iid):
    return iid.replace("astropy__astropy-", "ap-").replace("django__django-", "dj-")


def main(f, out=None):
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        r["turns"] = int(r["turns"]); r["e2e_s"] = float(r["e2e_s"])
        r["oh_s"] = float(r["openhands_s"]); r["reward"] = float(r["reward"])
        r["to"] = r["timed_out"] == "True"
    runs = sorted(set(r["run"] for r in rows))
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))  # preserve step order

    by_pr = defaultdict(list)   # (prompt,run)->rows
    by_p = defaultdict(list)    # prompt->rows
    for r in rows:
        by_pr[(r["prompt"], r["run"])].append(r)
        by_p[r["prompt"]].append(r)

    # ---- 1) pooled per-prompt (across all runs; 5x16=80 gens) ----
    print(f"\n[POOLED PER-PROMPT]  distribution over all {len(runs)}x16 gens (turns | e2e latency s)")
    print(f"{'prompt':10} {'solv/80':>7} | {'t.p10':>5} {'t.p50':>5} {'t.p90':>5} {'t.max':>5} | {'e.p10':>6} {'e.p50':>6} {'e.p90':>6} {'e.max':>6}")
    for p in prompts:
        rs = by_p[p]; t = stats([x["turns"] for x in rs]); e = stats([x["e2e_s"] for x in rs])
        sv = sum(x["reward"] > .5 for x in rs)
        print(f"{short(p):10} {sv:>3}/{len(rs):<3} | {t['p10']:>5.0f} {t['p50']:>5.0f} {t['p90']:>5.0f} {t['max']:>5.0f} | {e['p10']:>6.0f} {e['p50']:>6.0f} {e['p90']:>6.0f} {e['max']:>6.0f}")

    # ---- 2) turns p50 grid (prompt x run) + within-run spread ----
    def grid(metric, label, fmt):
        print(f"\n[{label} — p50 by prompt x run]  (cell = median over 16 gens)")
        print(f"{'prompt':10} " + " ".join(f"{r:>7}" for r in runs) + f"  {'gen-spread*':>10}")
        for p in prompts:
            cells = []
            spreads = []
            for r in runs:
                xs = [x[metric] for x in by_pr[(p, r)]]
                cells.append(pctl(xs, .5))
                spreads.append(pctl(xs, .9) - pctl(xs, .1))
            sp = sum(spreads) / len(spreads)
            print(f"{short(p):10} " + " ".join(fmt.format(c) for c in cells) + f"  {fmt.format(sp)}")
        print("  *gen-spread = mean over runs of (p90-p10) within a prompt's 16 gens")

    grid("turns", "TURNS", "{:>7.0f}")
    grid("e2e_s", "E2E LATENCY (s)", "{:>7.0f}")

    # ---- 3) full per-(prompt,run) table to CSV ----
    if out:
        w = csv.writer(open(out, "w", newline=""))
        w.writerow(["prompt", "run", "n", "solved", "timeouts",
                    "turns_min", "turns_p50", "turns_p90", "turns_max", "turns_std",
                    "e2e_min", "e2e_p50", "e2e_p90", "e2e_max", "e2e_std"])
        for p in prompts:
            for r in runs:
                rs = by_pr[(p, r)]
                if not rs:
                    continue
                t = stats([x["turns"] for x in rs]); e = stats([x["e2e_s"] for x in rs])
                w.writerow([p, r, len(rs), sum(x["reward"] > .5 for x in rs), sum(x["to"] for x in rs),
                            f"{t['min']:.0f}", f"{t['p50']:.0f}", f"{t['p90']:.0f}", f"{t['max']:.0f}", f"{t['std']:.1f}",
                            f"{e['min']:.0f}", f"{e['p50']:.0f}", f"{e['p90']:.0f}", f"{e['max']:.0f}", f"{e['std']:.1f}"])
        print(f"\nfull per-(prompt,run) table -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
