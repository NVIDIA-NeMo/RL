#!/usr/bin/env python3
"""Multi-granularity analysis of a per-request token table (from extract_requests.py).
Sections: PER-STEP, PER-TRAJECTORY, PER-REQUEST, TOKENS (prefill vs generated), LONG-TAIL.
Usage: python3 analyze_tokens.py <requests.jsonl>
"""
import json
import sys
from collections import defaultdict


def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def p(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    i = min(int(q * len(xs)), len(xs) - 1)
    return xs[i]


def dist(xs):
    return (f"p50={p(xs,.5):>8.0f} p90={p(xs,.9):>8.0f} p99={p(xs,.99):>8.0f} "
            f"max={max(xs) if xs else 0:>8.0f} mean={sum(xs)/len(xs) if xs else 0:>8.0f}")


def main(f):
    reqs = load(f)
    # group by rollout
    traj = defaultdict(list)
    for r in reqs:
        traj[r["rollout"]].append(r)
    for v in traj.values():
        v.sort(key=lambda x: x["turn"])
    rollouts = list(traj.values())
    N = len(rollouts)
    print(f"\n{'='*74}\n{f.split('/')[-1]}  |  {N} trajectories, {len(reqs)} requests\n{'='*74}")

    def rsum(v, k):
        return sum(x[k] for x in v)

    # ---------- PER-STEP ----------
    print("\n[PER-STEP]  (each step = 4 instances x 16 gens = 64 rollouts)")
    print(f"  {'step':>4} {'rollouts':>8} {'reward':>7} {'turns':>6} {'gen_tok/traj':>12} {'finalctx':>9} {'run_s':>7}")
    bystep = defaultdict(list)
    for v in rollouts:
        bystep[v[0]["step"]].append(v)
    for s in sorted(bystep):
        vs = bystep[s]
        rew = sum(v[0]["reward"] > 0.5 for v in vs) / len(vs)
        tn = sum(len(v) for v in vs) / len(vs)
        gt = sum(rsum(v, "gen_tok") for v in vs) / len(vs)
        fc = sum(v[-1]["prefill_tok"] for v in vs) / len(vs)
        rt = sum(v[0]["run_time"] for v in vs) / len(vs)
        print(f"  {s:>4} {len(vs):>8} {rew:>7.3f} {tn:>6.0f} {gt:>12.0f} {fc:>9.0f} {rt:>7.0f}")

    # ---------- PER-TRAJECTORY ----------
    print("\n[PER-TRAJECTORY]  distributions across", N, "trajectories")
    turns = [len(v) for v in rollouts]
    gtot = [rsum(v, "gen_tok") for v in rollouts]
    ptot = [rsum(v, "prefill_tok") for v in rollouts]          # raw re-prefill (sum over turns)
    fctx = [v[-1]["prefill_tok"] for v in rollouts]            # final context size
    incr = [v[-1]["prefill_tok"] - v[0]["prefill_tok"] for v in rollouts]  # context growth
    print(f"  turns/traj        {dist(turns)}")
    print(f"  gen_tok/traj      {dist(gtot)}   (total decoded)")
    print(f"  final_ctx_tok     {dist(fctx)}   (context at last request)")
    print(f"  ctx_growth_tok    {dist(incr)}   (last - first prompt)")
    print(f"  prefill_tok/traj  {dist(ptot)}   (SUM of prompts = raw re-prefill, no cache)")
    # split solved vs not
    sv = [v for v in rollouts if v[0]["reward"] > 0.5]
    fl = [v for v in rollouts if v[0]["reward"] <= 0.5]
    print(f"  gen_tok/traj  SOLVED({len(sv)})  {dist([rsum(v,'gen_tok') for v in sv])}")
    print(f"  gen_tok/traj  FAILED({len(fl)})  {dist([rsum(v,'gen_tok') for v in fl])}")

    # ---------- PER-REQUEST ----------
    print("\n[PER-REQUEST]  distributions across", len(reqs), "requests")
    pf = [r["prefill_tok"] for r in reqs]
    gn = [r["gen_tok"] for r in reqs]
    print(f"  prefill_tok/req   {dist(pf)}")
    print(f"  gen_tok/req       {dist(gn)}")
    # growth by turn bucket
    print("  prefill grows over turns (mean prefill_tok by turn index):")
    byturn = defaultdict(list)
    for r in reqs:
        byturn[r["turn"]].append(r["prefill_tok"])
    for t in (0, 5, 10, 20, 30, 40, 50, 59):
        if t in byturn:
            print(f"    turn {t:>2}: prefill p50={p(byturn[t],.5):>7.0f}  gen p50={p([x['gen_tok'] for x in reqs if x['turn']==t],.5):>5.0f}")

    # ---------- TOKENS: prefill vs generated ----------
    TP = sum(pf); TG = sum(gn); TI = sum(incr)
    print("\n[TOKENS: PREFILL vs GENERATED]  (whole run)")
    print(f"  total generated (decode)     = {TG:>14,}")
    print(f"  total prefill RAW (re-prefill every turn, no cache) = {TP:>14,}")
    print(f"  total prefill INCREMENTAL (with prefix cache ~ ctx growth) = {TI:>14,}")
    print(f"  raw    prefill : generated  = {TP/TG:>6.1f} : 1")
    print(f"  cached prefill : generated  = {TI/TG:>6.1f} : 1   (effective compute if prefix-cached)")
    # throughput: gen_tok / model_call_time per trajectory
    tps = [rsum(v, "gen_tok") / v[0]["model_call_time"] for v in rollouts if v[0]["model_call_time"] > 0]
    print(f"  decode throughput gen_tok/model_s per traj: {dist(tps)}")

    # ---------- LONG TAIL ----------
    print("\n[LONG TAIL]")
    print("  slowest trajectories (run_time s):")
    for v in sorted(rollouts, key=lambda v: -v[0]["run_time"])[:6]:
        h = v[0]
        print(f"    {h['run_time']:>6.0f}s  {len(v):>2}t  gen={rsum(v,'gen_tok'):>6}  ctx={v[-1]['prefill_tok']:>7}  {h['outcome']:<18} {h['instance'].replace('__','/')}")
    print("  largest single requests (prefill_tok):")
    for r in sorted(reqs, key=lambda r: -r["prefill_tok"])[:5]:
        print(f"    prefill={r['prefill_tok']:>7}  gen={r['gen_tok']:>5}  turn={r['turn']:>2}  rollout={r['rollout']}  {r['instance'].replace('__','/')}")
    print("  largest single generations (gen_tok):")
    for r in sorted(reqs, key=lambda r: -r["gen_tok"])[:5]:
        print(f"    gen={r['gen_tok']:>6}  prefill={r['prefill_tok']:>7}  turn={r['turn']:>2}  rollout={r['rollout']}")
    print(f"  tail ratios: prefill p99/p50={p(pf,.99)/max(p(pf,.5),1):.1f}x  gen p99/p50={p(gn,.99)/max(p(gn,.5),1):.1f}x "
          f"run_time p99/p50={p([v[0]['run_time'] for v in rollouts],.99)/max(p([v[0]['run_time'] for v in rollouts],.5),1):.1f}x")


if __name__ == "__main__":
    main(sys.argv[1])
