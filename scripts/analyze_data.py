#!/usr/bin/env python3
"""Deep data analysis of a rollout-benchmark run (or several).
Outcome taxonomy, timing decomposition, turns, per-instance difficulty / pass@k, patch stats.
Usage: python3 analyze_data.py <results.jsonl> [<results2.jsonl> ...]
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict


def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def fr(r):
    return r.get("full_result", {}) or {}


def turns(r):
    out = fr(r).get("response", {}).get("output", []) or []
    return sum(1 for x in out if isinstance(x, dict) and x.get("type") == "function_call")


def reasoning_items(r):
    out = fr(r).get("response", {}).get("output", []) or []
    return sum(1 for x in out if isinstance(x, dict) and x.get("type") == "reasoning")


def instance_id(r):
    return fr(r).get("responses_create_params", {}).get("metadata", {}).get("instance_id", "?")


def outcome(r):
    d = fr(r)
    if r.get("reward", 0) > 0.5 or d.get("resolved"):
        return "SOLVED"
    if d.get("agent_timed_out"):
        return "TIMEOUT"
    if d.get("patch_exists"):
        return "WRONG_PATCH"          # produced a diff, tests failed
    ek = d.get("agent_error_kind")
    if ek == "max_iteration":
        return "NO_PATCH:max_turns"
    if ek == "context_window":
        return "NO_PATCH:context"
    if ek:
        return f"NO_PATCH:{ek}"
    return "NO_PATCH:other"


def pct(n, d):
    return f"{100*n/d:4.1f}%" if d else "  -  "


def analyze(files):
    rows = [r for f in files for r in load(f)]
    n = len(rows)
    print(f"\n{'='*66}\nFILES: {', '.join(f.split('/')[-3] for f in files)}\nrollouts={n}\n{'='*66}")

    # --- outcome taxonomy ---
    oc = Counter(outcome(r) for r in rows)
    print("\n[OUTCOME TAXONOMY]")
    for k, c in oc.most_common():
        print(f"  {k:24s} {c:4d}  {pct(c,n)}")
    solved = [r for r in rows if outcome(r) == "SOLVED"]
    print(f"  {'-'*40}\n  mean_reward = {sum(r.get('reward',0) for r in rows)/n:.4f}")

    # --- timing decomposition (mean seconds) ---
    keys = ["openhands_run_time", "total_model_call_time", "total_command_exec_time",
            "connect_to_runtime_time", "initialize_runtime_time",
            "generation_apptainer_spinup_time", "ray_queue_time"]
    print("\n[TIMING mean seconds]  (model-call vs command-exec vs setup)")
    for k in keys:
        v = [fr(r).get(k) or 0 for r in rows]
        print(f"  {k:34s} p50={st.median(v):6.1f}  mean={st.mean(v):6.1f}  max={max(v):6.1f}")
    rt = [fr(r).get("openhands_run_time") or 0 for r in rows]
    mc = [fr(r).get("total_model_call_time") or 0 for r in rows]
    print(f"  → model-call is {100*sum(mc)/sum(rt):.0f}% of wall-clock")

    # --- turns ---
    tall = [turns(r) for r in rows]
    tsolved = [turns(r) for r in solved]
    print("\n[TURNS]")
    print(f"  all:    p50={st.median(tall):.0f} p90={sorted(tall)[int(0.9*n)-1]} max={max(tall)}")
    if tsolved:
        print(f"  solved: p50={st.median(tsolved):.0f} mean={st.mean(tsolved):.1f} max={max(tsolved)} range=[{min(tsolved)},{max(tsolved)}]")

    # --- per-instance difficulty / pass@k ---
    by = defaultdict(list)
    for r in rows:
        by[instance_id(r)].append(r.get("reward", 0) > 0.5)
    print(f"\n[PER-INSTANCE]  ({len(by)} instances, {n//len(by)} gens each)")
    solved_any = sum(1 for v in by.values() if any(v))
    solved_all = sum(1 for v in by.values() if all(v))
    print(f"  pass@1 (mean solve) = {sum(sum(v) for v in by.values())/n:.3f}")
    print(f"  pass@k (≥1 of {n//len(by)} solved) = {solved_any}/{len(by)} instances")
    print(f"  fully solved (all gens) = {solved_all};  never solved = {len(by)-solved_any}")
    print("  difficulty tiers (solve rate):")
    for iid in sorted(by, key=lambda k: -sum(by[k])):
        s = sum(by[iid]); tot = len(by[iid])
        if s:
            print(f"    {iid.replace('__','/'):26s} {s:2d}/{tot}")

    # --- patch production among failures ---
    fails = [r for r in rows if outcome(r) != "SOLVED"]
    with_patch = sum(1 for r in fails if fr(r).get("patch_exists"))
    print(f"\n[FAILURE MODES]  {len(fails)} failures")
    print(f"  produced-a-patch-but-failed: {with_patch}  ({pct(with_patch,len(fails))})")
    print(f"  no-patch (gave up/ran out):  {len(fails)-with_patch}  ({pct(len(fails)-with_patch,len(fails))})")


if __name__ == "__main__":
    analyze(sys.argv[1:])
