#!/usr/bin/env python3
"""Per-generation table for prompt-level distribution analysis.
One row per rollout (generation): run, prompt(instance), gen_index, turns, openhands_s, e2e_s, reward, timed_out.
e2e_s = full rollout wall-clock = sum of all phase timers (queue+spinup+connect+init+openhands+final_eval),
each clamped to [0,1e5] to drop unclosed timers on killed rollouts.
Usage: python3 extract_gen_latency.py <out.csv> <name>=<results.jsonl> [...]
"""
import csv
import json
import sys

PHASES = ["ray_queue_time", "generation_apptainer_spinup_time", "create_runtime_time",
          "connect_to_runtime_time", "initialize_runtime_time", "openhands_run_time",
          "final_eval_apptainer_spinup_time", "final_eval_time"]


def clamp(v):
    return v if isinstance(v, (int, float)) and 0 <= v < 1e5 else 0.0


def main(out, args):
    w = csv.writer(open(out, "w", newline=""))
    w.writerow(["run", "prompt", "gen_index", "turns", "openhands_s", "e2e_s", "reward", "timed_out"])
    for a in args:
        name, path = a.split("=", 1)
        n = 0
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            d = r.get("full_result", {}) or {}
            resp = d.get("response", {}) or {}
            md = d.get("responses_create_params", {}).get("metadata", {})
            out_items = resp.get("output") or []
            turns = sum(1 for x in out_items if x.get("type") == "function_call")
            e2e = sum(clamp(d.get(p)) for p in PHASES)
            oh = clamp(d.get("openhands_run_time"))
            def g(k):  # nested lookup for timed_out flag
                a2 = []
                def wlk(x):
                    if isinstance(x, dict):
                        for kk, v in x.items():
                            if kk == k: a2.append(v)
                            wlk(v)
                    elif isinstance(x, list):
                        [wlk(y) for y in x]
                wlk(d); return a2[0] if a2 else None
            w.writerow([name, md.get("instance_id", "?"), r.get("generation_index"), turns,
                        round(oh, 1), round(e2e, 1), r.get("reward", 0), bool(g("agent_timed_out"))])
            n += 1
        print(f"[done] {name}: {n} rows", flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
