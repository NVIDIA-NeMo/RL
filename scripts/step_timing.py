#!/usr/bin/env python3
"""Per-(run, step) phase-timing breakdown across all runs.
Streams each results.jsonl (huge), pulls only the timing fields, aggregates by eval_step.
Phases (mean seconds/rollout): queue, setup(connect+init+spinup), openhands(run; model/cmd sub),
final_eval(test scoring; only for rollouts that produced a patch), total wall-clock.
Usage: python3 step_timing.py <name>=<results.jsonl> [<name>=<results2.jsonl> ...]
"""
import json
import sys
from collections import defaultdict

FIELDS = ["ray_queue_time", "generation_apptainer_spinup_time", "create_runtime_time",
          "connect_to_runtime_time", "initialize_runtime_time", "openhands_run_time",
          "total_model_call_time", "total_command_exec_time",
          "final_eval_apptainer_spinup_time", "final_eval_time"]


def main(args):
    # acc[(run,step)][field] = [sum, count]; plus n rollouts
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    nroll = defaultdict(int)
    for a in args:
        name, path = a.split("=", 1)
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            d = r.get("full_result", {}) or {}
            step = r.get("eval_step")
            key = (name, step)
            nroll[key] += 1
            for f in FIELDS:
                v = d.get(f)
                if isinstance(v, (int, float)) and 0 <= v < 1e5:   # skip unclosed timers (negative) on killed rollouts
                    acc[key][f][0] += v
                    acc[key][f][1] += 1
        print(f"[done] {name}", flush=True)

    def m(key, f):
        s, c = acc[key][f]
        return s / c if c else 0.0

    print(f"\n{'run':6} {'step':>4} {'n':>4} | {'queue':>6} {'setup':>6} {'model':>7} {'cmd':>6} {'openhnd':>7} {'eval':>6} {'n_ev':>4} | {'TOTAL':>7}")
    print("-" * 92)
    tot_by_run = defaultdict(float)
    for key in sorted(acc):
        name, step = key
        n = nroll[key]
        queue = m(key, "ray_queue_time")
        setup = m(key, "connect_to_runtime_time") + m(key, "initialize_runtime_time") + \
            m(key, "generation_apptainer_spinup_time") + m(key, "create_runtime_time")
        model = m(key, "total_model_call_time")
        cmd = m(key, "total_command_exec_time")
        oh = m(key, "openhands_run_time")
        ev = m(key, "final_eval_time") + m(key, "final_eval_apptainer_spinup_time")
        n_ev = acc[key]["final_eval_time"][1]
        total = queue + setup + oh + ev
        tot_by_run[name] += total
        print(f"{name:6} {step:>4} {n:>4} | {queue:>6.1f} {setup:>6.1f} {model:>7.1f} {cmd:>6.1f} {oh:>7.1f} {ev:>6.1f} {n_ev:>4} | {total:>7.1f}")
    print("-" * 92)
    print("mean per-rollout TOTAL by run (s):", {k: round(v / 5, 1) for k, v in sorted(tot_by_run.items())})


if __name__ == "__main__":
    main(sys.argv[1:])
