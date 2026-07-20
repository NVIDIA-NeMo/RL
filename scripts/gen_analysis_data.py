#!/usr/bin/env python3
"""
gen_analysis_data.py — turn a NeMo-Gym SWE rollout-benchmark results file into
analysis-ready tables anyone can load into pandas / R / Excel.

INPUT
  <run>/nemo/nemo_gym_eval_results.jsonl   (one JSON row per rollout/trajectory; can be many GB)

OUTPUT (written to --outdir)
  trajectories.csv    one row per rollout        (e.g. 320 for a 5-step x 4-prompt x 16-gen run)
  requests.csv        one row per model request  (each 'function_call' turn; ~18k rows)
  DATA_DICTIONARY.md  column descriptions

HOW IT WORKS
  Each rollout is full_result.response.output = a list of items. Every item with
  type=='function_call' is ONE model request and carries:
     prompt_str      the full context sent that turn  -> PREFILL tokens
     generation_str  the model's decoded output       -> GENERATED tokens
  prompt_str re-grows every turn (context accumulates), so PREFILL is estimated from
  char count x a calibrated chars/tok ratio (fast, ~2%). GENERATED tokens are exact if a
  --tokenizer (HF tokenizer.json) is supplied, else also char-ratio.

USAGE
  python3 gen_analysis_data.py RESULTS.jsonl --outdir ./analysis_data \
      [--tokenizer /path/to/model/tokenizer.json] [--prompt-cpt 3.66] [--gen-cpt 3.89]

  # then, e.g. in pandas:
  #   import pandas as pd
  #   traj = pd.read_csv('analysis_data/trajectories.csv')
  #   req  = pd.read_csv('analysis_data/requests.csv')
"""
import argparse
import csv
import json
import os
import sys
import time

TRAJ_COLS = [
    "rollout_id", "eval_step", "instance_id", "gen_index",
    "reward", "resolved", "patch_exists", "outcome",
    "n_turns", "gen_tok_total", "prefill_tok_sum", "final_ctx_tok", "ctx_growth_tok",
    "run_time_s", "model_call_time_s", "cmd_exec_time_s", "runtime_setup_s", "queue_s",
    "agent_error_kind", "agent_timed_out",
]
REQ_COLS = [
    "rollout_id", "eval_step", "instance_id", "gen_index",
    "reward", "outcome", "turn", "prefill_tok", "gen_tok", "tool",
]

DATA_DICTIONARY = """# Data dictionary

## trajectories.csv — one row per rollout (SWE agent attempt on one instance)
| column | meaning |
|---|---|
| rollout_id | 0-based index of the rollout in the results file |
| eval_step | benchmark step (each step = num_prompts_per_step instances) |
| instance_id | SWE-bench instance (e.g. astropy__astropy-12907) |
| gen_index | which of the N generations for this instance (temperature sampling) |
| reward | 1.0 if the instance was resolved (tests pass), else 0.0 |
| resolved | bool, same as reward>0.5 |
| patch_exists | bool: the agent produced a code patch (may still fail tests) |
| outcome | SOLVED / WRONG_PATCH (patch, tests failed) / NO_PATCH:<reason> / TIMEOUT |
| n_turns | number of model requests (function_call turns) in the trajectory |
| gen_tok_total | total generated (decoded) tokens across all turns |
| prefill_tok_sum | sum of prefill tokens over turns = RAW re-prefill (no KV cache) |
| final_ctx_tok | prefill tokens of the LAST request = final context size |
| ctx_growth_tok | final_ctx_tok - first-request prefill = context accumulated |
| run_time_s | agent wall-clock (openhands_run_time) |
| model_call_time_s | time spent in model generation (total_model_call_time) |
| cmd_exec_time_s | time running shell/test commands in the SWE container |
| runtime_setup_s | connect + initialize + apptainer spinup |
| queue_s | time queued before the rollout started (ray_queue_time) |
| agent_error_kind | max_iteration / context_window / other / null |
| agent_timed_out | bool: hit swebench_agent_timeout (wall-clock cap) |

## requests.csv — one row per model request (one agent turn)
| column | meaning |
|---|---|
| rollout_id, eval_step, instance_id, gen_index | join keys back to trajectories.csv |
| reward, outcome | copied from the parent rollout (for easy filtering) |
| turn | 0-based request index within the trajectory |
| prefill_tok | context tokens sent this request (grows with turn) |
| gen_tok | tokens generated this request |
| tool | tool the model called this turn (think/bash/str_replace/finish/...) |

NOTES
- PREFILL is char-ratio estimated (prompt_str re-grows each turn -> exact tokenization is
  ~3B chars). GENERATED is exact when --tokenizer is given. See script header.
- With KV/prefix caching the effective prefill per turn is ~ the per-turn ctx growth, not
  prefill_tok; prefill_tok_sum is the no-cache upper bound.
"""


def get_tok(path):
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(path)
    except Exception as e:
        sys.stderr.write(f"[warn] could not load tokenizer ({e}); using char-ratio for gen too\n")
        return None


def outcome(d, reward):
    if reward > 0.5 or d.get("resolved"):
        return "SOLVED"
    if d.get("agent_timed_out"):
        return "TIMEOUT"
    if d.get("patch_exists"):
        return "WRONG_PATCH"
    ek = d.get("agent_error_kind")
    return f"NO_PATCH:{ek}" if ek else "NO_PATCH:other"


def main():
    ap = argparse.ArgumentParser(description="Generate analysis-ready CSVs from a rollout results.jsonl")
    ap.add_argument("results")
    ap.add_argument("--outdir", default="./analysis_data")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer.json for exact generated-token counts")
    ap.add_argument("--prompt-cpt", type=float, default=3.66, help="chars/token for prefill estimate")
    ap.add_argument("--gen-cpt", type=float, default=3.89, help="chars/token for gen fallback")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tok = get_tok(a.tokenizer)

    def gtok(s):
        if not s:
            return 0
        return len(tok.encode(s).ids) if tok else round(len(s) / a.gen_cpt)

    ft = open(os.path.join(a.outdir, "trajectories.csv"), "w", newline="")
    fr = open(os.path.join(a.outdir, "requests.csv"), "w", newline="")
    wt = csv.writer(ft); wt.writerow(TRAJ_COLS)
    wr = csv.writer(fr); wr.writerow(REQ_COLS)
    t0 = time.time(); nreq = 0
    for rid, line in enumerate(open(a.results)):
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("full_result", {}) or {}
        resp = d.get("response", {}) or {}
        md = d.get("responses_create_params", {}).get("metadata", {})
        rew = r.get("reward", 0)
        oc = outcome(d, rew)
        iid = md.get("instance_id", "?")
        gi = r.get("generation_index")
        fcs = [x for x in (resp.get("output") or []) if x.get("type") == "function_call"]
        pref = [round(len(x.get("prompt_str") or "") / a.prompt_cpt) for x in fcs]
        gen = [gtok(x.get("generation_str") or "") for x in fcs]
        for turn, fc in enumerate(fcs):
            wr.writerow([rid, r.get("eval_step"), iid, gi, rew, oc, turn, pref[turn], gen[turn], fc.get("name")])
            nreq += 1
        setup = sum((d.get(k) or 0) for k in
                    ("connect_to_runtime_time", "initialize_runtime_time", "generation_apptainer_spinup_time"))
        wt.writerow([
            rid, r.get("eval_step"), iid, gi, rew, bool(d.get("resolved")), bool(d.get("patch_exists")), oc,
            len(fcs), sum(gen), sum(pref), (pref[-1] if pref else 0), (pref[-1] - pref[0] if pref else 0),
            round(d.get("openhands_run_time") or 0, 1), round(d.get("total_model_call_time") or 0, 1),
            round(d.get("total_command_exec_time") or 0, 1), round(setup, 1), round(d.get("ray_queue_time") or 0, 2),
            d.get("agent_error_kind"), bool(d.get("agent_timed_out")),
        ])
        if (rid + 1) % 50 == 0:
            print(f"  {rid+1} rollouts, {nreq} requests, {time.time()-t0:.0f}s", flush=True)
    ft.close(); fr.close()
    open(os.path.join(a.outdir, "DATA_DICTIONARY.md"), "w").write(DATA_DICTIONARY)
    print(f"DONE: {rid+1} trajectories, {nreq} requests -> {a.outdir}/ "
          f"(trajectories.csv, requests.csv, DATA_DICTIONARY.md) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
