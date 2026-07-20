#!/usr/bin/env python3
"""Stream a (huge) rollout results.jsonl and emit a COMPACT per-request table with
EXACT prefill (prompt_str) and generated (generation_str) token counts.
Each function_call = one model request. Output: one jsonl row per request.
Usage: python3 extract_requests.py <results.jsonl> <out_requests.jsonl>
"""
import json
import sys
import time

from tokenizers import Tokenizer

MODEL_TOK = ("/lustre/fsw/portfolios/llmservice/users/pjin/devel/"
             "nemo-rl-ultra-v3-nano-opd-dev-20260513/results/"
             "mopd_ultrav3_to_nanov3_5_repro_v5_kd_opt_full-hsg-20260524-r1/step_18/hf/tokenizer.json")
# prompt_str is huge & re-grows every turn -> estimate prefill from chars (calibrated
# 3.66 chars/tok on 7.6M-token sample, stable ~2%). generated tokens are tokenized EXACTLY.
PROMPT_CPT = 3.66


def outcome(d, reward):
    if reward > 0.5 or d.get("resolved"):
        return "SOLVED"
    if d.get("agent_timed_out"):
        return "TIMEOUT"
    if d.get("patch_exists"):
        return "WRONG_PATCH"
    ek = d.get("agent_error_kind")
    return f"NO_PATCH:{ek}" if ek else "NO_PATCH:other"


def ntok(tok, s):
    return len(tok.encode(s).ids) if s else 0


def main(res, out):
    tok = Tokenizer.from_file(MODEL_TOK)
    fout = open(out, "w")
    t0 = time.time()
    nreq = 0
    for ridx, line in enumerate(open(res)):
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("full_result", {}) or {}
        resp = d.get("response", {}) or {}
        md = d.get("responses_create_params", {}).get("metadata", {})
        reward = r.get("reward", 0)
        base = dict(
            rollout=ridx,
            step=r.get("eval_step"),
            instance=md.get("instance_id", "?"),
            reward=reward,
            outcome=outcome(d, reward),
            run_time=round(d.get("openhands_run_time") or 0, 1),
            model_call_time=round(d.get("total_model_call_time") or 0, 1),
            cmd_exec_time=round(d.get("total_command_exec_time") or 0, 1),
        )
        fcs = [x for x in (resp.get("output") or []) if x.get("type") == "function_call"]
        for turn, fc in enumerate(fcs):
            pt = round(len(fc.get("prompt_str") or "") / PROMPT_CPT)   # prefill: char-ratio (fast)
            gt = ntok(tok, fc.get("generation_str") or "")             # generated: exact
            row = dict(base, turn=turn, prefill_tok=pt, gen_tok=gt, tool=fc.get("name"))
            fout.write(json.dumps(row) + "\n")
            nreq += 1
        if (ridx + 1) % 25 == 0:
            print(f"  {ridx+1} rollouts, {nreq} requests, {time.time()-t0:.0f}s", flush=True)
    fout.close()
    print(f"DONE: {ridx+1} rollouts, {nreq} requests -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
