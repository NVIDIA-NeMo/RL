#!/usr/bin/env python3
"""Build a NeMo-Gym train jsonl for SWE-bench_Verified instances that have a SIF
present on this cluster, matching the format of
Gym/responses_api_agents/swe_agents/data/example.jsonl (which IS SWE-bench_Verified).

Run on a node with `datasets` + internet (e.g. inside the nemo_rl_venv container):
    python make_swebench_jsonl.py            # build full jsonl (agent_ref=swe_agents_train)
    python make_swebench_jsonl.py --validate # only check format vs the shipped example record
"""
import json, os, sys

SIF_DIR = "/lustre/share/coreai_comparch_trtllm/verl-resource/gym_sifs/swebench"
SIF_PREFIX, SIF_SUFFIX = "swe-bench.eval.arm64.", ".sif"
OUT = "/lustre/fsw/coreai_comparch_trtllm/erinh/RL/swebench_verified_lyris_sif.jsonl"
DSN = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
# raw SWE-bench fields carried at top-level AND inside instance_dict (matches example.jsonl)
FIELDS = ["repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement",
          "hints_text", "created_at", "version", "FAIL_TO_PASS", "PASS_TO_PASS",
          "environment_setup_commit", "difficulty"]


def present_sif_ids():
    ids = set()
    for fn in os.listdir(SIF_DIR):
        if fn.startswith(SIF_PREFIX) and fn.endswith(SIF_SUFFIX):
            ids.add(fn[len(SIF_PREFIX):-len(SIF_SUFFIX)])
    return ids


def build_record(row, agent_name):
    raw = {k: row.get(k) for k in FIELDS}
    idict = dict(raw)
    idict["dataset_name"] = DSN
    idict["split"] = SPLIT
    idict["golden_patch"] = row.get("patch")
    metadata = {
        "instance_id": row["instance_id"],
        "base_commit": row.get("base_commit"),
        "dataset_name": DSN,
        "split": SPLIT,
        "problem_statement": row.get("problem_statement"),
        "golden_patch": row.get("patch"),
        "instance_dict": json.dumps(idict),
    }
    rcp = {
        "input": [],
        "metadata": metadata,
        "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "temperature": 0.7,
        "top_p": 0.8,
        "max_output_tokens": 12288,
    }
    rec = {"responses_create_params": rcp,
           "agent_ref": {"type": "responses_api_agents", "name": agent_name}}
    rec.update(raw)
    return rec


def norm(rec):
    """Structural compare helper: parse the embedded instance_dict so key-order doesn't matter."""
    r = json.loads(json.dumps(rec))
    md = r["responses_create_params"]["metadata"]
    md["instance_dict"] = json.loads(md["instance_dict"])
    return r


def main():
    from datasets import load_dataset
    validate_only = "--validate" in sys.argv
    sif_ids = present_sif_ids()
    sys.stderr.write(f"[build] present swebench SIFs: {len(sif_ids)}\n")
    ds = load_dataset(DSN, split=SPLIT)
    sys.stderr.write(f"[build] {DSN} {SPLIT} rows: {len(ds)}\n")
    by_id = {row["instance_id"]: row for row in ds}

    # --- self-validation: our astropy record must match the shipped example (agent_ref=swe_agents_val) ---
    ex_path = "/lustre/fsw/coreai_comparch_trtllm/erinh/RL/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/data/example.jsonl"
    ref = None
    if os.path.exists(ex_path):
        for line in open(ex_path):
            r = json.loads(line)
            if r.get("instance_id") == "astropy__astropy-12907":
                ref = r
                break
    if ref is not None and "astropy__astropy-12907" in by_id:
        mine = build_record(by_id["astropy__astropy-12907"], "swe_agents_val")
        if norm(mine) == norm(ref):
            sys.stderr.write("[validate] OK: generated astropy record matches example.jsonl exactly\n")
        else:
            sys.stderr.write("[validate] MISMATCH vs example.jsonl — dumping diffs\n")
            a, b = norm(mine), norm(ref)
            for k in set(a) | set(b):
                if a.get(k) != b.get(k):
                    sys.stderr.write(f"  top-level differs: {k}\n")
            ma = a["responses_create_params"]["metadata"]; mb = b["responses_create_params"]["metadata"]
            for k in set(ma) | set(mb):
                if ma.get(k) != mb.get(k):
                    sys.stderr.write(f"  metadata differs: {k}\n")
            ia = ma.get("instance_dict", {}); ib = mb.get("instance_dict", {})
            if isinstance(ia, dict) and isinstance(ib, dict):
                for k in set(ia) | set(ib):
                    if ia.get(k) != ib.get(k):
                        sys.stderr.write(f"  instance_dict differs: {k}\n")
            sys.exit(2)
    else:
        sys.stderr.write("[validate] (no reference record / astropy not in dataset — skipping format check)\n")

    if validate_only:
        return

    n = 0
    missing = sorted(sif_ids - set(by_id))
    with open(OUT, "w") as f:
        for iid in sorted(sif_ids):
            if iid not in by_id:
                continue
            f.write(json.dumps(build_record(by_id[iid], "swe_agents_train")) + "\n")
            n += 1
    sys.stderr.write(f"[build] wrote {n} records -> {OUT}\n")
    if missing:
        sys.stderr.write(f"[build] {len(missing)} SIFs had no SWE-bench_Verified row (skipped): {missing[:8]}{'...' if len(missing)>8 else ''}\n")


if __name__ == "__main__":
    main()
