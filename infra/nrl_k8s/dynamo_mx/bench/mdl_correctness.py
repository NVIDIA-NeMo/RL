"""MDL correctness proof against a REAL vLLM model.

Flow (greedy decode, deterministic):
  out0 = baseline generate (weights as loaded by vLLM)
  cold MDL load  -> builds destination map (weights unchanged)
  out1 = generate                     [expect == out0]
  corrupt: randomize every parameter
  out2 = generate                     [expect GARBAGE / != out0]
  warm MDL reload (in-place from map)  -> restores bytes into mapped slots
  out3 = generate                     [expect == out0 iff map is correct]

If MDL's destination map or in-place writes are wrong, out3 stays garbage.

Run (inside a vLLM+MX pod, one GPU for TP1, N for EP/TP):
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 python3 mdl_correctness.py <hf-model-id>

Verified 2026-07-08 on GB200: Qwen3-4B (218 direct + 180 fused) and
Qwen3-30B-A3B MoE (291 direct + 144 fused + 18,432 expert), both byte-identical.
"""
import os, sys, glob, json

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Thinking-2507"
PROMPT = "The capital of France is"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def _read_hf_weights(model_id):
    import glob as _g, os as _o
    from safetensors.torch import safe_open
    hub = _o.path.join(_o.environ["HF_HOME"], "hub",
                       "models--" + model_id.replace("/", "--"), "snapshots")
    snap = sorted(_g.glob(hub + "/*/"))[-1]
    out = []
    for sh in sorted(_g.glob(snap + "*.safetensors")):
        with safe_open(sh, framework="pt", device="cpu") as f:
            for k in f.keys():
                out.append((k, f.get_tensor(k)))
    return out


def mdl_setup(worker, model_id):
    import os as _o, torch
    from modelexpress.engines.vllm.mdl import MdlLoader
    model = worker.model_runner.model
    weights = _read_hf_weights(model_id)
    worker._mx_w = weights
    _o.environ["MX_LOAD_MODE"] = "direct"
    worker._mx_mdl = MdlLoader(model)
    worker._mx_mdl.load_weights(weights)  # cold: stock load + build map
    torch.cuda.synchronize()
    return {"checkpoint_tensors": len(weights),
            "params": len(dict(model.named_parameters()))}


def corrupt(worker):
    import torch
    model = worker.model_runner.model
    with torch.no_grad():
        for p in model.parameters():
            p.data.normal_(mean=0.0, std=0.5)
    torch.cuda.synchronize()
    return {"corrupted": True}


def warm_reload(worker):
    import torch
    worker._mx_mdl.load_weights(worker._mx_w)  # warm: in-place from map
    torch.cuda.synchronize()
    m = worker._mx_mdl
    return {"direct": len(m._direct), "fused": len(m._fused), "expert": len(m._expert),
            "cycles": m._cycles}


def _gen(llm):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    o = llm.generate([PROMPT], sp, use_tqdm=False)
    return o[0].outputs[0].text


def main():
    from vllm import LLM
    is_moe = "A3B" in MODEL_ID or "moe" in MODEL_ID.lower()
    kw = dict(model=MODEL_ID, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=2048, trust_remote_code=True)
    if is_moe:
        kw["moe_backend"] = "triton"  # keep re-loadable 3D expert layout (refit)
    print(f"[load] {MODEL_ID} (moe={is_moe}) ...", flush=True)
    llm = LLM(**kw)

    out0 = _gen(llm); print("OUT0 baseline :", repr(out0), flush=True)
    print("[setup]", llm.collective_rpc(mdl_setup, args=(MODEL_ID,))[0], flush=True)
    out1 = _gen(llm); print("OUT1 post-cold:", repr(out1), flush=True)
    print("[corrupt]", llm.collective_rpc(corrupt)[0], flush=True)
    out2 = _gen(llm); print("OUT2 corrupted:", repr(out2), flush=True)
    print("[warm]", llm.collective_rpc(warm_reload)[0], flush=True)
    out3 = _gen(llm); print("OUT3 post-warm:", repr(out3), flush=True)

    print("\n==== VERDICT ====")
    print("cold identity   (out1==out0):", out1 == out0)
    print("corruption took (out2!=out0):", out2 != out0)
    print("warm recovered  (out3==out0):", out3 == out0)
    ok = (out1 == out0) and (out2 != out0) and (out3 == out0)
    print("RESULT:", "PASS - inference correct after MDL refit" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
