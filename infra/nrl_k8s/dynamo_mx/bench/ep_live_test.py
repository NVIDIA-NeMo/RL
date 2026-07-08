"""Live EP>1 validation on a real vLLM expert-parallel engine (Qwen3-30B-A3B).

Two things this proves that transport/unit tests can't:
  1. PLACEMENT PARITY: vLLM's actual per-rank global-expert assignment on a
     real EP engine == our compute_local_expert_ids(..., "linear"). If these
     diverge, the production EP filter pulls the wrong experts.
  2. REFIT CORRECTNESS on an EP-sharded model: corrupt every param, warm MDL
     reload, generation returns byte-identical.

Run on an N-GPU node (N = EP size):
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 python3 ep_live_test.py 4

Verified 2026-07-08 on a 4-GPU GB200 node: EP=4 placement matched on all ranks
(rank r -> experts [r*32,(r+1)*32)); corrupt->warm-reload byte-identical
(rank 0: 241 direct + 4,608 expert writes). RESULT: PASS.
"""
import os, sys
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PROMPT = "The capital of France is"


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


def introspect_ep(worker):
    model = worker.model_runner.model
    info = {"ep_rank": None, "ep_size": None, "num_experts": None, "local_ids": None}
    for name, mod in model.named_modules():
        em = getattr(mod, "expert_map", None)
        if em is not None and hasattr(em, "numel") and em.numel() > 0:
            gids = [g for g in range(int(em.numel())) if int(em[g].item()) >= 0]
            info.update(ep_rank=int(getattr(mod, "ep_rank", -1)),
                        ep_size=int(getattr(mod, "ep_size", -1)),
                        num_experts=int(em.numel()), local_ids=sorted(gids))
            break
    return info


def mdl_setup(worker, model_id):
    import os as _o, torch
    from modelexpress.engines.vllm.mdl import MdlLoader
    model = worker.model_runner.model
    weights = _read_hf_weights(model_id)
    worker._mx_w = weights
    _o.environ["MX_LOAD_MODE"] = "direct"
    worker._mx_mdl = MdlLoader(model)
    worker._mx_mdl.load_weights(weights)
    torch.cuda.synchronize()
    return {"n_params": len(dict(model.named_parameters()))}


def corrupt(worker):
    import torch
    with torch.no_grad():
        for p in worker.model_runner.model.parameters():
            p.data.normal_(mean=0.0, std=0.5)
    torch.cuda.synchronize()
    return {"corrupted": True}


def warm_reload(worker):
    import torch
    worker._mx_mdl.load_weights(worker._mx_w)
    torch.cuda.synchronize()
    m = worker._mx_mdl
    return {"direct": len(m._direct), "fused": len(m._fused), "expert": len(m._expert)}


def _gen(llm):
    from vllm import SamplingParams
    return llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=48),
                        use_tqdm=False)[0].outputs[0].text


def main():
    from vllm import LLM
    from modelexpress.rl_expert_layout import compute_local_expert_ids
    print(f"[load] {MODEL_ID} EP={EP} (enable_expert_parallel, tp={EP}, triton) ...", flush=True)
    llm = LLM(model=MODEL_ID, enforce_eager=True, tensor_parallel_size=EP,
              enable_expert_parallel=True, moe_backend="triton",
              gpu_memory_utilization=0.85, max_model_len=2048, trust_remote_code=True)

    out0 = _gen(llm); print("OUT0 baseline :", repr(out0), flush=True)

    infos = llm.collective_rpc(introspect_ep)
    print("\n==== EP PLACEMENT PARITY ====")
    parity_ok = True
    for info in infos:
        if info["local_ids"] is None:
            print("  (a rank reported no expert_map)"); parity_ok = False; continue
        ours = list(compute_local_expert_ids(info["ep_rank"], info["ep_size"],
                                             info["num_experts"], "linear"))
        match = sorted(ours) == info["local_ids"]
        parity_ok = parity_ok and match
        print(f"  rank {info['ep_rank']}/{info['ep_size']}: vLLM {len(info['local_ids'])} "
              f"local {info['local_ids'][:3]}..{info['local_ids'][-1]}; "
              f"ours {ours[:3]}..{ours[-1]}; MATCH={match}")

    print("\n==== REFIT CORRECTNESS (EP engine) ====")
    print("[setup]", llm.collective_rpc(mdl_setup, args=(MODEL_ID,))[0], flush=True)
    out1 = _gen(llm); print("OUT1 post-cold:", repr(out1), flush=True)
    print("[corrupt]", llm.collective_rpc(corrupt)[0], flush=True)
    out2 = _gen(llm); print("OUT2 corrupted:", repr(out2), flush=True)
    print("[warm rank0]", llm.collective_rpc(warm_reload)[0], flush=True)
    out3 = _gen(llm); print("OUT3 post-warm:", repr(out3), flush=True)

    print("\n==== VERDICT ====")
    print("placement parity:", parity_ok)
    print("cold identity   (out1==out0):", out1 == out0)
    print("corruption took (out2!=out0):", out2 != out0)
    print("warm recovered  (out3==out0):", out3 == out0)
    ok = parity_ok and out1 == out0 and out2 != out0 and out3 == out0
    print("RESULT:", "PASS - EP>1 placement + refit correct" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
