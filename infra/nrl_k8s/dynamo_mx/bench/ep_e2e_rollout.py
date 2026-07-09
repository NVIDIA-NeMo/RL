"""Live end-to-end refit through the PRODUCTION path: EP Megatron trainer ->
real vLLM rollout engine, refitting via the production WeightTransferEngine
(MxVllmWeightUpdater + MxV2RefitReceiver + EP-gather) + MDL, then generating.

Unlike the receiver *harnesses* (ep_tp_receiver / ep_wire_bench), this drives the
actual production classes inside a live vLLM engine and proves generation
correctness — the integration, not just the transfer math.

Flow (greedy, deterministic) on a live vLLM engine:
  out0 = baseline generate
  setup: MxVllmWeightUpdater.initialize_weight_update_setup (per worker)
  refit #1 (cold): production update_weights (discover EP sources -> EP-gather +
                   global expert remap -> full HF) -> MDL builds map + loads
  out1 = generate                       [expect == out0: trainer has same weights]
  corrupt every parameter
  out2 = generate                       [expect GARBAGE]
  refit #2 (warm): production update_weights -> MDL in-place from map
  out3 = generate                       [expect == out0 iff the whole path is right]

Run inside a vLLM+MX worker pod (TP = tensor_parallel_size), with an EP Megatron
publisher (ep_publisher.py) already READY on the MX server. Env: MODEL_EXPRESS_URL,
TP (default 1), MODEL_ID.
"""
import os, sys, socket
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
# Leave GPU headroom for the EP-gather scratch + gathered HF weights (the refit
# pulls all experts across EP sources on TP1). NOTE: do NOT set
# expandable_segments — it remaps GPU VAs and invalidates NIXL/UCX RDMA memory
# registration (UCX "Local protection error" on RDMA_READ).
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.35"))

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TP = int(os.environ.get("TP", "1"))
MX_URL = os.environ["MODEL_EXPRESS_URL"]
PROMPT = "The capital of France is"


def _gen(llm):
    from vllm import SamplingParams
    return llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=48),
                        use_tqdm=False)[0].outputs[0].text


def setup(worker, mx_url, model):
    import torch
    from modelexpress.engines.vllm.weight_update import MxVllmWeightUpdater, MxInitInfo
    from modelexpress.engines.vllm.mdl import MdlLoader
    m = worker.model_runner.model
    dev = next(m.parameters()).device.index or 0
    rank = int(getattr(worker, "rank", 0) or 0)
    upd = MxVllmWeightUpdater()
    upd.initialize_weight_update_setup(MxInitInfo(
        mx_server_url=mx_url, model_name=model, worker_rank=rank,
        device_id=dev, same_rank_only=False,
    ))
    os.environ["MX_LOAD_MODE"] = "direct"
    worker._mx_upd = upd
    worker._mdl = MdlLoader(m)
    return {"rank": rank, "device": dev, "ready": upd.was_weight_update_setup_initialized()}


def refit(worker, model):
    import torch
    from modelexpress.engines.vllm.weight_update import MxUpdateInfo
    upd = worker._mx_upd
    upd.start_weight_update(1)
    n = {"n": 0}

    def _load(weights):
        n["n"] = len(weights)
        worker._mdl.load_weights(weights)

    upd.update_weights(
        MxUpdateInfo(version=1, min_version=1, moe_expert_filter=False, num_experts=0),
        load_weights=_load,
    )
    upd.finish_weight_update(1)
    torch.cuda.synchronize()
    m = worker._mdl
    return {"hf_tensors": n["n"], "direct": len(m._direct),
            "fused": len(m._fused), "expert": len(m._expert)}


def corrupt(worker):
    import torch
    with torch.no_grad():
        for p in worker.model_runner.model.parameters():
            p.data.normal_(mean=0.0, std=0.5)
    torch.cuda.synchronize()
    return {"corrupted": True}


def main():
    from vllm import LLM
    print(f"[e2e] loading vLLM {MODEL} TP={TP} (triton MoE) ...", flush=True)
    llm = LLM(model=MODEL, enforce_eager=True, tensor_parallel_size=TP,
              gpu_memory_utilization=GPU_UTIL, max_model_len=2048,
              trust_remote_code=True, moe_backend="triton")

    out0 = _gen(llm); print("OUT0 baseline :", repr(out0), flush=True)
    print("[setup]", llm.collective_rpc(setup, args=(MX_URL, MODEL))[0], flush=True)
    print("[refit cold]", llm.collective_rpc(refit, args=(MODEL,))[0], flush=True)
    out1 = _gen(llm); print("OUT1 post-cold:", repr(out1), flush=True)
    print("[corrupt]", llm.collective_rpc(corrupt)[0], flush=True)
    out2 = _gen(llm); print("OUT2 corrupted:", repr(out2), flush=True)
    print("[refit warm]", llm.collective_rpc(refit, args=(MODEL,))[0], flush=True)
    out3 = _gen(llm); print("OUT3 post-warm:", repr(out3), flush=True)

    print("\n==== VERDICT (live production-path refit: EP trainer -> vLLM rollout) ====")
    print("cold identity   (out1==out0):", out1 == out0)
    print("corruption took (out2!=out0):", out2 != out0)
    print("warm recovered  (out3==out0):", out3 == out0)
    ok = (out1 == out0) and (out2 != out0) and (out3 == out0)
    print("RESULT:", "PASS - live EP-trainer -> vLLM-rollout refit correct through production path"
          if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
