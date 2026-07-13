"""MDL FP8 warm-path correctness for exact Qwen3-30B-A3B architecture.

Same corrupt -> warm-reload -> generate proof as the bf16 case, but on an
FP8-quantized MoE. The input must be either Qwen/Qwen3-30B-A3B-FP8 or a
vLLM-supported checkpoint produced by quantize_qwen3_30b_fp8.py.

Run (inside a vLLM+MX pod, deep_gemm off for fast iteration):
  VLLM_USE_DEEP_GEMM=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    python3 fp8_h1_test.py /mnt/.../Qwen3-30B-A3B-FP8-block
"""
import glob
import json
import os
import sys

os.environ["VLLM_USE_DEEP_GEMM"] = "0"
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-FP8"
PROMPT = "The capital of France is"


def _read_hf_weights(model_id):
    from safetensors.torch import safe_open
    if os.path.isdir(model_id):
        snap = model_id
    else:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json"],
        )
    out = []
    for sh in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(sh, framework="pt", device="cpu") as f:
            for k in f.keys():
                out.append((k, f.get_tensor(k)))
    return out


def _validate_checkpoint(model_id):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False).to_dict()
    expected = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    }
    actual = dict(config)
    if actual.get("num_experts") is None:
        actual["num_experts"] = actual.get("num_local_experts")
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        mismatches["quantization_config"] = {
            "expected": "vLLM-supported FP8 metadata",
            "actual": quant,
        }
    if mismatches:
        raise RuntimeError(
            "checkpoint is not exact Qwen3-30B-A3B FP8: "
            + json.dumps(mismatches, sort_keys=True)
        )


def mdl_setup(worker, model_id):
    import os as _o, torch
    from modelexpress.engines.vllm.mdl import MdlLoader
    model = worker.model_runner.model
    n_fp8 = sum(1 for _, p in model.named_parameters() if p.dtype == torch.float8_e4m3fn)
    weights = _read_hf_weights(model_id)
    worker._mx_w = weights
    _o.environ["MX_LOAD_MODE"] = "direct"
    worker._mx_mdl = MdlLoader(model)
    worker._mx_mdl.load_weights(weights)
    torch.cuda.synchronize()
    return {"params": len(dict(model.named_parameters())), "fp8_params": n_fp8,
            "checkpoint_tensors": len(weights)}


def corrupt(worker):
    import torch
    with torch.no_grad():
        for p in worker.model_runner.model.parameters():
            if p.dtype == torch.float8_e4m3fn:
                p.data.copy_((torch.randn(p.shape, device=p.device) * 0.5).to(torch.float8_e4m3fn))
            else:
                p.data.normal_(mean=0.0, std=0.5)
    torch.cuda.synchronize()
    return {"corrupted": True}


def warm_reload(worker):
    import torch
    m = worker._mx_mdl
    m.load_weights(worker._mx_w)
    torch.cuda.synchronize()
    return {"direct": len(m._direct), "fused": len(m._fused), "expert": len(m._expert)}


def _gen(llm):
    from vllm import SamplingParams
    return llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=40),
                        use_tqdm=False)[0].outputs[0].text


def main():
    from vllm import LLM
    tp_size = int(os.environ.get("TP", os.environ.get("TENSOR_PARALLEL_SIZE", "1")))
    _validate_checkpoint(MODEL_ID)
    print(f"[load] {MODEL_ID} (language_model_only, triton, tp={tp_size}) ...", flush=True)
    llm = LLM(model=MODEL_ID, enforce_eager=True, tensor_parallel_size=tp_size,
              language_model_only=True, moe_backend="triton",
              gpu_memory_utilization=0.9, max_model_len=4096, trust_remote_code=True)
    out0 = _gen(llm); print("OUT0 baseline :", repr(out0), flush=True)
    print("[setup]", llm.collective_rpc(mdl_setup, args=(MODEL_ID,))[0], flush=True)
    out1 = _gen(llm); print("OUT1 post-cold:", repr(out1), flush=True)
    print("[corrupt]", llm.collective_rpc(corrupt)[0], flush=True)
    out2 = _gen(llm); print("OUT2 corrupted:", repr(out2), flush=True)
    print("[warm]", llm.collective_rpc(warm_reload)[0], flush=True)
    out3 = _gen(llm); print("OUT3 post-warm:", repr(out3), flush=True)
    ok = (out1 == out0) and (out2 != out0) and (out3 == out0)
    print("\nRESULT:", "PASS - fp8 warm path correct" if ok else "FAIL - fp8 warm path needs fix (H1)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
