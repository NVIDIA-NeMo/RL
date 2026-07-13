"""Qwen3-30B FP8 partial-refit value and generation correctness gate.

Loads the real block-FP8 checkpoint, builds MDL's full destination map, corrupts
only layers 0-5 in the live vLLM model, and reloads only the corresponding HF
tensors. The gate proves:

* selected destination parameters change after corruption and are rewritten;
* an unselected sentinel layer remains bit-exactly unchanged; and
* deterministic generation returns to the baseline.

The result also reports whether the complete live selected-parameter hash returns
bit-exactly. That is diagnostic rather than pass/fail because vLLM can rewrite
derived/absorbed parameters during the first forward.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys

os.environ["VLLM_USE_DEEP_GEMM"] = "0"
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

MODEL_ID = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/mnt/rl-workspace/kavink/Qwen3-30B-A3B-FP8-block"
)
SELECTED_LAYERS = set(range(6))
SENTINEL_LAYERS = {6}
PROMPT = "The capital of France is"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _read_hf_weights(model_id):
    from safetensors.torch import safe_open

    out = []
    for shard in sorted(glob.glob(os.path.join(model_id, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                out.append((name, handle.get_tensor(name)))
    return out


def _layer(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _hash_model_layers(model, layers: set[int]) -> tuple[str, int]:
    """Hash every byte of parameters owned by the requested local layers."""

    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if _layer(name) not in layers:
            continue
        raw = parameter.detach().contiguous().view(__import__("torch").uint8).cpu()
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(raw.numpy().tobytes())
        count += 1
    return digest.hexdigest(), count


def setup(worker, model_id):
    import torch

    from modelexpress.engines.vllm.mdl import MdlLoader

    model = worker.model_runner.model
    weights = _read_hf_weights(model_id)
    os.environ["MX_LOAD_MODE"] = "direct"
    worker._mx_partial_weights = weights
    worker._mx_partial_mdl = MdlLoader(model)
    worker._mx_partial_mdl.load_weights(weights)
    torch.cuda.synchronize()
    return {"checkpoint_tensors": len(weights)}


def capture_hashes(worker):
    """Capture after generation so lazy/derived first-forward state is stable."""

    model = worker.model_runner.model
    selected_hash, selected_count = _hash_model_layers(model, SELECTED_LAYERS)
    sentinel_hash, sentinel_count = _hash_model_layers(model, SENTINEL_LAYERS)
    return {
        "selected_hash": selected_hash,
        "selected_params": selected_count,
        "sentinel_hash": sentinel_hash,
        "sentinel_params": sentinel_count,
    }


def corrupt_selected(worker):
    import torch

    model = worker.model_runner.model
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if _layer(name) not in SELECTED_LAYERS:
                continue
            if parameter.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                value = (torch.randn(parameter.shape, device=parameter.device) * 0.5).to(
                    parameter.dtype
                )
                parameter.data.copy_(value)
            else:
                parameter.data.normal_(mean=0.0, std=0.5)
    torch.cuda.synchronize()
    selected_hash, _ = _hash_model_layers(model, SELECTED_LAYERS)
    sentinel_hash, _ = _hash_model_layers(model, SENTINEL_LAYERS)
    return {"selected_hash": selected_hash, "sentinel_hash": sentinel_hash}


def reload_selected(worker):
    import torch

    selected_weights = [
        (name, tensor)
        for name, tensor in worker._mx_partial_weights
        if _layer(name) in SELECTED_LAYERS
    ]
    worker._mx_partial_mdl.load_weights(selected_weights)
    torch.cuda.synchronize()
    return {"source_tensors": len(selected_weights)}


def _generate(llm):
    from vllm import SamplingParams

    return llm.generate(
        [PROMPT],
        SamplingParams(temperature=0.0, max_tokens=40),
        use_tqdm=False,
    )[0].outputs[0].text


def main() -> int:
    from vllm import LLM

    tp_size = int(os.environ.get("TP", "2"))
    llm = LLM(
        model=MODEL_ID,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        language_model_only=True,
        moe_backend="triton",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )
    baseline = _generate(llm)
    setup_rows = llm.collective_rpc(setup, args=(MODEL_ID,))
    post_cold = _generate(llm)
    baseline_rows = llm.collective_rpc(capture_hashes)
    corrupted_rows = llm.collective_rpc(corrupt_selected)
    corrupted = _generate(llm)
    reload_rows = llm.collective_rpc(reload_selected)
    restored = _generate(llm)
    restored_rows = llm.collective_rpc(capture_hashes)

    ranks_ok = all(
        corrupt["selected_hash"] != initial["selected_hash"]
        and restored_row["selected_hash"] != corrupt["selected_hash"]
        and corrupt["sentinel_hash"] == initial["sentinel_hash"]
        and restored_row["sentinel_hash"] == initial["sentinel_hash"]
        for initial, corrupt, restored_row in zip(
            baseline_rows, corrupted_rows, restored_rows, strict=True
        )
    )
    selected_live_hash_bit_exact = all(
        restored_row["selected_hash"] == initial["selected_hash"]
        for initial, restored_row in zip(
            baseline_rows, restored_rows, strict=True
        )
    )
    generation_ok = (
        post_cold == baseline and corrupted != baseline and restored == baseline
    )
    result = {
        "status": "pass" if ranks_ok and generation_ok else "fail",
        "tp_size": tp_size,
        "selected_layers": sorted(SELECTED_LAYERS),
        "sentinel_layers": sorted(SENTINEL_LAYERS),
        "rank_value_checks": ranks_ok,
        # vLLM may rewrite derived/absorbed live parameters on first forward;
        # generation recovery is authoritative. Keep this stricter diagnostic
        # visible without conflating derived-state bytes with source-weight
        # correctness.
        "selected_live_hash_bit_exact": selected_live_hash_bit_exact,
        "generation_check": generation_ok,
        "setup": setup_rows,
        "baseline": baseline_rows,
        "corrupted": corrupted_rows,
        "reload": reload_rows,
        "restored": restored_rows,
        "outputs": {
            "baseline": baseline,
            "post_cold": post_cold,
            "corrupted": corrupted,
            "restored": restored,
        },
    }
    print("FP8_PARTIAL_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
