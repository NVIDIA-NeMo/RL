# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compare real vLLM paged attention with Megatron/TE on identical Q/K/V.

Run in the mcore environment with FP8_KV_VLLM_PYTHON pointing to the vLLM
environment (the two dependency extras are incompatible). For example:

    FP8_KV_VLLM_PYTHON=/path/to/vllm/bin/python python -m pytest -s \
        -m mcore tests/unit/models/megatron/test_fp8_kv_attention_parity.py

FP8_KV_REPORT_DIR optionally preserves per-case JSON diagnostics and tensors.
The FP64 score/softmax references are diagnostics, not replacements for either
backend. Fused kernels do not expose their intermediate softmax probabilities;
identity-valued V provides an observable probability readout, including the
kernel's value accumulation rounding. No assertion assumes fake quantization
must improve every input, or that different attention kernels are bitwise equal.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nemo_rl.models.megatron.fp8_kv_quantization import (
    fake_quantize_kv_per_token_head,
    quantize_kv_per_token_head,
)
from nemo_rl.models.megatron.fp8_kv_attention import scaled_fp8_kv_attention


def _reference(q, k, v):
    """FP64 causal GQA, with bottom-right alignment for decode/chunked prefill."""
    groups = q.shape[1] // k.shape[1]
    k = k.double().repeat_interleave(groups, dim=1)
    v = v.double().repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.double(), k) / q.shape[-1] ** 0.5
    q_pos = torch.arange(q.shape[0]) + k.shape[0] - q.shape[0]
    mask = torch.arange(k.shape[0])[None, :] > q_pos[:, None]
    scores.masked_fill_(mask[None], -torch.inf)
    probabilities = scores.softmax(dim=-1)
    out = torch.einsum("hqk,khd->qhd", probabilities, v)
    return scores, probabilities, out


def _error(actual, expected):
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    assert torch.isfinite(actual).all()
    return {
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
    }


@pytest.mark.parametrize("q_length", [1, 3])
def test_reference_causal_alignment_and_gqa(q_length):
    """A uniform-score example checks decode alignment and KV-head expansion."""
    q = torch.zeros(q_length, 4, 8)
    k = torch.zeros(3, 2, 8)
    v = torch.tensor([[1, 2], [10, 20], [100, 200]])[:, :, None].expand(-1, -1, 8)
    _, probabilities, output = _reference(q, k, v)
    expected = torch.tensor([[1, 1, 2, 2], [5.5, 5.5, 11, 11], [37, 37, 74, 74]])
    torch.testing.assert_close(
        output,
        expected[-q_length:, :, None].expand(-1, -1, 8).double(),
    )
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(4, q_length).double())


def _vllm_worker(input_path, output_path):
    """Execute actual cache-write and unified-attention kernels in a subprocess."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    data = torch.load(input_path, weights_only=True)
    q, k, v = (data[name].cuda() for name in ("q", "k", "v"))
    length, heads, dim = k.shape
    block_size = 32
    num_blocks = (length + block_size - 1) // block_size
    # Nonidentity physical pages and a partial final block catch layout errors.
    pages = torch.arange(num_blocks - 1, -1, -1, device="cuda")
    positions = torch.arange(length, device="cuda")
    slots = pages[positions // block_size] * block_size + positions % block_size
    shape = (num_blocks, block_size, heads, dim)
    cache_k = torch.zeros(shape, device="cuda", dtype=torch.float8_e4m3fn)
    cache_v = torch.zeros_like(cache_k)
    scales_k = torch.ones(shape[:-1], device="cuda", dtype=torch.float32)
    scales_v = torch.ones_like(scales_k)
    table = pages.int().unsqueeze(0)

    def write(value):
        triton_reshape_and_cache_flash_per_token_head_quant(
            k,
            value,
            cache_k,
            cache_v,
            scales_k,
            scales_v,
            slots,
            kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        )

    def attend(k_cache, v_cache, quantized, segmented=False):
        out = torch.empty_like(q)
        segment_kwargs = {}
        if segmented:
            from vllm.v1.attention.ops.triton_unified_attention import (
                is_batch_invariant,
            )

            assert len(q) == 1
            assert not is_batch_invariant, (
                "segmented test requires VLLM_BATCH_INVARIANT=0"
            )
            # Include inactive segments for short contexts, and test the FP32
            # partial sums plus the cross-segment softmax/output reduction.
            partial_shape = (len(q), q.shape[1], 4)
            segment_kwargs = {
                "seq_threshold_3D": 1,
                "num_par_softmax_segments": 4,
                "softmax_segm_output": torch.empty(
                    (*partial_shape, dim), device="cuda", dtype=torch.float32
                ),
                "softmax_segm_max": torch.empty(
                    partial_shape, device="cuda", dtype=torch.float32
                ),
                "softmax_segm_expsum": torch.empty(
                    partial_shape, device="cuda", dtype=torch.float32
                ),
            }
        unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=torch.tensor([0, len(q)], device="cuda", dtype=torch.int32),
            max_seqlen_q=len(q),
            seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
            max_seqlen_k=length,
            softmax_scale=dim**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            kv_quant_mode=(
                KVQuantMode.FP8_PER_TOKEN_HEAD if quantized else KVQuantMode.NONE
            ),
            k_scale_cache=scales_k if quantized else None,
            v_scale_cache=scales_v if quantized else None,
            **segment_kwargs,
        )
        return out.cpu()

    def paged(tensor):
        cache = torch.zeros(shape, device="cuda", dtype=tensor.dtype)
        cache.view(-1, heads, dim)[slots] = tensor
        return cache

    write(v)
    # Gather the exact stored bytes/scales; Megatron dequantizes THESE values.
    # FP32 represents every stored E4M3 value exactly and is easy to serialize.
    stored_k = cache_k.float().view(-1, heads, dim)[slots]
    stored_v = cache_v.float().view(-1, heads, dim)[slots]
    stored_ks = scales_k.view(-1, heads)[slots].unsqueeze(-1)
    stored_vs = scales_v.view(-1, heads)[slots].unsqueeze(-1)
    result = {
        "k_quant": stored_k.cpu(),
        "v_quant": stored_v.cpu(),
        "k_scale": stored_ks.cpu(),
        "v_scale": stored_vs.cpu(),
        "fp8": attend(cache_k, cache_v, True),
        "bf16": attend(paged(k), paged(v), False),
        "dequant_bf16": attend(
            paged((stored_k * stored_ks).to(q.dtype)),
            paged((stored_v * stored_vs).to(q.dtype)),
            False,
        ),
    }
    if len(q) == 1:
        result["fp8_segmented"] = attend(cache_k, cache_v, True, segmented=True)
    if length <= dim:
        basis = torch.eye(length, dim, device="cuda", dtype=q.dtype)
        basis = basis[:, None, :].expand(-1, heads, -1).contiguous()
        write(basis)
        result["probability_readout"] = attend(cache_k, cache_v, True)
        result["basis_quant"] = cache_v.float().view(-1, heads, dim)[slots].cpu()
        result["basis_scale"] = scales_v.view(-1, heads)[slots].unsqueeze(-1).cpu()
    torch.save(result, output_path)


@pytest.fixture(scope="module")
def megatron_attention_state(tmp_path_factory):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # Do not tear down process groups owned by other tests in a larger suite.
    owns_dist = not torch.distributed.is_initialized()
    owns_parallel = not parallel_state.model_parallel_is_initialized()
    if owns_dist:
        rendezvous = tmp_path_factory.mktemp("kv-attention-dist") / "init"
        torch.distributed.init_process_group(
            "nccl",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
    if owns_parallel:
        parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    try:
        yield
    finally:
        if owns_parallel:
            parallel_state.destroy_model_parallel()
        if owns_dist:
            torch.distributed.destroy_process_group()


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", [1234, 4321])
@pytest.mark.parametrize(
    "length,q_length,q_heads,kv_heads",
    [(17, 17, 4, 4), (65, 65, 8, 2), (65, 1, 8, 2), (257, 1, 8, 2)],
    ids=["prefill-mha", "prefill-gqa", "decode-gqa", "decode-multipage-gqa"],
)
def test_fp8_kv_attention_parity(
    tmp_path,
    megatron_attention_state,
    length,
    q_length,
    q_heads,
    kv_heads,
    seed,
    record_property,
):
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    vllm_python = os.environ.get("FP8_KV_VLLM_PYTHON")
    if not vllm_python:
        pytest.skip("set FP8_KV_VLLM_PYTHON to the vLLM environment's Python")
    dim = 128
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(q_length, q_heads, dim, generator=generator).bfloat16()
    k = torch.randn(length, kv_heads, dim, generator=generator).bfloat16()
    v = torch.randn(length, kv_heads, dim, generator=generator).bfloat16()
    # Token/head-dependent scales, outlier channels, zeros, and the scale floor.
    factors = torch.linspace(0.25, 2, length)[:, None, None]
    k = (k.float() * factors).bfloat16()
    v = (v.float() * factors.flip(0)).bfloat16()
    k[0].zero_()
    v[1].mul_(1e-8)
    k[-1, :, 0] = 8
    v[-1, :, -1] = -6
    input_path, output_path = tmp_path / "input.pt", tmp_path / "vllm.pt"
    torch.save({"q": q, "k": k, "v": v}, input_path)
    completed = subprocess.run(
        [
            vllm_python,
            str(Path(__file__).resolve()),
            "--vllm-worker",
            str(input_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rollout = torch.load(output_path, weights_only=True)
    # Validate the independent training quantizer against the real cache writer.
    for name, original in (("k", k), ("v", v)):
        quant, scale = quantize_kv_per_token_head(original.cuda())
        torch.testing.assert_close(
            quant.float().cpu(), rollout[name + "_quant"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale.cpu(), rollout[name + "_scale"], rtol=1e-7, atol=0
        )
    kd_exact = rollout["k_quant"].double() * rollout["k_scale"].double()
    vd_exact = rollout["v_quant"].double() * rollout["v_scale"].double()
    # Match the production FP32 multiply -> BF16 cast, using vLLM's scales.
    kd = (rollout["k_quant"] * rollout["k_scale"]).bfloat16()
    vd = (rollout["v_quant"] * rollout["v_scale"]).bfloat16()
    for original, dequant in ((k, kd), (v, vd)):
        torch.testing.assert_close(
            fake_quantize_kv_per_token_head(original.cuda()).cpu(),
            dequant,
            rtol=0,
            atol=0,
        )

    config = TransformerConfig(
        num_layers=1,
        hidden_size=q_heads * dim,
        num_attention_heads=q_heads,
        num_query_groups=kv_heads,
        kv_channels=dim,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    mask_type = (
        AttnMaskType.causal if q_length == length else AttnMaskType.causal_bottom_right
    )
    attention = (
        TEDotProductAttention(
            config,
            layer_number=1,
            attn_mask_type=mask_type,
            attention_type="self",
        )
        .cuda()
        .eval()
    )

    def megatron(key, value):
        with torch.no_grad():
            output = attention(
                q.cuda().unsqueeze(1),
                key.cuda().unsqueeze(1),
                value.cuda().unsqueeze(1),
                attention_mask=None,
                attn_mask_type=mask_type,
            )
        return output.reshape(q_length, q_heads, dim).cpu()

    train_quantized, train_original = megatron(kd, vd), megatron(k, v)
    with torch.no_grad():
        train_scales_inside = scaled_fp8_kv_attention(
            q.cuda(), k.cuda(), v.cuda()
        ).cpu()
    scores_exact, probs_exact, ref_exact = _reference(q, kd_exact, vd_exact)
    scores_bf16, probs_bf16, ref_bf16 = _reference(q, kd, vd)
    _, _, ref_original = _reference(q, k, v)
    finite = torch.isfinite(scores_exact)
    report = {
        "seed": seed,
        "kv_length": length,
        "q_length": q_length,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": dim,
        "dequant_rounding_k": _error(kd, kd_exact),
        "dequant_rounding_v": _error(vd, vd_exact),
        "scores_dequant_rounding": _error(scores_bf16[finite], scores_exact[finite]),
        "softmax_dequant_rounding": _error(probs_bf16, probs_exact),
        "output_dequant_rounding": _error(ref_bf16, ref_exact),
        "vllm_fp8_vs_megatron_quantized": _error(rollout["fp8"], train_quantized),
        "vllm_fp8_vs_megatron_unquantized": _error(rollout["fp8"], train_original),
        "vllm_fp8_vs_exact_dequant_reference": _error(rollout["fp8"], ref_exact),
        "megatron_quantized_vs_bf16_reference": _error(train_quantized, ref_bf16),
        "megatron_unquantized_vs_original_reference": _error(
            train_original, ref_original
        ),
        "vllm_bf16_vs_original_reference": _error(rollout["bf16"], ref_original),
        "vllm_dequant_bf16_vs_megatron_quantized": _error(
            rollout["dequant_bf16"], train_quantized
        ),
        "vllm_scale_placement_effect": _error(rollout["fp8"], rollout["dequant_bf16"]),
        "vllm_fp8_vs_megatron_scales_inside": _error(
            rollout["fp8"], train_scales_inside
        ),
        "megatron_scales_inside_vs_exact_reference": _error(
            train_scales_inside, ref_exact
        ),
    }
    if "fp8_segmented" in rollout:
        report["vllm_segmented_vs_exact_dequant_reference"] = _error(
            rollout["fp8_segmented"], ref_exact
        )
        report["vllm_segmented_vs_2d"] = _error(
            rollout["fp8_segmented"], rollout["fp8"]
        )
        report["vllm_segmented_vs_megatron_quantized"] = _error(
            rollout["fp8_segmented"], train_quantized
        )
        report["vllm_segmented_vs_megatron_unquantized"] = _error(
            rollout["fp8_segmented"], train_original
        )
        report["vllm_segmented_vs_megatron_scales_inside"] = _error(
            rollout["fp8_segmented"], train_scales_inside
        )
        assert report["vllm_segmented_vs_exact_dequant_reference"]["relative_l2"] < 0.03
    if length <= dim:
        basis = (
            torch.eye(length, dim)
            .bfloat16()[:, None, :]
            .expand(-1, kv_heads, -1)
            .contiguous()
        )
        basis_dequant = (rollout["basis_quant"] * rollout["basis_scale"]).bfloat16()
        torch.testing.assert_close(basis_dequant, basis, rtol=0, atol=0)
        basis_output = megatron(kd, basis_dequant)
        # Identity V turns the output into causal probabilities (plus accumulation rounding).
        expected_exact = probs_exact.permute(1, 0, 2)
        expected_bf16 = probs_bf16.permute(1, 0, 2)
        observed = rollout["probability_readout"][..., :length]
        report["vllm_probability_readout"] = _error(observed, expected_exact)
        report["megatron_probability_readout"] = _error(
            basis_output[..., :length], expected_bf16
        )
        torch.testing.assert_close(
            observed.double(), expected_exact, rtol=0.03, atol=0.003
        )
        torch.testing.assert_close(
            basis_output[..., :length].double(), expected_bf16, rtol=0.03, atol=0.003
        )
        torch.testing.assert_close(
            observed.sum(-1).float(), torch.ones(q_length, q_heads), rtol=0, atol=0.015
        )

    print("FP8_KV_ATTENTION " + json.dumps(report, sort_keys=True))
    record_property("fp8_kv_attention", json.dumps(report, sort_keys=True))
    report_dir = Path(os.environ.get("FP8_KV_REPORT_DIR", tmp_path))
    report_dir.mkdir(parents=True, exist_ok=True)
    name = f"kv{length}-q{q_length}-h{q_heads}-kvh{kv_heads}-seed{seed}"
    (report_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(
        {
            "inputs": {"q": q, "k": k, "v": v},
            "vllm": rollout,
            "megatron_quantized": train_quantized,
            "megatron_unquantized": train_original,
            "megatron_scales_inside": train_scales_inside,
        },
        report_dir / f"{name}.pt",
    )
    # Normwise budgets avoid unstable elementwise relative errors near zero.
    # Validate each backend against its own FP64 target; report cross-backend
    # comparisons without prescribing which training scenario must be closer.
    for name in (
        "vllm_fp8_vs_exact_dequant_reference",
        "megatron_quantized_vs_bf16_reference",
        "megatron_unquantized_vs_original_reference",
        "vllm_bf16_vs_original_reference",
        "megatron_scales_inside_vs_exact_reference",
    ):
        assert report[name]["relative_l2"] < 0.03, (name, report[name])


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--vllm-worker":
        raise SystemExit("use pytest, or --vllm-worker INPUT.pt OUTPUT.pt")
    _vllm_worker(sys.argv[2], sys.argv[3])
