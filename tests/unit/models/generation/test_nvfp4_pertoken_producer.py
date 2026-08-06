"""Producer and refit-filter tests for the NVFP4 per-token rollout path."""

import types

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.models.generation.vllm.quantization import nvfp4_pertoken as M

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _vllm_reference():
    try:
        from vllm.model_executor.layers.quantization.online.nvfp4 import (  # noqa: PLC0415
            _quantize_moe_weight_to_nvfp4,
        )

        return _quantize_moe_weight_to_nvfp4
    except ImportError:
        return None


# ------------------------------------------------------------- producer


@cuda_only
def test_bitwise_vs_vllm_nvfp4_quant_kernel():
    """Verify that producer output matches vLLM's kernel bit for bit."""
    ref = _vllm_reference()
    if ref is None:
        pytest.skip("vLLM online-quant module unavailable")

    torch.manual_seed(0)
    for e, n, k in [(8, 256, 512), (4, 64, 128), (2, 96, 2048)]:
        w = torch.randn(e, n, k, dtype=torch.bfloat16, device="cuda") * 3.0
        q_ref, bs_ref, gs_ref = ref(w)
        q, bs, gs = M.quantize_nvfp4_weight(w)

        assert torch.equal(gs, gs_ref), f"global scales differ ({e},{n},{k})"
        assert torch.equal(bs.view(torch.uint8), bs_ref.view(torch.uint8)), (
            f"block scales differ ({e},{n},{k})"
        )
        mismatch = (q != q_ref).sum().item()
        assert mismatch == 0, (
            f"packed weights differ at {mismatch}/{q.numel()} bytes ({e},{n},{k})"
        )


@cuda_only
def test_2d_matches_3d_per_expert():
    torch.manual_seed(1)
    w = torch.randn(4, 128, 256, dtype=torch.bfloat16, device="cuda")
    q3, bs3, gs3 = M.quantize_nvfp4_weight(w)
    for e in range(4):
        q2, bs2, gs2 = M.quantize_nvfp4_weight(w[e])
        assert torch.equal(q2, q3[e])
        assert torch.equal(bs2.view(torch.uint8), bs3[e].view(torch.uint8))
        assert torch.equal(gs2, gs3[e])


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        M.quantize_nvfp4_weight(torch.randn(16))
    with pytest.raises(ValueError, match="multiple of 16"):
        M.quantize_nvfp4_weight(torch.randn(4, 20))  # K % 16 != 0


@cuda_only
def test_zero_block_yields_zero_codes():
    w = torch.zeros(16, 32, dtype=torch.bfloat16, device="cuda")
    w[0, 16:] = 1.0  # non-zero amax so global scale is finite
    q, bs, _ = M.quantize_nvfp4_weight(w)
    assert (q[0, :8] == 0).all()  # the all-zero block packs to zero codes


# ------------------------------------------------------------- refit filter


def _expert_stream(num_experts=4, n=32, k=64, layers=("model.layers.0",)):
    stream = []
    for layer in layers:
        for e in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                shape = (n, k) if proj != "down_proj" else (k, n * 2)
                stream.append(
                    (f"{layer}.mlp.experts.{e}.{proj}.weight", torch.randn(*shape))
                )
    return stream


def test_filter_emits_fused_tensors_and_passes_rest():
    stream = _expert_stream(num_experts=4, n=32, k=64)
    stream.insert(0, ("model.layers.0.self_attn.q_proj.weight", torch.randn(8, 8)))
    stream.append(("model.layers.0.self_attn.attn.k_scale", torch.tensor(1.0)))
    out = dict(
        M.iter_nvfp4_pertoken_weights(iter(stream), quant_patterns=["*.experts.*"])
    )
    p = "model.layers.0.mlp.experts"
    assert out[f"{p}.w13_weight"].shape == (4, 64, 32)  # (E, 2N, K/2)
    assert out[f"{p}.w13_weight"].dtype == torch.uint8
    assert out[f"{p}.w13_weight_scale"].shape == (4, 64, 4)  # (E, 2N, K/16)
    assert out[f"{p}.w13_weight_scale_2"].shape == (4, 2)
    assert out[f"{p}.w2_weight"].shape == (4, 64, 32)  # (E, K, 2N/2)
    assert out[f"{p}.w2_weight_scale_2"].shape == (4,)
    assert torch.equal(out["model.layers.0.self_attn.q_proj.weight"], stream[0][1])
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in out


def test_filter_w13_shares_one_global_scale_per_expert():
    """Gate+up must be quantized under ONE per-expert global scale.

    vLLM's ModelOptNvFp4FusedMoE.process_weights_after_loading keeps only
    w13_weight_scale_2[:, 0] for the whole fused tensor — per-projection
    scales silently decode the up half with the gate scale.
    """
    stream = _expert_stream(num_experts=2, n=16, k=32)
    tensors = {n: t for n, t in stream}
    out = dict(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))
    p = "model.layers.0.mlp.experts"
    s2 = out[f"{p}.w13_weight_scale_2"]
    assert torch.equal(s2[:, 0], s2[:, 1])
    for e in range(2):
        fused = torch.cat(
            [
                tensors[f"{p}.{e}.gate_proj.weight"],
                tensors[f"{p}.{e}.up_proj.weight"],
            ],
            dim=0,
        )
        fq, _, fs2 = M.quantize_nvfp4_weight(fused)
        assert torch.equal(out[f"{p}.w13_weight"][e], fq)
        assert torch.equal(s2[e, 0], fs2)


def test_filter_flushes_multiple_layers_in_order():
    stream = _expert_stream(
        num_experts=2, n=16, k=32, layers=("model.layers.0", "model.layers.1")
    )
    names = [n for n, _ in M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"])]
    assert names.index("model.layers.0.mlp.experts.w13_weight") < names.index(
        "model.layers.1.mlp.experts.w13_weight"
    )
    assert len(names) == 12


def test_filter_respects_complete_expert_layer_ignore():
    stream = _expert_stream(
        num_experts=2,
        n=16,
        k=32,
        layers=("model.layers.0", "model.layers.1"),
    )
    out = dict(
        M.iter_nvfp4_pertoken_weights(
            iter(stream),
            quant_patterns=["*.experts.*"],
            ignore_patterns=["*.layers.0.mlp.experts*"],
        )
    )
    assert out["model.layers.0.mlp.experts.gate_up_proj"].shape == (2, 32, 32)
    assert out["model.layers.0.mlp.experts.down_proj"].shape == (2, 32, 32)
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in out
    assert "model.layers.0.mlp.experts.w13_weight" not in out
    assert "model.layers.1.mlp.experts.w13_weight" in out


def test_filter_rejects_incomplete_expert_group():
    stream = _expert_stream(num_experts=2, n=16, k=32)
    stream.pop()
    with pytest.raises(RuntimeError, match="non-contiguous expert ids"):
        list(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))


def test_filter_raises_when_nothing_quantized():
    stream = [("m.self_attn.q_proj.weight", torch.randn(8, 16))]
    with pytest.raises(RuntimeError, match="quantized 0 params"):
        list(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))


def test_expand_fused_roundtrips_to_per_expert_checkpoint_names():
    """Fused transport tensors must expand back to exactly the per-expert
    ModelOpt names that RoutedExperts' expert mapping matches. Raw w13_/w2_
    names are silently dropped by vLLM's loader."""
    stream = _expert_stream(num_experts=2, n=16, k=32)
    stream.insert(0, ("m.self_attn.q_proj.weight", torch.randn(8, 8)))
    fused = list(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))
    expanded = dict(M.expand_fused_expert_weights(iter(fused)))
    fused = dict(fused)

    # Passthrough tensor is untouched.
    assert torch.equal(expanded["m.self_attn.q_proj.weight"], stream[0][1])
    # No fused transport names survive expansion.
    assert not any(".experts.w13_" in n or ".experts.w2_" in n for n in expanded)

    p = "model.layers.0.mlp.experts"
    # 1 passthrough + 2 experts x 3 projections x (3 tensors + input_scale)
    assert len(expanded) == 1 + 2 * 3 * 4
    # Neutral input scales complete each RoutedExperts layer during reload
    # (otherwise vLLM buffers the whole model and defers to finalize).
    for e in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            s = expanded[f"{p}.{e}.{proj}.input_scale"]
            assert s.dim() == 0 and s.item() == 1.0
    for e in range(2):
        assert torch.equal(
            expanded[f"{p}.{e}.gate_proj.weight"], fused[f"{p}.w13_weight"][e, :16]
        )
        assert torch.equal(
            expanded[f"{p}.{e}.up_proj.weight"], fused[f"{p}.w13_weight"][e, 16:]
        )
        assert torch.equal(
            expanded[f"{p}.{e}.down_proj.weight"], fused[f"{p}.w2_weight"][e]
        )
        assert torch.equal(
            expanded[f"{p}.{e}.gate_proj.weight_scale"].contiguous().view(torch.uint8),
            fused[f"{p}.w13_weight_scale"][e, :16].contiguous().view(torch.uint8),
        )
        assert torch.equal(
            expanded[f"{p}.{e}.down_proj.weight_scale_2"],
            fused[f"{p}.w2_weight_scale_2"][e],
        )
        # Shared gate/up global scale lands on both per-expert names as scalars.
        assert expanded[f"{p}.{e}.gate_proj.weight_scale_2"].dim() == 0
        assert torch.equal(
            expanded[f"{p}.{e}.gate_proj.weight_scale_2"],
            expanded[f"{p}.{e}.up_proj.weight_scale_2"],
        )


def test_rollout_config_defaults():
    cfg = M.NvFp4PerTokenRolloutConfig()
    assert cfg.enabled is False
    assert cfg.quant_patterns == ["*.experts.*"]
    assert cfg.resolved_ignore() == M.DEFAULT_NVFP4_IGNORE

    layer_ignore = "*.layers.0.mlp.experts*"
    cfg2 = M.NvFp4PerTokenRolloutConfig.model_validate(
        {"enabled": True, "additional_ignore": [layer_ignore]}
    )
    assert cfg2.resolved_ignore() == [*M.DEFAULT_NVFP4_IGNORE, layer_ignore]

    with pytest.raises(ValidationError, match="unknown_key"):
        M.NvFp4PerTokenRolloutConfig.model_validate({"enabled": True, "unknown_key": 1})
    with pytest.raises(ValidationError, match="ignore"):
        M.NvFp4PerTokenRolloutConfig.model_validate({"enabled": True, "ignore": []})


@pytest.mark.parametrize(
    "pattern",
    [
        "*.layers.0.mlp.experts.1*",
        "*.layers.0.mlp.experts.1.gate_proj*",
        "*.layers.0.mlp.experts.*.gate_proj*",
        "*self_attn*",
    ],
)
def test_rollout_config_rejects_partial_expert_ignore(pattern):
    with pytest.raises(ValidationError, match="complete expert layers"):
        M.NvFp4PerTokenRolloutConfig.model_validate(
            {"enabled": True, "additional_ignore": [pattern]}
        )


def test_prequantized_extension_uses_real_layerwise_reload_lifecycle(monkeypatch):
    """The retained path must restore stable kernel storage after each refit."""
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.reload import (
        record_metadata_for_reloading,
    )

    from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_vllm import (
        NvFp4PerTokenWorkerExtension,
    )

    model = torch.nn.Linear(4, 4, bias=False)
    record_metadata_for_reloading(model)
    original_data_ptr = model.weight.data_ptr()
    synchronized = []
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: synchronized.append(True)
    )

    extension = NvFp4PerTokenWorkerExtension.__new__(NvFp4PerTokenWorkerExtension)
    extension.device = torch.device("cpu")
    extension.model_runner = types.SimpleNamespace(model=model)
    extension.model_config = types.SimpleNamespace(dtype=torch.float32)

    with extension._weight_update_lifecycle("ipc") as finalize:
        assert model.weight.device.type == "meta"
        finalize()

    assert model.weight.device.type == "cpu"
    assert model.weight.data_ptr() == original_data_ptr
    assert synchronized == [True]
    assert extension._weight_update_errors_are_fatal()
