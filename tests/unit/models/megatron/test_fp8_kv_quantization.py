# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Per-token-head K/V rounding, attention hooks, and straight-through gradients."""

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo_rl.models.megatron.fp8_kv_quantization import (
    fake_quantize_kv_per_token_head,
    fp8_kv_attention_pre_hook,
    fp8_kv_quantization_enabled,
    install_fp8_kv_quantization,
    quantize_kv_per_token_head,
)


@pytest.mark.parametrize("shape", [(7, 3, 8), (7, 2, 3, 8)])
def test_scales_are_independent_of_tokens_heads_and_batch(shape):
    values = torch.zeros(shape)
    maxima = torch.arange(1, values.numel() // shape[-1] + 1).reshape(*shape[:-1], 1)
    values[..., :1] = maxima
    values[..., 1:2] = -maxima
    quantized, scales = quantize_kv_per_token_head(values)
    torch.testing.assert_close(scales, maxima.float() / 448, rtol=0, atol=0)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert torch.all(quantized[..., 0].float() == 448)
    assert torch.all(quantized[..., 1].float() == -448)


def test_known_e4m3_rounding():
    values = torch.tensor([[[1.0, 0.1, -0.3, 0.0]]])
    quantized, scales = quantize_kv_per_token_head(values)
    torch.testing.assert_close(
        quantized.float(), torch.tensor([[[448.0, 44.0, -128.0, 0.0]]])
    )
    torch.testing.assert_close(scales, torch.tensor([[[1.0 / 448]]]))
    result = fake_quantize_kv_per_token_head(values)
    assert not torch.equal(result, values)
    torch.testing.assert_close(
        result, torch.tensor([[[1.0, 44.0 / 448, -128.0 / 448, 0.0]]])
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_zero_and_tiny_values_are_finite(dtype):
    values = torch.zeros(3, 2, 8, dtype=dtype)
    values[1] = 1e-8
    quantized, scales = quantize_kv_per_token_head(values)
    assert torch.all(scales == 1e-6)
    assert torch.isfinite(quantized.float()).all()
    result = fake_quantize_kv_per_token_head(values)
    assert result.dtype == dtype
    assert torch.equal(result[0], values[0])
    assert torch.isfinite(result).all()


def test_noncontiguous_and_packed_layouts_have_identical_rounding():
    values = torch.randn(5, 2, 3, 8).transpose(0, 1)
    assert not values.is_contiguous()
    result = fake_quantize_kv_per_token_head(values)
    packed = fake_quantize_kv_per_token_head(values.reshape(-1, 3, 8))
    torch.testing.assert_close(result.reshape(-1, 3, 8), packed, rtol=0, atol=0)


def test_backward_is_identity_without_scale_gradients():
    values = torch.randn(4, 2, 8, requires_grad=True)
    upstream = torch.randn_like(values)
    result = fake_quantize_kv_per_token_head(values)
    result.backward(upstream)
    torch.testing.assert_close(values.grad, upstream, rtol=0, atol=0)


class RecordingAttention(nn.Module):
    def forward(self, query, key, value, attention_mask=None, **kwargs):
        self.inputs = (query, key, value)
        scores = (
            torch.einsum("thd,shd->hts", query.float(), key.float())
            / query.shape[-1] ** 0.5
        )
        probabilities = scores.softmax(dim=-1)
        return torch.einsum("hts,shd->thd", probabilities, value.float()).to(
            query.dtype
        )


@pytest.mark.parametrize("keyword_args", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_attention_quantizes_only_kv_with_gradients(keyword_args, recompute):
    core = RecordingAttention()
    core.register_forward_pre_hook(fp8_kv_attention_pre_hook, with_kwargs=True)
    query, key, value = [torch.randn(5, 2, 8, requires_grad=True) for _ in range(3)]

    def forward(q, k, v):
        if keyword_args:
            return core(query=q, key=k, value=v)
        return core(q, k, v)

    result = (
        checkpoint(forward, query, key, value, use_reentrant=True)
        if recompute
        else forward(query, key, value)
    )
    result.square().sum().backward()
    # Reentrant checkpointing detaches the input objects before recomputation.
    assert core.inputs[0].data_ptr() == query.data_ptr()
    torch.testing.assert_close(core.inputs[0], query, rtol=0, atol=0)
    for original, observed in zip((key, value), core.inputs[1:]):
        torch.testing.assert_close(
            observed, fake_quantize_kv_per_token_head(original), rtol=0, atol=0
        )
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_hook_also_applies_to_no_grad_logprob_passes():
    core = RecordingAttention().eval()
    core.register_forward_pre_hook(fp8_kv_attention_pre_hook, with_kwargs=True)
    q, k, v = [torch.randn(3, 2, 8) for _ in range(3)]
    with torch.no_grad():
        core(q, k, v)
    assert core.inputs[0] is q
    assert not torch.equal(core.inputs[1], k)
    assert not torch.equal(core.inputs[2], v)


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_switch_rejects_non_boolean_values(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        fp8_kv_quantization_enabled({"fp8_kv_per_token": value})


def test_switch_defaults_to_disabled():
    assert fp8_kv_quantization_enabled({}) is False
    assert fp8_kv_quantization_enabled({"fp8_kv_per_token": False}) is False
    assert fp8_kv_quantization_enabled({"fp8_kv_per_token": True}) is True


@pytest.mark.mcore
def test_installer_reaches_real_mcore_core_attention_without_changing_state():
    from megatron.core.transformer.attention import SelfAttention

    attention = SelfAttention.__new__(SelfAttention)
    nn.Module.__init__(attention)
    attention.core_attention = RecordingAttention()
    model = nn.Module()
    model.add_module("self_attention", attention)
    state_keys = list(model.state_dict())
    models = [model]
    assert install_fp8_kv_quantization(models) is models
    assert list(model.state_dict()) == state_keys
    q, k, v = [torch.randn(3, 2, 8) for _ in range(3)]
    attention._run_core_attention(q, k, v, None)
    assert attention.core_attention.inputs[0] is q
    torch.testing.assert_close(
        attention.core_attention.inputs[1], fake_quantize_kv_per_token_head(k)
    )


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_te_self_attention_forward_and_backward(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.distributed.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path}/distributed",
        rank=0,
        world_size=1,
    )
    try:
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            num_query_groups=1,
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        spec = get_gpt_layer_with_transformer_engine_spec().submodules.self_attention
        attention = build_module(spec, config=config, layer_number=1).cuda()
        hidden = torch.randn(
            16, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        baseline, _ = attention(hidden, attention_mask=None)
        original_kv = []
        observed_kv = []

        def capture_original(module, args):
            original_kv.append((args[1].detach().clone(), args[2].detach().clone()))

        def capture_quantized(module, args):
            observed_kv.append((args[1].detach().clone(), args[2].detach().clone()))

        attention.core_attention.register_forward_pre_hook(capture_original)
        install_fp8_kv_quantization([attention])
        attention.core_attention.register_forward_pre_hook(capture_quantized)
        output, _ = attention(hidden, attention_mask=None)
        output.float().square().sum().backward()
        assert torch.isfinite(output).all()
        assert not torch.equal(output, baseline)
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert hidden.grad.abs().sum() > 0
        for original, observed in zip(original_kv[0], observed_kv[0]):
            torch.testing.assert_close(
                observed, fake_quantize_kv_per_token_head(original), rtol=0, atol=0
            )
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.vllm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quantization_matches_actual_vllm_cache_write_kernel():
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode

    torch.manual_seed(42)
    tokens, heads, dim = 11, 3, 64
    key = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key) * 0.01
    key[0].zero_()
    key[1].mul_(1e-8)
    key_cache = torch.empty(
        1, tokens, heads, dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    value_cache = torch.empty_like(key_cache)
    k_scales = torch.empty(1, tokens, heads, device="cuda")
    v_scales = torch.empty_like(k_scales)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scales,
        v_scales,
        slots,
        kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    for tensor, cache, scales in (
        (key, key_cache, k_scales),
        (value, value_cache, v_scales),
    ):
        expected_q, expected_s = quantize_kv_per_token_head(tensor)
        torch.testing.assert_close(cache[0].float(), expected_q.float(), rtol=0, atol=0)
        torch.testing.assert_close(scales[0], expected_s.squeeze(-1), rtol=1e-7, atol=0)
