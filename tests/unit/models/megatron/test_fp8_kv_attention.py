# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in scale placement, gradients, packed sequences, and CP collectives."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo_rl.models.megatron.fp8_kv_attention import (
    FP8KVScaledAttention,
    cp_token_positions,
    scaled_fp8_kv_attention,
)
from nemo_rl.models.megatron.fp8_kv_quantization import (
    FP8_KV_IN_ATTENTION_ENV_VAR,
    fp8_kv_attention_env_vars,
    fp8_kv_in_attention_enabled,
    fp8_kv_quantization_enabled,
    install_fp8_kv_quantization,
)


def _config(cp=1, **overrides):
    return SimpleNamespace(
        **{
            "attention_dropout": 0.0,
            "window_size": None,
            "softmax_type": "vanilla",
            "attn_logit_softcapping": None,
            "qk_clip": False,
            "log_max_attention_logit": False,
            "context_parallel_size": cp,
            "cuda_graph_impl": "none",
            "enable_cuda_graph": False,
            "apply_query_key_layer_scaling": False,
            **overrides,
        }
    )


def _dense_reference(q, k, v):
    """Independent dense graph with STE through E4M3 and operand casts."""

    def unscaled(x):
        s = (x.detach().float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-6)
        z = x.float() / s
        encoded = z.detach().clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return (z + (encoded - z).detach()).to(x.dtype), s

    kr, ks = unscaled(k)
    vr, vs = unscaled(v)
    groups = q.shape[1] // k.shape[1]
    kr, ks, vr, vs = (t.repeat_interleave(groups, 1) for t in (kr, ks, vr, vs))
    scores = torch.einsum("qhd,khd->hqk", q.float(), kr.float()) * (q.shape[-1] ** -0.5)
    scores = scores * ks[:, :, 0].T[:, None, :]
    pos = torch.arange(len(q), device=q.device) + len(k) - len(q)
    mask = torch.arange(len(k), device=q.device)[None, :] > pos[:, None]
    p = scores.masked_fill(mask[None], -torch.inf).softmax(-1)
    weights = (p * vs[:, :, 0].T[:, None, :]).to(q.dtype)
    return torch.einsum("hqk,khd->qhd", weights.float(), vr.float()).to(q.dtype)


def test_environment_is_disabled_by_default_and_can_enable_quantization(monkeypatch):
    monkeypatch.delenv(FP8_KV_IN_ATTENTION_ENV_VAR, raising=False)
    assert not fp8_kv_in_attention_enabled()
    assert not fp8_kv_quantization_enabled({"fp8_kv_per_token": False})
    assert fp8_kv_quantization_enabled({"fp8_kv_per_token": True})
    monkeypatch.setenv(FP8_KV_IN_ATTENTION_ENV_VAR, "1")
    assert fp8_kv_in_attention_enabled()
    assert fp8_kv_quantization_enabled({"fp8_kv_per_token": False})
    assert fp8_kv_quantization_enabled({})
    assert fp8_kv_attention_env_vars({"OTHER": "keep"}) == {
        "OTHER": "keep",
        FP8_KV_IN_ATTENTION_ENV_VAR: "1",
    }
    assert fp8_kv_attention_env_vars({FP8_KV_IN_ATTENTION_ENV_VAR: "0"}) == {
        FP8_KV_IN_ATTENTION_ENV_VAR: "0"
    }
    monkeypatch.setenv(FP8_KV_IN_ATTENTION_ENV_VAR, "0")
    assert not fp8_kv_in_attention_enabled()


@pytest.mark.parametrize("setting", ["true", "false", "2", ""])
def test_invalid_environment_is_rejected(monkeypatch, setting):
    monkeypatch.setenv(FP8_KV_IN_ATTENTION_ENV_VAR, setting)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        fp8_kv_quantization_enabled({"fp8_kv_per_token": True})
    with pytest.raises(ValueError, match="must be 0 or 1"):
        fp8_kv_attention_env_vars(None)


@pytest.mark.parametrize("q_length", [1, 7, 69])
def test_scaled_attention_forward_and_ste_gradients(q_length):
    torch.manual_seed(15)
    q = torch.randn(q_length, 4, 16, requires_grad=True)
    k = torch.randn(73, 2, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    copies = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    actual = scaled_fp8_kv_attention(q, k, v)
    expected = _dense_reference(*copies)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for a, b in zip((q, k, v), copies):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-5)
    with torch.no_grad():
        torch.testing.assert_close(scaled_fp8_kv_attention(q, k, v), actual.detach())


def test_cp_mirrored_positions_cover_each_packed_sequence_once():
    positions = [cp_token_positions([16, 24], rank, 4, "cpu") for rank in range(4)]
    assert positions[0].tolist() == [0, 1, 14, 15, 16, 17, 18, 37, 38, 39]
    torch.testing.assert_close(torch.cat(positions).sort().values, torch.arange(40))
    with pytest.raises(ValueError, match="divisible"):
        cp_token_positions([17], 0, 4, "cpu")


def test_unpacked_batch_layout():
    torch.manual_seed(16)
    q = torch.randn(7, 2, 4, 16)
    k, v = [torch.randn(11, 2, 2, 16) for _ in range(2)]
    actual = FP8KVScaledAttention(_config())(q, k, v)
    expected = torch.stack(
        [_dense_reference(q[:, b], k[:, b], v[:, b]) for b in range(2)], dim=1
    )
    torch.testing.assert_close(actual, expected.flatten(-2), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"attention_dropout": 0.1},
        {"window_size": (3, 0)},
        {"softmax_type": "off-by-one"},
        {"attn_logit_softcapping": 30},
        {"qk_clip": True},
        {"cuda_graph_impl": "local"},
        {"apply_query_key_layer_scaling": True},
    ],
)
def test_unsupported_configurations_fail_at_install(overrides):
    with pytest.raises(ValueError):
        FP8KVScaledAttention(_config(**overrides))


@pytest.mark.mcore
def test_installer_paths_are_exclusive_and_state_keys_are_preserved(monkeypatch):
    from megatron.core.transformer.attention import SelfAttention

    def model():
        attention = SelfAttention.__new__(SelfAttention)
        torch.nn.Module.__init__(attention)
        attention.config = _config()
        attention.pg_collection = SimpleNamespace(cp=None)
        attention.core_attention = torch.nn.Identity()
        return attention

    legacy = model()
    monkeypatch.delenv(FP8_KV_IN_ATTENTION_ENV_VAR, raising=False)
    install_fp8_kv_quantization([legacy])
    assert len(legacy.core_attention._forward_pre_hooks) == 1
    monkeypatch.setenv(FP8_KV_IN_ATTENTION_ENV_VAR, "1")
    before = list(legacy.state_dict())
    install_fp8_kv_quantization([legacy])
    assert isinstance(legacy.core_attention, FP8KVScaledAttention)
    assert not legacy.core_attention._forward_pre_hooks
    assert list(legacy.state_dict()) == before
    # Even replacing a previously hooked core must never call BF16 dequantization.
    q = torch.randn(7, 1, 4, 16)
    k, v = [torch.randn(7, 1, 2, 16) for _ in range(2)]
    with patch(
        "nemo_rl.models.megatron.fp8_kv_quantization.fake_quantize_kv_per_token_head",
        side_effect=AssertionError("legacy dequantization called"),
    ):
        out = legacy._run_core_attention(q, k, v, None)
    torch.testing.assert_close(
        out,
        _dense_reference(q[:, 0], k[:, 0], v[:, 0]).flatten(-2)[:, None],
        rtol=2e-5,
        atol=2e-6,
    )
    install_fp8_kv_quantization([legacy])
    assert list(legacy.state_dict()) == before


@pytest.mark.mcore
def test_packed_padding_and_sequence_isolation():
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.manual_seed(17)
    q = torch.randn(16, 4, 16, requires_grad=True)
    k, v = [torch.randn(16, 2, 16, requires_grad=True) for _ in range(2)]
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 5, 11], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 5, 11], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 8, 16], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 8, 16], dtype=torch.int32),
    )
    actual = FP8KVScaledAttention(_config())(q, k, v, packed_seq_params=params).reshape(
        q.shape
    )
    expected = torch.zeros_like(q)
    expected[:5] = _dense_reference(q[:5], k[:5], v[:5])
    expected[8:14] = _dense_reference(q[8:14], k[8:14], v[8:14])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.sum().backward()
    for t in (q, k, v):
        assert torch.count_nonzero(t.grad[5:8]) == 0
        assert torch.count_nonzero(t.grad[14:]) == 0


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bf16_forward_and_backward():
    torch.manual_seed(18)
    q = torch.randn(69, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = [
        torch.randn(73, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    ]
    copies = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    actual = scaled_fp8_kv_attention(q, k, v)
    expected = _dense_reference(*copies)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.002
    for a, b in zip((q, k, v), copies):
        assert torch.isfinite(a.grad).all()
        assert (a.grad.float() - b.grad.float()).norm() / b.grad.float().norm() < 0.02


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_self_attention_with_recomputation(monkeypatch, tmp_path):
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    monkeypatch.setenv(FP8_KV_IN_ATTENTION_ENV_VAR, "1")
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{tmp_path}/init", rank=0, world_size=1
    )
    try:
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(20)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            num_query_groups=1,
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
        spec = get_gpt_layer_with_transformer_engine_spec().submodules.self_attention
        attention = build_module(spec, config=config, layer_number=1).cuda().train()
        state_keys = list(attention.state_dict())
        saved_state = attention.state_dict()
        install_fp8_kv_quantization([attention])
        assert isinstance(attention.core_attention, FP8KVScaledAttention)
        assert list(attention.state_dict()) == state_keys
        attention.load_state_dict(saved_state, strict=True)
        assert attention.core_attention.sharded_state_dict() == {}
        hidden = torch.randn(
            69, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with patch(
            "nemo_rl.models.megatron.fp8_kv_quantization.fake_quantize_kv_per_token_head",
            side_effect=AssertionError("legacy dequantization called"),
        ):
            output, bias = attention(hidden, attention_mask=None)
            projected = output if bias is None else output + bias
            projected.float().square().mean().backward()
        assert torch.isfinite(output).all()
        assert torch.isfinite(hidden.grad).all() and hidden.grad.norm() > 0
        for parameter in attention.parameters():
            if parameter.requires_grad:
                assert (
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                )
        with torch.no_grad():
            evaluated, _ = attention(hidden, attention_mask=None)
        torch.testing.assert_close(evaluated, output.detach(), rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _cp_worker():
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    for valid_lengths in ((16, 24), (1, 9)):
        # The padded case gives rank 3 no valid queries at all. It must still
        # participate in backward K/V reductions rather than deadlocking.
        torch.manual_seed(19)
        full = [
            torch.randn(40, h, 16).cuda().bfloat16().requires_grad_() for h in (4, 2, 2)
        ]
        positions = cp_token_positions([16, 24], rank, 4, "cuda")
        local = [t.detach().index_select(0, positions).requires_grad_() for t in full]
        cu = torch.tensor(
            [0, valid_lengths[0], sum(valid_lengths)], device="cuda", dtype=torch.int32
        )
        padded = torch.tensor([0, 16, 40], device="cuda", dtype=torch.int32)
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
            cp_group=torch.distributed.group.WORLD,
            local_cp_size=4,
        )
        actual = FP8KVScaledAttention(
            _config(cp=4), cp_group=torch.distributed.group.WORLD
        )(*local, packed_seq_params=params).reshape(10, 4, 16)
        expected = torch.zeros_like(full[0])
        for a, count in ((0, valid_lengths[0]), (16, valid_lengths[1])):
            expected[a : a + count] = scaled_fp8_kv_attention(
                *(t[a : a + count] for t in full)
            )
        torch.testing.assert_close(
            actual, expected.index_select(0, positions), rtol=0, atol=0
        )
        actual.float().square().sum().backward()
        expected.float().square().sum().backward()
        for a, b in zip(local, full):
            target = b.grad.index_select(0, positions).float()
            assert torch.isfinite(a.grad).all()
            assert (a.grad.float() - target).norm() / target.norm().clamp_min(
                1e-20
            ) < 0.02
    print(f"CP4_FORWARD_BACKWARD_PASSED rank={rank}", flush=True)
    torch.distributed.destroy_process_group()


@pytest.mark.mcore
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_context_parallel_four_ranks():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(Path(__file__).resolve()),
            "--cp-worker",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("CP4_FORWARD_BACKWARD_PASSED") == 4


if __name__ == "__main__":
    if sys.argv[1:] != ["--cp-worker"]:
        raise SystemExit("use pytest or --cp-worker")
    _cp_worker()
