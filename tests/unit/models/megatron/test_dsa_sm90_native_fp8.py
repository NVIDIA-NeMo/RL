# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron.patches import dsa_sm90_native_fp8 as native_fp8

pytestmark = pytest.mark.mcore


def test_fp8_ue8m0_quantize_handles_zero_small_and_large_vectors():
    values = torch.zeros(3, 128, dtype=torch.float32)
    values[1, 0] = 1.0e-10
    values[2, 0] = 448.0

    codes, scale = native_fp8._fp8_ue8m0_quantize(values)
    restored = codes.float() * scale

    assert torch.isfinite(restored).all()
    assert torch.count_nonzero(restored[0]) == 0
    torch.testing.assert_close(scale[0], torch.tensor([2.0**-22]), rtol=0, atol=0)
    torch.testing.assert_close(scale[2], torch.ones(1), rtol=0, atol=0)


class _CapturingCudnnDsa:
    def __init__(self):
        self.calls = []

    def indexer_forward_wrapper(
        self,
        index_q,
        index_k,
        weights,
        ratio,
        sm_scale,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        precision=None,
        q_scale=None,
        k_scale=None,
        q_causal_offsets=None,
    ):
        call = {
            "index_q": index_q,
            "index_k": index_k,
            "weights": weights,
            "ratio": ratio,
            "sm_scale": sm_scale,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "precision": precision,
            "q_scale": q_scale,
            "k_scale": k_scale,
            "q_causal_offsets": q_causal_offsets,
        }
        self.calls.append(call)
        if index_q.ndim == 4:
            shape = (index_q.size(0), index_q.size(1), index_k.size(1))
        else:
            shape = (index_q.size(0), max_seqlen_k)
        return {"scores": torch.zeros(shape, dtype=torch.float32)}


@pytest.fixture
def allow_cpu_native_operands(monkeypatch):
    def validate_without_device(q, k, weights, ratio):
        assert ratio == 1
        assert q.shape[:-1] == weights.shape
        assert k.size(-2) == 1
        assert q.size(-2) in (32, 64)
        assert q.size(-1) == k.size(-1) == 128

    monkeypatch.setattr(
        native_fp8, "_validate_native_operand_layout", validate_without_device
    )


def test_cudnn_proxy_passes_native_fp8_operands_and_folded_fp32_weights(
    allow_cpu_native_operands,
):
    torch.manual_seed(9)
    q = torch.randn(1, 4, 32, 128)
    k = torch.randn(1, 6, 1, 128)
    raw_weights = torch.randn(1, 4, 32)
    delegate = _CapturingCudnnDsa()

    result = native_fp8._NativeFp8CudnnDsaProxy(delegate).indexer_forward_wrapper(
        q, k, raw_weights, ratio=1, sm_scale=1.0
    )

    call = delegate.calls[-1]
    q_fp8, q_descale = native_fp8._fp8_ue8m0_quantize(q)
    k_fp8, k_descale = native_fp8._fp8_ue8m0_quantize(k)
    expected_weights = raw_weights * q_descale.squeeze(-1)
    expected_weights = expected_weights * (128**-0.5)
    expected_weights = (expected_weights * (32**-0.5)).contiguous()

    assert result["scores"].shape == (1, 4, 6)
    assert call["index_q"].dtype == torch.float8_e4m3fn
    assert call["index_k"].dtype == torch.float8_e4m3fn
    assert call["weights"].dtype == torch.float32
    assert call["q_scale"].dtype == torch.float32
    assert call["k_scale"].dtype == torch.float32
    assert call["precision"] == "fp8"
    assert call["sm_scale"] == 1.0
    torch.testing.assert_close(call["index_q"], q_fp8, rtol=0, atol=0)
    torch.testing.assert_close(call["index_k"], k_fp8, rtol=0, atol=0)
    torch.testing.assert_close(call["weights"], expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(call["q_scale"], torch.ones_like(call["q_scale"]))
    torch.testing.assert_close(call["k_scale"], k_descale.squeeze(-1), rtol=0, atol=0)
    torch.testing.assert_close(
        call["q_causal_offsets"], torch.tensor([2], dtype=torch.int32)
    )


def test_cudnn_proxy_derives_per_segment_thd_offsets(allow_cpu_native_operands):
    q = torch.randn(4, 32, 128)
    k = torch.randn(7, 1, 128)
    weights = torch.randn(4, 32)
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)
    delegate = _CapturingCudnnDsa()

    native_fp8._NativeFp8CudnnDsaProxy(delegate).indexer_forward_wrapper(
        q,
        k,
        weights,
        ratio=1,
        sm_scale=1.0,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=2,
        max_seqlen_k=4,
    )

    call = delegate.calls[-1]
    torch.testing.assert_close(
        call["q_causal_offsets"], torch.tensor([1, 2], dtype=torch.int32)
    )
    assert call["q_scale"].shape == (4, 32)
    assert call["k_scale"].shape == (7, 1)


def test_cudnn_proxy_delegates_bf16_without_native_arguments():
    q = torch.randn(1, 2, 32, 128).bfloat16()
    k = torch.randn(1, 2, 1, 128).bfloat16()
    weights = torch.randn(1, 2, 32).bfloat16()
    offsets = torch.tensor([3], dtype=torch.int32)
    delegate = _CapturingCudnnDsa()

    native_fp8._NativeFp8CudnnDsaProxy(delegate).indexer_forward_wrapper(
        q, k, weights, ratio=1, sm_scale=1.0, q_causal_offsets=offsets
    )

    call = delegate.calls[-1]
    assert call["index_q"] is q
    assert call["index_k"] is k
    assert call["weights"] is weights
    assert call["precision"] == "bf16"
    assert call["q_scale"] is None
    assert call["k_scale"] is None
    assert call["q_causal_offsets"] is offsets


def test_cudnn_proxy_delegates_existing_fp8_call_transparently():
    q = torch.randn(1, 2, 32, 128).to(torch.float8_e4m3fn)
    k = torch.randn(1, 2, 1, 128).to(torch.float8_e4m3fn)
    weights = torch.randn(1, 2, 32)
    q_scale = torch.ones(1, 2, 32)
    k_scale = torch.ones(1, 2, 1)
    offsets = torch.tensor([0], dtype=torch.int32)
    delegate = _CapturingCudnnDsa()

    native_fp8._NativeFp8CudnnDsaProxy(delegate).indexer_forward_wrapper(
        q,
        k,
        weights,
        ratio=1,
        sm_scale=1.0,
        precision="fp8",
        q_scale=q_scale,
        k_scale=k_scale,
        q_causal_offsets=offsets,
    )

    call = delegate.calls[-1]
    assert call["index_q"] is q
    assert call["index_k"] is k
    assert call["weights"] is weights
    assert call["precision"] == "fp8"
    assert call["q_scale"] is q_scale
    assert call["k_scale"] is k_scale
    assert call["q_causal_offsets"] is offsets


def test_cudnn_proxy_rejects_partially_converted_operands():
    q = torch.randn(1, 2, 32, 128)
    k = torch.randn(1, 2, 1, 128).bfloat16()
    weights = torch.randn(1, 2, 32)

    with pytest.raises(RuntimeError, match="all be FP32"):
        native_fp8._NativeFp8CudnnDsaProxy(
            _CapturingCudnnDsa()
        ).indexer_forward_wrapper(q, k, weights, ratio=1, sm_scale=1.0)


def test_global_row_helper_uses_explicit_chunk_offset(allow_cpu_native_operands):
    delegate = _CapturingCudnnDsa()
    module = SimpleNamespace(
        _cudnn_dsa=native_fp8._NativeFp8CudnnDsaProxy(delegate),
        _compute_indexer_scores_chunk_with_global_rows=lambda *_args, **_kwargs: None,
    )
    native_fp8._patch_global_row_score_helper(module)
    q = torch.randn(1, 2, 32, 128)
    k = torch.randn(1, 8, 1, 128)
    weights = torch.randn(1, 2, 32)

    scores = module._compute_indexer_scores_chunk_with_global_rows(
        q,
        k,
        weights,
        row_start=3,
        indexer_ratio=1,
        sm_scale=1.0,
    )

    torch.testing.assert_close(
        delegate.calls[-1]["q_causal_offsets"], torch.tensor([3], dtype=torch.int32)
    )
    assert torch.isfinite(scores[:, 0, :4]).all()
    assert torch.isneginf(scores[:, 0, 4:]).all()
    assert torch.isfinite(scores[:, 1, :5]).all()
    assert torch.isneginf(scores[:, 1, 5:]).all()

    module._compute_indexer_scores_chunk_with_global_rows(
        q,
        k,
        weights,
        row_start=0,
        indexer_ratio=1,
        sm_scale=1.0,
        seq_lens=torch.tensor([3, 4]),
    )
    torch.testing.assert_close(
        delegate.calls[-1]["q_causal_offsets"], torch.tensor([2], dtype=torch.int32)
    )


def test_global_row_helper_splits_zero_length_prefix(allow_cpu_native_operands):
    delegate = _CapturingCudnnDsa()
    module = SimpleNamespace(
        _cudnn_dsa=native_fp8._NativeFp8CudnnDsaProxy(delegate),
        _compute_indexer_scores_chunk_with_global_rows=lambda *_args, **_kwargs: None,
    )
    native_fp8._patch_global_row_score_helper(module)
    q = torch.randn(1, 3, 32, 128)
    k = torch.randn(1, 3, 1, 128)
    weights = torch.randn(1, 3, 32)

    scores = module._compute_indexer_scores_chunk_with_global_rows(
        q,
        k,
        weights,
        row_start=0,
        indexer_ratio=1,
        sm_scale=1.0,
        seq_lens=torch.tensor([0, 1, 2]),
    )

    assert len(delegate.calls) == 1
    torch.testing.assert_close(
        delegate.calls[-1]["q_causal_offsets"], torch.tensor([0], dtype=torch.int32)
    )
    assert torch.isneginf(scores[:, 0]).all()
    assert torch.isfinite(scores[:, 1, :1]).all()
    assert torch.isneginf(scores[:, 1, 1:]).all()
    assert torch.isfinite(scores[:, 2, :2]).all()
    assert torch.isneginf(scores[:, 2, 2:]).all()


def test_global_row_helper_handles_empty_query_and_rejects_ratio(
    allow_cpu_native_operands,
):
    delegate = _CapturingCudnnDsa()
    module = SimpleNamespace(
        _cudnn_dsa=native_fp8._NativeFp8CudnnDsaProxy(delegate),
        _compute_indexer_scores_chunk_with_global_rows=lambda *_args, **_kwargs: None,
    )
    native_fp8._patch_global_row_score_helper(module)
    q = torch.empty(1, 0, 32, 128)
    k = torch.randn(1, 3, 1, 128)
    weights = torch.empty(1, 0, 32)

    scores = module._compute_indexer_scores_chunk_with_global_rows(
        q, k, weights, row_start=0, indexer_ratio=1, sm_scale=1.0
    )
    assert scores.shape == (1, 0, 3)
    assert scores.dtype == torch.float32
    assert not delegate.calls

    with pytest.raises(RuntimeError, match="requires ratio=1"):
        module._compute_indexer_scores_chunk_with_global_rows(
            q, k, weights, row_start=0, indexer_ratio=2, sm_scale=1.0
        )


def test_backend_decline_cannot_fall_back_from_native_fp8():
    module = SimpleNamespace(
        run_fused_qk_topk=lambda *args, **kwargs: None,
        run_fused_dsa_attention=lambda *args, **kwargs: "full",
        run_fused_qk_topk_with_loss=lambda *args, **kwargs: "loss",
    )
    native_fp8._patch_backend_entrypoints(module)
    q = torch.randn(1, 1, 32, 128)
    k = torch.randn(1, 1, 128)
    weights = torch.randn(1, 1, 32)

    with pytest.raises(RuntimeError, match="refusing to fall back"):
        module.run_fused_qk_topk(q=q, k=k, weights=weights)
    assert (
        module.run_fused_dsa_attention(
            q_indexer=q, k_indexer=k, indexer_weights=weights
        )
        is None
    )
    with pytest.raises(RuntimeError, match="does not support Indexer auxiliary loss"):
        module.run_fused_qk_topk_with_loss(q=q, k=k, weights=weights)


def test_naive_score_fallback_rejects_native_fp8_operands():
    module = SimpleNamespace(fused_qk_topk_naive=lambda *args, **kwargs: "naive")
    native_fp8._patch_naive_score_fallback(module)
    q = torch.randn(1, 1, 32, 128)
    k = torch.randn(1, 1, 128)
    weights = torch.randn(1, 1, 32)

    with pytest.raises(RuntimeError, match="bypassed the cuDNN scorer"):
        module.fused_qk_topk_naive(q, k, weights)

    assert (
        module.fused_qk_topk_naive(q.bfloat16(), k.bfloat16(), weights.bfloat16())
        == "naive"
    )


def test_install_scorer_is_idempotent(monkeypatch):
    delegate = _CapturingCudnnDsa()
    cudnn_kernels = SimpleNamespace(
        _cudnn_dsa=delegate,
        _ensure_dsa_namespace=lambda: None,
        _compute_indexer_scores_chunk_with_global_rows=lambda *_args, **_kwargs: None,
        run_fused_qk_topk=lambda *_args, **_kwargs: None,
        run_fused_qk_topk_with_loss=lambda *_args, **_kwargs: None,
        run_fused_dsa_attention=lambda *_args, **_kwargs: None,
    )
    dsa_module = SimpleNamespace(
        fused_qk_topk_naive=lambda *_args, **_kwargs: None,
    )
    modules = {
        native_fp8._CUDNN_KERNEL_MODULE: cudnn_kernels,
        native_fp8._DSA_MODULE: dsa_module,
    }
    monkeypatch.setattr(native_fp8, "_validate_sm90_runtime", lambda: None)
    monkeypatch.setattr(native_fp8, "import_module", modules.__getitem__)

    native_fp8.install_dsa_sm90_fp8_scorer()
    first_install = (
        cudnn_kernels._cudnn_dsa,
        cudnn_kernels._compute_indexer_scores_chunk_with_global_rows,
        cudnn_kernels.run_fused_qk_topk,
        cudnn_kernels.run_fused_qk_topk_with_loss,
        cudnn_kernels.run_fused_dsa_attention,
        dsa_module.fused_qk_topk_naive,
    )
    native_fp8.install_dsa_sm90_fp8_scorer()

    assert isinstance(cudnn_kernels._cudnn_dsa, native_fp8._NativeFp8CudnnDsaProxy)
    assert cudnn_kernels._cudnn_dsa._delegate is delegate
    assert first_install == (
        cudnn_kernels._cudnn_dsa,
        cudnn_kernels._compute_indexer_scores_chunk_with_global_rows,
        cudnn_kernels.run_fused_qk_topk,
        cudnn_kernels.run_fused_qk_topk_with_loss,
        cudnn_kernels.run_fused_dsa_attention,
        dsa_module.fused_qk_topk_naive,
    )


@pytest.mark.parametrize(
    ("cuda_available", "capability", "error"),
    [
        (False, (9, 0), "CUDA SM90 runtime"),
        (True, (8, 0), "supports SM90 only"),
        (True, (9, 0), None),
        (True, (10, 0), "supports SM90 only"),
    ],
)
def test_scorer_runtime_validation_requires_sm90(
    monkeypatch, cuda_available, capability, error
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: capability if device == 7 else (-1, -1),
    )

    if error is None:
        native_fp8._validate_sm90_runtime()
    else:
        with pytest.raises(RuntimeError, match=error):
            native_fp8._validate_sm90_runtime()


def _valid_indexer_config():
    return SimpleNamespace(
        config=SimpleNamespace(
            experimental_attention_variant="dsa",
            dsa_kernel_backend="cudnn",
            attention_backend="auto",
            dsa_indexer_head_dim=128,
            dsa_indexer_scoring_relu=True,
            dsa_indexer_loss_coeff=0.0,
        ),
        index_n_heads=32,
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("dsa_kernel_backend", "tilelang"),
        ("attention_backend", "unfused"),
        ("dsa_indexer_scoring_relu", False),
    ],
)
def test_scorer_validator_rejects_unsupported_backend_config(attribute, value):
    indexer = _valid_indexer_config()
    setattr(indexer.config, attribute, value)

    with pytest.raises(ValueError, match=attribute):
        native_fp8.validate_dsa_sm90_fp8_indexer(indexer)


def test_scorer_validator_rejects_indexer_loss():
    indexer = _valid_indexer_config()
    indexer.config.dsa_indexer_loss_coeff = 0.001

    with pytest.raises(ValueError, match="dsa_indexer_loss_coeff=0"):
        native_fp8.validate_dsa_sm90_fp8_indexer(indexer)
