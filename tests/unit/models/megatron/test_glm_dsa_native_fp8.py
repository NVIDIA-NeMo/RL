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

from nemo_rl.models.megatron.patches import glm_dsa_native_fp8 as glm_adapter

pytestmark = pytest.mark.mcore


class _PairLinear(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, _inputs):
        return self.output.clone(), None


class _RotaryEmbedding:
    def get_rotary_seq_len(self, *_args):
        return 2

    def __call__(self, rotary_seq_len, *, packed_seq):
        del packed_seq
        return torch.zeros(rotary_seq_len, 1, 1, 64)


class _TensorParallelGroup:
    def size(self):
        return 1


class _FakeIndexer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            experimental_attention_variant="dsa",
            dsa_kernel_backend="cudnn",
            attention_backend="auto",
            dsa_indexer_head_dim=128,
            qk_pos_emb_head_dim=64,
            dsa_indexer_rope_interleaved=True,
            dsa_indexer_rotate_activation=False,
            dsa_indexer_scoring_relu=True,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_k_norm_epsilon=1.0e-6,
            rope_type="rope",
            rotary_scaling_factor=1.0,
            mscale=1.0,
            mscale_all_dim=1.0,
            rotary_interleaved=False,
            apply_rope_fusion=False,
            bf16=True,
            fp16=False,
            sequence_parallel=False,
            layernorm_epsilon=1.0e-5,
            layernorm_zero_centered_gamma=False,
        )
        self.qk_pos_emb_head_dim = 64
        self.index_head_dim = 128
        self.index_n_heads = 32
        self.softmax_scale = 128**-0.5
        self.pg_collection = SimpleNamespace(tp=_TensorParallelGroup(), cp=object())
        self.rotary_pos_emb = _RotaryEmbedding()

        torch.manual_seed(13)
        self.q_projection = torch.randn(2, 1, 32 * 128).bfloat16()
        self.k_projection = torch.randn(2, 1, 128).bfloat16()
        self.raw_weights = torch.randn(2, 1, 32).bfloat16()
        self.linear_wq_b = _PairLinear(self.q_projection)
        self.linear_wk = _PairLinear(self.k_projection)
        self.linear_weights_proj = _PairLinear(self.raw_weights)
        self.k_norm = torch.nn.LayerNorm(128, eps=1.0e-6, dtype=torch.bfloat16)
        with torch.no_grad():
            self.k_norm.weight.copy_(torch.randn(128).bfloat16())
            self.k_norm.bias.copy_(torch.randn(128).bfloat16())

    def forward_before_topk(self, x, qr, packed_seq_params=None):
        del x, qr, packed_seq_params
        raise AssertionError("unpatched forward_before_topk was called")

    def forward_with_scores(self, x, qr, mask=None, packed_seq_params=None):
        del x, qr, mask, packed_seq_params
        raise AssertionError("unpatched forward_with_scores was called")


def test_layer_norm_fp32_uses_configured_indexer_epsilon():
    indexer = _FakeIndexer()
    indexer.config.dsa_indexer_k_norm_epsilon = 0.25
    inputs = indexer.k_projection.float()

    actual = glm_adapter._layer_norm_fp32(indexer, inputs)

    centered = inputs - inputs.mean(dim=-1, keepdim=True)
    expected = centered * torch.rsqrt(
        (centered * centered).mean(dim=-1, keepdim=True) + 0.25
    )
    expected = expected * indexer.k_norm.weight.float() + indexer.k_norm.bias.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("rope_dim", "rope_interleaved"), [(32, False), (96, True)])
def test_rope_fp32_uses_configured_width_and_interleaving(
    monkeypatch, rope_dim, rope_interleaved
):
    indexer = _FakeIndexer()
    indexer.config.qk_pos_emb_head_dim = rope_dim
    indexer.config.dsa_indexer_rope_interleaved = rope_interleaved
    inputs = torch.randn(2, 1, 3, 128)
    call = {}

    def fake_apply_rope(tensor, _freqs, **kwargs):
        call["shape"] = tensor.shape
        call.update(kwargs)
        return tensor

    monkeypatch.setattr(glm_adapter, "apply_rotary_pos_emb", fake_apply_rope)
    actual = glm_adapter._apply_glm52_rope_fp32(indexer, inputs, torch.empty(0), 1.0)

    torch.testing.assert_close(actual, inputs, rtol=0, atol=0)
    assert call["shape"][-1] == rope_dim
    assert call["mla_rotary_interleaved"] is rope_interleaved
    assert call["mla_output_remove_interleaving"] is rope_interleaved


def test_enable_native_fp8_installs_generic_scorer_and_keeps_fp32_preprocessing(
    monkeypatch,
):
    monkeypatch.setattr(glm_adapter, "DSAIndexer", _FakeIndexer)
    installer_calls = []
    monkeypatch.setattr(
        glm_adapter,
        "install_dsa_sm90_fp8_scorer",
        lambda: installer_calls.append(True),
    )
    rope_dtypes = []

    def fake_apply_rope(_indexer, x, _freqs, _mscale, _cu_seqlens=None):
        rope_dtypes.append(x.dtype)
        return x

    monkeypatch.setattr(glm_adapter, "_apply_glm52_rope_fp32", fake_apply_rope)
    indexer = _FakeIndexer()
    root = torch.nn.Module()
    root.indexer = indexer

    assert glm_adapter.enable_glm52_dsa_native_fp8(root) == 1
    assert glm_adapter.enable_glm52_dsa_native_fp8(root) == 0
    assert installer_calls == [True]
    q, k, weights = indexer.forward_before_topk(
        torch.randn(2, 1, 8).bfloat16(), torch.randn(2, 1, 4).bfloat16()
    )

    k_projection = indexer.k_projection.float()
    centered_k = k_projection - k_projection.mean(dim=-1, keepdim=True)
    expected_k = centered_k * torch.rsqrt(
        (centered_k * centered_k).mean(dim=-1, keepdim=True) + 1.0e-6
    )
    expected_k = (
        expected_k * indexer.k_norm.weight.float() + indexer.k_norm.bias.float()
    )
    torch.testing.assert_close(
        q, indexer.q_projection.reshape(2, 1, 32, 128).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(weights, indexer.raw_weights.float(), rtol=0, atol=0)
    assert q.dtype == k.dtype == weights.dtype == torch.float32
    assert rope_dtypes == [torch.float32, torch.float32]
    with pytest.raises(RuntimeError, match="bypasses"):
        indexer.forward_with_scores(None, None)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("dsa_indexer_rotate_activation", True),
        ("rotary_interleaved", True),
        ("apply_rope_fusion", True),
        ("bf16", False),
        ("fp16", True),
    ],
)
def test_glm_adapter_rejects_unsupported_preprocessing_config(attribute, value):
    indexer = _FakeIndexer()
    setattr(indexer.config, attribute, value)

    with pytest.raises(ValueError, match=attribute):
        glm_adapter._validate_glm52_native_fp8_indexer(indexer)


def test_glm_adapter_accepts_configured_rope_width_layout_and_epsilon():
    indexer = _FakeIndexer()
    indexer.config.qk_pos_emb_head_dim = 32
    indexer.config.dsa_indexer_rope_interleaved = False
    indexer.config.dsa_indexer_k_norm_epsilon = 0.25

    glm_adapter._validate_glm52_native_fp8_indexer(indexer)


def test_glm_adapter_rejects_invalid_configured_indexer_epsilon():
    indexer = _FakeIndexer()
    indexer.config.dsa_indexer_k_norm_epsilon = -1.0

    with pytest.raises(ValueError, match="epsilon must be finite and positive"):
        glm_adapter._validate_glm52_native_fp8_indexer(indexer)


@pytest.mark.parametrize(
    ("hf_model_id", "expected"),
    [
        ("zai-org/GLM-5.2", True),
        ("/models/GLM-5.2/", True),
        ("/models/GLM-5.2/snapshots/revision", True),
        (
            "/cache/models--zai-org--GLM-5.2/snapshots/cf457fa734ab149f",
            True,
        ),
        ("zai-org/GLM-5.1", False),
        ("zai-org/GLM-5.3", False),
        ("/models/GLM-5.3/snapshots/revision", False),
        ("other/GLM-5.20", False),
    ],
)
def test_glm52_model_identity_detection(hf_model_id, expected):
    assert glm_adapter._is_glm52_model_id(hf_model_id) is expected


def _glm52_model_config():
    return SimpleNamespace(
        experimental_attention_variant="dsa",
        dsa_kernel_backend="cudnn",
        attention_backend="auto",
        dsa_indexer_head_dim=128,
        dsa_indexer_n_heads=32,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=3,
        dsa_indexer_rope_interleaved=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_k_norm_epsilon=1.0e-6,
        qk_pos_emb_head_dim=64,
        rotary_base=8_000_000,
        rope_type="yarn",
        rotary_scaling_factor=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        rotary_interleaved=False,
        apply_rope_fusion=False,
        bf16=True,
        fp16=False,
        layernorm_epsilon=1.0e-5,
    )


def test_local_glm52_contract_fallback_excludes_known_sibling_models():
    model_config = _glm52_model_config()

    assert glm_adapter._selects_glm52_adapter("/models/glm-moe-dsa-tiny", model_config)
    assert not glm_adapter._selects_glm52_adapter("zai-org/GLM-5.3", model_config)
    assert not glm_adapter._selects_glm52_adapter(
        "/models/GLM-5.3/snapshots/revision", model_config
    )

    model_config.dsa_indexer_topk_freq = 1
    assert not glm_adapter._selects_glm52_adapter(
        "/models/unknown-dsa-model", model_config
    )


@pytest.mark.parametrize(
    ("cuda_available", "capability", "expected"),
    [
        (False, (9, 0), False),
        (True, (8, 0), False),
        (True, (9, 0), True),
        (True, (10, 0), False),
    ],
)
def test_sm90_runtime_detection(monkeypatch, cuda_available, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: capability if device == 7 else (-1, -1),
    )

    assert glm_adapter._is_sm90_runtime() is expected


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("dsa_kernel_backend", "none"),
        ("dsa_indexer_loss_coeff", 0.001),
        ("apply_rope_fusion", True),
        ("bf16", False),
    ],
)
def test_automatic_selection_skips_incompatible_glm52_config(
    monkeypatch, attribute, value
):
    model_config = _glm52_model_config()
    setattr(model_config, attribute, value)
    monkeypatch.setattr(glm_adapter, "_is_sm90_runtime", lambda: True)
    monkeypatch.setattr(
        glm_adapter,
        "enable_glm52_dsa_native_fp8",
        lambda _model: pytest.fail("incompatible config must not enable the patch"),
    )

    assert not glm_adapter._supports_glm52_native_fp8_config(model_config)
    assert (
        glm_adapter.maybe_enable_glm52_dsa_native_fp8(
            object(),
            hf_model_id="zai-org/GLM-5.2",
            model_config=model_config,
        )
        is None
    )


@pytest.mark.parametrize(
    ("hf_model_id", "is_sm90", "expected"),
    [
        ("zai-org/GLM-5.2", True, 3),
        ("zai-org/GLM-5.2", False, None),
        ("zai-org/GLM-5.1", True, None),
    ],
)
def test_native_fp8_is_automatically_selected_only_for_glm52_sm90(
    monkeypatch, hf_model_id, is_sm90, expected
):
    calls = []
    monkeypatch.setattr(glm_adapter, "_is_sm90_runtime", lambda: is_sm90)
    monkeypatch.setattr(
        glm_adapter,
        "enable_glm52_dsa_native_fp8",
        lambda model: calls.append(model) or 3,
    )
    model = object()

    actual = glm_adapter.maybe_enable_glm52_dsa_native_fp8(
        model,
        hf_model_id=hf_model_id,
        model_config=_glm52_model_config(),
    )

    assert actual == expected
    assert calls == ([model] if expected is not None else [])


def test_automatic_enable_allows_pipeline_stage_without_local_indexer(monkeypatch):
    monkeypatch.setattr(glm_adapter, "DSAIndexer", _FakeIndexer)
    monkeypatch.setattr(glm_adapter, "_is_sm90_runtime", lambda: True)
    installer_calls = []
    monkeypatch.setattr(
        glm_adapter,
        "install_dsa_sm90_fp8_scorer",
        lambda: installer_calls.append(True),
    )

    assert (
        glm_adapter.maybe_enable_glm52_dsa_native_fp8(
            torch.nn.Module(),
            hf_model_id="zai-org/GLM-5.2",
            model_config=_glm52_model_config(),
        )
        == 0
    )
    assert installer_calls == []
