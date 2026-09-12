# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from nemo_rl.models.policy.lm_policy import _aggregate_megatron_flops_metrics
from nemo_rl.utils.flops_formulas import FLOPSConfig, glm_moe_dsa, qwen3
from nemo_rl.utils.flops_tracker import (
    FLOPTracker,
    convert_config_to_flops_config,
    get_hf_config,
    get_theoretical_tflops,
    is_using_tf32,
)


class GlmMoeDsaConfigForTest(PretrainedConfig):
    model_type = "glm_moe_dsa"

    def __init__(self, **kwargs):
        defaults = {
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "intermediate_size": 12288,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "num_experts_per_tok": 8,
            "vocab_size": 154880,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "first_k_dense_replace": 3,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


@pytest.mark.parametrize("overrides", [{}, {"qk_rope_head_dim": 64}])
def test_get_hf_config_forwards_overrides(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]
) -> None:
    expected_config = PretrainedConfig()
    captured_call: dict[str, Any] = {}

    def fake_from_pretrained(model_name: str, **kwargs: Any) -> PretrainedConfig:
        captured_call["model_name"] = model_name
        captured_call["kwargs"] = kwargs
        return expected_config

    monkeypatch.setattr(
        "nemo_rl.utils.flops_tracker.AutoConfig.from_pretrained",
        fake_from_pretrained,
    )

    actual_config = get_hf_config("test/model", **overrides)

    assert actual_config is expected_config
    assert captured_call == {
        "model_name": "test/model",
        "kwargs": {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            **overrides,
        },
    }


def _qwen3_flops_config(head_dim):
    # Qwen3-235B-A22B-like shape (smaller layer count for a cheap test).
    return FLOPSConfig(
        gbs=1,
        enc_seq_len=4096,
        hs=4096,
        layers=2,
        attention_heads=64,
        query_groups=8,
        head_dim=head_dim,
        moe_ffn_hidden_size=1536,
        moe_router_topk=8,
        vocab_size=151936,
    )


def test_qwen3_flops_head_dim_backward_compat():
    """head_dim=None falls back to hidden_size // num_heads, matching the old formula."""
    assert qwen3(_qwen3_flops_config(None)) == qwen3(_qwen3_flops_config(4096 // 64))


def test_qwen3_flops_wide_attention():
    """Wide attention (num_heads*head_dim > hidden) must count MORE attention FLOPs.

    Qwen3-235B-A22B has head_dim=128, num_heads=64, hidden=4096, so num_heads*head_dim=8192=2*hidden.
    The QKV/output projections and the O(seq^2) scores scale with num_heads*head_dim, not hidden_size,
    so the formula must not collapse head_dim to hidden_size/num_heads.
    """
    standard = qwen3(_qwen3_flops_config(4096 // 64))  # head_dim=64 == hidden/num_heads
    wide = qwen3(_qwen3_flops_config(128))  # head_dim=128 (Qwen3-235B)
    assert wide > standard


def _glm_moe_dsa_flops_config(
    index_topk: int = 2048, index_compute_layers: int = 78
) -> FLOPSConfig:
    return FLOPSConfig(
        gbs=1,
        enc_seq_len=4096,
        hs=6144,
        layers=78,
        ffn_hs=12288,
        attention_heads=64,
        moe_router_topk=8,
        query_groups=64,
        vocab_size=154880,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_head_dim=192,
        qk_pos_emb_head_dim=64,
        v_head_dim=256,
        moe_layer_freq=[0] * 3 + [1] * 75,
        moe_shared_expert_intermediate_size=2048,
        moe_ffn_hidden_size=2048,
        mtp_num_layers=None,
        causal_self_attn=True,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=index_topk,
        dsa_indexer_compute_layers=index_compute_layers,
    )


def test_glm_moe_dsa_flops_scale_with_sparse_topk():
    smaller_topk = glm_moe_dsa(_glm_moe_dsa_flops_config(index_topk=1024))
    larger_topk = glm_moe_dsa(_glm_moe_dsa_flops_config(index_topk=4096))
    assert larger_topk > smaller_topk


def test_glm_moe_dsa_flops_components():
    config = FLOPSConfig(
        gbs=2,
        enc_seq_len=8,
        hs=4,
        layers=2,
        ffn_hs=8,
        attention_heads=2,
        moe_router_topk=2,
        vocab_size=16,
        q_lora_rank=3,
        kv_lora_rank=2,
        qk_head_dim=2,
        qk_pos_emb_head_dim=1,
        v_head_dim=2,
        moe_layer_freq=[0, 1],
        moe_shared_expert_intermediate_size=2,
        moe_ffn_hidden_size=2,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=2,
        dsa_indexer_topk=4,
        dsa_indexer_compute_layers=1,
    )

    # Per input: 15,168 linear + 2,880 sparse attention + 736 indexer
    # + 3,072 vocabulary projection FLOPs; gbs=2.
    assert glm_moe_dsa(config) == 43_712


def _glm_5_2_indexer_types() -> list[str]:
    return [
        "full" if layer_number <= 3 or (layer_number - 3) % 4 == 0 else "shared"
        for layer_number in range(1, 79)
    ]


@pytest.mark.parametrize(
    ("model_config", "expected_index_compute_layers"),
    [
        (GlmMoeDsaConfigForTest(), 78),
        (
            GlmMoeDsaConfigForTest(
                mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
                indexer_types=_glm_5_2_indexer_types(),
            ),
            21,
        ),
    ],
    ids=["glm-5.1", "glm-5.2"],
)
def test_glm_moe_dsa_config_is_supported(model_config, expected_index_compute_layers):
    flops_config, flops_formula = convert_config_to_flops_config(model_config)

    assert flops_formula is glm_moe_dsa
    assert flops_config.moe_layer_freq == [0] * 3 + [1] * 75
    assert flops_config.moe_shared_expert_intermediate_size == 2048
    assert flops_config.dsa_indexer_n_heads == 32
    assert flops_config.dsa_indexer_head_dim == 128
    assert flops_config.dsa_indexer_topk == 2048
    assert flops_config.dsa_indexer_compute_layers == expected_index_compute_layers

    flops_tracker = FLOPTracker.from_config("glm-moe-dsa-test", model_config)
    flops_tracker.track(n_samples=1, padded_seq_len=4096)
    assert flops_tracker.total_flops > 0


def test_glm_5_2_reused_indices_reduce_indexer_flops():
    glm_5_1_config = _glm_moe_dsa_flops_config(index_compute_layers=78)
    glm_5_2_config = _glm_moe_dsa_flops_config(index_compute_layers=21)

    actual_difference = glm_moe_dsa(glm_5_1_config) - glm_moe_dsa(glm_5_2_config)
    seq_len = glm_5_1_config.enc_seq_len
    dense_causal_pairs = seq_len * (seq_len + 1) // 2
    projection_params = 2048 * 32 * 128 + 6144 * 128 + 6144 * 32
    per_layer_indexer_flops = 2 * (
        seq_len * projection_params + dense_causal_pairs * 32 * 128
    )
    assert actual_difference == (78 - 21) * per_layer_indexer_flops


def test_worker_total_flops_aggregation_megatron_path():
    """Verify _aggregate_megatron_flops_metrics for the basic case (no train_elapsed_seconds)."""
    world_size = 8
    results = [
        {
            "total_flops": 1.0e15,
            "num_ranks": world_size,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "model_dtype": torch.bfloat16,
        }
    ]

    aggregated_results = _aggregate_megatron_flops_metrics(results, world_size)

    assert aggregated_results["total_flops"] == pytest.approx(1.0e15)
    assert aggregated_results["num_ranks"] == 8
    assert "train_elapsed_seconds" not in aggregated_results
    # 8 GPUs × (1979/2 TFLOPS) for H100 bfloat16
    assert aggregated_results["theoretical_tflops"] == pytest.approx(8 * 1979 / 2)


def test_worker_total_flops_aggregation_megatron_path_with_elapsed():
    """Verify train_elapsed_seconds is forwarded when present in worker results."""
    world_size = 4
    results = [
        {
            "total_flops": 2.0e15,
            "num_ranks": world_size,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "model_dtype": torch.bfloat16,
            "train_elapsed_seconds": 3.5,
        }
    ]

    aggregated_results = _aggregate_megatron_flops_metrics(results, world_size)

    assert aggregated_results["total_flops"] == pytest.approx(2.0e15)
    assert aggregated_results["num_ranks"] == 4
    assert aggregated_results["train_elapsed_seconds"] == pytest.approx(3.5)
    assert aggregated_results["theoretical_tflops"] == pytest.approx(4 * 1979 / 2)


def test_worker_total_flops_aggregation_unknown_gpu_warns():
    """Verify a warning is emitted and theoretical_tflops is absent for unknown GPUs."""
    world_size = 2
    results = [
        {
            "total_flops": 1.0e14,
            "num_ranks": world_size,
            "gpu_name": "NVIDIA UNKNOWN GPU XYZ",
            "model_dtype": torch.bfloat16,
        }
    ]

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aggregated_results = _aggregate_megatron_flops_metrics(results, world_size)

    assert aggregated_results["total_flops"] == pytest.approx(1.0e14)
    assert aggregated_results["num_ranks"] == 2
    assert "theoretical_tflops" not in aggregated_results
    assert len(w) == 1
    assert "theoretical flops" in str(w[0].message).lower()


@pytest.mark.parametrize(
    "device_name, model_dtype, tflops",
    [
        ("NVIDIA A100 80GB PCIe", torch.bfloat16, 624 / 2),
        ("NVIDIA A100 80GB PCIe", torch.float32, 312 / 2 if is_using_tf32() else 19.5),
        ("NVIDIA H100 80GB HBM3", torch.bfloat16, 1979 / 2),
        ("NVIDIA H100 80GB HBM3", torch.float32, 989 / 2 if is_using_tf32() else 67.0),
        ("NVIDIA H200", torch.bfloat16, 1979 / 2),
        ("NVIDIA H200", torch.float32, 989 / 2 if is_using_tf32() else 67.0),
        ("NVIDIA B200", torch.bfloat16, 4500 / 2),
        ("NVIDIA B200", torch.float32, 2200 / 2 if is_using_tf32() else 80.0),
        ("NVIDIA B300", torch.bfloat16, 4500 / 2),
        ("NVIDIA B300", torch.float32, 2200 / 2 if is_using_tf32() else 80.0),
        ("NVIDIA GB200", torch.bfloat16, 4900 / 2),
        ("NVIDIA GB200", torch.float32, 2500 / 2 if is_using_tf32() else 80.0),
        ("NVIDIA GB300", torch.bfloat16, 4900 / 2),
        ("NVIDIA GB300", torch.float32, 2500 / 2 if is_using_tf32() else 80.0),
    ],
)
def test_theoretical_tflops(device_name, model_dtype, tflops):
    assert get_theoretical_tflops(device_name, model_dtype) == pytest.approx(tflops)
