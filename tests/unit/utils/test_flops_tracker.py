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

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from nemo_rl.utils.flops_formulas import FLOPSConfig, glm_moe_dsa, qwen3
from nemo_rl.utils.flops_tracker import (
    FLOPTracker,
    convert_config_to_flops_config,
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


def _glm_moe_dsa_flops_config(index_topk=2048):
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
    )


def test_glm_moe_dsa_flops_scale_with_sparse_topk():
    smaller_topk = glm_moe_dsa(_glm_moe_dsa_flops_config(index_topk=1024))
    larger_topk = glm_moe_dsa(_glm_moe_dsa_flops_config(index_topk=4096))
    assert larger_topk > smaller_topk


def test_glm_moe_dsa_config_is_supported():
    config = GlmMoeDsaConfigForTest()

    flops_config, flops_formula = convert_config_to_flops_config(config)
    assert flops_formula is glm_moe_dsa
    assert flops_config.moe_layer_freq == [0] * 3 + [1] * 75
    assert flops_config.moe_shared_expert_intermediate_size == 2048
    assert flops_config.dsa_indexer_n_heads == 32
    assert flops_config.dsa_indexer_head_dim == 128
    assert flops_config.dsa_indexer_topk == 2048

    flops_tracker = FLOPTracker.from_config("glm-moe-dsa-test", config)
    flops_tracker.track(n_samples=1, padded_seq_len=4096)
    assert flops_tracker.total_flops > 0


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
