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

from nemo_rl.utils.flops_tracker import FLOPTracker, get_hf_config


@pytest.mark.parametrize(
    "model_name, gbs, seqlen, expected_flops",
    [
        ("meta-llama/Llama-2-7b-hf", 128, 4096, 2.25e16),
        ("meta-llama/Llama-2-13b-hf", 128, 4096, 4.17e16),
        ("meta-llama/Llama-2-70b-hf", 128, 4096, 2.25e17),
        ("meta-llama/Meta-Llama-3-8B", 128, 8192, 5.31e16),
        ("meta-llama/Llama-3.1-70B-Instruct", 128, 8192, 4.71e17),
        ("meta-llama/Llama-3.1-405B-Instruct", 128, 8192, 2.65e18),
        ("Qwen/Qwen3-30B-A3B", 128, 4096, 1.21e16),
        ("Qwen/Qwen3-235B-A22B", 128, 4096, 7.76e16),
        ("deepseek-ai/DeepSeek-V3", 1, 4096, 1.023e15),
        ("moonshotai/Moonlight-16B-A3B-Instruct", 1, 4096, 6.45e13),
    ],
)
def test_flops_counter(model_name, gbs, seqlen, expected_flops):
    model_config = get_hf_config(model_name)
    flops_tracker = FLOPTracker.from_config(model_name, model_config)
    flops_tracker.track(gbs, seqlen)

    # check within 5% relative difference
    assert abs(flops_tracker.total_flops - expected_flops) / expected_flops <= 0.05, (
        f"Expected {expected_flops} flops, got {flops_tracker.total_flops}"
    )


def test_qwen2_flops_count_gqa_kv_projections():
    # Qwen2.5-Math-1.5B shapes: 12 query heads share 2 KV heads (head_dim 128).
    from nemo_rl.utils.flops_formulas import FLOPSConfig, qwen2

    hs, layers, ffn, vocab, seq = 1536, 28, 8960, 151936, 4096
    heads, kv_heads = 12, 2
    kv_proj = kv_heads * (hs // heads)
    # fwd+bwd = 3 * 2 * MACs; causal attention scores count half of QK^T and AV.
    attn = layers * seq * (hs * hs + 2 * hs * kv_proj + hs * hs + seq * hs)
    mlp = layers * seq * 3 * hs * ffn
    expected = 6 * (attn + mlp + seq * hs * vocab)

    config = FLOPSConfig(
        gbs=1,
        enc_seq_len=seq,
        hs=hs,
        layers=layers,
        ffn_hs=ffn,
        vocab_size=vocab,
        query_groups=kv_heads,
        attention_heads=heads,
    )
    assert qwen2(config) == pytest.approx(expected)
