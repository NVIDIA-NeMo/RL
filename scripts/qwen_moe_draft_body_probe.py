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

"""Exercise the real Qwen-MoE DFlash and DSpark body constructors without pytest."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from nemo_rl.models.megatron.draft.dflash import DFlashBodyConfig
from nemo_rl.models.megatron.draft.training import DSparkSpeculator
from nemo_rl.models.policy.draft_config import DSparkDraftConfig


def _qwen3_30b_a3b_body_config() -> DFlashBodyConfig:
    return DFlashBodyConfig(
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_hidden_layers=5,
        num_target_taps=5,
    )


def _build_asymmetric_dspark_body() -> None:
    provider = DSparkSpeculator(
        DSparkDraftConfig(
            enabled=True,
            model_name=None,
            block_size=8,
            anchors_per_sample=2,
            mask_token_id=151669,
            target_hidden_state_layer_ids=[1, 12, 23, 34, 45],
            num_layers=5,
            markov_rank=8,
            confidence_enabled=True,
            confidence_with_markov=True,
        )
    )
    target = SimpleNamespace(
        num_layers=48,
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        fp16=False,
        bf16=False,
        params_dtype=torch.float32,
        hidden_size=8,
        ffn_hidden_size=24,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=4,
        rotary_base=10_000.0,
        layernorm_epsilon=1e-6,
        init_method_std=0.02,
        vocab_size=32,
    )
    adapter = provider.build_model(
        model_provider=target,
        pg_collection=SimpleNamespace(tp=None),
        policy_model_chunk=SimpleNamespace(),
    )
    if adapter is None:
        raise RuntimeError("enabled DSpark provider did not build an adapter")
    q_proj_shape = tuple(adapter.body.layers[0].self_attn.q_proj.weight.shape)
    o_proj_shape = tuple(adapter.body.layers[0].self_attn.o_proj.weight.shape)
    if q_proj_shape != (16, 8) or o_proj_shape != (8, 16):
        raise AssertionError(
            f"unexpected asymmetric projection shapes: q={q_proj_shape}, o={o_proj_shape}"
        )


def main() -> None:
    config = _qwen3_30b_a3b_body_config()
    if config.num_attention_heads * config.head_dim != 4096:
        raise AssertionError("Qwen3-30B-A3B query projection width must be 4096")
    _build_asymmetric_dspark_body()
    print("QWEN_MOE_DRAFT_BODY_PROBE_PASS")


if __name__ == "__main__":
    main()
