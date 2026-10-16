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

"""Per-tensor MXFP8 selection rules.

``should_quantize`` is the contract offline conversion and online refit share,
so a change here silently desynchronizes a checkpoint from its refits.
"""

import pytest
import torch

from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    should_quantize,
    strip_weight_suffix,
)

_QUANTIZABLE = torch.ones((32, 64), dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "name",
    [
        # Every MoE router spelling in use across the model families this repo
        # runs. Routers drive a top-k argmax, so quantizing one shifts expert
        # selection rather than raising -- see ``DEFAULT_NVFP4_IGNORE`` in
        # ``nemo_rl/modelopt/utils.py``, which excludes the same names.
        "model.layers.0.mlp.gate.weight",  # Qwen3-MoE, DeepSeek
        "model.layers.0.mlp.router.weight",  # GPT-OSS
        "model.layers.0.feed_forward.router.weight",  # Llama-4
        "model.layers.0.block_sparse_moe.gate.weight",  # Mixtral
        "backbone.layers.24.mixer.gate.weight",  # NemotronH / nanov3
        "model.layers.0.mlp.shared_expert_gate.weight",  # Qwen2-MoE
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ],
)
def test_precision_sensitive_modules_stay_high_precision(name: str) -> None:
    assert not should_quantize(name, _QUANTIZABLE)


@pytest.mark.parametrize(
    "name",
    [
        # The expert MLPs are what quantization is for.
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        # ``shared_expert.gate_proj`` is an MLP weight; only the separate
        # ``shared_expert_gate`` is a router. The underscore is the whole
        # difference, so this pins that boundary.
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        # ``.gate.`` must not swallow the gate projections.
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "backbone.layers.24.mixer.experts.7.up_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ],
)
def test_projection_weights_are_still_quantized(name: str) -> None:
    assert should_quantize(name, _QUANTIZABLE)


@pytest.mark.parametrize(
    ("name", "weight"),
    [
        # Only ``.weight`` tensors get a block-scale companion.
        ("model.layers.0.self_attn.q_proj.bias", _QUANTIZABLE),
        # Already-quantized or integer tensors pass through untouched.
        (
            "model.layers.0.self_attn.q_proj.weight",
            torch.ones((32, 64), dtype=torch.uint8),
        ),
        # A 1-D tensor has no block dimension to scale.
        ("model.layers.0.self_attn.q_proj.weight", torch.ones(64)),
        # The last dim must tile the 1x32 MXFP8 block exactly.
        (
            "model.layers.0.self_attn.q_proj.weight",
            torch.ones((32, 48), dtype=torch.bfloat16),
        ),
    ],
)
def test_unquantizable_tensors_are_rejected(name: str, weight: torch.Tensor) -> None:
    assert not should_quantize(name, weight)


def test_strip_weight_suffix_rejects_a_non_weight_key() -> None:
    assert (
        strip_weight_suffix("model.layers.0.mlp.down_proj.weight")
        == "model.layers.0.mlp.down_proj"
    )
    with pytest.raises(ValueError):
        strip_weight_suffix("model.layers.0.mlp.down_proj.weight_scale_inv")
