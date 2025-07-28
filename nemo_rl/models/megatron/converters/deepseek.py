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

from nemo.lightning import io
from nemo.lightning.io.state import TransformFns, _ModelState


def get_export_mapping(source, source_config):
    mapping = {
        # Embed
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        # Attention
        "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
        "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
        "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
        # Dense MLP
        "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        # MoE
        "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
        "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        # LM Head
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    # For lite model
    if source_config["q_lora_rank"] is None:
        del mapping["decoder.layers.*.self_attention.linear_q_down_proj.weight"]
        del mapping["decoder.layers.*.self_attention.linear_q_up_proj.weight"]
        mapping["decoder.layers.*.self_attention.linear_q_proj.weight"] = (
            "model.layers.*.self_attn.q_proj.weight"
        )
    # Account for Mcore local spec
    if (
        source_config["q_lora_rank"] is not None
        and "decoder.layers.0.self_attention.q_layernorm.weight" in source
    ):
        mapping["decoder.layers.*.self_attention.q_layernorm.weight"] = mapping.pop(
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight"
        )

    if "decoder.layers.0.self_attention.kv_layernorm.weight" in source:
        mapping["decoder.layers.*.self_attention.kv_layernorm.weight"] = mapping.pop(
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight"
        )

    if source_config.get("moe_router_enable_expert_bias", False):
        mapping.update(
            {
                "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            }
        )
    return mapping


def get_export_transforms():
    transforms = [
        io.state_transform(
            source_key="decoder.layers.*.mlp.linear_fc1.weight",
            target_key=(
                "model.layers.*.mlp.gate_proj.weight",
                "model.layers.*.mlp.up_proj.weight",
            ),
            fn=TransformFns.split_fc1,
        ),
        io.state_transform(
            source_key="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            target_key=(
                "model.layers.*.mlp.experts.*.gate_proj.weight",
                "model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            fn=TransformFns.split_fc1,
        ),
        io.state_transform(
            source_key="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
            target_key=(
                "model.layers.*.mlp.shared_experts.gate_proj.weight",
                "model.layers.*.mlp.shared_experts.up_proj.weight",
            ),
            fn=TransformFns.split_fc1,
        ),
    ]
    return transforms


def get_source_fn(
    source_state_dict: dict[str, Any], source_config: dict[str, Any]
) -> _ModelState:
    """Modify source state_dict before conversion.

    In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to mcore weight
    a) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
    b) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE

    We rename decoder.layers.*.mlp.linear_fc1.layer_norm_weight in the first case to unify key names
    """
    for layer_i in range(source_config["num_layers"]):
        if (
            f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight"
            in source_state_dict
        ):
            weight = source_state_dict.pop(
                f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight"
            )
            source_state_dict[f"decoder.layers.{layer_i}.pre_mlp_layernorm.weight"] = (
                weight
            )
    modified_source = _ModelState(source_state_dict)
    return modified_source
