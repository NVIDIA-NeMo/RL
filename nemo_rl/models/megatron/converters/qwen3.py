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

from nemo.lightning import io
from nemo.lightning.io.state import TransformFns


def get_export_mapping(config):
    mapping = {
        "**.self_attention.linear_proj.weight": "**.self_attn.o_proj.weight",
        "**.self_attention.linear_qkv.layer_norm_weight": "**.input_layernorm.weight",
        "**.self_attention.q_layernorm.weight": "**.self_attn.q_norm.weight",
        "**.self_attention.k_layernorm.weight": "**.self_attn.k_norm.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
    }
    is_moe = getattr(config, "num_experts", 0) > 0
    if is_moe:
        mapping.update(
            {
                "**.mlp.experts.linear_fc2.weight*": "**.mlp.experts.*.down_proj.weight",
                "**.mlp.router.weight": "**.mlp.gate.weight",
                "**.pre_mlp_layernorm.weight": "**.post_attention_layernorm.weight",
            }
        )
    else:
        mapping.update(
            {
                "**.mlp.linear_fc2.weight": "**.mlp.down_proj.weight",
                "**.mlp.linear_fc1.layer_norm_weight": "**.post_attention_layernorm.weight",
            }
        )
    return mapping


def get_export_transforms(config):
    is_moe = getattr(config, "num_experts", 0) > 0
    transforms = [
        io.state_transform(
            source_key="**.self_attention.linear_qkv.weight",
            target_key=(
                "**.self_attn.q_proj.weight",
                "**.self_attn.k_proj.weight",
                "**.self_attn.v_proj.weight",
            ),
            fn=TransformFns.split_qkv,
        ),
        (
            io.state_transform(
                source_key="**.mlp.linear_fc1.weight",
                target_key=("**.mlp.gate_proj.weight", "**.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            )
            if not is_moe
            else io.state_transform(
                source_key="**.mlp.experts.linear_fc1.weight*",
                target_key=(
                    "**.mlp.experts.*.gate_proj.weight",
                    "**.mlp.experts.*.up_proj.weight",
                ),
                fn=TransformFns.split_fc1,
            )
        ),
        io.state_transform(
            source_key="embedding.word_embeddings.weight",
            target_key="model.embed_tokens.weight",
            fn=TransformFns.prune_padding,
        ),
    ]
    if not config.tie_word_embeddings:
        transforms.append(
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            )
        )

    return transforms
