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


def get_export_mapping():
    mapping = {
        "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
    }
    return mapping


def get_export_transforms(hf_config):
    transforms = [
        io.state_transform(
            source_key="decoder.layers.*.self_attention.linear_qkv.weight",
            target_key=(
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
            ),
            fn=TransformFns.split_qkv,
        ),
        io.state_transform(
            source_key="decoder.layers.*.mlp.linear_fc1.weight",
            target_key=(
                "model.layers.*.mlp.gate_proj.weight",
                "model.layers.*.mlp.up_proj.weight",
            ),
            fn=TransformFns.split_fc1,
        ),
        io.state_transform(
            source_key="embedding.word_embeddings.weight",
            target_key="model.embed_tokens.weight",
            fn=TransformFns.prune_padding,
        ),
    ]

    if not hf_config.tie_word_embeddings:
        transforms.append(
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            )
        )

    return transforms
