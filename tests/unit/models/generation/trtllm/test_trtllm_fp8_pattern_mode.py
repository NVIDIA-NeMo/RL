# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Refit scope of ``quantization_ignore_patterns`` (pattern mode) on Qwen3.5.

Builds tiny random-weight Qwen3.5 models (no checkpoint, no engine) with the
quantization config NeMo RL hands TRT-LLM, and checks which refit weights are FP8.
"""

import json

import pytest
import torch

pytestmark = pytest.mark.trtllm

pytest.importorskip("tensorrt_llm")
transformers = pytest.importorskip("transformers")

if not torch.cuda.is_available():
    pytest.skip("TRT-LLM model construction needs a GPU", allow_module_level=True)

from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM  # noqa: E402

from nemo_rl.models.generation.trtllm.quantization import fp8  # noqa: E402

_TEXT = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "dtype": "bfloat16",
    "eos_token_id": 1,
    "full_attention_interval": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 256,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 32,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 4096,
    "mlp_only_layers": [],
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
    "use_cache": True,
    "vocab_size": 1024,
}
_VISION = {
    "deepstack_visual_indexes": [],
    "depth": 1,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 64,
    "in_channels": 3,
    "intermediate_size": 128,
    "num_heads": 2,
    "num_position_embeddings": 2304,
    "out_hidden_size": 256,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
_MOE_TEXT = {
    "model_type": "qwen3_5_moe_text",
    "moe_intermediate_size": 128,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_expert_intermediate_size": 128,
}
_DENSE_TEXT = {"model_type": "qwen3_5_text", "intermediate_size": 512}


def _config(kind: str, quantization_config: dict) -> dict:
    moe = kind == "moe"
    return {
        "architectures": [
            "Qwen3_5MoeForConditionalGeneration"
            if moe
            else "Qwen3_5ForConditionalGeneration"
        ],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "text_config": {**_TEXT, **(_MOE_TEXT if moe else _DENSE_TEXT)},
        "tie_word_embeddings": False,
        "video_token_id": 248057,
        "vision_config": {
            **_VISION,
            "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
        "quantization_config": quantization_config,
    }


def _hf_state_dict_info(path) -> dict:
    config = transformers.AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = transformers.AutoModelForImageTextToText.from_config(config)
    return {
        name: (tensor.shape, torch.bfloat16)
        for name, tensor in model.state_dict().items()
        if not name.startswith("mtp.")
    }


@pytest.mark.parametrize("is_mx", [False, True])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_pattern_mode_quantizes_what_the_engine_holds_as_fp8(tmp_path, kind, is_mx):
    quantization_config = fp8.build_quant_config(is_mx, ["*lm_head*"])
    (tmp_path / "config.json").write_text(
        json.dumps(_config(kind, quantization_config))
    )

    model_config = ModelConfig.from_pretrained(str(tmp_path))
    model = AutoModelForCausalLM.from_config(model_config)
    info = _hf_state_dict_info(tmp_path)

    quantized = fp8.resolve_quantized_weights(model, model_config.quant_config, info)

    def scope(suffix):
        return {n for n in info if n.endswith(suffix)}

    # Attention and linear-attention output projections are FP8; the rest stay BF16.
    assert scope("self_attn.q_proj.weight") <= quantized
    assert scope("linear_attn.out_proj.weight") <= quantized
    for bf16 in (
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "lm_head.weight",
        "embed_tokens.weight",
    ):
        assert not scope(bf16) & quantized, bf16
    if kind == "moe":
        assert scope("mlp.experts.gate_up_proj") <= quantized
        assert scope("mlp.shared_expert.up_proj.weight") <= quantized
        for bf16 in ("mlp.gate.weight", "mlp.shared_expert_gate.weight"):
            assert not scope(bf16) & quantized, bf16
    else:
        assert scope("mlp.down_proj.weight") <= quantized
