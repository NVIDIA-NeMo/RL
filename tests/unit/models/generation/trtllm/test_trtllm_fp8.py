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

import json

import pytest
import torch

from nemo_rl.models.generation.trtllm.quantization.fp8 import (
    E4M3_MAX,
    FP8_BLOCK_QUANT_KWARGS,
    MXFP8_BLOCK_QUANT_KWARGS,
    MXFP8_BLOCK_SIZE,
    UE8M0_BIAS,
    build_quant_config,
    cast_tensor_to_fp8_blockwise,
    cast_tensor_to_mxfp8_blockwise,
    configure_fp8_llm_kwargs,
    configure_fp8_moe_backend,
    load_weights,
    resolve_quantized_weights,
    validate_fused_expert_layout,
    validate_routed_experts,
)

pytestmark = pytest.mark.trtllm


def _block_matrix(values: list[list[float]]) -> torch.Tensor:
    return torch.cat(
        [
            torch.cat(
                [torch.full((128, 128), value) for value in row],
                dim=1,
            )
            for row in values
        ],
        dim=0,
    )


def test_fused_expert_layout():
    prefix = "model.layers.0.mlp.experts"
    valid = {
        f"{prefix}.gate_up_proj": (torch.Size([256, 1024, 2048]), torch.bfloat16),
        f"{prefix}.down_proj": (torch.Size([256, 2048, 512]), torch.bfloat16),
    }

    validate_fused_expert_layout(valid)

    valid[f"{prefix}.gate_up_proj"] = (
        torch.Size([256, 2048, 1024]),
        torch.bfloat16,
    )
    with pytest.raises(ValueError, match=r"gate_up_proj=\[E,2I,H\]"):
        validate_fused_expert_layout(valid)


def test_missing_expert_layout():
    with pytest.raises(ValueError, match="No Qwen3.5 fused routed-expert weights"):
        validate_fused_expert_layout({})


def test_fp8_config_preserves_overrides():
    llm_kwargs = {
        "dtype": "fp8",
        "model_kwargs": {"pretrained_config": {"num_hidden_layers": 4}},
    }

    configure_fp8_llm_kwargs(llm_kwargs)

    assert llm_kwargs["dtype"] == "bfloat16"
    assert llm_kwargs["load_format"] == "dummy"
    assert llm_kwargs["use_cute_dsl_blockscaling_mm"] is True
    assert llm_kwargs["model_kwargs"]["pretrained_config"] == {"num_hidden_layers": 4}
    assert llm_kwargs["model_kwargs"]["quantization_config"] == FP8_BLOCK_QUANT_KWARGS


@pytest.mark.parametrize(
    "llm_kwargs",
    [
        {"load_format": "auto"},
        {"model_kwargs": {"quantization_config": {"quant_method": "modelopt"}}},
    ],
)
def test_configure_fp8_llm_kwargs_rejects_unsupported_contract(llm_kwargs):
    with pytest.raises(ValueError, match="precision='fp8'"):
        configure_fp8_llm_kwargs(llm_kwargs)


class _MoeConfig:
    def __init__(self, backend="AUTO", **kwargs) -> None:
        self.backend = backend
        self.kwargs = kwargs


def test_configure_fp8_moe_backend_preserves_other_fields():
    llm_kwargs = {
        "moe_config": {
            "backend": "trtllm",
            "load_balancer_config": {"num_slots": 16},
        }
    }

    configure_fp8_moe_backend(llm_kwargs, _MoeConfig)

    assert isinstance(llm_kwargs["moe_config"], _MoeConfig)
    assert llm_kwargs["moe_config"].backend == "TRTLLM"
    assert llm_kwargs["moe_config"].kwargs == {
        "load_balancer_config": {"num_slots": 16}
    }


@pytest.mark.parametrize(
    "moe_config",
    [
        {"backend": "DEEPGEMM"},
        _MoeConfig(backend="CUTLASS"),
    ],
)
def test_configure_fp8_moe_backend_rejects_non_trtllm(moe_config):
    with pytest.raises(ValueError, match="backend='TRTLLM'"):
        configure_fp8_moe_backend({"moe_config": moe_config}, _MoeConfig)


def test_mxfp8_scale_is_ue8m0_exponent_and_round_trips():
    torch.manual_seed(0)
    source = (torch.randn(4, 64, dtype=torch.float32) * 100.0).to(torch.bfloat16)

    quantized, scale_ue8m0 = cast_tensor_to_mxfp8_blockwise(source)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale_ue8m0.dtype == torch.uint8
    assert scale_ue8m0.shape == (4, 64 // MXFP8_BLOCK_SIZE)

    blocked = source.float().reshape(4, -1, MXFP8_BLOCK_SIZE)
    expected_exponent = torch.ceil(torch.log2(blocked.abs().amax(-1) / E4M3_MAX))
    assert torch.equal(scale_ue8m0, (expected_exponent + UE8M0_BIAS).to(torch.uint8))

    scale = torch.exp2(scale_ue8m0.float() - UE8M0_BIAS).repeat_interleave(
        MXFP8_BLOCK_SIZE, dim=-1
    )
    assert torch.allclose(
        quantized.float() * scale, source.float(), rtol=1.0 / 16, atol=1e-3
    )
    assert quantized.float().abs().max() <= E4M3_MAX


@pytest.mark.parametrize("backend", ["CUTLASS", "cutedsl", "CuteDSL"])
def test_configure_fp8_moe_backend_mxfp8_accepts_cutlass_and_cutedsl(backend):
    """MXFP8 experts run on CUTLASS or on the fused FC1+FC2 CuTe DSL kernel."""
    llm_kwargs = {"moe_config": {"backend": backend, "load_balancer_config": {"n": 1}}}
    configure_fp8_moe_backend(llm_kwargs, _MoeConfig, is_mx=True)
    assert llm_kwargs["moe_config"].backend == backend.upper()
    assert llm_kwargs["moe_config"].kwargs == {"load_balancer_config": {"n": 1}}

    llm_kwargs = {"moe_config": _MoeConfig(backend=backend)}
    configure_fp8_moe_backend(llm_kwargs, _MoeConfig, is_mx=True)
    assert llm_kwargs["moe_config"].backend == backend


def test_configure_fp8_moe_backend_mxfp8_defaults_to_cutlass():
    llm_kwargs = {}
    configure_fp8_moe_backend(llm_kwargs, _MoeConfig, is_mx=True)
    assert llm_kwargs["moe_config"].backend == "CUTLASS"


@pytest.mark.parametrize(
    "moe_config",
    [
        {"backend": "TRTLLM"},
        _MoeConfig(backend="DEEPGEMM"),
    ],
)
def test_configure_fp8_moe_backend_mxfp8_rejects_other_backends(moe_config):
    with pytest.raises(ValueError, match="backend='CUTLASS' or .*backend='CUTEDSL'"):
        configure_fp8_moe_backend({"moe_config": moe_config}, _MoeConfig, is_mx=True)


def test_mxfp8_llm_kwargs_use_the_mx_quant_contract():
    llm_kwargs = {}

    configure_fp8_llm_kwargs(llm_kwargs, is_mx=True)

    quant = llm_kwargs["model_kwargs"]["quantization_config"]
    assert quant["quant_method"] == "mxfp8"
    assert quant["weight_block_size"] == [1, MXFP8_BLOCK_SIZE]
    assert quant == MXFP8_BLOCK_QUANT_KWARGS


def test_block_fp8_scale_orientation_is_out_block_by_in_block():
    source = _block_matrix([[1.0, 2.0], [3.0, 4.0]])

    fp8_data, scale_inv = cast_tensor_to_fp8_blockwise(source)

    assert fp8_data.dtype == torch.float8_e4m3fn
    assert scale_inv.dtype == torch.float32
    assert scale_inv.shape == (2, 2)
    assert torch.equal(
        scale_inv,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 448.0,
    )
    for row_block in range(2):
        for column_block in range(2):
            block = fp8_data[
                row_block * 128 : (row_block + 1) * 128,
                column_block * 128 : (column_block + 1) * 128,
            ]
            dequantized = block.float() * scale_inv[row_block, column_block]
            assert torch.equal(
                dequantized,
                source[
                    row_block * 128 : (row_block + 1) * 128,
                    column_block * 128 : (column_block + 1) * 128,
                ],
            )


def test_block_fp8_handles_batched_non_aligned_zero_weights():
    source = torch.zeros((2, 130, 259), dtype=torch.bfloat16)

    fp8_data, scale_inv = cast_tensor_to_fp8_blockwise(source)

    assert fp8_data.shape == source.shape
    assert scale_inv.shape == (2, 2, 3)
    assert torch.count_nonzero(fp8_data.float()) == 0
    assert torch.equal(scale_inv, torch.ones_like(scale_inv))


def test_block_fp8_round_trip_preserves_intra_block_element_order():
    """Every element in the tests above is constant within its 128x128 block,
    so a dropped/wrong inverse permute at the reshape back to [rows, cols]
    would still pass them silently. Random (not periodic -- a ramp's
    symmetry can coincidentally survive a scramble) values make intra-block
    position matter."""
    torch.manual_seed(0)
    source = (torch.randn(256, 256, dtype=torch.float32) * 10.0).to(torch.bfloat16)

    fp8_data, scale_inv = cast_tensor_to_fp8_blockwise(source)

    for row_block in range(2):
        for column_block in range(2):
            block = fp8_data[
                row_block * 128 : (row_block + 1) * 128,
                column_block * 128 : (column_block + 1) * 128,
            ]
            dequantized = block.float() * scale_inv[row_block, column_block]
            expected = source[
                row_block * 128 : (row_block + 1) * 128,
                column_block * 128 : (column_block + 1) * 128,
            ].float()
            assert torch.allclose(dequantized, expected, rtol=1.0 / 8, atol=1.0)


def test_routed_expert_conversion():
    prefix = "model.language_model.layers.3.mlp.experts"
    gate = torch.stack([_block_matrix([[1.0, 2.0]]), _block_matrix([[3.0, 4.0]])])
    up = torch.stack([_block_matrix([[5.0, 6.0]]), _block_matrix([[7.0, 8.0]])])
    down = torch.stack(
        [_block_matrix([[9.0], [10.0]]), _block_matrix([[11.0], [12.0]])]
    )
    gate_up = torch.cat((gate, up), dim=1).to(torch.bfloat16)
    down = down.to(torch.bfloat16)
    mtp_expert_name = "mtp.layers.0.mlp.experts.7.down_proj.weight"
    mtp_expert = torch.randn(128, 128, dtype=torch.bfloat16)
    passthrough = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.0.linear_attn.in_proj_qkvz.weight": torch.randn(128, 128),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 128),
        "model.language_model.layers.3.mlp.gate.weight": torch.randn(2, 256),
    }

    converted = load_weights(
        [
            (f"{prefix}.gate_up_proj", gate_up),
            (f"{prefix}.down_proj", down),
            (mtp_expert_name, mtp_expert),
            *passthrough.items(),
        ]
    )

    assert f"{prefix}.gate_up_proj" not in converted
    assert f"{prefix}.down_proj" not in converted

    source_projections = {
        "gate_proj": gate,
        "up_proj": up,
        "down_proj": down,
    }
    for expert_index in range(2):
        for projection_name, source in source_projections.items():
            weight_name = f"{prefix}.{expert_index}.{projection_name}.weight"
            scale_name = weight_name.removesuffix(".weight") + ".weight_scale_inv"
            expected_weight, expected_scale = cast_tensor_to_fp8_blockwise(
                source[expert_index]
            )
            assert torch.equal(converted[weight_name].float(), expected_weight.float())
            assert torch.equal(converted[scale_name], expected_scale)
            assert converted[weight_name].dtype == torch.float8_e4m3fn
            assert converted[scale_name].dtype == torch.float32
    assert converted[mtp_expert_name].dtype == torch.float8_e4m3fn
    assert (
        converted["mtp.layers.0.mlp.experts.7.down_proj.weight_scale_inv"].dtype
        == torch.float32
    )
    for name, tensor in passthrough.items():
        assert converted[name] is tensor
        assert name.removesuffix(".weight") + ".weight_scale_inv" not in converted


def test_qwen3_moe_per_expert_weights_are_quantized():
    prefix = "model.layers.1.mlp.experts"
    experts = {
        f"{prefix}.{i}.{proj}.weight": torch.randn(128, 128, dtype=torch.bfloat16)
        for i in range(2)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    passthrough = {
        "model.layers.1.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.1.mlp.gate.weight": torch.randn(2, 128),
    }
    validate_routed_experts(
        {name: (tensor.shape, tensor.dtype) for name, tensor in experts.items()}
    )

    converted = load_weights([*experts.items(), *passthrough.items()])

    for name in experts:
        assert converted[name].dtype == torch.float8_e4m3fn
        scale = converted[name.removesuffix(".weight") + ".weight_scale_inv"]
        assert scale.dtype == torch.float32
    for name, tensor in passthrough.items():
        assert converted[name] is tensor


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (["model.layers.0.self_attn.q_proj.weight"], "no routed-expert weights"),
        (
            [
                "model.layers.0.mlp.experts.0.down_proj.weight",
                "model.layers.1.mlp.down_proj.weight",
            ],
            "dense MLP weights",
        ),
    ],
)
def test_validate_routed_experts_rejects_unquantizable_models(names, match):
    state_dict_info = {name: (torch.Size([128, 128]), torch.bfloat16) for name in names}

    with pytest.raises(ValueError, match=match):
        validate_routed_experts(state_dict_info)


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
    import transformers

    config = transformers.AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = transformers.AutoModelForImageTextToText.from_config(config)
    return {
        name: (tensor.shape, torch.bfloat16)
        for name, tensor in model.state_dict().items()
        if not name.startswith("mtp.")
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TRT-LLM model construction needs a GPU"
)
@pytest.mark.parametrize("is_mx", [False, True])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_pattern_mode_quantizes_what_the_engine_holds_as_fp8(tmp_path, kind, is_mx):
    # Tiny random-weight Qwen3.5 built with the quantization config NeMo RL hands
    # TRT-LLM (no checkpoint, no engine); checks which refit weights are FP8.
    pytest.importorskip("transformers")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM

    if is_mx and kind == "moe" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("TRT-LLM MXFP8 MoE needs SM100+")
    quantization_config = build_quant_config(is_mx, ["*lm_head*"])
    (tmp_path / "config.json").write_text(
        json.dumps(_config(kind, quantization_config))
    )

    model_config = ModelConfig.from_pretrained(str(tmp_path))
    model = AutoModelForCausalLM.from_config(model_config)
    info = _hf_state_dict_info(tmp_path)

    quantized = resolve_quantized_weights(model, model_config.quant_config, info)

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
