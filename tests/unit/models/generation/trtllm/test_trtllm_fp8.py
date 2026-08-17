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

import pytest
import torch

from nemo_rl.models.generation.trtllm.quantization.fp8 import (
    FP8_BLOCK_QUANT_KWARGS,
    cast_tensor_to_fp8_blockwise,
    configure_fp8_llm_kwargs,
    configure_fp8_moe_backend,
    load_weights,
    validate_fused_expert_layout,
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

    configure_fp8_llm_kwargs(llm_kwargs, model_type="qwen3_5_moe")

    assert llm_kwargs["dtype"] == "bfloat16"
    assert llm_kwargs["load_format"] == "dummy"
    assert llm_kwargs["use_cute_dsl_blockscaling_mm"] is True
    assert llm_kwargs["model_kwargs"]["pretrained_config"] == {"num_hidden_layers": 4}
    assert llm_kwargs["model_kwargs"]["quantization_config"] == FP8_BLOCK_QUANT_KWARGS


@pytest.mark.parametrize(
    ("llm_kwargs", "model_type"),
    [
        ({"load_format": "auto"}, "qwen3_5_moe"),
        (
            {"model_kwargs": {"quantization_config": {"quant_method": "modelopt"}}},
            "qwen3_5_moe",
        ),
        ({}, "qwen3_moe"),
    ],
)
def test_configure_fp8_llm_kwargs_rejects_unsupported_contract(llm_kwargs, model_type):
    with pytest.raises(ValueError, match="precision='fp8'"):
        configure_fp8_llm_kwargs(llm_kwargs, model_type=model_type)


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
