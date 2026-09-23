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
    build_local_expert_lookup,
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


def _fused_expert_stacks(num_experts: int):
    gate = torch.stack([_block_matrix([[float(2 * e + 1), 2.0]]) for e in range(num_experts)])
    up = torch.stack([_block_matrix([[5.0, float(e + 1)]]) for e in range(num_experts)])
    down = torch.stack(
        [_block_matrix([[9.0], [float(10 + e)]]) for e in range(num_experts)]
    )
    gate_up = torch.cat((gate, up), dim=1).to(torch.bfloat16)
    return gate, up, down.to(torch.bfloat16), gate_up


@pytest.mark.parametrize("is_mx", [False, True])
def test_local_expert_lookup_converts_only_local_experts(is_mx):
    prefix = "model.language_model.layers.3.mlp.experts"
    gate, up, down, gate_up = _fused_expert_stacks(4)
    local_ids = [3, 1]

    full = load_weights(
        [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)],
        is_mx=is_mx,
    )
    local = load_weights(
        [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)],
        is_mx=is_mx,
        local_experts=lambda p: local_ids if p == prefix else None,
    )

    expected_names = {
        f"{prefix}.{e}.{proj}.{leaf}"
        for e in local_ids
        for proj in ("gate_proj", "up_proj", "down_proj")
        for leaf in ("weight", "weight_scale_inv")
    }
    assert set(local) == expected_names
    for name, tensor in local.items():
        assert torch.equal(tensor.float(), full[name].float())
        assert tensor.dtype == full[name].dtype
    assert f"{prefix}.0.gate_proj.weight" not in local
    assert f"{prefix}.2.down_proj.weight" not in local


def test_local_expert_lookup_unknown_prefix_converts_everything():
    prefix = "model.layers.0.mlp.experts"
    _gate, _up, down, gate_up = _fused_expert_stacks(3)
    converted = load_weights(
        [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)],
        local_experts=lambda p: None,
    )
    assert {f"{prefix}.{e}.down_proj.weight" for e in range(3)} <= set(converted)


def test_local_expert_lookup_filters_split_expert_names():
    prefix = "mtp.layers.0.mlp.experts"
    weights = [
        (f"{prefix}.{e}.down_proj.weight", torch.randn(128, 128, dtype=torch.bfloat16))
        for e in range(3)
    ]
    passthrough = ("model.layers.0.mlp.gate.weight", torch.randn(2, 256))
    converted = load_weights(
        weights + [passthrough],
        local_experts=lambda p: [2] if p == prefix else None,
    )
    assert f"{prefix}.2.down_proj.weight" in converted
    assert f"{prefix}.2.down_proj.weight_scale_inv" in converted
    assert f"{prefix}.0.down_proj.weight" not in converted
    assert f"{prefix}.1.down_proj.weight" not in converted
    assert converted[passthrough[0]] is passthrough[1]


def test_local_expert_lookup_rejects_out_of_range_ids():
    prefix = "model.layers.0.mlp.experts"
    _gate, _up, down, gate_up = _fused_expert_stacks(2)
    with pytest.raises(ValueError, match="outside"):
        load_weights(
            [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)],
            local_experts=lambda p: [0, 5],
        )


class _MoeStub(torch.nn.Module):
    def __init__(self, layer_idx, local_ids, load_balancer=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.initial_local_expert_ids = list(local_ids)
        self.layer_load_balancer = load_balancer


class _ModelStub(torch.nn.Module):
    def __init__(self, layers, num_hidden_layers=None):
        super().__init__()
        self.blocks = torch.nn.ModuleList(layers)
        pretrained = type("Pretrained", (), {"num_hidden_layers": num_hidden_layers})()
        self.model_config = type("ModelConfig", (), {"pretrained_config": pretrained})()


def test_build_local_expert_lookup_maps_decoder_and_mtp_layers():
    # The owner module and its backend both carry the attributes (same ids).
    owner = _MoeStub(3, [32, 33, 34, 35])
    owner.backend = _MoeStub(3, [35, 34, 33, 32])
    mtp = _MoeStub(60, [0, 1])
    lookup = build_local_expert_lookup(_ModelStub([owner, mtp], num_hidden_layers=60))

    assert lookup is not None
    assert list(lookup("model.language_model.layers.3.mlp.experts")) == [32, 33, 34, 35]
    assert list(lookup("model.layers.3.mlp.experts")) == [32, 33, 34, 35]
    assert list(lookup("mtp.layers.0.mlp.experts")) == [0, 1]
    assert lookup("model.layers.7.mlp.experts") is None
    assert lookup("model.layers.3.mlp.shared_expert") is None


def test_build_local_expert_lookup_without_moe_or_with_load_balancer():
    assert build_local_expert_lookup(_ModelStub([torch.nn.Linear(2, 2)])) is None
    balanced = _MoeStub(0, [0, 1], load_balancer=object())
    assert build_local_expert_lookup(_ModelStub([balanced])) is None


def test_build_local_expert_lookup_rejects_disagreeing_modules():
    owner = _MoeStub(1, [0, 1])
    owner.backend = _MoeStub(1, [2, 3])
    with pytest.raises(ValueError, match="disagree"):
        build_local_expert_lookup(_ModelStub([owner]))
