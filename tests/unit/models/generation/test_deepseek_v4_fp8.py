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

import types

import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.fixture
def deepseek_v4_fp8():
    from nemo_rl.models.generation.vllm.quantization import deepseek_v4_fp8

    return deepseek_v4_fp8


@pytest.fixture
def skip_tensors():
    """Guard the process-global reload skip list that prepare_refit mutates."""
    from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS

    original = set(SKIP_TENSORS)
    try:
        yield SKIP_TENSORS
    finally:
        SKIP_TENSORS.clear()
        SKIP_TENSORS.update(original)


class DeepSeekV4ForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="deepseek_v4")


class OtherForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="deepseek_v3")


def test_is_model_detects_config_model_type(deepseek_v4_fp8):
    model = OtherForCausalLM()
    model.config = types.SimpleNamespace(model_type="deepseek_v4")

    assert deepseek_v4_fp8.is_model(model) is True


def test_is_model_rejects_other_architectures(deepseek_v4_fp8):
    model = OtherForCausalLM()
    model.config = types.SimpleNamespace(model_type="deepseek_v3")

    assert deepseek_v4_fp8.is_model(model) is False


def test_map_checkpoint_name_passes_through_other_architectures(deepseek_v4_fp8):
    model = OtherForCausalLM()
    model.hf_to_vllm_mapper = types.SimpleNamespace(apply_list=lambda _names: [])

    assert (
        deepseek_v4_fp8.map_checkpoint_name(model, "layers.0.attn.wq_a.weight")
        == "layers.0.attn.wq_a.weight"
    )


def test_map_checkpoint_name_applies_the_vllm_mapper(deepseek_v4_fp8):
    model = DeepSeekV4ForCausalLM()
    model.hf_to_vllm_mapper = types.SimpleNamespace(
        apply_list=lambda names: [f"model.{names[0]}"]
    )

    assert (
        deepseek_v4_fp8.map_checkpoint_name(model, "layers.0.attn.wq_a.weight")
        == "model.layers.0.attn.wq_a.weight"
    )


@pytest.mark.parametrize(
    ("module_path", "expected"),
    [
        (
            ["model", "layers", "0", "attn", "wq_a"],
            ["model", "layers", "0", "attn", "fused_wqa_wkv"],
        ),
        (
            ["model", "layers", "0", "attn", "wkv"],
            ["model", "layers", "0", "attn", "fused_wqa_wkv"],
        ),
        (
            ["model", "layers", "0", "compressor", "wgate"],
            ["model", "layers", "0", "compressor", "fused_wkv_wgate"],
        ),
        (
            ["model", "layers", "0", "ffn", "shared_experts", "w3"],
            ["model", "layers", "0", "ffn", "shared_experts", "gate_up_proj"],
        ),
        (
            ["model", "layers", "0", "attn", "o_proj"],
            ["model", "layers", "0", "attn", "o_proj"],
        ),
    ],
)
def test_remap_packed_module_path(deepseek_v4_fp8, module_path, expected):
    assert (
        deepseek_v4_fp8.remap_packed_module_path(DeepSeekV4ForCausalLM(), module_path)
        == expected
    )


def test_remap_packed_module_path_leaves_other_architectures_alone(deepseek_v4_fp8):
    module_path = ["model", "layers", "0", "attn", "wq_a"]

    assert (
        deepseek_v4_fp8.remap_packed_module_path(OtherForCausalLM(), list(module_path))
        == module_path
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.3.ffn.experts.17.w1.weight", 17),
        ("ffn.experts.0.w2.weight", 0),
        ("model.layers.3.ffn.experts.17.w1.weight_scale_inv", None),
        ("model.layers.3.ffn.shared_experts.w1.weight", None),
        ("model.layers.3.ffn.gate.weight", None),
    ],
)
def test_get_exported_expert_id(deepseek_v4_fp8, name, expected):
    assert deepseek_v4_fp8.get_exported_expert_id(DeepSeekV4ForCausalLM(), name) == (
        expected
    )


def test_get_exported_expert_id_is_none_for_other_architectures(deepseek_v4_fp8):
    assert (
        deepseek_v4_fp8.get_exported_expert_id(
            OtherForCausalLM(), "model.layers.3.ffn.experts.17.w1.weight"
        )
        is None
    )


def test_is_nonlocal_expert(deepseek_v4_fp8, monkeypatch):
    class FakeRoutedExperts(torch.nn.Module):
        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 4 else -1

    monkeypatch.setattr(deepseek_v4_fp8, "RoutedExperts", FakeRoutedExperts)
    experts = FakeRoutedExperts()

    assert deepseek_v4_fp8.is_nonlocal_expert(experts, 4) is False
    assert deepseek_v4_fp8.is_nonlocal_expert(experts, 5) is True


class FakeRoutedExpertsLayer(torch.nn.Module):
    """Routed-expert layer whose tensors start in the post-load kernel layout."""

    def __init__(self, weight_block_size=None, process_calls=None):
        super().__init__()
        self.num_experts = 2
        self.hidden_size = 4
        self.intermediate_size_per_partition = 2
        self.weight_block_size = weight_block_size
        # Kernel layout: transposed and bf16, i.e. not directly loadable.
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 4, 4, dtype=torch.bfloat16), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(2, 2, 4, dtype=torch.bfloat16), requires_grad=False
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.zeros(2, 1, 1), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.zeros(2, 1, 1), requires_grad=False
        )
        self.quant_method = types.SimpleNamespace(
            process_weights_after_loading=lambda layer: (
                process_calls.append(layer) if process_calls is not None else None
            )
        )


def test_prepare_refit_restores_raw_expert_shapes_and_dtypes(
    deepseek_v4_fp8, monkeypatch, skip_tensors
):
    monkeypatch.setattr(deepseek_v4_fp8, "RoutedExperts", FakeRoutedExpertsLayer)
    layer = FakeRoutedExpertsLayer(weight_block_size=[2, 2])
    model = torch.nn.Sequential(layer)

    added_skip_tensors = deepseek_v4_fp8.prepare_refit(model)

    assert layer.w13_weight.shape == (2, 4, 4)
    assert layer.w13_weight.dtype == torch.float8_e4m3fn
    assert layer.w2_weight.shape == (2, 4, 2)
    assert layer.w2_weight.dtype == torch.float8_e4m3fn
    # ceil(dim / block) per weight dimension.
    assert layer.w13_weight_scale_inv.shape == (2, 2, 2)
    assert layer.w2_weight_scale_inv.shape == (2, 2, 1)
    assert layer.w13_weight_scale_inv.dtype == torch.float32
    deepseek_v4_fp8.restore_refit(added_skip_tensors)


def test_prepare_refit_marks_expert_and_sink_tensors_for_immediate_load(
    deepseek_v4_fp8, monkeypatch, skip_tensors
):
    monkeypatch.setattr(deepseek_v4_fp8, "RoutedExperts", FakeRoutedExpertsLayer)
    model = torch.nn.Sequential(FakeRoutedExpertsLayer(weight_block_size=[2, 2]))

    added_skip_tensors = deepseek_v4_fp8.prepare_refit(model)

    assert "attn_sink" in skip_tensors
    assert {"w13_weight", "w2_weight", "w13_weight_scale_inv"} <= skip_tensors

    deepseek_v4_fp8.restore_refit(added_skip_tensors)
    assert "attn_sink" not in skip_tensors
    assert "w13_weight" not in skip_tensors


def test_restore_refit_preserves_preexisting_global_skip_names(
    deepseek_v4_fp8, monkeypatch, skip_tensors
):
    monkeypatch.setattr(deepseek_v4_fp8, "RoutedExperts", FakeRoutedExpertsLayer)
    baseline = set(skip_tensors)
    skip_tensors.update({"attn_sink", "preexisting"})
    model = torch.nn.Sequential(FakeRoutedExpertsLayer(weight_block_size=[2, 2]))

    added_skip_tensors = deepseek_v4_fp8.prepare_refit(model)
    deepseek_v4_fp8.restore_refit(added_skip_tensors)

    assert skip_tensors == baseline | {"attn_sink", "preexisting"}


def test_finalize_refit_requantizes_block_scaled_experts(deepseek_v4_fp8, monkeypatch):
    monkeypatch.setattr(deepseek_v4_fp8, "RoutedExperts", FakeRoutedExpertsLayer)
    process_calls = []
    quantized = FakeRoutedExpertsLayer(
        weight_block_size=[2, 2], process_calls=process_calls
    )
    model = torch.nn.Sequential(quantized)

    deepseek_v4_fp8.finalize_refit(model)

    assert process_calls == [quantized]
