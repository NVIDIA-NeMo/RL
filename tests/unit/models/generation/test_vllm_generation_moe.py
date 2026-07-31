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

import torch

from nemo_rl.models.generation.vllm.quantization import fp8


class _Container(torch.nn.Module):
    pass


class _FakeRoutedExperts(torch.nn.Module):
    pass


def test_qwen35_fp8_refit_resolves_wrapped_expert(monkeypatch):
    expert = _FakeRoutedExperts()
    layer = _Container()
    layer.mlp = _Container()
    layer.mlp.experts = expert

    model = _Container()
    model.packed_modules_mapping = {}
    model.language_model = _Container()
    model.language_model.model = _Container()
    model.language_model.model.layers = torch.nn.ModuleList([layer])

    monkeypatch.setattr(fp8, "RoutedExperts", _FakeRoutedExperts)

    module = fp8._get_module_from_param_name(
        model,
        "model.language_model.layers.0.mlp.experts.7.gate_proj.weight",
    )

    assert module is expert


def test_qwen35_fp8_refit_expands_grouped_experts(monkeypatch):
    def fake_quantize(grouped_experts):
        scales_shape = (grouped_experts.shape[0], 1, 1)
        return grouped_experts, torch.ones(scales_shape)

    monkeypatch.setattr(fp8, "_quantize_grouped_experts_blockwise", fake_quantize)
    grouped = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    entries = fp8._expand_grouped_moe_expert_to_fp8(
        "model.language_model.layers.0.mlp.experts.gate_up_proj", grouped
    )

    assert [name for name, _ in entries] == [
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.language_model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
        "model.language_model.layers.0.mlp.experts.1.up_proj.weight",
        "model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale_inv",
    ]


def test_qwen35_fp8_refit_quantizes_grouped_experts_in_bounded_chunks(
    monkeypatch,
):
    quantized_shapes = []

    def fake_cast(data, weight_block_size):
        quantized_shapes.append(tuple(data.shape))
        block0, block1 = weight_block_size
        scales = torch.ones(
            data.shape[0] // block0, data.shape[1] // block1, 1
        )
        return torch.empty_like(data, dtype=torch.float8_e4m3fn), scales

    monkeypatch.setattr(fp8, "cast_tensor_to_fp8_blockwise", fake_cast)
    grouped = torch.zeros(33, 128, 128, dtype=torch.bfloat16)

    weight_fp8, scale_inv = fp8._quantize_grouped_experts_blockwise(grouped)

    assert quantized_shapes == [(32 * 128, 128), (128, 128)]
    assert weight_fp8.shape == grouped.shape
    assert scale_inv.shape == (33, 1, 1)
