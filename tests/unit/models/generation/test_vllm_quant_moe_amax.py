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
import pytest
import torch

from nemo_rl.modelopt.models.generation.vllm_quant_moe_amax import (
    route_moe_input_quantizer_amax,
)


class _Quantizer(torch.nn.Module):
    def __init__(self, amax: float) -> None:
        super().__init__()
        self.register_buffer("_amax", torch.tensor(amax))


class _FakeRoutedExperts(torch.nn.Module):
    """Mimics vLLM 0.28's RoutedExperts naming contract for a non-gated MoE.

    ``layer_name`` is the vLLM module path, ``get_expert_mapping`` yields
    ``(param_name, weight_name, expert_id, shard_id)`` with the same string
    shapes vLLM builds: ``experts.<id>.<proj>.`` -> ``experts.w13_`` / ``w2_``.
    """

    def __init__(self, layer_name: str, num_experts: int) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.num_experts = num_experts
        self.w13_input_quantizer = _Quantizer(1.0)
        self.w2_input_quantizer = _Quantizer(1.0)
        self.w13_weight = torch.nn.Parameter(torch.zeros(num_experts, 4, 2))

    def get_expert_mapping(self, include_fused: bool = True):
        mapping = []
        for expert_id in range(self.num_experts):
            for shard_id, proj in (("w1", "up_proj"), ("w2", "down_proj")):
                param = "experts.w13_" if shard_id == "w1" else "experts.w2_"
                mapping.append(
                    (param, f"experts.{expert_id}.{proj}.", expert_id, shard_id)
                )
        return mapping


class _Mixer(torch.nn.Module):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.experts = _FakeRoutedExperts(f"{prefix}.experts", num_experts=3)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Mixer("model.layers.0")])
        # A dense quantizer that must be left alone for vLLM's own loader.
        self.dense_input_quantizer = _Quantizer(1.0)

    # vLLM names the top-level module `model`, so layer_name carries that prefix.


def test_expert_amax_fans_in_with_max_and_is_removed_from_the_stream():
    model = _Model()
    experts = model.layers[0].experts
    weights = [
        ("model.layers.0.experts.0.up_proj.input_quantizer._amax", torch.tensor(3.0)),
        ("model.layers.0.experts.2.up_proj.input_quantizer._amax", torch.tensor(7.0)),
        ("model.layers.0.experts.1.down_proj.input_quantizer._amax", torch.tensor(0.5)),
        ("model.layers.0.experts.w13_weight", torch.zeros(3, 4, 2)),
        ("model.dense.input_quantizer._amax", torch.tensor(9.0)),
    ]

    remaining = route_moe_input_quantizer_amax(model, weights)

    assert experts.w13_input_quantizer._amax.item() == 7.0
    # 0.5 < initial 1.0: max keeps the larger existing value.
    assert experts.w2_input_quantizer._amax.item() == 1.0
    assert [name for name, _ in remaining] == [
        "model.layers.0.experts.w13_weight",
        "model.dense.input_quantizer._amax",
    ]


def test_missing_quantizer_path_raises_with_resolved_name():
    model = _Model()
    del model.layers[0].experts.w2_input_quantizer
    weights = [
        ("model.layers.0.experts.1.down_proj.input_quantizer._amax", torch.tensor(2.0)),
    ]
    with pytest.raises(KeyError, match="w2_input_quantizer._amax"):
        route_moe_input_quantizer_amax(model, weights)


def test_models_without_expert_modules_pass_everything_through():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    weights = [("0.input_quantizer._amax", torch.tensor(1.0))]
    assert route_moe_input_quantizer_amax(model, weights) == weights


class _BackboneMapper:
    """Stands in for vLLM's WeightsMapper: Nemotron-H maps `backbone.` -> `model.`."""

    def _map_name(self, key: str) -> str | None:
        if key.startswith("mtp."):
            return None
        return key.replace("backbone.", "model.", 1)


def test_checkpoint_names_are_mapped_to_vllm_names_before_matching():
    """Nemotron-H exports `backbone.layers.*`; vLLM's layer_name is `model.layers.*`.

    Without the mapper the amax passed straight through to vLLM's loader, which
    then failed with `Layer model.layers.N.mixer.experts has no parameter
    'w13_input_quantizer._amax'` on the nano3 fakequant nightlies.
    """
    model = _Model()
    experts = model.layers[0].experts
    weights = [
        (
            "backbone.layers.0.experts.2.up_proj.input_quantizer._amax",
            torch.tensor(5.0),
        ),
        (
            "backbone.layers.0.experts.1.down_proj.input_quantizer._amax",
            torch.tensor(4.0),
        ),
        ("mtp.layers.0.experts.0.up_proj.input_quantizer._amax", torch.tensor(9.0)),
        ("backbone.layers.0.experts.w13_weight", torch.zeros(3, 4, 2)),
    ]

    remaining = route_moe_input_quantizer_amax(model, weights, mapper=_BackboneMapper())

    assert experts.w13_input_quantizer._amax.item() == 5.0
    assert experts.w2_input_quantizer._amax.item() == 4.0
    # Dropped-by-mapper names and non-amax weights are left for vLLM's loader.
    assert [name for name, _ in remaining] == [
        "mtp.layers.0.experts.0.up_proj.input_quantizer._amax",
        "backbone.layers.0.experts.w13_weight",
    ]


def test_unmapped_checkpoint_prefix_passes_through_without_a_mapper():
    """Documents the pass-3/5 failure mode: no mapper, no match, vLLM gets the name."""
    model = _Model()
    weights = [
        (
            "backbone.layers.0.experts.2.up_proj.input_quantizer._amax",
            torch.tensor(5.0),
        ),
    ]
    assert route_moe_input_quantizer_amax(model, weights) == weights
    assert model.layers[0].experts.w13_input_quantizer._amax.item() == 1.0
