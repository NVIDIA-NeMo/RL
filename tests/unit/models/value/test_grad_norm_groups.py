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
"""critic/gnorm/* helper (single process: the model-parallel reduce is a no-op)."""

from __future__ import annotations

import math

import pytest
import torch

from nemo_rl.models.value.grad_norm_groups import (
    GRAD_NORM_GROUP_ORDER,
    grad_norm_group_of,
    hybrid_pattern_of,
    pre_clip_grad_norms_by_group,
)


class _Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _Mixer()


class _Decoder(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(n)])


class _Model(torch.nn.Module):
    """decoder.layers.N.mixer.weight + embedding + value head, pattern 'M*E'."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 2)
        self.decoder = _Decoder(3)
        self.output_layer = torch.nn.Linear(2, 1, bias=False)
        self.config = type("Cfg", (), {"hybrid_layer_pattern": "M*E"})()


def test_grad_norm_group_of_uses_pattern_then_name_hints():
    assert grad_norm_group_of("output_layer.weight", "M*E") == "value_head"
    assert grad_norm_group_of("embedding.word_embeddings.weight", "M") == "embedding"
    assert grad_norm_group_of("decoder.layers.0.mixer.in_proj.weight", "M*E") == "mamba"
    assert grad_norm_group_of("decoder.layers.1.mixer.linear_qkv.weight", "M*E") == (
        "attention"
    )
    assert grad_norm_group_of("decoder.layers.2.mixer.x", "M*E") == "moe"
    assert grad_norm_group_of("decoder.layers.0.mixer.x", "-") == "mlp"
    # No pattern: Nemotron-H submodule names.
    assert grad_norm_group_of("decoder.layers.9.mixer.experts.fc1", None) == "moe"
    assert grad_norm_group_of("decoder.final_norm.weight", None) == "other"
    assert set(GRAD_NORM_GROUP_ORDER) >= {"attention", "mamba", "moe", "other"}


def test_hybrid_pattern_lookup_tolerates_wrappers():
    model = _Model()
    assert hybrid_pattern_of(model) == "M*E"
    wrapped = type("DDP", (), {"module": model})()
    assert hybrid_pattern_of([wrapped]) == "M*E"
    assert hybrid_pattern_of(torch.nn.Linear(1, 1)) is None


def test_pre_clip_grad_norms_by_group_prefers_main_grad():
    model = _Model()
    model.decoder.layers[0].mixer.weight.grad = torch.tensor([3.0, 4.0])
    model.decoder.layers[1].mixer.weight.grad = torch.tensor([1.0, 0.0])
    # mcore DDP accumulates into main_grad; it wins over .grad.
    model.decoder.layers[2].mixer.weight.main_grad = torch.tensor([0.0, 2.0])
    model.decoder.layers[2].mixer.weight.grad = torch.tensor([100.0, 0.0])
    model.output_layer.weight.grad = torch.tensor([[6.0, 8.0]])
    norms = pre_clip_grad_norms_by_group(model, hybrid_pattern_of(model), None)
    assert float(norms["mamba"]) == pytest.approx(5.0)
    assert float(norms["attention"]) == pytest.approx(1.0)
    assert float(norms["moe"]) == pytest.approx(2.0)
    assert float(norms["value_head"]) == pytest.approx(10.0)
    # Groups with no gradient are dropped, not reported as 0.
    assert "embedding" not in norms and "other" not in norms
    total = math.sqrt(sum(float(v) ** 2 for v in norms.values()))
    assert total == pytest.approx(math.sqrt(25 + 1 + 4 + 100))
