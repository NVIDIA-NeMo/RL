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

"""Refit-time handling of LoRA on Automodel's grouped MoE experts.

Grouped-expert LoRA does not follow the LinearLoRA layout: the adapters are bare parameters
on the expert module (`lora_gate_and_up_A`, ...) next to fused base weights
(`gate_and_up_projs`, `down_projs`). These tests pin down that such adapters are merged into
the base weights before refit, and are not themselves streamed to the inference engine.
"""

import pytest
import torch
import torch.nn as nn

from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    _grouped_expert_lora_adapters,
    _is_lora_adapter_tensor,
    _maybe_merge_lora_weight,
)

N_EXPERTS = 3
DIM = 8
UP_DIM = 12
RANK = 2
ALPHA = 8


class StubGroupedExpertsLoRA(nn.Module):
    """Minimal stand-in for GroupedExperts(DeepEP)LoRA's parameter layout."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.gate_and_up_projs = nn.Parameter(
            torch.randn(N_EXPERTS, DIM, UP_DIM, dtype=dtype)
        )
        self.down_projs = nn.Parameter(torch.randn(N_EXPERTS, UP_DIM, DIM, dtype=dtype))
        self.lora_gate_and_up_A = nn.Parameter(
            torch.randn(N_EXPERTS, DIM, RANK, dtype=dtype)
        )
        self.lora_gate_and_up_B = nn.Parameter(
            torch.randn(N_EXPERTS, RANK, UP_DIM, dtype=dtype)
        )
        self.lora_down_A = nn.Parameter(
            torch.randn(N_EXPERTS, UP_DIM, RANK, dtype=dtype)
        )
        self.lora_down_B = nn.Parameter(torch.randn(N_EXPERTS, RANK, DIM, dtype=dtype))
        self.scale = ALPHA / RANK


@pytest.fixture
def module_map():
    module = StubGroupedExpertsLoRA()
    return {"model.layers.0.moe.experts": module}, module


@pytest.mark.parametrize(
    "fqn,expected",
    [
        ("model.layers.0.self_attn.q_proj.lora_A.weight", True),
        ("model.layers.0.self_attn.q_proj.lora_B.weight", True),
        ("model.layers.0.moe.experts.lora_gate_and_up_A", True),
        ("model.layers.0.moe.experts.lora_gate_and_up_B", True),
        ("model.layers.0.moe.experts.lora_down_A", True),
        ("model.layers.0.moe.experts.lora_down_B", True),
        ("model.layers.0.moe.experts.gate_and_up_projs", False),
        ("model.layers.0.moe.experts.down_projs", False),
        ("model.layers.0.self_attn.q_proj.weight", False),
    ],
)
def test_is_lora_adapter_tensor(fqn, expected):
    """Adapter tensors of both layouts are recognised; base weights are not."""
    assert _is_lora_adapter_tensor(fqn) is expected


def test_grouped_expert_adapters_are_resolved(module_map):
    mapping, module = module_map
    for leaf, expected in (
        ("gate_and_up_projs", (module.lora_gate_and_up_A, module.lora_gate_and_up_B)),
        ("down_projs", (module.lora_down_A, module.lora_down_B)),
    ):
        found_module, adapters = _grouped_expert_lora_adapters(
            mapping, f"model.layers.0.moe.experts.{leaf}"
        )
        assert found_module is module
        assert adapters == expected


def test_non_expert_fqn_resolves_to_nothing(module_map):
    mapping, _ = module_map
    assert _grouped_expert_lora_adapters(
        mapping, "model.layers.0.moe.experts.something"
    ) == (
        None,
        None,
    )
    assert _grouped_expert_lora_adapters(mapping, "model.lm_head.weight") == (
        None,
        None,
    )


@pytest.mark.parametrize(
    "leaf,a_attr,b_attr",
    [
        ("gate_and_up_projs", "lora_gate_and_up_A", "lora_gate_and_up_B"),
        ("down_projs", "lora_down_A", "lora_down_B"),
    ],
)
def test_merge_adds_per_expert_delta(module_map, leaf, a_attr, b_attr):
    """The merge must be per-expert `W + (A @ B) * scale`, in the base weight's orientation."""
    mapping, module = module_map
    base = getattr(module, leaf).detach()
    merged = _maybe_merge_lora_weight(
        mapping, f"model.layers.0.moe.experts.{leaf}", base
    )

    expected = (
        base
        + torch.bmm(getattr(module, a_attr), getattr(module, b_attr)) * module.scale
    )
    assert merged.shape == base.shape
    torch.testing.assert_close(merged, expected)
    # Each expert must get its own delta, not a shared one.
    for expert in range(N_EXPERTS):
        torch.testing.assert_close(merged[expert], expected[expert])


def test_merge_is_out_of_place(module_map):
    """The refit generator may hand over a parameter's own storage; it must not be mutated."""
    mapping, module = module_map
    base = module.gate_and_up_projs.detach()
    before = base.clone()
    _maybe_merge_lora_weight(
        mapping, "model.layers.0.moe.experts.gate_and_up_projs", base
    )
    torch.testing.assert_close(base, before)


def test_zero_initialised_b_is_a_no_op(module_map):
    """LoRA B starts at zero, so a freshly adapted model must refit unchanged."""
    mapping, module = module_map
    with torch.no_grad():
        module.lora_gate_and_up_B.zero_()
    base = module.gate_and_up_projs.detach()
    merged = _maybe_merge_lora_weight(
        mapping, "model.layers.0.moe.experts.gate_and_up_projs", base
    )
    torch.testing.assert_close(merged, base)


def test_module_without_adapters_passes_through():
    """A plain grouped-expert module (no LoRA) must be returned untouched."""

    class PlainExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_and_up_projs = nn.Parameter(torch.randn(N_EXPERTS, DIM, UP_DIM))

    module = PlainExperts()
    mapping = {"model.layers.0.moe.experts": module}
    base = module.gate_and_up_projs.detach()
    merged = _maybe_merge_lora_weight(
        mapping, "model.layers.0.moe.experts.gate_and_up_projs", base
    )
    assert merged is base


def test_merge_preserves_dtype(module_map):
    """Adapters may be held in a different dtype than the base weights (fp32 masters)."""
    mapping, module = module_map
    with torch.no_grad():
        module.gate_and_up_projs.data = module.gate_and_up_projs.data.to(torch.bfloat16)
    base = module.gate_and_up_projs.detach()
    merged = _maybe_merge_lora_weight(
        mapping, "model.layers.0.moe.experts.gate_and_up_projs", base
    )
    assert merged.dtype == torch.bfloat16
