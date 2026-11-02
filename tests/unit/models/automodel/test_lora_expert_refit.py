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

from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("nemo_automodel")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from nemo_automodel.components._peft.lora_experts import (
    GroupedExpertsDeepEPLoRA,
    GroupedExpertsLoRA,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
    _assert_lora_modules_reachable,
    _grouped_expert_lora_adapters,
    _is_lora_adapter_tensor,
    _maybe_merge_lora_weight,
    automodel_params_generator,
)

pytestmark = pytest.mark.automodel

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


@pytest.mark.parametrize("leaf", ["gate_and_up_projs", "down_projs"])
@pytest.mark.parametrize("adapter_dtype", [torch.float32, torch.bfloat16])
def test_bf16_merge_matches_offline_fp32_merge(
    leaf: str, adapter_dtype: torch.dtype
) -> None:
    torch.manual_seed(1)
    module = StubGroupedExpertsLoRA(dtype=adapter_dtype)
    base = getattr(module, leaf).detach().to(torch.bfloat16)
    prefix = "lora_gate_and_up" if leaf == "gate_and_up_projs" else "lora_down"
    a = getattr(module, f"{prefix}_A").detach().float()
    b = getattr(module, f"{prefix}_B").detach().float()
    expected = (base.float() + torch.bmm(a, b) * module.scale).to(base.dtype)

    merged = _maybe_merge_lora_weight({"experts": module}, f"experts.{leaf}", base)

    assert torch.equal(merged, expected)
    assert not merged.requires_grad


@pytest.mark.parametrize("checkpointed", [False, True])
def test_generator_merges_experts_and_filters_adapters(checkpointed: bool) -> None:
    block = nn.Module()
    block.experts = StubGroupedExpertsLoRA()
    model = nn.Module()
    model.layers = nn.ModuleList([checkpoint_wrapper(block) if checkpointed else block])
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    results = dict(automodel_params_generator(model, torch.float32))
    expected_names = {
        "layers.0.experts.gate_and_up_projs",
        "layers.0.experts.down_projs",
    }
    assert set(results) == expected_names
    for leaf, a, b in (
        ("gate_and_up_projs", "lora_gate_and_up_A", "lora_gate_and_up_B"),
        ("down_projs", "lora_down_A", "lora_down_B"),
    ):
        prefix = "layers.0.experts."
        expected = before[prefix + leaf] + block.experts.scale * torch.bmm(
            before[prefix + a], before[prefix + b]
        )
        torch.testing.assert_close(results[prefix + leaf], expected)
    # Repeated refits must not accumulate the update in training weights.
    again = dict(automodel_params_generator(model, torch.float32))
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    for name in results:
        torch.testing.assert_close(again[name], results[name], rtol=0, atol=0)


def test_refit_metadata_excludes_grouped_adapters() -> None:
    model = nn.Module()
    model.experts = StubGroupedExpertsLoRA()
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    worker.model = model
    worker.dtype = torch.bfloat16
    worker.lora_enabled = True
    worker.lora_refit_mode = "merged"
    worker.draft_model = None

    metadata = worker.prepare_refit_info()
    tensors = dict(automodel_params_generator(model, worker.dtype))

    assert set(metadata) == {"experts.gate_and_up_projs", "experts.down_projs"}
    assert metadata == {
        name: (tensor.shape, tensor.dtype) for name, tensor in tensors.items()
    }


@pytest.mark.parametrize("missing", ["gate_and_up_projs", "down_projs"])
def test_guard_rejects_missing_expert_base_weight(missing: str) -> None:
    model = nn.Module()
    model.experts = StubGroupedExpertsLoRA()
    state_dict = model.state_dict()
    del state_dict[f"experts.{missing}"]
    with pytest.raises(RuntimeError, match=f"experts.{missing}"):
        _assert_lora_modules_reachable(dict(model.named_modules()), state_dict)


def test_generator_rejects_unrecognized_wrapper() -> None:
    class RenamingWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = StubGroupedExpertsLoRA()
            self._register_state_dict_hook(self._strip_inner)

        @staticmethod
        def _strip_inner(module, state_dict, prefix, local_metadata) -> None:
            inner_prefix = f"{prefix}inner."
            for key in [k for k in state_dict if k.startswith(inner_prefix)]:
                state_dict[prefix + key[len(inner_prefix) :]] = state_dict.pop(key)

    model = nn.Module()
    model.experts = RenamingWrapper()
    with pytest.raises(RuntimeError, match="experts.inner"):
        next(automodel_params_generator(model, torch.float32))


def test_root_grouped_experts_are_reachable() -> None:
    model = StubGroupedExpertsLoRA()
    results = dict(automodel_params_generator(model, torch.float32))
    assert set(results) == {"gate_and_up_projs", "down_projs"}
    expected = model.down_projs + model.scale * torch.bmm(
        model.lora_down_A, model.lora_down_B
    )
    torch.testing.assert_close(results["down_projs"], expected)


@pytest.mark.parametrize("deep_ep", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "relu2"])
def test_pinned_automodel_expert_layout(deep_ep: bool, activation: str) -> None:
    """Exercise the pinned Automodel's actual gated and non-gated parameter layouts."""
    config = MoEConfig(
        n_routed_experts=N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=DIM,
        inter_dim=UP_DIM,
        moe_inter_dim=UP_DIM,
        norm_topk_prob=True,
        expert_activation=activation,
        dtype=torch.float32,
    )
    if deep_ep:
        experts = GroupedExpertsDeepEPLoRA(
            GroupedExpertsDeepEP(config), lora_dim=RANK, alpha=ALPHA
        )
    else:
        experts = GroupedExpertsLoRA(GroupedExperts(config), lora_dim=RANK, alpha=ALPHA)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.normal_()
    model = nn.Module()
    model.experts = experts

    results = dict(automodel_params_generator(model, torch.float32))

    assert set(results) == {"experts.gate_and_up_projs", "experts.down_projs"}
    for leaf, a, b in (
        ("gate_and_up_projs", experts.lora_gate_and_up_A, experts.lora_gate_and_up_B),
        ("down_projs", experts.lora_down_A, experts.lora_down_B),
    ):
        base = getattr(experts, leaf)
        # Compare each exported expert's linear action with the unmerged adapter.
        x = torch.randn(N_EXPERTS, 2, base.shape[1])
        expected = x @ base + ((x @ a) @ b) * experts.scale
        torch.testing.assert_close(x @ results[f"experts.{leaf}"], expected)


def _check_expert_parallel_refit(rank: int, init_method: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(7)
        model = nn.Module()
        model.experts = StubGroupedExpertsLoRA()
        expected = dict(automodel_params_generator(model, torch.float32))
        mesh = init_device_mesh("cpu", (2,))
        # Three experts on two ranks also exercise an uneven expert shard.
        for name, parameter in list(model.experts.named_parameters()):
            sharded = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
            setattr(model.experts, name, nn.Parameter(sharded))
        results = dict(automodel_params_generator(model, torch.float32))
        assert set(results) == set(expected)
        for name, tensor in results.items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
def test_expert_parallel_refit_gathers_matching_adapters(tmp_path: Path) -> None:
    mp.spawn(
        _check_expert_parallel_refit,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
