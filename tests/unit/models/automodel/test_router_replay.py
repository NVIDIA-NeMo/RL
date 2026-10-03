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
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

try:
    from nemo_automodel._transformers.registry import ModelRegistry
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.router_replay import (
        RouterReplay,
        RouterReplayMode,
    )
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.models.automodel.router_replay import (
    enable_routing_replay_in_automodel_kwargs,
    microbatch_replay_context,
    microbatch_routed_experts,
    replay_routes,
    require_automodel_moe_model,
    router_replay_gates,
    validate_automodel_router_replay_config,
)

NUM_EXPERTS = 32


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_experts = NUM_EXPERTS
        self.router_replay = RouterReplay()


class _MoEMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()


class _Block(nn.Module):
    def __init__(self, moe: bool):
        super().__init__()
        self.mixer = _MoEMixer() if moe else nn.Identity()


class _Model(nn.Module):
    """Hybrid stack: MoE gates in decoder layers 1 and 3, plus an MTP gate vLLM never routes."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(moe=i in (1, 3)) for i in range(4)])
        self.mtp = nn.ModuleList([_Gate()])

    def gates(self):
        return [self.layers[1].mixer.gate, self.layers[3].mixer.gate]


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    RouterReplay.clear_registry()


def _config(**overrides):
    cfg = {
        "router_replay": {"enabled": True},
        "generation": {"backend": "vllm"},
        "dtensor_cfg": {"context_parallel_size": 1, "sequence_parallel": False},
        "sequence_packing": {"enabled": False},
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.automodel
@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"generation": {"backend": "megatron"}}, ValueError),
        (
            {"sequence_packing": {"enabled": True, "train_mb_tokens": 1024}},
            NotImplementedError,
        ),
        (
            {"dtensor_cfg": {"context_parallel_size": 2, "sequence_parallel": False}},
            NotImplementedError,
        ),
        (
            {"dtensor_cfg": {"context_parallel_size": 1, "sequence_parallel": True}},
            NotImplementedError,
        ),
    ],
)
def test_validate_rejects_unsupported_setups(overrides, error):
    with pytest.raises(error):
        validate_automodel_router_replay_config(_config(**overrides))


@pytest.mark.automodel
def test_validate_accepts_supported_layout_and_disabled_replay():
    validate_automodel_router_replay_config(_config())
    validate_automodel_router_replay_config(
        _config(
            router_replay={"enabled": False},
            generation={"backend": "megatron"},
            sequence_packing={"enabled": True, "train_mb_tokens": 1024},
        )
    )


@pytest.mark.automodel
@pytest.mark.parametrize(
    "architectures, force_hf",
    [(["NotAnAutomodelArch"], None), ([], None), ("registered", True)],
    ids=["unregistered-arch", "no-arch", "force-hf"],
)
def test_require_automodel_moe_model_rejects_hf_models(architectures, force_hf):
    registered = "NemotronHForCausalLM"
    assert registered in ModelRegistry.model_arch_name_to_cls
    if architectures == "registered":
        architectures = [registered]
    with pytest.raises(ValueError, match="loads through HF"):
        require_automodel_moe_model(
            {"force_hf": force_hf}, SimpleNamespace(architectures=architectures)
        )
    # A registered architecture without force_hf uses Automodel's own gates.
    require_automodel_moe_model({}, SimpleNamespace(architectures=[registered]))


@pytest.mark.automodel
def test_enable_routing_replay_keeps_other_moe_overrides():
    kwargs = {"moe_overrides": {"aux_loss_coeff": 0.0}}
    enable_routing_replay_in_automodel_kwargs(kwargs)
    assert kwargs["moe_overrides"] == {
        "aux_loss_coeff": 0.0,
        "enable_routing_replay": True,
    }


@pytest.mark.automodel
def test_gates_carry_layer_and_experts_and_exclude_mtp():
    model = _Model()
    gates = router_replay_gates(model)
    assert [(g.layer, g.num_experts) for g in gates] == [
        (1, NUM_EXPERTS),
        (3, NUM_EXPERTS),
    ]
    assert [g.handle for g in gates] == [g.router_replay for g in model.gates()]


_LIVE = torch.tensor([[0, 1], [2, 3], [4, 5]])


def _apply_all(model):
    g1, g3 = model.gates()
    # The MTP gate is never replayed.
    assert torch.equal(model.mtp[0].router_replay.apply(_LIVE), _LIVE)
    return g1.router_replay.apply(_LIVE).cpu(), g3.router_replay.apply(_LIVE).cpu()


@pytest.mark.automodel
def test_replay_moe_layer_layout_with_missing_route_fallback():
    model = _Model()
    gates = router_replay_gates(model)
    # [B=1, S=3, 2 MoE-layer slots, topk=2]; token 1 of the first MoE layer is the
    # all -1 missing-route sentinel and keeps the live selection.
    routed = torch.tensor(
        [[[[7, 8], [9, 10]], [[-1, -1], [11, 12]], [[13, 14], [15, 16]]]]
    )
    with replay_routes(gates, routed):
        first, second = _apply_all(model)
    assert torch.equal(first, torch.tensor([[7, 8], [2, 3], [13, 14]]))
    assert torch.equal(second, torch.tensor([[9, 10], [11, 12], [15, 16]]))
    # Outside the context the gates select live again.
    for gate in model.gates():
        assert torch.equal(gate.router_replay.apply(_LIVE), _LIVE)


@pytest.mark.automodel
def test_replay_decoder_layer_layout():
    model = _Model()
    gates = router_replay_gates(model)
    # vLLM's hybrid layout: one slot per decoder layer (4); gates read slots 1 and 3.
    routed = torch.arange(1 * 3 * 4 * 2).reshape(1, 3, 4, 2) % NUM_EXPERTS
    with replay_routes(gates, routed):
        first, second = _apply_all(model)
    assert torch.equal(first, routed[0, :, 1])
    assert torch.equal(second, routed[0, :, 3])


@pytest.mark.automodel
@pytest.mark.parametrize(
    "bad_route",
    [[-1, 5], [3, 3], [NUM_EXPERTS, 2]],
    ids=["partially-negative", "repeated-expert", "out-of-range"],
)
def test_replay_rejects_corrupt_routes(bad_route):
    model = _Model()
    gates = router_replay_gates(model)
    routed = torch.tensor([[[[1, 2], [3, 4]], [bad_route, [5, 6]], [[7, 8], [9, 10]]]])
    with pytest.raises(ValueError, match="Corrupt routed_experts .* decoder layer 1"):
        with replay_routes(gates, routed):
            pass


@pytest.mark.automodel
def test_replay_rejects_unmatched_layer_slots():
    model = _Model()
    gates = router_replay_gates(model)
    with pytest.raises(ValueError, match="matches neither"):
        with replay_routes(gates, torch.zeros(1, 3, 3, 2, dtype=torch.long)):
            pass


@pytest.mark.automodel
def test_replay_is_installed_per_instance_only():
    model = _Model()
    router_replay_gates(model)
    # Automodel's own RouterReplay semantics are untouched for any other instance.
    other = RouterReplay()
    other.mode = RouterReplayMode.REPLAY
    other.set_target(torch.tensor([[-1, -1]]))
    assert torch.equal(other.apply(torch.tensor([[0, 1]])), torch.tensor([[-1, -1]]))


@pytest.mark.automodel
def test_microbatch_routed_experts_masks_padding_and_checks_alignment():
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    lengths = torch.tensor([4, 2])
    routed = torch.arange(2 * 4 * 1 * 2, dtype=torch.int8).reshape(2, 4, 1, 2)
    out = microbatch_routed_experts(
        {"routed_experts": routed, "input_lengths": lengths}, input_ids
    )
    assert torch.equal(out[0], routed[0])
    assert torch.equal(out[1, :2], routed[1, :2])
    # Right-padding past input_lengths keeps the live selection.
    assert (out[1, 2:] == -1).all()
    with pytest.raises(ValueError, match="no routed_experts"):
        microbatch_routed_experts({"input_lengths": lengths}, input_ids)
    with pytest.raises(ValueError, match="aligned with input_ids"):
        microbatch_routed_experts(
            {"routed_experts": routed[:, :3], "input_lengths": lengths}, input_ids
        )


@pytest.mark.automodel
def test_replay_through_a_real_automodel_gate():
    torch.manual_seed(0)
    config = MoEConfig(
        n_routed_experts=8,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=4,
        inter_dim=8,
        moe_inter_dim=8,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.float32,
        enable_routing_replay=True,
    )
    gate = Gate(config)
    nn.init.normal_(gate.weight)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].gate = gate
    gates = router_replay_gates(model)
    x = torch.randn(3, 4)
    token_mask = torch.ones(3, dtype=torch.bool)
    _, live_indices, _ = gate(x, token_mask, None)

    # Tokens 0 and 2 replay rollout experts; token 1 has no route and stays live.
    routed = torch.tensor([[[[5, 6]], [[-1, -1]], [[0, 7]]]])
    with replay_routes(gates, routed):
        weights, indices, _ = gate(x, token_mask, None)

    assert torch.equal(indices[0], torch.tensor([5, 6]))
    assert torch.equal(indices[1], live_indices[1])
    assert torch.equal(indices[2], torch.tensor([0, 7]))
    # Weights come from the live router at the replayed experts, so it still trains.
    probs = (x @ gate.weight.T).softmax(dim=-1)
    assert torch.allclose(weights, probs.gather(1, indices))
    weights.sum().backward()
    assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0


@pytest.mark.automodel
def test_microbatch_replay_context_is_noop_without_gates():
    assert isinstance(microbatch_replay_context(None, SimpleNamespace()), nullcontext)
