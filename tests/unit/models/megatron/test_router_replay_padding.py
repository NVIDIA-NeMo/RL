# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import router_replay


def _config(**overrides):
    return SimpleNamespace(
        **{
            "num_layers": 1,
            "moe_layer_freq": 1,
            "sequence_parallel": False,
            "num_moe_experts": 512,
            "moe_expert_capacity_factor": None,
            "moe_expert_rank_capacity_factor": None,
            "moe_pad_experts_for_cuda_graph_inference": False,
            "moe_token_dropping": False,
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            **overrides,
        }
    )


def test_long_padding_balances_ep64_and_preserves_owned_routes():
    # The failing real row was 41,881 tokens wide with 21,793 valid tokens.
    routes = torch.arange(22).expand(41888, -1).clone()
    routes[:21793] = (routes[:21793] + 107) % 512
    original = routes.clone()
    padding = torch.arange(len(routes)) >= 21793
    balanced = router_replay._balance_replay_padding(routes, padding, _config())

    assert torch.equal(routes, original)
    assert torch.equal(balanced[~padding], original[~padding])
    assert torch.all(balanced[padding].sort(-1).values.diff(dim=-1) > 0)
    counts = torch.bincount(balanced[padding].flatten(), minlength=512)
    assert int(counts.max() - counts.min()) <= 1
    ep_counts = counts.reshape(64, 8).sum(-1)
    old_peak = int(padding.sum()) * 8
    assert old_peak > 20 * int(ep_counts.max())


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_expert_capacity_factor": 1.0},
        {"moe_expert_rank_capacity_factor": 1.0},
        {"moe_pad_experts_for_cuda_graph_inference": True},
        {"moe_token_dropping": True},
        {"moe_router_load_balancing_type": "seq_aux_loss"},
        {"moe_router_bias_update_rate": 0.001},
    ],
)
def test_dummy_sensitive_routers_keep_original_routes(overrides):
    routes = torch.arange(22).expand(20, -1)
    padding = torch.ones(20, dtype=torch.bool)
    assert (
        router_replay._balance_replay_padding(routes, padding, _config(**overrides))
        is routes
    )


def test_balanced_padding_preserves_valid_loss_and_expert_gradients():
    routes = torch.arange(3).expand(25, -1).clone()
    padding = torch.arange(25) >= 7
    balanced = router_replay._balance_replay_padding(
        routes, padding, _config(num_moe_experts=8)
    )
    inputs = torch.randn(25, dtype=torch.float64)
    base = torch.randn(8, dtype=torch.float64)
    results = []
    for indices in (routes, balanced):
        weights = base.clone().requires_grad_()
        output = (weights[indices] * inputs[:, None]).sum(-1)
        loss = (output.square() * ~padding).sum()
        loss.backward()
        results.append((loss.detach(), weights.grad))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_assignments_align_dense_batch_padding_before_tp_split(monkeypatch, tp_rank):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _config(sequence_parallel=True)
            self.layer_number = 1
            self.router_replay = SimpleNamespace()

    monkeypatch.setattr(
        router_replay, "_get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        router_replay, "_get_tensor_model_parallel_rank", lambda: tp_rank
    )
    routes = torch.arange(3).expand(2, 8, 1, 3).clone()
    padding = torch.arange(8)[None, :] >= torch.tensor([3, 5])[:, None]
    source = routes.transpose(0, 1).reshape(16, 1, 3)[tp_rank * 8 : (tp_rank + 1) * 8]
    local_padding = padding.transpose(0, 1).reshape(-1)[tp_rank * 8 : (tp_rank + 1) * 8]
    assignment = router_replay.build_router_replay_assignments(
        Model(), routes, padding_mask=padding
    )[0][1]
    assert torch.equal(assignment[~local_padding], source[~local_padding, 0])
    expected = router_replay._balance_replay_padding(
        source[:, 0], local_padding, _config()
    )
    assert torch.equal(assignment, expected)
