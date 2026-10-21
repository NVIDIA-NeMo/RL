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

"""Numerical checks for transient expert gradients and BF16 update accumulation."""

import copy

import pytest
import torch
from logra.compression import install_sketches
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class TinyExperts(nn.Module):
    """Same three-dimensional parameter layout as AutoModel grouped experts."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_and_up_projs = nn.Parameter(torch.randn(3, 7, 5) * 0.1)
        self.down_projs = nn.Parameter(torch.randn(3, 5, 7) * 0.1)
        self.fail = False

    def forward(self, x: Tensor) -> Tensor:
        if self.fail:
            raise RuntimeError("test forward failure")
        hidden = torch.einsum("ebi,eio->ebo", x, self.gate_and_up_projs).tanh()
        return torch.einsum("ebi,eio->ebo", hidden, self.down_projs)


def expert_config() -> LoGRAConfig:
    return LoGRAConfig(
        rank=3,
        optimizer="sgd",
        target_modules=[],
        target_parameters=["gate_and_up_projs", "down_projs"],
    )


@pytest.mark.parametrize("recompute", [False, True])
def test_fused_sketch_and_input_gradient(recompute: bool) -> None:
    torch.manual_seed(42)
    dense = TinyExperts()
    compressed = copy.deepcopy(dense)
    originals = dict(compressed.named_parameters())
    states = install_sketches(compressed, expert_config())
    x = torch.randn(3, 6, 7, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    dense(x).square().sum().backward()
    for chunk in z.chunk(2, dim=1):
        output = (
            checkpoint(compressed, chunk, use_reentrant=False)
            if recompute
            else compressed(chunk)
        )
        output.square().sum().backward()
    for state in states:
        gradient = dict(dense.named_parameters())[state.weight_name].grad
        torch.testing.assert_close(
            state.sketch, gradient.reshape(-1, gradient.shape[-1]) @ state.projection.T
        )
        assert state.weight is originals[state.weight_name]
        assert state.weight.grad is None
        assert state.swapped_weight is None
    torch.testing.assert_close(x.grad, z.grad)


def test_expert_forward_failure_restores_parameters() -> None:
    net = TinyExperts()
    originals = dict(net.named_parameters())
    states = install_sketches(net, expert_config())
    net.fail = True
    with pytest.raises(RuntimeError, match="test forward failure"):
        net(torch.randn(3, 2, 7))
    for state in states:
        assert state.weight is originals[state.weight_name]
        assert state.swapped_weight is None
    assert set(net.state_dict()) == set(originals)


def test_expert_sgd_update_matches_dense_projection() -> None:
    torch.manual_seed(6)
    reference = TinyExperts()
    net = copy.deepcopy(reference)
    opt = LoGRAOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0),
        expert_config(),
    )
    x = torch.randn(3, 4, 7, requires_grad=True)
    reference(x).square().sum().backward()
    net(x).square().sum().backward()
    expected = {}
    for layer in opt.layers:
        parameter = dict(reference.named_parameters())[layer.weight_name]
        gradient = parameter.grad.reshape(-1, parameter.shape[-1])
        update = gradient @ layer.projection.T @ layer.projection
        expected[layer.weight_name] = parameter.detach() - 0.01 * update.view_as(
            parameter
        )
    opt.synchronize_and_clip(None, None)
    opt.step()
    for name, parameter in net.named_parameters():
        torch.testing.assert_close(parameter, expected[name])
        assert parameter.grad is None


@pytest.mark.parametrize("rule", ["sgd", "row_adam"])
def test_bf16_small_updates_and_resume_preserve_fp32_state(rule: str) -> None:
    def create() -> tuple[nn.Sequential, LoGRAOptimizer]:
        net = nn.Sequential(nn.Linear(1, 1, bias=False, dtype=torch.bfloat16))
        with torch.no_grad():
            net[0].weight.fill_(1)
        opt = LoGRAOptimizer(
            net,
            torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0),
            LoGRAConfig(rank=1, target_modules=["0"], optimizer=rule, refresh=False),
        )
        return net, opt

    def step(opt: LoGRAOptimizer) -> None:
        opt.zero_grad()
        opt.layers[0].sketch.copy_(opt.layers[0].projection)
        opt.synchronize_and_clip(None, None)
        opt.step()

    net, opt = create()
    for _ in range(9):
        step(opt)
    saved = copy.deepcopy(opt.state_dict())
    clone, other = create()
    clone.load_state_dict(net.state_dict())
    other.load_state_dict(saved)
    for key, value in opt.state[net[0].weight].items():
        restored = other.state[clone[0].weight][key]
        assert restored.dtype == torch.float32
        torch.testing.assert_close(value, restored, rtol=0, atol=0)
    for _ in range(16):
        step(opt)
        step(other)
    assert net[0].weight.item() < 1.0  # Each individual BF16-only update would be lost.
    torch.testing.assert_close(net[0].weight, clone[0].weight, rtol=0, atol=0)
    for key, value in opt.state[net[0].weight].items():
        torch.testing.assert_close(
            value, other.state[clone[0].weight][key], rtol=0, atol=0
        )


def test_old_dense_config_checkpoint_remains_loadable() -> None:
    net = nn.Sequential(nn.Linear(2, 2))
    opt = LoGRAOptimizer(
        net, torch.optim.AdamW(net.parameters()), LoGRAConfig(target_modules=["0"])
    )
    state = opt.state_dict()
    del state["config"]["target_parameters"]
    del state["config"]["target_prefixes"]
    opt.load_state_dict(state)
