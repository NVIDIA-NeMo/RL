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

"""CPU tests for the opt-in Super vision activation override."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_rl.models.megatron.vision_activation import set_super_vision_exact_gelu


def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_te_activation_func=False, bias_activation_fusion=False
        )
        self.activation_func = fast_gelu
        self.weight = nn.Parameter(torch.ones(7, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_func(x * self.weight)


def model_with_vision(layers: int = 32) -> nn.Module:
    model = nn.Module()
    model.vision_model = nn.Module()
    model.vision_model.layers = nn.ModuleList()
    for _ in range(layers):
        layer = nn.Module()
        layer.mlp = MLP()
        model.vision_model.layers.append(layer)
    model.language_model = nn.Module()
    model.language_model.mlp = MLP()
    return model


@pytest.mark.parametrize("container", [lambda m: m, lambda m: [m], lambda m: (m,)])
def test_exact_output_gradients_and_unchanged_parameters(container) -> None:
    model = model_with_vision()
    before = {name: parameter for name, parameter in model.named_parameters()}
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    x = torch.linspace(-3, 3, 7, dtype=torch.float64, requires_grad=True)
    approximate = model.vision_model.layers[0].mlp(x).detach()
    set_super_vision_exact_gelu(container(model))
    assert all(
        layer.mlp.activation_func is F.gelu for layer in model.vision_model.layers
    )
    assert model.language_model.mlp.activation_func is fast_gelu
    for name, parameter in model.named_parameters():
        assert parameter is before[name]
        torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)
    mlp = model.vision_model.layers[0].mlp
    output = mlp(x)
    expected_x = x.detach().clone().requires_grad_()
    expected_weight = mlp.weight.detach().clone().requires_grad_()
    expected = F.gelu(expected_x * expected_weight, approximate="none")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert not torch.equal(output.detach(), approximate)
    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(mlp.weight.grad, expected_weight.grad, rtol=0, atol=0)
    assert not mlp._forward_hooks and not mlp._forward_pre_hooks
    set_super_vision_exact_gelu(container(model))  # Idempotent.


@pytest.mark.parametrize("layers", [0, 31, 33])
def test_rejects_unexpected_layout_without_partial_update(layers: int) -> None:
    model = model_with_vision(layers)
    with pytest.raises(ValueError, match="requires 32 vision MLPs"):
        set_super_vision_exact_gelu(model)
    assert all(
        layer.mlp.activation_func is fast_gelu for layer in model.vision_model.layers
    )


@pytest.mark.parametrize("flag", ["use_te_activation_func", "bias_activation_fusion"])
def test_rejects_fused_activation_without_partial_update(flag: str) -> None:
    model = model_with_vision()
    setattr(model.vision_model.layers[-1].mlp.config, flag, True)
    with pytest.raises(ValueError, match="requires unfused"):
        set_super_vision_exact_gelu(model)
    assert all(
        layer.mlp.activation_func is fast_gelu for layer in model.vision_model.layers
    )


def test_rejects_non_gelu_activation_without_partial_update() -> None:
    model = model_with_vision()
    model.vision_model.layers[-1].mlp.activation_func = F.silu
    with pytest.raises(ValueError, match="Unexpected vision activation"):
        set_super_vision_exact_gelu(model)
    assert all(
        layer.mlp.activation_func is fast_gelu
        for layer in model.vision_model.layers[:-1]
    )
    assert model.vision_model.layers[-1].mlp.activation_func is F.silu
