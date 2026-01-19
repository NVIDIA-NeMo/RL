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


import math

import pytest

# Skip entire module if nemo_automodel is not available
pytest_plugins = []
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

import torch
import torch.nn as nn
from nemo_automodel.components._peft.lora import (
    LinearLoRA,
    PeftConfig,
    apply_lora_to_linear_modules,
)
from nemo_automodel.components._peft.module_matcher import ModuleMatcher


class SimpleLoraMock(nn.Module):
    """Simple mock LoRA module for testing initialization."""

    def __init__(self, in_features=128, out_features=256, lora_dim=8):
        super().__init__()
        self.lora_A = nn.Linear(in_features, lora_dim, bias=False)
        self.lora_B = nn.Linear(lora_dim, out_features, bias=False)


def test_lora_init_statistical_properties():
    """
    Additional test to verify the statistical properties of the patched initialization.
    This ensures our fix produces reasonable weight distributions.
    """
    torch.manual_seed(42)

    lora = SimpleLoraMock(in_features=512, out_features=1024, lora_dim=32)

    # Test xavier initialization
    LinearLoRA.init_lora_weights(lora, "xavier")

    # Xavier normal should have mean ≈ 0 and specific std
    mean_A = lora.lora_A.weight.data.mean().item()
    std_A = lora.lora_A.weight.data.std().item()

    assert abs(mean_A) < 0.1, f"Xavier normal should have mean ≈ 0, got {mean_A}"
    # Xavier normal std = sqrt(2 / (fan_in + fan_out))
    expected_std = math.sqrt(2.0 / (512 + 32))
    assert abs(std_A - expected_std) < 0.05, (
        f"Xavier normal std should be ≈ {expected_std}, got {std_A}"
    )

    # LoRA B should be all zeros
    assert torch.all(lora.lora_B.weight.data == 0), "LoRA B should be zero-initialized"

    # Test kaiming initialization
    lora2 = SimpleLoraMock(in_features=512, out_features=1024, lora_dim=32)
    LinearLoRA.init_lora_weights(lora2, "kaiming")

    mean_A2 = lora2.lora_A.weight.data.mean().item()
    assert abs(mean_A2) < 0.1, f"Kaiming should have mean ≈ 0, got {mean_A2}"
    assert torch.all(lora2.lora_B.weight.data == 0), "LoRA B should be zero-initialized"


class DummyModel(nn.Module):
    """A dummy neural network model with two linear layers used for testing LoRA injection."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)
        self.config = {}

    def forward(self, x):
        """Forward pass through two linear layers with ReLU activation in between."""
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x


class DummyModelNoConfig(nn.Module):
    """Same as DummyModel but without a `config` attribute."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)

    def forward(self, x):
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x


@pytest.fixture
def dummy_input():
    """Provides a dummy input tensor for model testing."""
    return torch.randn(2, 16, requires_grad=True)


@pytest.fixture
def model():
    """Instantiates and returns a DummyModel instance."""
    return DummyModel()


@pytest.fixture
def model_no_config():
    """Instantiates a model that has no `config` attr."""
    return DummyModelNoConfig()


def test_lora_patch_applies_to_selected_module(model):
    """Tests that LoRA is only applied to specified target modules."""
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8)
    )
    assert isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)


def test_lora_patch_on_model_without_config(model_no_config):
    """LoRA should still patch correctly even if the model lacks `config`."""
    apply_lora_to_linear_modules(
        model_no_config, PeftConfig(target_modules=["linear1"], dim=4, alpha=8)
    )
    assert isinstance(model_no_config.linear1, LinearLoRA)
    assert not isinstance(model_no_config.linear2, LinearLoRA)


def test_lora_layers_are_trainable():
    """Ensures that LoRA layers are trainable while base weights remain frozen."""
    base = nn.Linear(16, 16)
    lora = LinearLoRA(base, dim=4, alpha=8)

    assert lora.weight.requires_grad is False
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad
    if lora.bias is not None:
        assert lora.bias.requires_grad is False


def test_forward_output_consistency(dummy_input):
    """Verifies that model output shape remains the same after LoRA patching,
    but values change due to the added LoRA components.
    """
    base = DummyModel()
    model = DummyModel()
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8)
    )

    base.eval()
    model.eval()

    with torch.no_grad():
        out1 = base(dummy_input)
        out2 = model(dummy_input)

    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2), "Output should differ due to LoRA injection"


def test_backward_pass(dummy_input):
    """Checks that backpropagation works and gradients are correctly computed
    when LoRA is applied.
    """
    model = DummyModel()
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8)
    )
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "Some parameters should receive gradients"
    assert all(torch.isfinite(g).all() for g in grads if g is not None), (
        "Gradients should be finite"
    )


def test_backward_pass_without_config(dummy_input, model_no_config):
    """Backward pass must succeed on a model without `config`."""
    apply_lora_to_linear_modules(
        model_no_config, PeftConfig(target_modules=["linear1"], dim=4, alpha=8)
    )
    out = model_no_config(dummy_input)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in model_no_config.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_apply_lora_respects_wildcard(model):
    """Validates that wildcard matching correctly applies LoRA to all matching modules."""
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=[".*"], dim=4, alpha=8)
    )
    assert isinstance(model.linear1, LinearLoRA)
    assert isinstance(model.linear2, LinearLoRA)


def test_no_patch_on_non_matching_module(model):
    """Confirms that no modules are patched if target pattern doesn't match any names."""
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=["nonexistent_module"], dim=4, alpha=8)
    )
    assert not isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)


def test_lora_patch_with_dtype_string(model):
    """Tests that LoRA can be applied with dtype specified as string."""
    apply_lora_to_linear_modules(
        model,
        PeftConfig(
            target_modules=["linear1"], dim=4, alpha=8, lora_dtype="torch.bfloat16"
        ),
    )
    assert isinstance(model.linear1, LinearLoRA)
    assert model.linear1.lora_A.weight.dtype == torch.bfloat16
    assert model.linear1.lora_B.weight.dtype == torch.bfloat16
    assert not isinstance(model.linear2, LinearLoRA)


def test_dropout_pre_post_effects(dummy_input):
    """Tests that different dropout positions ('pre' vs 'post') lead to different outputs."""
    base = nn.Linear(16, 16)
    lora_pre = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position="pre")
    lora_post = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position="post")

    with torch.no_grad():
        lora_pre.lora_A.weight.uniform_()
        lora_pre.lora_B.weight.uniform_()

        lora_post.lora_A.weight.copy_(lora_pre.lora_A.weight)
        lora_post.lora_B.weight.copy_(lora_pre.lora_B.weight)

    lora_pre.train()
    lora_post.train()

    out_pre = lora_pre(dummy_input)
    out_post = lora_post(dummy_input)

    assert out_pre.shape == out_post.shape
    assert not torch.allclose(out_pre, out_post), (
        "Dropout positions should affect output differently"
    )


def test_patched_get_supported_lora_modules_needed():
    target_modules = ["*"]
    exclude_modules = []
    match_all_linear = True
    is_causal_lm = True
    module_matcher = ModuleMatcher(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        match_all_linear=match_all_linear,
        is_causal_lm=is_causal_lm,
    )

    model = nn.Module()
    model.lm_head = nn.Linear(10, 10)
    is_lm_head_supported = module_matcher.match(model.lm_head, "lm_head")
    assert is_lm_head_supported is False, (
        "LoRA Adapter should not be applied to lm_head. "
        "If this assertion fails, the upstream bug has been fixed in Automodel. "
        "You can:\n"
        "1. Remove the patch patched_get_supported_lora_modules in nemo_rl/models/generation/vllm/lora.py\n"
        "2. Remove the patching call\n"
        "3. Retest the reward in train and accuracy in validation at the first step should be exactly equal for Llama3.2-3B-Instruct model.\n"
        "4. Delete this test"
    )
