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

from nemo_rl.models.generation.vllm import batch_invariant


def test_true_on_policy_patches_disabled():
    result = batch_invariant.install_true_on_policy_patches(
        torch.nn.Module(),
        bf16_true_on_policy=False,
    )

    assert result == {}


def test_true_on_policy_patches_install_requested_components(monkeypatch):
    calls = []

    def _patch_result(name):
        def _installer(model):
            calls.append((name, model))
            return {"patched": name}

        return _installer

    monkeypatch.setattr(
        batch_invariant,
        "install_megatron_style_rmsnorm_patch",
        _patch_result("rmsnorm"),
    )
    monkeypatch.setattr(
        batch_invariant,
        "install_megatron_style_rope_patch",
        _patch_result("rope"),
    )
    monkeypatch.setattr(
        batch_invariant,
        "install_megatron_style_swiglu_patch",
        _patch_result("swiglu"),
    )
    monkeypatch.setenv(
        batch_invariant.G_TRUE_ON_POLICY_COMPONENTS_ENV,
        "rmsnorm, swiglu",
    )

    model = torch.nn.Module()
    result = batch_invariant.install_true_on_policy_patches(
        model,
        bf16_true_on_policy=True,
    )

    assert calls == [("rmsnorm", model), ("swiglu", model)]
    assert result == {
        "bf16_components": ("rmsnorm", "swiglu"),
        "megatron_style_rmsnorm": {"patched": "rmsnorm"},
        "megatron_style_swiglu": {"patched": "swiglu"},
    }


def test_true_on_policy_patches_default_to_all_components(monkeypatch):
    calls = []

    monkeypatch.setattr(
        batch_invariant,
        "install_megatron_style_rmsnorm_patch",
        lambda model: calls.append("rmsnorm") or {"patched": "rmsnorm"},
    )
    monkeypatch.setattr(
        batch_invariant,
        "install_megatron_style_rope_patch",
        lambda model: calls.append("rope") or {"patched": "rope"},
    )
    monkeypatch.setattr(
        batch_invariant,
        "install_megatron_style_swiglu_patch",
        lambda model: calls.append("swiglu") or {"patched": "swiglu"},
    )
    monkeypatch.delenv(batch_invariant.G_TRUE_ON_POLICY_COMPONENTS_ENV, raising=False)

    result = batch_invariant.install_true_on_policy_patches(
        torch.nn.Module(),
        bf16_true_on_policy=True,
    )

    assert calls == ["rmsnorm", "rope", "swiglu"]
    assert result["bf16_components"] == ("rmsnorm", "rope", "swiglu")


def test_true_on_policy_patches_reject_unknown_component(monkeypatch):
    monkeypatch.setenv(
        batch_invariant.G_TRUE_ON_POLICY_COMPONENTS_ENV,
        "rmsnorm,not_a_patch",
    )

    with pytest.raises(
        ValueError,
        match="Unknown vLLM true-on-policy patch components",
    ):
        batch_invariant.install_true_on_policy_patches(
            torch.nn.Module(),
            bf16_true_on_policy=True,
        )
