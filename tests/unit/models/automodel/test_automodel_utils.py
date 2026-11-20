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
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
)

from nemo_rl.models.automodel.utils import (
    release_moe_dispatcher_state,
    resolve_model_class,
)


@pytest.mark.automodel
@pytest.mark.parametrize(
    ("model_type", "class_name"),
    [
        ("qwen2_5_vl", "nemo_image_text"),
        ("qwen2_5_omni", "nemo_text_waveform"),
        ("unknown_model", "nemo_causal_lm"),
    ],
)
def test_resolve_model_class_maps_model_type_to_loader(
    monkeypatch: pytest.MonkeyPatch, model_type: str, class_name: str
) -> None:
    """An unmapped model type falls back to the causal-LM loader."""
    classes = {
        "nemo_image_text": object(),
        "nemo_text_waveform": object(),
        "nemo_causal_lm": object(),
    }
    monkeypatch.setattr(
        "nemo_rl.models.automodel.utils.AUTOMODEL_FACTORY",
        {
            "qwen2_5_vl": classes["nemo_image_text"],
            "qwen2_5_omni": classes["nemo_text_waveform"],
        },
    )
    monkeypatch.setattr(
        "nemo_rl.models.automodel.utils.NeMoAutoModelForCausalLM",
        classes["nemo_causal_lm"],
    )

    assert resolve_model_class(model_type) is classes[class_name]


@pytest.mark.automodel
def test_resolve_model_class_routes_gemma4_unified_to_image_text_model():
    assert "ImageTextToText" in resolve_model_class("gemma4_unified").__name__


def _experts_with_manager(manager: SimpleNamespace) -> nn.Module:
    """An experts-like module holding a dispatcher the way Automodel does.

    The dispatcher is a plain attribute, not a submodule, and is built without
    __init__ because a real one needs an expert-parallel process group.
    """
    dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
    dispatcher._comm_manager = manager
    experts = nn.Linear(2, 2)
    experts.token_dispatcher = dispatcher
    return experts


@pytest.mark.automodel
def test_release_moe_dispatcher_state_clears_shared_and_per_layer_managers():
    """A manager shared by several layers is cleared once; config attrs stay."""
    shared = SimpleNamespace(
        num_experts=8,
        routing_map=torch.ones(4, 8, dtype=torch.bool),
        token_probs=torch.rand(4, 8),
        dispatched_probs=torch.rand(6),
        tokens_per_expert=torch.ones(8, dtype=torch.long),
        handle=object(),
        num_permuted_tokens=torch.tensor(6),
    )
    per_layer = SimpleNamespace(
        num_experts=8,
        token_probs=torch.rand(4, 2),
        token_indices=torch.zeros(4, 2, dtype=torch.long),
        dispatched_indices=torch.zeros(3, 2, dtype=torch.long),
        dispatched_routing_map=None,
        reversed_mapping_for_combine=torch.arange(3),
    )
    model = nn.Sequential(
        _experts_with_manager(shared),
        _experts_with_manager(shared),
        _experts_with_manager(per_layer),
        nn.Linear(2, 2),
    )

    assert release_moe_dispatcher_state(model) == 6 + 4

    for manager in (shared, per_layer):
        assert manager.num_experts == 8
        assert manager.token_probs is None
    assert shared.routing_map is None
    assert shared.dispatched_probs is None
    assert shared.tokens_per_expert is None
    assert shared.handle is None
    assert shared.num_permuted_tokens is None
    assert per_layer.token_indices is None
    assert per_layer.dispatched_indices is None
    assert per_layer.reversed_mapping_for_combine is None
    # A second call has nothing left to clear.
    assert release_moe_dispatcher_state(model) == 0


@pytest.mark.automodel
def test_release_moe_dispatcher_state_frees_activations_held_by_grad_fn():
    """Dispatch state with a grad_fn keeps saved activations alive until cleared.

    This mirrors the leak: after backward, ``token_probs`` comes from the
    activation-checkpoint recompute and its graph still holds the activations
    that the recompute saved.
    """
    hidden = torch.randn(16, 32, requires_grad=True)
    activation = hidden * 2.0
    activation_ref = weakref.ref(activation)
    # sigmoid saves its output, matmul saves ``activation``.
    weight = torch.randn(32, 8, requires_grad=True)
    probs = (activation @ weight).sigmoid()
    manager = SimpleNamespace(token_probs=probs, dispatched_probs=probs * 1.0)
    model = nn.Sequential(_experts_with_manager(manager))
    del activation, probs
    gc.collect()
    assert activation_ref() is not None

    assert release_moe_dispatcher_state(model) == 2
    gc.collect()
    assert activation_ref() is None


@pytest.mark.automodel
def test_release_moe_dispatcher_state_ignores_dense_models():
    """Models without MoE dispatchers are a no-op."""
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    model[0].token_dispatcher = None
    assert release_moe_dispatcher_state(model) == 0
