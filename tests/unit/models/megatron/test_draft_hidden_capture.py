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

"""Target-SP reconstruction tests for online draft hidden capture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch

_MODULE_NAME = "nemo_rl.models.megatron.draft.hidden_capture"
_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/megatron/draft/hidden_capture.py"
)


def _load_hidden_capture_module() -> ModuleType:
    megatron_core = ModuleType("megatron.core")
    megatron_core.parallel_state = MagicMock()
    megatron_core_utils = ModuleType("megatron.core.utils")
    megatron_core_utils.unwrap_model = lambda model: model
    prior_modules = {
        name: sys.modules.get(name)
        for name in ("megatron.core", "megatron.core.utils", _MODULE_NAME)
    }
    sys.modules["megatron.core"] = megatron_core
    sys.modules["megatron.core.utils"] = megatron_core_utils
    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load hidden-capture production module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in prior_modules.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def test_target_sp_capture_reconstructs_once_over_tp_group() -> None:
    module = _load_hidden_capture_module()
    capture = object.__new__(module.HiddenStateCapture)
    capture.pp_size = 1
    capture.aux_layer_indices = (1, 3)
    capture._captured = {
        "layer_1": torch.full((2, 1, 2), 11.0),
        "layer_3": torch.full((2, 1, 2), 33.0),
        "embeds": torch.full((2, 1, 2), 55.0),
        "output_hidden": torch.full((2, 1, 2), 77.0),
    }
    tp_group = MagicMock(name="tp_group")
    module.parallel_state.get_tensor_model_parallel_group.return_value = tp_group
    reconstructed_hidden = torch.full((4, 1, 4), 7.0)
    reconstructed_embeds = torch.full((4, 1, 2), 9.0)
    reconstructed_output_hidden = torch.full((4, 1, 2), 13.0)
    module._reconstruct_tp_sequence = MagicMock(
        side_effect=(
            reconstructed_hidden,
            reconstructed_embeds,
            reconstructed_output_hidden,
        )
    )
    layout = SimpleNamespace(tp_size=2)

    states = capture.get_captured_states(sequence_layout=layout)

    assert states.hidden_states is reconstructed_hidden
    assert states.inputs_embeds is reconstructed_embeds
    assert states.output_hidden is reconstructed_output_hidden
    assert states.sequence_layout is layout
    assert states.sequence_is_reconstructed is True
    assert module._reconstruct_tp_sequence.call_count == 3
    hidden_call, embed_call, output_hidden_call = (
        module._reconstruct_tp_sequence.call_args_list
    )
    assert hidden_call.kwargs["tp_group"] is tp_group
    assert embed_call.kwargs["tp_group"] is tp_group
    assert hidden_call.kwargs["sequence_layout"] is layout
    assert embed_call.kwargs["sequence_layout"] is layout
    assert hidden_call.kwargs["sequence_dim"] == 0
    assert embed_call.kwargs["sequence_dim"] == 0
    assert output_hidden_call.kwargs["sequence_dim"] == 0
    assert hidden_call.args[0].shape == (2, 1, 4)
    assert embed_call.args[0].shape == (2, 1, 2)
    assert output_hidden_call.args[0].shape == (2, 1, 2)


def test_capture_hooks_keep_detached_views_without_per_tap_clones() -> None:
    module = _load_hidden_capture_module()
    capture = object.__new__(module.HiddenStateCapture)
    capture._captured = {}
    hidden = torch.randn(3, 1, 4, requires_grad=True)
    embeds = torch.randn(3, 1, 4, requires_grad=True)

    capture._make_layer_output_hook(2)(None, None, hidden)
    capture._make_embedding_hook()(None, None, embeds)
    capture._make_output_hidden_hook()(None, (hidden,))

    captured_hidden = capture._captured["layer_2"]
    captured_embeds = capture._captured["embeds"]
    captured_output_hidden = capture._captured["output_hidden"]
    assert captured_hidden.data_ptr() == hidden.data_ptr()
    assert captured_embeds.data_ptr() == embeds.data_ptr()
    assert captured_output_hidden.data_ptr() == hidden.data_ptr()
    assert captured_hidden.requires_grad is False
    assert captured_embeds.requires_grad is False
    assert captured_output_hidden.requires_grad is False


def _local_capture(module: ModuleType, captured: dict[str, torch.Tensor]) -> Any:
    capture = object.__new__(module.HiddenStateCapture)
    capture.pp_size = 1
    capture.aux_layer_indices = (1,)
    capture._captured = captured
    return capture


def test_capture_without_layout_returns_local_states_unchanged() -> None:
    module = _load_hidden_capture_module()
    hidden = torch.randn(2, 1, 3)
    capture = _local_capture(module, {"layer_1": hidden})

    states = capture.get_captured_states()

    assert torch.equal(states.hidden_states, hidden)
    assert states.sequence_layout is None
    assert states.sequence_is_reconstructed is False


def test_empty_capture_with_layout_skips_reconstruction() -> None:
    module = _load_hidden_capture_module()
    module._reconstruct_tp_sequence = MagicMock()
    capture = _local_capture(module, {})
    layout = SimpleNamespace(tp_size=2)

    states = capture.get_captured_states(sequence_layout=layout)

    assert states.hidden_states is None
    assert states.inputs_embeds is None
    assert states.output_hidden is None
    assert states.sequence_layout is layout
    module._reconstruct_tp_sequence.assert_not_called()


def test_output_hidden_hook_ignores_calls_without_positional_input() -> None:
    module = _load_hidden_capture_module()
    capture = _local_capture(module, {})

    capture._make_output_hidden_hook()(None, ())

    assert "output_hidden" not in capture._captured


def test_registered_hooks_capture_first_and_last_stage_tensors_then_detach() -> None:
    module = _load_hidden_capture_module()
    capture = _local_capture(module, {})
    capture.is_first_stage = True
    capture.is_last_stage = True
    capture._local_aux_indices = []
    capture._hooks = []
    capture.model = SimpleNamespace(
        embedding=torch.nn.Identity(),
        output_layer=torch.nn.Identity(),
    )
    embeds = torch.randn(2, 1, 3)
    output_hidden = torch.randn(2, 1, 3)

    with capture.capture_context():
        capture.model.embedding(embeds)
        capture.model.output_layer(output_hidden)
        assert torch.equal(capture._captured["embeds"], embeds)
        assert torch.equal(capture._captured["output_hidden"], output_hidden)
    capture._captured.clear()
    capture.model.embedding(embeds)

    assert capture._hooks == []
    assert capture._captured == {}


def test_reconstruct_wrapper_forwards_to_sequence_parallel() -> None:
    module = _load_hidden_capture_module()
    sp_module_name = "nemo_rl.models.megatron.draft.sequence_parallel"
    prior = sys.modules.get(sp_module_name)
    sp_module = ModuleType(sp_module_name)
    sp_module.reconstruct_tp_sequence = MagicMock(return_value="rebuilt")
    sys.modules[sp_module_name] = sp_module
    try:
        local = torch.randn(4, 2)
        layout = SimpleNamespace(tp_size=2)
        group = object()

        result = module._reconstruct_tp_sequence(
            local,
            sequence_layout=layout,
            tp_group=group,
            sequence_dim=0,
        )
    finally:
        if prior is None:
            sys.modules.pop(sp_module_name, None)
        else:
            sys.modules[sp_module_name] = prior

    assert result == "rebuilt"
    sp_module.reconstruct_tp_sequence.assert_called_once_with(
        local,
        sequence_layout=layout,
        tp_group=group,
        sequence_dim=0,
    )
