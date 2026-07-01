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

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest


def _load_refit_helpers():
    """Load the small refit helpers without importing Megatron/Ray/Torch."""
    source = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
    )
    module = ast.parse(source.read_text())
    helper_names = {
        "_should_merge_adapter_weights_for_refit",
        "_validate_merged_refit_weight_name",
        "_iter_refit_base_weights",
    }
    helper_nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {node.name for node in helper_nodes} == helper_names

    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    ast.fix_missing_locations(helper_module)
    namespace = {
        "Any": Any,
        "Iterator": Iterator,
        "PolicyConfig": dict,
        "torch": SimpleNamespace(Tensor=object),
    }
    exec(compile(helper_module, str(source), "exec"), namespace)
    return namespace["_iter_refit_base_weights"]


_iter_refit_base_weights = _load_refit_helpers()


class RecordingBridge:
    def __init__(self, names: list[str]):
        self.names = names
        self.merge_adapter_weights = None

    def export_hf_weights(
        self,
        model,
        *,
        show_progress,
        conversion_tasks,
        merge_adapter_weights,
    ):
        self.merge_adapter_weights = merge_adapter_weights
        for name in self.names:
            yield name, object()


def _policy_config(peft_enabled: bool):
    return {
        "megatron_cfg": {
            "peft": {
                "enabled": peft_enabled,
            },
        },
    }


def test_refit_export_keeps_full_weight_fast_path_without_peft():
    bridge = RecordingBridge(["model.layers.0.self_attn.q_proj.base_layer.weight"])

    names = [
        name
        for name, _ in _iter_refit_base_weights(
            megatron_bridge=bridge,
            model=object(),
            conversion_tasks=[],
            config=_policy_config(peft_enabled=False),
        )
    ]

    assert bridge.merge_adapter_weights is False
    assert names == ["model.layers.0.self_attn.q_proj.base_layer.weight"]


def test_refit_export_requests_merged_weights_with_peft():
    bridge = RecordingBridge(["model.layers.0.self_attn.q_proj.weight"])

    names = [
        name
        for name, _ in _iter_refit_base_weights(
            megatron_bridge=bridge,
            model=object(),
            conversion_tasks=[],
            config=_policy_config(peft_enabled=True),
        )
    ]

    assert bridge.merge_adapter_weights is True
    assert names == ["model.layers.0.self_attn.q_proj.weight"]


@pytest.mark.parametrize(
    "bad_name",
    [
        "model.layers.0.self_attn.q_proj.base_layer.weight",
        "model.layers.0.self_attn.q_proj.adapter.linear_in.weight",
        "model.layers.0.self_attn.q_proj.lora_A.weight",
    ],
)
def test_refit_export_rejects_unmerged_peft_keys_with_peft(bad_name):
    bridge = RecordingBridge([bad_name])

    with pytest.raises(RuntimeError, match="unmerged PEFT wrapper key"):
        list(
            _iter_refit_base_weights(
                megatron_bridge=bridge,
                model=object(),
                conversion_tasks=[],
                config=_policy_config(peft_enabled=True),
            )
        )

    assert bridge.merge_adapter_weights is True
