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

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_overlay_module():
    module_path = (
        Path(__file__).resolve().parents[4]
        / "infra/nrl_k8s/dynamo_mx/mx_refit_extension_megatron.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_mx_refit_extension_megatron",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecordingModel:
    def __init__(self):
        self.loaded_weights = None

    def load_weights(self, *, weights):
        self.loaded_weights = list(weights)


class RecordingDraftModel(RecordingModel):
    def __init__(self, *, org_vocab_size: int | None = None):
        super().__init__()
        self._org_vocab_size = org_vocab_size

    def named_modules(self):
        if self._org_vocab_size is None:
            return []
        return [
            (
                "lm_head",
                types.SimpleNamespace(org_vocab_size=self._org_vocab_size),
            )
        ]


def _worker_with(model_runner):
    module = _load_overlay_module()
    worker = object.__new__(module.MxRefitWorkerExtension)
    worker.model_runner = model_runner
    return worker


def test_torch_dtype_handles_draft_d2t_int64():
    module = _load_overlay_module()

    assert module._torch_dtype("int64") is torch.int64
    assert module._torch_dtype("torch.int64") is torch.int64


def test_derives_qwen_llama_qkv_hf_names_without_sidecar_map():
    module = _load_overlay_module()

    assert module._derive_qwen_llama_hf_names(
        "module.decoder.layers.0.self_attention.linear_qkv.weight",
        "qkv_column",
    ) == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]


def test_derives_qwen_llama_gated_mlp_hf_names_without_sidecar_map():
    module = _load_overlay_module()

    assert module._derive_qwen_llama_hf_names(
        "module.decoder.layers.1.mlp.linear_fc1.weight",
        "gated_mlp_column",
    ) == [
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
    ]


def test_resolves_identity_megatron_name_map_with_derived_hf_name():
    module = _load_overlay_module()

    assert module._resolve_hf_names(
        "module.decoder.layers.2.input_layernorm.weight",
        "replicated",
        {
            "decoder.layers.2.input_layernorm.weight": [
                "module.decoder.layers.2.input_layernorm.weight"
            ]
        },
    ) == ["model.layers.2.input_layernorm.weight"]


def test_resolves_module_prefixed_hf_name_map_without_wrapper_prefix():
    module = _load_overlay_module()

    assert module._resolve_hf_names(
        "module.decoder.layers.2.input_layernorm.weight",
        "replicated",
        {
            "decoder.layers.2.input_layernorm.weight": [
                "module.model.layers.2.input_layernorm.weight"
            ]
        },
    ) == ["model.layers.2.input_layernorm.weight"]


def test_derives_fallback_name_without_module_wrapper_prefix():
    module = _load_overlay_module()

    assert module._derive_qwen_llama_hf_names(
        "module.decoder.layers.2.unhandled.weight",
        "replicated",
    ) == ["decoder.layers.2.unhandled.weight"]


def test_mx_load_weights_routes_draft_weights_to_drafter():
    policy_model = RecordingModel()
    draft_model = RecordingDraftModel(org_vocab_size=2)
    worker = _worker_with(
        types.SimpleNamespace(
            model=policy_model,
            drafter=types.SimpleNamespace(model=draft_model),
        )
    )

    policy_tensor = torch.ones(2, 2)
    draft_tensor = torch.arange(8).reshape(4, 2)

    worker._mx_load_weights(
        [
            ("model.layers.0.weight", policy_tensor),
            ("draft.lm_head.weight", draft_tensor),
        ]
    )

    assert len(policy_model.loaded_weights) == 1
    policy_name, loaded_policy_tensor = policy_model.loaded_weights[0]
    assert policy_name == "model.layers.0.weight"
    assert loaded_policy_tensor is policy_tensor
    assert len(draft_model.loaded_weights) == 1
    draft_name, loaded_draft_tensor = draft_model.loaded_weights[0]
    assert draft_name == "lm_head.weight"
    assert torch.equal(loaded_draft_tensor, draft_tensor[:2])


def test_mx_load_weights_routes_draft_weights_to_speculator_model():
    policy_model = RecordingModel()
    draft_model = RecordingDraftModel(org_vocab_size=2)
    worker = _worker_with(
        types.SimpleNamespace(
            model=policy_model,
            speculator=types.SimpleNamespace(model=draft_model),
        )
    )

    draft_tensor = torch.arange(8).reshape(4, 2)

    worker._mx_load_weights([("draft.lm_head.weight", draft_tensor)])

    assert policy_model.loaded_weights == []
    assert len(draft_model.loaded_weights) == 1
    draft_name, loaded_draft_tensor = draft_model.loaded_weights[0]
    assert draft_name == "lm_head.weight"
    assert torch.equal(loaded_draft_tensor, draft_tensor[:2])


def test_mx_load_weights_rejects_draft_weights_without_drafter():
    policy_model = RecordingModel()
    worker = _worker_with(types.SimpleNamespace(model=policy_model))

    with pytest.raises(RuntimeError, match="vLLM has no drafter"):
        worker._mx_load_weights([("draft.eagle_module.fc.weight", torch.ones(2, 2))])

    assert policy_model.loaded_weights == []
