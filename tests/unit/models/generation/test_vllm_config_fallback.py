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
"""Config loading on the vLLM side for model types Transformers does not register."""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.interfaces import get_num_routed_experts
from nemo_rl.models.generation.vllm.worker_utils import load_hf_config_for_vllm


def _write_config(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(tmp_path)


@pytest.fixture
def fake_vllm_get_config(monkeypatch):
    calls = []

    def get_config(model, trust_remote_code=False, **_):
        calls.append((model, trust_remote_code))
        return SimpleNamespace(model_type="from_vllm", moe_num_experts=352)

    mod = types.ModuleType("vllm.transformers_utils.config")
    mod.get_config = get_config
    for name in ("vllm", "vllm.transformers_utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.config", mod)
    return calls


def test_unregistered_model_type_falls_back_to_vllm(tmp_path, fake_vllm_get_config):
    path = _write_config(tmp_path, "nemo_rl_test_unregistered_type")
    cfg = load_hf_config_for_vllm(path, trust_remote_code=True)
    assert cfg.model_type == "from_vllm"
    assert fake_vllm_get_config == [(path, True)]


def test_registered_model_type_uses_transformers(tmp_path, fake_vllm_get_config):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 64, "num_attention_heads": 4})
    )
    cfg = load_hf_config_for_vllm(str(tmp_path), trust_remote_code=True)
    assert cfg.model_type == "llama"
    assert fake_vllm_get_config == []


def test_other_errors_are_not_swallowed(tmp_path, fake_vllm_get_config):
    with pytest.raises(OSError):
        load_hf_config_for_vllm(str(tmp_path / "missing"), trust_remote_code=True)
    assert fake_vllm_get_config == []


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (SimpleNamespace(moe_num_experts=352), 352),
        (SimpleNamespace(text_config=SimpleNamespace(moe_num_experts=288)), 288),
        (SimpleNamespace(num_experts=8, moe_num_experts=352), 8),
        (SimpleNamespace(hidden_size=64), None),
    ],
)
def test_get_num_routed_experts_reads_moe_num_experts(cfg, expected):
    assert get_num_routed_experts(cfg) == expected
