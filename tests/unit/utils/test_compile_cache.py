# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from pathlib import Path

import pytest

from nemo_rl.utils import compile_cache

SCRIPT = Path(__file__).parents[3] / "tools/external_gym_vllm/cache_sync.sh"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NRL_VLLM_TRITON_CACHE_DIR", "/local/policy/triton")
    monkeypatch.setenv("NRL_MEGATRON_TRITON_CACHE_DIR", "/local/megatron/triton")
    assert compile_cache.env_overrides("VLLM") == {
        "TRITON_CACHE_DIR": "/local/policy/triton"
    }


def test_seed_is_noop_unless_configured_and_rejects_partial_config(monkeypatch):
    for var in compile_cache._SEED_VARS:
        monkeypatch.delenv(var, raising=False)
    compile_cache.seed("policy")
    monkeypatch.setenv("NRL_CACHE_SYNC", str(SCRIPT))
    with pytest.raises(ValueError):
        compile_cache.seed("policy")


def test_seed_copies_once_per_node(tmp_path, monkeypatch):
    (tmp_path / "persistent/policy").mkdir(parents=True)
    (tmp_path / "persistent/policy/kernel").write_text("compiled")
    monkeypatch.setenv("NRL_CACHE_SYNC", str(SCRIPT))
    monkeypatch.setenv("NRL_COMPILE_CACHE_ROOT", str(tmp_path / "persistent"))
    monkeypatch.setenv("NRL_LOCAL_COMPILE_CACHE", str(tmp_path / "local"))

    compile_cache.seed("policy")
    local = tmp_path / "local/policy/kernel"
    assert local.read_text() == "compiled"
    local.write_text("changed here")
    compile_cache.seed("policy")  # already seeded on this node: no copy
    assert local.read_text() == "changed here"
