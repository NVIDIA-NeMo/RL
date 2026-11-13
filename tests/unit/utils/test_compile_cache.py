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
import pytest

from nemo_rl.utils import compile_cache


@pytest.fixture
def caches(tmp_path, monkeypatch):
    for var in ("NRL_COMPILE_CACHE_VLLM_DIR", "NRL_COMPILE_CACHE_FORCE_FRESH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(compile_cache.SHARED_VAR, str(tmp_path / "shared"))
    monkeypatch.setenv(compile_cache.LOCAL_VAR, str(tmp_path / "local"))
    return tmp_path


def test_off_unless_configured_and_rejects_half_config(monkeypatch):
    monkeypatch.delenv(compile_cache.SHARED_VAR, raising=False)
    monkeypatch.delenv(compile_cache.LOCAL_VAR, raising=False)
    assert compile_cache.env_overrides("vllm") == {}
    compile_cache.seed("vllm")  # no-op
    monkeypatch.setenv(compile_cache.LOCAL_VAR, "/local")
    with pytest.raises(ValueError):
        compile_cache.seed("vllm")


def test_env_overrides_default_under_the_local_dir(caches):
    env = compile_cache.env_overrides("megatron")
    assert env["TRITON_CACHE_DIR"] == f"{caches}/local/megatron/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == f"{caches}/local/megatron/inductor"
    assert compile_cache.vllm_engine_dir(3) == f"{caches}/local/vllm_engines/3"


def test_missing_script_raises(caches, monkeypatch):
    monkeypatch.setenv(compile_cache.SCRIPT_VAR, str(caches / "missing.sh"))
    with pytest.raises(FileNotFoundError):
        compile_cache.seed("megatron")


def test_seed_copies_once_per_node_from_the_type_dir(caches, monkeypatch):
    other = caches / "other_vllm"
    (other / "vllm").mkdir(parents=True)
    (other / "vllm/kernel").write_text("compiled")
    monkeypatch.setenv("NRL_COMPILE_CACHE_VLLM_DIR", str(other))

    compile_cache.seed("vllm")
    local = caches / "local/vllm/vllm/kernel"
    assert local.read_text() == "compiled"
    local.write_text("changed here")
    compile_cache.seed("vllm")  # already seeded on this node: no copy
    assert local.read_text() == "changed here"


def test_engine_folder_is_filled_from_the_node_copy(caches, monkeypatch):
    (caches / "shared/vllm/vllm/torch_compile_cache").mkdir(parents=True)
    (caches / "shared/vllm/vllm/torch_compile_cache/entry").write_text("compiled")
    engine = compile_cache.vllm_engine_dir(0)

    compile_cache._prepare_vllm_engine_on_this_node(engine)
    assert (caches / "local/vllm_engines/0/torch_compile_cache/entry").exists()
