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
import threading
import time
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


def test_autotune_and_job_id_reach_only_the_prefixed_worker(monkeypatch):
    monkeypatch.setenv("NRL_MEGATRON_TRITON_CACHE_AUTOTUNING", "1")
    monkeypatch.setenv("NRL_MEGATRON_TORCH_COMPILE_JOB_ID", "my-rl-run")
    monkeypatch.setenv("NRL_MEGATRON_TRITON_PRINT_AUTOTUNING", "")
    monkeypatch.delenv("NRL_VLLM_TRITON_CACHE_AUTOTUNING", raising=False)

    megatron = compile_cache.env_overrides("MEGATRON")
    vllm = compile_cache.env_overrides("VLLM")

    assert megatron["TRITON_CACHE_AUTOTUNING"] == "1"
    assert megatron["TORCH_COMPILE_JOB_ID"] == "my-rl-run"
    # Empty means unset, like the launcher's ${VAR:-} defaults.
    assert "TRITON_PRINT_AUTOTUNING" not in megatron
    assert "TRITON_CACHE_AUTOTUNING" not in vllm


@pytest.fixture
def megatron_worker_env(monkeypatch, tmp_path):
    """The Megatron worker's cache env; seed() only records its commands."""
    monkeypatch.setenv("NRL_CACHE_SYNC", str(SCRIPT))
    monkeypatch.setenv("NRL_COMPILE_CACHE_ROOT", str(tmp_path / "shared"))
    monkeypatch.setenv("NRL_LOCAL_COMPILE_CACHE", str(tmp_path / "local"))
    monkeypatch.setenv(
        "TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "local/megatron/inductor")
    )
    monkeypatch.delenv("TORCH_COMPILE_JOB_ID", raising=False)
    monkeypatch.setattr(compile_cache, "_PROFILE_SEEDS", set())
    calls = []
    monkeypatch.setattr(
        compile_cache.subprocess, "Popen", lambda cmd, **kwargs: calls.append(cmd)
    )
    return calls


def test_seed_copies_shape_profiles_first_only_when_torch_keeps_them_in_the_cache(
    megatron_worker_env, monkeypatch, tmp_path
):
    seed_once = [
        "bash",
        str(SCRIPT),
        "seed-once",
        f"{tmp_path}/shared/megatron",
        f"{tmp_path}/local/megatron",
    ]

    compile_cache.seed("megatron", background=True)
    monkeypatch.setenv("TORCH_COMPILE_JOB_ID", "my-rl-run")
    compile_cache.seed("megatron", background=True)
    # Profiles kept outside the type's cache dir are not part of its copy.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "elsewhere"))
    compile_cache.seed("megatron", background=True)

    assert megatron_worker_env == [
        seed_once,
        seed_once + ["inductor/dynamo"],
        seed_once,
    ]
    assert compile_cache._PROFILE_SEEDS == {"megatron"}


def test_wait_for_shape_profiles_returns_once_seed_once_has_placed_them(
    megatron_worker_env, monkeypatch, tmp_path
):
    monkeypatch.setenv("TORCH_COMPILE_JOB_ID", "my-rl-run")
    compile_cache.seed("megatron", background=True)
    marker = tmp_path / "local/megatron.seeded-first"
    marker.parent.mkdir(parents=True)
    timer = threading.Timer(0.3, marker.touch)
    timer.start()

    start = time.monotonic()
    compile_cache.wait_for_shape_profiles("megatron", timeout_s=30)
    waited = time.monotonic() - start
    timer.join()

    assert 0.2 <= waited < 10


def test_wait_for_shape_profiles_returns_at_once_without_a_profile_seed(
    megatron_worker_env, monkeypatch
):
    compile_cache.seed("megatron", background=True)
    # A job id set after the seed, or a cache type that was never seeded.
    monkeypatch.setenv("TORCH_COMPILE_JOB_ID", "my-rl-run")

    start = time.monotonic()
    compile_cache.wait_for_shape_profiles("megatron", timeout_s=30)
    compile_cache.wait_for_shape_profiles("value", timeout_s=30)

    assert time.monotonic() - start < 1


def test_wait_for_shape_profiles_gives_up_after_the_timeout(
    megatron_worker_env, monkeypatch, capsys
):
    monkeypatch.setenv("TORCH_COMPILE_JOB_ID", "my-rl-run")
    # The copy never starts: Popen is only recorded.
    compile_cache.seed("megatron", background=True)

    start = time.monotonic()
    compile_cache.wait_for_shape_profiles("megatron", timeout_s=0.3)

    assert 0.3 <= time.monotonic() - start < 10
    assert "compiling without them" in capsys.readouterr().out
