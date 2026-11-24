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
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nemo_rl.utils import compile_cache


class _Group:
    """Object collectives between threads, one thread per rank."""

    def __init__(self, world_size: int):
        self._slots: list = [None] * world_size
        self._barrier = threading.Barrier(world_size, timeout=30)

    def member(self, rank: int):
        group = self

        class Member:
            @staticmethod
            def all_gather(obj):
                group._slots[rank] = obj
                group._barrier.wait()
                out = list(group._slots)
                group._barrier.wait()
                return out

            @staticmethod
            def broadcast(obj):
                return Member.all_gather(obj)[0]

        return Member()


def _ranks(tmp_path: Path, world_size: int) -> list[compile_cache.ShapeProfiles]:
    group = _Group(world_size)
    return [
        compile_cache.ShapeProfiles(
            tmp_path / f"local/{rank}/code_state.pkl",
            tmp_path / "sets/key",
            rank,
            world_size,
            group.member(rank),
        )
        for rank in range(world_size)
    ]


def _on_every_rank(ranks, call) -> None:
    with ThreadPoolExecutor(len(ranks)) as pool:
        list(pool.map(call, ranks))


def _put_set(sets: Path, name: str, profiles: dict, world_size: int, age_s=0) -> Path:
    set_dir = sets / name
    set_dir.mkdir(parents=True)
    for rank, text in profiles.items():
        (set_dir / f"{rank}.pkl").write_text(text)
    manifest = set_dir / "manifest.json"
    manifest.write_text(json.dumps({"world_size": world_size, "ranks": list(profiles)}))
    t = time.time() - age_s
    os.utime(manifest, (t, t))
    return set_dir


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


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
    assert compile_cache.type_dir("vllm") == f"{caches}/shared/vllm"


def test_autotune_and_job_id_reach_only_their_cache_type(caches, monkeypatch):
    monkeypatch.setenv("NRL_MEGATRON_TRITON_CACHE_AUTOTUNING", "1")
    monkeypatch.setenv("NRL_MEGATRON_TORCH_COMPILE_JOB_ID", "my-rl-run")
    monkeypatch.setenv("NRL_MEGATRON_TRITON_PRINT_AUTOTUNING", "")
    monkeypatch.delenv("NRL_VLLM_TRITON_CACHE_AUTOTUNING", raising=False)

    megatron = compile_cache.env_overrides("megatron")
    vllm = compile_cache.env_overrides("vllm")

    assert megatron["TRITON_CACHE_AUTOTUNING"] == "1"
    assert megatron["TORCH_COMPILE_JOB_ID"] == "my-rl-run"
    # An empty value, as VAR="${VAR:-}" in a launch script exports, means unset.
    assert "TRITON_PRINT_AUTOTUNING" not in megatron
    assert "TRITON_CACHE_AUTOTUNING" not in vllm


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


def test_deterministic_runs_warn_when_autotune_results_are_shared(
    caches, monkeypatch, caplog
):
    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "1")
    with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
        compile_cache.warn_if_autotuning_is_shared(False)
        assert not caplog.records
        compile_cache.warn_if_autotuning_is_shared(True)
    assert "whichever job tuned them first" in caplog.text


def test_shape_profile_key_changes_with_the_layout():
    key = compile_cache.shape_profile_key("run", {"tp": 2, "world_size": 8})
    assert key.startswith("run-")
    assert key == compile_cache.shape_profile_key("run", {"world_size": 8, "tp": 2})
    assert key != compile_cache.shape_profile_key("run", {"tp": 4, "world_size": 8})


def test_shape_profiles_are_off_without_a_job_id(caches, monkeypatch):
    monkeypatch.delenv("TORCH_COMPILE_JOB_ID", raising=False)
    assert compile_cache.ShapeProfiles.load("megatron", {}) is None


_COMPILE_ON_ANOTHER_THREAD = """
import sys
import threading

import torch
import torch.distributed as dist

from nemo_rl.utils import compile_cache

dist.init_process_group("gloo", init_method=sys.argv[1], rank=0, world_size=1)
profiles = compile_cache.ShapeProfiles.load("megatron", {})


@torch.compile(backend="eager")
def double(x):
    return x * 2


def step():
    for n in (2, 3):
        double(torch.ones(n))


thread = threading.Thread(target=step)
thread.start()
thread.join()
profiles.step_done(True)
dist.destroy_process_group()
"""


def test_a_profile_torch_writes_on_another_thread_is_saved(caches, monkeypatch):
    """Ray runs a worker's steps on another thread than the one that loads."""
    monkeypatch.setenv("NRL_MEGATRON_TORCH_COMPILE_JOB_ID", "run")
    result = subprocess.run(
        [sys.executable, "-c", _COMPILE_ON_ANOTHER_THREAD, f"file://{caches}/pg"],
        capture_output=True,
        text=True,
        env={**os.environ, **compile_cache.env_overrides("megatron")},
        cwd=Path(compile_cache.__file__).parents[2],
    )

    assert result.returncode == 0, result.stderr
    [set_dir] = (caches / "shared/megatron_shape_profiles").glob("*/*")
    assert json.loads((set_dir / "manifest.json").read_text())["ranks"] == [0]


def test_shape_profiles_load_the_newest_complete_set(tmp_path):
    ranks = _ranks(tmp_path, 2)
    sets = tmp_path / "sets/key"
    _put_set(sets, "old", {0: "old 0", 1: "old 1"}, 2, age_s=600)
    _put_set(sets, "new", {0: "new 0"}, 2, age_s=300)
    _put_set(sets, "other_world_size", {0: "x", 1: "x", 2: "x"}, 3)
    _write(sets / "unfinished/0.pkl", "no manifest yet")
    _write(ranks[1].path, "left by an earlier job")

    _on_every_rank(ranks, lambda p: p.load_newest())

    assert ranks[0].path.read_text() == "new 0"
    assert not ranks[1].path.exists()


def test_a_rank_that_cannot_load_drops_the_set_on_every_rank(tmp_path):
    ranks = _ranks(tmp_path, 2)
    set_dir = _put_set(tmp_path / "sets/key", "s", {0: "p0", 1: "p1"}, 2)
    (set_dir / "1.pkl").unlink()

    _on_every_rank(ranks, lambda p: p.load_newest())

    assert not ranks[0].path.exists()
    assert not ranks[1].path.exists()


def test_shape_profiles_are_saved_as_one_set_after_steps_1_2_4(tmp_path):
    ranks = _ranks(tmp_path, 2)
    sets = tmp_path / "sets/key"
    _on_every_rank(ranks, lambda p: p.load_newest())

    _write(ranks[0].path, "rank 0 after step 1")
    _on_every_rank(ranks, lambda p: p.step_done(True))
    [first] = sets.iterdir()
    assert json.loads((first / "manifest.json").read_text())["ranks"] == [0]
    assert (first / "0.pkl").read_text() == "rank 0 after step 1"

    _on_every_rank(ranks, lambda p: p.step_done(True))  # step 2: nothing changed
    _write(ranks[1].path, "rank 1 after step 3")
    _on_every_rank(ranks, lambda p: p.step_done(True))  # step 3: no save
    assert len(list(sets.iterdir())) == 1

    _on_every_rank(ranks, lambda p: p.step_done(True))
    [second] = [d for d in sets.iterdir() if d != first]
    assert json.loads((second / "manifest.json").read_text())["ranks"] == [0, 1]
    assert (second / "1.pkl").read_text() == "rank 1 after step 3"


def test_no_set_is_saved_after_a_step_that_failed_on_any_rank(tmp_path):
    ranks = _ranks(tmp_path, 2)
    for p in ranks:
        _write(p.path, f"rank {p.rank}")

    _on_every_rank(ranks, lambda p: p.step_done(p.rank == 0))

    assert not (tmp_path / "sets").exists()
