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
import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "tools/external_gym_vllm/cache_sync.sh"
PROFILE = "inductor/dynamo/code_state_job_0_.pkl"


def _run(*args: object, env: dict[str, str] | None = None) -> None:
    subprocess.run(["bash", str(SCRIPT), *map(str, args)], check=True, env=env)


def _set_age(path: Path, minutes: float) -> None:
    t = time.time() - minutes * 60
    os.utime(path, (t, t))


def _put(path: Path, text: str, minutes: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if minutes:
        _set_age(path, minutes)


def test_seed_then_writeback_publishes_only_new_settled_files(tmp_path: Path):
    persistent, local = tmp_path / "persistent", tmp_path / "local"
    (persistent / "a").mkdir(parents=True)
    (persistent / "a/old").write_text("persistent")

    _run("seed", persistent, local)
    assert (local / "a/old").read_text() == "persistent"
    _set_age(tmp_path / "local.seeded", 10)

    (local / "a/old").write_text("changed locally")  # exists: never replaced
    (local / "a/new").write_text("compiled")  # new, settled: published
    (local / "a/busy").write_text("being written")  # new, < 1 min old: skipped
    _set_age(local / "a/old", 2)
    _set_age(local / "a/new", 2)
    _run("writeback", local, persistent)

    assert (persistent / "a/old").read_text() == "persistent"
    assert (persistent / "a/new").read_text() == "compiled"
    assert not (persistent / "a/busy").exists()
    assert not (tmp_path / "persistent.lock").exists()


def test_seed_once_copies_listed_subdirs_first_and_keeps_rewritten_profiles(
    tmp_path: Path,
):
    persistent, local = tmp_path / "persistent/megatron", tmp_path / "local/megatron"
    _put(persistent / PROFILE, "published profile")
    _put(persistent / "inductor/fxgraph/entry", "graph")
    # Notes what is in place when the full copy starts, then rewrites the profile
    # like a rank that compiles once .seeded-first exists.
    bin_dir, seen = tmp_path / "bin", tmp_path / "seen"
    bin_dir.mkdir()
    fake_rsync = bin_dir / "rsync"
    fake_rsync.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            if [[ "${{@: -2:1}}" == "{persistent}/" ]]; then
              [[ -f "{local}/{PROFILE}" ]] && echo profile >>"{seen}"
              [[ -e "{local}.seeded-first" ]] && echo marker >>"{seen}"
              echo rewritten >"{local}/{PROFILE}"
            fi
            exec {shutil.which("rsync")} "$@"
            """
        )
    )
    fake_rsync.chmod(0o755)

    _run(
        "seed-once",
        persistent,
        local,
        "inductor/dynamo",
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert seen.read_text().split() == ["profile", "marker"]
    assert (local / PROFILE).read_text() == "rewritten\n"
    assert (local / "inductor/fxgraph/entry").read_text() == "graph"
    assert Path(f"{local}.seeded").exists()


def test_writeback_replaces_only_shape_profiles_this_node_rewrote(tmp_path: Path):
    persistent, local = tmp_path / "persistent/megatron", tmp_path / "local/megatron"
    other_rank = "inductor/dynamo/code_state_job_1_.pkl"
    raced_rank = "inductor/dynamo/code_state_job_2_.pkl"
    # Published by earlier jobs; this node's seed copied them with the same mtimes.
    for rel, text in [
        (PROFILE, "old profile"),
        (other_rank, "old profile"),
        ("inductor/fxgraph/entry", "old graph"),
    ]:
        _put(persistent / rel, text, minutes=60)
        _put(local / rel, text, minutes=60)
    _put(Path(f"{local}.seeded-first"), "", minutes=30)
    _put(Path(f"{local}.seeded"), "", minutes=29)
    # Rank 0 rewrote its profile just now: profiles have no age rule.
    _put(local / PROFILE, "new profile")
    # Published entries are never replaced, even when the local copy changed.
    _put(local / "inductor/fxgraph/entry", "new graph", minutes=10)
    # A later job published rank 2's profile after this node rewrote it.
    _put(local / raced_rank, "this job", minutes=5)
    _put(persistent / raced_rank, "later job", minutes=1)
    # New entries, one still inside the 1-minute age rule.
    _put(local / "triton/kernel.cubin", "kernel", minutes=10)
    _put(local / "triton/fresh.cubin", "fresh")

    _run("writeback", local, persistent)

    assert (persistent / PROFILE).read_text() == "new profile"
    assert (persistent / other_rank).read_text() == "old profile"
    assert (persistent / raced_rank).read_text() == "later job"
    assert (persistent / "inductor/fxgraph/entry").read_text() == "old graph"
    assert (persistent / "triton/kernel.cubin").read_text() == "kernel"
    assert not (persistent / "triton/fresh.cubin").exists()
    assert not (tmp_path / "persistent/megatron.lock").exists()
