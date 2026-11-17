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
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "nemo_rl/utils/cache_sync.sh"


def _run(*args: object, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def _set_age(path: Path, minutes: float) -> None:
    t = time.time() - minutes * 60
    os.utime(path, (t, t))


def _fake_rsync(tmp_path: Path, body: str) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "rsync").write_text(f"#!/bin/bash\n{body}\n")
    (bin_dir / "rsync").chmod(0o755)
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}"}


def test_writeback_publishes_new_settled_files_once(tmp_path: Path):
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

    # Sent files are not sent again; a file that was too young is, once settled.
    (persistent / "a/new").unlink()
    _set_age(local / "a/busy", 0.5)
    _run("writeback", local, persistent, env={"NRL_COMPILE_CACHE_MIN_AGE_MIN": "0"})
    assert not (persistent / "a/new").exists()
    assert (persistent / "a/busy").read_text() == "being written"


def test_seed_marks_the_start_of_the_copy(tmp_path: Path):
    (tmp_path / "persistent").mkdir()
    env = _fake_rsync(tmp_path, f'sleep 2; exec {shutil.which("rsync")} "$@"')
    start = time.time()
    _run("seed", tmp_path / "persistent", tmp_path / "local", env=env)
    # A file compiled during the copy is newer than the mark, so it is published.
    assert (tmp_path / "local.seeded").stat().st_mtime < start + 1


def test_failed_seed_warns_and_continues(tmp_path: Path):
    (tmp_path / "persistent").mkdir()
    env = _fake_rsync(tmp_path, "exit 23")
    out = _run("seed", tmp_path / "persistent", tmp_path / "local", env=env)
    assert "WARNING" in out.stdout
    assert (tmp_path / "local.seeded").exists()


def test_seed_type_and_vllm_engine_merge(tmp_path: Path):
    env = {
        "NRL_COMPILE_CACHE_SHARED_DIR": str(tmp_path / "shared"),
        "NRL_COMPILE_CACHE_LOCAL_DIR": str(tmp_path / "local"),
        "NRL_COMPILE_CACHE_FORCE_FRESH": "1",
    }
    _run("seed-type", "vllm", env=env)
    assert (tmp_path / "local/vllm.seeded").exists()
    engine = tmp_path / "local/vllm_engines/0/torch_compile_cache"
    engine.mkdir(parents=True)
    (engine / "entry").write_text("compiled")
    _set_age(engine / "entry", 2)
    stop = tmp_path / "ENDED"
    stop.touch()

    # Merged with a new time, so the pass after the merge publishes it.
    _run("sidecar-types", 0, stop, "vllm", env=env)
    merged = tmp_path / "local/vllm/vllm/torch_compile_cache/entry"
    assert merged.read_text() == "compiled"
    _set_age(merged, 0.5)  # newer than the first pass's mark
    _run(
        "sidecar-types",
        0,
        stop,
        "vllm",
        env={**env, "NRL_COMPILE_CACHE_MIN_AGE_MIN": "0"},
    )
    published = tmp_path / "shared/vllm/vllm/torch_compile_cache/entry"
    assert published.read_text() == "compiled"


def test_missing_flock_fails(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("rsync", "hostname"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), "seed-type", "vllm"],
        capture_output=True,
        text=True,
        env={"PATH": str(bin_dir)},
    )
    assert result.returncode == 1
    assert "flock is required" in result.stdout
