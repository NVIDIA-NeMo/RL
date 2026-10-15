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
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "tools/external_gym_vllm/cache_sync.sh"


def _run(*args: object) -> None:
    subprocess.run(["bash", str(SCRIPT), *map(str, args)], check=True)


def _set_age(path: Path, minutes: float) -> None:
    t = time.time() - minutes * 60
    os.utime(path, (t, t))


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
