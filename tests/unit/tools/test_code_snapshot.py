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
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "code_snapshot.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="code_snapshot.sh requires rsync"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    # The script resolves PROJECT_ROOT as its own parent dir, so it must live in the repo.
    (tmp_path / "tools").mkdir()
    shutil.copy(SCRIPT, tmp_path / "tools" / "code_snapshot.sh")
    (tmp_path / "ray.sub").write_text("sub\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def _snapshot(repo: Path, exp_name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "tools/code_snapshot.sh", exp_name],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_fresh_snapshot_then_reuse(repo):
    result = _snapshot(repo, "exp")
    assert result.returncode == 0, result.stderr
    snapshot = Path(result.stdout.strip())
    assert (snapshot / "ray.sub").is_file()
    assert (snapshot / ".snapshot_complete").is_file()

    (snapshot / "continue.sh").write_text("#!/bin/bash\n")
    result = _snapshot(repo, "exp")
    assert result.returncode == 0, result.stderr
    assert (snapshot / "continue.sh").is_file()


def test_unmarked_snapshot_is_repaired_without_losing_run_outputs(repo):
    # Mimics a snapshot from before the marker existed that tools/launch already used.
    snapshot = repo / "code_snapshots" / "exp"
    (snapshot / "logs").mkdir(parents=True)
    (snapshot / "logs" / "run.log").write_text("log\n")
    (snapshot / "continue.sh").write_text("#!/bin/bash\n")

    result = _snapshot(repo, "exp")
    assert result.returncode == 0, result.stderr
    assert (snapshot / "ray.sub").is_file()
    assert (snapshot / ".snapshot_complete").is_file()
    assert (snapshot / "logs" / "run.log").read_text() == "log\n"
    assert (snapshot / "continue.sh").is_file()


@pytest.mark.parametrize(
    "exp_name",
    [
        "",
        ".",
        "..",
        "./exp",
        "exp/.",
        "../exp",
        "exp/..",
        "team/../../exp",
        "team/./exp",
        "/abs/exp",
        "exp\nother",
    ],
)
def test_invalid_experiment_name_is_rejected(repo, exp_name):
    other = repo / "code_snapshots" / "other_exp"
    other.mkdir(parents=True)
    (other / "continue.sh").write_text("#!/bin/bash\n")

    result = _snapshot(repo, exp_name)
    assert result.returncode != 0
    assert "Invalid experiment name" in result.stderr
    assert (other / "continue.sh").is_file()
    assert not (repo / ".snapshot_complete").exists()
    assert not (repo / "exp").exists()


def test_nested_experiment_name_is_allowed(repo):
    result = _snapshot(repo, "team/run1")
    assert result.returncode == 0, result.stderr
    snapshot = repo / "code_snapshots" / "team" / "run1"
    assert Path(result.stdout.strip()).resolve() == snapshot.resolve()
    assert (snapshot / "ray.sub").is_file()
    assert (snapshot / ".snapshot_complete").is_file()
