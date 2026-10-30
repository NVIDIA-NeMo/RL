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
"""tools/launch submits a heterogeneous job when a test asks for external vLLM pools."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="tools/code_snapshot.sh requires rsync"
)

CONFIG = """#!/bin/bash
# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=1
NUM_MINUTES=30
{extra}
# ===== END CONFIG =====
"""

POOLS = """register_external_vllm_pool JUDGE \\
  --model "$HF_HOME/judge" \\
  --container /shared/judge.sqsh \\
  --python /usr/bin/python \\
  --replicas 1 \\
  --tensor-parallel-size 4 \\
  --lb-port 9300 \\
  --url-placeholder __JUDGE_BASE_URL__
EXTERNAL_VLLM_SCRIPT_ARGS="++env.judge.base_url=__JUDGE_BASE_URL__"
"""

# Records what tools/launch hands to sbatch.
FAKE_SBATCH = """#!/bin/bash
printf '%s\\n' "$@" > "$SBATCH_RECORD.args"
printf '%s\\n' "$COMMAND" > "$SBATCH_RECORD.command"
printf '%s\\n' "${EXTERNAL_VLLM_POOLS:-}" > "$SBATCH_RECORD.pools"
echo "Submitted batch job 4242"
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    # tools/launch resolves PROJECT_ROOT as its own parent dir, so it must live in the repo.
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    for name in ("launch", "code_snapshot.sh"):
        shutil.copy(ROOT / "tools" / name, repo / "tools" / name)
    shutil.copytree(
        ROOT / "tools" / "external_gym_vllm", repo / "tools" / "external_gym_vllm"
    )
    (repo / "ray.sub").write_text("sub\n")
    (repo / "tests").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(FAKE_SBATCH)
    (bin_dir / "sbatch").chmod(0o755)
    return repo


def _launch(repo: Path, script: str, dryrun: str = "") -> subprocess.CompletedProcess:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"],
        cwd=repo,
        check=True,
    )
    env = {
        **os.environ,
        "PATH": f"{repo.parent / 'bin'}:{os.environ['PATH']}",
        "SBATCH_RECORD": str(repo.parent / "sbatch"),
        "HF_HOME": str(repo.parent / "hf_home"),
        "HF_DATASETS_CACHE": "/shared/hf_datasets",
        "CONTAINER": "/shared/rl.sqsh",
        "ACCOUNT": "acct",
        "PARTITION": "batch",
        # The snapshot and HF_HOME stand in for the shared filesystem.
        "EXTERNAL_VLLM_SHARED_ROOT": str(repo.parent),
        "DRYRUN": dryrun,
    }
    return subprocess.run(
        ["bash", "tools/launch", script],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_external_pools_submit_a_heterogeneous_job(repo):
    (repo / "tests" / "judge.sh").write_text(
        CONFIG.format(extra="NUM_EXTERNAL_SERVICE_NODES=1")
    )
    (repo / "tests" / "judge.pools").write_text(POOLS)

    result = _launch(repo, "tests/judge.sh")
    assert result.returncode == 0, result.stderr

    record = repo.parent / "sbatch"
    args = (record.with_suffix(".args")).read_text().split("\n")
    assert args[args.index(":") + 1] == "--nodes=1"
    assert args[-2] == "tools/external_gym_vllm/run_in_allocation.sh"
    assert (record.with_suffix(".pools")).read_text().strip() == "JUDGE"
    command = (record.with_suffix(".command")).read_text()
    assert "++env.judge.base_url=__JUDGE_BASE_URL__" in command
    assert command.strip().endswith("__EXTERNAL_SERVICE_READINESS__")
    # GPU hours count the external nodes too: 1 run x 3 nodes x 4 GPUs x 0.5 h.
    assert "[INFO]: 6 GPUhrs" in result.stdout


def test_without_external_pools_launch_submits_ray_sub(repo):
    (repo / "tests" / "plain.sh").write_text(CONFIG.format(extra=""))

    result = _launch(repo, "tests/plain.sh")
    assert result.returncode == 0, result.stderr

    record = repo.parent / "sbatch"
    args = (record.with_suffix(".args")).read_text().split("\n")
    assert ":" not in args
    assert args[-2] == "ray.sub"
    assert (
        (record.with_suffix(".command"))
        .read_text()
        .startswith("uv run tests/plain.sh ")
    )


def test_external_pools_need_a_pools_file(repo):
    (repo / "tests" / "judge.sh").write_text(
        CONFIG.format(extra="NUM_EXTERNAL_SERVICE_NODES=1")
    )

    result = _launch(repo, "tests/judge.sh", dryrun="1")
    assert result.returncode == 1
    assert "needs pool definitions in tests/judge.pools" in result.stderr
