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

"""Tests for ray.sub's node-scoping logic, including the dedicated Ray head.

``ray.sub`` cannot be run end to end without Slurm, so these tests execute only
its configuration prologue (everything before the first ``srun``) against stub
``scontrol``/``sinfo`` binaries and assert the derived topology and per-component
Slurm identity.
"""

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
RAY_SUB = REPO_ROOT / "ray.sub"

# `scontrol show hostnames` expansion, per-node CPU counts (the head node type
# deliberately differs from the workers), and per-component job fields.
STUB_SCONTROL = """\
#!/bin/bash
set -euo pipefail
expand() {
  local item
  IFS=',' read -ra items <<< "$1"
  for item in "${items[@]}"; do
    if [[ "$item" =~ ^([a-z]+)\\[([0-9]+)-([0-9]+)\\]$ ]]; then
      local prefix="${BASH_REMATCH[1]}" lo="${BASH_REMATCH[2]}" hi="${BASH_REMATCH[3]}" i
      for (( i = 10#$lo; i <= 10#$hi; i++ )); do printf '%s%02d\\n' "$prefix" "$i"; done
    else
      printf '%s\\n' "$item"
    fi
  done
}
case "$1 $2" in
  "show hostnames") expand "$3" ;;
  "show node")
    if [[ "$3" == head* ]]; then echo "NodeName=$3 CPUTot=32 State=IDLE"
    else echo "NodeName=$3 CPUTot=224 State=IDLE"; fi ;;
  "show job")
    case "$3" in
      *"+0") echo "JobId=$3 Partition=${STUB_HEAD_PARTITION:-cpu} Account=acct WorkDir=/work" ;;
      *)     echo "JobId=$3 Partition=${STUB_WORKER_PARTITION:-batch} Account=acct WorkDir=/work" ;;
    esac ;;
  *) echo "scontrol stub: unhandled '$*'" >&2; exit 1 ;;
esac
"""

# The GPU partition reports GRES; the CPU partition does not.
STUB_SINFO = """\
#!/bin/bash
set -euo pipefail
partition=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) partition="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case "$partition" in
  cpu) echo "(null)" ;;
  *)   echo "gpu:8(S:0-1)" ;;
esac
"""

# Variables the prologue derives that these tests assert on.
_CAPTURED = [
    "HET_HEAD",
    "NUM_WORKER_NODES",
    "NUM_CLUSTER_NODES",
    "NUM_ACTORS",
    "HEAD_PARTITION",
    "WORKER_PARTITION",
    "HEAD_GRES_ARG",
    "WORKER_GRES_ARG",
    "CPUS_PER_WORKER",
    "CPUS_PER_HEAD",
    "HEAD_SRUN_ARGS",
    "WORKER_SRUN_ARGS",
    "HEAD_NUM_GPUS_ARG",
    "head_node",
]


@pytest.fixture(scope="module")
def prologue(tmp_path_factory) -> Path:
    """ray.sub truncated just before its first srun, with `set -x` removed."""
    lines = RAY_SUB.read_text().splitlines()
    cut = next(i for i, line in enumerate(lines) if "Optional sandbox sidecar" in line)
    body = "\n".join(lines[: cut - 1]).replace(
        "set -eoux pipefail", "set -eou pipefail"
    )
    path = tmp_path_factory.mktemp("prologue") / "prologue.sh"
    path.write_text(body)
    return path


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("slurm_stubs")
    for name, body in (("scontrol", STUB_SCONTROL), ("sinfo", STUB_SINFO)):
        stub = path / name
        stub.write_text(body)
        stub.chmod(0o755)
    return path


def _run(prologue: Path, stub_dir: Path, tmp_path: Path, env: dict[str, str]):
    """Source the prologue with `env` and return (CompletedProcess, vars)."""
    full_env = {
        "PATH": f"{stub_dir}:/usr/bin:/bin",
        "BASE_LOG_DIR": str(tmp_path / "logs"),
        "CONTAINER": "img.sqsh",
        "MOUNTS": "/x:/x",
        "COMMAND": "echo hi",
        "SLURM_JOB_ID": "42",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "SLURM_JOB_ACCOUNT": "acct",
        "SLURM_JOB_PARTITION": "batch",
        **env,
    }
    script = textwrap.dedent(f"""
        source {prologue} >/dev/null 2>&1 || exit $?
        for name in {" ".join(_CAPTURED)}; do
          printf '%s=%s\\n' "$name" "${{!name}}"
        done
        printf 'nodes_array=%s\\n' "${{nodes_array[*]}}"
    """)
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=full_env
    )
    parsed = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    return proc, parsed


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to exercise ray.sub"
)


def test_flat_allocation_is_unchanged(prologue, stub_dir, tmp_path):
    """Without DEDICATED_RAY_HEAD the head is still a compute node."""
    proc, v = _run(
        prologue,
        stub_dir,
        tmp_path,
        {"SLURM_JOB_NODELIST": "node[01-04]", "SLURM_JOB_NUM_NODES": "4"},
    )
    assert proc.returncode == 0, proc.stderr

    assert v["HET_HEAD"] == "0"
    assert v["NUM_WORKER_NODES"] == "3"
    assert v["NUM_CLUSTER_NODES"] == "4"
    # The head contributes GPUs, so all four nodes count toward worker units.
    assert v["NUM_ACTORS"] == "32"
    assert v["head_node"] == "node01"
    assert v["HEAD_GRES_ARG"] == v["WORKER_GRES_ARG"] == "--gres=gpu:8"
    assert v["CPUS_PER_HEAD"] == v["CPUS_PER_WORKER"] == "224"
    assert v["HEAD_NUM_GPUS_ARG"] == ""
    assert "--het-group" not in v["HEAD_SRUN_ARGS"]
    assert "--het-group" not in v["WORKER_SRUN_ARGS"]


def test_dedicated_head_scopes_each_component(prologue, stub_dir, tmp_path):
    """The head and workers get their own nodelist, partition, GRES, and CPUs."""
    proc, v = _run(
        prologue,
        stub_dir,
        tmp_path,
        {
            "DEDICATED_RAY_HEAD": "1",
            "SLURM_HET_SIZE": "2",
            "SLURM_JOB_NODELIST_HET_GROUP_0": "head01",
            "SLURM_JOB_NODELIST_HET_GROUP_1": "node[01-08]",
            "SLURM_JOB_NODELIST": "head01,node[01-08]",
            "SLURM_JOB_NUM_NODES": "9",
            "STUB_HEAD_PARTITION": "cpu",
            "STUB_WORKER_PARTITION": "batch",
        },
    )
    assert proc.returncode == 0, proc.stderr

    assert v["HET_HEAD"] == "1"
    assert v["NUM_WORKER_NODES"] == "8"
    assert v["NUM_CLUSTER_NODES"] == "9"
    # The head advertises no GPUs, so it must not inflate the expected count.
    assert v["NUM_ACTORS"] == "64"
    assert v["head_node"] == "head01"
    assert v["nodes_array"].split()[1] == "node01"

    # Per-component identity: a CPU-only head beside GPU workers.
    assert v["HEAD_PARTITION"] == "cpu"
    assert v["WORKER_PARTITION"] == "batch"
    assert v["HEAD_GRES_ARG"] == ""
    assert v["WORKER_GRES_ARG"] == "--gres=gpu:8"
    assert v["CPUS_PER_HEAD"] == "32"
    assert v["CPUS_PER_WORKER"] == "224"

    assert v["HEAD_NUM_GPUS_ARG"] == "--num-gpus=0"
    assert "--het-group=0" in v["HEAD_SRUN_ARGS"]
    assert "--het-group=1" in v["WORKER_SRUN_ARGS"]
    assert "-p cpu" in v["HEAD_SRUN_ARGS"]
    assert "-p batch" in v["WORKER_SRUN_ARGS"]


def test_dedicated_head_stays_first_regardless_of_hostname(
    prologue, stub_dir, tmp_path
):
    """The head is fixed by the allocation, so the hostname sort must not move it."""
    proc, v = _run(
        prologue,
        stub_dir,
        tmp_path,
        {
            "DEDICATED_RAY_HEAD": "1",
            "SLURM_HET_SIZE": "2",
            # Sorts after every worker; the flat path would have picked node01.
            "SLURM_JOB_NODELIST_HET_GROUP_0": "zzz01",
            "SLURM_JOB_NODELIST_HET_GROUP_1": "node[01-04]",
            "SLURM_JOB_NODELIST": "zzz01,node[01-04]",
            "SLURM_JOB_NUM_NODES": "5",
            "STUB_HEAD_PARTITION": "cpu",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert v["head_node"] == "zzz01"
    assert v["nodes_array"] == "zzz01 node01 node02 node03 node04"


@pytest.mark.parametrize(
    "env,expected",
    [
        pytest.param(
            {
                "DEDICATED_RAY_HEAD": "1",
                "SLURM_JOB_NODELIST": "node[01-04]",
                "SLURM_JOB_NUM_NODES": "4",
            },
            "requires a heterogeneous job",
            id="not_a_het_job",
        ),
        pytest.param(
            {
                "DEDICATED_RAY_HEAD": "yes",
                "SLURM_JOB_NODELIST": "node[01-04]",
                "SLURM_JOB_NUM_NODES": "4",
            },
            "must be 0 or 1",
            id="invalid_value",
        ),
        pytest.param(
            {
                "DEDICATED_RAY_HEAD": "1",
                "SLURM_HET_SIZE": "2",
                "SLURM_JOB_NODELIST_HET_GROUP_0": "head[01-02]",
                "SLURM_JOB_NODELIST_HET_GROUP_1": "node[01-04]",
                "SLURM_JOB_NODELIST": "head[01-02],node[01-04]",
                "SLURM_JOB_NUM_NODES": "6",
            },
            "exactly one node",
            id="multi_node_head_component",
        ),
    ],
)
def test_dedicated_head_misconfiguration_fails_fast(
    prologue, stub_dir, tmp_path, env, expected
):
    """Misconfiguration must abort before any srun, with an actionable message."""
    full_env = {
        "PATH": f"{stub_dir}:/usr/bin:/bin",
        "BASE_LOG_DIR": str(tmp_path / "logs"),
        "CONTAINER": "img.sqsh",
        "MOUNTS": "/x:/x",
        "COMMAND": "echo hi",
        "SLURM_JOB_ID": "42",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "SLURM_JOB_ACCOUNT": "acct",
        "SLURM_JOB_PARTITION": "batch",
        **env,
    }
    proc = subprocess.run(
        ["bash", str(prologue)], capture_output=True, text=True, env=full_env
    )
    assert proc.returncode != 0
    assert expected in proc.stdout + proc.stderr
