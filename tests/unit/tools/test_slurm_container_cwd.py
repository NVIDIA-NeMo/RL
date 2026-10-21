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

"""Run the Slurm launcher with local command stubs and inspect its srun arguments."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("different_submit_directory", [False, True])
def test_container_uses_job_workdir(
    tmp_path: Path, different_submit_directory: bool
) -> None:
    repo = Path(__file__).resolve().parents[3]
    workdir = tmp_path / "job_checkout"
    submit_dir = tmp_path / "submit_checkout" if different_submit_directory else workdir
    workdir.mkdir()
    submit_dir.mkdir(exist_ok=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stubs = {
        "sinfo": "printf 'gpu:1\\n'\n",
        "scontrol": "printf 'test-node\\n'\n",
        # Stop before creating any container, Ray process, or training worker.
        "srun": 'printf "%s\\n" "$@" > "$TEST_SRUN_ARGS"\nexit 73\n',
    }
    for name, body in stubs.items():
        executable = binaries / name
        executable.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        executable.chmod(0o755)
    arguments = tmp_path / "srun_args.txt"
    env = {
        **os.environ,
        "PATH": f"{binaries}:/usr/bin:/bin",
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_ACCOUNT": "test-account",
        "SLURM_JOB_PARTITION": "test-partition",
        "SLURM_JOB_NODELIST": "test-node",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NNODES": "1",
        "SLURM_SUBMIT_DIR": str(submit_dir),
        "BASE_LOG_DIR": str(tmp_path / "logs"),
        "GPUS_PER_NODE": "1",
        "CPUS_PER_WORKER": "1",
        "DEDICATED_RAY_HEAD": "0",
        "CONTAINER": "unused-test-image.sqsh",
        "MOUNTS": "",
        "SETUP_COMMAND": "",
        "COMMAND": "exit 0",
        "TEST_SRUN_ARGS": str(arguments),
    }
    env.pop("SLURM_RESTART_COUNT", None)
    env.pop("UV_CACHE_DIR_OVERRIDE", None)
    result = subprocess.run(
        ["bash", str(repo / "ray.sub")],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0  # The stub deliberately rejects the launch.
    assert arguments.is_file(), result.stderr
    container_cwd = [
        argument.removeprefix("--container-workdir=")
        for argument in arguments.read_text().splitlines()
        if argument.startswith("--container-workdir=")
    ]
    assert container_cwd == [str(workdir)]
