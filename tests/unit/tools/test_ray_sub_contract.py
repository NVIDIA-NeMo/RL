# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute launcher shell fragments without Slurm, containers, or GPUs."""

import os
from pathlib import Path
import subprocess

import pytest

LAUNCHER = Path(__file__).resolve().parents[3] / "ray.sub"


def fragment(start, end):
    text = LAUNCHER.read_text()
    return text[text.index(start) : text.index(end)]


@pytest.mark.parametrize(
    "container_cwd", ["/mounted/run", "/mounted/run with spaces", None]
)
def test_container_cwd_and_identity_survive_slurm_cleanup(container_cwd):
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": "123",
        "SLURM_SUBMIT_DIR": "/host/alias",
    }
    if container_cwd is not None:
        env["NRL_CONTAINER_WORKDIR"] = container_cwd
    code = fragment("# Preserve allocation identity", "# Record job-start epoch")
    code += "\nunset SLURM_JOB_ID SLURM_SUBMIT_DIR\n"
    code += 'printf "%s\\n" "$NRL_SLURM_JOB_ID" "$NRL_SLURM_SUBMIT_DIR" "$NRL_CONTAINER_WORKDIR"'
    result = subprocess.run(
        ["bash", "-eu", "-c", code], env=env, check=True, capture_output=True, text=True
    )
    assert result.stdout.splitlines() == [
        "123",
        "/host/alias",
        container_cwd or "/host/alias",
    ]


def test_reject_relative_container_cwd():
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": "123",
        "SLURM_SUBMIT_DIR": "/host",
        "NRL_CONTAINER_WORKDIR": "relative",
    }
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            fragment("# Preserve allocation identity", "# Record job-start epoch"),
        ],
        env=env,
        capture_output=True,
    )
    assert result.returncode != 0


def test_common_arguments_preserve_container_path_as_one_argument():
    env = {
        "PATH": os.environ["PATH"],
        "GRES_ARG": "--gres=gpu:4",
        "MOUNTS": "/host:/mounted",
        "CONTAINER": "/images/rl.sqsh",
        "NRL_CONTAINER_WORKDIR": "/mounted/run with spaces",
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_ACCOUNT": "test_account",
    }
    code = fragment("COMMON_SRUN_ARGS=()", "# Number of CPUs per worker node.")
    code += '\nprintf "%s\\n" "${COMMON_SRUN_ARGS[@]}"'
    result = subprocess.run(
        ["bash", "-eu", "-c", code], env=env, check=True, capture_output=True, text=True
    )
    assert "--container-workdir=/mounted/run with spaces" in result.stdout.splitlines()
    assert LAUNCHER.read_text().count('srun "${COMMON_SRUN_ARGS[@]}"') == 2


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_affinity_guard_checks_effective_mask():
    function = fragment(
        "check_cpu_affinity() {",
        "########################################################\n# User defined variables",
    )
    cpus = len(os.sched_getaffinity(0))
    for expected, success in [(cpus, True), (cpus + 1, False)]:
        result = subprocess.run(
            ["bash", "-eu", "-c", function + f"\ncheck_cpu_affinity {expected}"],
            capture_output=True,
        )
        assert (result.returncode == 0) is success
    text = LAUNCHER.read_text()
    assert text.count("$(declare -f check_cpu_affinity)") == 3
    sandbox = fragment(
        'srun --output "$SANDBOX_PORTS_DIR', 'srun "${COMMON_SRUN_ARGS[@]}"'
    )
    assert '--cpus-per-task="$CPUS_PER_WORKER"' in sandbox
    assert "--cpu-bind=cores" in sandbox
    assert "COMMON_SRUN_ARGS+=(--overlap --cpu-bind=cores)" in text
