from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(path: Path, *args: str, **environment: str) -> str:
    result = subprocess.run(
        ["bash", str(path), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_compat_smoke_uses_short_compute_node_tmpdir() -> None:
    output = _run_script(
        REPO_ROOT / "scripts" / "submit_vllm_024_compat_smoke.sh",
        REPO_DIR=str(REPO_ROOT),
        CONTAINER="/unused/nemo-rl.sqsh",
        DRY_RUN="true",
    )

    assert "TMPDIR=/tmp" in output


def test_performance_launcher_uses_short_compute_node_tmpdir() -> None:
    output = _run_script(
        REPO_ROOT
        / "experiments"
        / "vllm_024_upgrade"
        / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
    )

    assert "TMPDIR=/tmp" in output


def test_performance_launcher_preserves_compute_visible_workdir() -> None:
    output = _run_script(
        REPO_ROOT
        / "experiments"
        / "vllm_024_upgrade"
        / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "CONTAINER_WORKDIR=/lustre/users/sna/RL" in output


def test_performance_launcher_imports_nemo_rl_from_the_checkout() -> None:
    output = _run_script(
        REPO_ROOT
        / "experiments"
        / "vllm_024_upgrade"
        / "submit_performance_step10.sh",
        "dry-run",
        "qwen32b",
        REPO_DIR="/lustre/users/sna/RL",
    )

    assert "PYTHONPATH=/lustre/users/sna/RL" in output


def test_ray_launcher_accepts_an_explicit_container_workdir() -> None:
    source = (REPO_ROOT / "ray.sub").read_text(encoding="utf-8")

    assert 'CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$SLURM_SUBMIT_DIR}' in source
    assert 'COMMON_SRUN_ARGS+=" --container-workdir=$CONTAINER_WORKDIR"' in source
