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
