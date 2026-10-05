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
"""ray.sub ends the job when the head srun outlives the driver.

The batch-side wait loop is extracted from ray.sub and run against a fake head
process, so the test exercises the real shell code without Slurm.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RAY_SUB = REPO_ROOT / "ray.sub"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _batch_wait_loop() -> str:
    lines = RAY_SUB.read_text().splitlines()
    start = lines.index('  _driver_exited_at=""')
    end = lines.index("  done", start)
    return "\n".join(lines[start : end + 1])


def _run(tmp_path: Path, *, head_seconds: int, grace_s: int, driver_exited: bool):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    if driver_exited:
        (log_dir / "DRIVER_EXITED").write_text("3\n")
    script = f"""
set -u
declare -A SRUN_PIDS
LOG_DIR={log_dir}
DRIVER_EXIT_GRACE_S={grace_s}
sleep {head_seconds} >/dev/null 2>&1 &
SRUN_PIDS["ray-head"]=$!
trap 'kill "${{SRUN_PIDS["ray-head"]}}" 2>/dev/null' EXIT
{_batch_wait_loop()}
echo head-exited
"""
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60
    )


def test_ends_job_with_driver_exit_code_when_head_outlives_driver(tmp_path):
    result = _run(tmp_path, head_seconds=30, grace_s=1, driver_exited=True)

    assert result.returncode == 3
    assert "head-exited" not in result.stdout
    assert "ending the job" in result.stderr
    assert (tmp_path / "logs" / "ENDED").exists()


@pytest.mark.parametrize(
    "grace_s,driver_exited",
    [
        (0, True),  # watchdog disabled
        (120, True),  # head exits within the grace period
        (1, False),  # driver still running
    ],
)
def test_waits_for_head(tmp_path, grace_s, driver_exited):
    result = _run(
        tmp_path, head_seconds=1, grace_s=grace_s, driver_exited=driver_exited
    )

    assert result.returncode == 0
    assert "head-exited" in result.stdout
    assert not (tmp_path / "logs" / "ENDED").exists()
