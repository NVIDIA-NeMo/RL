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

The batch-side head wait (the poll loop through the final `wait` and `exit`)
and the DRIVER_EXIT_GRACE_S validation are extracted from ray.sub and run in
bash against fake processes, so the tests exercise the real shell code without
Slurm. The head container's marker write is checked in place inside the
generated head command.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RAY_SUB = REPO_ROOT / "ray.sub"
MARKER_WRITE = '  echo "\\$exit_code" > "$LOG_DIR/DRIVER_EXITED" 2>/dev/null || true'
# Distinct from every driver code the tests use, so the job's exit status tells
# whether it came from the head (`wait` propagation) or from the watchdog.
HEAD_EXIT_CODE = 7

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _lines() -> list[str]:
    return RAY_SUB.read_text().splitlines()


def _batch_head_wait() -> str:
    """The COMMAND branch from the watchdog state through `wait` and `exit`."""
    lines = _lines()
    start = lines.index('  _driver_exited_at=""')
    end = lines.index("  exit $exit_code", start)
    return "\n".join(lines[start : end + 1])


def _grace_validation() -> str:
    lines = _lines()
    start = lines.index("DRIVER_EXIT_GRACE_S=${DRIVER_EXIT_GRACE_S:-180}")
    end = lines.index("fi", start)
    return "\n".join(lines[start : end + 1])


def _head_cmd() -> list[str]:
    lines = _lines()
    start = lines.index("head_cmd=$(cat <<EOF")
    end = lines.index("EOF", start)
    return lines[start + 1 : end]


def _run(
    tmp_path: Path,
    *,
    head_seconds: int,
    grace_s: int,
    marker: str | None,
    sidecar_dies_after_s: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Runs the extracted head wait against a fake head that exits HEAD_EXIT_CODE.

    `sidecar_dies_after_s` adds a worker srun that exits before the head, as after
    a node failure during bringup: 0 means it is already gone when the loop starts,
    N > 0 that it dies N seconds in (the loop polls every 2 s).
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    if marker is not None:
        (log_dir / "DRIVER_EXITED").write_text(marker)
    if sidecar_dies_after_s is None:
        sidecar = ""
    else:
        sidecar = f'sleep {sidecar_dies_after_s} & SRUN_PIDS["ray-workers"]=$!'
        if sidecar_dies_after_s == 0:
            sidecar += '; wait "${SRUN_PIDS["ray-workers"]}"'
    # The extracted tail turns `set -e` on, so a failed kill (head already gone)
    # in the trap must not replace the exit status.
    script = f"""
set -u
declare -A SRUN_PIDS
LOG_DIR={log_dir}
DRIVER_EXIT_GRACE_S={grace_s}
sh -c 'sleep {head_seconds}; exit {HEAD_EXIT_CODE}' >/dev/null 2>&1 &
SRUN_PIDS["ray-head"]=$!
echo "head-pid=$!"
trap 'kill "${{SRUN_PIDS["ray-head"]}}" 2>/dev/null || true' EXIT
{sidecar}
{_batch_head_wait()}
"""
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60
    )


def _fake_head_pid(result: subprocess.CompletedProcess[str]) -> int:
    (line,) = [
        line for line in result.stdout.splitlines() if line.startswith("head-pid=")
    ]
    return int(line.removeprefix("head-pid="))


def _assert_not_running(pid: int) -> None:
    """The harness's EXIT trap must still kill the fake head when the loop `exit`s."""
    if not Path("/proc/self").exists():  # no procfs: nothing to check
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state in ("Z", "X"):
            return
        time.sleep(0.05)
    pytest.fail(f"fake head (pid {pid}) is still running after the wait loop exited")


@pytest.mark.parametrize(
    "marker,expected_code",
    [
        ("3\n", 3),  # failed driver
        ("0\n", 0),  # successful driver: the job still ends, with status 0
        ("garbage\n", 1),  # unreadable marker falls back to 1
    ],
)
def test_ends_job_with_driver_exit_code_when_head_outlives_driver(
    tmp_path: Path, marker: str, expected_code: int
) -> None:
    result = _run(tmp_path, head_seconds=30, grace_s=1, marker=marker)

    assert result.returncode == expected_code  # the driver's, not the head's
    assert "ending the job" in result.stderr
    assert (tmp_path / "logs" / "ENDED").exists()
    _assert_not_running(_fake_head_pid(result))


@pytest.mark.parametrize(
    "grace_s,marker",
    [
        (0, "3\n"),  # watchdog disabled
        (120, "3\n"),  # head exits within the grace period
        (1, None),  # driver still running: no marker
    ],
)
def test_waits_for_head(tmp_path: Path, grace_s: int, marker: str | None) -> None:
    """The head's own status is what the job exits with."""
    result = _run(tmp_path, head_seconds=1, grace_s=grace_s, marker=marker)

    assert result.returncode == HEAD_EXIT_CODE, result.stderr
    assert "ending the job" not in result.stderr
    assert not (tmp_path / "logs" / "ENDED").exists()


def test_sidecar_failure_signals_ended_and_still_propagates_the_head_status(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path, head_seconds=1, grace_s=120, marker=None, sidecar_dies_after_s=0
    )

    # The loop breaks out at the first poll, then `wait`s on the head as before.
    assert result.returncode == HEAD_EXIT_CODE, result.stderr
    assert "exited before the head; failing fast" in result.stderr
    assert "ending the job" not in result.stderr
    assert (tmp_path / "logs" / "ENDED").exists()
    _assert_not_running(_fake_head_pid(result))


def test_sidecar_failure_wins_over_an_expired_grace_period(tmp_path: Path) -> None:
    """Both conditions hold on the t=2 s poll: the sidecar check runs first.

    Pins the watchdog's placement after the sidecar check, so a dying sidecar
    keeps its fail-fast path and the head's status instead of the driver's.
    """
    result = _run(
        tmp_path, head_seconds=4, grace_s=1, marker="3\n", sidecar_dies_after_s=1
    )

    assert result.returncode == HEAD_EXIT_CODE, result.stderr
    assert "exited before the head; failing fast" in result.stderr
    assert "ending the job" not in result.stderr
    assert (tmp_path / "logs" / "ENDED").exists()
    _assert_not_running(_fake_head_pid(result))


@pytest.mark.parametrize(
    "value,expected_code,expected_value",
    [
        (None, 0, "180"),  # unset: default
        ("0", 0, "0"),  # disabled
        ("600", 0, "600"),
        ("08", 0, "8"),  # a leading zero must not be read as octal ((( 08 )) errors)
        ("010", 0, "10"),  # ... nor silently mean 8
        ("abc", 1, None),
        ("1.5", 1, None),
        ("-1", 1, None),
    ],
)
def test_grace_period_must_be_a_non_negative_integer(
    value: str | None, expected_code: int, expected_value: str | None
) -> None:
    script = f"""
set -u
{_grace_validation()}
echo "grace=$DRIVER_EXIT_GRACE_S"
"""
    env = {k: v for k, v in os.environ.items() if k != "DRIVER_EXIT_GRACE_S"}
    if value is not None:
        env["DRIVER_EXIT_GRACE_S"] = value
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30, env=env
    )

    assert result.returncode == expected_code
    if expected_value is None:
        assert "DRIVER_EXIT_GRACE_S must be a non-negative integer" in result.stderr
    else:
        assert f"grace={expected_value}" in result.stdout


def test_head_writes_the_marker_right_after_the_driver_returns() -> None:
    """The marker must carry the driver's status and be written before anything else.

    The Alloy EXIT trap and `exit` follow; if teardown hangs there, the marker is
    the only thing telling the submit host that the driver is done.
    """
    head = _head_cmd()
    captured = head.index("  exit_code=\\$?")
    statements = [
        line for line in head[captured + 1 :] if not line.lstrip().startswith("#")
    ]
    assert statements[0] == MARKER_WRITE, (
        "the DRIVER_EXITED write must directly follow `exit_code=$?` in the head "
        f"command; found {statements[0]!r}"
    )
    assert "  exit \\$exit_code" in statements[1:4], (
        "the head command must still exit with the driver's status after the marker"
    )
