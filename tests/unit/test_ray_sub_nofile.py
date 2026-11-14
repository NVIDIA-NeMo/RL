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

"""ray.sub's file-descriptor limit block, executed against a stub ``ulimit``.

The block is cut out of ray.sub verbatim and run under ``set -eou pipefail``
with a shell function shadowing the ``ulimit`` builtin, so the tests pin the
soft limit each hard limit resolves to without touching real process limits.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

RAY_SUB = Path(__file__).resolve().parents[2] / "ray.sub"

RAY_FLOOR = 65535
DEFAULT_TARGET = 131072
STARTING_SOFT = 1024

# Functions win over builtins in bash, so the block under test reads and writes
# this stub. Command substitution forks the shell, so `$(ulimit -Sn)` in the
# block's final echo sees the STUB_SOFT its own earlier call assigned.
_ULIMIT_STUB = """
ulimit() {
  case "$1" in
    -Hn) echo "${STUB_HARD}" ;;
    -Sn)
      if [[ -n "${2:-}" ]]; then STUB_SOFT="$2"; else echo "${STUB_SOFT}"; fi
      ;;
    *) echo "unexpected ulimit $*" >&2; return 2 ;;
  esac
}
"""


def _block_text() -> str:
    match = re.search(
        r'^NRL_TARGET_NOFILE=.*?^echo "\[INFO\] file descriptor limit:.*?\n',
        RAY_SUB.read_text(),
        re.S | re.M,
    )
    assert match, "file descriptor limit block not found in ray.sub"
    return match.group(0)


def _run(hard, soft=STARTING_SOFT, target=None):
    env = {k: v for k, v in os.environ.items() if k != "NRL_TARGET_NOFILE"}
    env["STUB_HARD"] = str(hard)
    env["STUB_SOFT"] = str(soft)
    if target is not None:
        env["NRL_TARGET_NOFILE"] = str(target)
    script = f"set -eou pipefail\n{_ULIMIT_STUB}\n{_block_text()}"
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )


def _resolved_soft(result) -> str:
    match = re.search(r"soft=(\S+) hard=", result.stdout)
    assert match, f"no [INFO] line in {result.stdout!r} / {result.stderr!r}"
    return match.group(1)


@pytest.mark.parametrize("hard", ["unlimited", DEFAULT_TARGET, 1048576])
def test_a_hard_limit_at_or_above_the_target_takes_the_target(hard):
    result = _run(hard)
    assert result.returncode == 0, result.stderr
    assert _resolved_soft(result) == str(DEFAULT_TARGET)


def test_a_hard_limit_between_the_floor_and_the_target_takes_everything():
    """Ray's 65535 is a floor, so a host that allows more should hand it over."""
    result = _run(100000)
    assert result.returncode == 0, result.stderr
    assert _resolved_soft(result) == "100000"


def test_a_hard_limit_exactly_at_the_floor_is_still_raised():
    """The previous form compared strictly on both branches, so a host whose
    hard limit was exactly 65535 fell through without raising the soft limit."""
    result = _run(RAY_FLOOR)
    assert result.returncode == 0, result.stderr
    assert _resolved_soft(result) == str(RAY_FLOOR)


def test_a_hard_limit_below_the_floor_warns_and_changes_nothing():
    result = _run(4096)
    assert result.returncode == 0, result.stderr
    assert "[WARNING]" in result.stdout
    assert _resolved_soft(result) == str(STARTING_SOFT)


@pytest.mark.parametrize(
    ("soft", "hard"),
    [
        (1048576, 1048576),
        (200000, 1048576),
        ("unlimited", "unlimited"),
        (DEFAULT_TARGET, DEFAULT_TARGET),
    ],
)
def test_a_soft_limit_already_at_or_above_the_target_is_never_lowered(soft, hard):
    """Raising a limit must not lower one. A container image can start well
    above the target, and Ray's own ray.init and Gym's set_ulimit both only
    ever raise."""
    result = _run(hard, soft=soft)
    assert result.returncode == 0, result.stderr
    assert _resolved_soft(result) == str(soft)


def test_a_soft_limit_below_the_target_is_still_raised():
    """The no-lowering guard must not swallow the case the block exists for."""
    result = _run(1048576, soft=70000)
    assert result.returncode == 0, result.stderr
    assert _resolved_soft(result) == str(DEFAULT_TARGET)


def test_the_target_is_overridable():
    result = _run("unlimited", target=262144)
    assert result.returncode == 0, result.stderr
    assert _resolved_soft(result) == "262144"


def test_the_resolved_limits_are_reported():
    """A silent ulimit leaves no way to tell which branch a run took."""
    result = _run(1048576)
    assert "[INFO] file descriptor limit: soft=131072 hard=1048576" in result.stdout
