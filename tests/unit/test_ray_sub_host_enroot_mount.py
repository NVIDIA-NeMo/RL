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

"""ray.sub's ``NRL_MOUNT_HOST_ENROOT`` handling, executed against a fake root.

The knob validation and the mount block are cut out of ray.sub verbatim; the
block's ``/usr`` and ``/etc`` paths are redirected into a temporary root so the
tests can stage an enroot installation, and the block runs under
``set -eou pipefail`` exactly as ray.sub runs it.
"""

import os
import re
import subprocess
from pathlib import Path

RAY_SUB = Path(__file__).resolve().parents[2] / "ray.sub"
BLOCK_START = "# NRL_MOUNT_HOST_ENROOT=1: a process inside the training container"


def _validation_text() -> str:
    match = re.search(
        r"^NRL_MOUNT_HOST_ENROOT=\$\{NRL_MOUNT_HOST_ENROOT:-\}.*?^esac\n",
        RAY_SUB.read_text(),
        re.S | re.M,
    )
    assert match, "the NRL_MOUNT_HOST_ENROOT validation was not found in ray.sub"
    return match.group(0)


def _block_text() -> str:
    text = RAY_SUB.read_text()
    start = text.index(BLOCK_START)
    end = text.index("\nfi\n", start) + len("\nfi\n")
    return text[start:end]


def _stage(root: Path, *, dirs=(), bins=()):
    for rel in dirs:
        (root / rel.lstrip("/")).mkdir(parents=True, exist_ok=True)
    for rel in bins:
        path = root / rel.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\necho 'enroot version 3.5.0'\n")
        path.chmod(0o755)


def _run_block(tmp_path, knob, *, mounts="", dirs=(), bins=()):
    root = tmp_path / "root"
    _stage(root, dirs=dirs, bins=bins)
    block = (
        _block_text().replace("/usr/", f"{root}/usr/").replace("/etc/", f"{root}/etc/")
    )
    knob_line = (
        "unset NRL_MOUNT_HOST_ENROOT\n"
        if knob is None
        else f"export NRL_MOUNT_HOST_ENROOT='{knob}'\n"
    )
    # The block's own [INFO] line goes to stdout too; MOUNTS is printed last,
    # on its own line, so the caller can read it back.
    script = f"set -eou pipefail\n{knob_line}MOUNTS='{mounts}'\n{block}\nprintf '\\nMOUNTS=%s' \"$MOUNTS\"\n"
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{root}/usr/bin:{os.environ['PATH']}"},
    )


def _mounts(result) -> str:
    """The MOUNTS value the block left behind."""
    last_line = result.stdout.splitlines()[-1]
    assert last_line.startswith("MOUNTS="), result.stdout
    return last_line[len("MOUNTS=") :]


def _run_validation(knob):
    knob_line = (
        "unset NRL_MOUNT_HOST_ENROOT\n"
        if knob is None
        else f"export NRL_MOUNT_HOST_ENROOT='{knob}'\n"
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            f"set -eou pipefail\n{knob_line}{_validation_text()}echo accepted\n",
        ],
        capture_output=True,
        text=True,
    )


def test_knob_defaults_to_unset_and_the_block_precedes_the_container_mounts():
    text = RAY_SUB.read_text()
    assert "NRL_MOUNT_HOST_ENROOT=${NRL_MOUNT_HOST_ENROOT:-}" in text
    assert text.index(BLOCK_START) < text.index(
        'COMMON_SRUN_ARGS+=" --container-mounts=$MOUNTS"'
    )


def test_unset_and_zero_leave_the_mounts_untouched(tmp_path):
    for knob in (None, "0"):
        result = _run_block(tmp_path / str(knob), knob, mounts="/a:/a")
        assert result.returncode == 0, result.stderr
        assert _mounts(result) == "/a:/a"
        assert "bind-mounting" not in result.stdout


def test_other_values_are_refused_at_startup():
    """Any value other than 1 used to be a silent no-op whose first symptom was
    'enroot: command not found' inside the container, minutes to hours later."""
    for knob in ("true", "yes", "on"):
        result = _run_validation(knob)
        assert result.returncode == 1
        assert "NRL_MOUNT_HOST_ENROOT must be unset, 0, or 1" in result.stderr
    for knob in (None, "0", "1"):
        assert _run_validation(knob).stdout.strip() == "accepted"


def test_a_node_without_enroot_fails_naming_the_knob(tmp_path):
    result = _run_block(tmp_path, "1")
    assert result.returncode == 1
    assert "NRL_MOUNT_HOST_ENROOT=1" in result.stderr
    assert "no enroot to mount" in result.stderr


def test_present_directories_and_binaries_become_mount_entries(tmp_path):
    """Directories in loop order, then every /usr/bin/enroot* file in glob order;
    absent directories are skipped without tripping set -e, and an empty MOUNTS
    gets no leading comma."""
    root = tmp_path / "root"
    result = _run_block(
        tmp_path,
        "1",
        dirs=("/usr/lib/enroot", "/etc/enroot"),
        bins=("/usr/bin/enroot", "/usr/bin/enroot-mount"),
    )
    assert result.returncode == 0, result.stderr
    expected = [
        f"{root}/usr/lib/enroot",
        f"{root}/etc/enroot",
        f"{root}/usr/bin/enroot",
        f"{root}/usr/bin/enroot-mount",
    ]
    assert _mounts(result) == ",".join(f"{p}:{p}" for p in expected)
    assert "bind-mounting" in result.stdout
    assert "2 binaries" in result.stdout


def test_existing_mounts_are_kept_in_front(tmp_path):
    root = tmp_path / "root"
    result = _run_block(tmp_path, "1", mounts="/x:/x", bins=("/usr/bin/enroot",))
    assert result.returncode == 0, result.stderr
    assert _mounts(result) == f"/x:/x,{root}/usr/bin/enroot:{root}/usr/bin/enroot"
