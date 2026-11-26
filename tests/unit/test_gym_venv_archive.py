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

"""``gym_venv_archive.sh unpack``, run against small archives in a temporary directory.

ray.sub ignores its setup command's exit status, so the unpack must touch the ENDED file on every failure.
A stub ``rm`` placed first on ``PATH`` refuses to delete chosen paths, so the tests also cover cleanups that fail,
which do not depend on file permissions and so also fail when the tests run as root.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "nemo_gym"
    / "gym_venv_archive.sh"
)
SERVER = "resources_servers/mcqa"

_RM_STUB = """#!/usr/bin/env bash
for arg in "$@"; do
    # Unquoted on purpose: STUB_RM_REFUSE is a glob.
    if [[ "${arg}" == ${STUB_RM_REFUSE} ]]; then
        echo "rm: stub refuses to delete ${arg}" >&2
        exit 1
    fi
done
exec REAL_RM "$@"
"""


@pytest.fixture
def node(tmp_path):
    """A node whose venv directory already holds venvs; tests put the archive and ENDED next to it."""
    venv_dir = tmp_path / "opt" / "gym_venvs"
    venv_dir.mkdir(parents=True)
    (venv_dir / "existing").write_text("existing venvs\n")
    return tmp_path


def _make_archive(tmp_path, *, servers=(SERVER,), listed=(SERVER,), marker_dir=False):
    src = tmp_path / "src" / "gym_venvs"
    for server in servers:
        python = src / server / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
    if marker_dir:
        # A directory where the unpack writes its marker makes the marker write fail after the swap.
        (src / ".from-archive").mkdir(parents=True)
    archive = tmp_path / "gym_venvs.tar"
    subprocess.run(
        ["tar", "-C", str(src.parent), "-cf", str(archive), "gym_venvs"], check=True
    )
    Path(f"{archive}.size").write_text(f"{archive.stat().st_size}\n")
    Path(f"{archive}.servers").write_text("".join(f"{s}\n" for s in listed))
    return archive


def _unpack(tmp_path, archive, *, refuse_rm=None):
    env = dict(os.environ, NEMO_GYM_VENV_DIR=str(tmp_path / "opt" / "gym_venvs"))
    if refuse_rm is not None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "rm"
        stub.write_text(_RM_STUB.replace("REAL_RM", shutil.which("rm")))
        stub.chmod(0o755)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        env["STUB_RM_REFUSE"] = refuse_rm
    return subprocess.run(
        ["bash", str(SCRIPT), "unpack", str(archive), str(tmp_path / "ENDED")],
        capture_output=True,
        text=True,
        env=env,
    )


def _staging_dirs(tmp_path):
    return sorted((tmp_path / "opt").glob(".gym_venvs.*"))


def test_unpack_replaces_the_existing_venvs_and_a_rerun_skips_it(node):
    archive = _make_archive(node)

    first = _unpack(node, archive)
    assert first.returncode == 0, first.stderr
    venv_dir = node / "opt" / "gym_venvs"
    assert (venv_dir / SERVER / ".venv" / "bin" / "python").is_file()
    assert not (venv_dir / "existing").exists()
    assert (venv_dir / ".from-archive").read_text() == f"{archive}\n"
    assert _staging_dirs(node) == []

    rerun = _unpack(node, archive)
    assert rerun.returncode == 0, rerun.stderr
    assert "unpacked" not in rerun.stdout
    assert not (node / "ENDED").exists()


def test_a_missing_server_keeps_the_existing_venvs_and_ends_the_job(node):
    archive = _make_archive(node, listed=(SERVER, "resources_servers/missing"))

    result = _unpack(node, archive)

    assert result.returncode != 0
    assert "missing: resources_servers/missing" in result.stderr
    assert (node / "ENDED").exists()
    assert (node / "opt" / "gym_venvs" / "existing").is_file()
    assert _staging_dirs(node) == []


def test_a_failed_marker_write_puts_the_existing_venvs_back_and_ends_the_job(node):
    archive = _make_archive(node, marker_dir=True)

    result = _unpack(node, archive)

    assert result.returncode != 0
    assert (node / "ENDED").exists()
    assert (node / "opt" / "gym_venvs" / "existing").is_file()
    assert _staging_dirs(node) == []


def test_a_failing_staging_cleanup_still_ends_the_job(node):
    archive = _make_archive(node, listed=(SERVER, "resources_servers/missing"))

    result = _unpack(node, archive, refuse_rm="*/.gym_venvs.*")

    assert result.returncode != 0
    assert "stub refuses" in result.stderr
    assert (node / "ENDED").exists()
    assert (node / "opt" / "gym_venvs" / "existing").is_file()


def test_a_failing_rollback_still_ends_the_job_and_keeps_the_existing_venvs(node):
    archive = _make_archive(node, marker_dir=True)
    venv_dir = node / "opt" / "gym_venvs"

    result = _unpack(node, archive, refuse_rm=str(venv_dir))

    assert result.returncode != 0
    assert "stub refuses" in result.stderr
    assert (node / "ENDED").exists()
    # The new venvs could not be removed, so the existing ones stay in staging instead of moving inside them.
    [staging] = _staging_dirs(node)
    assert (staging / ".previous" / "existing").is_file()
    assert not (venv_dir / ".previous").exists()
