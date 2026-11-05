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

"""Tests for :mod:`nrl_k8s.workdir` — staging the Ray ``working_dir`` upload.

These tests build a small fixture repo tree under ``tmp_path`` and verify:

* ``include_paths`` filters the staged subset.
* Missing optional paths are skipped silently.
* Ignore patterns drop ``.gitignore`` (Ray's packager uses it and silently
  loses training JSONLs), ``__pycache__``, and related caches.
* ``extra_files`` lands the merged recipe into the staged root.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from nrl_k8s.workdir import stage_workdir

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A minimal repo tree with one source dir + a cache dir + a .gitignore."""
    root = tmp_path / "repo"
    (root / "nemo_rl").mkdir(parents=True)
    (root / "nemo_rl" / "mod.py").write_text("x = 1\n")
    (root / "nemo_rl" / ".gitignore").write_text("data.jsonl\n")
    (root / "nemo_rl" / "data.jsonl").write_text("{}\n")
    (root / "nemo_rl" / "__pycache__").mkdir()
    (root / "nemo_rl" / "__pycache__" / "mod.cpython.pyc").write_text("bin")

    (root / "examples").mkdir()
    (root / "examples" / "run.py").write_text("print('hi')\n")

    # a single-file include target (like tests/check_metrics.py)
    (root / "tests").mkdir()
    (root / "tests" / "check_metrics.py").write_text("# metrics\n")

    return root


# =============================================================================
# Tests
# =============================================================================


class TestStageWorkdir:
    def test_respects_include_paths(self, fake_repo: Path) -> None:
        """Only the requested subtrees end up in the staged dir."""
        dest = stage_workdir(fake_repo, include_paths=["nemo_rl"])
        try:
            assert (dest / "nemo_rl" / "mod.py").is_file()
            # 'examples' was not requested, must not leak in.
            assert not (dest / "examples").exists()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_skips_missing_optional_paths(self, fake_repo: Path) -> None:
        """A path in ``include_paths`` that doesn't exist is silently skipped."""
        dest = stage_workdir(
            fake_repo,
            include_paths=["nemo_rl", "3rdparty/Gym-workspace/Gym/uv.lock"],
        )
        try:
            assert (dest / "nemo_rl" / "mod.py").is_file()
            assert not (dest / "3rdparty").exists()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_strips_gitignore_and_pycache(self, fake_repo: Path) -> None:
        """``.gitignore`` and ``__pycache__`` are scrubbed from the staged copy.

        Ray's packager honours ``.gitignore`` and would drop data files.
        """
        dest = stage_workdir(fake_repo, include_paths=["nemo_rl"])
        try:
            # The source files that would otherwise be dropped must remain.
            assert (dest / "nemo_rl" / "data.jsonl").is_file()
            # But the ignore-triggering files must be gone.
            assert not (dest / "nemo_rl" / ".gitignore").exists()
            assert not (dest / "nemo_rl" / "__pycache__").exists()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_writes_extra_files(self, fake_repo: Path) -> None:
        """``extra_files`` seeds additional paths (used for the merged recipe YAML)."""
        dest = stage_workdir(
            fake_repo,
            include_paths=["nemo_rl"],
            extra_files={"nrl_k8s_run.yaml": "policy:\n  x: 1\n"},
        )
        try:
            out = dest / "nrl_k8s_run.yaml"
            assert out.is_file()
            assert "policy:" in out.read_text()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_single_file_include_copies_file(self, fake_repo: Path) -> None:
        """An ``include_paths`` entry pointing at a file copies just that file."""
        dest = stage_workdir(fake_repo, include_paths=["tests/check_metrics.py"])
        try:
            assert (dest / "tests" / "check_metrics.py").is_file()
        finally:
            shutil.rmtree(dest, ignore_errors=True)
