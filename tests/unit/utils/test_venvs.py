# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import errno
import os
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pathlib import Path

from nemo_rl.utils.venvs import (
    _build_or_wait_for_venv,
    _clear_incomplete_venv,
    _get_venv_marker_paths,
    create_local_venv,
)
from tests.unit.conftest import TEST_ASSETS_DIR


def test_create_local_venv():
    # The temporary directory is created within the project.
    # For some reason, creating a virtual environment outside of the project
    # doesn't work reliably.
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        # Mock os.environ to set NEMO_RL_VENV_DIR for this test
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_python = create_local_venv(
                py_executable="uv run --group docs", venv_name="test_venv"
            )
            assert os.path.exists(venv_python)
            assert venv_python == f"{tempdir}/test_venv/bin/python"
            # Check if sphinx package is installed in the created venv

            # Run a Python command to check if sphinx can be imported
            result = subprocess.run(
                [
                    venv_python,
                    "-c",
                    "import sphinx; print('Sphinx package is installed')",
                ],
                capture_output=True,
                text=True,
            )

            # Verify the command executed successfully (return code 0)
            assert result.returncode == 0, f"Failed to import sphinx: {result.stderr}"
            assert "Sphinx package is installed" in result.stdout


def _fake_create_local_venv(tempdir: str):
    def _create(py_executable: str, venv_name: str, force_rebuild: bool = False) -> str:
        venv_path = Path(tempdir) / venv_name
        python_path = venv_path / "bin" / "python"
        python_path.parent.mkdir(parents=True, exist_ok=True)
        python_path.write_text("")
        return str(python_path)

    return _create


def test_clear_incomplete_venv_ignores_disappearing_files():
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        venv_path = Path(tempdir) / "test_venv"
        venv_path.mkdir()
        _, ready_file = _get_venv_marker_paths(venv_path)
        ready_file.touch()

        def fake_rmtree(path, onexc=None):
            assert path == venv_path
            assert onexc is not None
            onexc(
                os.unlink,
                str(venv_path / "bin" / "python"),
                FileNotFoundError("gone"),
            )

        with patch("nemo_rl.utils.venvs.shutil.rmtree", side_effect=fake_rmtree):
            _clear_incomplete_venv(venv_path, ready_file)

        assert not ready_file.exists()


def test_clear_incomplete_venv_ignores_stale_file_handles():
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        venv_path = Path(tempdir) / "test_venv"
        venv_path.mkdir()
        _, ready_file = _get_venv_marker_paths(venv_path)
        ready_file.touch()

        def fake_rmtree(path, onexc=None):
            assert path == venv_path
            assert onexc is not None
            onexc(
                os.unlink,
                str(venv_path / "bin" / "python"),
                OSError(errno.ESTALE, "Stale file handle"),
            )

        with patch("nemo_rl.utils.venvs.shutil.rmtree", side_effect=fake_rmtree):
            _clear_incomplete_venv(venv_path, ready_file)

        assert not ready_file.exists()


def test_clear_incomplete_venv_raises_non_missing_errors():
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        venv_path = Path(tempdir) / "test_venv"
        venv_path.mkdir()
        _, ready_file = _get_venv_marker_paths(venv_path)
        ready_file.touch()

        def fake_rmtree(path, onexc=None):
            assert path == venv_path
            assert onexc is not None
            onexc(
                os.unlink,
                str(venv_path / "bin" / "python"),
                PermissionError("denied"),
            )

        with patch("nemo_rl.utils.venvs.shutil.rmtree", side_effect=fake_rmtree):
            try:
                _clear_incomplete_venv(venv_path, ready_file)
            except PermissionError:
                pass
            else:
                assert False, "Expected PermissionError to propagate"


def test_build_or_wait_for_venv_reuses_ready_venv():
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_path = Path(tempdir) / "test_venv"
            python_path = venv_path / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")
            _, ready_file = _get_venv_marker_paths(venv_path)
            ready_file.touch()

            with patch("nemo_rl.utils.venvs.create_local_venv") as mock_create:
                result = _build_or_wait_for_venv(
                    py_executable="uv run --group docs",
                    venv_name="test_venv",
                    node_idx=0,
                )

            assert result == str(python_path)
            mock_create.assert_not_called()


def test_build_or_wait_for_venv_rebuilds_incomplete_venv():
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_path = Path(tempdir) / "test_venv"
            stale_python = venv_path / "bin" / "python"
            stale_python.parent.mkdir(parents=True, exist_ok=True)
            stale_python.write_text("")

            with patch(
                "nemo_rl.utils.venvs.create_local_venv",
                side_effect=_fake_create_local_venv(tempdir),
            ) as mock_create, patch("nemo_rl.utils.venvs.time.sleep", return_value=None):
                result = _build_or_wait_for_venv(
                    py_executable="uv run --group docs",
                    venv_name="test_venv",
                    node_idx=0,
                )

            _, ready_file = _get_venv_marker_paths(venv_path)
            assert result == str(venv_path / "bin" / "python")
            assert ready_file.exists()
            mock_create.assert_called_once()


def test_build_or_wait_for_venv_force_rebuild_ignores_stale_marker():
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_path = Path(tempdir) / "test_venv"
            python_path = venv_path / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")
            started_file, ready_file = _get_venv_marker_paths(venv_path)
            started_file.touch()

            with patch(
                "nemo_rl.utils.venvs.create_local_venv",
                side_effect=_fake_create_local_venv(tempdir),
            ) as mock_create, patch("nemo_rl.utils.venvs.time.sleep", return_value=None):
                result = _build_or_wait_for_venv(
                    py_executable="uv run --group docs",
                    venv_name="test_venv",
                    node_idx=0,
                    force_rebuild=True,
                )

            assert result == str(python_path)
            assert ready_file.exists()
            assert not started_file.exists()
            mock_create.assert_called_once()


def test_build_or_wait_for_venv_recovers_from_stale_builder_marker():
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        with patch.dict(
            os.environ,
            {
                "NEMO_RL_VENV_DIR": tempdir,
                "NEMO_RL_VENV_STALE_TIMEOUT_SECS": "0",
            },
        ):
            venv_path = Path(tempdir) / "test_venv"
            python_path = venv_path / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")
            started_file, ready_file = _get_venv_marker_paths(venv_path)
            started_file.touch()

            with patch(
                "nemo_rl.utils.venvs.create_local_venv",
                side_effect=_fake_create_local_venv(tempdir),
            ) as mock_create, patch("nemo_rl.utils.venvs.time.sleep", return_value=None):
                result = _build_or_wait_for_venv(
                    py_executable="uv run --group docs",
                    venv_name="test_venv",
                    node_idx=0,
                )

            assert result == str(python_path)
            assert ready_file.exists()
            assert not started_file.exists()
            mock_create.assert_called_once()
