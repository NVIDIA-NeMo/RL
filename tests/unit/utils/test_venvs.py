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
import fcntl
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nemo_rl.utils.venvs import create_local_venv
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


def test_create_local_venv_forwards_uv_no_install_packages():
    create_local_venv.cache_clear()
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        with (
            patch.dict(
                os.environ,
                {
                    "NEMO_RL_VENV_DIR": tempdir,
                    "NRL_UV_NO_INSTALL_PACKAGES": (
                        "deep-ep,causal-conv1d,,fast-hadamard-transform"
                    ),
                },
            ),
            patch("nemo_rl.utils.venvs.subprocess.run") as run_mock,
        ):
            Path(tempdir, "test_venv").mkdir()
            venv_python = create_local_venv(
                py_executable="uv run --group mcore", venv_name="test_venv"
            )

    assert venv_python == f"{tempdir}/test_venv/bin/python"
    sync_cmd = run_mock.call_args_list[1].args[0]
    exec_cmd = run_mock.call_args_list[2].args[0]
    expected_args = [
        "--no-install-package",
        "deep-ep",
        "--no-install-package",
        "causal-conv1d",
        "--no-install-package",
        "fast-hadamard-transform",
    ]
    assert sync_cmd[:2] == ["uv", "sync"]
    assert "--locked" not in sync_cmd
    assert "--group" in sync_cmd
    assert "mcore" in sync_cmd
    assert sync_cmd[-len(expected_args) :] == expected_args
    assert exec_cmd[:3] == ["uv", "run", "--no-sync"]
    assert "--no-install-package" not in exec_cmd
    create_local_venv.cache_clear()


def test_create_local_venv_can_serialize_uv_sync():
    create_local_venv.cache_clear()
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        lock_path = Path(tempdir) / "uv-sync.lock"
        with (
            patch.dict(
                os.environ,
                {
                    "NEMO_RL_VENV_DIR": tempdir,
                    "NRL_SERIALIZE_UV_SYNC": "true",
                    "NRL_UV_SYNC_LOCK_PATH": str(lock_path),
                },
            ),
            patch("nemo_rl.utils.venvs.subprocess.run") as run_mock,
            patch("fcntl.flock") as flock_mock,
        ):
            Path(tempdir, "test_venv").mkdir()
            venv_python = create_local_venv(
                py_executable="uv run --group mcore", venv_name="test_venv"
            )

    assert venv_python == f"{tempdir}/test_venv/bin/python"
    assert run_mock.call_args_list[1].args[0][:2] == ["uv", "sync"]
    assert run_mock.call_args_list[2].args[0][:3] == ["uv", "run", "--no-sync"]
    assert flock_mock.call_args_list[0].args[1] == fcntl.LOCK_EX
    assert flock_mock.call_args_list[-1].args[1] == fcntl.LOCK_UN
    create_local_venv.cache_clear()
