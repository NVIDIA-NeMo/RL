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
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from nemo_rl.utils.venvs import (
    add_hf_modules_cache_to_pythonpath,
    create_local_venv,
    make_actor_runtime_env,
)
from tests.unit.conftest import TEST_ASSETS_DIR


def _mock_uv_venv(command, **_kwargs):
    if command[:2] == ["uv", "venv"]:
        python_path = Path(command[-1]) / "bin" / "python"
        python_path.parent.mkdir(parents=True, exist_ok=True)
        python_path.touch()


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


def test_create_local_venv_refreshes_when_uv_spec_changes(tmp_path):
    vllm = "uv run --locked --extra vllm"
    vllm_gym = "uv run --locked --extra vllm --extra nemo_gym"
    venv_path = tmp_path / "worker"

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}),
        patch(
            "nemo_rl.utils.venvs.subprocess.run", side_effect=_mock_uv_venv
        ) as mock_run,
    ):
        create_local_venv(vllm, "worker")
        first_fingerprint = (venv_path / ".nemo_rl_venv_spec").read_text()

        mock_run.reset_mock()
        create_local_venv(vllm, "worker")
        mock_run.assert_not_called()

        create_local_venv(vllm_gym, "worker")
        assert mock_run.call_count == 3
        assert (venv_path / ".nemo_rl_venv_spec").read_text() != first_fingerprint


def test_create_local_venv_refreshes_legacy_cache_without_spec(tmp_path):
    venv_path = tmp_path / "worker"
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}),
        patch(
            "nemo_rl.utils.venvs.subprocess.run", side_effect=_mock_uv_venv
        ) as mock_run,
    ):
        create_local_venv("uv run --locked --extra vllm", "worker")

    assert mock_run.call_count == 3
    assert (venv_path / ".nemo_rl_venv_spec").is_file()


def test_create_local_venv_refreshes_when_lockfile_changes(tmp_path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    lock_path = project_path / "uv.lock"
    lock_path.write_text("version = 1\n")
    venv_root = tmp_path / "venvs"
    venv_path = venv_root / "worker"
    py_executable = "uv run --locked --extra vllm"

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(venv_root)}),
        patch("nemo_rl.utils.venvs.git_root", str(project_path)),
        patch(
            "nemo_rl.utils.venvs.subprocess.run", side_effect=_mock_uv_venv
        ) as mock_run,
    ):
        create_local_venv(py_executable, "worker")
        first_fingerprint = (venv_path / ".nemo_rl_venv_spec").read_text()

        mock_run.reset_mock()
        lock_path.write_text("version = 2\n")
        create_local_venv(py_executable, "worker")

    assert mock_run.call_count == 3
    assert (venv_path / ".nemo_rl_venv_spec").read_text() != first_fingerprint


def test_create_local_venv_refreshes_when_pyproject_changes(tmp_path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    pyproject_path = project_path / "pyproject.toml"
    pyproject_path.write_text("[project]\nname = 'test'\n")
    (project_path / "uv.lock").write_text("version = 1\n")
    venv_root = tmp_path / "venvs"
    venv_path = venv_root / "worker"
    py_executable = "uv run --locked --extra vllm"

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(venv_root)}),
        patch("nemo_rl.utils.venvs.git_root", str(project_path)),
        patch(
            "nemo_rl.utils.venvs.subprocess.run", side_effect=_mock_uv_venv
        ) as mock_run,
    ):
        create_local_venv(py_executable, "worker")
        first_fingerprint = (venv_path / ".nemo_rl_venv_spec").read_text()

        mock_run.reset_mock()
        pyproject_path.write_text(
            "[project]\nname = 'test'\ndependencies = ['new-dependency']\n"
        )
        create_local_venv(py_executable, "worker")

    assert mock_run.call_count == 3
    assert (venv_path / ".nemo_rl_venv_spec").read_text() != first_fingerprint


def test_create_local_venv_does_not_cache_failed_sync(tmp_path):
    venv_path = tmp_path / "worker"

    def fail_sync(command, **kwargs):
        _mock_uv_venv(command, **kwargs)
        if command[:2] == ["uv", "sync"]:
            raise subprocess.CalledProcessError(2, command)

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}),
        patch("nemo_rl.utils.venvs.subprocess.run", side_effect=fail_sync),
        pytest.raises(subprocess.CalledProcessError),
    ):
        create_local_venv("uv run --locked --extra vllm", "worker")

    assert not (venv_path / ".nemo_rl_venv_spec").exists()

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}),
        patch(
            "nemo_rl.utils.venvs.subprocess.run", side_effect=_mock_uv_venv
        ) as mock_run,
    ):
        create_local_venv("uv run --locked --extra vllm", "worker")

    assert mock_run.call_count == 3
    assert (venv_path / ".nemo_rl_venv_spec").is_file()


def test_add_hf_modules_cache_to_pythonpath():
    result = add_hf_modules_cache_to_pythonpath(
        {
            "HF_MODULES_CACHE": "/hf/modules",
            "PYTHONPATH": f"/project{os.pathsep}/other",
        }
    )

    assert result["PYTHONPATH"].split(os.pathsep) == [
        "/hf/modules",
        "/project",
        "/other",
    ]


def test_add_hf_modules_cache_does_not_duplicate_pythonpath_entry():
    pythonpath = f"/project{os.pathsep}/hf/modules"

    result = add_hf_modules_cache_to_pythonpath(
        {"HF_MODULES_CACHE": "/hf/modules", "PYTHONPATH": pythonpath}
    )

    assert result["PYTHONPATH"] == pythonpath


def test_make_actor_runtime_env_builds_local_venv_for_uv_python_executable():
    """Mirrors the inline venv-creation logic that used to live in grpo.py."""
    with (
        patch(
            "nemo_rl.distributed.ray_actor_environment_registry.get_actor_python_env",
            return_value="uv run --group vllm",
        ) as mock_get_env,
        patch(
            "nemo_rl.utils.venvs.create_local_venv_on_each_node",
            return_value="/fake/venv/bin/python",
        ) as mock_create_venv,
    ):
        runtime_env = make_actor_runtime_env("some.module.SomeActor")

    mock_get_env.assert_called_once_with("some.module.SomeActor")
    mock_create_venv.assert_called_once_with(
        "uv run --group vllm", "some.module.SomeActor"
    )
    assert runtime_env["py_executable"] == "/fake/venv/bin/python"
    assert runtime_env["env_vars"]["VIRTUAL_ENV"] == "/fake/venv"
    assert runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"] == "/fake/venv"
