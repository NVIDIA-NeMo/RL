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
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import nemo_rl.utils.venvs as venvs
from nemo_rl.utils.venvs import (
    create_local_venv,
    create_local_venv_on_each_node,
    make_python_runtime_env,
)
from tests.unit.conftest import TEST_ASSETS_DIR


def test_make_python_runtime_env_activates_configured_venv_without_mutating_input(
    tmp_path,
):
    venv = tmp_path / "configured-venv"
    python = venv / "bin" / "python"
    base_env = {
        "PATH": f"/opt/nemo_rl_venv/bin{os.pathsep}/usr/bin",
        "PYTHONPATH": "/explicit/source/tree",
        "VIRTUAL_ENV": "/opt/nemo_rl_venv",
        "UV_PROJECT_ENVIRONMENT": "/opt/nemo_rl_venv",
    }
    original_base_env = base_env.copy()

    runtime_env = make_python_runtime_env(str(python), base_env=base_env)

    assert runtime_env["py_executable"] == str(python)
    assert runtime_env["env_vars"] == {
        "PATH": (
            f"{venv / 'bin'}{os.pathsep}/opt/nemo_rl_venv/bin{os.pathsep}/usr/bin"
        ),
        "PYTHONPATH": "/explicit/source/tree",
        "VIRTUAL_ENV": str(venv),
        "UV_PROJECT_ENVIRONMENT": str(venv),
    }
    assert base_env == original_base_env


def test_make_python_runtime_env_preserves_venv_python_symlink(tmp_path):
    interpreter = tmp_path / "python3.13"
    interpreter.touch()
    venv = tmp_path / "configured-venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.symlink_to(interpreter)

    runtime_env = make_python_runtime_env(str(python), base_env={})

    assert runtime_env["py_executable"] == str(python)
    assert runtime_env["py_executable"] != str(python.resolve())
    assert runtime_env["env_vars"]["VIRTUAL_ENV"] == str(venv)


def test_create_local_venv_on_each_node_propagates_configured_root(
    tmp_path, monkeypatch
):
    venv_root = tmp_path / "shared-venvs"
    venv_name = "example.Worker"
    python = venv_root / venv_name / "bin" / "python"
    monkeypatch.setenv("NEMO_RL_VENV_DIR", str(venv_root))

    placement = MagicMock()
    placement.ready.return_value = "placement-ready"
    builder = MagicMock()
    builder.options.return_value.remote.return_value = "builder-result"

    with (
        patch.object(
            venvs.ray,
            "nodes",
            return_value=[{"Alive": True, "Resources": {"CPU": 1}}],
        ),
        patch.object(venvs, "placement_group", return_value=placement),
        patch.object(
            venvs.ray,
            "get",
            side_effect=[None, [str(python)]],
        ) as ray_get,
        patch.object(venvs, "_env_builder", builder),
        patch.object(venvs.ray.util, "remove_placement_group") as remove_pg,
    ):
        result = create_local_venv_on_each_node("uv run --locked", venv_name)

    assert result == str(python)
    builder.options.assert_called_once_with(
        placement_group=placement,
        runtime_env={"env_vars": {"NEMO_RL_VENV_DIR": str(venv_root)}},
    )
    builder.options.return_value.remote.assert_called_once_with(
        "uv run --locked", venv_name, 0, False
    )
    assert ray_get.call_args_list == [call("placement-ready"), call(["builder-result"])]
    remove_pg.assert_called_once_with(placement)


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
