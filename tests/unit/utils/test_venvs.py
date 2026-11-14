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
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

import nemo_rl.utils.venvs as venv_module
from nemo_rl.utils.venvs import (
    add_hf_modules_cache_to_pythonpath,
    create_local_venv,
    make_actor_runtime_env,
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


def test_force_rebuild_replaces_a_link_to_a_shared_venv(tmp_path: Path) -> None:
    shared = tmp_path / "extras-vllm"
    (shared / "bin").mkdir(parents=True)
    (shared / "bin" / "python").touch()
    (tmp_path / "actor.Worker").symlink_to("extras-vllm")

    def fake_run(cmd: list[str], **kwargs: Any) -> None:
        if cmd[:2] == ["uv", "venv"]:
            os.makedirs(cmd[-1], exist_ok=True)

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}),
        patch.object(venv_module.subprocess, "run", side_effect=fake_run),
    ):
        create_local_venv.__wrapped__(
            py_executable="uv run --extra vllm",
            venv_name="actor.Worker",
            force_rebuild=True,
        )

    assert not (tmp_path / "actor.Worker").is_symlink()
    assert (tmp_path / "actor.Worker").is_dir()
    assert (shared / "bin" / "python").exists()


def test_force_rebuild_on_each_node_replaces_every_link_to_a_shared_venv(
    tmp_path: Path,
) -> None:
    """Two actors that link to one shared venv must each get their own on force rebuild.

    Drives the Ray task body, not just create_local_venv, because the task touches its
    STARTED_ENV_BUILDER marker through the actor's path before create_local_venv runs.
    """
    shared = tmp_path / "extras-vllm"
    (shared / "bin").mkdir(parents=True)
    (shared / "bin" / "python").touch()
    for actor in ("a.Worker", "b.Worker"):
        (tmp_path / actor).symlink_to("extras-vllm")

    def fake_run(cmd: list[str], **kwargs: Any) -> None:
        if cmd[:2] == ["uv", "venv"]:
            os.makedirs(cmd[-1], exist_ok=True)

    with (
        patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}),
        patch.object(venv_module.subprocess, "run", side_effect=fake_run),
        patch.object(venv_module.time, "sleep"),
        patch.object(
            venv_module, "create_local_venv", venv_module.create_local_venv.__wrapped__
        ),
    ):
        for actor in ("a.Worker", "b.Worker"):
            venv_module._env_builder._function("uv run --extra vllm", actor, 0, True)

    for actor in ("a.Worker", "b.Worker"):
        assert not (tmp_path / actor).is_symlink(), actor
        assert (tmp_path / actor).is_dir(), actor
    assert (shared / "bin" / "python").exists()
    assert not (shared / "STARTED_ENV_BUILDER").exists()


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


@pytest.fixture
def node_venv_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, str]]:
    monkeypatch.setenv("NEMO_RL_VENV_DIR", str(tmp_path))
    monkeypatch.setenv("NRL_FORCE_REBUILD_VENVS", "true")
    monkeypatch.setattr(venv_module, "G_BUILT_VENVS", {})
    monkeypatch.setattr(venv_module, "G_VENV_BUILD_LOCKS", {})
    nodes = [{"NodeID": "node-a", "Alive": True, "Resources": {"CPU": 4}}]
    pg = MagicMock()
    builder = MagicMock()
    python_path = str(tmp_path / "gym/bin/python")

    def ray_get(ref: Any) -> list[str] | None:
        return None if ref is pg.ready.return_value else [python_path]

    with (
        patch.object(venv_module.ray, "nodes", return_value=nodes) as get_nodes,
        patch.object(venv_module, "placement_group", return_value=pg) as make_pg,
        patch.object(venv_module, "_env_builder", builder),
        patch.object(venv_module.ray, "get", side_effect=ray_get) as get,
        patch.object(venv_module.ray.util, "remove_placement_group") as remove_pg,
    ):
        yield (
            get_nodes,
            make_pg,
            builder.options.return_value.remote,
            get,
            remove_pg,
            python_path,
        )


def test_repeated_force_rebuild_does_not_touch_prepared_venv(node_venv_builder) -> None:
    _, make_pg, build, _, remove_pg, python_path = node_venv_builder
    first_path = venv_module.create_local_venv_on_each_node(
        "uv run --extra nemo_gym", "gym"
    )
    second_path = venv_module.create_local_venv_on_each_node(
        "uv run --extra nemo_gym", "gym"
    )
    assert first_path == second_path == python_path
    make_pg.assert_called_once_with(bundles=[{"CPU": 1}], strategy="STRICT_SPREAD")
    build.assert_called_once_with("uv run --extra nemo_gym", "gym", 0, True)
    remove_pg.assert_called_once()


@pytest.mark.parametrize("changed_input", ["command", "directory", "nodes"])
def test_venv_cache_distinguishes_build_inputs(
    node_venv_builder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_input: str,
) -> None:
    get_nodes, make_pg, build, _, _, _ = node_venv_builder
    venv_module.create_local_venv_on_each_node("uv run --extra nemo_gym", "gym")
    command = "uv run --extra nemo_gym"
    if changed_input == "command":
        command = "uv run --extra vllm"
    elif changed_input == "directory":
        monkeypatch.setenv("NEMO_RL_VENV_DIR", str(tmp_path / "other"))
    else:
        get_nodes.return_value = [
            {"NodeID": "node-b", "Alive": True, "Resources": {"CPU": 4}}
        ]
    venv_module.create_local_venv_on_each_node(command, "gym")
    assert make_pg.call_count == 2
    assert build.call_count == 2


def test_failed_venv_build_is_retried_and_releases_placement_group(
    node_venv_builder,
) -> None:
    _, make_pg, build, get, remove_pg, python_path = node_venv_builder
    get.side_effect = [None, RuntimeError("uv failed"), None, [python_path]]
    with pytest.raises(RuntimeError, match="uv failed"):
        venv_module.create_local_venv_on_each_node("uv run --extra nemo_gym", "gym")
    assert venv_module.G_BUILT_VENVS == {}
    assert remove_pg.call_count == 1
    assert (
        venv_module.create_local_venv_on_each_node("uv run --extra nemo_gym", "gym")
        == python_path
    )
    assert make_pg.call_count == build.call_count == remove_pg.call_count == 2
