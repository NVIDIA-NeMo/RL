# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.super_rl.prepare_node import NATIVE_IMPORT_CHECK, prepare


@pytest.mark.parametrize("matching_source", [True, False])
def test_native_import_checks_actual_source(tmp_path: Path, matching_source: bool):
    source = tmp_path / "fixture_component.py"
    source.write_text("VALUE = 1\n")
    expected = source if matching_source else tmp_path / "different.py"
    result = subprocess.run(
        [sys.executable, "-c", NATIVE_IMPORT_CHECK, "fixture_component", str(expected)],
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    if matching_source:
        assert result.returncode == 0
        assert str(source) in result.stdout
    else:
        assert result.returncode != 0
        assert "Native Gym import source differs" in result.stderr


def test_prepare_selects_staged_component_root(tmp_path: Path):
    config = tmp_path / "recipe.yaml"
    config.write_text(
        "env:\n  nemo_gym:\n    config_paths: []\n    fixture:\n"
        "      resources_servers:\n        example:\n          entrypoint: app.py\n"
    )
    gym = tmp_path / "staged"
    gym.mkdir()
    image_venvs = tmp_path / "image_venvs"
    interpreter = image_venvs / "resources_servers/example/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    runtime_venvs = tmp_path / "runtime_venvs"
    with patch("tools.super_rl.prepare_node.subprocess.run") as run:
        receipt = prepare(config, gym, image_venvs, runtime_venvs)
    command = run.call_args.args[0]
    assert command[0] == str(interpreter)
    assert command[-2:] == [
        "resources_servers.example.app",
        str(gym / "resources_servers/example/app.py"),
    ]
    assert run.call_args.kwargs["env"]["NEMO_GYM_EXTRA_ROOTS"] == str(gym)
    assert receipt["complete"] is True
    assert receipt["components"] == ["resources_servers.example.app"]
