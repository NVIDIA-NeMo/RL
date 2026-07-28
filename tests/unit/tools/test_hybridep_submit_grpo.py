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

import os
import shlex
import subprocess
from pathlib import Path

import pytest


def _run_launcher(
    tmp_path: Path,
    *,
    dispatcher_mode: str,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    )
    model_config = launcher.parent / "models" / "qwen3-30ba3b-4n4g.env"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/bin/sh
case "$*" in
  *"rev-parse --show-toplevel"*) printf '%s\\n' "$FAKE_PROJECT_ROOT" ;;
  *"rev-parse HEAD"*) printf '%s\\n' "1342aa09a0e0903f4299390e5d9fc1d88f3b75eb" ;;
esac
"""
    )
    fake_git.chmod(0o755)

    fake_sshare = fake_bin / "sshare"
    fake_sshare.write_text(
        """#!/bin/sh
printf '%s|%s|%s|\\n' 'nemotron_sw_pre' "$FAKE_USER" '0.900000'
"""
    )
    fake_sshare.chmod(0o755)

    command_capture = tmp_path / "command.txt"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/bin/sh
if [ "$1" = "--test-only" ]; then
  printf '%s\\n' 'test-only accepted'
  exit 0
fi
printf '%s\\n' "$COMMAND" > "$COMMAND_CAPTURE"
printf '%s\\n' '999999'
"""
    )
    fake_sbatch.chmod(0o755)

    container = tmp_path / "nightly.sqsh"
    container.touch()
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_PROJECT_ROOT": str(project_root),
        "FAKE_USER": subprocess.check_output(["id", "-un"], text=True).strip(),
        "COMMAND_CAPTURE": str(command_capture),
        "CONTAINER": str(container),
        "RUN_ROOT": str(tmp_path / "run"),
        "RUN_SUFFIX": "test",
        "DISPATCHER_MODE": dispatcher_mode,
    }
    if extra_env:
        env.update(extra_env)

    subprocess.run(
        ["bash", str(launcher), str(model_config)],
        check=True,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )
    return shlex.split(command_capture.read_text())


def test_recipe_dispatcher_preserves_recipe_default(tmp_path: Path) -> None:
    driver_args = _run_launcher(tmp_path, dispatcher_mode="recipe")

    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" not in driver_args
    assert "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" not in driver_args
    assert "++policy.megatron_cfg.moe_hybridep_num_sms=32" not in driver_args


def test_unknown_dispatcher_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _run_launcher(tmp_path, dispatcher_mode="unknown")


def test_padding_logging_reaches_megatron_worker_environment(tmp_path: Path) -> None:
    driver_args = _run_launcher(
        tmp_path,
        dispatcher_mode="hybridep",
        extra_env={
            "NEMO_RL_HYBRIDEP_LOG_PACKING": "1",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS": "4096",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS": "0",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE": "1",
        },
    )

    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING='1'"
        in driver_args
    )
    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS='4096'"
        in driver_args
    )
    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS='0'"
        in driver_args
    )
    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE='1'"
        in driver_args
    )


def test_submission_metadata_records_dispatcher_and_padding_logging(
    tmp_path: Path,
) -> None:
    _run_launcher(
        tmp_path,
        dispatcher_mode="hybridep",
        extra_env={
            "NEMO_RL_HYBRIDEP_LOG_PACKING": "1",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS": "4096",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS": "0",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE": "1",
        },
    )

    metadata = (tmp_path / "run" / "submission.env").read_text()
    assert "dispatcher_mode=hybridep\n" in metadata
    assert "padding_log_enabled=1\n" in metadata
    assert "padding_log_max_calls=4096\n" in metadata
    assert "padding_log_ranks=0\n" in metadata
    assert "padding_log_reduce=1\n" in metadata


def test_hybridep_launcher_preserves_project_and_bridge_pythonpath() -> None:
    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    )
    source = launcher.read_text()

    project_path = 'PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"'
    bridge_path = 'PYTHONPATH="${BRIDGE_SRC}:${PYTHONPATH}"'
    overlay_path = 'PYTHONPATH="${DEEPEP_OVERLAY}:${PYTHONPATH}"'
    export_path = "export PYTHONPATH"

    assert project_path in source
    assert bridge_path in source
    assert overlay_path in source
    assert export_path in source
    assert source.index(project_path) < source.index(bridge_path)
    assert source.index(bridge_path) < source.index(overlay_path)
    assert source.index(overlay_path) < source.index(export_path)
