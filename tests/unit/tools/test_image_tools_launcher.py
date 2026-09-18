# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import os
import json
import shlex
import subprocess

import pytest


@pytest.mark.parametrize("resume_step", [None, 20, 40])
def test_training_resume_restores_checkpoint_root_and_requires_wandb(tmp_path, resume_step):
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "tools/launch_image_tools_train_hsg.sh").read_text()
    block = launcher.split('if [[ -n "${RESUME_CHECKPOINT_DIR:-}" ]]; then', 1)[1].split(
        '\nexport CONTAINER=', 1
    )[0]
    block = 'if [[ -n "${RESUME_CHECKPOINT_DIR:-}" ]]; then' + block
    checkpoint_root = tmp_path / 'results/previous/checkpoints'
    checkpoint = checkpoint_root / 'step_20'
    checkpoint.mkdir(parents=True)
    (checkpoint_root / 'latest_checkpoint_status.json').write_text(json.dumps({'last_checkpoint_step': 20}))
    (checkpoint / 'training_info.json').write_text(json.dumps({'current_step': 20}))
    for name in ['config.yaml', 'train_dataloader.pt', 'rollouts.pt', 'replay_buffer.pt',
                 'policy/weights/iter_0000000/.metadata']:
        target = checkpoint / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('fixture')
    env = {k: v for k, v in os.environ.items() if k not in {'RESUME_CHECKPOINT_DIR', 'RESUME_STEP'}}
    env.update(OPERATION_ROOT=str(tmp_path), RUN_DIR=str(tmp_path / 'results/new'))
    if resume_step is not None:
        env.update(RESUME_CHECKPOINT_DIR=str(checkpoint_root), RESUME_STEP=str(resume_step))
    result = subprocess.run(['/bin/bash', '-ec', block + '\nprintf "%s|%s" "$CHECKPOINT_DIR" "$IMAGE_TOOLS_WANDB_RESUME"'],
                            env=env, text=True, capture_output=True)
    if resume_step == 40:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
        expected = f'{checkpoint_root}|must' if resume_step else f'{tmp_path}/results/new/checkpoints|allow'
        assert result.stdout.endswith(expected)


def test_actor_module_prewarm_is_serial_and_precedes_distributed_driver(tmp_path):
    root = Path(__file__).resolve().parents[3]
    driver = (root / "tools/image_tools_train_hsg.sh").read_text()
    block = driver.split("# Transformers versions", 1)[1].split(
        "exec /opt/nemo_rl_venv", 1
    )[0]
    # Execute the real loop against tiny interpreters, checking order and env.
    names = (
        "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
    )
    for name in names:
        executable = tmp_path / name / "bin/python"
        executable.parent.mkdir(parents=True)
        executable.write_text(
            '#!/bin/bash\nset -eu\ntest "${CUDA_VISIBLE_DEVICES}" = ""\n'
            f'printf "%s\\n" "{name}" "$@"\n'
        )
        executable.chmod(0o700)
    block = block[block.index("for component") :].replace(
        "/opt/ray_venvs", str(tmp_path)
    )
    output = subprocess.check_output(
        ["/bin/bash", "-ec", block],
        env={
            **os.environ,
            "MODEL_CHECKPOINT": "/base/model",
            "VLLM_TOKENIZER": "/run/tokenizer",
        },
        text=True,
    ).splitlines()
    assert output == [
        value
        for name in names
        for value in (
            name,
            "tools/prewarm_image_tools_model_modules.py",
            "--model",
            "/base/model",
            "--tokenizer",
            "/run/tokenizer",
        )
    ]


@pytest.mark.parametrize("requested, expected", [("INFO", "INFO"), (None, "WARN")])
def test_submission_nccl_debug_wins_over_credential_defaults(requested, expected):
    root = Path(__file__).resolve().parents[3]
    driver = (root / "tools/image_tools_train_hsg.sh").read_text()
    # Execute the actual shell block with a synthetic default, never real secrets.
    setup = driver.split("cd /opt/nemo-rl\n", 1)[1].split(': "${WANDB_API_KEY:', 1)[0]
    setup = setup.replace("source /opt/nemo-rl/.env", "export NCCL_DEBUG=WARN")
    env = {key: value for key, value in os.environ.items() if key != "NCCL_DEBUG"}
    if requested is not None:
        env["NCCL_DEBUG"] = requested
    output = subprocess.check_output(
        ["/bin/bash", "-c", "set -eu\n" + setup + 'printf "%s" "$NCCL_DEBUG"'],
        env=env,
        text=True,
    )
    assert output == expected


def test_slurm_wrap_is_posix_quoted_and_preserves_bash_arguments():
    launcher = (
        Path(__file__).resolve().parents[3] / "tools/launch_image_tools_runtime_hsg.sh"
    )
    assignment = next(
        line
        for line in launcher.read_text().splitlines()
        if line.startswith("wrapped_command=")
    )
    values = ["space value", "quote'value", "line\nbreak", "$not_expanded"]
    script = (
        "printf -v command '%q ' printf '%s\\n' \"$@\"\n"
        + assignment
        + '\nprintf "%s" "$wrapped_command"\n'
    )
    wrapper = subprocess.check_output(
        ["/bin/bash", "-c", script, "test", *values], text=True
    )
    tokens = shlex.split(wrapper, posix=True)
    assert tokens[:3] == ["exec", "/bin/bash", "-c"]
    assert len(tokens) == 4
    result = subprocess.check_output(["/bin/sh", "-c", wrapper], text=True)
    assert result == "\n".join(values) + "\n"


@pytest.mark.parametrize("suite", ["standalone", "visual-games"])
def test_training_preserves_bundled_uv_archive_and_checks_real_imports(tmp_path, suite):
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "tools/launch_image_tools_train_hsg.sh").read_text()
    # Execute the actual environment/mount setup, with a conflicting inherited
    # override, without submitting a job or touching cluster paths.
    setup = launcher.split("export NEMO_RL_VENV_DIR=", 1)[1].split("submit=(", 1)[0]
    script = "set -eu\nexport NEMO_RL_VENV_DIR=" + setup + "\n"
    script += 'test -z "${UV_CACHE_DIR_OVERRIDE+x}"\n'
    script += 'printf "%s\\n%s\\n" "$MOUNTS" "$SETUP_COMMAND"\n'
    env = {
        **os.environ,
        "RUN_DIR": "/fixture/results/run",
        "OPERATION_ROOT": "/fixture",
        "HSG_ROOT": "/fixture",
        "PROJECT_ROOT": "/fixture/snapshots/source",
        "MODEL_CHECKPOINT": "/fixture/model",
        "RUN_LOG_DIR": "/fixture/results/run/logs",
        "OVERLAY_DIR": "/fixture/overlay",
        "UV_CACHE_DIR_OVERRIDE": "/fixture/empty-cache",
        "IMAGE_TOOLS_SUITE": suite,
        "GAMES_VENV_ROOT": str(tmp_path / "game-venvs"),
        "EVAL_MANIFEST": "/fixture/heldout.jsonl",
        "VISGYM_ASSET_ARCHIVE": "/fixture/assets.bundle",
    }
    components = (
        "resources_servers/gym_v",
        "resources_servers/visgym",
        "responses_api_agents/gymv_agent",
        "responses_api_agents/visgym_agent",
    )
    for component in components:
        (tmp_path / "game-venvs" / component / ".venv").mkdir(parents=True)
    output = subprocess.check_output(["/bin/bash", "-c", script], env=env, text=True)
    assert ":/root" not in output.splitlines()[0]
    if suite == "visual-games":
        for component in components:
            assert f":/opt/gym_venvs/{component}:ro" in output.splitlines()[0]
    assert (
        "test -r /opt/nemo_rl_venv/lib/python3.13/site-packages/transformers/__init__.py"
        in output
    )
    assert "from transformers import AutoConfig, AutoProcessor, AutoTokenizer" in output
    driver = (root / "tools/image_tools_train_hsg.sh").read_text()
    assert 'export UV_CACHE_DIR="$RUN_DIR/cache/uv"' in driver
