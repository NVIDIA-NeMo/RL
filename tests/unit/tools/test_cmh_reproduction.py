# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the portable wrapper, not a GPU certification."""

from pathlib import Path
import subprocess

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tools/super_rl/cmh_repro"


def test_fullscale_contract():
    cfg = OmegaConf.load(
        ROOT / "training_configs/super_rl/experiments/regular_s120_cmh_fullscale.yaml"
    )
    assert cfg.cluster.num_nodes == 64 and cfg.cluster.gpus_per_node == 4
    assert cfg.cluster.segment_size == 16
    assert cfg.policy.generation.colocated.resources.num_nodes == 48
    assert cfg.policy.train_global_batch_size == 4096
    assert (
        cfg.grpo.num_prompts_per_step == 256
        and cfg.grpo.num_generations_per_prompt == 16
    )
    assert (
        cfg.grpo.max_num_steps == 100
        and cfg.grpo.async_grpo.max_trajectory_age_steps == 2
    )
    assert cfg.policy.megatron_cfg.context_parallel_size == 4
    assert cfg.policy.megatron_cfg.expert_model_parallel_size == 16
    assert (
        cfg.policy.router_replay.enabled and cfg.policy.router_replay.transport == "ray"
    )
    assert cfg.policy.generation.max_new_tokens == 102400
    assert cfg.policy.max_total_sequence_length == 131072
    assert cfg.env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking
    assert not cfg.checkpointing.load_replay_buffer
    assert (
        cfg.checkpointing.save_period,
        cfg.checkpointing.ft_save_period,
        cfg.checkpointing.ft_keep_latest_k,
    ) == (10, 1, 1)
    assert cfg.env.nemo_gym.global_aiohttp_connector_limit_per_host == 16384
    assert "global_aiohttp_connector_limit" not in cfg.env.nemo_gym
    for name in ("math_with_judge", "equivalence_llm_judge"):
        judge = cfg.env.nemo_gym[name].resources_servers[name]
        assert judge.judge_max_attempts == 3 and judge.fail_on_missing_judge_verdict
        assert judge.judge_responses_create_params.max_output_tokens == 8192
        assert (
            judge.judge_responses_create_params.temperature
            == judge.judge_responses_create_params.top_p
            == 1.0
        )


@pytest.mark.parametrize(
    "name", ["submit.sh", "job.sbatch", "node.sh", "driver.sh", "profile.env"]
)
def test_shell_syntax(name):
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def test_unfilled_template_fails_before_submission():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "submit.sh"),
            str(ROOT / "training_configs/super_rl/cmh_repro.user.example.env"),
            "fresh",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Missing/unfilled/unsupported input: SUPER_RL_ROOT" in result.stderr


def test_no_personal_runtime_lineage_in_launch_files():
    for path in SCRIPTS.iterdir():
        if not path.is_file():
            continue
        body = path.read_text()
        assert "/users/jiachengx/" not in body
        assert "s120-cot-cmh-s100-3791850" not in body
    driver = (SCRIPTS / "driver.sh").read_text()
    assert "WANDB_RESUME=must" in driver and "WANDB_RESUME=never" in driver
    assert "checkpointing.load_replay_buffer=false" in driver


def test_clean_gym_staging(tmp_path):
    from tools.super_rl import stage_gym

    source = ROOT / "3rdparty/Gym-workspace/Gym"
    probe = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", stage_gym.GYM_BASE],
        capture_output=True,
    )
    if probe.returncode:
        pytest.fail("This release test requires the exact initialized Gym dependency")
    destination = tmp_path / "gym"
    stage_gym.stage(source, destination)
    assert (destination / "nemo_gym/judge_verdict.py").read_bytes() == (
        stage_gym.ASSETS / "gym_overlays/judge_verdict.py"
    ).read_bytes()
    assert (
        "budget.consume("
        in (destination / "responses_api_agents/simple_agent/app.py").read_text()
    )
