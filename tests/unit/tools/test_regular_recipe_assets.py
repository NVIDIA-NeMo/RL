# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import register_omegaconf_resolvers


def test_equivalence_template_uses_gym_component_working_directory():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    )
    resource = config.env.nemo_gym.equivalence_llm_judge.resources_servers.equivalence_llm_judge
    assert (
        resource.judge_prompt_template_fpath
        == "prompt_templates/equivalence_llm_judge.txt"
    )


def test_short_smoke_preserves_ten_update_warmup():
    register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    )
    scheduler = config.policy.megatron_cfg.scheduler
    assert config.grpo.max_num_steps == 3
    assert scheduler.lr_warmup_iters == 10
    assert scheduler.lr_decay_iters == 11
    assert scheduler.lr_decay_style == "constant"
