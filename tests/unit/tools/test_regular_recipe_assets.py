# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf


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
