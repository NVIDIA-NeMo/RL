# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the strict Search-R1 training recipe."""

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import (
    load_config_with_inheritance,
    register_omegaconf_resolvers,
)


def test_search_r1_recipe_instantiates_master_config() -> None:
    repo_root = Path(__file__).parents[4]
    recipe = repo_root / "examples/nemo_gym/ai_search/grpo_qwen2_5_7b_search_r1.yaml"

    register_omegaconf_resolvers()
    raw_config = load_config_with_inheritance(recipe)
    resolved_config = OmegaConf.to_container(raw_config, resolve=True)

    assert isinstance(resolved_config, dict)
    config = MasterConfig(**resolved_config)
    assert config.policy["tokenizer"]["chat_template"] is None
    assert config.policy["generation"]["vllm_cfg"]["enforce_eager"] is True
