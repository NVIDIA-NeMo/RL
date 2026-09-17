# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Validate that the Qwen30B MoE dynamic/static CP pair is matched."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.models.policy.dynamic_cp import _minimum_cp_size_for_experts
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


RECIPE_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples/configs/recipes/llm/performance"
)


@pytest.mark.parametrize(
    "recipe_name,dynamic_enabled,context_parallel_size,pad_factor",
    (
        (
            "grpo-qwen3-30ba3b-4n4g-async-1off-megatron-dynamiccp-10step.yaml",
            True,
            1,
            4,
        ),
        (
            "grpo-qwen3-30ba3b-4n4g-async-1off-megatron-staticcp-10step.yaml",
            False,
            2,
            16,
        ),
    ),
)
def test_cp_comparison_recipe_pair(
    recipe_name, dynamic_enabled, context_parallel_size, pad_factor
):
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(
        load_config(RECIPE_DIR / recipe_name), resolve=True
    )
    assert isinstance(resolved, dict)

    assert resolved["grpo"]["max_num_steps"] == 10
    assert resolved["grpo"]["num_prompts_per_step"] == 16
    assert resolved["grpo"]["num_generations_per_prompt"] == 32
    assert resolved["policy"]["train_global_batch_size"] == 512
    assert resolved["policy"]["max_total_sequence_length"] == 8192
    assert resolved["policy"]["make_sequence_length_divisible_by"] == pad_factor
    assert resolved["policy"]["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert resolved["cluster"]["num_nodes"] == 4
    assert resolved["policy"]["generation"]["colocated"]["enabled"] is False
    assert resolved["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 2

    megatron = resolved["policy"]["megatron_cfg"]
    assert megatron["tensor_model_parallel_size"] == 4
    assert megatron["pipeline_model_parallel_size"] == 1
    assert megatron["expert_model_parallel_size"] == 4
    assert megatron["expert_tensor_parallel_size"] == 1
    assert _minimum_cp_size_for_experts(megatron, 1) == 1
    assert megatron["context_parallel_size"] == context_parallel_size
    assert megatron["dynamic_context_parallel"]["enabled"] is dynamic_enabled
    assert megatron["dynamic_context_parallel"]["max_size"] == 2
    assert megatron["dynamic_context_parallel"]["tokens_per_rank"] == 4096

    assert resolved["logger"]["wandb_enabled"] is True
    assert resolved["logger"]["wandb"]["project"] == "nemo-rl-cp-comparison"
