# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Validate matched Qwen30B MoE and Qwen32B dense CP pairs."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


RECIPE_DIR = (
    Path(__file__).resolve().parents[2] / "examples/configs/recipes/llm/performance"
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
    # Policy imports require the optional worker/transformers dependencies;
    # keep pure YAML comparisons runnable in the lightweight CPU environment.
    from nemo_rl.models.policy.dynamic_cp import _minimum_cp_size_for_experts

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


@pytest.mark.parametrize("model", ["qwen3-30ba3b", "qwen3-32b"])
def test_comparison_changes_only_cp_and_logging(model):
    register_omegaconf_resolvers()
    stem = f"grpo-{model}-4n4g-async-1off-megatron-"
    dynamic, static = [
        OmegaConf.to_container(
            load_config(RECIPE_DIR / f"{stem}{mode}cp-10step.yaml"), resolve=True
        )
        for mode in ("dynamic", "static")
    ]
    for config in (dynamic, static):
        config.pop("logger")
        config["policy"].pop("make_sequence_length_divisible_by")
        config["policy"]["megatron_cfg"].pop("context_parallel_size")
        config["policy"]["megatron_cfg"]["dynamic_context_parallel"].pop("enabled")
    assert dynamic == static


def test_dense_comparison_capacity_and_async_resources():
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(
        load_config(
            RECIPE_DIR / "grpo-qwen3-32b-4n4g-async-1off-megatron-dynamiccp-10step.yaml"
        ),
        resolve=True,
    )
    policy = resolved["policy"]
    megatron = policy["megatron_cfg"]
    assert policy["model_name"] == "Qwen/Qwen3-32B"
    assert policy["train_global_batch_size"] == 512
    assert policy["max_total_sequence_length"] == 16384
    assert megatron["tensor_model_parallel_size"] == 2
    assert megatron["pipeline_model_parallel_size"] == 1
    assert megatron["expert_model_parallel_size"] == 1
    assert megatron["activation_checkpointing"] is True
    assert megatron["dynamic_context_parallel"]["max_size"] == 4
    assert megatron["dynamic_context_parallel"]["tokens_per_rank"] == 4096
    assert resolved["cluster"]["num_nodes"] == 4
    assert resolved["grpo"]["async_grpo"]["enabled"] is True
    assert policy["generation"]["colocated"]["resources"]["num_nodes"] == 2
    assert policy["generation"]["vllm_cfg"]["tensor_parallel_size"] == 2
    assert resolved["logger"]["wandb_enabled"] is True
