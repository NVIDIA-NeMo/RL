# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Resolve the real-model dynamic-CP MoE smoke recipes and validate topology."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.models.policy.dynamic_cp import (
    _minimum_cp_size_for_experts,
    validate_dynamic_cp,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


RECIPE_DIR = (
    Path(__file__).resolve().parents[2] / "examples/configs/recipes/llm/performance"
)
RECIPES = (
    (
        "grpo-qwen3-30ba3b-4n4g-async-1off-megatron-dynamiccp-quick.yaml",
        "aux_loss",
        8,
        4,
        2,
    ),
    (
        "grpo-qwen3-235b-32n4g-async-1off-megatron-dynamiccp-quick.yaml",
        "seq_aux_loss",
        2,
        32,
        16,
    ),
    (
        "grpo-nemotron3-nano-30ba3b-8n4g-megatron-dynamiccp-quick.yaml",
        "none",
        4,
        8,
        8,
    ),
)

# These benchmark recipes are intentionally local/ignored. Keep the topology
# checks useful for local recipe development without breaking a clean checkout.
pytestmark = pytest.mark.skipif(
    any(not (RECIPE_DIR / recipe_name).exists() for recipe_name, *_ in RECIPES),
    reason="local dynamic-CP benchmark recipes are not present",
)


@pytest.mark.parametrize(
    "recipe_name,routing_type,effective_minimum,total_nodes,expected_policy_nodes",
    RECIPES,
)
def test_dynamic_cp_moe_recipe_topology(
    recipe_name,
    routing_type,
    effective_minimum,
    total_nodes,
    expected_policy_nodes,
):
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(
        load_config(RECIPE_DIR / recipe_name), resolve=True
    )
    assert isinstance(resolved, dict)
    policy = resolved["policy"]
    megatron = policy["megatron_cfg"]
    generation = policy["generation"]["colocated"]
    cluster = resolved["cluster"]
    generation_nodes = generation["resources"]["num_nodes"] or 0
    policy_nodes = (
        cluster["num_nodes"]
        if generation["enabled"]
        else cluster["num_nodes"] - generation_nodes
    )
    policy_world_size = policy_nodes * cluster["gpus_per_node"]
    lanes = policy_world_size // (
        megatron["tensor_model_parallel_size"]
        * megatron["pipeline_model_parallel_size"]
    )

    assert megatron["pipeline_model_parallel_size"] == 1
    assert megatron["context_parallel_size"] == 1
    assert megatron["expert_tensor_parallel_size"] == 1
    assert megatron["moe_router_load_balancing_type"] == routing_type
    assert _minimum_cp_size_for_experts(megatron, 1) == effective_minimum
    assert cluster["num_nodes"] == total_nodes
    assert policy_nodes == expected_policy_nodes
    assert resolved["logger"]["wandb_enabled"] is True
    assert resolved["logger"]["wandb"]["project"] == "nemo-rl"
    assert resolved["logger"]["wandb"]["name"]
    validate_dynamic_cp(policy, lanes=lanes)
