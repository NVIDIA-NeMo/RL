# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    validate_single_controller_config,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    ROOT
    / "examples/configs/recipes/vlm"
    / "vlm_grpo-nemotron-nano-omni-30ba3b-2n8g-megatron-agentic-vbench-sc-256k.v1.yaml"
)
SUPER_RECIPE = (
    ROOT
    / "examples/configs/recipes/vlm"
    / "vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml"
)
LAUNCHER = ROOT / "scripts/agentic_vbench/debug_subagents_sync_2n_nano.sh"


def _load_resolved_recipe() -> tuple[dict, MasterConfig]:
    register_omegaconf_resolvers()
    env = {
        "MODEL_PATH": "/tmp/nano-omni-model",
        "TRAIN_PATH": "/tmp/agentic-vbench-train.jsonl",
        "VAL_PATH": "/tmp/agentic-vbench-train.jsonl",
    }
    with patch.dict(os.environ, env, clear=False):
        resolved = OmegaConf.to_container(load_config(RECIPE), resolve=True)
    assert isinstance(resolved, dict)
    # Match the subagent wrapper: framework-owned parent/child context segments
    # and exact routed-expert replay are on, while OpenCode compaction stays off.
    resolved["token_capture"]["context_compaction"] = True
    resolved["policy"].setdefault("router_replay", {})["enabled"] = True
    config = MasterConfig.model_validate(resolved)
    validate_single_controller_config(config)
    return resolved, config


def test_nano_sync2_recipe_preserves_agentic_subagent_gate_contract() -> None:
    cfg, config = _load_resolved_recipe()

    assert {
        key: cfg["cluster"][key]
        for key in ("num_nodes", "gpus_per_node", "segment_size")
    } == {
        "num_nodes": 2,
        "gpus_per_node": 8,
        "segment_size": 1,
    }
    assert cfg["policy"]["generation"]["colocated"]["resources"] == {
        "gpus_per_node": 8,
        "num_nodes": 1,
    }
    assert cfg["policy"]["train_global_batch_size"] == 16
    assert cfg["policy"]["sequence_packing"]["enabled"] is False
    assert cfg["policy"]["dynamic_batching"]["enabled"] is False
    parallel = cfg["policy"]["megatron_cfg"]
    assert {
        key: parallel[key]
        for key in (
            "tensor_model_parallel_size",
            "expert_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        )
    } == {
        "tensor_model_parallel_size": 8,
        "expert_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
    }
    assert config.policy["megatron_cfg"]["context_parallel_size"] == 1
    assert not config.policy["sequence_packing"]["enabled"]
    assert cfg["grpo"]["num_prompts_per_step"] == 4
    assert cfg["grpo"]["num_generations_per_prompt"] == 4
    assert cfg["grpo"]["max_num_steps"] == 1_000_000
    assert cfg["grpo"]["max_num_epochs"] is None
    assert cfg["grpo"]["async_grpo"] is None
    assert cfg["grpo"]["debug_payload_metrics"] is True

    # Diagnostics must not hide parity problems or change the objective.
    assert cfg["grpo"]["seq_logprob_error_threshold"] is None
    assert cfg["grpo"]["advantage_clip_low"] is None
    assert cfg["grpo"]["advantage_clip_high"] is None
    assert cfg["grpo"]["use_dynamic_sampling"] is False
    assert cfg["grpo"]["zero_variance_prompt_filtering"] is False
    assert cfg["async_rl"]["sampler"] == {
        "name": "in_order",
        "max_lookahead_versions": 0,
    }
    assert cfg["env"]["nemo_gym"]["config_paths"][-1].endswith(
        "agentic_vbench_train.yaml"
    )
    assert cfg["logger"]["wandb"]["project"] == ("nemorl-v2-subagent-comapction-debug")


def test_nano_sync2_launcher_reuses_exact_subagent_prompt_and_r3_wrapper() -> None:
    text = LAUNCHER.read_text()
    assert "debug_subagents_sync_8n.sh" in text
    assert "NUM_NODES=2" in text
    assert "NUM_GEN_NODES=1" in text
    assert "SEGMENT_SIZE=1" in text
    assert "NUM_PROMPTS=4" in text
    assert "NUM_GENERATIONS=4" in text
    assert "MAX_STEPS=1000000" in text
    assert "AGENTIC_VBENCH_TASKS=assembly" in text
    assert "assembly_first4_x4.jsonl" in text
    assert "backend-subagent-sync8-20261014-v20-long" in text
    assert "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" in text
    assert "TOKENIZER_CHAT_TEMPLATE" in text
    assert "VLLM_CHAT_TEMPLATE" in text


def test_exact_nano_and_super_recipes_accept_shared_optional_overrides() -> None:
    register_omegaconf_resolvers()
    env = {
        "MODEL_PATH": "/tmp/omni-model",
        "TRAIN_PATH": "/tmp/agentic-vbench-train.jsonl",
        "VAL_PATH": "/tmp/agentic-vbench-train.jsonl",
    }
    overrides = [
        "++policy.generation.vllm_cfg.load_format=auto",
        "++policy.tokenizer.chat_template=/tmp/chat_template.jinja",
        "++policy.generation.vllm_cfg.http_server_serving_chat_kwargs."
        "chat_template=/tmp/chat_template.jinja",
    ]

    with patch.dict(os.environ, env, clear=False):
        for recipe in (RECIPE, SUPER_RECIPE):
            updated = parse_hydra_overrides(load_config(recipe), overrides)
            assert updated.policy.generation.vllm_cfg.load_format == "auto"
            assert updated.policy.tokenizer.chat_template.endswith(".jinja")
            assert updated.policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template.endswith(
                ".jinja"
            )
