# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE = (
    REPO_ROOT / "examples/configs/recipes/llm/"
    "grpo-nanov3-30ba3b-3n8g-megatron-dynamo-swe1.yaml"
)
DRIVER = (
    REPO_ROOT / "tests/test_suites/llm/grpo-nanov3-30ba3b-3n8g-megatron-dynamo-swe1.sh"
)


def _load_recipe() -> dict:
    register_omegaconf_resolvers()
    return OmegaConf.to_container(load_config(RECIPE), resolve=True)


def test_public_swe_recipe_has_supported_topology_and_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "/hf-home")
    config = _load_recipe()
    generation = config["policy"]["generation"]

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    configured_generation = configure_generation_config(generation, Tokenizer())
    validated = DynamoConfig.model_validate(configured_generation)

    assert config["policy"]["model_name"] == (
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    assert config["cluster"]["gpus_per_node"] == 8
    assert config["cluster"]["num_nodes"] == 3
    assert config["cluster"]["segment_size"] == 1
    assert config["grpo"]["max_num_steps"] == 4
    assert config["grpo"]["async_grpo"]["enabled"] is True
    assert config["grpo"]["async_grpo"]["in_flight_weight_updates"] is False
    assert (
        config["grpo"]["async_grpo"]["recompute_kv_cache_after_weight_updates"] is True
    )
    assert generation["colocated"]["resources"] == {
        "gpus_per_node": 8,
        "num_nodes": 1,
    }
    assert validated.engine_world_size == 4
    assert generation["vllm_cfg"]["expert_parallel_size"] == 4
    assert validated.dynamo_cfg.frontend_args.router_mode == "kv"
    assert validated.dynamo_cfg.control_timeout_s == 600
    assert validated.vllm_cfg.enable_vllm_metrics_logger is True
    assert validated.vllm_cfg.load_format == "dummy"
    assert config["env"]["nemo_gym"]["config_paths"][-1].endswith(
        "swe_pivot_single_step_tool_use_with_argument_comparison.yaml"
    )
    assert (
        config["env"]["nemo_gym"]["policy_model"]["responses_api_models"]["vllm_model"][
            "chat_template_kwargs"
        ]["force_nonempty_content"]
        is True
    )
    assert (
        config["env"]["nemo_gym"]["single_step_tool_use_with_argument_comparison_swe"][
            "responses_api_agents"
        ]["tool_simulation_agent"]["resources_server"]["name"]
        == "swe_pivot_single_step_tool_use_with_argument_comparison_resources_server"
    )
    assert config["logger"]["wandb_enabled"] is True
    assert config["logger"]["tensorboard_enabled"] is True
    assert config["logger"]["wandb"]["project"] == "nemo-rl"
    assert config["data"]["train"]["data_path"].endswith(
        "/superv3_data/swe1/train-split.jsonl"
    )
    assert config["data"]["validation"]["data_path"].endswith(
        "/superv3_data/swe1/val-split.jsonl"
    )


def test_recipe_and_driver_have_no_removed_swe_modes() -> None:
    recipe_text = RECIPE.read_text(encoding="utf-8")
    text = recipe_text + DRIVER.read_text(encoding="utf-8")
    for forbidden in (
        "/lustre/",
        "/path/to",
        "/home/",
        "jthomson",
        "openhands",
        "container_formatter",
        "SIF_",
        "effort_levels",
        "hsg_r2",
        "USES_SANDBOX",
    ):
        assert forbidden.lower() not in text.lower()
    assert "load_format:" not in recipe_text
    assert "--require-tag-prefix generation_metrics/" in text
    assert "project: nemo-rl" in text
    assert "Expected step ${MAX_STEPS}" in text
    assert 'median(data["train/token_mult_prob_error"]) < 1.1' in text
    assert "data['train/token_mult_prob_error']['${MAX_STEPS}'] < 1.1" in text
    assert 'mean(data["train/gen_kl_error"]) < 0.02' in text
    assert "Expected one cache invalidation per refit" in text
