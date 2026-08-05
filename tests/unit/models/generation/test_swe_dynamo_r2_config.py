# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE = (
    REPO_ROOT / "examples/configs/recipes/llm/"
    "grpo-nanov3-30ba3b-swe-6n4g-megatron-dynamo-wandb.yaml"
)
LAUNCHER = (
    REPO_ROOT / "examples/swe_bench/run_grpo_nanov3_30ba3b_swe_dynamo_hsg_r2_wandb.sh"
)


def _load_recipe() -> dict:
    register_omegaconf_resolvers()
    return OmegaConf.to_container(load_config(RECIPE), resolve=True)


def test_public_swe_recipe_has_supported_topology_and_telemetry() -> None:
    config = _load_recipe()
    generation = config["policy"]["generation"]
    validated = DynamoConfig.model_validate(generation)

    assert config["policy"]["model_name"] == (
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    assert config["cluster"]["gpus_per_node"] == 4
    assert config["cluster"]["num_nodes"] == 6
    assert config["cluster"]["segment_size"] == 1
    assert config["grpo"]["max_num_steps"] == 4
    assert config["grpo"]["async_grpo"]["enabled"] is True
    assert config["grpo"]["async_grpo"]["in_flight_weight_updates"] is False
    assert generation["colocated"]["resources"] == {
        "gpus_per_node": 4,
        "num_nodes": 2,
    }
    assert validated.engine_world_size == 4
    assert validated.dynamo_cfg.frontend_args.router_mode == "kv"
    assert validated.dynamo_cfg.control_timeout_s == 600
    assert validated.vllm_cfg.enable_vllm_metrics_logger is True
    assert validated.vllm_cfg.load_format == "dummy"
    assert config["env"]["nemo_gym"]["effort_levels"]["low_ub"] == 15000
    assert config["reward_penalties"]["penalize_unwanted_tokens"] is False
    assert config["logger"]["wandb_enabled"] is True


def test_recipe_and_launcher_have_no_private_or_interpolated_defaults() -> None:
    text = RECIPE.read_text(encoding="utf-8") + LAUNCHER.read_text(encoding="utf-8")
    for forbidden in (
        "/lustre/",
        "/path/to",
        "/home/",
        "jthomson",
        "${oc.env:",
        "NEMO_RL_PY_EXECUTABLES_SYSTEM",
        "engine_world_size:",
        "dynamo_python:",
        "penalize_eos_token",
    ):
        assert forbidden not in text
    assert "WANDB_API_KEY" in text
    assert "BUILD_DYNAMO=1" in text
