# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

import yaml
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE = (
    REPO_ROOT / "examples/configs/recipes/llm/"
    "grpo-nemotron-nano-v3.5-swe-6n4g-megatron-dynamo-wandb.yaml"
)
LAUNCHER = (
    REPO_ROOT / "examples/swe_bench/run_grpo_nano_v3_5_swe_dynamo_hsg_r2_wandb.sh"
)


def _load_recipe(monkeypatch) -> dict:
    monkeypatch.setenv("MODEL_PATH", "/models/nemotron")
    monkeypatch.setenv("TRAIN_PATH", "/data/train.jsonl")
    monkeypatch.setenv("VAL_PATH", "/data/val.jsonl")
    monkeypatch.setenv("SIF_FORMATTERS", '["/shared/{instance_id}.sif"]')
    register_omegaconf_resolvers()
    return OmegaConf.to_container(load_config(RECIPE), resolve=True)


def test_swe_recipe_topology_refit_and_wandb_contract(monkeypatch) -> None:
    config = _load_recipe(monkeypatch)
    generation = config["policy"]["generation"]

    with RECIPE.open(encoding="utf-8") as config_file:
        assert yaml.safe_load(config_file)["defaults"] == (
            "../../../nemo_gym/grpo_nanov3.yaml"
        )
    assert config["cluster"]["gpus_per_node"] == 4
    assert config["cluster"]["num_nodes"] == 6
    assert config["grpo"]["max_num_steps"] == 4
    assert config["grpo"]["async_grpo"]["enabled"] is True
    assert config["policy"]["train_global_batch_size"] == 4
    assert generation["backend"] == "dynamo"
    assert generation["colocated"] == {
        "enabled": False,
        "resources": {"gpus_per_node": 4, "num_nodes": 2},
    }
    assert generation["dynamo_cfg"]["engine_world_size"] == 4
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 4
    assert generation["vllm_cfg"]["pipeline_parallel_size"] == 1
    assert generation["vllm_cfg"]["enable_vllm_metrics_logger"] is True
    assert generation["vllm_cfg"]["vllm_metrics_logger_interval"] == 0.5
    assert config["logger"]["wandb_enabled"] is True

    training_world_size = 4 * 4
    inference_world_size = 2 * generation["dynamo_cfg"]["engine_world_size"]
    assert (training_world_size, inference_world_size) == (16, 8)
    assert [training_world_size + worker * 4 for worker in range(2)] == [16, 20]


def test_swe_recipe_and_launcher_have_no_private_paths() -> None:
    text = RECIPE.read_text(encoding="utf-8") + LAUNCHER.read_text(encoding="utf-8")

    assert "/lustre/" not in text
    assert "portfolios/" not in text
    assert "coreai_" not in text
    assert "${oc.env:MODEL_PATH}" in text
    assert "${oc.env:TRAIN_PATH}" in text
    assert "${oc.env:VAL_PATH}" in text
    assert "WANDB_API_KEY" in text
