# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
R1_CONFIG = REPO_ROOT / "examples/swe_bench/grpo_nano_v3_5_swe_dynamo_hsg.yaml"
R2_CONFIG = (
    REPO_ROOT
    / "examples/swe_bench/grpo_nano_v3_5_swe_dynamo_hsg_r2_wandb.yaml"
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _remove(config: dict, *path: str) -> None:
    parent = config
    for key in path[:-1]:
        parent = parent[key]
    parent.pop(path[-1], None)


def test_r2_only_changes_scale_steps_and_wandb() -> None:
    r1 = _load(R1_CONFIG)
    r2 = _load(R2_CONFIG)
    allowed_paths = (
        ("cluster", "num_nodes"),
        ("grpo", "num_prompts_per_step"),
        ("grpo", "max_num_steps"),
        ("policy", "train_global_batch_size"),
        (
            "policy",
            "generation",
            "vllm_cfg",
            "enable_vllm_metrics_logger",
        ),
        (
            "policy",
            "generation",
            "vllm_cfg",
            "vllm_metrics_logger_interval",
        ),
        ("policy", "generation", "colocated", "resources", "num_nodes"),
        ("logger", "wandb_enabled"),
        ("logger", "wandb", "name"),
    )
    normalized_r1 = deepcopy(r1)
    normalized_r2 = deepcopy(r2)
    for path in allowed_paths:
        _remove(normalized_r1, *path)
        _remove(normalized_r2, *path)

    assert normalized_r2 == normalized_r1


def test_r2_topology_batch_and_wandb_contract() -> None:
    config = _load(R2_CONFIG)
    generation = config["policy"]["generation"]

    assert config["cluster"] == {
        "gpus_per_node": 4,
        "num_nodes": 6,
        "segment_size": 1,
    }
    assert config["grpo"]["num_prompts_per_step"] == 2
    assert config["grpo"]["num_generations_per_prompt"] == 2
    assert config["grpo"]["max_num_steps"] == 4
    assert config["policy"]["train_global_batch_size"] == 4
    assert generation["dynamo_cfg"]["engine_world_size"] == 4
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 4
    assert generation["vllm_cfg"]["pipeline_parallel_size"] == 1
    assert generation["colocated"]["resources"] == {
        "gpus_per_node": 4,
        "num_nodes": 2,
    }
    assert generation["vllm_cfg"]["enable_vllm_metrics_logger"] is True
    assert generation["vllm_cfg"]["vllm_metrics_logger_interval"] == 0.5
    assert config["logger"]["wandb_enabled"] is True
    assert config["logger"]["wandb"]["project"] == "nemo-rl-dynamo-swe"

    training_world_size = (6 - 2) * 4
    inference_world_size = 2 * generation["dynamo_cfg"]["engine_world_size"]
    assert training_world_size == 16
    assert inference_world_size == 8
    assert training_world_size + inference_world_size == 24
    assert [training_world_size + worker * 4 for worker in range(2)] == [16, 20]
    assert config["policy"]["train_global_batch_size"] == (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
    )
