# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run NeMo Gym trajectory collection against an existing Dynamo deployment.

This entrypoint is intentionally narrower than ``run_grpo_nemo_gym.py``. It
does not construct the train policy or logprob/training stack, so smoke tests
can reserve all GPUs for Dynamo rollout serving.
"""

import argparse
import os
import pprint
from typing import Any

import ray
import wandb.util
from omegaconf import OmegaConf
from wandb import Table

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import NemoGymConfig, setup_nemo_gym_config
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo import DynamoGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.dynamo_prometheus import maybe_start_dynamo_prometheus_monitor
from nemo_rl.utils.logger import Logger, get_next_experiment_dir

wandb.util.VALUE_BYTES_LIMIT = 10_000_000


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run NeMo Gym rollout-only trajectory collection with Dynamo"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_known_args()


def _materialize_dynamo_generation_config(config: MasterConfig) -> dict[str, Any]:
    generation_config = config["policy"]["generation"]
    generation_config["model_name"] = config["policy"]["model_name"]
    vllm_cfg = generation_config.setdefault("vllm_cfg", {})
    vllm_cfg.setdefault("max_model_len", config["policy"]["max_total_sequence_length"])
    return generation_config


def _build_one_batch(dataset: Any):
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("No samples available for rollout-only trajectory collection")
    return rl_collate_fn([dataset[i] for i in range(len(dataset))])


def _log_trajectory_collection(logger: Logger, rollout_result) -> None:
    rows_to_log: list[str] = []
    scalar_metrics: dict[str, float | int | bool] = {}

    for key, value in rollout_result.rollout_metrics.items():
        if "full_result" in key:
            value: Table
            rows_to_log.extend(v[0] for v in value.data)
        elif isinstance(value, (bool, int, float)):
            scalar_metrics[key] = value

    logger.log_string_list_as_jsonl(rows_to_log, "trajectory_collection.jsonl")
    logger.log_metrics(
        scalar_metrics,
        step=0,
        prefix="trajectory_collection",
        step_finished=True,
    )


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for Dynamo rollout-only collection"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )
    generation_config = _materialize_dynamo_generation_config(config)

    setup_nemo_gym_config(config, tokenizer)
    config["env"]["nemo_gym"].pop("is_trajectory_collection", None)

    print("\nSetting up data...")
    train_dataset, val_dataset = setup_response_data(
        tokenizer, config["data"], env_configs=None
    )
    rollout_dataset = val_dataset if val_dataset is not None else train_dataset
    rollout_batch = _build_one_batch(rollout_dataset)

    print("Final rollout-only config:")
    pprint.pprint(config)

    init_ray()
    logger = Logger(config["logger"])
    logger.log_hyperparams(config)
    dynamo_prometheus_monitor = maybe_start_dynamo_prometheus_monitor(config, logger)

    policy_generation = DynamoGeneration(
        cluster=None,
        config=generation_config,
    )

    try:
        nemo_gym_config = NemoGymConfig(
            model_name=generation_config["model_name"],
            base_urls=policy_generation.dp_openai_server_base_urls,
            initial_global_config_dict=config["env"]["nemo_gym"],
        )
        nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
        ray.get(nemo_gym.health_check.remote())

        policy_generation.prepare_for_generation()
        print("\nRunning Dynamo rollout-only trajectory collection...", flush=True)
        rollout_result = run_async_nemo_gym_rollout(
            policy_generation=policy_generation,
            input_batch=rollout_batch,
            tokenizer=tokenizer,
            task_to_env={"nemo_gym": nemo_gym},
            max_seq_len=None,
            generation_config=generation_config,
            max_rollout_turns=None,
            greedy=False,
        )
        _log_trajectory_collection(logger, rollout_result)
        print("Rollout metrics:")
        pprint.pprint(rollout_result.rollout_metrics)
    finally:
        policy_generation.finish_generation()
        if dynamo_prometheus_monitor is not None:
            dynamo_prometheus_monitor.stop()


if __name__ == "__main__":
    main()
