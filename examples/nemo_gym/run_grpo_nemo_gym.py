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

import argparse
import os
import pprint
from typing import Dict, Optional

# Increase the W&B single object size warning threshold. Initially 100_000 (100 KB) -> 10_000_000 (10 MB)
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

import ray
from datasets import concatenate_datasets
from omegaconf import OmegaConf
from wandb import Table

from nemo_rl.algorithms.grpo import (
    ColocatablePolicyInterface,
    EnvironmentInterface,
    GenerationInterface,
    Logger,
    MasterConfig,
    StatefulDataLoader,
    TokenizerType,
    _should_use_nemo_gym,
    grpo_train,
    refit_policy_generation,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    extract_necessary_env_names,
    load_response_dataset,
    update_single_dataset_config,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    NemoGymConfig,
    setup_nemo_gym_config,
)
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_data(
    tokenizer: TokenizerType,
    data_config: Dict,
    env_configs: Dict,
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up envs...")
    env_name_list = extract_necessary_env_names(data_config)
    envs = {
        env_name: create_env(env_name=env_name, env_config=env_configs[env_name])
        for env_name in env_name_list
        if env_name != "nemo_gym"
    }
    print("\n▶ Setting up data...")
    # setup train dataset
    task_data_processors = {}
    task_to_env = {}
    data_list = []

    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]
    for cfg in data_config["train"]:
        update_single_dataset_config(cfg, data_config["default"])
        data = load_response_dataset(cfg, seed)
        data_list.append(data)
        # bind task_name to task_data_processors and task_to_env
        task_name = data.task_name
        task_data_processors[task_name] = (data.task_spec, data.processor)
        # Skip binding nemo_gym env to task_to_env, nemo_gym env need to initialize policy first
        if cfg["env_name"] != "nemo_gym":
            task_to_env[task_name] = envs[cfg["env_name"]]

    merged_data = concatenate_datasets([data.dataset for data in data_list])
    dataset = AllTaskProcessedDataset(
        merged_data,
        tokenizer,
        None,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(dataset)} samples.")

    # setup validation dataset
    val_task_data_processors = {}
    val_task_to_env = {}
    val_data_list = []

    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            val_data_list.append(data.val_dataset)
            # bind task_name to task_data_processors
            task_name = data.task_name
            val_task_data_processors[task_name] = task_data_processors[task_name]
            if task_name in task_to_env:
                val_task_to_env[task_name] = task_to_env[task_name]

    if data_config["validation"] is not None:
        if isinstance(data_config["validation"], dict):
            data_config["validation"] = [data_config["validation"]]

        for cfg in data_config["validation"]:
            update_single_dataset_config(cfg, data_config["default"])
            val_data = load_response_dataset(cfg, seed)
            val_data_list.append(val_data.dataset)
            # bind task_name to task_data_processors
            task_name = val_data.task_name
            val_task_data_processors[task_name] = (
                val_data.task_spec,
                val_data.processor,
            )
            if cfg["env_name"] != "nemo_gym":
                val_task_to_env[task_name] = envs[cfg["env_name"]]

    val_dataset = None
    if len(val_data_list) > 0:
        merged_val_data = concatenate_datasets(val_data_list)
        val_dataset = AllTaskProcessedDataset(
            merged_val_data,
            tokenizer,
            None,
            val_task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        print(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples.")

    return dataset, val_dataset, task_to_env, val_task_to_env


# These types are directly imported from grpo_train since if something about the architecture changes we want to immediately fail.
def collect_trajectories(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer: TokenizerType,
    val_task_to_env: dict[str, EnvironmentInterface],
    logger: Logger,
    master_config: MasterConfig,
) -> None:
    """Run trajectory collection."""
    # common config/state items
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    refit_policy_generation(policy, policy_generation, colocated_inference)

    log_filename = "trajectory_collection.jsonl"

    print("\n🔍 Running trajectory collection...", flush=True)
    generation_config = master_config["policy"]["generation"]
    for val_batch in val_dataloader:
        nemo_gym_rollout_result = run_async_nemo_gym_rollout(
            policy_generation=policy_generation,
            input_batch=val_batch,
            tokenizer=tokenizer,
            task_to_env=val_task_to_env,
            max_seq_len=None,
            generation_config=generation_config,
            max_rollout_turns=None,
            greedy=False,
        )

        rows_to_log: list[str] = []
        for key, value in nemo_gym_rollout_result.rollout_metrics.items():
            if "full_result" not in key:
                continue

            value: Table
            data: list[list[str]] = value.data  # (n, 1)
            rows_to_log.extend(v[0] for v in data)

        logger.log_string_list_as_jsonl(rows_to_log, log_filename)

        # TODO: eventually as trajectory collection use cases exceed 4 hours, we can leverage the dataloader save functionality to resume
        # And also leverage the TimeoutChecker functionality as well

    policy_generation.finish_generation()


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # NeMo-Gym specific config setup.
    setup_nemo_gym_config(config, tokenizer)

    # We assert here since this is right after the final config has been materialized.
    assert _should_use_nemo_gym(config)

    print("\n▶ Setting up data...")
    train_dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        tokenizer=tokenizer,
        data_config=config["data"],
        env_configs=config["env"],
        seed=config["grpo"]["seed"],
    )

    # Validation dataset config setup.
    if config["grpo"]["max_val_samples"] is not None:
        raise ValueError(
            """A non-null `grpo.max_val_samples` parameter is not supported.

Gym principle is that there is no hidden data pre or post processing from you. What you see is what you get.

The validation set you pass in will directly be used for validation with no additional preprocessing. If you want to have some number of repetitions, please include that in your dataset, via ``num_repeats``, in your dataset config and `ng_prepare_data` will prepare it accordingly."""
        )

    print(
        f"Setting `grpo.max_val_samples` and `grpo.val_batch_size` to the length of the validation dataset, which is {len(val_dataset)}"
    )
    config["grpo"]["max_val_samples"] = len(val_dataset)
    config["grpo"]["val_batch_size"] = config["grpo"]["max_val_samples"]

    # Print config
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    is_trajectory_collection = (
        config["env"]["nemo_gym"].pop("is_trajectory_collection", False) or False
    )
    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    # Default nemo_gym env is used for trajectory collection
    nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
    # Blocking wait for NeMo-Gym to spin up
    ray.get(nemo_gym.health_check.remote())
    task_to_env["nemo_gym"] = nemo_gym
    val_task_to_env["nemo_gym"] = nemo_gym

    if is_trajectory_collection:
        collect_trajectories(
            policy=policy,
            policy_generation=policy_generation,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            val_task_to_env=val_task_to_env,
            logger=logger,
            master_config=master_config,
        )
    else:
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
