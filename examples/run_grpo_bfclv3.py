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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.interfaces import (
    TaskDataProcessFnCallable,
    TaskDataSpec,
    DatumSpec,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.data.processors import bfcl_multiturn_hf_data_processor
from nemo_rl.environments.multi_turn_tool_environment import MultiTurnToolEnvironment
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

# ===============================================================================
#                            BFCL Data Processing
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


def parse_args():
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
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up data...")
    
    # Define task specification for your HuggingFace dataset
    task_spec = TaskDataSpec(
        task_name="bfcl_multiturn",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # Load dataset from HuggingFace
    data: Any = load_response_dataset(data_config, seed)

    # Set up data processors 
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (task_spec, bfcl_multiturn_hf_data_processor))
    )
    task_name = 'bfcl_multiturn'
    task_data_processors[task_name] = (task_spec, bfcl_multiturn_hf_data_processor)

    # Create training dataset
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    # Create validation dataset if available
    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds.get("validation"):
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    # Set up environment - customize based on your needs
    task_to_env: dict[str, EnvironmentInterface] = {}
    
    # If you have specific environment configs, you can add them here
    if 'bfcl_multiturn' in env_configs and env_configs['bfcl_multiturn'].get('enable', False):
        multi_turn_tool_env = MultiTurnToolEnvironment.options(
            runtime_env={
                "py_executable": get_actor_python_env(
                    "nemo_rl.environments.multi_turn_tool_environment.MultiTurnToolEnvironment"
                ),
                "env_vars": dict(os.environ),
            },
        ).remote(env_configs['bfcl_multiturn'])
        task_to_env[task_name] = multi_turn_tool_env
        task_to_env['bfcl_multiturn'] = multi_turn_tool_env
    
    # Create a default environment mapping
    if not task_to_env:
        # You can set up a default environment here if needed
        task_to_env = defaultdict(lambda: None)
        task_to_env[task_name] = None

    return dataset, val_dataset, task_to_env, task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_bfclv3.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], config["grpo"]["seed"])

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
    ) = setup(config, tokenizer, dataset, val_dataset)

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