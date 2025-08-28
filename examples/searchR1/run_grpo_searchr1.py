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
from typing import Any

from datasets import load_dataset
from environment.search import SearchEnv, SearchEnvConfig
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    TaskDataSpec,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

TokenizerType = PreTrainedTokenizerBase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def parquet_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    message_logs = list(datum_dict["prompt"])
    message_logs = message_logs[1:]  # neglect the system prompt
    for idx, message in enumerate(message_logs):
        last_message = idx == len(message_logs) - 1
        content = tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=last_message,
            add_special_tokens=False,
        )
        token_ids = tokenizer(content, return_tensors="pt")["input_ids"][0]
        message["content"] = content
        message["token_ids"] = token_ids

    length = sum(len(message["token_ids"]) for message in message_logs)
    loss_multiplier = 1.0
    if max_seq_length and length > max_seq_length:
        for chat_message in message_logs:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_logs))
            ]
        loss_multiplier = 0.0

    return DatumSpec(
        message_log=message_logs,
        length=length,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name=task_data_spec.task_name,
        stop_strings=["</search>", "</answer>", "</think>"],  # ?
        extra_env_info={
            "ground_truth": datum_dict["reward_spec"]["ground_truth"],
            "num_turns": 0,
        },
    )


def setup_data(
    env_cfg: dict[str, Any],
    tokenizer: AutoTokenizer,
    train_data_path: str,
    valid_data_path: str,
    task_name: str = "searchR1",
) -> tuple[
    AllTaskProcessedDataset,
    AllTaskProcessedDataset,
    dict[str, SearchEnv],
    dict[str, SearchEnv],
]:
    """Setup the dataloader, which reads from parquet files."""
    task_spec = TaskDataSpec(task_name=task_name)

    train_dataset = load_dataset("parquet", data_files=train_data_path)["train"]
    valid_dataset = load_dataset("parquet", data_files=valid_data_path)["train"]

    env = SearchEnv.options(num_gpus=0).remote(
        cfg=SearchEnvConfig(**dict(env_cfg["search"]))
    )
    task_to_env = {task_name: env}

    dataset = AllTaskProcessedDataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        default_task_data_spec=task_spec,
        task_data_processors=parquet_data_processor,
    )

    valid_dataset = AllTaskProcessedDataset(
        dataset=valid_dataset,
        tokenizer=tokenizer,
        default_task_data_spec=task_spec,
        task_data_processors=parquet_data_processor,
    )
    return dataset, valid_dataset, task_to_env, task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "../", "configs", "grpo_searchr1.yaml"
        )

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

    # return
    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        config["env"],
        tokenizer,
        config["data"]["train_data_path"],
        config["data"]["valid_data_path"],
    )

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
