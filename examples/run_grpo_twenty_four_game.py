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
import itertools
import os
import pprint
import random
from typing import Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.games.twenty_four_game import (
    TwentyFourGameEnv,
    has_solution,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

SYSTEM_PROMPT = (
    "You are a math puzzle solver. You will be given 4 numbers and must find a way "
    "to combine them using +, -, *, / to make exactly 24. You must use each number "
    "exactly once. Put your final expression inside <answer></answer> tags."
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for 24 Game")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def generate_solvable_puzzle() -> tuple[list[int], str]:
    """Generate a random solvable 24-game puzzle.

    Returns (numbers, solution_expression).
    """
    while True:
        numbers = [random.randint(1, 13) for _ in range(4)]
        solution = has_solution(numbers)
        if solution is not None:
            return numbers, solution


def generate_twenty_four_datum(
    tokenizer: AutoTokenizer,
    idx: int,
) -> DatumSpec:
    """Generate a single 24-game datum with prompt and metadata."""
    numbers, solution = generate_solvable_puzzle()

    prompt_content = (
        f"You are given the numbers: {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]}.\n"
        f"Find a mathematical expression using all four numbers exactly once and "
        f"the operators +, -, *, / (and parentheses) that equals 24.\n"
        f"Think step by step, then put your final expression inside <answer></answer> tags.\n"
        f"For example: <answer>(1+2+3)*4</answer>"
    )

    message_list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content},
    ]

    initial_prompt_content = tokenizer.apply_chat_template(
        message_list,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    tokenized_prompt = tokenizer(
        initial_prompt_content, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]

    metadata = {
        "numbers": numbers,
        "solution": solution,
    }

    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": "twenty_four_game",
    }
    return datum


class IterableTwentyFourDataset(IterableDataset):
    """An IterableDataset that generates solvable 24-game puzzles indefinitely."""

    def __init__(self, tokenizer: AutoTokenizer, length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_twenty_four_datum(
                tokenizer=self.tokenizer,
                idx=i,
            )

    def __len__(self):
        return self.length


def setup_twenty_four_data(
    tokenizer: AutoTokenizer,
    env_cfg: dict,
    task_name: str,
    length: int,
    val_length: int,
) -> tuple[IterableDataset, IterableDataset, dict, dict]:
    """Set up the iterable data generator and env map for the 24 game task."""
    env_config = env_cfg[task_name]
    env = TwentyFourGameEnv.options(num_gpus=0).remote(
        config=dict(env_config.get("cfg", {}))
    )
    task_to_env = {task_name: env}

    training_dataset = IterableTwentyFourDataset(
        tokenizer=tokenizer,
        length=length,
    )
    validation_dataset = IterableTwentyFourDataset(
        tokenizer=tokenizer,
        length=val_length,
    )

    return training_dataset, validation_dataset, task_to_env, task_to_env


def main():
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_twenty_four_game.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()
    set_seed(config["grpo"]["seed"])

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_twenty_four_data(
        tokenizer=tokenizer,
        env_cfg=config["env"],
        task_name="twenty_four_game",
        length=ds_length,
        val_length=config["grpo"]["max_val_samples"],
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
