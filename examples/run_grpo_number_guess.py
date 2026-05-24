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
from nemo_rl.environments.games.number_guess import NumberGuessEnv, NumberGuessMetadata
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

SYSTEM_PROMPT = (
    "You are playing a number guessing game. "
    "Each turn, output exactly one guess inside <guess></guess> tags, like <guess>10</guess>. "
    "You will receive feedback in <feedback></feedback> tags telling you if your guess is too high or too low. "
    "Then make another guess. Always use binary search: guess the middle of the remaining range."
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Number Guessing Game"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def generate_number_guess_datum(
    tokenizer: AutoTokenizer,
    min_number: int,
    max_number: int,
    max_guesses: int,
    idx: int,
) -> DatumSpec:
    """Generate a single number guessing game datum."""
    target = random.randint(min_number, max_number)

    prompt_content = (
        f"I'm thinking of a number between {min_number} and {max_number}. "
        f"You have {max_guesses} guesses. Make your first guess now."
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

    metadata = NumberGuessMetadata(
        target=target,
        guesses_remaining=max_guesses,
        max_guesses=max_guesses,
        last_guess=None,
    )

    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": "number_guess",
        "stop_strings": ["</guess>"],
    }
    return datum


class IterableNumberGuessDataset(IterableDataset):
    """An IterableDataset that generates number guessing game data indefinitely."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        min_number: int,
        max_number: int,
        max_guesses: int,
        length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.min_number = min_number
        self.max_number = max_number
        self.max_guesses = max_guesses
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_number_guess_datum(
                tokenizer=self.tokenizer,
                min_number=self.min_number,
                max_number=self.max_number,
                max_guesses=self.max_guesses,
                idx=i,
            )

    def __len__(self):
        return self.length


def setup_number_guess_data(
    tokenizer: AutoTokenizer,
    env_cfg: dict,
    task_name: str,
    length: int,
    val_length: int,
) -> tuple[IterableDataset, IterableDataset, dict, dict]:
    """Set up the iterable data generator and env map for the number guessing game."""
    env_config = env_cfg[task_name]
    cfg = dict(env_config.get("cfg", {}))

    env = NumberGuessEnv.options(num_gpus=0).remote(config=cfg)
    task_to_env = {task_name: env}

    min_number = cfg.get("min_number", 1)
    max_number = cfg.get("max_number", 50)
    max_guesses = cfg.get("max_guesses", 7)

    training_dataset = IterableNumberGuessDataset(
        tokenizer=tokenizer,
        min_number=min_number,
        max_number=max_number,
        max_guesses=max_guesses,
        length=length,
    )
    validation_dataset = IterableNumberGuessDataset(
        tokenizer=tokenizer,
        min_number=min_number,
        max_number=max_number,
        max_guesses=max_guesses,
        length=val_length,
    )

    return training_dataset, validation_dataset, task_to_env, task_to_env


def main():
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_number_guess.yaml"
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

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"Using checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    ds_length = (
        config.grpo["num_prompts_per_step"]
        * config.grpo["num_generations_per_prompt"]
        * config.grpo["max_num_steps"]
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_number_guess_data(
        tokenizer=tokenizer,
        env_cfg=config.env,
        task_name="number_guess",
        length=ds_length,
        val_length=config.grpo["max_val_samples"],
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
