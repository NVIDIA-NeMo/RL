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

"""GRPO training script for Wordle game.

Trains an LLM to play Wordle using GRPO with dense reward shaping.
The model learns to guess 5-letter words based on feedback patterns.

Usage:
    # Smoke test with small model
    uv run python examples/run_grpo_wordle.py \
        grpo.max_num_steps=5 \
        grpo.num_prompts_per_step=4 \
        grpo.num_generations_per_prompt=4 \
        policy.model_name="Qwen/Qwen2.5-0.5B-Instruct" \
        cluster.gpus_per_node=1

    # Full training with Nemotron-Nano-9B-v2
    uv run python examples/run_grpo_wordle.py \
        logger.wandb_enabled=True \
        logger.wandb.name="wordle-nemotron-9b"
"""

import argparse
import itertools
import os
import pprint
import random
from typing import Any, Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.games.wordle import (
    WordleConfig,
    WordleEnv,
    WordleGameLogic,
    WordleMetadata,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for Wordle")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def generate_wordle_datum(
    tokenizer: AutoTokenizer,
    game_config: WordleConfig,
    max_turns: int,
    task_name: str,
    idx: int,
    add_system_prompt: bool,
) -> DatumSpec:
    """Generate a single Wordle game datum (prompt and metadata)."""
    # Generate initial game state
    initial_game_state = WordleGameLogic.generate(game_config)
    target_word = initial_game_state["target_word"]
    welcome_message = WordleGameLogic.init(initial_game_state)

    # Create initial prompt - end with <guess> to prime the model
    prompt_content = f"{welcome_message}\nYour guess: <guess>"

    # Apply chat template
    initial_prompt_content = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()

    # Tokenize
    tokenized_prompt = tokenizer(
        initial_prompt_content, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    # Create message log
    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]

    # Create initial metadata
    metadata = WordleMetadata(
        target_word=target_word,
        guesses=[],
        feedback_history=[],
        known_greens={},
        known_yellows=set(),
        eliminated_letters=set(),
        turn=0,
        max_turns=max_turns,
    )

    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
        "stop_strings": ["</guess>"],
    }

    return datum


class IterableWordleDataset(IterableDataset):
    """An IterableDataset that generates Wordle games indefinitely."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        game_config: WordleConfig,
        max_turns: int,
        task_name: str,
        add_system_prompt: bool,
        length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.game_config = game_config
        self.max_turns = max_turns
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        print("Starting IterableWordleDataset (indefinite generation).")
        for i in itertools.count():
            yield generate_wordle_datum(
                tokenizer=self.tokenizer,
                game_config=self.game_config,
                max_turns=self.max_turns,
                task_name=self.task_name,
                idx=i,
                add_system_prompt=self.add_system_prompt,
            )

    def __len__(self):
        return self.length


def setup_wordle_data(
    tokenizer: AutoTokenizer,
    env_cfg: dict[str, Any],
    task_name: str,
    length: int,
    val_length: int,
    add_system_prompt: bool,
) -> tuple[IterableDataset, IterableDataset | None, dict, dict]:
    """Set up the iterable data generator and env map for Wordle."""
    print("Setting up Wordle iterable data and environment...")
    env_config = env_cfg[task_name]

    print(f"Instantiating environment for task '{task_name}'...")
    env = WordleEnv.options(num_gpus=0).remote(cfg=dict(env_config["cfg"]))
    task_to_env = {task_name: env}
    print(f"Environment '{task_name}' created.")

    print("Creating Wordle dataset...")
    training_dataset = IterableWordleDataset(
        tokenizer=tokenizer,
        game_config=dict(env_config["cfg"]["game_config"]),
        max_turns=env_config["cfg"]["max_turns"],
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=length,
    )
    print("Wordle training dataset created.")

    validation_dataset = IterableWordleDataset(
        tokenizer=tokenizer,
        game_config=dict(env_config["cfg"]["game_config"]),
        max_turns=env_config["cfg"]["max_turns"],
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=val_length,
    )
    print("Wordle validation dataset created.")

    val_task_to_env = task_to_env

    return training_dataset, validation_dataset, task_to_env, val_task_to_env


def main():
    """Main entry point."""
    args, overrides = parse_args()

    # Default config path
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_wordle.yaml"
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

    # Get the next experiment directory
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    set_seed(config["grpo"]["seed"])

    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Setup data & env map
    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_wordle_data(
        tokenizer=tokenizer,
        env_cfg=config["env"],
        task_name="wordle_game",
        length=ds_length,
        val_length=config["grpo"]["max_val_samples"],
        add_system_prompt=config["data"]["add_system_prompt"],
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
