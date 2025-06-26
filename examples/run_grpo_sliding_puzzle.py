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
from typing import Any, Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.games.sliding_puzzle import (
    SlidingPuzzleConfig,
    SlidingPuzzleEnv,
    SlidingPuzzleGameLogic,
    SlidingPuzzleMetadata,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def generate_puzzle_datum(
    tokenizer,
    game_config: SlidingPuzzleConfig,
    max_moves: int,
    task_name: str,
    idx: int,
    add_system_prompt: bool,
) -> DatumSpec:
    """Generates a single sliding puzzle datum (prompt and metadata)."""

    def generate_random_config(max_config: dict[str, Any]) -> dict[str, Any]:
        """Generate a random config for the sliding puzzle game."""
        shuffle_moves = random.randint(1, max_config.get("shuffle_moves"))
        if shuffle_moves % 2 == 0:
            shuffle_moves += 1
        return {
            "size": random.randint(2, max_config.get("size", 3)),
            "shuffle_moves": shuffle_moves,
        }

    game_config = generate_random_config(game_config)
    initial_game_state = SlidingPuzzleGameLogic.generate(game_config)
    initial_render = SlidingPuzzleGameLogic.render(initial_game_state)
    welcome_message = SlidingPuzzleGameLogic.init(initial_game_state)
    puzzle_size = game_config.get("size", 3)
    prompt_instructions = (
        f"{welcome_message}\n\n"
        f"Current Board State:\n{initial_render}\n\n"
        f"Reach the goal state where numbers are ordered 1 through {puzzle_size**2 - 1} "
        f"with the empty space (0) at the bottom right.\n"
        f"Valid actions: 'up', 'down', 'left', 'right', or 'slide row col' (e.g., 'slide 1 2').\n"
        f"After thinking, output your chosen action on a new line starting with '<action></action>' like this:\n<action>your_action</action>"
        f"\nIf you just want to see the board, output <action>view</action>"
        f"\nThink carefully step-by-step before acting.\n"
    )
    initial_prompt_content = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_instructions}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
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
    metadata = SlidingPuzzleMetadata(
        game_state=initial_game_state, num_moves=0, max_moves=max_moves
    )
    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
        "stop_strings": ["</action>"],
    }
    return datum


class IterablePuzzleDataset(IterableDataset):
    """An IterableDataset that generates sliding puzzle data indefinitely."""

    def __init__(
        self, tokenizer, game_config, max_moves, task_name, add_system_prompt, length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.game_config = game_config
        self.max_moves = max_moves
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        print("Starting IterablePuzzleDataset (indefinite generation).")
        # Use itertools.count for an infinite index generator
        for i in itertools.count():
            yield generate_puzzle_datum(
                tokenizer=self.tokenizer,
                game_config=self.game_config,
                max_moves=self.max_moves,
                task_name=self.task_name,
                idx=i,
                add_system_prompt=self.add_system_prompt,
            )

    def __len__(self):
        return self.length


def setup_puzzle_data(
    tokenizer: AutoTokenizer,
    env_cfg: dict[str, Any],
    task_name: str,
    length: int,
    val_length: int,
    add_system_prompt: bool,
) -> tuple[IterableDataset, IterableDataset | None, dict, dict]:
    """Sets up the iterable data generator and env map for the sliding puzzle task."""
    print("Setting up Sliding Puzzle iterable data and environment...")
    env_config = env_cfg[task_name]

    print(f"Instantiating environment for task '{task_name}'...")
    env = SlidingPuzzleEnv.options(num_gpus=0).remote(cfg=dict(env_config["cfg"]))
    task_to_env = {task_name: env}
    print(f"Environment '{task_name}' created.")

    print("Creating Sliding Puzzle dataset...")
    training_dataset = IterablePuzzleDataset(
        tokenizer=tokenizer,
        game_config=dict(env_config["cfg"]["game_config"]),
        max_moves=env_config["cfg"]["max_moves"],
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=length,
    )
    print("Sliding Puzzle dataset created.")

    validation_dataset = IterablePuzzleDataset(
        tokenizer=tokenizer,
        game_config=dict(env_config["cfg"]["game_config"]),
        max_moves=env_config["cfg"]["max_moves"],
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=val_length,
    )
    val_task_to_env = task_to_env

    return training_dataset, validation_dataset, task_to_env, val_task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_sliding_puzzle.yaml"
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

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data & env map
    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_puzzle_data(
        tokenizer=tokenizer,
        env_cfg=config["env"],
        task_name="sliding_puzzle_game",
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
