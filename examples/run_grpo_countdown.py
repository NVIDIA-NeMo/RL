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

"""GRPO training script for the Countdown Game.

Given a list of numbers and a target, the model must combine the numbers
using +, -, *, / to reach the target. Each number may be used at most once.

Puzzles are procedurally generated and guaranteed to be solvable.
"""

import argparse
import itertools
import operator
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
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

# Number pool following the classic Countdown format
SMALL_NUMBERS = list(range(1, 11))
LARGE_NUMBERS = [25, 50, 75, 100]

# Operators for building solvable expressions
_OPS = [
    (operator.add, "+"),
    (operator.sub, "-"),
    (operator.mul, "*"),
]


def _generate_solvable_puzzle(
    num_count: int,
) -> tuple[list[int], int, str]:
    """Generate a solvable Countdown puzzle by building an expression first.

    Picks random numbers and combines a random subset with random operators
    to produce a target. This guarantees solvability.

    Returns (numbers, target, solution_expr).
    """
    while True:
        # Pick numbers: mix of small and large
        num_large = random.randint(0, min(2, num_count))
        num_small = num_count - num_large
        numbers = random.choices(SMALL_NUMBERS, k=num_small) + random.choices(
            LARGE_NUMBERS, k=num_large
        )
        random.shuffle(numbers)

        # Use 2-4 of the available numbers to build an expression
        use_count = random.randint(2, min(4, num_count))
        used_indices = random.sample(range(num_count), use_count)
        used_nums = [numbers[i] for i in used_indices]

        # Build a random expression tree
        result = used_nums[0]
        expr_parts = [str(used_nums[0])]
        for i in range(1, len(used_nums)):
            op_func, op_sym = random.choice(_OPS)
            new_result = op_func(result, used_nums[i])
            # Wrap in parens to maintain correctness
            expr_parts = [f"({' '.join(expr_parts)} {op_sym} {used_nums[i]})"]
            result = new_result

        target = result
        # Only accept integer targets in a reasonable range
        if isinstance(target, int) and 1 <= target <= 999:
            return numbers, target, expr_parts[0]


def generate_countdown_datum(
    tokenizer: AutoTokenizer,
    num_count_min: int,
    num_count_max: int,
    task_name: str,
    idx: int,
    add_system_prompt: bool,
) -> DatumSpec:
    """Generate a single Countdown game datum."""
    num_count = random.randint(num_count_min, num_count_max)
    numbers, target, _ = _generate_solvable_puzzle(num_count)

    prompt = (
        f"Using the numbers {numbers}, create an expression that equals {target}. "
        f"You may use each number at most once. "
        f"You may use +, -, *, / and parentheses.\n"
        f"Put your final expression in <answer>EXPR</answer> tags. "
        f"For example: <answer>(1 + 2) * 3</answer>"
    )

    initial_prompt_content = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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

    metadata = {"numbers": numbers, "target": target}
    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
    }
    return datum


class IterableCountdownDataset(IterableDataset):
    """An IterableDataset that generates Countdown game data indefinitely."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_count_min: int,
        num_count_max: int,
        task_name: str,
        add_system_prompt: bool,
        length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_count_min = num_count_min
        self.num_count_max = num_count_max
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_countdown_datum(
                tokenizer=self.tokenizer,
                num_count_min=self.num_count_min,
                num_count_max=self.num_count_max,
                task_name=self.task_name,
                idx=i,
                add_system_prompt=self.add_system_prompt,
            )

    def __len__(self):
        return self.length


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for Countdown Game")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_countdown.yaml"
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

    # Setup environment — load the CountdownEnv class from this project tree
    # (needed when the editable install points to a different copy of the repo)
    task_name = "countdown"
    env_config = config["env"][task_name]
    import importlib.util

    _countdown_path = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "nemo_rl",
        "environments",
        "games",
        "countdown.py",
    )
    _spec = importlib.util.spec_from_file_location(
        "nemo_rl.environments.games.countdown", os.path.abspath(_countdown_path)
    )
    _countdown_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_countdown_mod)
    CountdownEnv = _countdown_mod.CountdownEnv

    env = CountdownEnv.options(num_gpus=0).remote(cfg=dict(env_config.get("cfg", {})))
    task_to_env = {task_name: env}

    # Setup datasets
    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    countdown_cfg = env_config.get("cfg", {})
    dataset = IterableCountdownDataset(
        tokenizer=tokenizer,
        num_count_min=countdown_cfg.get("num_count_min", 3),
        num_count_max=countdown_cfg.get("num_count_max", 6),
        task_name=task_name,
        add_system_prompt=config["data"]["add_system_prompt"],
        length=ds_length,
    )
    val_dataset = IterableCountdownDataset(
        tokenizer=tokenizer,
        num_count_min=countdown_cfg.get("num_count_min", 3),
        num_count_max=countdown_cfg.get("num_count_max", 6),
        task_name=task_name,
        add_system_prompt=config["data"]["add_system_prompt"],
        length=config["grpo"]["max_val_samples"],
    )
    val_task_to_env = task_to_env

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
