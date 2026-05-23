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
from nemo_rl.environments.games.calculator_tool import (
    STOP_STRINGS,
    CalculatorMetadata,
    CalculatorToolEnv,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

SYSTEM_PROMPT = (
    "You have access to a calculator tool. To use it, write:\n"
    '<tool_call>{"name":"calculate","arguments":{"expression":"MATH_EXPR"}}</tool_call>\n'
    "where MATH_EXPR is a valid arithmetic expression (e.g., 3*4+5*3).\n"
    "After computing, give your final answer in <answer>NUMBER</answer> tags.\n"
    "You may make multiple calculator calls before giving your final answer."
)

# Item pools for generating word problems
ITEM_NAMES = [
    "apples",
    "oranges",
    "bananas",
    "pens",
    "notebooks",
    "cookies",
    "muffins",
    "pencils",
    "erasers",
    "stickers",
]


def _generate_problem() -> tuple[str, float]:
    """Generate a simple shopping math word problem and its answer."""
    num_items = random.randint(2, 3)
    chosen_items = random.sample(ITEM_NAMES, num_items)

    parts = []
    expr_parts = []
    for item in chosen_items:
        price = random.randint(1, 5)
        qty = random.randint(1, 5)
        parts.append(f"{qty} {item} at ${price} each")
        expr_parts.append(price * qty)

    total = sum(expr_parts)
    problem = (
        f"A store sells items. Someone buys {', and '.join(parts)}. "
        f"How much do they spend in total?"
    )
    return problem, float(total)


def generate_calculator_datum(
    tokenizer: AutoTokenizer,
    max_tool_calls: int,
    task_name: str,
    idx: int,
) -> DatumSpec:
    """Generate a single calculator tool-call datum."""
    problem, target_answer = _generate_problem()

    message_list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    initial_prompt_content = tokenizer.apply_chat_template(
        message_list,
        tokenize=False,
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

    metadata = CalculatorMetadata(
        target_answer=target_answer,
        num_tool_calls=0,
        max_tool_calls=max_tool_calls,
    )

    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
        "stop_strings": STOP_STRINGS,
    }
    return datum


class IterableCalculatorDataset(IterableDataset):
    """An IterableDataset that generates calculator tool-call data indefinitely."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_tool_calls: int,
        task_name: str,
        length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_tool_calls = max_tool_calls
        self.task_name = task_name
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_calculator_datum(
                tokenizer=self.tokenizer,
                max_tool_calls=self.max_tool_calls,
                task_name=self.task_name,
                idx=i,
            )

    def __len__(self):
        return self.length


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GRPO training for calculator tool-call task"
    )
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
            os.path.dirname(__file__), "configs", "grpo_calculator_tool.yaml"
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

    # Setup env
    task_name = "calculator_tool"
    env_config = config.env[task_name]
    max_tool_calls = env_config["cfg"]["max_tool_calls"]

    env = CalculatorToolEnv.options(num_gpus=0).remote(cfg=dict(env_config["cfg"]))
    task_to_env = {task_name: env}

    # Setup datasets
    ds_length = (
        config.grpo["num_prompts_per_step"]
        * config.grpo["num_generations_per_prompt"]
        * config.grpo["max_num_steps"]
    )
    dataset = IterableCalculatorDataset(
        tokenizer=tokenizer,
        max_tool_calls=max_tool_calls,
        task_name=task_name,
        length=ds_length,
    )
    val_dataset = IterableCalculatorDataset(
        tokenizer=tokenizer,
        max_tool_calls=max_tool_calls,
        task_name=task_name,
        length=config.grpo["max_val_samples"],
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
