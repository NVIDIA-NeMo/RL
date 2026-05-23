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
from typing import Any, Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.games.wordle import (
    WORD_LIST,
    WordleEnv,
    WordleGameLogic,
    WordleMetadata,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for Wordle")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def generate_wordle_datum(
    tokenizer,
    word_list: list[str],
    max_guesses: int,
    task_name: str,
    idx: int,
    add_system_prompt: bool,
) -> DatumSpec:
    """Generate a single Wordle datum (prompt and metadata)."""
    game_state = WordleGameLogic.generate(word_list)
    target_word = game_state["target_word"]

    system_prompt = (
        "You are playing Wordle. You must guess a secret 5-letter word. "
        "After each guess, you'll receive feedback:\n"
        "  G = letter is correct and in the correct position (Green)\n"
        "  Y = letter is in the word but in the wrong position (Yellow)\n"
        "  X = letter is not in the word (Gray)\n\n"
        "Wrap every guess in <guess>WORD</guess> tags. "
        "Think carefully about the feedback before each guess."
    )

    user_prompt = (
        "I'm thinking of a 5-letter word. You have 6 guesses.\n"
        "Make your first guess by wrapping it in <guess>WORD</guess> tags.\n"
        "For example: <guess>crane</guess>"
    )

    messages = []
    if add_system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        user_prompt = system_prompt + "\n\n" + user_prompt
    messages.append({"role": "user", "content": user_prompt})

    initial_prompt_content = tokenizer.apply_chat_template(
        messages,
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

    metadata = WordleMetadata(
        target_word=target_word,
        guesses_made=0,
        max_guesses=max_guesses,
        feedback_history=[],
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
    """An IterableDataset that generates Wordle data indefinitely."""

    def __init__(
        self,
        tokenizer,
        word_list: list[str],
        max_guesses: int,
        task_name: str,
        add_system_prompt: bool,
        length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.word_list = word_list
        self.max_guesses = max_guesses
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_wordle_datum(
                tokenizer=self.tokenizer,
                word_list=self.word_list,
                max_guesses=self.max_guesses,
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
    """Set up the iterable data generator and env map for the Wordle task."""
    env_config = env_cfg[task_name]

    env = WordleEnv.options(num_gpus=0).remote(cfg=dict(env_config.get("cfg", {})))
    task_to_env = {task_name: env}

    word_list = env_config.get("cfg", {}).get("word_list", None)
    if word_list is None:
        word_list = WORD_LIST
    max_guesses = env_config.get("cfg", {}).get("max_guesses", 6)

    training_dataset = IterableWordleDataset(
        tokenizer=tokenizer,
        word_list=word_list,
        max_guesses=max_guesses,
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=length,
    )

    validation_dataset = IterableWordleDataset(
        tokenizer=tokenizer,
        word_list=word_list,
        max_guesses=max_guesses,
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=val_length,
    )

    val_task_to_env = task_to_env
    return training_dataset, validation_dataset, task_to_env, val_task_to_env


def main():
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_wordle.yaml"
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
    dataset, val_dataset, task_to_env, val_task_to_env = setup_wordle_data(
        tokenizer=tokenizer,
        env_cfg=config.env,
        task_name="wordle",
        length=ds_length,
        val_length=config.grpo["max_val_samples"],
        add_system_prompt=config.data["add_system_prompt"],
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
