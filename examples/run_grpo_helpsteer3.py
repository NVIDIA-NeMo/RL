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
from typing import Any, Optional

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_preference_dataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.helpsteer3_environment import HelpSteer3Environment
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GRPO training with HelpSteer3 configuration"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                             HelpSteer3 Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


def helpsteer3_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a HelpSteer3 preference datum into a DatumSpec for GRPO training.

    This function converts HelpSteer3 preference data to work with GRPO by:
    1. Using the context as the prompt
    2. Using the preferred completion as the target response
    3. Creating a reward signal based on preference scores
    """
    # Extract context and completions from HelpSteer3 format
    context = datum_dict["context"]
    completions = datum_dict["completions"]

    # Sort completions by rank (0 is preferred, 1 is rejected)
    completions = sorted(completions, key=lambda x: x["rank"])
    preferred_completion = completions[0]["completion"]

    # Build the conversation from context
    message_log: LLMMessageLogType = []

    # Add context messages
    if isinstance(context, list):
        for msg in context:
            message_log.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )
    else:
        # If context is a string, treat it as a user message
        message_log.append(
            {
                "role": "user",
                "content": context,
            }
        )

    # Add the preferred completion as the target
    for completion_msg in preferred_completion:
        message_log.append(
            {
                "role": completion_msg["role"],
                "content": completion_msg["content"],
            }
        )

    # Apply chat template and tokenize
    formatted_conversation = tokenizer.apply_chat_template(
        message_log,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=True,
    )

    # Tokenize the entire conversation
    full_tokens = tokenizer(
        formatted_conversation,
        return_tensors="pt",
        add_special_tokens=False,  # Already added by chat template
    )["input_ids"][0]

    # For simplicity, assign all tokens to the first message
    # In a more sophisticated implementation, you might want to split tokens properly
    message_log[0]["token_ids"] = full_tokens
    message_log[0]["content"] = formatted_conversation

    # Clear token_ids for other messages to avoid double counting
    for i in range(1, len(message_log)):
        message_log[i]["token_ids"] = tokenizer("", return_tensors="pt")["input_ids"][
            0
        ]  # Empty tensor

    length = sum(len(m["token_ids"]) for m in message_log)

    # Create ground truth from the preferred completion for environment evaluation
    ground_truth = " ".join([msg["content"] for msg in preferred_completion])
    extra_env_info = {"ground_truth": ground_truth}

    loss_multiplier = 1.0
    if length > max_seq_length:
        # Truncate if too long
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(
                    max_seq_length // len(message_log), len(chat_message["token_ids"])
                )
            ]
        loss_multiplier = 0.1  # Reduce loss for truncated sequences

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_data_spec.task_name,
    }
    return output


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
    print("\n▶ Setting up HelpSteer3 data and environment...")
    helpsteer3_task_spec = TaskDataSpec(
        task_name="helpsteer3",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )

    # Load HelpSteer3 preference dataset
    data: Any = load_preference_dataset(data_config)

    # Data processor for HelpSteer3
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (helpsteer3_task_spec, helpsteer3_data_processor))
    )
    task_data_processors["helpsteer3"] = (
        helpsteer3_task_spec,
        helpsteer3_data_processor,
    )

    # Setup dedicated HelpSteer3Environment
    helpsteer3_env = HelpSteer3Environment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.helpsteer3_environment.HelpSteer3Environment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs.get("helpsteer3", {"num_workers": 8}))

    # Create training dataset
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        helpsteer3_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    # Create validation dataset if available
    val_dataset: Optional[AllTaskProcessedDataset] = None
    if "validation" in data.formatted_ds and data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            helpsteer3_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )

    # Map tasks to environments
    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: helpsteer3_env)
    task_to_env["helpsteer3"] = helpsteer3_env

    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "recipes",
            "llm",
            "grpo-helpsteer3-llama-3.2-1b-1n8g-fsdp2tp1.yaml",
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

    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Setup data
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
