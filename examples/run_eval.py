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
import sys
from typing import Any, cast

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from examples.run_grpo_math import math_data_processor
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.eval_datasets import (
    aime2024,
    gpqa,
    math,
    mmlu,
    mmlu_pro,
)
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config

TokenizerType = PreTrainedTokenizerBase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


def _construct_prompt(prompt: str, question: str, options: dict[str, str]) -> str:
    """Construct prompt from question and options."""
    output = prompt
    output += f"\n\nQuestion: {question}\nOptions:\n"
    output += "\n".join(
        [
            f"{letter}) {option}"
            for letter, option in options.items()
            if option is not None
        ]
    )
    return output


def multichoice_qa_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for multiple-choice problems."""
    question = datum_dict["question"]
    answer = str(datum_dict["answer"])
    options = datum_dict["options"]
    extra_env_info = {"ground_truth": answer}
    if "subject" in datum_dict:
        extra_env_info.update({"subject": datum_dict["subject"]})

    message_log = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(sys, return_tensors="pt")["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        problem = _construct_prompt(task_data_spec.prompt, question, options)
    user_message = {"role": "user", "content": problem}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def setup_data(tokenizer: AutoTokenizer, data_config, env_configs):
    print("Setting up data...")

    # load dataset
    dataset_name = data_config["dataset_name"]
    data_processor_fn = multichoice_qa_processor
    if dataset_name == "mmlu":
        base_dataset = mmlu.MMLUDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "aime2024":
        base_dataset = aime2024.AIME2024Dataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
        data_processor_fn = math_data_processor
    elif dataset_name == "gpqa":
        base_dataset = gpqa.GPQADataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mmlu_pro":
        base_dataset = mmlu_pro.MMLUProDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math":
        base_dataset = math.MathDataset(
            variant="math_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
        data_processor_fn = math_data_processor
    elif dataset_name == "math500":
        base_dataset = math.MathDataset(
            variant="math_500_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
        data_processor_fn = math_data_processor
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    rekeyed_ds = base_dataset.rekeyed_ds

    env = MathEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            )
        }
    ).remote(env_configs["math"])

    dataset = AllTaskProcessedDataset(
        dataset=rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=data_processor_fn,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, env, tokenizer


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "evals", "eval.yaml"
        )

    config = OmegaConf.load(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Init ray
    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )

    # Setup data
    (
        dataset,
        env,
        tokenizer,
    ) = setup_data(tokenizer, config["data"], config["env"])

    # Setup
    (
        vllm_generation,
        dataloader,
        master_config,
    ) = setup(config, tokenizer, dataset)

    # Run evaluation
    run_env_eval(
        vllm_generation,
        dataloader,
        env,
        master_config,
    )


if __name__ == "__main__":
    main()
