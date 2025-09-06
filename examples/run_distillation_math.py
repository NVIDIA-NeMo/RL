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
from collections import defaultdict
from typing import Any, Optional

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.distillation import MasterConfig, distillation_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.hf_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.hf_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
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
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run distillation training with configuration"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                             Math Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


# TaskDataProcessFnCallable
def hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,  # required parameter
    idx: int,
) -> DatumSpec:
    """Process Hugging Face format data into NeMo-RL format."""
    # safety check: ensure prompt exists
    if task_data_spec.prompt is None:
        if task_data_spec.prompt_file:
            if os.path.exists(task_data_spec.prompt_file):
                try:
                    with open(task_data_spec.prompt_file, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    raise ValueError(f"Failed to read file: {e}")
            else:
                raise ValueError(
                    f"Prompt file does not exist: {task_data_spec.prompt_file}"
                )

        raise ValueError(
            f"TaskDataSpec.prompt is None. This usually means the prompt file "
            f"'{task_data_spec.prompt_file}' could not be loaded or is empty. "
            f"Current working directory: {os.getcwd()}, "
            f"Absolute prompt file path: {os.path.abspath(task_data_spec.prompt_file) if task_data_spec.prompt_file else 'None'}"
        )

    messages = datum_dict["messages"]
    problem = messages[0]["content"]
    extra_env_info = {"ground_truth": messages[1]["content"]}

    message_log: LLMMessageLogType = []

    try:
        formatted_content = task_data_spec.prompt.format(problem)
    except Exception as e:
        raise ValueError(f"Failed to format prompt: {e}")

    user_message = {
        "role": "user",
        "content": formatted_content,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # calculate the number of tokens that can be retained for each message
        tokens_per_message = max_seq_length // len(message_log)
        for chat_message in message_log:
            # retain each message's token, but not exceed the limit
            chat_message["token_ids"] = chat_message["token_ids"][:tokens_per_message]
        # recalculate the length
        length = sum(len(m["token_ids"]) for m in message_log)
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
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
    print("\n▶ Setting up data...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # Load OpenMathInstruct2Dataset using nemo rl datasets
    if data_config["dataset_name"] == "OpenMathInstruct-2":
        print("Loading nvidia/OpenMathInstruct2Dataset for training and validation")
        data: Any = OpenMathInstruct2Dataset(seed=seed)
    elif data_config["dataset_name"] == "DeepScaler":
        print(
            "Loading agentica-org/DeepScaleR-Preview-Dataset for training and validation"
        )
        data: Any = DeepScalerDataset(seed=seed)
    else:
        raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (math_task_spec, hf_data_processor))
    )
    task_data_processors["math"] = (math_task_spec, hf_data_processor)

    math_env = MathEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["math"])
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            math_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "distillation_math.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    if config["policy"]["generation"] is not None:
        from nemo_rl.models.generation import configure_generation_config

        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], tokenizer
        )
    else:
        print("  ⚠️ No generation config found, this may cause issues")

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], 42)

    (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        # tokenizer,  # add tokenizer
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    distillation_train(
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        tokenizer,  # pass tokenizer parameter
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    )


if __name__ == "__main__":
    main()
