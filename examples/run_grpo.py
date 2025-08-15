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

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import JsonlinesDataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.ifeval_environment import IFEvalEnvironment
from nemo_rl.environments.llm_judge_async_environment import LLMJudgeAsyncEnvironment
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, env_configs):
    print("\n▶ Setting up data...")
    val_ds = None

    train_ds = JsonlinesDataset(
        data_config["train"]["jsonl_path"],
        data_config["train"]["seed"],
        tokenizer,
        max_seq_length=data_config["max_input_seq_length"],
        filter_long_samples=data_config["train"]["filter_long_samples"],
    )

    if "val" in data_config:
        val_ds = JsonlinesDataset(
            data_config["val"]["jsonl_path"],
            data_config["val"]["seed"],
            tokenizer,
            max_seq_length=data_config["max_input_seq_length"],
            filter_long_samples=data_config["val"]["filter_long_samples"],
        )

    task_to_env = {}

    if "math" in env_configs and env_configs["math"]["enable"]:
        math_env = MathEnvironment.options(
            runtime_env={
                "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(
                    os.environ
                ),  # Pass thru all user environment variables
            }
        ).remote(env_configs["math"])
        task_to_env["math"] = math_env

    if "ifeval" in env_configs and env_configs["ifeval"]["enable"]:
        ifeval_env = IFEvalEnvironment.options(
            runtime_env={
                "py_executable": IFEvalEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(os.environ),
            },
        ).remote(env_configs["ifeval"])
        task_to_env["ifeval"] = ifeval_env
    if "llm_judge_async" in env_configs and env_configs["llm_judge_async"]["enable"]:
        # Extract max_concurrency from config, default to 16 if not specified
        max_concurrency = env_configs["llm_judge_async"].get("max_concurrency", 16)

        llm_judge_async_env = LLMJudgeAsyncEnvironment.options(
            max_concurrency=max_concurrency,
            runtime_env={
                "py_executable": LLMJudgeAsyncEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(os.environ),
            },
        ).remote(env_configs["llm_judge_async"])
        task_to_env["llm_judge"] = llm_judge_async_env

    return train_ds, val_ds, task_to_env, task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_1B.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    if (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        != config["policy"]["train_global_batch_size"]
        and config["policy"]["num_global_batch_repeats"] > 1
    ):
        raise ValueError(
            f"num prompts per step ({config['grpo']['num_prompts_per_step']}) * "
            f"num generations per prompt ({config['grpo']['num_generations_per_prompt']}) "
            f"must be equal to train global batch size ({config['policy']['train_global_batch_size']}) "
            f"if num global batch repeats ({config['policy']['num_global_batch_repeats']}) > 1 "
            "because I don't shuffle the data so the batch will be seen in the same order like B1 B1 B1 B2 B2 B2 instead of B1 B2 B1 B2"
        )

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

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"])

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
