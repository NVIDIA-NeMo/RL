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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset, RandomDataset
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.dummy_environment import DummyEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides

TokenizerType = PreTrainedTokenizerBase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_data(tokenizer: AutoTokenizer, data_config, env_configs):
    print("Setting up data...")

    # load dataset
    base_dataset = RandomDataset(data_config["input_len_or_input_len_generator"])

    env = DummyEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            )
        }
    ).remote()

    dataset = AllTaskProcessedDataset(
        dataset=base_dataset.formatted_ds["train"],
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=base_dataset.processor,
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

    # Init ray
    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )
    config["generation"]["vllm_cfg"]["load_format"] = (
        "dummy"  # for random dataset eval, we use dummy weight initialization
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
