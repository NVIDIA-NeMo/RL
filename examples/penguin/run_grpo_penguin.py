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
import json
from typing import Optional
from itertools import chain, repeat
from time import sleep

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup, _should_use_penguin
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset, DatumSpec
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.penguin import (
    Penguin,
    PenguinConfig,
    setup_qwen3_penguin_config,
    setup_penguin_config,
    penguin_example_to_nemo_rl_datum_spec,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_single_penguin_dataset(jsonl_fpath: str, tokenizer, num_repeats: Optional[int] = None):
    with open(jsonl_fpath) as f:
        penguin_examples = list(map(json.loads, f))

    print(f"Loaded data at {jsonl_fpath}. Found {len(penguin_examples)} examples")

    if num_repeats:
        previous_length = len(penguin_examples)
        penguin_examples = list(chain.from_iterable(repeat(penguin_example, num_repeats) for penguin_example in penguin_examples))
        print(f"Repeating examples (in a pattern of abc to aabbcc) for {jsonl_fpath} from {previous_length} to {len(penguin_examples)}!")

    nemo_rl_compatible_examples: list[DatumSpec] = [
        penguin_example_to_nemo_rl_datum_spec(penguin_example, idx)
        for idx, penguin_example in enumerate(penguin_examples)
    ]

    passthrough_task_processor = lambda datum_dict, *args, **kwargs: datum_dict
    return AllTaskProcessedDataset(
        nemo_rl_compatible_examples,
        tokenizer,
        None,
        passthrough_task_processor,
    )


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "grpo_math_1B_vllm_http_server.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Penguin specific config setup.
    setup_penguin_config(config, tokenizer)
    setup_qwen3_penguin_config(config, tokenizer)

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # We assert here since this is right after the final config has been materialized.
    assert _should_use_penguin(config)

    print("\n▶ Setting up data...")
    train_dataset = setup_single_penguin_dataset(
        jsonl_fpath=config["data"]["train_jsonl_fpath"],
        tokenizer=tokenizer,
    )
    val_dataset = setup_single_penguin_dataset(
        jsonl_fpath=config["data"]["validation_jsonl_fpath"],
        tokenizer=tokenizer,
        num_repeats=config["grpo"]["num_generations_per_prompt"],
    )

    init_ray()

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
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    penguin_config = PenguinConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["penguin"],
    )
    penguin = Penguin.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.penguin.Penguin"
            ),
        }
    ).remote(penguin_config)
    task_to_env = {"penguin": penguin}
    val_task_to_env = task_to_env

    # TODO: Eventually there will be a better way to check if the servers have spun up on Penguin side.
    sleep_time = int(master_config["env"].get("init_sleep_time", 30))  # 30s default.
    print(f"Sleeping {sleep_time}s to let Penguin environments spin up.")
    sleep(sleep_time)

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
