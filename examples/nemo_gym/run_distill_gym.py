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
r"""On-policy distillation training with NeMo Gym.

This script runs distillation training using NeMo Gym environments for
multi-turn rollouts. NeMo Gym handles the orchestration of agent-environment
interactions, tool calling, and verification.

Example usage:
    python examples/nemo_gym/run_distill_gym.py --config examples/nemo_gym/distillation_math_gym.yaml

Or with overrides:
    python examples/nemo_gym/run_distill_gym.py \
        --config examples/nemo_gym/distillation_math_gym.yaml \
        distillation.max_num_steps=100
"""

import argparse
import json
import os
import pprint
from itertools import chain, repeat
from typing import Optional

# Increase the W&B single object size warning threshold. Initially 100_000 (100 KB) -> 10_000_000 (10 MB)
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.distill_gym import (
    MasterConfig,
    distillation_gym_train,
    setup,
)
from nemo_rl.algorithms.grpo import _should_use_nemo_gym
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    nemo_gym_example_to_nemo_rl_datum_spec,
    setup_nemo_gym_config,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run distillation training with NeMo Gym"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_single_nemo_gym_dataset(
    jsonl_fpath: str, tokenizer, num_repeats: Optional[int] = None
):
    """Load a NeMo Gym dataset from a JSONL file.

    Args:
        jsonl_fpath: Path to the JSONL file containing NeMo Gym examples
        tokenizer: Tokenizer for the model
        num_repeats: Optional number of times to repeat each example

    Returns:
        AllTaskProcessedDataset ready for training/validation
    """
    with open(jsonl_fpath) as f:
        nemo_gym_examples = list(map(json.loads, f))

    print(f"Loaded data at {jsonl_fpath}. Found {len(nemo_gym_examples)} examples")

    if num_repeats:
        previous_length = len(nemo_gym_examples)
        nemo_gym_examples = list(
            chain.from_iterable(
                repeat(nemo_gym_example, num_repeats)
                for nemo_gym_example in nemo_gym_examples
            )
        )
        print(
            f"Repeating examples (in a pattern of abc to aabbcc) for {jsonl_fpath} from {previous_length} to {len(nemo_gym_examples)}!"
        )

    nemo_rl_compatible_examples: list[DatumSpec] = [
        nemo_gym_example_to_nemo_rl_datum_spec(nemo_gym_example, idx)
        for idx, nemo_gym_example in enumerate(nemo_gym_examples)
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
        # Default config - you should create this file
        args.config = os.path.join(
            os.path.dirname(__file__),
            "distillation_math_gym.yaml",
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
        "A generation config is required for distillation"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # NeMo-Gym specific config setup.
    setup_nemo_gym_config(config, tokenizer)

    # We assert here since this is right after the final config has been materialized.
    assert _should_use_nemo_gym(config), (
        "NeMo Gym must be enabled for this script. "
        "Set env.should_use_nemo_gym: true in your config."
    )

    print("\n▶ Setting up data...")
    train_dataset = setup_single_nemo_gym_dataset(
        jsonl_fpath=config["data"]["train_jsonl_fpath"],
        tokenizer=tokenizer,
    )
    val_dataset = setup_single_nemo_gym_dataset(
        jsonl_fpath=config["data"]["validation_jsonl_fpath"],
        tokenizer=tokenizer,
    )

    # Validation dataset config setup.
    if config["distillation"]["max_val_samples"] is not None:
        raise ValueError(
            """A non-null `distillation.max_val_samples` parameter is not supported.

Gym principle is that there is no hidden data pre or post processing from you. What you see is what you get.

The validation set you pass in will directly be used for validation with no additional preprocessing. If you want to have some number of repetitions, please include that in your dataset, via ``num_repeats``, in your dataset config and `ng_prepare_data` will prepare it accordingly."""
        )

    print(
        f"Setting `distillation.max_val_samples` and `distillation.val_batch_size` to the length of the validation dataset, which is {len(val_dataset)}"
    )
    config["distillation"]["max_val_samples"] = len(val_dataset)
    config["distillation"]["val_batch_size"] = config["distillation"]["max_val_samples"]

    # Print config
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # Setup NeMo Gym environment
    nemo_gym_config = NemoGymConfig(
        model_name=student_generation.cfg["model_name"],
        base_urls=student_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    nemo_gym = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.nemo_gym.NemoGym"
            ),
        }
    ).remote(nemo_gym_config)
    # Blocking wait for NeMo-Gym to spin up
    ray.get(nemo_gym.health_check.remote())
    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    # Run distillation training with NeMo Gym
    distillation_gym_train(
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        tokenizer,
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
