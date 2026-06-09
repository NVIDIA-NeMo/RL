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

# Increase the W&B single object size warning threshold. Initially 100_000 (100 KB) -> 10_000_000 (10 MB)
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.distillation import (
    MasterConfig,
    distillation_train,
    setup,
)
from nemo_rl.algorithms.grpo import _should_use_nemo_gym
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    NemoGymConfig,
    setup_nemo_gym_config,
)
from nemo_rl.environments.utils import create_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


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


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "distillation_math.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    # setup tokenizer
    tokenizer = get_tokenizer(config.policy["tokenizer"])

    if config.policy["generation"] is not None:
        config.policy["generation"] = configure_generation_config(
            config.policy["generation"], tokenizer
        )
    else:
        raise ValueError(
            "A vLLM generation config is required for NeMo-Gym distillation"
        )

    # NeMo-Gym specific config setup.
    setup_nemo_gym_config(config, tokenizer)

    # We assert here since this is right after the final config has been materialized.
    assert _should_use_nemo_gym(config)

    # NeMo-Gym environment needs to get dp_openai_server_base_urls from
    # student_generation, so we don't setup env here.
    print("\n▶ Setting up data...")
    train_dataset, val_dataset = setup_response_data(
        tokenizer, config.data, env_configs=None
    )

    # Validation dataset config setup. Same Gym principle as run_grpo_nemo_gym.py:
    # max_val_samples is derived from len(val_dataset); user-set values are rejected.
    if config.distillation["max_val_samples"] is not None:
        raise ValueError(
            """A non-null `distillation.max_val_samples` parameter is not supported.

Gym principle is that there is no hidden data pre or post processing from you. What you see is what you get.

The validation set you pass in will directly be used for validation with no additional preprocessing. If you want to have some number of repetitions, please include that in your dataset, via ``num_repeats``, in your dataset config and `ng_prepare_data` will prepare it accordingly."""
        )

    if val_dataset is not None:
        print(
            f"Setting `distillation.max_val_samples` and `distillation.val_batch_size` to the length of the validation dataset, which is {len(val_dataset)}"
        )
        config.distillation["max_val_samples"] = len(val_dataset)
        config.distillation["val_batch_size"] = config.distillation["max_val_samples"]

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

    if student_generation is None:
        raise ValueError("NeMo-Gym distillation requires a vLLM generation backend")

    nemo_gym_config = NemoGymConfig(
        model_name=student_generation.cfg["model_name"],
        base_urls=student_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config.env["nemo_gym"],
    )
    nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
    # Blocking wait for NeMo-Gym to spin up
    ray.get(nemo_gym.health_check.remote())

    # Bind task_to_env and val_task_to_env for nemo_gym env
    # Hardcode here to match `run_async_nemo_gym_rollout`
    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    print("🚀 Running distillation training")
    distillation_train(
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
