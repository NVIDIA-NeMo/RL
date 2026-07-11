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

from nemo_rl.algorithms.ppo import MasterConfig, async_ppo_train, ppo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run PPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def _async_ppo_enabled(config: MasterConfig) -> bool:
    """Whether async PPO is requested in the config."""
    return "async_ppo" in config.ppo and config.ppo["async_ppo"]["enabled"]


def _validate_async_ppo_config(config: MasterConfig) -> None:
    """Reject async-incompatible config up front (before setup()/vLLM init).

    Async PPO does not support DAPO-style dynamic sampling / reward scaling /
    reward shaping, nor multiple dataloaders.
    """
    if not _async_ppo_enabled(config):
        return
    for feature in ("use_dynamic_sampling", "reward_scaling", "reward_shaping"):
        if feature not in config.ppo:
            continue
        if feature == "use_dynamic_sampling":
            if config.ppo[feature]:
                raise NotImplementedError(f"{feature} is not supported with async PPO")
        elif config.ppo[feature]["enabled"]:
            raise NotImplementedError(f"{feature} is not supported with async PPO")
    if config.data.get("use_multiple_dataloader"):
        raise NotImplementedError(
            "use_multiple_dataloader is not supported with async PPO"
        )


def main() -> None:
    """Main entry point."""
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "ppo_math_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    # Fail fast (before the expensive setup()/vLLM init) on async-incompatible
    # config, so users don't wait through a full worker init to be rejected.
    _validate_async_ppo_config(config)

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for PPO"
    )
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_response_data(tokenizer, config.data, config.env)

    (
        policy,
        policy_generation,
        value_model,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        value_loss_fn,
        logger,
        checkpointer,
        ppo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Dispatch to async PPO when enabled, otherwise standard synchronous PPO.
    if _async_ppo_enabled(config):
        print("🚀 Running asynchronous PPO training")
        async_config = config.ppo["async_ppo"]
        async_ppo_train(
            policy,
            policy_generation,
            value_model,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            value_loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            ppo_state,
            master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        print("🚀 Running synchronous PPO training")

        # Run standard PPO training
        ppo_train(
            policy,
            policy_generation,
            value_model,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            value_loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            ppo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
