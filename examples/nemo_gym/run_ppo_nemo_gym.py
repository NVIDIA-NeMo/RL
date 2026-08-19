# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_rl.algorithms.ppo import (
    MasterConfig,
    async_ppo_train,
    ppo_train,
    setup,
    validate_async_ppo_config,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    setup_nemo_gym_config,
    should_use_nemo_gym,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PPO training with NeMo Gym")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_known_args()


def main() -> None:
    """Run synchronous or asynchronous PPO with NeMo Gym rollouts."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "ppo_math_rlvr_nemo_gym.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    print("Applied CLI overrides")

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for PPO"
    )
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )
    setup_nemo_gym_config(config, tokenizer)
    assert should_use_nemo_gym(config)

    print("\n▶ Setting up data...")
    dataset, val_dataset = setup_response_data(tokenizer, config.data, env_configs=None)

    if config.ppo.max_val_samples is not None:
        raise ValueError(
            "ppo.max_val_samples must be null for NeMo Gym; validation uses "
            "the complete prepared validation dataset"
        )
    if val_dataset is not None:
        config.ppo.max_val_samples = len(val_dataset)
        config.ppo.val_batch_size = len(val_dataset)

    print("Final config:")
    pprint.pprint(config)
    init_ray()

    (
        policy,
        policy_generation,
        nemo_gym,
        value_model,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        value_loss_fn,
        logger,
        checkpointer,
        ppo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)
    assert nemo_gym is not None

    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env
    async_ppo_enabled = config.ppo.async_ppo.enabled
    if async_ppo_enabled:
        validate_async_ppo_config(config, policy_generation)

    with checkpointer:
        if async_ppo_enabled:
            print("🚀 Running asynchronous PPO training with NeMo Gym")
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
            )
        else:
            print("🚀 Running synchronous PPO training with NeMo Gym")
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
