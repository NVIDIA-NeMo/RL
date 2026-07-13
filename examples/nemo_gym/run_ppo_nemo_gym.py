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
import time

# Increase the W&B single object size warning threshold. Initially 100_000 (100 KB) -> 10_000_000 (10 MB)
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import _should_use_nemo_gym
from nemo_rl.algorithms.ppo import (
    MasterConfig,
    async_ppo_train,
    ppo_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run PPO training with NeMo-Gym")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main() -> None:
    """Main entry point."""
    main_start = time.perf_counter()
    log_container_init_timing()
    rl_init_timer = Timer(context={"worker": "driver"})

    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "ppo_math_rlvr_nemo_gym.yaml",
        )

    with rl_init_timer.time("config"):
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

    with rl_init_timer.time("tokenizer"):
        tokenizer = get_tokenizer(config.policy["tokenizer"])
        assert config.policy["generation"] is not None, (
            "A generation config is required for PPO"
        )
        config.policy["generation"] = configure_generation_config(
            config.policy["generation"], tokenizer
        )

        # NeMo-Gym specific config setup (forces async_engine / expose_http_server,
        # nulls stop_strings / stop_token_ids).
        setup_nemo_gym_config(config, tokenizer)

    # Assert here since this is right after the final config has been materialized.
    assert _should_use_nemo_gym(config)

    # NeMo-Gym env needs dp_openai_server_base_urls from policy_generation, so the
    # gym actor is created inside setup(); we don't build the env here.
    with rl_init_timer.time("data"):
        print("\n▶ Setting up data...")
        train_dataset, val_dataset = setup_response_data(
            tokenizer, config.data, env_configs=None
        )

    # Validation dataset config setup. Gym principle: what you pass in is used
    # verbatim (no hidden pre/post processing), so a preset max_val_samples is
    # rejected and the full val set is used as one batch.
    if config.ppo["max_val_samples"] is not None:
        raise ValueError(
            """A non-null `ppo.max_val_samples` parameter is not supported.

Gym principle is that there is no hidden data pre or post processing from you. What you see is what you get.

The validation set you pass in will directly be used for validation with no additional preprocessing. If you want to have some number of repetitions, please include that in your dataset, via ``num_repeats``, in your dataset config and `ng_prepare_data` will prepare it accordingly."""
        )

    if val_dataset is not None:
        print(
            f"Setting `ppo.max_val_samples` and `ppo.val_batch_size` to the length of the validation dataset, which is {len(val_dataset)}"
        )
        config.ppo["max_val_samples"] = len(val_dataset)
        config.ppo["val_batch_size"] = config.ppo["max_val_samples"]

    # Print config
    print("Final config:")
    pprint.pprint(config)

    with rl_init_timer.time("ray_connect"):
        init_ray()

    with rl_init_timer.time("setup"):
        (
            policy,
            policy_generation,
            nemo_gym,
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
        ) = setup(config, tokenizer, train_dataset, val_dataset)

    rl_init_timer.record("total", time.perf_counter() - main_start)
    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)

    # NeMo-Gym is spun up inside setup() (overlapped with vLLM model load).
    # Bind task_to_env / val_task_to_env for the nemo_gym env. Hardcode here to
    # match `run_async_nemo_gym_rollout`.
    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    if "async_ppo" in config.ppo and config.ppo["async_ppo"]["enabled"]:
        # Async PPO does not support dynamic sampling / reward scaling / reward shaping.
        unsupported_features = [
            "use_dynamic_sampling",
            "reward_scaling",
            "reward_shaping",
        ]
        for feature in unsupported_features:
            if feature not in config.ppo:
                continue
            if feature == "use_dynamic_sampling":
                if config.ppo[feature]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async PPO"
                    )
            elif config.ppo[feature]["enabled"]:
                raise NotImplementedError(f"{feature} is not supported with async PPO")

        print("🚀 Running async PPO training with NeMo-Gym")
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
        print("🚀 Running synchronous PPO training with NeMo-Gym")
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
