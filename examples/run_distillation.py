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

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.distillation import MasterConfig, distillation_train, setup
from nemo_rl.algorithms.grpo import _should_use_nemo_gym
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import NemoGymConfig, setup_nemo_gym_config
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
    # Parse arguments
    register_omegaconf_resolvers()
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
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], tokenizer
        )
    else:
        print("  ⚠️ No generation config found, this may cause issues")

    use_nemo_gym = bool((config.get("env") or {}).get("should_use_nemo_gym"))
    if use_nemo_gym:
        setup_nemo_gym_config(config, tokenizer)
        assert _should_use_nemo_gym(config)

    # setup data
    if use_nemo_gym:
        dataset, val_dataset = setup_response_data(
            tokenizer, config["data"], env_configs=None
        )
        task_to_env = {}
        val_task_to_env = None
    else:
        (
            dataset,
            val_dataset,
            task_to_env,
            val_task_to_env,
        ) = setup_response_data(tokenizer, config["data"], config["env"])

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
    ) = setup(config, tokenizer, dataset, val_dataset)

    if use_nemo_gym:
        if student_generation is None:
            raise ValueError("NeMo-Gym distillation requires a vLLM generation backend")
        nemo_gym_config = NemoGymConfig(
            model_name=student_generation.cfg["model_name"],
            base_urls=student_generation.dp_openai_server_base_urls,
            initial_global_config_dict=config["env"]["nemo_gym"],
        )
        nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
        ray.get(nemo_gym.health_check.remote())
        task_to_env = {"nemo_gym": nemo_gym}
        val_task_to_env = task_to_env

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
