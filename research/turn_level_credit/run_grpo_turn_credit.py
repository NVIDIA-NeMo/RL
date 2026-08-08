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

"""Run legacy synchronous GRPO with experimental native turn-level credit."""

import argparse
import os
import pprint
import time

from omegaconf import OmegaConf
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.integration import install_turn_credit_runtime
from turn_level_credit.validation import validate_supported_path

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the config path and Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Run synchronous GRPO with native turn-level credit"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file",
    )
    return parser.parse_known_args()


def load_master_and_turn_credit_config(
    config_path: str,
    overrides: list[str],
) -> tuple[MasterConfig, TurnCreditConfig]:
    """Load core and research configuration from one YAML file."""
    register_omegaconf_resolvers()
    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved configuration must be a dictionary")
    turn_credit_config = TurnCreditConfig.model_validate(
        resolved.get("turn_credit", {})
    )
    master_config = MasterConfig.model_validate(resolved)
    validate_supported_path(master_config, turn_credit_config)
    return master_config, turn_credit_config


def main() -> None:
    """Set up NeMo-RL and run the supported synchronous GRPO experiment."""
    main_start = time.perf_counter()
    log_container_init_timing()
    init_timer = Timer(context={"worker": "driver"})

    args, overrides = parse_args()
    if args.config is None:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_math_0.5b_turn_credit.yaml",
        )

    with init_timer.time("config"):
        config, turn_credit_config = load_master_and_turn_credit_config(
            args.config,
            overrides,
        )
    print(f"Loaded configuration from: {args.config}")
    print("Final core config:")
    pprint.pprint(config)
    print("Turn-credit config:")
    pprint.pprint(turn_credit_config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    with init_timer.time("ray_connect"):
        init_ray()

    with init_timer.time("tokenizer"):
        tokenizer = get_tokenizer(config.policy["tokenizer"])
        generation_config = config.policy["generation"]
        if generation_config is None:
            raise ValueError("A generation config is required for GRPO")
        has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
        megatron_config = config.policy.get("megatron_cfg") or {}
        config.policy["generation"] = configure_generation_config(
            generation_config,
            tokenizer,
            has_refit_draft_weights=has_refit_draft_weights,
            trains_mtp=bool(megatron_config.get("mtp_num_layers")),
        )

    with init_timer.time("data"):
        data_result = setup_response_data(
            tokenizer,
            config.data,
            config.env,
        )
        if len(data_result) != 4:
            raise RuntimeError(
                "Native environment setup must return datasets and environment maps"
            )
        dataset, val_dataset, task_to_env, val_task_to_env = data_result

    with install_turn_credit_runtime(turn_credit_config):
        with init_timer.time("setup"):
            (
                policy,
                policy_generation,
                _nemo_gym,
                _cluster,
                dataloader,
                val_dataloader,
                loss_fn,
                logger,
                checkpointer,
                grpo_state,
                master_config,
                _teacher_worker_groups,
                _alias_to_group_alias,
            ) = setup(
                config,
                tokenizer,
                dataset,
                val_dataset,
            )

        init_timer.record("total", time.perf_counter() - main_start)
        if policy_generation is None:
            raise RuntimeError("Policy generation was not initialized")
        with checkpointer:
            grpo_train(
                policy=policy,
                policy_generation=policy_generation,
                wrapped_dataloader=dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                task_to_env=task_to_env,
                val_task_to_env=val_task_to_env,
                logger=logger,
                checkpointer=checkpointer,
                grpo_save_state=grpo_state,
                master_config=master_config,
            )


if __name__ == "__main__":
    main()
