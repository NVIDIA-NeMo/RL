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

"""Launch offline GRPO over grouped, rewarded teacher trajectories."""

import argparse
import os
import pprint

from omegaconf import OmegaConf

from nemo_rl.algorithms.offline_grpo import MasterConfig, offline_grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.offline_grpo import setup_offline_grpo_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the config path and preserve Hydra-style overrides."""
    parser = argparse.ArgumentParser(description="Run offline GRPO training")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_known_args()


def main() -> None:
    """Load configuration, initialize NeMo RL, and train."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if args.config is None:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "offline_grpo.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    raw_config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw_config, dict):
        raise TypeError("The resolved offline GRPO config must be a mapping")
    master_config = MasterConfig.model_validate(raw_config)
    pprint.pprint(master_config)

    master_config.logger["log_dir"] = get_next_experiment_dir(
        master_config.logger["log_dir"]
    )
    init_ray()
    tokenizer = get_tokenizer(master_config.policy["tokenizer"])
    train_dataset, val_dataset = setup_offline_grpo_data(tokenizer, master_config.data)

    (
        policy,
        _cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        advantage_estimator,
        logger,
        checkpointer,
        save_state,
        master_config,
    ) = setup(master_config, tokenizer, train_dataset, val_dataset)
    with checkpointer:
        offline_grpo_train(
            policy,
            train_dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            advantage_estimator,
            master_config,
            logger,
            checkpointer,
            save_state,
        )


if __name__ == "__main__":
    main()
