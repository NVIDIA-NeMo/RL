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

"""CLI entrypoint for offline trajectory-value critic pretraining."""

import argparse
import os
import pprint
from typing import cast

from omegaconf import OmegaConf

from nemo_rl.algorithms.critic_pretraining import (
    MasterConfig,
    critic_pretrain,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Pretrain a scalar trajectory-prefix critic"
    )
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_known_args()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "critic_pretraining.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("the resolved critic-pretraining config must be a mapping")
    master_config = MasterConfig.model_validate(resolved)
    master_config.logger["log_dir"] = get_next_experiment_dir(
        master_config.logger["log_dir"]
    )
    print("Final config:")
    pprint.pprint(master_config)

    init_ray()
    tokenizer = get_tokenizer(master_config.value["tokenizer"])
    train_dataset, validation_dataset = cast(
        tuple[
            AllTaskProcessedDataset | dict[str, AllTaskProcessedDataset],
            AllTaskProcessedDataset | None,
        ],
        setup_response_data(tokenizer, master_config.data, env_configs=None),
    )
    if isinstance(train_dataset, dict):
        raise ValueError("critic pretraining requires one merged training dataloader")
    if validation_dataset is None:
        validation_datasets: dict[str, AllTaskProcessedDataset] = {}
    else:
        validation_datasets = {"test": validation_dataset}

    (
        value_model,
        _cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
    ) = setup(
        master_config,
        tokenizer,
        train_dataset,
        validation_datasets,
    )
    critic_pretrain(
        value_model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        save_state,
    )


if __name__ == "__main__":
    main()
