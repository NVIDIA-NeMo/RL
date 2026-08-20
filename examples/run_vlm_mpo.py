# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Run offline image MPO with the canonical Nemotron Omni Megatron model."""

import argparse
import os
import pprint

from omegaconf import OmegaConf

from nemo_rl.algorithms.mpo import MasterConfig, mpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.processors import vlm_preference_preprocessor
from nemo_rl.data.utils import setup_preference_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run VLM MPO training")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_known_args()


def main() -> None:
    log_container_init_timing()
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if args.config is None:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "recipes",
            "vlm",
            "vlm_mpo-nemotron-omni-30ba3b-mmpr-1n8g-megatron-tp8.v1.yaml",
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    master_config = MasterConfig.model_validate(
        OmegaConf.to_container(config, resolve=True)
    )
    pprint.pprint(master_config.model_dump())

    master_config.logger["log_dir"] = get_next_experiment_dir(
        master_config.logger["log_dir"]
    )
    init_ray()

    processor = get_tokenizer(master_config.policy["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer
    train_dataset, val_dataset = setup_preference_data(
        processor,
        master_config.data,
        processor_fn=vlm_preference_preprocessor,
    )

    (
        policy,
        _cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        mpo_save_state,
        master_config,
    ) = setup(master_config, tokenizer, train_dataset, val_dataset)

    with checkpointer:
        mpo_train(
            policy,
            train_dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            master_config,
            logger,
            checkpointer,
            mpo_save_state,
        )


if __name__ == "__main__":
    main()
