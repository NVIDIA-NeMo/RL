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

"""Native synchronous GRPO with a research-local LoGRA worker."""

import argparse

from logra.actor_environments import register_actor_environments
from logra.config import LoGRAConfig
from logra.policy import make_policy_factory
from omegaconf import OmegaConf
from pydantic import Field

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
from nemo_rl.utils.logger import get_next_experiment_dir


class ResearchConfig(MasterConfig):
    logra: LoGRAConfig = Field(default_factory=LoGRAConfig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_logra.yaml")
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    raw = parse_hydra_overrides(load_config(args.config), overrides)
    config = ResearchConfig.model_validate(OmegaConf.to_container(raw, resolve=True))
    if config.grpo.async_grpo.enabled or (
        config.data_plane is not None and config.data_plane["enabled"]
    ):
        raise ValueError(
            "Initial research support is native synchronous GRPO without the data plane"
        )
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    register_actor_environments()
    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )
    datasets = setup_response_data(tokenizer, config.data, config.env)
    if len(datasets) != 4:
        raise ValueError("GRPO requires training and validation reward environments")
    dataset, val_dataset, task_to_env, val_task_to_env = datasets
    (
        policy,
        generation,
        _,
        cluster,
        loader,
        val_loader,
        loss,
        logger,
        checkpointer,
        state,
        master,
        _,
        _,
    ) = setup(
        config,
        tokenizer,
        dataset,
        val_dataset,
        policy_factory=make_policy_factory(config.logra),
    )
    try:
        with checkpointer:
            grpo_train(
                policy,
                generation,
                loader,
                val_loader,
                tokenizer,
                loss,
                task_to_env,
                val_task_to_env,
                logger,
                checkpointer,
                state,
                master,
            )
    finally:
        generation.shutdown()
        policy.shutdown()


if __name__ == "__main__":
    main()
