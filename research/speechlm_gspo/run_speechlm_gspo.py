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

"""GSPO training of native NeMo SpeechLM ASR models."""

import argparse
import os
import pprint

from omegaconf import OmegaConf
from speechlm_gspo.actor_environments import (
    ASR_ENVIRONMENT,
    register_actor_environments,
)
from speechlm_gspo.data import Nemotron3_5TranscribeProcessor, asr_data_processor

from nemo_rl.algorithms.grpo import MasterConfig, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.processors import register_processor
from nemo_rl.data.utils import setup_response_data
from nemo_rl.data_plane.factory import (
    make_policy_factory,
    maybe_configure_data_plane_env,
    select_sync_trainer,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.utils import register_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from nemo_rl.utils.outdated_config_checks import check_outdated_config

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "configs",
    "recipes",
    "gspo-nemotron-3.5-transcribe-30b-a4b-4n8g-automodel-ep8.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Recipe YAML")
    args, overrides = parser.parse_known_args()

    register_omegaconf_resolvers()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    check_outdated_config(config)
    config = MasterConfig(**config)
    if config.grpo.async_grpo.enabled:
        raise NotImplementedError("SpeechLM GSPO supports synchronous GRPO only")
    pprint.pprint(config)
    config.logger.log_dir = get_next_experiment_dir(config.logger.log_dir)

    register_actor_environments()
    register_env("asr", ASR_ENVIRONMENT)
    register_processor("asr", asr_data_processor)
    maybe_configure_data_plane_env(config.data_plane)
    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    # Workers resolve the same class through AutoProcessor in the policy env.
    processor = Nemotron3_5TranscribeProcessor(tokenizer)
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        processor, config.data, config.env
    )
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
        processor=processor,
        policy_factory=make_policy_factory(config.data_plane),
    )
    trainer = select_sync_trainer(master_config, label="SpeechLM GSPO")
    with checkpointer:
        trainer(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            processor=processor,
        )


if __name__ == "__main__":
    main()
