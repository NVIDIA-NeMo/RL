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
"""Run GDPO for a masked diffusion language model."""

import argparse
import os
import pprint

from gdpo.generation import AutomodelGeneration
from gdpo.setup import validate_gdpo_config
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.utils import shutdown_environments
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.telemetry.instrumentation import setup_span, startup_span
from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from nemo_rl.utils.outdated_config_checks import check_outdated_config

WORKER_FQN = "gdpo.worker.AutomodelGDPOPolicyWorker"
ACTOR_ENVIRONMENT_REGISTRY[WORKER_FQN] = PY_EXECUTABLES.AUTOMODEL


def _make_policy(**kwargs):
    """Construct a policy using the GDPO worker extension."""
    return Policy(**kwargs, worker_extension_cls_fqn=WORKER_FQN)


def _make_generation(policy_config, policy):
    """Wrap the live Automodel policy as a generation interface."""
    return AutomodelGeneration(policy_config, policy)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the config path and Hydra overrides."""
    parser = argparse.ArgumentParser(description="Run GDPO")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_known_args()


def main() -> None:
    """Set up and train a masked diffusion policy with GDPO."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if args.config is None:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "gdpo_llada_8b.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    check_outdated_config(config)
    config = MasterConfig(**config)
    pprint.pprint(config)

    validate_gdpo_config(config.policy, config.loss_fn, config.grpo)
    if config.data_plane is not None and config.data_plane["enabled"]:
        raise ValueError("GDPO does not support data_plane.enabled=true")

    config.logger.log_dir = get_next_experiment_dir(config.logger.log_dir)
    init_telemetry_driver(config, algorithm="grpo")
    try:
        with startup_span():
            init_ray()
            with setup_span("tokenizer"):
                tokenizer = get_tokenizer(config.policy["tokenizer"])
                assert config.policy["generation"] is not None, (
                    "A generation config is required for GDPO"
                )
                config.policy["generation"] = configure_generation_config(
                    config.policy["generation"], tokenizer
                )
            with setup_span("data"):
                dataset, val_dataset, task_to_env, val_task_to_env = (
                    setup_response_data(tokenizer, config.data, config.env)
                )
            with setup_span("workers"):
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
                    policy_factory=_make_policy,
                    generation_factory=_make_generation,
                    generation_logprobs_available=False,
                )
        assert policy_generation is not None

        # grpo_train tears down its own environments once it starts.
        trainer_owns_environment_shutdown = False
        try:
            with checkpointer:
                trainer_owns_environment_shutdown = True
                grpo_train(
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
                )
        finally:
            if not trainer_owns_environment_shutdown:
                shutdown_environments(task_to_env, val_task_to_env)
            policy_generation.shutdown()
            policy.shutdown()
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
