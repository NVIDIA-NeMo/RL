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

"""Run single-controller SFT with colocated Energon loaders."""

from __future__ import annotations

import argparse
import os
import pprint
import warnings

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.sft_v2 import (
    MasterConfig,
    SFTSingleControllerActor,
    setup_sft_v2,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the config path and Hydra-style overrides."""
    parser = argparse.ArgumentParser(description="Run SFTv2 training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "configs",
            "recipes",
            "vlm",
            "vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml",
        ),
    )
    return parser.parse_known_args()


def main() -> None:
    """Load configuration and start the SFT single controller."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    master_config = MasterConfig.model_validate(resolved)
    master_config.logger["log_dir"] = get_next_experiment_dir(
        master_config.logger["log_dir"]
    )
    pprint.pprint(master_config.model_dump())

    # NEMO_RL_OTEL_* env is snapshotted into the Ray runtime_env and inherited
    # by every worker, so this has to run before init_ray().
    init_telemetry_driver(master_config, algorithm="sft_v2")

    init_ray()
    processor = get_tokenizer(master_config.policy["tokenizer"], get_processor=True)
    actor_args = setup_sft_v2(master_config, processor)
    controller = SFTSingleControllerActor.remote(master_config, actor_args)
    try:
        result = ray.get(controller.run.remote())
        pprint.pprint(result)
    finally:
        try:
            actor_args.trainer.shutdown()
        except Exception as error:  # teardown must preserve the controller failure
            warnings.warn(f"SFTv2 trainer shutdown failed: {error}", stacklevel=2)
        shutdown_telemetry()


if __name__ == "__main__":
    main()
