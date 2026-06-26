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

"""AReaL decoupled-PPO GRPO launcher driven by SingleControllerArealActor.

Identical wiring to run_grpo_single_controller.py (same config loading,
setup_single_controller bundle, data_plane.enabled=true requirement); it only
swaps the orchestrator to SingleControllerArealActor, which runs the AReaL
two-phase decoupled-PPO train loop + interruptible refit (AREAL.md §9 P0-P4).
Pair it with an AReaL recipe (use_importance_sampling_correction=true, the
startup assert in single_controller_areal.py requires it).
"""

import argparse
import os
import pprint

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_areal import SingleControllerArealActor
from nemo_rl.algorithms.single_controller_utils import (
    MasterConfig,
    setup_single_controller,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import init_ray
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
        description="Run AReaL decoupled-PPO GRPO training via SingleControllerArealActor"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "recipes",
            "llm",
            "grpo-llama3.1-8b-instruct-2n4g-areal-single-controller.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    dp_cfg = config.data_plane
    if not dp_cfg.get("enabled", False):
        raise ValueError(
            "run_grpo_areal requires data_plane.enabled=true. "
            "Use examples/run_grpo.py for the legacy / sync paths."
        )

    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for SC-driven AReaL GRPO"
    )
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
    )

    bundle = setup_single_controller(config, tokenizer)

    print("🚀 Launching SingleControllerArealActor")
    sc = SingleControllerArealActor.remote(master_config=config, bundle=bundle)
    result = ray.get(sc.run.remote())
    print(f"AReaL SC run complete: {result}")


if __name__ == "__main__":
    main()
