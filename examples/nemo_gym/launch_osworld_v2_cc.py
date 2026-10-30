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

"""Validate or launch the minimal OSWorld v2 SingleController recipe.

Required environment variables:
  NANO_OMNI_MODEL_NAME
  NANO_OMNI_CHAT_TEMPLATE
  OSWORLD_GRPO_TRAIN_DATA

Examples:
  uv run --no-sync examples/nemo_gym/launch_osworld_v2_cc.py --validate-only
  uv run --no-sync examples/nemo_gym/launch_osworld_v2_cc.py --molt-b8k8 --validate-only
  uv run --no-sync examples/nemo_gym/launch_osworld_v2_cc.py --molt-b8k8-checkpoint --validate-only
  uv run --no-sync examples/nemo_gym/launch_osworld_v2_cc.py logger.log_dir=/logs/run
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

CONFIG_PATH = Path(__file__).with_name("grpo_nemotron_omni_30ba3b_osworld_v2_cc.yaml")
MOLT_CONFIG_PATH = Path(__file__).with_name(
    "grpo_nemotron_omni_30ba3b_osworld_v2_cc_molt_b8k8.yaml"
)
MOLT_CHECKPOINT_CONFIG_PATH = Path(__file__).with_name(
    "grpo_nemotron_omni_30ba3b_osworld_v2_cc_molt_b8k8_checkpoint.yaml"
)


def compose_and_validate_config(
    overrides: Sequence[str], *, config_path: Path = CONFIG_PATH
) -> MasterConfig:
    """Compose the recipe and apply all SingleController startup checks."""
    register_omegaconf_resolvers()
    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, list(overrides))
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("OSWorld recipe did not resolve to a mapping")
    master_config = MasterConfig.model_validate(resolved)
    validate_single_controller_config(master_config)
    return master_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher flags while retaining Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compose and validate without initializing Ray or loading the model.",
    )
    parser.add_argument(
        "--molt-b8k8",
        action="store_true",
        help="Use the asynchronous Molt B8/K8 recipe instead of the minimal smoke.",
    )
    parser.add_argument(
        "--molt-b8k8-checkpoint",
        action="store_true",
        help="Use the Molt B8/K8 recipe with full step-boundary checkpointing.",
    )
    return parser.parse_known_args()


def main() -> None:
    """Validate the dedicated recipe, then optionally run its SC entrypoint."""
    args, overrides = parse_args()
    if args.molt_b8k8 and args.molt_b8k8_checkpoint:
        raise ValueError(
            "--molt-b8k8 and --molt-b8k8-checkpoint are mutually exclusive"
        )
    if args.molt_b8k8_checkpoint:
        config_path = MOLT_CHECKPOINT_CONFIG_PATH
    elif args.molt_b8k8:
        config_path = MOLT_CONFIG_PATH
    else:
        config_path = CONFIG_PATH
    config = compose_and_validate_config(overrides, config_path=config_path)
    if args.validate_only:
        assert config.grpo is not None
        print(
            "Validated OSWorld v2 CC recipe: "
            f"{config.grpo.num_prompts_per_step} prompt x "
            f"{config.grpo.num_generations_per_prompt} generations, "
            f"sampler={config.async_rl.sampler.name}, "
            f"data_plane={config.data_plane['impl']}."
        )
        return

    # Import the runtime driver only after validation so --validate-only remains
    # a model-free configuration check.
    from examples import run_grpo_single_controller  # noqa: PLC0415

    sys.argv = [
        str(Path(run_grpo_single_controller.__file__)),
        "--config",
        str(config_path),
        *overrides,
    ]
    run_grpo_single_controller.main()


if __name__ == "__main__":
    main()
