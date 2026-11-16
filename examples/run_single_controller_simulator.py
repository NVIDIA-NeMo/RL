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

"""SingleController simulator: run one side of the RL loop with the other mocked.

Keeps the SingleController's own pumps, TransferQueue data plane, sampler and
refit lifecycle, and swaps one side of the loop for a stand-in that uses no GPU:

  rollout_only    real generation and environments, ``cpu`` train backend
  train_only      real trainer fed by a ``cpu`` rollout backend (not implemented)
  train_rollout   both real: the ordinary run_grpo_single_controller.py run

Only ``rollout_only`` is implemented. It serves a recipe's rollouts exactly as
training would -- same engines, same Gym, same sampler, same pause/resume around
every refit -- while a CpuTrainer stands in for the train step and the weight
version advances without moving any weights.

Takes two YAML files: ``--config`` is an SC recipe, loaded as
run_grpo_single_controller.py loads it, and ``--simulator-config`` holds what
the stand-ins do (examples/configs/simulator/rollout_only.yaml). Hydra-style
overrides apply to the recipe, or to the simulator config when prefixed with
``simulator.``.
"""

import argparse
import os
import sys

from nemo_rl.algorithms.single_controller_utils.entrypoint_utils import (
    run_single_controller,
)

# Drop examples/ from sys.path so examples/nemo_gym/ (no __init__.py) doesn't
# shadow the real nemo_gym package as a namespace package.
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the two config paths; anything else is a Hydra-style override."""
    parser = argparse.ArgumentParser(
        description="Run one side of the SingleController RL loop with the other mocked"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to an SC YAML recipe"
    )
    parser.add_argument(
        "--simulator-config",
        type=str,
        default=None,
        help="Path to the simulator YAML. Default: "
        "examples/configs/simulator/rollout_only.yaml.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    args, overrides = parse_args()

    if not args.simulator_config:
        args.simulator_config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "simulator",
            "rollout_only.yaml",
        )

    run_single_controller(
        args.config, overrides, simulator_config_path=args.simulator_config
    )


if __name__ == "__main__":
    main()
