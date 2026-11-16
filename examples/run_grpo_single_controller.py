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

"""Async GRPO / PPO launcher driven by the SingleController actor.

Builds the full SC actor args driver-side via setup_single_controller and hands them
to SingleControllerActor. Mirrors run_grpo.py for config loading so the same YAML
files apply. data_plane.enabled=true is mandatory. A config carrying a `ppo:` block
additionally brings up the PPO critic and trains it alongside the policy.
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run async GRPO / PPO training via SingleController"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_math_1B_megatron_single_controller.yaml",
        )

    run_single_controller(args.config, overrides)


if __name__ == "__main__":
    main()
