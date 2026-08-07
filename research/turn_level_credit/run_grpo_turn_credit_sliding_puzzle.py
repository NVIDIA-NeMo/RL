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

"""Run the native sliding-puzzle GRPO example with turn-level credit."""

import os
import sys

from examples import run_grpo_sliding_puzzle
from run_grpo_turn_credit import (
    load_master_and_turn_credit_config,
    parse_args,
)
from turn_level_credit.integration import install_turn_credit_runtime


def main() -> None:
    """Validate the research config, then delegate to the upstream runner."""
    args, overrides = parse_args()
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_sliding_puzzle_turn_credit.yaml",
        )
        # The upstream runner owns data construction and parses argv itself.
        sys.argv[1:1] = ["--config", config_path]

    _master_config, turn_credit_config = load_master_and_turn_credit_config(
        config_path,
        overrides,
    )
    with install_turn_credit_runtime(turn_credit_config):
        run_grpo_sliding_puzzle.main()


if __name__ == "__main__":
    main()
