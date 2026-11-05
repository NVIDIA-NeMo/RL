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
"""Run the research-local Block JustGRPO Sudoku experiment."""

import argparse
from pathlib import Path


def main() -> None:
    from just_grpo.train import run
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml",
    )
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    parser.add_argument("--generation-python")
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    raw = parse_hydra_overrides(load_config(args.config), overrides)
    for argument, path in (
        ("model", "policy.model_name"),
        ("output_dir", "logger.log_dir"),
        ("generation_python", "just_grpo.generation_python"),
    ):
        value = vars(args)[argument]
        if value is not None:
            OmegaConf.update(raw, path, value)
    OmegaConf.resolve(raw)
    run(raw)


if __name__ == "__main__":
    main()
