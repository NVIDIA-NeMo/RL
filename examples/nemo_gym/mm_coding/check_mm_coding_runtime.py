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
"""Check Gym API and component discovery before allocating MM-coding training GPUs."""

import argparse
import contextlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

REQUIRED_MODULES = (
    "nemo_gym.core",
    "nemo_gym.config",
    "nemo_gym.server_runtime",
    "nemo_gym.server_runtime.sandbox_lending",
    "resources_servers.mm_coding_training.app",
    "resources_servers.mm_coding_training.session_app",
    "responses_api_agents.opencode_agent.app",
)


def check_runtime(
    *,
    mode: str = "single",
    harnesses: tuple[str, ...] = ("opencode",),
) -> list[str]:
    """Return missing API/component modules without starting services or importing candidate code."""
    missing = []
    modules = list(REQUIRED_MODULES)
    if mode in {"multi", "mixed"} and "opencode" in harnesses:
        modules.append("nemo_gym.models.token_capture.harness_notes")
    for module in modules:
        try:
            found = importlib.util.find_spec(module)
        except (ModuleNotFoundError, ImportError):
            found = None
        if found is None:
            missing.append(module)
    return missing


def check_training_config(config_path: Path) -> None:
    """Validate the resolved learner/generator before launching model workers."""
    # Actor discovery does not need the optional training dependencies.
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.single_controller_utils.config import (
        MasterConfig,
        validate_single_controller_config,
    )
    from nemo_rl.utils.config import (
        load_config_with_inheritance,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = MasterConfig.model_validate(
        OmegaConf.to_container(load_config_with_inheritance(config_path), resolve=True)
    )
    if (
        not config.policy["megatron_cfg"]["enabled"]
        or config.policy["automodel_cfg"]["enabled"]
    ):
        raise ValueError("MM-coding acceptance requires a Megatron learner")
    if config.policy["generation"]["backend"] != "vllm":
        raise ValueError("MM-coding acceptance requires vLLM generation")
    if (
        config.policy["generation"]["max_new_tokens"]
        >= config.policy["max_total_sequence_length"]
    ):
        raise ValueError(
            "Reserve input context: max_new_tokens must be below max_total_sequence_length"
        )
    if not config.token_capture.enabled:
        raise ValueError("MM-coding acceptance requires token_capture.enabled=true")
    validate_single_controller_config(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("single", "multi", "mixed"), default="single"
    )
    parser.add_argument(
        "--harnesses",
        nargs="+",
        choices=("opencode",),
        default=("opencode",),
    )
    parser.add_argument(
        "--config", type=Path, help="Resolved MM-coding training recipe"
    )
    args = parser.parse_args()
    if args.config is not None:
        with contextlib.redirect_stdout(sys.stderr):
            check_training_config(args.config)
    missing = check_runtime(mode=args.mode, harnesses=tuple(args.harnesses))
    print(
        json.dumps(
            {
                "compatible": not missing,
                "mode": args.mode,
                "missing_modules": missing,
                "harnesses": args.harnesses,
            }
        )
    )
    if missing:
        raise SystemExit(
            "Install the public Gym MM-coding revision and its dependencies in the driver, "
            "Gym actor, generation actor and component environments. Check the RL Gym pin: it may "
            "predate these APIs. OpenCode multi/mixed also needs Gym's harness-note token-capture "
            "fix (upstream PR #5833 or its merged successor). Re-run this check with the dataset's "
            "--mode and --harnesses in every runtime before allocating GPUs."
        )


if __name__ == "__main__":
    main()
