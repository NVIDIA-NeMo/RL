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
"""Check Gym API and component discovery before allocating Super VL training GPUs."""

import argparse
import importlib.util
import json

REQUIRED_MODULES = (
    "nemo_gym.core",
    "nemo_gym.config",
    "nemo_gym.server_runtime",
    "nemo_gym.resources_server_sandbox_sessions",
    "resources_servers.mm_coding_training.app",
    "resources_servers.mm_coding_training.session_app",
    "responses_api_agents.hermes_agent.image_observations",
)


def check_runtime(
    *,
    mode: str = "single",
    harnesses: tuple[str, ...] = ("opencode", "codex", "hermes"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("single", "multi", "mixed"), default="single"
    )
    parser.add_argument(
        "--harnesses",
        nargs="+",
        choices=("opencode", "codex", "hermes"),
        default=("opencode", "codex", "hermes"),
    )
    args = parser.parse_args()
    missing = check_runtime(mode=args.mode, harnesses=tuple(args.harnesses))
    print(
        json.dumps(
            {"compatible": not missing, "mode": args.mode, "missing_modules": missing}
        )
    )
    if missing:
        raise SystemExit(
            "Install the submitted internal Gym revision and its dependencies in the driver, "
            "Gym actor, generation actor and component environments. The stock RL Gym pin "
            "predates these APIs. OpenCode multi/mixed also needs Gym's harness-note token-capture "
            "fix (upstream PR #5833 or its merged successor). Re-run this check with the dataset's "
            "--mode and --harnesses in every runtime before allocating GPUs."
        )


if __name__ == "__main__":
    main()
