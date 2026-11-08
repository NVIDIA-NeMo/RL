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


def check_runtime() -> list[str]:
    """Return missing API/component modules without starting services or importing candidate code."""
    missing = []
    for module in REQUIRED_MODULES:
        try:
            found = importlib.util.find_spec(module)
        except (ModuleNotFoundError, ImportError):
            found = None
        if found is None:
            missing.append(module)
    return missing


def main() -> None:
    missing = check_runtime()
    print(json.dumps({"compatible": not missing, "missing_modules": missing}))
    if missing:
        raise SystemExit(
            "Install the submitted internal Gym revision and its dependencies in the driver, "
            "Gym actor, generation actor and component environments. The stock RL Gym pin "
            "predates these APIs. Re-run this check in every runtime before allocating GPUs."
        )


if __name__ == "__main__":
    main()
