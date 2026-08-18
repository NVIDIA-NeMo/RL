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

"""Central agent loop for NeMo-Gym rollouts.

The loop replaces the per-rollout NeMo-Gym agent server (``simple_agent/app.py``)
with one loop owned by NeMo-RL and run inside the ``NemoGym`` Ray actor. Nothing
here imports ``nemo_gym``: the loop takes an injected HTTP client, so it is
importable and unit-testable from the driver venv.

Each of the four pluggable pieces has a registry and a factory keyed off its
config block, so a new implementation is one dict entry:

    ROLLOUT_TREES        <- central_agent.rollout_tree.type
    TOOL_CALL_MANAGERS   <- central_agent.tool_calls.mode
    STEERING_MANAGERS    <- central_agent.steering_message.enabled / .type
    TERMINATION_MANAGERS <- central_agent.termination.type

Design: ``ideas/central-controller/index.html`` in the design-docs repo.
"""

from nemo_rl.environments.central_agent_helpers.agent_loop import CentralAgent
from nemo_rl.environments.central_agent_helpers.config import (
    DEFAULT_CENTRAL_AGENT_CONFIG,
    SUPPORTED_AGENT_IMPLEMENTATIONS,
    merge_central_agent_config,
    pop_agent_registry,
    resolve_central_agent_config,
)
from nemo_rl.environments.central_agent_helpers.rollout_tree import (
    ROLLOUT_TREES,
    build_rollout_tree,
)
from nemo_rl.environments.central_agent_helpers.steering_manager import (
    STEERING_MANAGERS,
    build_steering_manager,
)
from nemo_rl.environments.central_agent_helpers.termination_manager import (
    TERMINATION_MANAGERS,
    build_termination_manager,
)
from nemo_rl.environments.central_agent_helpers.tool_call_manager import (
    TOOL_CALL_MANAGERS,
    build_tool_call_manager,
)

__all__ = [
    "CentralAgent",
    "DEFAULT_CENTRAL_AGENT_CONFIG",
    "ROLLOUT_TREES",
    "STEERING_MANAGERS",
    "SUPPORTED_AGENT_IMPLEMENTATIONS",
    "TERMINATION_MANAGERS",
    "TOOL_CALL_MANAGERS",
    "build_rollout_tree",
    "build_steering_manager",
    "build_termination_manager",
    "build_tool_call_manager",
    "merge_central_agent_config",
    "pop_agent_registry",
    "resolve_central_agent_config",
]
