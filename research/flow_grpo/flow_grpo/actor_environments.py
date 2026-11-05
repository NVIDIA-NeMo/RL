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

"""Select the research project's dependencies for policy and reward actors."""

import os
import shlex
from pathlib import Path

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

POLICY_WORKER = "flow_grpo.models.policy.workers.flow_grpo_worker.FlowGRPOPolicyWorker"
REWARD_WORKER = "flow_grpo.environments.image_reward_environment._RewardWorker"


def register_actor_environments() -> None:
    """Register project-local actor commands before constructing any workers."""
    project_dir = Path(__file__).resolve().parents[1]
    executable = (
        PY_EXECUTABLES.SYSTEM
        if os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"
        else f"uv run --locked --directory {shlex.quote(str(project_dir))}"
    )
    ACTOR_ENVIRONMENT_REGISTRY[POLICY_WORKER] = executable
    ACTOR_ENVIRONMENT_REGISTRY[REWARD_WORKER] = executable
