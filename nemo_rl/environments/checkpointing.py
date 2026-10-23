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

"""Pick the environment backend's rollout checkpoint for a run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemo_rl.experience.environment_checkpoint import (
    EnvironmentCheckpoint,
    NoEnvironmentCheckpoint,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.experience.rollout_manager import RolloutManager


def build_environment_checkpoint(
    master_config: MasterConfig,
    *,
    env_handles: dict[str, Any],
    rollout_manager: RolloutManager,
    dp_client: Any,
) -> EnvironmentCheckpoint:
    """Return the environment checkpoint this run's config asks for.

    Only turn-level recovery keeps environment state, and setup already
    requires NeMo Gym for it. A new backend that keeps rollout state adds its
    branch here.
    """
    if not master_config.rollout_recovery.turn_checkpointing_enabled:
        return NoEnvironmentCheckpoint()
    from nemo_rl.environments.gym_environment_checkpoint import (
        build_gym_environment_checkpoint,
    )

    return build_gym_environment_checkpoint(
        master_config,
        env_handles=env_handles,
        rollout_manager=rollout_manager,
        dp_client=dp_client,
    )
