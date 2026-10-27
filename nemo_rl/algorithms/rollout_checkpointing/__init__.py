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

"""Rollout checkpointing for the single controller, behind one small API."""

from nemo_rl.algorithms.rollout_checkpointing.controller_state import (
    ControllerRolloutState,
    ControllerSidecars,
    write_controller_sidecars,
)
from nemo_rl.algorithms.rollout_checkpointing.data_plane import (
    DataPlaneRolloutCheckpointing,
    RolloutCheckpointCut,
    RolloutCheckpointSaveResult,
)
from nemo_rl.algorithms.rollout_checkpointing.interface import (
    NoRolloutCheckpointing,
    RolloutCheckpointing,
    RolloutCheckpointingHost,
)
from nemo_rl.algorithms.rollout_checkpointing.inventory import (
    capture_reward_settings,
    validate_replay_inventory,
    validate_rollout_recovery_inventory,
)

__all__ = [
    "ControllerRolloutState",
    "ControllerSidecars",
    "DataPlaneRolloutCheckpointing",
    "NoRolloutCheckpointing",
    "RolloutCheckpointCut",
    "RolloutCheckpointSaveResult",
    "RolloutCheckpointing",
    "RolloutCheckpointingHost",
    "capture_reward_settings",
    "validate_replay_inventory",
    "validate_rollout_recovery_inventory",
    "write_controller_sidecars",
]
