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

"""The controller state a checkpoint saves beside its TQ snapshot."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from nemo_rl.algorithms.async_utils.replay_buffer import (
    REPLACEMENT_RESERVE_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    TQReplayMetadataState,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.utils import save_dataloader_state
from nemo_rl.experience.rollout_recovery import ROLLOUT_RECOVERY_STATE_FILENAME
from nemo_rl.utils.path_utils import save_checkpoint_object

if TYPE_CHECKING:
    from nemo_rl.experience.reward_penalties import FinalizedReward
    from nemo_rl.utils.checkpoint import PathLike


@dataclass(frozen=True)
class ControllerRolloutState:
    """The controller's own state that a checkpoint persists.

    The controller builds it when asked, under the data-plane checkpoint cut, so
    it matches the TQ snapshot taken in the same cut. The dictionaries are the
    controller's live objects, not copies; they are serialized under that cut.

    Attributes:
        train_steps: Completed train steps.
        trainer_version: The policy version the trainer has published.
        current_epoch: The rollout pump's dataloader epoch.
        dataloader_state: ``state_dict()`` of the training dataloader.
        sampler_dispatch_index: The sampler's admission cursor.
        sampler_supports_buffer_checkpoint: Whether the sampler can restore
            buffered groups, which decides whether a trainer checkpoint saves
            the replay index.
        replacement_reserve: Spare prompts pulled from the dataloader but not
            yet dispatched.
        batch_shortfall: Groups each stamped step will never receive.
        sampler_stamps_target_steps: Whether the sampler has stamped a step.
        finalizer_metrics_by_group: Pending finalizer metrics per group.
        finalizer_rewards_by_group: Pending finalizer rewards per group.
        rollout_recovery_enabled: Whether dispatch records unfinished groups
            in the recovery ledger, so a trainer checkpoint must save it.
    """

    train_steps: int
    trainer_version: int
    current_epoch: int
    dataloader_state: dict[str, Any]
    sampler_dispatch_index: int
    sampler_supports_buffer_checkpoint: bool
    replacement_reserve: list[DatumSpec]
    batch_shortfall: dict[int, int]
    sampler_stamps_target_steps: bool
    finalizer_metrics_by_group: dict[str, dict[str, float]]
    finalizer_rewards_by_group: dict[str, list[FinalizedReward]]
    rollout_recovery_enabled: bool


@dataclass(frozen=True)
class ControllerSidecars:
    """Controller files a checkpoint writes beside its TQ snapshot.

    Attributes:
        dataloader_state: Saved as ``train_dataloader.pt``.
        replacement_reserve: Saved only when not empty.
        replay_metadata: The metadata-only replay index, or ``None`` when the
            checkpoint does not save one.
        rollout_recovery_payload: The serialized recovery ledger, or ``None``
            when the checkpoint does not save one.
    """

    dataloader_state: dict[str, Any]
    replacement_reserve: list[DatumSpec]
    replay_metadata: Optional[TQReplayMetadataState] = None
    rollout_recovery_payload: Optional[bytes] = None


async def write_controller_sidecars(
    checkpoint_path: PathLike, sidecars: ControllerSidecars
) -> list[str]:
    """Write controller state beside TQ and return the paths written.

    Works for local and ``msc://`` checkpoint paths. Everything here must be on
    disk before the checkpoint is finalized.
    """
    written_paths: list[str] = []
    await asyncio.to_thread(
        save_dataloader_state, sidecars.dataloader_state, checkpoint_path
    )
    written_paths.append(os.path.join(checkpoint_path, "train_dataloader.pt"))
    if sidecars.replacement_reserve:
        replacement_reserve_path = os.path.join(
            checkpoint_path, REPLACEMENT_RESERVE_FILENAME
        )
        await asyncio.to_thread(
            save_checkpoint_object,
            sidecars.replacement_reserve,
            replacement_reserve_path,
        )
        written_paths.append(replacement_reserve_path)
    if sidecars.replay_metadata is not None:
        replay_metadata_path = os.path.join(
            checkpoint_path, REPLAY_BUFFER_METADATA_FILENAME
        )
        await asyncio.to_thread(
            save_checkpoint_object,
            sidecars.replay_metadata,
            replay_metadata_path,
        )
        written_paths.append(replay_metadata_path)
    if sidecars.rollout_recovery_payload is not None:
        rollout_recovery_path = os.path.join(
            checkpoint_path, ROLLOUT_RECOVERY_STATE_FILENAME
        )
        await asyncio.to_thread(
            Path(rollout_recovery_path).write_bytes,
            sidecars.rollout_recovery_payload,
        )
        written_paths.append(rollout_recovery_path)
    return written_paths
