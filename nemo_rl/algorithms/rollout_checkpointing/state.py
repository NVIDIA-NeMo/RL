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

"""Rollout state the single controller saves and restores, and the parts that hold it."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
)

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
    DataPlaneCheckpointMetadata,
    TQReplayMetadataState,
)
from nemo_rl.algorithms.metric_utils import (
    RolloutCheckpointAttemptReason,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.environments.checkpointing.interfaces import (
    EnvironmentCheckpoint,
)
from nemo_rl.experience.reward_penalties import (
    FinalizedReward,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
    from nemo_rl.algorithms.async_utils.staleness_sampler import PromptGroupSampler
    from nemo_rl.experience.rollout_manager import RolloutManager


@dataclass(frozen=True)
class RolloutCheckpointSaveResult:
    """Outcome returned by one rollout checkpoint save attempt."""

    saved: bool
    reason: RolloutCheckpointAttemptReason


@dataclass
class RolloutRuntimeState:
    """Rollout state that the pumps change and every checkpoint saves.

    The controller owns the one instance; restore fills it before the pumps
    start, and checkpoints read it under the exclusive data-plane cut.
    """

    # Spare prompts that on_dropped_prompt="replace" substitutes for dropped
    # ones. Filled only by the rollout pump, which owns the dataloader iterator;
    # see the config docstring for why a dispatch task cannot pull from it.
    replacement_reserve: deque[DatumSpec] = field(default_factory=deque)
    # Prompts dropped from each stamped target step. A stamping sampler will
    # only ever select num_prompts_per_step groups carrying that stamp, so a
    # dropped prompt leaves that step permanently one short and the train pump
    # waits on a group no one is generating. The pump subtracts these to close
    # the step short. With an unstamped sampler the batch fills from whatever
    # is ready, so a drop costs throughput but strands nothing.
    batch_shortfall: dict[int, int] = field(default_factory=dict)
    # Whether the sampler has ever handed back a target step. Only stamped
    # prompts can strand a step, so this gates the spare pool: filling it for a
    # sampler that never stamps would divert prompts nothing can draw on.
    # Learned from admit rather than the sampler's type, because a custom
    # sampler's stamping is not knowable until it answers.
    sampler_stamps_target_steps: bool = False
    # Finalizer results kept per group until the group is trained or dropped.
    finalizer_metrics_by_group: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    finalizer_rewards_by_group: dict[str, list[FinalizedReward]] = field(
        default_factory=dict
    )


# Trainer checkpoints save model state and refuse rows a streamed step still
# owns; rollout snapshots reference a durable trainer checkpoint and re-index
# those rows. Both capture the rest of the rollout state the same way.
CheckpointKind = Literal["trainer_checkpoint", "rollout_snapshot"]


@dataclass(frozen=True)
class CapturedRolloutState:
    """Controller state captured under one exclusive cut, with its TQ save."""

    dataloader_state: dict[str, Any]
    sampler_dispatch_index: int
    next_nemo_gym_task_index: int
    replacement_reserve: list[DatumSpec]
    replay_metadata: Optional[TQReplayMetadataState]
    rollout_recovery_payload: Optional[bytes]
    rollout_recovery_group_count: Optional[int]
    replay_row_count: int
    staging_row_count: int
    rolled_back_train_group_count: int
    mutation_version: int
    tq_save_seconds: float


@dataclass(frozen=True)
class TrainingProgress:
    """The controller's training counters, read when a checkpoint needs them."""

    train_steps: int
    trainer_version: int
    current_epoch: int


@dataclass(frozen=True)
class ResumeSource:
    """The checkpoint setup restored from; the defaults describe a fresh run."""

    path: Optional[str] = None
    from_rollout_snapshot: bool = False
    data_plane_metadata: Optional[DataPlaneCheckpointMetadata] = None
    # Setup's load timings, logged with the restore timings.
    load_metrics: Optional[dict[str, float]] = None


@dataclass(frozen=True)
class RolloutParts:
    """The controller's parts that both saving and restoring rollout state use.

    Neither side changes which parts these are; ``runtime_state`` is the one
    mutable object, shared with the controller's pumps.
    """

    master_config: MasterConfig
    dp_client: Any
    partition_id: str
    buffer: TQReplayBuffer
    dataloader: Any
    sampler: PromptGroupSampler
    rollout_manager: RolloutManager
    data_plane_barrier: DataPlaneCheckpointBarrier
    environment_checkpoint: EnvironmentCheckpoint
    runtime_state: RolloutRuntimeState
    # Returns the controller's current training counters.
    read_progress: Callable[[], TrainingProgress]
    # The controller's telemetry logger.
    log_metrics: Callable[..., None]


class _UsesRolloutParts:
    """Unpack the shared parts into the attributes both sides read."""

    def __init__(self, parts: RolloutParts) -> None:
        self._master_config = parts.master_config
        self._async_cfg = parts.master_config.async_rl
        self._dp_client = parts.dp_client
        self._partition_id = parts.partition_id
        self._buffer = parts.buffer
        self._dataloader = parts.dataloader
        self._sampler = parts.sampler
        self._rollout_manager = parts.rollout_manager
        self._rollout_recovery_ledger = parts.rollout_manager.recovery_ledger
        self._data_plane_checkpoint_barrier = parts.data_plane_barrier
        self._environment_checkpoint = parts.environment_checkpoint
        self._rollout_runtime = parts.runtime_state
        self._read_progress = parts.read_progress
        self._log_metrics = parts.log_metrics
