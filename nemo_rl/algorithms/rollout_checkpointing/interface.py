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

"""What the single controller calls to checkpoint rollouts, and what it provides.

The controller runs a train pump and a rollout pump. Everything that saves
rollout state sits behind ``RolloutCheckpointing``: periodic rollout snapshots,
the TQ part of a trainer checkpoint, the lock that keeps the two from publishing
at once, and the window in which a train step's optimizer update has landed but
its counters have not. The controller never names the environment backend; the
component reaches it through ``EnvironmentCheckpoint``.

The component never reads controller fields. It asks through
``RolloutCheckpointingHost``, and the controller answers with a
``ControllerRolloutState`` built under the data-plane cut.

A run that saves no rollout state gets ``NoRolloutCheckpointing``.
"""

from __future__ import annotations

import abc
import contextlib
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Any

from nemo_rl.algorithms.rollout_checkpointing.controller_state import (
    ControllerRolloutState,
    ControllerSidecars,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneMutationCut
    from nemo_rl.utils.checkpoint import PathLike


class RolloutCheckpointingHost(abc.ABC):
    """What the rollout checkpointing component needs from the controller."""

    @property
    @abc.abstractmethod
    def train_steps(self) -> int:
        """Completed train steps."""
        ...

    @property
    @abc.abstractmethod
    def trainer_version(self) -> int:
        """The policy version the trainer has published."""
        ...

    @property
    @abc.abstractmethod
    def current_epoch(self) -> int:
        """The rollout pump's dataloader epoch."""
        ...

    @abc.abstractmethod
    def rollout_state(self) -> ControllerRolloutState:
        """Return the controller's state to save. Call it under the data-plane cut."""
        ...

    @abc.abstractmethod
    def log_telemetry_metrics(
        self, metrics: dict[str, float], *, step: int, prefix: str
    ) -> None:
        """Log metrics on the telemetry axis, which does not follow train steps."""
        ...


class RolloutCheckpointing(abc.ABC):
    """The rollout checkpointing API the single controller calls.

    An implementation subclasses this and implements every member.
    """

    @abc.abstractmethod
    def background_tasks(self) -> list[Coroutine[Any, Any, None]]:
        """Coroutines the controller runs beside its pumps, such as the snapshot pump.

        A task that returns normally has asked the run to stop; see
        ``stop_requested``. A task that raises fails the run.
        """
        ...

    @property
    @abc.abstractmethod
    def stop_requested(self) -> bool:
        """Whether a background task stopped the run on purpose.

        Set after the checkpoint deadline passed before the first train step and
        a rollout snapshot made the run's state durable.
        """
        ...

    @abc.abstractmethod
    def optimizer_commit(
        self, *, drain_snapshot: bool
    ) -> contextlib.AbstractAsyncContextManager[None]:
        """Mark a train step's optimizer update as landed until its counters publish.

        Rollout snapshots skip this window, and one already pausing the
        environment stops. Enter it once the update is applied or about to be,
        and exit it once the step's TQ cleanup and trainer counters are
        published. A step may enter it more than once.

        Args:
            drain_snapshot: First wait for a snapshot that is already saving the
                old trainer identity, as a step that publishes a new trainer
                version before its cleanup must.
        """
        ...

    @abc.abstractmethod
    def trainer_checkpoint(self) -> contextlib.AbstractAsyncContextManager[None]:
        """Hold off rollout snapshots for the whole of one trainer checkpoint.

        Trainer checkpoints and rollout snapshots share one namespace and must
        never publish at the same time.
        """
        ...

    @abc.abstractmethod
    async def save_with_trainer_checkpoint(
        self,
        cut: DataPlaneMutationCut,
        checkpoint_path: PathLike,
        state: ControllerRolloutState,
    ) -> ControllerSidecars:
        """Save the rollout state that belongs in a trainer checkpoint.

        Called under the exclusive data-plane cut, with ``state`` built under
        the same cut. Saves TQ now and returns the controller files to write
        with ``write_controller_sidecars`` before the checkpoint is finalized.
        """
        ...


class NoRolloutCheckpointing(RolloutCheckpointing):
    """For a run that saves no rollout state.

    No snapshot pump, no lock and no commit window. A trainer checkpoint still
    gets the dataloader state and spare prompts every checkpoint needs, and
    nothing from the data plane.
    """

    def background_tasks(self) -> list[Coroutine[Any, Any, None]]:
        return []

    @property
    def stop_requested(self) -> bool:
        return False

    @contextlib.asynccontextmanager
    async def optimizer_commit(self, *, drain_snapshot: bool) -> AsyncIterator[None]:
        del drain_snapshot
        yield

    @contextlib.asynccontextmanager
    async def trainer_checkpoint(self) -> AsyncIterator[None]:
        yield

    async def save_with_trainer_checkpoint(
        self,
        cut: DataPlaneMutationCut,
        checkpoint_path: PathLike,
        state: ControllerRolloutState,
    ) -> ControllerSidecars:
        del cut, checkpoint_path
        return ControllerSidecars(
            dataloader_state=state.dataloader_state,
            replacement_reserve=state.replacement_reserve,
        )
