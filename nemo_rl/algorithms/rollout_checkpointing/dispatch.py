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

"""How the rollout pump reserves, admits and gives up prompt groups.

The rollout pump runs one dispatch loop. Whether that loop records each
prompt group in the recovery ledger is decided here, not in the pump:
``PromptDispatch.reservation`` hands out a ``LedgerReservation`` when it does
and an ``UnrecordedReservation`` when it does not.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Optional

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    TransactionalAdmissionSampler,
)
from nemo_rl.algorithms.rollout_checkpointing.interface import PromptReservation

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import (
        CheckpointMutationKind,
        DataPlaneCheckpointBarrier,
        DataPlaneMutationCut,
    )
    from nemo_rl.algorithms.async_utils.staleness_sampler import PromptGroupSampler
    from nemo_rl.algorithms.rollout_checkpointing.restore import RestoreReport
    from nemo_rl.data.interfaces import DatumSpec
    from nemo_rl.experience.rollout_manager import RolloutManager


class LedgerReservation(PromptReservation):
    """Record every change in the recovery ledger, under one mutation cut."""

    def __init__(
        self, cut: DataPlaneMutationCut, rollout_manager: RolloutManager
    ) -> None:
        """Store the live cut and the rollout manager that owns the ledger."""
        self._cut = cut
        self._rollout_manager = rollout_manager

    def reserve_prompt_group(
        self,
        prompt: DatumSpec,
        *,
        target_step: Optional[int],
        admitted: bool = True,
        admission_id: Optional[str] = None,
    ) -> Optional[str]:
        return self._rollout_manager.reserve_prompt_group(
            self._cut,
            prompt,
            target_step=target_step,
            admitted=admitted,
            admission_id=admission_id,
        )

    async def admit(
        self,
        sampler: PromptGroupSampler,
        *,
        trainer_version_fn: Callable[[], int],
    ) -> Optional[int]:
        if isinstance(sampler, TransactionalAdmissionSampler):
            return sampler.commit_admission(self._cut)
        # Custom samplers retain their existing monolithic admission API. Hold
        # the mutation cut across it for correctness. Contract: a custom admit()
        # must not wait on anything beyond a single trainer_version increment --
        # a checkpoint drains mutation slots while blocking the train pump, so a
        # longer wait deadlocks the run. Implement TransactionalAdmissionSampler
        # to keep the gate wait outside the mutation cut entirely.
        return await sampler.admit(trainer_version_fn=trainer_version_fn)

    def mark_prompt_group_admitted(
        self, group_id: Optional[str], *, target_step: Optional[int]
    ) -> None:
        assert group_id is not None, "a ledger reservation always has a group ID"
        self._rollout_manager.mark_prompt_group_admitted(
            self._cut,
            group_id,
            target_step=target_step,
        )

    def discard_prompt_group(self, group_id: Optional[str]) -> None:
        assert group_id is not None, "a ledger reservation always has a group ID"
        self._rollout_manager.discard_prompt_group(self._cut, group_id)

    async def discard_recovery_group(self, group_id: str) -> None:
        await self._rollout_manager.discard_recovery_group(self._cut, group_id)


class UnrecordedReservation(PromptReservation):
    """Record nothing. The rollout manager mints each group ID when it runs it."""

    def reserve_prompt_group(
        self,
        prompt: DatumSpec,
        *,
        target_step: Optional[int],
        admitted: bool = True,
        admission_id: Optional[str] = None,
    ) -> Optional[str]:
        del prompt, target_step, admitted, admission_id
        return None

    async def admit(
        self,
        sampler: PromptGroupSampler,
        *,
        trainer_version_fn: Callable[[], int],
    ) -> Optional[int]:
        return await sampler.admit(trainer_version_fn=trainer_version_fn)

    def mark_prompt_group_admitted(
        self, group_id: Optional[str], *, target_step: Optional[int]
    ) -> None:
        del group_id, target_step

    def discard_prompt_group(self, group_id: Optional[str]) -> None:
        del group_id

    async def discard_recovery_group(self, group_id: str) -> None:
        del group_id


class PromptDispatch:
    """Hand the rollout pump a ledger or an unrecorded reservation.

    The choice is not tied to which ``RolloutCheckpointing`` was built. A run
    that saves no rollout state still dispatches through the ledger when the
    checkpoint it resumed from holds unfinished groups.
    """

    def __init__(
        self,
        *,
        data_plane_barrier: DataPlaneCheckpointBarrier,
        rollout_manager: RolloutManager,
        records_prompt_groups: bool,
    ) -> None:
        """Store the barrier and ledger owner, and whether to record from the start.

        Args:
            data_plane_barrier: The barrier every TQ mutation goes through.
            rollout_manager: Owns the recovery ledger.
            records_prompt_groups: Whether dispatch records groups in the
                ledger from the start. Restore may turn it on later.
        """
        self._data_plane_checkpoint_barrier = data_plane_barrier
        self._rollout_manager = rollout_manager
        self._records_prompt_groups = records_prompt_groups
        self._unrecorded = UnrecordedReservation()

    @property
    def records_prompt_groups(self) -> bool:
        """Whether dispatch records each prompt group in the recovery ledger."""
        return self._records_prompt_groups

    def apply_restore_report(self, report: RestoreReport) -> RestoreReport:
        """Turn ledger dispatch on when restore queued groups only it can launch."""
        if report.requires_recovery_dispatch:
            # Restored ownership must be dispatched even when future checkpoint
            # saving is disabled and the native snapshot predates recovery sidecars.
            self._records_prompt_groups = True
        return report

    @contextlib.asynccontextmanager
    async def reservation(
        self, kind: CheckpointMutationKind
    ) -> AsyncIterator[PromptReservation]:
        """Yield a reservation for one group of dispatch changes.

        Args:
            kind: The mutation kind the ledger reservation's cut is counted as.
        """
        if not self._records_prompt_groups:
            # Nothing is recorded, so a checkpoint has nothing here to see
            # whole. Dispatch without the ledger never took a mutation slot,
            # and a run without recovery must not start paying for one.
            yield self._unrecorded
            return
        async with self._data_plane_checkpoint_barrier.mutation(kind) as cut:
            yield LedgerReservation(cut, self._rollout_manager)
