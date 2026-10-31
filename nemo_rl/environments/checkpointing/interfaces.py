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

"""Save and restore rollout state that lives in the environment backend.

At turn-level recovery the environment backend holds part of every unfinished
rollout: agent sessions, tool state, the turns already run. The single
controller saves and restores that state only through ``EnvironmentCheckpoint``
and never names the backend. NeMo Gym implements it in
``nemo_rl.environments.checkpointing.nemo_gym``; a backend that keeps no
rollout state uses ``NoEnvironmentCheckpoint``.

A rollout snapshot uses the interface in this order::

    async with env_checkpoint.snapshot(path) as snapshot:  # backend paused
        async with data_plane_barrier.checkpoint() as cut:
            if snapshot is not None:
                await snapshot.check_against_cut(cut)
            ...  # RL saves TQ and its own state
    # backend resumed
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneMutationCut


@dataclass(frozen=True)
class RestoredEnvironmentRollouts:
    """What an environment restore continued, so RL can match it.

    Attributes:
        continued_attempts: ``(instance_id, rollout_id, attempt)`` of every
            rollout the backend restored. The recovery ledger continues exactly
            these and starts every other unfinished rollout over.
        staging_keys: Token-capture staging rows the restored rollouts still
            reference. Restore keeps them instead of clearing them as orphans.
    """

    continued_attempts: frozenset[tuple[str, str, int]] = frozenset()
    staging_keys: frozenset[str] = frozenset()


class EnvironmentSnapshot(Protocol):
    """The backend's part of one rollout snapshot, while the backend is paused."""

    @property
    def restarted_rollouts(self) -> int:
        """Rollouts the backend could not capture; they start over after a crash."""
        ...

    async def check_against_cut(self, cut: DataPlaneMutationCut) -> None:
        """Under the data-plane cut, require the captured state to match RL's.

        Raises when the backend's rollouts and RL's ledger disagree, which
        would make the snapshot unrestorable.
        """
        ...

    def metrics(self, paused_seconds: float) -> dict[str, float]:
        """Backend metrics for this snapshot, logged with the snapshot timings."""
        ...


class EnvironmentCheckpoint(Protocol):
    """Rollout state an environment backend keeps between turns."""

    @property
    def holds_rollout_state(self) -> bool:
        """Whether rollout snapshots must capture this backend.

        True means a trainer checkpoint, which never pauses the backend, does
        not stand in for a rollout snapshot.
        """
        ...

    async def check_startup(self) -> None:
        """Refuse, or warn about, a deployment that cannot recover as configured."""
        ...

    def snapshot(
        self, checkpoint_path: Path
    ) -> contextlib.AbstractAsyncContextManager[Optional[EnvironmentSnapshot]]:
        """Pause the backend and save its rollouts under ``checkpoint_path``.

        Yields ``None`` when there is nothing to save. The backend resumes,
        and accepts new rollouts again, when the context exits.
        """
        ...

    def restored(
        self, checkpoint_path: Path
    ) -> contextlib.AbstractAsyncContextManager[RestoredEnvironmentRollouts]:
        """Restore the backend's rollouts from ``checkpoint_path``.

        The restore is kept only if the context exits cleanly. On an exception
        the backend drops what it restored before it resumes.
        """
        ...

    def is_unavailable(self, error: BaseException) -> bool:
        """Whether a failed snapshot only met a backend that was briefly unreachable.

        Such a failure leaves nothing RL holds changed, so the next attempt
        can retry.
        """
        ...


class NoEnvironmentCheckpoint:
    """For a backend that keeps no rollout state between turns."""

    @property
    def holds_rollout_state(self) -> bool:
        return False

    async def check_startup(self) -> None:
        return None

    @contextlib.asynccontextmanager
    async def snapshot(
        self, checkpoint_path: Path
    ) -> AsyncIterator[Optional[EnvironmentSnapshot]]:
        del checkpoint_path
        yield None

    @contextlib.asynccontextmanager
    async def restored(
        self, checkpoint_path: Path
    ) -> AsyncIterator[RestoredEnvironmentRollouts]:
        raise RuntimeError(
            f"rollout checkpoint {checkpoint_path} contains turn-level environment "
            "state, but this run's environment backend keeps none to restore"
        )
        yield RestoredEnvironmentRollouts()  # pragma: no cover

    def is_unavailable(self, error: BaseException) -> bool:
        del error
        return False
