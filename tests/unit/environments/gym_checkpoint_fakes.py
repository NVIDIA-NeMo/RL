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

"""A fake NeMo-Gym actor to drive the real ``GymCheckpointCoordinator`` in tests,
and builders for the real ``GymEnvironmentCheckpoint`` over it.

Tests set the per-test Gym behavior (a commit that blocks, a restore that fails,
a prepare that is not ready, sessions Gym keeps until retired) on the actor, so
the coordinator under test is always the real one.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from nemo_rl.environments.checkpointing.nemo_gym import GymEnvironmentCheckpoint
from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointCommitSummary,
    GymCheckpointEpisode,
    GymCheckpointPrepareSummary,
)
from nemo_rl.environments.gym_checkpoint_coordinator import (
    GymCheckpointCommitResult,
    GymCheckpointCoordinator,
)
from nemo_rl.environments.nemo_gym import NemoGymShardSet
from nemo_rl.experience.rollout_manager import RequestDeadlineRegistry
from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger

INSTANCE_ID = "tools/replica-0"


class _RemoteMethod:
    def __init__(self, function) -> None:
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class FakeGymActor:
    """One NeMo-Gym actor's checkpoint control calls, recorded in ``calls``.

    Args:
        prepared: Whether a prepare parks every participant.
        restore_error: Raised by restore.
        exported_episodes: What a commit exports; by default the episodes in
            scope, or ``live`` when that is set.
        staging_keys_by_episode: The staged rows a commit attributes.
        hang: Operations accepted but never answered, like a stuck actor.
        resume_errors: Raised by the first resume calls, one per call.
        live: Sessions Gym keeps until RL retires them. A commit exports all of
            them and a retire refuses while a checkpoint is open, as Gym does.
        unowned_live_episodes: Parked sessions no participant owns.
        block_commit: Hold each commit until ``release_commit`` is set;
            ``commit_started`` is set when one arrives.
        while_prepared: Awaited once a prepare has parked Gym.
        restarts: Episodes a prepare reports as restarts. A commit that names
            one is refused, as Gym does, and a commit never exports one.
        forget_errors: Raised by the first forget calls, one per call.
        restart_only_servers: Participants whose status says their rollouts
            start over after a crash.
        participants: The server names of the participants a commit reports.
    """

    def __init__(
        self,
        *,
        prepared: bool = True,
        restore_error: BaseException | None = None,
        exported_episodes: tuple[GymCheckpointEpisode, ...] | None = None,
        staging_keys_by_episode: dict[GymCheckpointEpisode, tuple[str, ...]]
        | None = None,
        hang: frozenset[str] = frozenset(),
        resume_errors: tuple[BaseException, ...] = (),
        live: Optional[set[GymCheckpointEpisode]] = None,
        unowned_live_episodes: tuple[GymCheckpointEpisode, ...] = (),
        block_commit: bool = False,
        while_prepared: Optional[Callable[[], Awaitable[Any]]] = None,
        restarts: tuple[GymCheckpointEpisode, ...] = (),
        forget_errors: tuple[BaseException, ...] = (),
        restart_only_servers: tuple[str, ...] = (),
        participants: tuple[str, ...] = ("tools-agent", "tools-model"),
    ) -> None:
        self.prepared = prepared
        self.resume_errors = list(resume_errors)
        self.hang = hang
        self.restore_error = restore_error
        self.exported_episodes = exported_episodes
        self.staging_keys_by_episode = staging_keys_by_episode or {}
        self.live = None if live is None else set(live)
        self.unowned_live_episodes = unowned_live_episodes
        self.commit_started = asyncio.Event() if block_commit else None
        self.release_commit = asyncio.Event() if block_commit else None
        self.while_prepared = while_prepared
        self.restarts = restarts
        self.forget_errors = list(forget_errors)
        self.restart_only_servers = list(restart_only_servers)
        self.participants = participants
        # Between a prepare that parked Gym and the resume that reopens it.
        self.checkpoint_open = False
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.checkpoint_prepare = _RemoteMethod(self._prepare)
        self.checkpoint_renew = _RemoteMethod(self._renew)
        self.checkpoint_retire = _RemoteMethod(self._retire)
        self.checkpoint_commit = _RemoteMethod(self._commit)
        self.checkpoint_restore = _RemoteMethod(self._restore)
        self.checkpoint_resume = _RemoteMethod(self._resume)
        self.checkpoint_forget = _RemoteMethod(self._forget)
        self.checkpoint_restart_only_servers = _RemoteMethod(self._restart_only_servers)

    def operations(self) -> list[str]:
        """The control operations received, in order."""
        return [call[0] for call in self.calls]

    def mark(self, label: str) -> None:
        """Record a test event in ``calls`` to check its order against Gym's."""
        self.calls.append((label, (), {}))

    async def _record(self, operation: str, args, kwargs) -> None:
        self.calls.append((operation, args, kwargs))
        if operation in self.hang:
            await asyncio.Event().wait()

    async def _prepare(self, *args, **kwargs) -> GymCheckpointPrepareSummary:
        await self._record("prepare", args, kwargs)
        if self.prepared:
            self.checkpoint_open = True
            if self.while_prepared is not None:
                await self.while_prepared()
        return GymCheckpointPrepareSummary(
            prepared=self.prepared,
            blockers={} if self.prepared else {"agent": ("rollout-1",)},
            restarts=self.restarts,
        )

    async def _renew(self, *args, **kwargs) -> None:
        await self._record("renew", args, kwargs)

    async def _restart_only_servers(self) -> list[str]:
        await self._record("status", (), {})
        return self.restart_only_servers

    async def _commit(self, *args, **kwargs) -> GymCheckpointCommitSummary:
        await self._record("commit", args, kwargs)
        restarted = set(args[2]) & set(self.restarts)
        if restarted:
            raise RuntimeError(f"restart_in_scope: {sorted(restarted)!r}")
        if self.commit_started is not None and self.release_commit is not None:
            self.commit_started.set()
            await self.release_commit.wait()
        if self.live is not None:
            episodes = tuple(sorted(self.live - set(self.restarts)))
        elif self.exported_episodes is not None:
            episodes = self.exported_episodes
        else:
            episodes = args[2]
        return GymCheckpointCommitSummary(
            exported_episodes=episodes,
            participants=self.participants,
            staging_keys=tuple(
                sorted(
                    {k for keys in self.staging_keys_by_episode.values() for k in keys}
                )
            ),
            staging_keys_by_episode=self.staging_keys_by_episode,
            unowned_live_episodes=self.unowned_live_episodes,
        )

    async def _retire(self, *args, **kwargs) -> None:
        await self._record("retire", args, kwargs)
        if self.live is None:
            return
        if self.checkpoint_open:
            # Gym stops a retired episode by waiting for its final cleanup,
            # whose calls wait for resume, so it refuses a retire mid-checkpoint.
            raise RuntimeError("cannot retire while a checkpoint is open; resume first")
        self.live.difference_update(args[1])

    async def _forget(self, *args, **kwargs) -> None:
        await self._record("forget", args, kwargs)
        if self.forget_errors:
            raise self.forget_errors.pop(0)

    async def _restore(self, *args, **kwargs) -> None:
        await self._record("restore", args, kwargs)
        if self.restore_error is not None:
            raise self.restore_error

    async def _resume(self, *args, **kwargs) -> None:
        await self._record("resume", args, kwargs)
        if self.resume_errors:
            raise self.resume_errors.pop(0)
        self.checkpoint_open = False


def gym_coordinator(
    *actors: FakeGymActor,
    control_timeout_s: float = 30.0,
    prepare_timeout_s: float = 30.0,
) -> GymCheckpointCoordinator:
    """The real coordinator over fake actors of one shard, ``tools``."""
    return GymCheckpointCoordinator(
        NemoGymShardSet(handles={"tools": list(actors)}),
        control_timeout_s=control_timeout_s,
        prepare_timeout_s=prepare_timeout_s,
    )


async def no_staging_keys() -> list[str]:
    return []


def deadline_pausing_manager(
    registry: Optional[RequestDeadlineRegistry] = None,
) -> SimpleNamespace:
    """Rollout-manager stand-in for the deadline pause a Gym checkpoint takes."""
    registry = registry if registry is not None else RequestDeadlineRegistry()
    return SimpleNamespace(
        suspend_request_deadlines=registry.suspend,
        resume_request_deadlines=registry.resume,
    )


def gym_environment_checkpoint(
    coordinator: GymCheckpointCoordinator,
    ledger: Optional[RolloutRecoveryLedger] = None,
    *,
    manager: Optional[Any] = None,
    list_staging_keys: Callable[[], Awaitable[list[str]]] = no_staging_keys,
    allow_restart_only_servers: bool = False,
) -> GymEnvironmentCheckpoint:
    """Gym's checkpoint over ``coordinator``, with a stand-in rollout manager."""
    return GymEnvironmentCheckpoint(
        coordinator,
        ledger=ledger if ledger is not None else RolloutRecoveryLedger(),
        rollout_manager=manager if manager is not None else deadline_pausing_manager(),
        list_staging_keys=list_staging_keys,
        allow_restart_only_servers=allow_restart_only_servers,
    )


@contextlib.asynccontextmanager
async def snapshot_commit(
    gym: GymEnvironmentCheckpoint, checkpoint_path: Path, *, checkpoint_id: str
) -> AsyncIterator[GymCheckpointCommitResult]:
    """Gym's commit for one snapshot, while Gym is parked."""
    async with gym.snapshot(checkpoint_path, checkpoint_id=checkpoint_id) as snapshot:
        yield snapshot.commit
