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

"""NeMo Gym's side of turn-level rollout snapshots.

Implements ``EnvironmentCheckpoint`` (``nemo_rl.environments.checkpointing.interfaces``)
over a ``GymCheckpointCoordinator``. Everything that knows Gym episodes, Gym
participants, retire/forget, and ``gym_checkpoint.json`` lives here, so the
single controller only pauses, snapshots, validates and restores through the
interface.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.environments.checkpointing.interfaces import RestoredEnvironmentRollouts
from nemo_rl.environments.gym_checkpoint_adapter import GymCheckpointEpisode
from nemo_rl.environments.gym_checkpoint_coordinator import (
    GymCheckpointCommitResult,
    GymCheckpointCoordinator,
    GymCheckpointOperationError,
    is_transient_gym_checkpoint_failure,
    load_gym_checkpoint_manifest,
    staging_keys_of,
    write_gym_checkpoint_manifest,
)
from nemo_rl.environments.nemo_gym import as_nemo_gym_shard_set

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneMutationCut
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.experience.rollout_manager import RolloutManager
    from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger

# Rollout-deadline pause owned by a Gym checkpoint while it holds rollouts parked.
_GYM_CHECKPOINT_DEADLINE_HOLDER = "gym_checkpoint"

# Episodes named in a Gym checkpoint error; the rest are only counted.
_GYM_EPISODE_SAMPLE_SIZE = 5


def _summarize_gym_episodes(
    episodes_by_instance: Mapping[str, Iterable[Any]],
) -> str:
    """Count episodes per instance and name a few, however many there are."""
    parts = []
    for instance_id, episodes in sorted(episodes_by_instance.items()):
        ordered = sorted(episodes, key=repr)
        if not ordered:
            continue
        sample = ordered[:_GYM_EPISODE_SAMPLE_SIZE]
        more = len(ordered) - len(sample)
        parts.append(
            f"{instance_id}: {len(ordered)} episode(s), e.g. {sample!r}"
            + (f" and {more} more" if more else "")
        )
    return "; ".join(parts) or "none"


def build_gym_environment_checkpoint(
    master_config: MasterConfig,
    *,
    env_handles: dict[str, Any],
    rollout_manager: RolloutManager,
    dp_client: Any,
) -> GymEnvironmentCheckpoint:
    """Build the Gym checkpoint and close Gym dispatch through its admission gate."""
    staging_partition = master_config.token_capture.staging_partition

    async def list_staging_keys() -> list[str]:
        return await call_data_plane(
            dp_client,
            "list_sample_ids",
            offload_sync=True,
            partition_id=staging_partition,
        )

    environment_checkpoint = GymEnvironmentCheckpoint(
        GymCheckpointCoordinator(
            as_nemo_gym_shard_set(env_handles["nemo_gym"]),
            control_timeout_s=master_config.token_capture.control_timeout_s,
            prepare_timeout_s=master_config.rollout_recovery.gym_prepare_timeout_s,
        ),
        ledger=rollout_manager.recovery_ledger,
        rollout_manager=rollout_manager,
        list_staging_keys=list_staging_keys,
        allow_restart_only_servers=(
            master_config.rollout_recovery.allow_restart_only_gym_servers
        ),
    )
    rollout_manager.set_rollout_dispatch_admission_gate(
        environment_checkpoint.admission_gate
    )
    return environment_checkpoint


@dataclass
class GymEnvironmentSnapshot:
    """Gym's committed part of one rollout snapshot."""

    checkpoint: GymEnvironmentCheckpoint
    commit: GymCheckpointCommitResult
    checkpoint_path: Path

    @property
    def restarted_rollouts(self) -> int:
        return sum(len(episodes) for episodes in self.commit.manifest.restarts.values())

    async def check_against_cut(self, cut: DataPlaneMutationCut) -> None:
        await self.checkpoint.validate_cut(
            cut, self.commit, checkpoint_path=self.checkpoint_path
        )

    def metrics(self, paused_seconds: float) -> dict[str, float]:
        return {
            "gym_parked_seconds": paused_seconds,
            # Episodes Gym could not capture; they start over after a crash.
            "gym_restarted_episodes": float(self.restarted_rollouts),
        }


class GymEnvironmentCheckpoint:
    """Pause, commit, validate and restore Gym's episodes for RL's snapshots."""

    def __init__(
        self,
        coordinator: GymCheckpointCoordinator,
        *,
        ledger: RolloutRecoveryLedger,
        rollout_manager: Any,
        list_staging_keys: Callable[[], Awaitable[Iterable[str]]],
        allow_restart_only_servers: bool = False,
    ) -> None:
        """Wrap ``coordinator``.

        Args:
            coordinator: Talks to every Gym deployment's checkpoint participants.
            ledger: RL's record of which Gym episodes it owns.
            rollout_manager: Pauses rollout deadlines while Gym is parked.
            list_staging_keys: Returns the token-capture staging rows TQ holds,
                to check the ones the committed episodes reference still exist.
            allow_restart_only_servers: Accept Gym servers whose rollouts start
                over after a crash, with a warning, instead of refusing them.
        """
        self.coordinator = coordinator
        self._ledger = ledger
        self._rollout_manager = rollout_manager
        self._list_staging_keys = list_staging_keys
        self._allow_restart_only_servers = allow_restart_only_servers
        # Separate from the data-plane barrier: it pauses only Gym /run
        # submissions while a Gym checkpoint is prepared.
        self.admission_gate = DataPlaneCheckpointBarrier()

    @property
    def holds_rollout_state(self) -> bool:
        return True

    def is_unavailable(self, error: BaseException) -> bool:
        # Gym was resumed, or its lease reopens it; nothing RL holds changed.
        return is_transient_gym_checkpoint_failure(error)

    # ── ledger views ──────────────────────────────────────────────────────

    def inventory(self) -> dict[str, tuple[GymCheckpointEpisode, ...]]:
        """Return the Gym-owned episode attempts of every live deployment."""
        return self._ledger.gym_checkpoint_inventory(self.coordinator.instance_ids)

    def _queued_retirements(self) -> dict[str, frozenset[GymCheckpointEpisode]]:
        """Return dropped episodes Gym may still hold, awaiting their retire."""
        return {
            instance_id: frozenset(episodes)
            for instance_id, episodes in (
                self._ledger.gym_checkpoint_retirements(
                    self.coordinator.instance_ids
                ).items()
            )
        }

    # ── startup ───────────────────────────────────────────────────────────

    async def check_startup(self) -> None:
        """Refuse, or warn about, Gym servers whose rollouts start over after a crash.

        Such a server's in-flight rollouts are reported as restarts at every
        checkpoint, so turn-level recovery would quietly become
        restart-from-task for them.
        """
        allowed = self._allow_restart_only_servers
        restart_only = {
            instance_id: servers
            for instance_id, servers in (
                await self.coordinator.restart_only_servers()
            ).items()
            if servers
        }
        if restart_only and not allowed:
            raise ValueError(
                "rollout_recovery.target_level='turn' cannot continue rollouts on "
                "these restart-only Gym servers, which start every in-flight "
                f"rollout over after a crash: {restart_only!r}. Set "
                "rollout_recovery.allow_restart_only_gym_servers=true to accept "
                "that, or use target_level='sibling'."
            )
        if restart_only:
            warnings.warn(
                "rollout_recovery.allow_restart_only_gym_servers=true: in-flight "
                "rollouts on these restart-only Gym servers start over after a "
                f"crash: {restart_only!r}",
                stacklevel=2,
            )
        elif allowed:
            warnings.warn(
                "rollout_recovery.allow_restart_only_gym_servers=true, but no Gym "
                "server is restart-only; the setting has no effect",
                stacklevel=2,
            )

    # ── snapshot ──────────────────────────────────────────────────────────

    async def _retire_dropped_episodes(self, checkpoint_id: str) -> None:
        """Retire episodes RL dropped while Gym may still be running them.

        Abandoning or discarding a group leaves an admitted episode running in
        Gym, which would export it at commit, outside the candidate set. Gym
        accepts a retire only while idle, so this runs before prepare. A drop
        recorded after it stays queued: this checkpoint commits it in scope but
        leaves it out of the saved manifest, and a later retire stops it.
        """
        ledger = self._ledger
        retirements = ledger.gym_checkpoint_retirements(self.coordinator.instance_ids)
        if any(retirements.values()):
            await self.coordinator.retire(checkpoint_id, retirements)
            # Only what was retired: a drop recorded meanwhile stays for next time.
            ledger.mark_gym_retired(retirements)
        await self._forget_finished_rollouts(checkpoint_id)

    async def _forget_finished_rollouts(self, checkpoint_id: str) -> None:
        """Let Gym drop its marks for rollouts RL will never send again.

        Gym refuses a retired or restored rollout by ID until it is forgotten,
        so each finished one would otherwise stay on every participant for the
        rest of the run. Gym refuses a forget while a checkpoint is open, so this
        runs right after the retire. A deployment that refuses, for example
        because a retire has not finished freeing a rollout, gets the retire
        again and then the forget. Forget is best effort: a mark costs one small
        entry, so a deployment that still refuses keeps its rollouts queued for
        the next snapshot instead of failing this one.
        """
        coordinator = self.coordinator
        ledger = self._ledger
        forgets = ledger.gym_checkpoint_forgets(coordinator.instance_ids)
        if not any(forgets.values()):
            return
        try:
            await coordinator.forget(checkpoint_id, forgets)
        except GymCheckpointOperationError as error:
            refused = {
                instance_id: forgets[instance_id] for instance_id in error.failures
            }
            ledger.mark_gym_forgotten(
                {
                    instance_id: episodes
                    for instance_id, episodes in forgets.items()
                    if instance_id not in refused
                }
            )
            try:
                await coordinator.retire(
                    checkpoint_id,
                    {
                        instance_id: refused.get(instance_id, ())
                        for instance_id in coordinator.instance_ids
                    },
                )
                await coordinator.forget(checkpoint_id, refused)
            except GymCheckpointOperationError as retry_error:
                warnings.warn(
                    "Gym could not forget finished rollouts on "
                    f"{sorted(retry_error.failures)!r}; they stay queued for the next "
                    f"snapshot: {retry_error}",
                    stacklevel=2,
                )
                return
            ledger.mark_gym_forgotten(refused)
            return
        ledger.mark_gym_forgotten(forgets)

    async def _drain_non_exported_candidates(
        self,
        candidates: dict[str, tuple[GymCheckpointEpisode, ...]],
        commit: GymCheckpointCommitResult,
    ) -> None:
        """Wait for on-wire replies that Gym did not retain in its checkpoint.

        A restart is still running in Gym, parked until resume, so it is not
        waited for: the snapshot records it as starting over.
        """
        coordinator = self.coordinator
        pending = {
            instance_id: set(candidates[instance_id])
            - set(commit.manifest.instances[instance_id])
            - set(commit.manifest.restarts[instance_id])
            for instance_id in coordinator.instance_ids
        }
        if not any(pending.values()):
            return
        # Only an episode whose reply is already on the wire can resolve here. A
        # session Gym still holds is parked by prepare, so waiting would only time out.
        stranded = {
            instance_id: pending[instance_id].intersection(
                commit.instances[instance_id].unowned_live_episodes
            )
            for instance_id in coordinator.instance_ids
        }
        if any(stranded.values()):
            raise RuntimeError(
                "Gym checkpoint holds parked agent sessions that are not an "
                "agent's own /run episode, "
                "so they can neither be restored nor finish while the deployment is "
                f"prepared: stranded={_summarize_gym_episodes(stranded)}"
            )

        # Only the pending episodes are checked: rescanning every live rollout
        # on each poll would block the event loop that delivers the replies
        # this waits for.
        ledger = self._ledger
        waiting = pending
        deadline = time.monotonic() + coordinator.control_timeout_s
        while True:
            waiting = ledger.gym_episodes_still_held(waiting)
            if not any(waiting.values()):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Gym checkpoint could not classify every candidate before the "
                    "drain deadline: "
                    f"unresolved={_summarize_gym_episodes(waiting)}"
                )
            await asyncio.sleep(min(0.05, remaining))

    @contextlib.asynccontextmanager
    async def snapshot(
        self,
        checkpoint_path: Path,
        *,
        checkpoint_id: Optional[str] = None,
    ) -> AsyncIterator[GymEnvironmentSnapshot]:
        """Close Gym admission, commit owned episodes, and drain on-wire replies."""
        checkpoint_id = (
            checkpoint_id
            if checkpoint_id is not None
            else f"rollout-{uuid.uuid4().hex}"
        )
        coordinator = self.coordinator
        async with self.admission_gate.checkpoint():
            # Gym parks every live rollout until resume, so the wait must not
            # count against their deadlines. A separate holder keeps a
            # colocated engine's own pause intact when this one ends.
            self._rollout_manager.suspend_request_deadlines(
                _GYM_CHECKPOINT_DEADLINE_HOLDER
            )
            try:
                # Gym refuses a retire once a checkpoint is open, so retire while
                # it is still idle; dispatch is already closed, so nothing new
                # races in.
                await self._retire_dropped_episodes(checkpoint_id)
                async with coordinator.prepared(checkpoint_id) as restarts:
                    candidates = self.inventory()
                    # A row dropped since that retire is still running in Gym,
                    # which exports it; naming it keeps the export in scope, and
                    # the cut leaves it out of the saved manifest.
                    dropped = self._queued_retirements()
                    # Gym refuses a commit that names an episode it cannot
                    # capture; those start over after a crash instead.
                    scope = {
                        instance_id: tuple(
                            episode
                            for episode in episodes
                            + tuple(sorted(dropped[instance_id] - set(episodes)))
                            if episode not in restarts[instance_id]
                        )
                        for instance_id, episodes in candidates.items()
                    }
                    commit = await coordinator.commit(
                        checkpoint_id,
                        Path(checkpoint_path),
                        scope,
                        restarts={
                            instance_id: tuple(
                                episode
                                for episode in episodes
                                if episode in restarts[instance_id]
                            )
                            for instance_id, episodes in candidates.items()
                        },
                    )
                    await self._drain_non_exported_candidates(candidates, commit)
                    yield GymEnvironmentSnapshot(
                        checkpoint=self,
                        commit=commit,
                        checkpoint_path=Path(checkpoint_path),
                    )
            finally:
                self._rollout_manager.resume_request_deadlines(
                    _GYM_CHECKPOINT_DEADLINE_HOLDER
                )

    async def validate_cut(
        self,
        cut: DataPlaneMutationCut,
        commit: GymCheckpointCommitResult,
        *,
        checkpoint_path: Path,
    ) -> None:
        """Require Gym ownership and referenced staging rows to match this cut.

        Gym may have exported episodes RL dropped while it was prepared: it
        refuses their retire until resume, so they are still queued for one.
        They and their staged rows are left out of the saved manifest, since
        restore requires it to match RL's inventory and RL no longer continues
        them; a dropped group may already have cleaned those rows up.
        """
        cut.require_live()
        coordinator = self.coordinator
        actual = self.inventory()
        expected = commit.manifest.instances
        dropped = self._queued_retirements()
        mismatches = {}
        kept: dict[str, tuple[GymCheckpointEpisode, ...]] = {}
        kept_restarts: dict[str, tuple[GymCheckpointEpisode, ...]] = {}
        for instance_id in coordinator.instance_ids:
            owned = set(actual[instance_id])
            exported = set(expected[instance_id])
            restarted = set(commit.manifest.restarts[instance_id])
            if (
                owned - exported - restarted
                or not exported - owned <= dropped[instance_id]
            ):
                mismatches[instance_id] = {
                    "expected": sorted(exported, key=repr),
                    "restarts": sorted(restarted, key=repr),
                    "actual": sorted(owned, key=repr),
                }
            kept[instance_id] = tuple(
                episode for episode in expected[instance_id] if episode in owned
            )
            # A restart that finished during the snapshot is no longer RL's.
            kept_restarts[instance_id] = tuple(
                episode
                for episode in commit.manifest.restarts[instance_id]
                if episode in owned
            )
        if mismatches:
            raise RuntimeError(
                "Gym episode ownership changed after commit and before the TQ cut: "
                f"{mismatches!r}"
            )
        staging_keys = {
            instance_id: staging_keys_of(
                commit.instances[instance_id], kept[instance_id]
            )
            for instance_id in coordinator.instance_ids
        }
        if kept != expected or kept_restarts != commit.manifest.restarts:
            await asyncio.to_thread(
                write_gym_checkpoint_manifest,
                Path(checkpoint_path),
                replace(
                    commit.manifest,
                    instances=kept,
                    staging_keys=staging_keys,
                    restarts=kept_restarts,
                ),
            )

        expected_staging_keys = {key for keys in staging_keys.values() for key in keys}
        if not expected_staging_keys:
            return
        actual_staging_keys = set(await self._list_staging_keys())
        missing = sorted(expected_staging_keys - actual_staging_keys)
        if missing:
            raise RuntimeError(
                "Gym checkpoint references staging rows missing from live TQ state: "
                f"missing={missing[:10]!r} (total={len(missing)})"
            )

    # ── restore ───────────────────────────────────────────────────────────

    @contextlib.asynccontextmanager
    async def restored(
        self, checkpoint_path: Path
    ) -> AsyncIterator[RestoredEnvironmentRollouts]:
        """Restore Gym's saved episodes; on failure retire them before Gym reopens."""
        coordinator = self.coordinator
        manifest = await asyncio.to_thread(
            load_gym_checkpoint_manifest, checkpoint_path
        )
        expected_inventory = {
            instance_id: set(episodes)
            for instance_id, episodes in self.inventory().items()
        }
        # Restarts are not restored: prepare_for_restart starts every
        # episode Gym did not restore over from its task.
        saved_inventory = {
            instance_id: set(episodes) | set(manifest.restarts.get(instance_id, ()))
            for instance_id, episodes in manifest.instances.items()
        }
        if saved_inventory != expected_inventory:
            differing = sorted(
                instance_id
                for instance_id in saved_inventory.keys() | expected_inventory.keys()
                if saved_inventory.get(instance_id)
                != expected_inventory.get(instance_id)
            )
            raise ValueError(
                "Gym checkpoint manifest does not match the rollout "
                f"recovery ledger for Gym instances {differing!r}"
            )
        async with coordinator.restored(
            f"restore-{uuid.uuid4().hex}",
            checkpoint_path,
            manifest,
        ):
            yield RestoredEnvironmentRollouts(
                continued_attempts=frozenset(
                    (instance_id, episode.rollout_id, episode.attempt)
                    for instance_id, episodes in manifest.instances.items()
                    for episode in episodes
                ),
                staging_keys=frozenset(
                    key for keys in manifest.staging_keys.values() for key in keys
                ),
            )
