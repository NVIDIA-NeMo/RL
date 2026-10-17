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

"""NeMo RL coordination across independent Gym checkpoint deployments."""

from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ray

from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointCommitSummary,
    GymCheckpointEpisode,
    GymCheckpointPrepareSummary,
    GymCheckpointUnavailable,
)
from nemo_rl.environments.nemo_gym import NemoGymShardSet
from nemo_rl.experience.failures import FailureClass, classify_rollout_failure

GYM_CHECKPOINT_MANIFEST_FILENAME = "gym_checkpoint.json"
# 2: records the participants that committed each instance.
GYM_CHECKPOINT_SCHEMA_VERSION = 2
# Upper bound on how long past a call's deadline RL waits for Gym's reply. Gym
# answers by the deadline once it runs the call, so this only covers the reply
# crossing Ray; a call still unanswered after it was never run.
_MAX_REPLY_GRACE_S = 5.0
# A lost resume leaves its deployment closed until the checkpoint lease expires,
# so a transient failure is retried once. Gym answers a repeated resume
# idempotently.
_RESUME_ATTEMPTS = 2
_RESUME_RETRY_PAUSE_S = 1.0


class GymCheckpointOperationError(RuntimeError):
    """One or more independent Gym deployments failed one operation."""

    def __init__(self, operation: str, failures: Mapping[str, BaseException]) -> None:
        self.operation = operation
        self.failures = dict(failures)
        detail = "; ".join(
            f"{instance}: {type(error).__name__}: {error}"
            for instance, error in sorted(self.failures.items())
        )
        super().__init__(
            f"Gym checkpoint {operation} failed on "
            f"{len(self.failures)} instance(s): {detail}"
        )

    def __reduce__(self):
        # Rebuild from the constructor arguments, not the formatted message, and
        # keep attributes such as notes added after construction.
        return (type(self), (self.operation, self.failures), self.__dict__)


class GymCheckpointNotReady(RuntimeError):
    """Gym participants could not all park before the checkpoint deadline."""

    def __init__(
        self,
        blockers: Mapping[str, Mapping[str, tuple[str, ...]]],
    ) -> None:
        self.blockers = {
            instance: dict(instance_blockers)
            for instance, instance_blockers in blockers.items()
        }
        super().__init__(
            f"Gym checkpoint participants are not ready: {self.blockers!r}"
        )

    def __reduce__(self):
        return (type(self), (self.blockers,), self.__dict__)


def is_transient_gym_checkpoint_failure(error: BaseException) -> bool:
    """Whether a failed Gym checkpoint may succeed if it is simply tried again.

    Participants that could not park in time, calls that went unanswered, and
    actors or control planes that failed a call leave nothing behind: the
    coordinator resumes Gym, and Gym's lease reopens it if that resume is lost
    too. Anything else, such as a malformed reply or an ownership or topology
    mismatch, is a broken invariant that a retry would only hide.
    """
    if isinstance(error, GymCheckpointNotReady):
        return True
    if isinstance(error, GymCheckpointOperationError):
        return bool(error.failures) and all(
            isinstance(failure, GymCheckpointUnavailable)
            or classify_rollout_failure(failure) is FailureClass.INFRA
            for failure in error.failures.values()
        )
    return False


@dataclass(frozen=True)
class GymCheckpointManifest:
    """Durable mapping from Gym actor instances to continued episodes.

    ``restarts`` are the episodes RL still owned that Gym could not capture: a
    restore leaves them out and starts them over from their tasks.
    """

    schema_version: int
    checkpoint_id: str
    instances: dict[str, tuple[GymCheckpointEpisode, ...]]
    staging_keys: dict[str, tuple[str, ...]]
    restarts: dict[str, tuple[GymCheckpointEpisode, ...]]
    # The server names of the Gym participants that committed each instance.
    participants: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if not self.checkpoint_id:
            raise ValueError("Gym checkpoint_id must not be empty")
        if not self.instances:
            raise ValueError(
                "Gym checkpoint manifest must contain at least one instance"
            )
        for instance_id, episodes in self.instances.items():
            if not instance_id:
                raise ValueError("Gym checkpoint instance ID must not be empty")
            episode_keys = [
                (episode.rollout_id, episode.attempt) for episode in episodes
            ]
            if len(episode_keys) != len(set(episode_keys)):
                raise ValueError(
                    "Gym checkpoint instance "
                    f"{instance_id!r} contains duplicate episodes"
                )
        if set(self.staging_keys) != set(self.instances):
            raise ValueError(
                "Gym checkpoint staging-key topology must match its episode topology"
            )
        if set(self.restarts) != set(self.instances):
            raise ValueError(
                "Gym checkpoint restart topology must match its episode topology"
            )
        if set(self.participants) != set(self.instances):
            raise ValueError(
                "Gym checkpoint participant topology must match its episode topology"
            )
        for instance_id, restarts in self.restarts.items():
            if len(tuple(restarts)) != len(set(restarts)):
                raise ValueError(
                    "Gym checkpoint instance "
                    f"{instance_id!r} contains duplicate restarts"
                )
            continued = set(restarts) & set(self.instances[instance_id])
            if continued:
                raise ValueError(
                    "Gym checkpoint instance "
                    f"{instance_id!r} both continues and restarts "
                    f"{sorted(continued)!r}"
                )
        for instance_id, keys in self.staging_keys.items():
            if any(not key for key in keys):
                raise ValueError(
                    "Gym checkpoint instance "
                    f"{instance_id!r} contains an empty staging key"
                )
            # pyrefly widens the tuple ``keys`` to Iterable inside this loop.
            if len(keys) != len(set(keys)):  # pyrefly: ignore[bad-argument-type]
                raise ValueError(
                    "Gym checkpoint instance "
                    f"{instance_id!r} contains duplicate staging keys"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "instances": {
                instance_id: [asdict(episode) for episode in episodes]
                for instance_id, episodes in sorted(self.instances.items())
            },
            "staging_keys": {
                instance_id: list(keys)
                for instance_id, keys in sorted(self.staging_keys.items())
            },
            "restarts": {
                instance_id: [asdict(episode) for episode in episodes]
                for instance_id, episodes in sorted(self.restarts.items())
            },
            "participants": {
                instance_id: list(names)
                for instance_id, names in sorted(self.participants.items())
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GymCheckpointManifest:
        """Parse a manifest RL wrote; __post_init__ checks what it means."""
        # Before any field: an older manifest lacks fields this version requires.
        _require_schema_version(raw.get("schema_version"))
        return cls(
            schema_version=raw["schema_version"],
            checkpoint_id=raw["checkpoint_id"],
            instances=_episodes_by_instance(raw["instances"]),
            staging_keys={
                instance_id: tuple(keys)
                for instance_id, keys in raw["staging_keys"].items()
            },
            restarts=_episodes_by_instance(raw["restarts"]),
            participants={
                instance_id: tuple(names)
                for instance_id, names in raw["participants"].items()
            },
        )


def _require_schema_version(version: Any) -> None:
    if version != GYM_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Gym checkpoint schema version {version!r}: this NeMo RL "
            f"reads version {GYM_CHECKPOINT_SCHEMA_VERSION}. Resume with the NeMo RL "
            "that wrote the checkpoint, or start without the Gym checkpoint."
        )


def _episodes_by_instance(
    raw: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, tuple[GymCheckpointEpisode, ...]]:
    return {
        instance_id: tuple(
            GymCheckpointEpisode(episode["rollout_id"], episode["attempt"])
            for episode in episodes
        )
        for instance_id, episodes in raw.items()
    }


@dataclass(frozen=True)
class GymCheckpointCommitResult:
    """Outer manifest plus each Gym actor's validated commit summary."""

    manifest: GymCheckpointManifest
    instances: dict[str, GymCheckpointCommitSummary]


def staging_keys_of(
    summary: GymCheckpointCommitSummary, episodes: Iterable[GymCheckpointEpisode]
) -> tuple[str, ...]:
    """Return the staged rows of ``episodes``, the only rows a restore continues.

    Gym attributes each row to the episodes whose records hold it, so a
    generation-cut row needs no parsing and a carried-over row stays while any
    attempt that uses it is kept.
    """
    by_episode = summary.staging_keys_by_episode
    return tuple(
        sorted({key for episode in episodes for key in by_episode.get(episode, ())})
    )


def load_gym_checkpoint_manifest(checkpoint_root: str | Path) -> GymCheckpointManifest:
    """Load and validate the Gym manifest from one committed RL snapshot."""
    manifest_path = Path(checkpoint_root) / GYM_CHECKPOINT_MANIFEST_FILENAME
    raw = json.loads(manifest_path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("Gym checkpoint manifest must contain a mapping")
    return GymCheckpointManifest.from_mapping(raw)


def write_gym_checkpoint_manifest(
    checkpoint_root: str | Path,
    manifest: GymCheckpointManifest,
) -> None:
    """Atomically write a Gym manifest inside an unpublished RL snapshot."""
    root = Path(checkpoint_root)
    path = root / GYM_CHECKPOINT_MANIFEST_FILENAME
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n")
    os.replace(tmp_path, path)


class GymCheckpointCoordinator:
    """Drive one Gym checkpoint operation across every NeMo-Gym actor."""

    def __init__(
        self,
        shard_set: NemoGymShardSet,
        *,
        control_timeout_s: float,
        prepare_timeout_s: float,
    ) -> None:
        if control_timeout_s <= 0:
            raise ValueError("Gym checkpoint control_timeout_s must be positive")
        self._handles = shard_set.checkpoint_handles
        if not self._handles:
            raise ValueError("Gym checkpoint coordinator requires at least one actor")
        self._control_timeout_s = control_timeout_s
        self._prepare_timeout_s = prepare_timeout_s

    @property
    def instance_ids(self) -> frozenset[str]:
        return frozenset(self._handles)

    @property
    def control_timeout_s(self) -> float:
        """Maximum wall-clock budget for one coordinated control call but prepare."""
        return self._control_timeout_s

    def _deadline(self) -> float:
        return time.time() + self._control_timeout_s

    async def _collect(
        self,
        operation: str,
        references: Mapping[str, Any],
        *,
        deadline_ts: float,
    ) -> dict[str, Any]:
        """Wait for every actor's reply, but never past the call's deadline.

        Gym enforces ``deadline_ts`` only once it runs the call. An actor whose
        event loop is stuck, or a call Ray never schedules, would otherwise be
        waited on forever. Giving up is safe: a Gym deployment left parked
        reopens by itself when its checkpoint lease expires.
        """
        tasks = {
            instance_id: asyncio.ensure_future(reference)
            for instance_id, reference in references.items()
        }
        grace_s = min(self._control_timeout_s, _MAX_REPLY_GRACE_S)
        timeout_s = max(0.0, deadline_ts - time.time()) + grace_s
        try:
            await asyncio.wait(tasks.values(), timeout=timeout_s)
        except BaseException:
            for task in tasks.values():
                task.cancel()
            raise
        results: dict[str, Any] = {}
        failures: dict[str, BaseException] = {}
        for instance_id, task in tasks.items():
            if not task.done():
                task.cancel()
                _cancel_remote(references[instance_id])
                failures[instance_id] = TimeoutError(
                    f"no reply within {timeout_s:.1f}s of the {operation} call"
                )
            elif task.cancelled():
                failures[instance_id] = asyncio.CancelledError()
            elif task.exception() is not None:
                failures[instance_id] = task.exception()
            else:
                results[instance_id] = task.result()
        if failures:
            raise GymCheckpointOperationError(operation, failures)
        return results

    async def prepare(
        self,
        checkpoint_id: str,
    ) -> dict[str, GymCheckpointPrepareSummary]:
        # Parking waits for in-flight agent and environment steps to reach a
        # boundary, which can take far longer than one control call.
        deadline_ts = time.time() + self._prepare_timeout_s
        results = await self._collect(
            "prepare",
            {
                instance_id: handle.checkpoint_prepare.remote(
                    checkpoint_id,
                    deadline_ts=deadline_ts,
                )
                for instance_id, handle in self._handles.items()
            },
            deadline_ts=deadline_ts,
        )
        summaries = {instance_id: result for instance_id, result in results.items()}
        blockers = {
            instance_id: summary.blockers
            for instance_id, summary in summaries.items()
            if not summary.prepared
        }
        if blockers:
            raise GymCheckpointNotReady(blockers)
        return summaries

    async def renew(self, checkpoint_id: str) -> None:
        deadline_ts = self._deadline()
        await self._collect(
            "renew",
            {
                instance_id: handle.checkpoint_renew.remote(
                    checkpoint_id,
                    deadline_ts=deadline_ts,
                )
                for instance_id, handle in self._handles.items()
            },
            deadline_ts=deadline_ts,
        )

    def _validate_inventory(
        self,
        episodes_by_instance: Mapping[str, tuple[GymCheckpointEpisode, ...]],
    ) -> dict[str, tuple[GymCheckpointEpisode, ...]]:
        actual = set(episodes_by_instance)
        expected = set(self._handles)
        if actual != expected:
            raise ValueError(
                "Gym checkpoint inventory does not match the live actor topology: "
                f"missing={sorted(expected - actual)!r}, "
                f"unknown={sorted(actual - expected)!r}"
            )
        return {
            instance_id: tuple(episodes_by_instance[instance_id])
            for instance_id in self._handles
        }

    async def commit(
        self,
        checkpoint_id: str,
        checkpoint_root: str | Path,
        episodes_by_instance: Mapping[str, tuple[GymCheckpointEpisode, ...]],
        *,
        restarts: Mapping[str, tuple[GymCheckpointEpisode, ...]],
    ) -> GymCheckpointCommitResult:
        """Save ``episodes_by_instance`` and record ``restarts`` as starting over.

        Gym refuses a commit whose scope names an episode it reported as a
        restart, so the two must not overlap.
        """
        inventory = self._validate_inventory(episodes_by_instance)
        restarted = self._validate_inventory(restarts)
        overlap = {
            instance_id: sorted(set(inventory[instance_id]) & set(episodes))
            for instance_id, episodes in restarted.items()
            if set(inventory[instance_id]) & set(episodes)
        }
        if overlap:
            raise ValueError(f"Gym checkpoint commit scope names restarts: {overlap!r}")
        deadline_ts = self._deadline()
        raw_summaries = await self._collect(
            "commit",
            {
                instance_id: self._handles[instance_id].checkpoint_commit.remote(
                    checkpoint_id,
                    str(checkpoint_root),
                    episodes,
                    deadline_ts=deadline_ts,
                )
                for instance_id, episodes in inventory.items()
            },
            deadline_ts=deadline_ts,
        )
        summaries: dict[str, GymCheckpointCommitSummary] = {}
        for instance_id, summary in raw_summaries.items():
            if not isinstance(summary, GymCheckpointCommitSummary):
                raise TypeError(
                    "Gym checkpoint actor returned an invalid commit summary: "
                    f"instance={instance_id!r}, type={type(summary).__name__}"
                )
            summaries[instance_id] = summary
            expected = set(inventory[instance_id])
            actual = set(summary.exported_episodes)
            unexpected = actual - expected
            if unexpected:
                raise RuntimeError(
                    "Gym checkpoint actor exported an episode outside the "
                    "candidate set: "
                    f"instance={instance_id!r}, "
                    f"unexpected={sorted(unexpected, key=repr)!r}"
                )
        manifest = GymCheckpointManifest(
            schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id=checkpoint_id,
            instances={
                instance_id: summary.exported_episodes
                for instance_id, summary in summaries.items()
            },
            # Gym's model participant reports the rows of every candidate in
            # scope, including one whose reply was already on the wire. Only
            # saved episodes are restored; RL's ledger owns the rest once sealed.
            staging_keys={
                instance_id: staging_keys_of(summary, summary.exported_episodes)
                for instance_id, summary in summaries.items()
            },
            restarts=restarted,
            participants={
                instance_id: summary.participants
                for instance_id, summary in summaries.items()
            },
        )
        await asyncio.to_thread(
            write_gym_checkpoint_manifest,
            checkpoint_root,
            manifest,
        )
        return GymCheckpointCommitResult(
            manifest=manifest,
            instances=summaries,
        )

    async def retire(
        self,
        checkpoint_id: str,
        episodes_by_instance: Mapping[str, tuple[GymCheckpointEpisode, ...]],
    ) -> None:
        """Retire selected physical attempts from every Gym deployment."""
        inventory = self._validate_inventory(episodes_by_instance)
        deadline_ts = self._deadline()
        await self._collect(
            "retire",
            {
                instance_id: self._handles[instance_id].checkpoint_retire.remote(
                    checkpoint_id,
                    episodes,
                    deadline_ts=deadline_ts,
                )
                for instance_id, episodes in inventory.items()
            },
            deadline_ts=deadline_ts,
        )

    async def forget(
        self,
        checkpoint_id: str,
        episodes_by_instance: Mapping[str, tuple[GymCheckpointEpisode, ...]],
    ) -> None:
        """Let Gym stop refusing these rollouts: RL will never send them again.

        One call per deployment with anything to forget, naming each rollout once.
        """
        unknown = set(episodes_by_instance) - set(self._handles)
        if unknown:
            raise ValueError(
                "Gym forget names instances outside the live actor topology: "
                f"unknown={sorted(unknown)!r}"
            )
        deadline_ts = self._deadline()
        await self._collect(
            "forget",
            {
                instance_id: self._handles[instance_id].checkpoint_forget.remote(
                    checkpoint_id,
                    tuple(sorted({episode.rollout_id for episode in episodes})),
                    deadline_ts=deadline_ts,
                )
                for instance_id, episodes in episodes_by_instance.items()
                if episodes
            },
            deadline_ts=deadline_ts,
        )

    async def retire_restored(
        self,
        restore_id: str,
        manifest: GymCheckpointManifest,
    ) -> None:
        """Retire the replacement attempts installed by a successful restore."""
        await self.retire(
            restore_id,
            {
                instance_id: tuple(episode.next_attempt() for episode in episodes)
                for instance_id, episodes in manifest.instances.items()
            },
        )

    async def restore(
        self,
        restore_id: str,
        checkpoint_root: str | Path,
        manifest: GymCheckpointManifest,
    ) -> None:
        inventory = self._validate_inventory(manifest.instances)
        deadline_ts = self._deadline()
        try:
            await self._collect(
                "restore",
                {
                    instance_id: self._handles[instance_id].checkpoint_restore.remote(
                        restore_id,
                        str(checkpoint_root),
                        episodes,
                        participants=manifest.participants[instance_id],
                        source_checkpoint_id=manifest.checkpoint_id,
                        deadline_ts=deadline_ts,
                    )
                    for instance_id, episodes in inventory.items()
                },
                deadline_ts=deadline_ts,
            )
        except GymCheckpointOperationError as error:
            # Gym makes one deployment's restore atomic. RL must extend that
            # guarantee across independent Gym actors. Retire replacement
            # attempts on every actor before reopening them: a failed Ray reply
            # or actor-local validation error is an uncertain outcome, so the
            # remote restore may have succeeded before the failure reached this
            # coordinator.
            cleanup_errors: list[BaseException] = []
            try:
                cleanup_deadline_ts = self._deadline()
                await self._collect(
                    "restore_retire",
                    {
                        instance_id: self._handles[
                            instance_id
                        ].checkpoint_retire.remote(
                            restore_id,
                            tuple(
                                episode.next_attempt()
                                for episode in inventory[instance_id]
                            ),
                            deadline_ts=cleanup_deadline_ts,
                        )
                        for instance_id in inventory
                    },
                    deadline_ts=cleanup_deadline_ts,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                await self.resume(restore_id)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            for cleanup_error in cleanup_errors:
                error.add_note(
                    "Gym cross-instance restore cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    @asynccontextmanager
    async def restored(
        self,
        restore_id: str,
        checkpoint_root: str | Path,
        manifest: GymCheckpointManifest,
    ) -> AsyncIterator[None]:
        """Restore every deployment for the body, then always reopen it once.

        A failed restore cleans up and reopens Gym itself. A body that fails
        could not take over the restored attempts, so they are retired before
        Gym reopens and never continue.
        """
        await self.restore(restore_id, checkpoint_root, manifest)
        try:
            yield
        except BaseException as error:
            try:
                await self.retire_restored(restore_id, manifest)
            except Exception as retire_error:
                error.add_note(
                    "Gym retire of restored attempts also failed: "
                    f"{type(retire_error).__name__}: {retire_error}"
                )
            try:
                await self.resume(restore_id)
            except Exception as resume_error:
                error.add_note(
                    "Gym restore cleanup also failed: "
                    f"{type(resume_error).__name__}: {resume_error}"
                )
            raise
        else:
            await self.resume(restore_id)

    async def resume(self, checkpoint_id: str) -> None:
        """Reopen every deployment, retrying the ones a transient failure left closed."""
        handles = dict(self._handles)
        for attempt in range(1, _RESUME_ATTEMPTS + 1):
            deadline_ts = self._deadline()
            try:
                await self._collect(
                    "resume",
                    {
                        instance_id: handle.checkpoint_resume.remote(
                            checkpoint_id,
                            deadline_ts=deadline_ts,
                        )
                        for instance_id, handle in handles.items()
                    },
                    deadline_ts=deadline_ts,
                )
                return
            except GymCheckpointOperationError as error:
                if (
                    attempt == _RESUME_ATTEMPTS
                    or not is_transient_gym_checkpoint_failure(error)
                ):
                    raise
                handles = {
                    instance_id: self._handles[instance_id]
                    for instance_id in error.failures
                }
            await asyncio.sleep(_RESUME_RETRY_PAUSE_S)

    async def _renew_until_stopped(
        self,
        checkpoint_id: str,
        stop: asyncio.Event,
    ) -> None:
        """Keep a prepared Gym deployment parked during a slow outer save."""
        interval_s = max(0.01, min(self._control_timeout_s / 3.0, 10.0))
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except TimeoutError:
                await self.renew(checkpoint_id)
            else:
                return

    @asynccontextmanager
    async def prepared(
        self, checkpoint_id: str
    ) -> AsyncIterator[dict[str, frozenset[GymCheckpointEpisode]]]:
        """Prepare every deployment, renew its lease, and always release it.

        Yields each instance's restarts: the episodes a commit must leave out.
        Gym resumes when the body exits, before the caller publishes, so new
        rollouts wait only for Gym's own commit. Once the body succeeded every
        commit is written and resume only reopens admission: a resume that
        still fails is reported, not raised, since Gym's lease reopens that
        deployment on its own and the checkpoint stays valid.
        """
        try:
            summaries = await self.prepare(checkpoint_id)
            # A prepare may use most of its original deadline. Refresh once before
            # the outer controller begins disk I/O, then keep the lease alive.
            await self.renew(checkpoint_id)
            stop = asyncio.Event()
            renew_task = asyncio.create_task(
                self._renew_until_stopped(checkpoint_id, stop)
            )
            body_error: BaseException | None = None
            try:
                yield {
                    instance_id: frozenset(summary.restarts)
                    for instance_id, summary in summaries.items()
                }
            except BaseException as error:
                body_error = error
                raise
            finally:
                stop.set()
                (renew_result,) = await asyncio.gather(
                    renew_task,
                    return_exceptions=True,
                )
                if isinstance(renew_result, BaseException):
                    if body_error is None:
                        raise renew_result
                    body_error.add_note(
                        "Gym checkpoint lease renewal also failed: "
                        f"{type(renew_result).__name__}: {renew_result}"
                    )
        except BaseException as error:
            try:
                await self.resume(checkpoint_id)
            except Exception as resume_error:
                error.add_note(
                    "Gym checkpoint cleanup also failed: "
                    f"{type(resume_error).__name__}: {resume_error}"
                )
            raise
        else:
            try:
                await self.resume(checkpoint_id)
            except GymCheckpointOperationError as error:
                warnings.warn(
                    f"Gym checkpoint {checkpoint_id!r} is saved, but resume failed on "
                    f"{sorted(error.failures)!r}; those deployments stay closed until "
                    f"their lease expires: {error}",
                    stacklevel=2,
                )


def _cancel_remote(reference: Any) -> None:
    """Ask Ray to stop a control call RL stopped waiting for; best effort."""
    if not isinstance(reference, ray.ObjectRef):
        return
    try:
        ray.cancel(reference)
    except Exception:
        # The lease still reopens Gym; a failed cancel only leaves the call queued.
        pass
