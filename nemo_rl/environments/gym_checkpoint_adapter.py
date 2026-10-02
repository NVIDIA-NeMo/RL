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
"""Thin NeMo RL adapter over one NeMo Gym v2 checkpoint coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nemo_gym._checkpoint.coordination import Participants, PrepareResult
    from nemo_gym.server_utils import ServerClient


@dataclass(frozen=True)
class GymCheckpointInstance:
    """Stable identity and filesystem namespace for one logical Gym actor."""

    shard_name: str
    replica_index: int

    def __post_init__(self) -> None:
        if (
            not self.shard_name
            or self.shard_name in {".", ".."}
            or "/" in self.shard_name
            or "\\" in self.shard_name
        ):
            raise ValueError(
                "Gym checkpoint shard_name must be one safe path component"
            )
        if self.replica_index < 0:
            raise ValueError("Gym checkpoint replica_index must be non-negative")

    @property
    def instance_id(self) -> str:
        return f"{self.shard_name}/replica-{self.replica_index}"

    def live_capture_dir(self, capture_root: str | Path) -> Path:
        return Path(capture_root) / "gym-instances" / self.instance_id

    def checkpoint_dir(self, checkpoint_root: str | Path) -> Path:
        return Path(checkpoint_root) / "gym-instances" / self.instance_id


@dataclass(frozen=True)
class GymCheckpointParticipantSummary:
    """Serializable discovery result; the live Gym client remains actor-local."""

    instance_id: str
    members: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GymCheckpointParticipantManifest:
    """Integrity metadata for one Gym participant's committed state."""

    schema_version: int
    kind: str
    instance: str
    checkpoint_id: str
    records_file: str
    records_sha256: str
    record_count: int


GymCheckpointParticipantKind = Literal[
    "environment",
    "model",
    "agent",
    "resources",
]


@dataclass(frozen=True)
class GymCheckpointParticipantCommitSummary:
    """Validated commit reply from one discovered Gym participant."""

    server_name: str
    kind: GymCheckpointParticipantKind
    phase: str
    capture_keys: tuple[str, ...]
    manifest: GymCheckpointParticipantManifest
    staging_keys: tuple[str, ...]


@dataclass(frozen=True)
class GymCheckpointCommitSummary:
    """Actor-local Gym state selected by one checkpoint commit."""

    exported_episodes: tuple[GymCheckpointEpisode, ...]
    staging_keys: tuple[str, ...]
    participants: tuple[GymCheckpointParticipantCommitSummary, ...]
    # Each committed episode's staged rows, as Gym's records attribute them. A
    # row carried over from an earlier attempt belongs to both attempts.
    staging_keys_by_episode: Mapping[GymCheckpointEpisode, tuple[str, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        attributed = {
            key for keys in self.staging_keys_by_episode.values() for key in keys
        }
        if attributed != set(self.staging_keys):
            raise ValueError(
                "Gym checkpoint staging_keys_by_episode must attribute exactly the "
                "committed staging keys: "
                f"unattributed={sorted(set(self.staging_keys) - attributed)[:10]!r}, "
                f"unlisted={sorted(attributed - set(self.staging_keys))[:10]!r}"
            )

    # Parked agent sessions that no environment or legacy agent participant owns.
    # They cannot be restored and cannot finish while the deployment is prepared.
    unowned_live_episodes: tuple[GymCheckpointEpisode, ...] = ()


@dataclass(frozen=True)
class GymCheckpointEpisode:
    """Serializable identity of one physical Gym episode attempt."""

    rollout_id: str
    attempt: int

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, str) or not self.rollout_id:
            raise ValueError("Gym checkpoint rollout_id must not be empty")
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 0
        ):
            raise ValueError("Gym checkpoint attempt must be non-negative")

    def next_attempt(self) -> GymCheckpointEpisode:
        """Return the physical episode Gym creates when this one is restored."""
        return GymCheckpointEpisode(self.rollout_id, self.attempt + 1)


@dataclass(frozen=True)
class GymCheckpointPrepareSummary:
    """Serializable result of preparing one actor's Gym deployment."""

    prepared: bool
    blockers: dict[str, tuple[str, ...]]


class GymCheckpointUnavailable(RuntimeError):
    """Gym's control plane could not complete one live checkpoint call.

    Raised in place of Gym's ``CoordinationError``, which cannot be pickled and
    so reaches the driver as an untyped ``UnserializableException``. The
    participants' failures arrive as text, so this says only that the call
    failed, not why; nothing RL holds changed, and a later checkpoint may
    succeed.
    """


@contextmanager
def _live_checkpoint_call() -> Iterator[None]:
    """Type a Gym coordination failure before it crosses the Ray boundary."""
    from nemo_gym._checkpoint.coordination import CoordinationError

    try:
        yield
    except CoordinationError as error:
        raise GymCheckpointUnavailable(str(error)) from error


class GymCheckpointAdapter:
    """Delegate checkpoint operations for exactly one Gym deployment to Gym."""

    def __init__(
        self,
        *,
        instance: GymCheckpointInstance,
        client: ServerClient,
        auth_token: str,
    ) -> None:
        self._instance = instance
        self._client = client
        self._auth_token = auth_token
        self._participants: Participants | None = None

    async def discover(self) -> GymCheckpointParticipantSummary:
        from nemo_gym._checkpoint.coordination import discover

        if self._participants is None:
            self._participants = await discover(
                self._client, auth_token=self._auth_token
            )
        participants = self._participants
        return GymCheckpointParticipantSummary(
            instance_id=self._instance.instance_id,
            members=tuple(
                (member.server_name, member.kind) for member in participants.members
            ),
        )

    def _require_participants(self) -> Participants:
        if self._participants is None:
            raise RuntimeError(
                f"Gym checkpoint instance {self._instance.instance_id!r} has not "
                "discovered its deployment"
            )
        return self._participants

    async def prepare(self, checkpoint_id: str, *, deadline_ts: float) -> PrepareResult:
        from nemo_gym._checkpoint.coordination import prepare

        with _live_checkpoint_call():
            return await prepare(
                self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
            )

    async def renew(self, checkpoint_id: str, *, deadline_ts: float) -> None:
        from nemo_gym._checkpoint.coordination import renew

        with _live_checkpoint_call():
            await renew(
                self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
            )

    @staticmethod
    def _episode_ids(episodes: Iterable[GymCheckpointEpisode]) -> list[Any]:
        """Convert the RL wire identity to Gym's validated API model."""
        from nemo_gym.episode_types import EpisodeId

        return [
            EpisodeId(rollout_id=episode.rollout_id, attempt=episode.attempt)
            for episode in episodes
        ]

    @staticmethod
    def _participant_manifest(
        raw: object,
        *,
        participant: str,
        kind: GymCheckpointParticipantKind,
        checkpoint_id: str,
    ) -> GymCheckpointParticipantManifest:
        """Validate Gym's participant manifest before it leaves the actor."""
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "Gym checkpoint commit returned no manifest for participant "
                f"{participant!r}"
            )
        schema_version = raw.get("schema_version")
        manifest_kind = raw.get("kind")
        instance = raw.get("instance")
        manifest_checkpoint_id = raw.get("checkpoint_id")
        records_file = raw.get("records_file")
        records_sha256 = raw.get("records_sha256")
        record_count = raw.get("record_count")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise RuntimeError(
                "Gym checkpoint participant manifest has invalid schema_version: "
                f"participant={participant!r}, schema_version={schema_version!r}"
            )
        if manifest_kind != kind:
            raise RuntimeError(
                "Gym checkpoint participant manifest has the wrong kind: "
                f"participant={participant!r}, expected={kind!r}, "
                f"actual={manifest_kind!r}"
            )
        if not isinstance(instance, str) or not instance:
            raise RuntimeError(
                "Gym checkpoint participant manifest has invalid instance: "
                f"participant={participant!r}, instance={instance!r}"
            )
        if (
            not isinstance(manifest_checkpoint_id, str)
            or manifest_checkpoint_id != checkpoint_id
        ):
            raise RuntimeError(
                "Gym checkpoint participant manifest has the wrong checkpoint ID: "
                f"participant={participant!r}, expected={checkpoint_id!r}, "
                f"actual={manifest_checkpoint_id!r}"
            )
        if not isinstance(records_file, str) or not records_file:
            raise RuntimeError(
                "Gym checkpoint participant manifest has invalid records_file: "
                f"participant={participant!r}, records_file={records_file!r}"
            )
        if not isinstance(records_sha256, str) or not records_sha256:
            raise RuntimeError(
                "Gym checkpoint participant manifest has invalid records_sha256: "
                f"participant={participant!r}, records_sha256={records_sha256!r}"
            )
        if (
            not isinstance(record_count, int)
            or isinstance(record_count, bool)
            or record_count < 0
        ):
            raise RuntimeError(
                "Gym checkpoint participant manifest has invalid record_count: "
                f"participant={participant!r}, record_count={record_count!r}"
            )
        return GymCheckpointParticipantManifest(
            schema_version=schema_version,
            kind=kind,
            instance=instance,
            checkpoint_id=manifest_checkpoint_id,
            records_file=records_file,
            records_sha256=records_sha256,
            record_count=record_count,
        )

    @staticmethod
    def _legacy_agent_capture_keys(
        checkpoint_dir: Path,
        *,
        participant: str,
        manifest: GymCheckpointParticipantManifest,
    ) -> frozenset[str]:
        """Return the legacy ``/run`` episodes one agent participant committed.

        An agent exports every parked session, but only a legacy ``/run`` session
        (a record with a non-null ``episode`` step boundary) owns its episode. A
        native session runs under an environment participant, which owns it.
        """
        from nemo_gym._checkpoint.store import read_participant_state
        from nemo_gym.episode_types import EpisodeId

        stored, records = read_participant_state(
            checkpoint_dir, kind="agent", instance=manifest.instance
        )
        if stored.get("records_sha256") != manifest.records_sha256:
            raise RuntimeError(
                "Gym checkpoint agent records on disk do not match the commit "
                f"reply: participant={participant!r}, "
                f"expected={manifest.records_sha256!r}, "
                f"actual={stored.get('records_sha256')!r}"
            )
        return frozenset(
            EpisodeId.model_validate(record["episode_id"]).capture_key
            for record in records
            if record.get("episode") is not None
        )

    def _commit_summary(
        self,
        checkpoint_id: str,
        episode_ids: list[Any],
        replies: dict[str, dict[str, Any]],
        *,
        checkpoint_dir: Path,
    ) -> GymCheckpointCommitSummary:
        """Convert participant replies into an actor-local ownership summary."""
        participants = self._require_participants()
        expected_participants = {
            participant.server_name for participant in participants.members
        }
        if set(replies) != expected_participants:
            raise RuntimeError(
                "Gym checkpoint commit returned an unexpected participant set: "
                f"expected={sorted(expected_participants)!r}, "
                f"actual={sorted(replies)!r}"
            )

        requested_keys = {episode_id.capture_key for episode_id in episode_ids}
        exported_environment_keys: set[str] = set()
        # A legacy_agent environment server only relays /run, so it never holds
        # the episode; the agent's legacy session owns it instead.
        legacy_agent_keys: set[str] = set()
        agent_session_keys: set[str] = set()
        staging_keys: set[str] = set()
        keys_by_capture_key: dict[str, set[str]] = {}
        participant_summaries: list[GymCheckpointParticipantCommitSummary] = []
        for member in participants.members:
            reply = replies.get(member.server_name)
            if not isinstance(reply, Mapping):
                raise RuntimeError(
                    "Gym checkpoint commit returned an invalid reply for participant "
                    f"{member.server_name!r}"
                )
            phase = reply.get("phase")
            if phase != "committed":
                raise RuntimeError(
                    "Gym checkpoint participant did not enter committed phase: "
                    f"participant={member.server_name!r}, phase={phase!r}"
                )
            exported = reply.get("episode_ids")
            if not isinstance(exported, list) or not all(
                isinstance(capture_key, str) for capture_key in exported
            ):
                raise RuntimeError(
                    "Gym checkpoint commit returned invalid episode_ids for "
                    f"participant {member.server_name!r}"
                )
            capture_keys = tuple(sorted(set(exported)))
            if len(capture_keys) != len(exported):
                raise RuntimeError(
                    "Gym checkpoint commit returned duplicate episode IDs for "
                    f"participant {member.server_name!r}"
                )
            unexpected = set(capture_keys) - requested_keys
            if unexpected:
                raise RuntimeError(
                    "Gym checkpoint participant exported episodes outside the "
                    f"requested scope: participant={member.server_name!r}, "
                    f"unexpected={sorted(unexpected)!r}"
                )
            manifest = self._participant_manifest(
                reply.get("manifest"),
                participant=member.server_name,
                kind=member.kind,
                checkpoint_id=checkpoint_id,
            )
            if member.kind == "agent" and capture_keys:
                agent_session_keys.update(capture_keys)
                legacy_keys = self._legacy_agent_capture_keys(
                    checkpoint_dir,
                    participant=member.server_name,
                    manifest=manifest,
                )
                unreported = legacy_keys - set(capture_keys)
                if unreported:
                    raise RuntimeError(
                        "Gym checkpoint agent records name episodes absent from its "
                        f"commit reply: participant={member.server_name!r}, "
                        f"unreported={sorted(unreported)!r}"
                    )
                duplicate_owners = legacy_agent_keys.intersection(legacy_keys)
                if duplicate_owners:
                    raise RuntimeError(
                        "Gym checkpoint legacy episode was exported by multiple agent "
                        f"participants: {sorted(duplicate_owners)!r}"
                    )
                legacy_agent_keys.update(legacy_keys)
            if member.kind == "environment":
                duplicate_owners = exported_environment_keys.intersection(capture_keys)
                if duplicate_owners:
                    raise RuntimeError(
                        "Gym checkpoint episode was exported by multiple environment "
                        f"participants: {sorted(duplicate_owners)!r}"
                    )
                exported_environment_keys.update(capture_keys)

            participant_keys: tuple[str, ...] = ()
            if member.kind == "model":
                raw_staging_keys = reply.get("staging_keys")
                if not isinstance(raw_staging_keys, list) or not all(
                    isinstance(key, str) for key in raw_staging_keys
                ):
                    raise RuntimeError(
                        "Gym checkpoint policy model returned invalid staging_keys: "
                        f"participant={member.server_name!r}, "
                        f"staging_keys={raw_staging_keys!r}"
                    )
                participant_keys = tuple(sorted(set(raw_staging_keys)))
                if len(participant_keys) != len(raw_staging_keys):
                    raise RuntimeError(
                        "Gym checkpoint policy model returned duplicate staging_keys: "
                        f"participant={member.server_name!r}"
                    )
                staging_keys.update(participant_keys)
                raw_by_episode = reply.get("staging_keys_by_episode")
                if not isinstance(raw_by_episode, dict) or not all(
                    isinstance(capture_key, str)
                    and isinstance(keys, list)
                    and all(isinstance(key, str) for key in keys)
                    for capture_key, keys in raw_by_episode.items()
                ):
                    raise RuntimeError(
                        "Gym checkpoint policy model returned invalid "
                        f"staging_keys_by_episode: participant={member.server_name!r}"
                    )
                attributed = {key for keys in raw_by_episode.values() for key in keys}
                if attributed != set(participant_keys):
                    raise RuntimeError(
                        "Gym checkpoint policy model staging_keys_by_episode does not "
                        "attribute exactly its staging_keys: "
                        f"participant={member.server_name!r}, "
                        f"unattributed={sorted(set(participant_keys) - attributed)[:10]!r}, "
                        f"unlisted={sorted(attributed - set(participant_keys))[:10]!r}"
                    )
                for capture_key, keys in raw_by_episode.items():
                    keys_by_capture_key.setdefault(capture_key, set()).update(keys)

            participant_summaries.append(
                GymCheckpointParticipantCommitSummary(
                    server_name=member.server_name,
                    kind=member.kind,
                    phase=phase,
                    capture_keys=capture_keys,
                    manifest=manifest,
                    staging_keys=participant_keys,
                )
            )

        from nemo_gym.episode_types import EpisodeId

        def _episodes(capture_keys: set[str]) -> tuple[GymCheckpointEpisode, ...]:
            return tuple(
                GymCheckpointEpisode(parsed.rollout_id, parsed.attempt)
                for parsed in (
                    EpisodeId.from_capture_key(capture_key)
                    for capture_key in sorted(capture_keys)
                )
            )

        # An environment participant owns every episode it exported; a legacy
        # agent session owns its episode only when no environment does.
        owned_keys = exported_environment_keys | legacy_agent_keys
        return GymCheckpointCommitSummary(
            exported_episodes=_episodes(owned_keys),
            staging_keys=tuple(sorted(staging_keys)),
            participants=tuple(participant_summaries),
            staging_keys_by_episode={
                GymCheckpointEpisode(parsed.rollout_id, parsed.attempt): tuple(
                    sorted(keys)
                )
                for parsed, keys in (
                    (EpisodeId.from_capture_key(capture_key), keys)
                    for capture_key, keys in keys_by_capture_key.items()
                )
            },
            unowned_live_episodes=_episodes(agent_session_keys - owned_keys),
        )

    async def retire(
        self,
        checkpoint_id: str,
        episodes: Iterable[GymCheckpointEpisode],
        *,
        deadline_ts: float,
    ) -> None:
        from nemo_gym._checkpoint.coordination import retire

        with _live_checkpoint_call():
            await retire(
                self._require_participants(),
                checkpoint_id,
                self._episode_ids(episodes),
                deadline_ts=deadline_ts,
            )

    async def commit(
        self,
        checkpoint_id: str,
        checkpoint_root: str | Path,
        episodes: Iterable[GymCheckpointEpisode],
        *,
        deadline_ts: float,
    ) -> GymCheckpointCommitSummary:
        from nemo_gym._checkpoint.coordination import commit

        episode_ids = self._episode_ids(episodes)
        checkpoint_dir = self._instance.checkpoint_dir(checkpoint_root)
        with _live_checkpoint_call():
            replies = await commit(
                self._require_participants(),
                checkpoint_id,
                str(checkpoint_dir),
                episode_ids,
                deadline_ts=deadline_ts,
            )
        # Reads committed agent records from disk, so keep it off the event loop.
        return await asyncio.to_thread(
            self._commit_summary,
            checkpoint_id,
            episode_ids,
            replies,
            checkpoint_dir=checkpoint_dir,
        )

    async def restore(
        self,
        checkpoint_id: str,
        checkpoint_root: str | Path,
        episodes: Iterable[GymCheckpointEpisode],
        *,
        source_checkpoint_id: str,
        deadline_ts: float,
    ) -> None:
        from nemo_gym._checkpoint.coordination import restore

        replies = await restore(
            self._require_participants(),
            checkpoint_id,
            str(self._instance.checkpoint_dir(checkpoint_root)),
            self._episode_ids(episodes),
            deadline_ts=deadline_ts,
        )
        expected_participants = {
            participant.server_name
            for participant in self._require_participants().members
        }
        if set(replies) != expected_participants:
            raise RuntimeError(
                "Gym checkpoint restore returned an unexpected participant set: "
                f"expected={sorted(expected_participants)!r}, "
                f"actual={sorted(replies)!r}"
            )
        mismatched_sources: dict[str, object] = {}
        for participant, reply in replies.items():
            if not isinstance(reply, Mapping):
                mismatched_sources[participant] = type(reply).__name__
                continue
            actual_source = reply.get("source_checkpoint_id")
            if actual_source != source_checkpoint_id:
                mismatched_sources[participant] = actual_source
        if mismatched_sources:
            raise RuntimeError(
                "Gym checkpoint restore loaded participant state from the wrong "
                f"checkpoint: expected={source_checkpoint_id!r}, "
                f"actual={mismatched_sources!r}"
            )

    async def resume(self, checkpoint_id: str, *, deadline_ts: float) -> None:
        from nemo_gym._checkpoint.coordination import resume

        with _live_checkpoint_call():
            await resume(
                self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
            )
