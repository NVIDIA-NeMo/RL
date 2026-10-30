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

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
class GymCheckpointCommitSummary:
    """Actor-local Gym state selected by one checkpoint commit."""

    exported_episodes: tuple[GymCheckpointEpisode, ...]
    staging_keys: tuple[str, ...]
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

    # Parked agent sessions that are not an agent's own /run episode. They
    # cannot be restored and cannot finish while the deployment is prepared.
    unowned_live_episodes: tuple[GymCheckpointEpisode, ...] = ()


@dataclass(frozen=True, order=True)
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

    @property
    def capture_key(self) -> str:
        """Gym's attempt-qualified key: ``EpisodeId.capture_key``, mirrored."""
        return (
            self.rollout_id
            if self.attempt == 0
            else f"{self.rollout_id}-a{self.attempt}"
        )

    def next_attempt(self) -> GymCheckpointEpisode:
        """Return the physical episode Gym creates when this one is restored."""
        return GymCheckpointEpisode(self.rollout_id, self.attempt + 1)


@dataclass(frozen=True)
class GymCheckpointPrepareSummary:
    """Serializable result of preparing one actor's Gym deployment.

    ``restarts`` are in-flight episodes some participant cannot capture: Gym
    refuses a commit that names one, and they start over after a crash.
    """

    prepared: bool
    blockers: dict[str, tuple[str, ...]]
    restarts: tuple[GymCheckpointEpisode, ...]

    @classmethod
    def from_result(cls, result: PrepareResult) -> GymCheckpointPrepareSummary:
        return cls(
            prepared=result.prepared,
            blockers={
                server: tuple(blockers)
                for server, blockers in result.blockers().items()
            },
            restarts=tuple(
                sorted(
                    GymCheckpointEpisode(episode.rollout_id, episode.attempt)
                    for episode in result.restarts()
                )
            ),
        )


class GymCheckpointUnavailable(RuntimeError):
    """Gym's control plane could not complete one checkpoint control call.

    Raised in place of Gym's ``CoordinationError``, which cannot be pickled and
    so reaches the driver as an untyped ``UnserializableException``. The
    participants' failures arrive as text, so this says only that the call
    failed, not why; nothing RL holds changed, and a later checkpoint may
    succeed.
    """


@contextmanager
def _reraise_as_gym_checkpoint_unavailable() -> Iterator[None]:
    """Turn Gym's unpicklable ``CoordinationError`` into ``GymCheckpointUnavailable``.

    Every Gym coordination call goes through this, so its failure reaches the
    driver typed, with Gym's message, instead of as ``UnserializableException``.
    """
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

    async def discover(self) -> None:
        from nemo_gym._checkpoint.coordination import discover

        if self._participants is not None:
            return
        with _reraise_as_gym_checkpoint_unavailable():
            participants = await discover(self._client, auth_token=self._auth_token)
        # Only typed environment servers take part in checkpoints, and turn
        # recovery refuses them at setup; the legacy_agent relay is not a
        # participant. Commits therefore assume every episode is an agent's own
        # /run episode, so fail rather than misattribute one an environment
        # server owns.
        environment_servers = sorted(
            member.server_name
            for member in participants.members
            if member.kind == "environment"
        )
        if environment_servers:
            raise RuntimeError(
                "Gym turn recovery does not support environment-server checkpoint "
                f"participants yet: {environment_servers!r}. Route these rows through "
                "a legacy_agent environment server or use target_level='sibling'."
            )
        self._participants = participants

    async def restart_only_servers(self) -> list[str]:
        """Return the participants whose rollouts start over after a crash.

        Gym reports their in-flight rollouts as restarts at every checkpoint, so
        turn-level recovery quietly becomes restart-from-task for them. Each
        participant declares this as its checkpoint ``mode``, which discovery
        already returned.
        """
        return sorted(
            member.server_name
            for member in self._require_participants().members
            if member.mode == "restart_only"
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

        with _reraise_as_gym_checkpoint_unavailable():
            return await prepare(
                self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
            )

    async def renew(self, checkpoint_id: str, *, deadline_ts: float) -> None:
        from nemo_gym._checkpoint.coordination import renew

        with _reraise_as_gym_checkpoint_unavailable():
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

    def _commit_summary(
        self,
        checkpoint_id: str,
        episode_ids: list[Any],
        replies: dict[str, dict[str, Any]],
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
        # A legacy_agent environment server only relays /run, so it never holds
        # the episode: it is the agent's own /run episode instead.
        run_episode_keys: set[str] = set()
        agent_session_keys: set[str] = set()
        staging_keys: set[str] = set()
        keys_by_capture_key: dict[str, set[str]] = {}
        # Gym's control plane builds these replies; only how they relate to the
        # request and to each other is checked here.
        for member in participants.members:
            reply = replies[member.server_name]
            capture_keys = set(reply["episode_ids"])
            unexpected = capture_keys - requested_keys
            if unexpected:
                raise RuntimeError(
                    "Gym checkpoint participant exported episodes outside the "
                    f"requested scope: participant={member.server_name!r}, "
                    f"unexpected={sorted(unexpected)!r}"
                )
            if member.kind == "agent" and capture_keys:
                agent_session_keys.update(capture_keys)
                # The committed episodes the agent's own /run drives.
                own_keys = set(reply["run_episode_ids"])
                unexported = own_keys - capture_keys
                if unexported:
                    raise RuntimeError(
                        "Gym checkpoint agent names /run episodes it did not export: "
                        f"participant={member.server_name!r}, "
                        f"unexported={sorted(unexported)!r}"
                    )
                duplicate_owners = run_episode_keys.intersection(own_keys)
                if duplicate_owners:
                    raise RuntimeError(
                        "Gym checkpoint episode was exported as their own /run episode "
                        f"by multiple agent participants: {sorted(duplicate_owners)!r}"
                    )
                run_episode_keys.update(own_keys)
            if member.kind == "model":
                staging_keys.update(reply["staging_keys"])
                for capture_key, keys in reply["staging_keys_by_episode"].items():
                    keys_by_capture_key.setdefault(capture_key, set()).update(keys)

        from nemo_gym.episode_types import EpisodeId

        def _episodes(capture_keys: set[str]) -> tuple[GymCheckpointEpisode, ...]:
            return tuple(
                GymCheckpointEpisode(parsed.rollout_id, parsed.attempt)
                for parsed in (
                    EpisodeId.from_capture_key(capture_key)
                    for capture_key in sorted(capture_keys)
                )
            )

        # An agent owns only its own /run episodes; discover refuses
        # environment servers, the other kind of owner.
        owned_keys = run_episode_keys
        return GymCheckpointCommitSummary(
            exported_episodes=_episodes(owned_keys),
            staging_keys=tuple(sorted(staging_keys)),
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

        with _reraise_as_gym_checkpoint_unavailable():
            await retire(
                self._require_participants(),
                checkpoint_id,
                self._episode_ids(episodes),
                deadline_ts=deadline_ts,
            )

    async def forget(
        self,
        checkpoint_id: str,
        rollout_ids: Iterable[str],
        *,
        deadline_ts: float,
    ) -> None:
        from nemo_gym._checkpoint.coordination import forget

        with _reraise_as_gym_checkpoint_unavailable():
            await forget(
                self._require_participants(),
                checkpoint_id,
                list(rollout_ids),
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
        with _reraise_as_gym_checkpoint_unavailable():
            replies = await commit(
                self._require_participants(),
                checkpoint_id,
                str(checkpoint_dir),
                episode_ids,
                deadline_ts=deadline_ts,
            )
        return self._commit_summary(checkpoint_id, episode_ids, replies)

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

        with _reraise_as_gym_checkpoint_unavailable():
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

        with _reraise_as_gym_checkpoint_unavailable():
            await resume(
                self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
            )
