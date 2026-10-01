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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_gym._checkpoint.coordination import Participants, PrepareResult
    from nemo_gym.episode_types import EpisodeId
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
class GymCheckpointCommitSummary:
    """Checkpoint-owned TQ rows reported by Gym's policy participants."""

    staging_keys: tuple[str, ...]


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

        return await prepare(
            self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
        )

    async def renew(self, checkpoint_id: str, *, deadline_ts: float) -> None:
        from nemo_gym._checkpoint.coordination import renew

        await renew(
            self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
        )

    async def retire(
        self,
        checkpoint_id: str,
        episode_ids: Iterable[EpisodeId],
        *,
        deadline_ts: float,
    ) -> None:
        from nemo_gym._checkpoint.coordination import retire

        await retire(
            self._require_participants(),
            checkpoint_id,
            episode_ids,
            deadline_ts=deadline_ts,
        )

    async def commit(
        self,
        checkpoint_id: str,
        checkpoint_root: str | Path,
        episode_ids: Iterable[EpisodeId],
        *,
        deadline_ts: float,
    ) -> GymCheckpointCommitSummary:
        from nemo_gym._checkpoint.coordination import commit

        replies = await commit(
            self._require_participants(),
            checkpoint_id,
            str(self._instance.checkpoint_dir(checkpoint_root)),
            episode_ids,
            deadline_ts=deadline_ts,
        )
        staging_keys: set[str] = set()
        for participant in self._require_participants().members:
            if participant.kind != "model":
                continue
            reply = replies.get(participant.server_name)
            if not isinstance(reply, Mapping):
                raise RuntimeError(
                    "Gym checkpoint commit returned no reply for policy model "
                    f"participant {participant.server_name!r}"
                )
            participant_keys = reply.get("staging_keys")
            if not isinstance(participant_keys, list) or not all(
                isinstance(key, str) for key in participant_keys
            ):
                raise RuntimeError(
                    "Gym checkpoint policy model returned invalid staging_keys: "
                    f"participant={participant.server_name!r}, "
                    f"staging_keys={participant_keys!r}"
                )
            staging_keys.update(participant_keys)
        return GymCheckpointCommitSummary(staging_keys=tuple(sorted(staging_keys)))

    async def restore(
        self,
        checkpoint_id: str,
        checkpoint_root: str | Path,
        episode_ids: Iterable[EpisodeId],
        *,
        deadline_ts: float,
    ) -> dict[str, dict[str, Any]]:
        from nemo_gym._checkpoint.coordination import restore

        return await restore(
            self._require_participants(),
            checkpoint_id,
            str(self._instance.checkpoint_dir(checkpoint_root)),
            episode_ids,
            deadline_ts=deadline_ts,
        )

    async def resume(self, checkpoint_id: str, *, deadline_ts: float) -> None:
        from nemo_gym._checkpoint.coordination import resume

        await resume(
            self._require_participants(), checkpoint_id, deadline_ts=deadline_ts
        )
