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

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointCommitSummary,
    GymCheckpointEpisode,
    GymCheckpointPrepareSummary,
)
from nemo_rl.environments.gym_checkpoint_coordinator import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymCheckpointCoordinator,
    GymCheckpointManifest,
    GymCheckpointNotReady,
    GymCheckpointOperationError,
    load_gym_checkpoint_manifest,
)
from nemo_rl.environments.nemo_gym import NemoGymShardSet

pytestmark = pytest.mark.nemo_gym


class _RemoteMethod:
    def __init__(self, function) -> None:
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class _FakeGymActor:
    def __init__(
        self,
        *,
        prepared: bool = True,
        restore_error: BaseException | None = None,
        exported_episodes: tuple[GymCheckpointEpisode, ...] | None = None,
        staging_keys: tuple[str, ...] = (),
    ) -> None:
        self.prepared = prepared
        self.restore_error = restore_error
        self.exported_episodes = exported_episodes
        self.staging_keys = staging_keys
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.checkpoint_prepare = _RemoteMethod(self._prepare)
        self.checkpoint_renew = _RemoteMethod(self._renew)
        self.checkpoint_retire = _RemoteMethod(self._retire)
        self.checkpoint_commit = _RemoteMethod(self._commit)
        self.checkpoint_restore = _RemoteMethod(self._restore)
        self.checkpoint_resume = _RemoteMethod(self._resume)

    async def _prepare(self, *args, **kwargs) -> GymCheckpointPrepareSummary:
        self.calls.append(("prepare", args, kwargs))
        return GymCheckpointPrepareSummary(
            prepared=self.prepared,
            blockers={} if self.prepared else {"agent": ("rollout-1",)},
        )

    async def _renew(self, *args, **kwargs) -> None:
        self.calls.append(("renew", args, kwargs))

    async def _commit(self, *args, **kwargs) -> GymCheckpointCommitSummary:
        self.calls.append(("commit", args, kwargs))
        episodes = args[2] if self.exported_episodes is None else self.exported_episodes
        return GymCheckpointCommitSummary(
            exported_episodes=episodes,
            staging_keys=self.staging_keys,
            participants=(),
        )

    async def _retire(self, *args, **kwargs) -> None:
        self.calls.append(("retire", args, kwargs))

    async def _restore(self, *args, **kwargs) -> None:
        self.calls.append(("restore", args, kwargs))
        if self.restore_error is not None:
            raise self.restore_error

    async def _resume(self, *args, **kwargs) -> None:
        self.calls.append(("resume", args, kwargs))


def _coordinator(*actors: _FakeGymActor) -> GymCheckpointCoordinator:
    return GymCheckpointCoordinator(
        NemoGymShardSet(handles={"tools": list(actors)}),
        control_timeout_s=30.0,
    )


def test_manifest_rejects_duplicate_episode_identity() -> None:
    episode = GymCheckpointEpisode("rollout-0", 2)

    with pytest.raises(ValueError, match="duplicate episodes"):
        GymCheckpointManifest(
            schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id="save-1",
            instances={"tools/replica-0": (episode, episode)},
            staging_keys={"tools/replica-0": ()},
        )


def test_coordinator_commits_and_restores_each_instance_inventory(
    tmp_path: Path,
) -> None:
    first = _FakeGymActor(staging_keys=("rollout-0/call-0",))
    second = _FakeGymActor()
    coordinator = _coordinator(first, second)
    inventory = {
        "tools/replica-0": (GymCheckpointEpisode("rollout-0", 0),),
        "tools/replica-1": (GymCheckpointEpisode("rollout-1", 2),),
    }

    async def exercise() -> None:
        async with coordinator.prepared("save-1"):
            result = await coordinator.commit("save-1", tmp_path, inventory)
            assert result.manifest.instances == inventory
            assert set(result.instances) == set(inventory)
            assert result.instances["tools/replica-0"].exported_episodes == (
                GymCheckpointEpisode("rollout-0", 0),
            )
            assert result.manifest.staging_keys == {
                "tools/replica-0": ("rollout-0/call-0",),
                "tools/replica-1": (),
            }
            await coordinator.renew("save-1")
        manifest = load_gym_checkpoint_manifest(tmp_path)
        assert manifest.checkpoint_id == "save-1"
        assert manifest.instances == inventory
        assert manifest.staging_keys == {
            "tools/replica-0": ("rollout-0/call-0",),
            "tools/replica-1": (),
        }
        await coordinator.restore("restore-1", tmp_path, manifest)
        await coordinator.discard_restored("restore-1", manifest)
        await coordinator.resume("restore-1")

    asyncio.run(exercise())

    for replica, actor in enumerate((first, second)):
        operations = [call[0] for call in actor.calls]
        assert operations == [
            "prepare",
            "renew",
            "commit",
            "renew",
            "resume",
            "restore",
            "retire",
            "resume",
        ]
        commit_args = actor.calls[2][1]
        assert commit_args[0] == "save-1"
        assert commit_args[2] == inventory[f"tools/replica-{replica}"]
        retire_args = actor.calls[-2][1]
        assert retire_args[0] == "restore-1"
        assert retire_args[1] == tuple(
            episode.next_attempt() for episode in inventory[f"tools/replica-{replica}"]
        )
        assert actor.calls[5][2]["source_checkpoint_id"] == "save-1"


def test_coordinator_resumes_every_actor_when_prepare_is_not_ready() -> None:
    ready = _FakeGymActor()
    blocked = _FakeGymActor(prepared=False)
    coordinator = _coordinator(ready, blocked)

    async def exercise() -> None:
        with pytest.raises(GymCheckpointNotReady, match="rollout-1"):
            async with coordinator.prepared("save-1"):
                raise AssertionError("unreachable")

    asyncio.run(exercise())

    assert [call[0] for call in ready.calls] == ["prepare", "resume"]
    assert [call[0] for call in blocked.calls] == ["prepare", "resume"]


def test_coordinator_renews_lease_while_outer_checkpoint_is_slow() -> None:
    actor = _FakeGymActor()
    coordinator = GymCheckpointCoordinator(
        NemoGymShardSet(handles={"tools": [actor]}),
        control_timeout_s=0.03,
    )

    async def exercise() -> None:
        async with coordinator.prepared("save-1"):
            await asyncio.sleep(0.04)

    asyncio.run(exercise())

    operations = [call[0] for call in actor.calls]
    assert operations[0:2] == ["prepare", "renew"]
    assert operations.count("renew") >= 2
    assert operations[-1] == "resume"


def test_coordinator_rejects_inventory_for_the_wrong_actor_topology(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(_FakeGymActor())

    with pytest.raises(ValueError, match="live actor topology"):
        asyncio.run(
            coordinator.commit(
                "save-1",
                tmp_path,
                {"other/replica-0": ()},
            )
        )


def test_coordinator_uses_environment_exported_subset_as_manifest(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(_FakeGymActor(exported_episodes=()))

    result = asyncio.run(
        coordinator.commit(
            "save-1",
            tmp_path,
            {"tools/replica-0": (GymCheckpointEpisode("rollout-on-wire", 0),)},
        )
    )

    assert result.manifest.instances == {"tools/replica-0": ()}
    assert load_gym_checkpoint_manifest(tmp_path).instances == {"tools/replica-0": ()}


def test_coordinator_rejects_episode_outside_candidate_set(tmp_path: Path) -> None:
    coordinator = _coordinator(
        _FakeGymActor(
            exported_episodes=(GymCheckpointEpisode("unexpected-rollout", 0),)
        )
    )

    with pytest.raises(RuntimeError, match="outside the candidate set"):
        asyncio.run(
            coordinator.commit(
                "save-1",
                tmp_path,
                {"tools/replica-0": (GymCheckpointEpisode("rollout-on-wire", 0),)},
            )
        )


def test_restore_failure_discards_every_actor_under_uncertain_outcome() -> None:
    restored = _FakeGymActor()
    failed = _FakeGymActor(restore_error=RuntimeError("restore failed"))
    coordinator = _coordinator(restored, failed)
    manifest = GymCheckpointManifest(
        schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id="save-1",
        instances={
            "tools/replica-0": (GymCheckpointEpisode("rollout-0", 0),),
            "tools/replica-1": (GymCheckpointEpisode("rollout-1", 0),),
        },
        staging_keys={
            "tools/replica-0": (),
            "tools/replica-1": (),
        },
    )

    with pytest.raises(GymCheckpointOperationError, match="restore failed"):
        asyncio.run(coordinator.restore("restore-1", "/unused", manifest))

    assert [call[0] for call in restored.calls] == ["restore", "retire", "resume"]
    assert [call[0] for call in failed.calls] == ["restore", "retire", "resume"]
