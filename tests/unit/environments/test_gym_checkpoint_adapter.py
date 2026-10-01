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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointAdapter,
    GymCheckpointCommitSummary,
    GymCheckpointInstance,
)

pytestmark = pytest.mark.nemo_gym


def _participants(client: object, label: str) -> SimpleNamespace:
    return SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name=f"{label}-environment", kind="environment"),
            SimpleNamespace(server_name=f"{label}-model", kind="model"),
        ),
    )


def test_checkpoint_instance_paths_are_stable_and_isolated() -> None:
    first = GymCheckpointInstance(shard_name="tools", replica_index=0)
    replacement = GymCheckpointInstance(shard_name="tools", replica_index=0)
    sibling = GymCheckpointInstance(shard_name="tools", replica_index=1)

    assert first.instance_id == replacement.instance_id == "tools/replica-0"
    assert first.live_capture_dir("/capture") == replacement.live_capture_dir(
        "/capture"
    )
    assert first.live_capture_dir("/capture") != sibling.live_capture_dir("/capture")
    assert first.checkpoint_dir("/checkpoint") == (
        replacement.checkpoint_dir("/checkpoint")
    )


@pytest.mark.parametrize(
    ("shard_name", "replica_index"),
    [("", 0), (".", 0), ("..", 0), ("a/b", 0), ("a\\b", 0), ("a", -1)],
)
def test_checkpoint_instance_rejects_unsafe_identity(
    shard_name: str, replica_index: int
) -> None:
    with pytest.raises(ValueError):
        GymCheckpointInstance(
            shard_name=shard_name,
            replica_index=replica_index,
        )


def test_adapter_delegates_the_complete_gym_v2_lifecycle(monkeypatch) -> None:
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = _participants(client, "actor-a")
    prepare_result = object()
    commit_result = {
        "actor-a-environment": {"episode_ids": []},
        "actor-a-model": {"staging_keys": ["stage-2", "stage-1"]},
    }
    restore_result = {"model": {"restored": True}}

    discover = AsyncMock(return_value=participants)
    prepare = AsyncMock(return_value=prepare_result)
    renew = AsyncMock()
    retire = AsyncMock()
    commit = AsyncMock(return_value=commit_result)
    restore = AsyncMock(return_value=restore_result)
    resume = AsyncMock()
    for name, operation in (
        ("discover", discover),
        ("prepare", prepare),
        ("renew", renew),
        ("retire", retire),
        ("commit", commit),
        ("restore", restore),
        ("resume", resume),
    ):
        monkeypatch.setattr(coordination, name, operation)

    instance = GymCheckpointInstance(shard_name="actor-a", replica_index=0)
    adapter = GymCheckpointAdapter(
        instance=instance,
        client=client,
        auth_token="secret",
    )
    episode_ids = [object()]

    async def exercise() -> None:
        summary = await adapter.discover()
        assert summary.instance_id == "actor-a/replica-0"
        assert summary.members == (
            ("actor-a-environment", "environment"),
            ("actor-a-model", "model"),
        )
        # Discovery is cached inside the actor-local adapter.
        assert await adapter.discover() == summary
        assert await adapter.prepare("save-1", deadline_ts=10.0) is prepare_result
        await adapter.renew("save-1", deadline_ts=11.0)
        await adapter.retire("save-1", episode_ids, deadline_ts=12.0)
        assert await adapter.commit(
            "save-1",
            "/checkpoints/step-1",
            episode_ids,
            deadline_ts=13.0,
        ) == GymCheckpointCommitSummary(
            staging_keys=("stage-1", "stage-2"),
        )
        assert (
            await adapter.restore(
                "restore-1",
                "/checkpoints/step-1",
                episode_ids,
                deadline_ts=14.0,
            )
            == restore_result
        )
        await adapter.resume("save-1", deadline_ts=15.0)

    asyncio.run(exercise())

    discover.assert_awaited_once_with(client, auth_token="secret")
    prepare.assert_awaited_once_with(participants, "save-1", deadline_ts=10.0)
    renew.assert_awaited_once_with(participants, "save-1", deadline_ts=11.0)
    retire.assert_awaited_once_with(
        participants,
        "save-1",
        episode_ids,
        deadline_ts=12.0,
    )
    instance_dir = "/checkpoints/step-1/gym-instances/actor-a/replica-0"
    commit.assert_awaited_once_with(
        participants,
        "save-1",
        instance_dir,
        episode_ids,
        deadline_ts=13.0,
    )
    restore.assert_awaited_once_with(
        participants,
        "restore-1",
        instance_dir,
        episode_ids,
        deadline_ts=14.0,
    )
    resume.assert_awaited_once_with(participants, "save-1", deadline_ts=15.0)


def test_two_adapters_keep_their_clients_and_participants_isolated(monkeypatch) -> None:
    from nemo_gym._checkpoint import coordination

    client_a = object()
    client_b = object()
    participants_a = _participants(client_a, "actor-a")
    participants_b = _participants(client_b, "actor-b")

    async def discover(client, *, auth_token):
        assert auth_token == "secret"
        return participants_a if client is client_a else participants_b

    prepare = AsyncMock(return_value=object())
    monkeypatch.setattr(coordination, "discover", discover)
    monkeypatch.setattr(coordination, "prepare", prepare)

    adapter_a = GymCheckpointAdapter(
        instance=GymCheckpointInstance("actor-a", 0),
        client=client_a,
        auth_token="secret",
    )
    adapter_b = GymCheckpointAdapter(
        instance=GymCheckpointInstance("actor-b", 0),
        client=client_b,
        auth_token="secret",
    )

    async def exercise() -> None:
        await asyncio.gather(adapter_a.discover(), adapter_b.discover())
        await asyncio.gather(
            adapter_a.prepare("save-1", deadline_ts=10.0),
            adapter_b.prepare("save-1", deadline_ts=10.0),
        )

    asyncio.run(exercise())

    assert prepare.await_args_list[0].args[0] is participants_a
    assert prepare.await_args_list[1].args[0] is participants_b


def test_adapter_rejects_operations_before_discovery() -> None:
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("actor-a", 0),
        client=object(),
        auth_token="secret",
    )

    with pytest.raises(RuntimeError, match="has not discovered"):
        asyncio.run(adapter.prepare("save-1", deadline_ts=10.0))
