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
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointAdapter,
    GymCheckpointEpisode,
    GymCheckpointInstance,
    GymCheckpointParticipantManifest,
)
from nemo_rl.experience.rollout_recovery import (
    PromptGroupPhase,
    PromptGroupRecoveryRecord,
    PromptGroupStatus,
    PromptRef,
    RecoveryGranularity,
    RecoveryTargetLevel,
    RolloutAttemptRecord,
    RolloutAttemptStatus,
    RolloutSiblingRecord,
)

pytestmark = pytest.mark.nemo_gym


@pytest.mark.parametrize("gym_attempt", [0, 1, 17])
def test_rollout_capture_key_matches_gym_episode_identity(gym_attempt: int) -> None:
    """Fail loudly if RL's legacy string carrier drifts from Gym's codec."""
    from nemo_gym.episode_types import EpisodeId

    group = PromptGroupRecoveryRecord(
        group_id="group-7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_ref=PromptRef(sample_id="7", task_name=None),
        task_source=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        restore_level=RecoveryTargetLevel.TURN,
        runtime_prompt_payload=None,
        expected_generations=1,
        target_step=7,
        start_weight_version=6,
        siblings=[
            RolloutSiblingRecord(
                generation_index=0,
                attempts=[
                    RolloutAttemptRecord(
                        attempt_uuid=uuid.UUID(int=1),
                        status=RolloutAttemptStatus.DISPATCHED,
                        gym_instance_id="tools/replica-0",
                        gym_attempt=gym_attempt,
                    )
                ],
            )
        ],
        phase=PromptGroupPhase.ADMITTED,
        status=PromptGroupStatus.GENERATING,
    )

    rollout_id, attempt = group.gym_episode(0)
    gym_episode = EpisodeId(rollout_id=rollout_id, attempt=attempt)
    capture_key = group.gate_rollout_id(0)

    assert capture_key == gym_episode.capture_key
    assert EpisodeId.from_capture_key(capture_key) == gym_episode


def _participants(client: object, label: str) -> SimpleNamespace:
    return SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name=f"{label}-environment", kind="environment"),
            SimpleNamespace(server_name=f"{label}-model", kind="model"),
        ),
    )


def _participant_manifest(
    label: str,
    kind: str,
    *,
    checkpoint_id: str = "save-1",
    record_count: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": kind,
        "instance": f"{label}-{kind}",
        "checkpoint_id": checkpoint_id,
        "records_file": "records.jsonl",
        "records_sha256": "0" * 64,
        "record_count": record_count,
    }


def _commit_reply(
    label: str,
    kind: str,
    episode_ids: list[str],
    *,
    staging_keys: list[str] | None = None,
    staging_keys_by_episode: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    reply: dict[str, object] = {
        "phase": "committed",
        "manifest": _participant_manifest(
            label,
            kind,
            record_count=len(episode_ids),
        ),
        "episode_ids": episode_ids,
    }
    if staging_keys is not None:
        reply["staging_keys"] = staging_keys
    if staging_keys_by_episode is not None:
        reply["staging_keys_by_episode"] = staging_keys_by_episode
    return reply


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
        "actor-a-environment": _commit_reply(
            "actor-a", "environment", ["rollout-1-a2"]
        ),
        "actor-a-model": _commit_reply(
            "actor-a",
            "model",
            ["rollout-1-a2"],
            staging_keys=["stage-2", "stage-1"],
            staging_keys_by_episode={"rollout-1-a2": ["stage-2", "stage-1"]},
        ),
    }
    restore_result = {
        "actor-a-environment": {"source_checkpoint_id": "save-1"},
        "actor-a-model": {"source_checkpoint_id": "save-1"},
    }

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
    episodes = [GymCheckpointEpisode("rollout-1", 2)]

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
        await adapter.retire("save-1", episodes, deadline_ts=12.0)
        commit_summary = await adapter.commit(
            "save-1",
            "/checkpoints/step-1",
            episodes,
            deadline_ts=13.0,
        )
        assert commit_summary.exported_episodes == (
            GymCheckpointEpisode("rollout-1", 2),
        )
        assert commit_summary.staging_keys == ("stage-1", "stage-2")
        assert commit_summary.staging_keys_by_episode == {
            GymCheckpointEpisode("rollout-1", 2): ("stage-1", "stage-2")
        }
        assert [
            participant.server_name for participant in commit_summary.participants
        ] == ["actor-a-environment", "actor-a-model"]
        assert commit_summary.participants[0].manifest == (
            GymCheckpointParticipantManifest(
                schema_version=1,
                kind="environment",
                instance="actor-a-environment",
                checkpoint_id="save-1",
                records_file="records.jsonl",
                records_sha256="0" * 64,
                record_count=1,
            )
        )
        await adapter.restore(
            "restore-1",
            "/checkpoints/step-1",
            episodes,
            source_checkpoint_id="save-1",
            deadline_ts=14.0,
        )
        await adapter.resume("save-1", deadline_ts=15.0)

    asyncio.run(exercise())

    discover.assert_awaited_once_with(client, auth_token="secret")
    prepare.assert_awaited_once_with(participants, "save-1", deadline_ts=10.0)
    renew.assert_awaited_once_with(participants, "save-1", deadline_ts=11.0)
    (retire_episode,) = retire.await_args.args[2]
    assert retire_episode.rollout_id == "rollout-1"
    assert retire_episode.attempt == 2
    retire.assert_awaited_once_with(
        participants,
        "save-1",
        retire.await_args.args[2],
        deadline_ts=12.0,
    )
    instance_dir = "/checkpoints/step-1/gym-instances/actor-a/replica-0"
    (commit_episode,) = commit.await_args.args[3]
    assert commit_episode == retire_episode
    commit.assert_awaited_once_with(
        participants,
        "save-1",
        instance_dir,
        commit.await_args.args[3],
        deadline_ts=13.0,
    )
    (restore_episode,) = restore.await_args.args[3]
    assert restore_episode == retire_episode
    restore.assert_awaited_once_with(
        participants,
        "restore-1",
        instance_dir,
        restore.await_args.args[3],
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


def test_adapter_reports_candidates_not_exported_by_the_environment(
    monkeypatch,
) -> None:
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = _participants(client, "actor-a")
    monkeypatch.setattr(
        coordination,
        "discover",
        AsyncMock(return_value=participants),
    )
    monkeypatch.setattr(
        coordination,
        "commit",
        AsyncMock(
            return_value={
                "actor-a-environment": _commit_reply("actor-a", "environment", []),
                "actor-a-model": _commit_reply(
                    "actor-a", "model", [], staging_keys=[], staging_keys_by_episode={}
                ),
            }
        ),
    )
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("actor-a", 0),
        client=client,
        auth_token="secret",
    )

    async def exercise() -> None:
        await adapter.discover()
        summary = await adapter.commit(
            "save-1",
            "/checkpoints/step-1",
            [GymCheckpointEpisode("rollout-1", 0)],
            deadline_ts=10.0,
        )
        assert summary.exported_episodes == ()
        assert summary.staging_keys == ()

    asyncio.run(exercise())


def test_adapter_does_not_require_optional_agent_or_resource_episode_state(
    monkeypatch,
) -> None:
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name="environment", kind="environment"),
            SimpleNamespace(server_name="model", kind="model"),
            SimpleNamespace(server_name="agent", kind="agent"),
            SimpleNamespace(server_name="resources", kind="resources"),
        ),
    )
    monkeypatch.setattr(
        coordination,
        "discover",
        AsyncMock(return_value=participants),
    )
    monkeypatch.setattr(
        coordination,
        "commit",
        AsyncMock(
            return_value={
                "environment": _commit_reply(
                    "deployment", "environment", ["rollout-1"]
                ),
                "model": _commit_reply(
                    "deployment",
                    "model",
                    ["rollout-1"],
                    staging_keys=["rollout-1/call-1"],
                    staging_keys_by_episode={"rollout-1": ["rollout-1/call-1"]},
                ),
                "agent": _commit_reply("deployment", "agent", []),
                "resources": _commit_reply("deployment", "resources", []),
            }
        ),
    )
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("deployment", 0),
        client=client,
        auth_token="secret",
    )

    async def exercise() -> None:
        await adapter.discover()
        summary = await adapter.commit(
            "save-1",
            "/checkpoints/step-1",
            [GymCheckpointEpisode("rollout-1", 0)],
            deadline_ts=10.0,
        )
        assert summary.exported_episodes == (GymCheckpointEpisode("rollout-1", 0),)
        assert summary.staging_keys == ("rollout-1/call-1",)
        assert {
            participant.kind: participant.episode_keys
            for participant in summary.participants
        } == {
            "environment": ("rollout-1",),
            "model": ("rollout-1",),
            "agent": (),
            "resources": (),
        }

    asyncio.run(exercise())


def _commit_model_reply(monkeypatch, model_reply: dict[str, object]):
    """Commit two episodes with an environment and the given policy-model reply."""
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name="environment", kind="environment"),
            SimpleNamespace(server_name="model", kind="model"),
        ),
    )
    monkeypatch.setattr(coordination, "discover", AsyncMock(return_value=participants))
    monkeypatch.setattr(
        coordination,
        "commit",
        AsyncMock(
            return_value={
                "environment": _commit_reply(
                    "deployment", "environment", ["rollout-1", "rollout-2-a1"]
                ),
                "model": model_reply,
            }
        ),
    )
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("deployment", 0),
        client=client,
        auth_token="secret",
    )

    async def exercise():
        await adapter.discover()
        return await adapter.commit(
            "save-1",
            "/checkpoints/step-1",
            [
                GymCheckpointEpisode("rollout-1", 0),
                GymCheckpointEpisode("rollout-2", 1),
            ],
            deadline_ts=10.0,
        )

    return asyncio.run(exercise())


def test_staging_keys_are_attributed_to_their_episodes_by_gym(monkeypatch) -> None:
    """A cut key names no rollout and a carried-over row serves two attempts."""
    cut_key = "__generation_cut__/save-1/c9"
    summary = _commit_model_reply(
        monkeypatch,
        _commit_reply(
            "deployment",
            "model",
            ["rollout-1", "rollout-2-a1"],
            staging_keys=["rollout-1/c1", "rollout-2/c1", cut_key],
            staging_keys_by_episode={
                "rollout-1": ["rollout-1/c1"],
                "rollout-2-a1": ["rollout-2/c1", cut_key],
            },
        ),
    )

    assert summary.staging_keys_by_episode == {
        GymCheckpointEpisode("rollout-1", 0): ("rollout-1/c1",),
        GymCheckpointEpisode("rollout-2", 1): (cut_key, "rollout-2/c1"),
    }


@pytest.mark.parametrize(
    "staging_keys_by_episode",
    [
        pytest.param(None, id="missing"),
        pytest.param({"rollout-1": ["rollout-1/c1"]}, id="a-key-with-no-episode"),
        pytest.param(
            {"rollout-1": ["rollout-1/c1"], "rollout-2-a1": ["rollout-2/c1", "extra"]},
            id="a-key-outside-the-flat-list",
        ),
    ],
)
def test_staging_keys_gym_does_not_attribute_fail_the_commit(
    monkeypatch, staging_keys_by_episode
) -> None:
    """Keeping or dropping a row whose owner is unknown would both be a guess."""
    with pytest.raises(RuntimeError, match="staging_keys_by_episode"):
        _commit_model_reply(
            monkeypatch,
            _commit_reply(
                "deployment",
                "model",
                ["rollout-1", "rollout-2-a1"],
                staging_keys=["rollout-1/c1", "rollout-2/c1"],
                staging_keys_by_episode=staging_keys_by_episode,
            ),
        )


def test_adapter_rejects_restore_from_another_checkpoint(monkeypatch) -> None:
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = _participants(client, "actor-a")
    monkeypatch.setattr(
        coordination,
        "discover",
        AsyncMock(return_value=participants),
    )
    monkeypatch.setattr(
        coordination,
        "restore",
        AsyncMock(
            return_value={
                "actor-a-environment": {"source_checkpoint_id": "save-other"},
                "actor-a-model": {"source_checkpoint_id": "save-other"},
            }
        ),
    )
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("actor-a", 0),
        client=client,
        auth_token="secret",
    )

    async def exercise() -> None:
        await adapter.discover()
        with pytest.raises(RuntimeError, match="wrong checkpoint"):
            await adapter.restore(
                "restore-1",
                "/checkpoints/step-1",
                [GymCheckpointEpisode("rollout-1", 0)],
                source_checkpoint_id="save-1",
                deadline_ts=10.0,
            )

    asyncio.run(exercise())


def test_adapter_rejects_operations_before_discovery() -> None:
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("actor-a", 0),
        client=object(),
        auth_token="secret",
    )

    with pytest.raises(RuntimeError, match="has not discovered"):
        asyncio.run(adapter.prepare("save-1", deadline_ts=10.0))
