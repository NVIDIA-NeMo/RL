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
import json
import pickle
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointAdapter,
    GymCheckpointEpisode,
    GymCheckpointInstance,
    GymCheckpointPrepareSummary,
    GymCheckpointUnavailable,
)
from nemo_rl.experience.rollout_recovery import (
    PromptGroupPhase,
    PromptGroupRecoveryRecord,
    PromptGroupStatus,
    PromptRef,
    RecoveryTargetLevel,
    RolloutAttemptRecord,
    RolloutAttemptStatus,
    RolloutSiblingRecord,
)

pytestmark = pytest.mark.nemo_gym


@pytest.mark.parametrize("gym_attempt", [0, 1, 17])
def test_rollout_capture_key_matches_gym_episode_identity(gym_attempt: int) -> None:
    """Fail loudly if the string form RL gives a Gym episode drifts from Gym's codec."""
    from nemo_gym.episode_types import EpisodeId

    group = PromptGroupRecoveryRecord(
        group_id="group-7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_ref=PromptRef(sample_id="7", task_name=None),
        task_source=None,
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

    episode = group.gym_episode(0)
    rollout_id, attempt = episode.rollout_id, episode.attempt
    gym_episode = EpisodeId(rollout_id=rollout_id, attempt=attempt)
    capture_key = group.gate_rollout_id(0)

    assert capture_key == gym_episode.capture_key
    assert EpisodeId.from_capture_key(capture_key) == gym_episode


def _participants(client: object, label: str) -> SimpleNamespace:
    return SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name=f"{label}-agent", kind="agent"),
            SimpleNamespace(server_name=f"{label}-model", kind="model"),
        ),
    )


def _commit_reply(
    label: str,
    kind: str,
    episode_ids: list[str],
    *,
    staging_keys: list[str] | None = None,
    staging_keys_by_episode: dict[str, list[str]] | None = None,
    run_episode_ids: list[str] | None = None,
) -> dict[str, object]:
    """A participant's commit reply, for states real Gym participants never produce."""
    del label, kind
    reply: dict[str, object] = {"phase": "committed", "episode_ids": episode_ids}
    if run_episode_ids is not None:
        reply["run_episode_ids"] = run_episode_ids
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


_CONTROL_TOKEN = "control-token"


class _NoResourceSessions:
    """Resources server hooks with no per-session state to save."""

    async def export_session_states(self, session_ids):
        return {}

    async def restore_session_states(self, states):
        return None

    async def retire_session_state(self, session_id):
        return None


def _stateless_resources() -> dict[str, Any]:
    from nemo_gym._checkpoint.resources import ResourcesParticipant

    return {"tools": ResourcesParticipant(_NoResourceSessions(), "stateless")}


def _in_process_gym(
    ledger_dir: Path,
    *,
    restart_only_agent: Any = None,
    resources: dict[str, Any] | None = None,
):
    """A policy model and an agent running Gym's real checkpoint participants.

    ``restart_only_agent`` adds a second agent server with that participant, and
    ``resources`` adds a resources server per entry.

    Control calls go to in-process ASGI apps through Gym's own control routes,
    so the adapter sees exactly the replies Gym's participants build. Server
    names are fixed, as a restarted deployment keeps them.
    """
    import httpx
    from fastapi import FastAPI
    from nemo_gym._checkpoint.agent import AgentSessionParticipant
    from nemo_gym._checkpoint.control import install_participant
    from nemo_gym._checkpoint.model import PolicyModelParticipant
    from nemo_gym.server_utils import BaseServerConfig, ServerClient
    from nemo_gym.token_id_capture.lineage import FileLineageStore
    from omegaconf import OmegaConf

    class _Reply:
        def __init__(self, response: httpx.Response) -> None:
            self.status = response.status_code
            self._content = response.content

        async def read(self) -> bytes:
            return self._content

    class _InProcessClient(ServerClient):
        apps: dict[str, Any]

        async def request(self, server_name, url_path, method, **kwargs):
            transport = httpx.ASGITransport(app=self.apps[server_name])
            async with httpx.AsyncClient(
                transport=transport, base_url=f"http://{server_name}"
            ) as client:
                # Gym's coordination sends each control body pre-encoded as data.
                return _Reply(
                    await client.request(
                        method,
                        url_path,
                        json=kwargs.get("json"),
                        content=kwargs.get("data"),
                        headers=kwargs.get("headers"),
                    )
                )

    class _AgentHooks:
        def __init__(self) -> None:
            self.restored: list[Any] = []

        async def export_agent_sessions(self, session_keys):
            return {key: {"turns": 1} for key in session_keys}

        async def restore_agent_sessions(self, sessions):
            self.restored.extend(sessions)

        async def retire_agent_session(self, session_key):
            return None

    hooks = _AgentHooks()
    agent = AgentSessionParticipant(hooks)
    participants = {
        "policy": PolicyModelParticipant(FileLineageStore(ledger_dir)),
        "agent": agent,
    }
    servers = {
        "policy": {"responses_api_models": {"policy": {}}},
        "agent": {"responses_api_agents": {"agent": {}}},
    }
    if restart_only_agent is not None:
        participants["restart_only"] = restart_only_agent
        servers["restart_only"] = {"responses_api_agents": {"restart_only": {}}}
    for name, participant in (resources or {}).items():
        participants[name] = participant
        servers[name] = {"resources_servers": {name: {}}}
    apps = {}
    for name, participant in participants.items():
        apps[name] = FastAPI()
        install_participant(
            apps[name],
            participant,
            auth_token=_CONTROL_TOKEN,
            instance_name=name,
            lease_grace_seconds=60,
        )
    client = _InProcessClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(servers),
        apps=apps,
    )
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance(shard_name="nemo_gym", replica_index=0),
        client=client,
        auth_token=_CONTROL_TOKEN,
    )
    return adapter, agent, hooks


def test_adapter_drives_gyms_real_participants_through_save_and_restore(
    tmp_path: Path,
) -> None:
    """Fail if a reply Gym's participants build stops matching what RL reads."""
    from nemo_gym.episode_types import EpisodeId

    episode = GymCheckpointEpisode("rollout-1", 0)
    checkpoint_root = tmp_path / "checkpoint"

    async def exercise() -> None:
        # A stateless resources server exports no episode state; the commit
        # must not need any from it.
        source, agent, _ = _in_process_gym(
            tmp_path / "ledger-before", resources=_stateless_resources()
        )
        await source.discover()

        # The agent's own /run episode, inside a replayable verify step: parked for
        # the checkpoint.
        in_verify = asyncio.Event()
        release = asyncio.Event()

        async def legacy_run() -> None:
            async with agent.legacy_run(
                "run:rollout-1", EpisodeId(rollout_id="rollout-1")
            ) as run:
                await run.boundary({"next": "verify", "response": {}})
                async with run.step("replay"):
                    in_verify.set()
                    await release.wait()

        run = asyncio.create_task(legacy_run())
        await asyncio.wait_for(in_verify.wait(), timeout=5)

        prepared = await source.prepare("save-1", deadline_ts=time.time() + 5)
        assert prepared.prepared, prepared.blockers()
        summary = await source.commit(
            "save-1", checkpoint_root, [episode], deadline_ts=time.time() + 5
        )
        await source.resume("save-1", deadline_ts=time.time() + 5)
        release.set()
        await asyncio.wait_for(run, timeout=5)

        # The agent's own /run episode is exported; nothing else is live.
        assert summary.exported_episodes == (episode,)
        assert summary.unowned_live_episodes == ()
        assert summary.staging_keys == ()

        # A fresh deployment, as after a crash.
        restored, restored_agent, hooks = _in_process_gym(
            tmp_path / "ledger-after", resources=_stateless_resources()
        )
        await restored.discover()
        await restored.restore(
            "restore-1",
            checkpoint_root,
            [episode],
            source_checkpoint_id="save-1",
            deadline_ts=time.time() + 5,
        )
        await restored.resume("restore-1", deadline_ts=time.time() + 5)

        replacement = EpisodeId(rollout_id="rollout-1", attempt=1)
        assert await restored_agent.restored_pending() == [replacement]
        assert [(s.session_key, s.episode_id, s.session) for s in hooks.restored] == [
            ("run:rollout-1", replacement, {"turns": 1})
        ]

        # The restore refuses the attempt it replaced, and a retire refuses its
        # rollout, on every participant until RL forgets them.
        await restored.retire(
            "retire-1",
            [GymCheckpointEpisode("rollout-2", 0)],
            deadline_ts=time.time() + 5,
        )
        assert await _retired_rollouts(restored) == {"policy": 2, "agent": 2}
        await restored.forget(
            "forget-1", ["rollout-1", "rollout-2"], deadline_ts=time.time() + 5
        )
        assert await _retired_rollouts(restored) == {"policy": 0, "agent": 0}

    asyncio.run(exercise())


async def _retired_rollouts(adapter: GymCheckpointAdapter) -> dict[str, int]:
    """Each participant's count of rollouts it still refuses, from its /status."""
    counts = {}
    for server_name in ("policy", "agent"):
        reply = await adapter._client.request(
            server_name,
            "/ng-control/v1/checkpoint/status",
            "GET",
            headers={"Authorization": f"Bearer {_CONTROL_TOKEN}"},
        )
        assert reply.status == 200
        counts[server_name] = json.loads(await reply.read())["retired_rollouts"]
    return counts


def test_a_rollout_gym_cannot_capture_is_reported_as_a_restart(tmp_path: Path) -> None:
    """RL reads the restarts of Gym's real coordination and leaves them out."""
    from nemo_gym._checkpoint.agent import RestartOnlyAgentParticipant

    restart = GymCheckpointEpisode("rollout-1", 0)

    async def exercise() -> None:
        restart_only = RestartOnlyAgentParticipant()
        adapter, _, _ = _in_process_gym(tmp_path, restart_only_agent=restart_only)
        await adapter.discover()

        release = asyncio.Event()
        in_call = asyncio.Event()

        async def call() -> None:
            async with restart_only.track(restart.capture_key):
                in_call.set()
                await release.wait()

        tracked = asyncio.create_task(call())
        await asyncio.wait_for(in_call.wait(), timeout=5)

        # Gym refuses a commit that names the restart; the controller resumes
        # and prepares again under a new checkpoint ID.
        await adapter.prepare("save-0", deadline_ts=time.time() + 5)
        with pytest.raises(GymCheckpointUnavailable, match="restart_in_scope"):
            await adapter.commit(
                "save-0", tmp_path / "refused", [restart], deadline_ts=time.time() + 5
            )
        await adapter.resume("save-0", deadline_ts=time.time() + 5)

        result = await adapter.prepare("save-1", deadline_ts=time.time() + 5)
        summary = GymCheckpointPrepareSummary.from_result(result)
        assert summary.prepared
        assert summary.restarts == (restart,)
        committed = await adapter.commit(
            "save-1", tmp_path / "checkpoint", [], deadline_ts=time.time() + 5
        )
        await adapter.resume("save-1", deadline_ts=time.time() + 5)
        release.set()
        await asyncio.wait_for(tracked, timeout=5)

        assert committed.exported_episodes == ()

    asyncio.run(exercise())


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


def _commit_model_reply(monkeypatch, model_reply: dict[str, object]):
    """Commit two agent /run episodes with the given policy-model reply."""
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name="agent", kind="agent"),
            SimpleNamespace(server_name="model", kind="model"),
        ),
    )
    episodes = ["rollout-1", "rollout-2-a1"]
    monkeypatch.setattr(coordination, "discover", AsyncMock(return_value=participants))
    monkeypatch.setattr(
        coordination,
        "commit",
        AsyncMock(
            return_value={
                "agent": _commit_reply(
                    "deployment", "agent", episodes, run_episode_ids=episodes
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
    with pytest.raises(ValueError, match="attribute exactly the committed staging"):
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


def _commit_with_agent_session(monkeypatch, *, own_run_episode_ids: list[str]):
    """Commit one agent session, as Gym's agent participant replies for it."""
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name="model", kind="model"),
            SimpleNamespace(server_name="agent", kind="agent"),
        ),
    )
    monkeypatch.setattr(coordination, "discover", AsyncMock(return_value=participants))
    monkeypatch.setattr(
        coordination,
        "commit",
        AsyncMock(
            return_value={
                "model": _commit_reply(
                    "deployment",
                    "model",
                    [],
                    staging_keys=[],
                    staging_keys_by_episode={},
                ),
                "agent": {
                    **_commit_reply("deployment", "agent", ["rollout-1"]),
                    "run_episode_ids": own_run_episode_ids,
                },
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
            "/unused",
            [GymCheckpointEpisode("rollout-1", 0)],
            deadline_ts=10.0,
        )

    return asyncio.run(exercise())


def test_adapter_assigns_a_relayed_run_episode_to_its_agent(monkeypatch) -> None:
    """A legacy_agent environment server relays /run: the agent's own episode."""
    summary = _commit_with_agent_session(monkeypatch, own_run_episode_ids=["rollout-1"])

    assert summary.exported_episodes == (GymCheckpointEpisode("rollout-1", 0),)
    assert summary.unowned_live_episodes == ()


def test_adapter_reports_an_agent_session_that_is_not_its_own_run_episode(
    monkeypatch,
) -> None:
    """A parked session nobody can restore must not be mistaken for an export."""
    summary = _commit_with_agent_session(monkeypatch, own_run_episode_ids=[])

    assert summary.exported_episodes == ()
    assert summary.unowned_live_episodes == (GymCheckpointEpisode("rollout-1", 0),)


def test_adapter_rejects_a_run_episode_the_agent_did_not_export(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="names /run episodes it did not export"):
        _commit_with_agent_session(
            monkeypatch, own_run_episode_ids=["rollout-1", "rollout-2"]
        )


def _commit_replies(monkeypatch, members: list[tuple[str, str]], replies: dict):
    """Commit ``rollout-1`` against hand-written replies from ``members``."""
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = SimpleNamespace(
        client=client,
        members=tuple(
            SimpleNamespace(server_name=name, kind=kind) for name, kind in members
        ),
    )
    monkeypatch.setattr(coordination, "discover", AsyncMock(return_value=participants))
    monkeypatch.setattr(coordination, "commit", AsyncMock(return_value=replies))
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("deployment", 0),
        client=client,
        auth_token="secret",
    )

    async def exercise():
        await adapter.discover()
        return await adapter.commit(
            "save-1",
            "/unused",
            [GymCheckpointEpisode("rollout-1", 0)],
            deadline_ts=10.0,
        )

    return asyncio.run(exercise())


_EMPTY_MODEL_REPLY = _commit_reply(
    "deployment", "model", [], staging_keys=[], staging_keys_by_episode={}
)


def test_adapter_rejects_a_commit_from_an_unexpected_participant_set(
    monkeypatch,
) -> None:
    with pytest.raises(RuntimeError, match="unexpected participant set"):
        _commit_replies(
            monkeypatch,
            [("agent", "agent"), ("model", "model")],
            {"model": _EMPTY_MODEL_REPLY, "stranger": _EMPTY_MODEL_REPLY},
        )


def test_adapter_rejects_an_export_outside_the_requested_scope(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="outside the requested scope"):
        _commit_replies(
            monkeypatch,
            [("agent", "agent"), ("model", "model")],
            {
                "agent": _commit_reply(
                    "deployment",
                    "agent",
                    ["rollout-1", "rollout-9"],
                    run_episode_ids=["rollout-1"],
                ),
                "model": _EMPTY_MODEL_REPLY,
            },
        )


def test_adapter_rejects_a_run_episode_two_agents_claim(monkeypatch) -> None:
    """Restoring it on either agent would leave the other's session behind."""
    claim = _commit_reply(
        "deployment", "agent", ["rollout-1"], run_episode_ids=["rollout-1"]
    )
    with pytest.raises(RuntimeError, match="multiple agent participants"):
        _commit_replies(
            monkeypatch,
            [("agent-a", "agent"), ("agent-b", "agent"), ("model", "model")],
            {"agent-a": claim, "agent-b": claim, "model": _EMPTY_MODEL_REPLY},
        )


def test_adapter_refuses_environment_server_participants(monkeypatch) -> None:
    """Commits assume an agent owns every episode; an environment server would own its own."""
    from nemo_gym._checkpoint import coordination

    client = object()
    participants = SimpleNamespace(
        client=client,
        members=(
            SimpleNamespace(server_name="episodes", kind="environment"),
            SimpleNamespace(server_name="model", kind="model"),
        ),
    )
    monkeypatch.setattr(coordination, "discover", AsyncMock(return_value=participants))
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance("deployment", 0),
        client=client,
        auth_token="secret",
    )

    with pytest.raises(
        RuntimeError, match="environment-server checkpoint participants"
    ):
        asyncio.run(adapter.discover())
    # Nothing was discovered, so no later checkpoint can run against it.
    with pytest.raises(RuntimeError, match="has not discovered"):
        asyncio.run(adapter.prepare("save-1", deadline_ts=10.0))


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
                "actor-a-agent": {"source_checkpoint_id": "save-other"},
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


@pytest.mark.parametrize(
    "operation",
    ["discover", "prepare", "renew", "retire", "commit", "restore", "resume"],
)
def test_a_failed_checkpoint_control_call_crosses_ray_typed(
    monkeypatch, operation: str
) -> None:
    """Gym's CoordinationError cannot be pickled and reaches RL untyped."""
    from nemo_gym._checkpoint import coordination

    failure = coordination.CoordinationError(
        operation, {"actor-a-agent": "HTTP 409 deadline_exceeded"}
    )
    monkeypatch.setattr(coordination, operation, AsyncMock(side_effect=failure))
    adapter = GymCheckpointAdapter(
        instance=GymCheckpointInstance(shard_name="actor-a", replica_index=0),
        client=object(),
        auth_token="secret",
    )
    if operation != "discover":
        adapter._participants = _participants(object(), "actor-a")
    episodes = [GymCheckpointEpisode("rollout-1", 0)]
    calls = {
        "discover": lambda: adapter.discover(),
        "restore": lambda: adapter.restore(
            "restore-1",
            "/unused",
            episodes,
            source_checkpoint_id="save-1",
            deadline_ts=1.0,
        ),
        "prepare": lambda: adapter.prepare("save-1", deadline_ts=1.0),
        "renew": lambda: adapter.renew("save-1", deadline_ts=1.0),
        "retire": lambda: adapter.retire("save-1", episodes, deadline_ts=1.0),
        "commit": lambda: adapter.commit(
            "save-1", "/unused", episodes, deadline_ts=1.0
        ),
        "resume": lambda: adapter.resume("save-1", deadline_ts=1.0),
    }

    with pytest.raises(GymCheckpointUnavailable, match="deadline_exceeded") as raised:
        asyncio.run(calls[operation]())

    restored = pickle.loads(pickle.dumps(raised.value))
    assert type(restored) is GymCheckpointUnavailable
    assert str(restored) == str(raised.value)


def test_adapter_names_the_restart_only_servers_gym_reports(tmp_path: Path) -> None:
    """Read the field each participant kind reports, through Gym's own status route."""
    from nemo_gym._checkpoint.agent import RestartOnlyAgentParticipant
    from nemo_gym._checkpoint.resources import ResourcesParticipant

    restart_only_agent = RestartOnlyAgentParticipant()
    if "restart_only" not in restart_only_agent.status_extra():
        # Gyms before NVIDIA-NeMo/Gym#4957 do not report it; the adapter then
        # cannot tell, which is checked below.
        restart_only_agent.status_extra = lambda: {"restart_only": True}

    async def exercise() -> tuple[list[str], list[str]]:
        adapter, _, _ = _in_process_gym(
            tmp_path / "ledger",
            restart_only_agent=restart_only_agent,
            resources={
                "restarting_tools": ResourcesParticipant(
                    _NoResourceSessions(), "restart_only"
                ),
                "exported_tools": ResourcesParticipant(
                    _NoResourceSessions(), "exported"
                ),
                "stateless_tools": ResourcesParticipant(
                    _NoResourceSessions(), "stateless"
                ),
            },
        )
        await adapter.discover()
        named = await adapter.restart_only_servers()

        older_agent = RestartOnlyAgentParticipant()
        older_agent.status_extra = lambda: {}
        older, _, _ = _in_process_gym(
            tmp_path / "ledger-older", restart_only_agent=older_agent
        )
        await older.discover()
        return named, await older.restart_only_servers()

    named, older = asyncio.run(exercise())

    # The session-capable agent and the policy model are not restart-only.
    assert named == ["restart_only", "restarting_tools"]
    assert older == []
