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

import asyncio
import json
import pickle
import socket
import time
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from nemo_rl.environments.nemo_gym_checkpoint import (
    ActorCheckpointError,
    ActorCheckpointPhase,
    AgentCompletionReceipt,
    CheckpointCapabilities,
    CheckpointParticipant,
    CheckpointParticipantError,
    CheckpointParticipantKind,
    CheckpointTransport,
    ExecutionIdentity,
    LiveExecutionRegistry,
    NemoGymCheckpointCoordinator,
    discover_checkpoint_participants,
    load_checkpoint_manifest,
)


def _capabilities(
    component: str,
    name: str,
    *,
    checkpoint_mode: str = "export_restore",
    instance_role: str | None = None,
    multi_process_mode: str = "single_worker",
) -> dict[str, Any]:
    return {
        "component": component,
        "name": name,
        "schema_version": 1,
        "admission_states": (
            ["accepting", "draining", "paused"]
            if component == "responses_api_models"
            else ["accepting"]
        ),
        "checkpoint_mode": checkpoint_mode,
        "concurrency_contract": "stateless",
        "multi_process": {
            "mode": multi_process_mode,
            "num_workers": 2 if multi_process_mode == "unmanaged" else 1,
        },
        "instance_role": instance_role,
    }


def _participant(
    participant_id: str,
    kind: CheckpointParticipantKind,
) -> CheckpointParticipant:
    component = {
        CheckpointParticipantKind.POLICY_MODEL: "responses_api_models",
        CheckpointParticipantKind.AGENT: "responses_api_agents",
        CheckpointParticipantKind.RESOURCES: "resources_servers",
    }[kind]
    return CheckpointParticipant(
        participant_id=participant_id,
        name=participant_id,
        base_url=f"http://{participant_id}.test",
        kind=kind,
        capabilities=CheckpointCapabilities.from_mapping(
            _capabilities(
                component,
                participant_id,
                instance_role=(
                    "policy" if kind == CheckpointParticipantKind.POLICY_MODEL else None
                ),
            )
        ),
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.json_bodies: list[dict[str, Any] | None] = []
        self.capabilities: dict[str, dict[str, Any]] = {}
        self.fail_once: dict[tuple[str, str], CheckpointParticipantError] = {}
        self.responses: dict[tuple[str, str], dict[str, Any]] = {}

    async def request(
        self,
        participant: CheckpointParticipant,
        method: str,
        endpoint: str,
        *,
        deadline_ts: float,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del method, deadline_ts, params
        key = (participant.participant_id, endpoint)
        self.calls.append(key)
        self.json_bodies.append(json_body)
        error = self.fail_once.pop(key, None)
        if error is not None:
            raise error
        if key in self.responses:
            return self.responses[key]
        if endpoint.endswith("/capabilities"):
            return self.capabilities[participant.participant_id]
        if endpoint.endswith("/status"):
            return {"state": "paused"}
        if endpoint.endswith("/commit") and json_body is not None:
            checkpoint_dir = Path(json_body["checkpoint_dir"])
            subdirectory = {
                CheckpointParticipantKind.POLICY_MODEL: "model-ledger",
                CheckpointParticipantKind.AGENT: "agent",
                CheckpointParticipantKind.RESOURCES: "resources",
            }[participant.kind]
            output = checkpoint_dir / subdirectory / participant.name / "manifest.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"participant": participant.participant_id}))
        return {"state": "ok", "participant": participant.participant_id}


def _global_config() -> dict[str, Any]:
    return {
        "policy": {
            "responses_api_models": {
                "implementation": {"host": "model.test", "port": 8001}
            }
        },
        "judge": {
            "responses_api_models": {
                "implementation": {"host": "judge.test", "port": 8002}
            }
        },
        "agent": {
            "responses_api_agents": {
                "implementation": {"host": "agent.test", "port": 8003}
            }
        },
        "resources": {
            "resources_servers": {
                "implementation": {"host": "resources.test", "port": 8004}
            }
        },
        "head_server": {"host": "head.test", "port": 8000},
    }


@pytest.mark.asyncio
async def test_discovery_filters_auxiliary_and_stateless_servers() -> None:
    fake = _FakeTransport()
    fake.capabilities = {
        "policy": _capabilities(
            "responses_api_models", "policy", instance_role="policy"
        ),
        "judge": _capabilities(
            "responses_api_models", "judge", instance_role="auxiliary"
        ),
        "agent": _capabilities("responses_api_agents", "agent"),
        "resources": _capabilities(
            "resources_servers", "resources", checkpoint_mode="stateless"
        ),
    }

    participants = await discover_checkpoint_participants(
        _global_config(),
        fake,  # type: ignore[arg-type]
        deadline_ts=time.time() + 10,
    )

    assert [
        (participant.participant_id, participant.kind) for participant in participants
    ] == [
        ("agent", CheckpointParticipantKind.AGENT),
        ("policy", CheckpointParticipantKind.POLICY_MODEL),
    ]
    assert len(fake.calls) == 4


@pytest.mark.asyncio
async def test_discovery_rejects_unmanaged_workers() -> None:
    fake = _FakeTransport()
    fake.capabilities = {
        "policy": _capabilities(
            "responses_api_models",
            "policy",
            instance_role="policy",
            multi_process_mode="unmanaged",
        ),
        "judge": _capabilities(
            "responses_api_models", "judge", instance_role="auxiliary"
        ),
        "agent": _capabilities(
            "responses_api_agents", "agent", checkpoint_mode="stateless"
        ),
        "resources": _capabilities(
            "resources_servers", "resources", checkpoint_mode="stateless"
        ),
    }

    with pytest.raises(ValueError, match="unmanaged multi-process"):
        await discover_checkpoint_participants(
            _global_config(),
            fake,  # type: ignore[arg-type]
            deadline_ts=time.time() + 10,
        )


def test_live_registry_validates_identity_and_restores_frozen_membership() -> None:
    registry = LiveExecutionRegistry()
    row = {"_ng_rollout_id": "rollout-1", "_ng_attempt_index": 3}
    identity = registry.register(row)
    assert identity == ExecutionIdentity("rollout-1", 3)
    registry.mark_terminal(identity)
    frozen = registry.freeze("checkpoint-1")
    assert frozen[0].to_dict()["state"] == "terminal"
    with pytest.raises(RuntimeError, match="dispatch is frozen"):
        registry.register({"_ng_rollout_id": "other", "_ng_attempt_index": 0})
    registry.release(identity)
    registry.unfreeze("checkpoint-1")

    registry.restore("restore-1", [execution.to_dict() for execution in frozen])
    assert registry.status()["frozen_membership"] == 1
    registry.unfreeze("restore-1")
    assert registry.status()["frozen_checkpoint_id"] is None


@pytest.mark.parametrize(
    "row,match",
    [
        ({"_ng_attempt_index": 0}, "_ng_rollout_id"),
        (
            {"_ng_rollout_id": "rollout", "_ng_attempt_index": -1},
            "_ng_attempt_index",
        ),
        (
            {"_ng_rollout_id": "rollout", "_ng_attempt_index": True},
            "_ng_attempt_index",
        ),
    ],
)
def test_live_registry_rejects_missing_or_invalid_identity(
    row: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        LiveExecutionRegistry().register(row)


@pytest.mark.asyncio
async def test_transport_preserves_bearer_status_and_gym_error_code() -> None:
    from aiohttp import web

    seen_authorization: list[str | None] = []

    async def rejected(request: web.Request) -> web.Response:
        seen_authorization.append(request.headers.get("Authorization"))
        return web.json_response(
            {"error": {"code": "checkpoint_conflict", "detail": "busy"}},
            status=409,
        )

    app = web.Application()
    app.router.add_post("/reject", rejected)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.SockSite(runner, _listening_socket())
    await site.start()
    participant = _participant("policy", CheckpointParticipantKind.POLICY_MODEL)
    participant = CheckpointParticipant(
        participant.participant_id,
        participant.name,
        _site_url(site),
        participant.kind,
        participant.capabilities,
    )
    try:
        with pytest.raises(CheckpointParticipantError) as exc_info:
            await CheckpointTransport("secret").request(
                participant,
                "POST",
                "/reject",
                deadline_ts=time.time() + 5,
            )
        assert exc_info.value.status == 409
        assert exc_info.value.gym_error_code == "checkpoint_conflict"
        assert exc_info.value.participant == "policy"
        assert seen_authorization == ["Bearer secret"]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_transport_refuses_an_expired_deadline_before_io() -> None:
    participant = _participant("policy", CheckpointParticipantKind.POLICY_MODEL)
    with pytest.raises(
        CheckpointParticipantError,
        match="absolute checkpoint deadline expired",
    ) as exc_info:
        await CheckpointTransport("secret").request(
            participant,
            "GET",
            "/never",
            deadline_ts=time.time() - 1,
        )
    assert exc_info.value.gym_error_code == "deadline_exceeded"


def test_participant_error_pickling_preserves_structured_fields() -> None:
    error = CheckpointParticipantError(
        "policy",
        "/prepare",
        "busy",
        status=409,
        gym_error_code="checkpoint_conflict",
    )
    restored = pickle.loads(pickle.dumps(error))
    assert restored.participant == "policy"
    assert restored.endpoint == "/prepare"
    assert restored.status == 409
    assert restored.gym_error_code == "checkpoint_conflict"

    actor_error = ActorCheckpointError(
        "checkpoint-1",
        ActorCheckpointPhase.PREPARING,
        ["agent"],
        error,
    )
    restored_actor_error = pickle.loads(pickle.dumps(actor_error))
    assert restored_actor_error.phase == ActorCheckpointPhase.PREPARING
    assert restored_actor_error.completed_participants == ("agent",)
    assert "checkpoint_conflict" in restored_actor_error.cause_detail


@pytest.mark.parametrize("attempt_index", [0, 1])
@pytest.mark.asyncio
async def test_delivery_acknowledges_exact_completion_receipt_before_release(
    attempt_index: int,
) -> None:
    coordinator = NemoGymCheckpointCoordinator("secret")
    fake = _FakeTransport()
    coordinator._transport = fake  # type: ignore[assignment]
    coordinator._participants = [_participant("agent", CheckpointParticipantKind.AGENT)]
    identity = coordinator.registry.register(
        {
            "_ng_rollout_id": "rollout-1",
            "_ng_attempt_index": attempt_index,
        }
    )
    coordinator.registry.mark_terminal(identity)
    receipt = {
        "rollout_id": "rollout-1",
        "attempt_index": attempt_index,
        "execution_generation": attempt_index + 1,
        "result_identity": f"result-{attempt_index}",
        "result_digest": f"{attempt_index + 1:064x}",
    }
    fake.responses[("agent", "/ng-control/v1/agent-checkpoint/status")] = {
        "completed_unacknowledged_attempts": [{"completion_receipt": receipt}]
    }
    fake.responses[("agent", "/ng-control/v1/agent-checkpoint/acknowledge")] = {
        "acknowledged": True,
        "idempotent": False,
    }

    await coordinator.release(identity, agent_name="agent")

    assert coordinator.registry.status()["live"] == 0
    assert fake.calls[-2:] == [
        ("agent", "/ng-control/v1/agent-checkpoint/status"),
        ("agent", "/ng-control/v1/agent-checkpoint/acknowledge"),
    ]
    assert fake.json_bodies[-1] == receipt


@pytest.mark.asyncio
async def test_delivery_receipt_mismatch_retains_terminal_membership() -> None:
    coordinator = NemoGymCheckpointCoordinator("secret")
    fake = _FakeTransport()
    coordinator._transport = fake  # type: ignore[assignment]
    coordinator._participants = [_participant("agent", CheckpointParticipantKind.AGENT)]
    identity = coordinator.registry.register(
        {"_ng_rollout_id": "rollout-1", "_ng_attempt_index": 0}
    )
    coordinator.registry.mark_terminal(identity)
    fake.responses[("agent", "/ng-control/v1/agent-checkpoint/status")] = {
        "completed_unacknowledged_attempts": []
    }

    with pytest.raises(
        CheckpointParticipantError,
        match="expected exactly one completion receipt",
    ):
        await coordinator.release(identity, agent_name="agent")

    assert coordinator.registry.status()["terminal_unreleased"] == 1
    assert not any(endpoint.endswith("/acknowledge") for _, endpoint in fake.calls)


@pytest.mark.asyncio
async def test_prepare_fails_closed_for_yielded_unacknowledged_result() -> None:
    coordinator, fake = _coordinator_with_all_participants()
    identity = coordinator.registry.register(
        {"_ng_rollout_id": "rollout-1", "_ng_attempt_index": 0}
    )
    coordinator.registry.mark_terminal(identity)

    with pytest.raises(ActorCheckpointError, match="remain unacknowledged"):
        await coordinator.prepare("checkpoint-1", time.time() + 10)

    assert coordinator.status("checkpoint-1")["phase"] == "preparing"
    assert coordinator.status("checkpoint-1")["registry"]["frozen_checkpoint_id"] == (
        "checkpoint-1"
    )
    assert fake.calls == []


def test_completion_receipt_accepts_current_and_final_result_id_names() -> None:
    common = {
        "rollout_id": "rollout-1",
        "attempt_index": 1,
        "execution_generation": 2,
        "result_digest": "a" * 64,
    }
    current = AgentCompletionReceipt.from_mapping(
        {**common, "result_identity": "result-current"}
    )
    final = AgentCompletionReceipt.from_mapping({**common, "result_id": "result-final"})

    assert current.to_request()["result_identity"] == "result-current"
    assert final.to_request()["result_id"] == "result-final"


@pytest.mark.asyncio
async def test_prepare_commit_and_resume_are_idempotent_and_ordered(
    tmp_path: Path,
) -> None:
    coordinator, fake = _coordinator_with_all_participants()
    identity = coordinator.registry.register(
        {"_ng_rollout_id": "rollout-1", "_ng_attempt_index": 0}
    )

    first_prepare = await coordinator.prepare("checkpoint-1", time.time() + 10)
    second_prepare = await coordinator.prepare("checkpoint-1", time.time() + 10)
    assert first_prepare == second_prepare
    assert first_prepare["phase"] == "prepared"
    assert coordinator.status("checkpoint-1")["registry"]["frozen_membership"] == 1

    first_commit = await coordinator.commit("checkpoint-1", tmp_path)
    second_commit = await coordinator.commit("checkpoint-1", tmp_path)
    assert first_commit == second_commit
    manifest = load_checkpoint_manifest(tmp_path)
    assert manifest["checkpoint_id"] == "checkpoint-1"
    assert len(manifest["files"]) == 3

    first_resume = await coordinator.resume("checkpoint-1")
    calls_after_resume = len(fake.calls)
    second_resume = await coordinator.resume("checkpoint-1")
    assert second_resume == first_resume
    assert len(fake.calls) == calls_after_resume
    resume_calls = [
        participant
        for participant, endpoint in fake.calls
        if endpoint.endswith("/resume")
    ]
    assert resume_calls[-3:] == ["policy", "resources", "agent"]
    assert coordinator.status("checkpoint-1")["phase"] == "idle"

    coordinator.registry.release(identity)
    restored = await coordinator.restore("restore-1", tmp_path)
    assert restored["phase"] == "restored_paused"
    assert coordinator.status("restore-1")["registry"]["frozen_membership"] == 1
    await coordinator.resume("restore-1")
    assert coordinator.status("restore-1")["phase"] == "idle"

    first_file = tmp_path / next(iter(manifest["files"]))
    first_file.write_text("corrupted")
    with pytest.raises(ValueError, match="missing or corrupted"):
        load_checkpoint_manifest(tmp_path)


@pytest.mark.asyncio
async def test_partial_prepare_can_abort_only_touched_participants_in_order() -> None:
    coordinator, fake = _coordinator_with_all_participants()
    fake.fail_once[("policy", "/ng-control/v1/model-admission/pause")] = (
        CheckpointParticipantError(
            "policy",
            "/ng-control/v1/model-admission/pause",
            "unavailable",
            status=503,
            gym_error_code="unavailable",
        )
    )

    with pytest.raises(ActorCheckpointError) as exc_info:
        await coordinator.prepare("checkpoint-1", time.time() + 10)
    assert exc_info.value.completed_participants == ("agent", "policy")

    await coordinator.abort("checkpoint-1")
    resume_calls = [
        participant
        for participant, endpoint in fake.calls
        if endpoint.endswith("/resume")
    ]
    assert resume_calls == ["policy", "agent"]
    assert coordinator.status("checkpoint-1")["phase"] == "idle"


def _coordinator_with_all_participants() -> tuple[
    NemoGymCheckpointCoordinator,
    _FakeTransport,
]:
    coordinator = NemoGymCheckpointCoordinator("secret")
    fake = _FakeTransport()
    coordinator._transport = fake  # type: ignore[assignment]
    coordinator._participants = [
        _participant("policy", CheckpointParticipantKind.POLICY_MODEL),
        _participant("agent", CheckpointParticipantKind.AGENT),
        _participant("resources", CheckpointParticipantKind.RESOURCES),
    ]
    return coordinator, fake


def _listening_socket() -> socket.socket:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    return sock


def _site_url(site: Any) -> str:
    server = site._server
    assert server is not None
    host, port = server.sockets[0].getsockname()[:2]
    return f"http://{host}:{port}"


@pytest.mark.asyncio
async def test_embedded_gym_model_participant_prepare_and_abort() -> None:
    import uvicorn
    from fastapi import FastAPI
    from nemo_gym._checkpoint import (
        AdmissionLimiter,
        ControlCapabilities,
        ControlFence,
        MultiProcessCapability,
        install_control_plane,
        install_model_admission,
    )

    token = "embedded-secret"
    app = FastAPI()
    fence = ControlFence()
    limiter = AdmissionLimiter()
    capabilities = ControlCapabilities(
        component="responses_api_models",
        name="policy",
        admission_states=["accepting", "draining", "paused"],
        checkpoint_mode="export_restore",
        multi_process=MultiProcessCapability(mode="single_worker"),
        instance_role="policy",
    )
    install_control_plane(app, capabilities=capabilities, fence=fence)
    install_model_admission(
        app,
        limiter=limiter,
        fence=fence,
        instance_role="policy",
        auth_token=token,
    )
    sock = _listening_socket()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    host, port = sock.getsockname()[:2]
    config = {
        "policy": {
            "responses_api_models": {"implementation": {"host": host, "port": port}}
        }
    }
    coordinator = NemoGymCheckpointCoordinator(token)
    try:
        await coordinator.discover(config, deadline_ts=time.time() + 5)
        prepared = await coordinator.prepare("checkpoint-1", time.time() + 5)
        assert prepared["phase"] == "prepared"
        assert limiter.counts()["state"] == "paused"
        await coordinator.abort("checkpoint-1")
        assert limiter.counts()["state"] == "accepting"
    finally:
        server.should_exit = True
        await server_task


@pytest.mark.asyncio
async def test_integrated_gym_delivery_checkpoint_and_receipt_replay(
    tmp_path: Path,
) -> None:
    import uvicorn
    from fastapi import FastAPI
    from nemo_gym import _checkpoint as gym_checkpoint

    if not hasattr(gym_checkpoint, "AgentAcknowledgeRequest"):
        pytest.skip("requires the integrated Gym completion-acknowledgment contract")

    from nemo_gym._checkpoint import (
        RESOURCE_REQUEST_ID_HEADER,
        AdmissionLimiter,
        AgentCheckpointParticipant,
        ControlCapabilities,
        ControlFence,
        MultiProcessCapability,
        ResourcesCheckpointParticipant,
        ResourceSnapshot,
        install_agent_checkpoint,
        install_control_plane,
        install_model_admission,
        install_model_checkpoint,
        install_resources_checkpoint,
    )
    from nemo_gym.rollout_correlation import (
        ATTEMPT_INDEX_HEADER,
        ROLLOUT_ID_HEADER,
    )
    from nemo_gym.token_id_capture.lineage import FileLineageStore

    token = "integrated-secret"

    def capabilities(
        component: str,
        name: str,
        *,
        instance_role: str | None = None,
    ) -> ControlCapabilities:
        return ControlCapabilities(
            component=component,
            name=name,
            admission_states=(
                ["accepting", "draining", "paused"]
                if component == "responses_api_models"
                else ["accepting"]
            ),
            checkpoint_mode="export_restore",
            concurrency_contract=(
                "serialized_per_session"
                if component != "responses_api_models"
                else "stateless"
            ),
            multi_process=MultiProcessCapability(
                mode="single_worker",
                num_workers=1,
            ),
            instance_role=instance_role,
        )

    model_app = FastAPI()
    model_fence = ControlFence()
    limiter = AdmissionLimiter()
    ledger = FileLineageStore(tmp_path / "ledger")
    install_control_plane(
        model_app,
        capabilities=capabilities(
            "responses_api_models",
            "policy",
            instance_role="policy",
        ),
        fence=model_fence,
    )
    install_model_admission(
        model_app,
        limiter=limiter,
        fence=model_fence,
        instance_role="policy",
        auth_token=token,
    )
    install_model_checkpoint(
        model_app,
        fence=model_fence,
        limiter=limiter,
        ledger_provider=lambda: ledger,
        file_ledger_root_provider=lambda: ledger.checkpoint_root,
        instance_role="policy",
        server_name="policy",
        auth_token=token,
    )

    agent_app = FastAPI()
    agent_fence = ControlFence()
    agent_participant = AgentCheckpointParticipant("agent")
    install_control_plane(
        agent_app,
        capabilities=capabilities("responses_api_agents", "agent"),
        fence=agent_fence,
    )
    install_agent_checkpoint(
        agent_app,
        participant=agent_participant,
        fence=agent_fence,
        auth_token=token,
    )

    resource_state: dict[tuple[str, int], dict[str, int]] = {}
    resource_calls = 0

    async def export_resource_state(
        rollout_id: str,
        attempt_index: int,
    ) -> dict[str, Any]:
        return dict(resource_state[(rollout_id, attempt_index)])

    async def restore_resource_states(snapshots: list[ResourceSnapshot]) -> None:
        resource_state.clear()
        resource_state.update(
            {
                (snapshot.rollout_id, snapshot.attempt_index): dict(snapshot.state)
                for snapshot in snapshots
            }
        )

    resources_participant = ResourcesCheckpointParticipant(
        export_state=export_resource_state,
        restore_states=restore_resource_states,
    )
    resources_app = FastAPI()
    resources_fence = ControlFence()
    install_control_plane(
        resources_app,
        capabilities=capabilities("resources_servers", "resources"),
        fence=resources_fence,
    )

    @resources_app.post("/seed")
    async def seed_resource() -> dict[str, int]:
        nonlocal resource_calls
        resource_calls += 1
        resource_state[("resource-rollout", 0)] = {"value": resource_calls}
        return {"value": resource_calls}

    install_resources_checkpoint(
        resources_app,
        participant=resources_participant,
        fence=resources_fence,
        auth_token=token,
        server_name="resources",
        route_kind=lambda path, method: (
            "start" if path == "/seed" and method == "POST" else None
        ),
    )

    apps = [model_app, agent_app, resources_app]
    sockets = [_listening_socket() for _ in apps]
    servers = [
        uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        for app in apps
    ]
    server_tasks = [
        asyncio.create_task(server.serve(sockets=[sock]))
        for server, sock in zip(servers, sockets)
    ]
    while not all(server.started for server in servers):
        await asyncio.sleep(0.01)
    urls = [
        f"http://{sock.getsockname()[0]}:{sock.getsockname()[1]}" for sock in sockets
    ]
    config = {
        "policy": {
            "responses_api_models": {
                "implementation": {
                    "host": sockets[0].getsockname()[0],
                    "port": sockets[0].getsockname()[1],
                }
            }
        },
        "agent": {
            "responses_api_agents": {
                "implementation": {
                    "host": sockets[1].getsockname()[0],
                    "port": sockets[1].getsockname()[1],
                }
            }
        },
        "resources": {
            "resources_servers": {
                "implementation": {
                    "host": sockets[2].getsockname()[0],
                    "port": sockets[2].getsockname()[1],
                }
            }
        },
    }
    coordinator = NemoGymCheckpointCoordinator(token)
    try:
        await coordinator.discover(config, deadline_ts=time.time() + 10)

        for attempt_index in (0, 1):
            execution = await agent_participant.begin(
                "delivery-rollout",
                attempt_index,
                task=asyncio.current_task(),
            )
            await agent_participant.finish(
                execution,
                outcome="completed",
                result={"id": f"result-{attempt_index}", "reward": 1.0},
            )
            identity = coordinator.registry.register(
                {
                    "_ng_rollout_id": "delivery-rollout",
                    "_ng_attempt_index": attempt_index,
                }
            )
            coordinator.registry.mark_terminal(identity)
            await coordinator.release(identity, agent_name="agent")
        assert agent_participant.status()["acknowledged_completed"] == 2
        assert coordinator.registry.status()["live"] == 0

        resource_headers = {
            ROLLOUT_ID_HEADER: "resource-rollout",
            ATTEMPT_INDEX_HEADER: "0",
            RESOURCE_REQUEST_ID_HEADER: "seed-1",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{urls[2]}/seed",
                json={"value": 1},
                headers=resource_headers,
            ) as response:
                first_resource_result = await response.json()
            async with session.post(
                f"{urls[2]}/seed",
                json={"value": 1},
                headers=resource_headers,
            ) as response:
                replayed_resource_result = await response.json()
        assert replayed_resource_result == first_resource_result
        assert resource_calls == 1
        resources_participant.retire("resource-rollout", 0)
        resource_state.clear()

        live_identity = coordinator.registry.register(
            {
                "_ng_rollout_id": "checkpoint-rollout",
                "_ng_attempt_index": 0,
            }
        )
        checkpoint_dir = tmp_path / "checkpoint"
        prepared = await coordinator.prepare(
            "checkpoint-1",
            time.time() + 10,
        )
        assert prepared["phase"] == "prepared"
        committed = await coordinator.commit("checkpoint-1", checkpoint_dir)
        assert committed["phase"] == "committed_paused"
        manifest = load_checkpoint_manifest(checkpoint_dir)
        assert manifest["checkpoint_id"] == "checkpoint-1"
        await coordinator.resume("checkpoint-1")
        coordinator.registry.release(live_identity)

        servers[2].should_exit = True
        await server_tasks[2]
        resources_participant = ResourcesCheckpointParticipant(
            export_state=export_resource_state,
            restore_states=restore_resource_states,
            restore_expected=True,
        )
        resources_app = FastAPI()
        resources_fence = ControlFence()
        install_control_plane(
            resources_app,
            capabilities=capabilities("resources_servers", "resources"),
            fence=resources_fence,
        )
        install_resources_checkpoint(
            resources_app,
            participant=resources_participant,
            fence=resources_fence,
            auth_token=token,
            server_name="resources",
            route_kind=lambda path, method: (
                "start" if path == "/seed" and method == "POST" else None
            ),
        )
        sockets[2] = _listening_socket()
        servers[2] = uvicorn.Server(
            uvicorn.Config(
                resources_app,
                log_level="error",
                lifespan="off",
            )
        )
        server_tasks[2] = asyncio.create_task(servers[2].serve(sockets=[sockets[2]]))
        while not servers[2].started:
            await asyncio.sleep(0.01)
        config["resources"]["resources_servers"]["implementation"] = {
            "host": sockets[2].getsockname()[0],
            "port": sockets[2].getsockname()[1],
        }
        await coordinator.discover(config, deadline_ts=time.time() + 10)

        restored = await coordinator.restore("restore-1", checkpoint_dir)
        assert restored["phase"] == "restored_paused"
        await coordinator.resume("restore-1")

        checkpoint_file = checkpoint_dir / next(iter(manifest["files"]))
        checkpoint_file.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="missing or corrupted"):
            load_checkpoint_manifest(checkpoint_dir)
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*server_tasks)
