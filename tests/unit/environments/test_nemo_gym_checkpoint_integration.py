# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Exercise RL checkpoint orchestration against real Gym participants."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

import nemo_gym.server_utils
from nemo_gym._checkpoint import (
    CHECKPOINT_CONTROL_TOKEN_ENV,
    GATED_MODEL_ROUTE_SUFFIXES,
    AdmissionLimiter,
    AdmissionMiddleware,
    ControlCapabilities,
    ControlFence,
    MultiProcessCapability,
    install_control_plane,
    install_model_admission,
    install_model_checkpoint,
)
from nemo_gym.config_types import BaseServerConfig, ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import (
    ATTEMPT_INDEX_HEADER,
    MODEL_CALL_ID_HEADER,
    PARENT_MODEL_CALL_ID_HEADER,
    ROLLOUT_ID_HEADER,
    SOURCE_CAPTURE_KEY_HEADER,
    RolloutContextMiddleware,
    current_attempt_index,
    current_logical_rollout_id,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit
from resources_servers.example_session_state_mgmt.app import (
    IncrementCounterRequest,
    IncrementCounterResponse,
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)
from resources_servers.genrm_compare.app import (
    GenRMCompareConfig,
    GenRMCompareResourcesServer,
)
from responses_api_agents.checkpoint_test_agent.app import CheckpointTestAgent
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig

from nemo_rl.environments.nemo_gym import NemoGym


AUTH_TOKEN = "checkpoint-token"
ROLLOUT_ID = "partial-rollout"
SAVE_ID = "checkpoint-1"
RESTORE_ID = "restore-1"

pytestmark = pytest.mark.nemo_gym


class _AsyncBody:
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _Response:
    """Expose the aiohttp-like response surface used by Gym and RL."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status = response.status_code
        self.ok = response.is_success
        self.cookies = response.cookies
        self.headers = response.headers
        self.content = _AsyncBody(response.content)
        self._content = response.content

    async def read(self) -> bytes:
        return self._content

    async def text(self) -> str:
        return self._response.text

    async def json(self) -> Any:
        return self._response.json()

    def raise_for_status(self) -> None:
        self._response.raise_for_status()


class _CountingResourcesServer(StatefulCounterResourcesServer):
    incremented: ClassVar[list[tuple[str, int]]] = []

    async def increment_counter(
        self,
        request: Request,
        body: IncrementCounterRequest,
    ) -> IncrementCounterResponse:
        identity = (current_logical_rollout_id(), current_attempt_index())
        assert identity[0] is not None and identity[1] is not None
        self.incremented.append((identity[0], identity[1]))
        return await super().increment_counter(request, body)


def _global_config(
    *,
    resources_name: str = "resources",
    resources_host: str = "resources.test",
) -> Any:
    return OmegaConf.create(
        {
            "observability_enabled": True,
            "policy": {
                "responses_api_models": {
                    "policy": {
                        "host": "policy.test",
                        "port": 80,
                        "base_url": ["http://policy.test/v1"],
                        "instance_role": "policy",
                    }
                }
            },
            resources_name: {
                "resources_servers": {
                    resources_name: {"host": resources_host, "port": 80}
                }
            },
            "agent": {
                "responses_api_agents": {"agent": {"host": "agent.test", "port": 80}}
            },
        }
    )


def _server_client(
    *,
    resources_name: str = "resources",
    resources_host: str = "resources.test",
) -> ServerClient:
    return ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=_global_config(
            resources_name=resources_name,
            resources_host=resources_host,
        ),
    )


def _checkpoint_env(server_client: ServerClient) -> NemoGym:
    # Instantiate Ray's underlying class locally: the test is about the RL/Gym
    # protocol, not Ray transport or process scheduling.
    env_cls = NemoGym.__ray_metadata__.modified_class
    env = env_cls(
        {
            "model_name": "deterministic-policy",
            "base_urls": ["http://policy.test/v1"],
            "turn_recovery_enabled": True,
            "checkpoint_control_auth_token": AUTH_TOKEN,
        }
    )
    env.rh = object()
    env._server_client = server_client
    env._checkpoint_control_headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    return env


def _agent(
    *,
    resources_name: str = "resources",
    resources_host: str = "resources.test",
    checkpoint_replayable_verify: bool = False,
    agent_cls: type[SimpleAgent] = SimpleAgent,
) -> SimpleAgent:
    return agent_cls(
        config=SimpleAgentConfig(
            host="agent.test",
            port=80,
            entrypoint="app.py",
            name="agent",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            resources_server=ResourcesServerRef(
                type="resources_servers", name=resources_name
            ),
            checkpoint_replayable_verify=checkpoint_replayable_verify,
        ),
        server_client=_server_client(
            resources_name=resources_name,
            resources_host=resources_host,
        ),
    )


def _resources(*, restore_expected: bool = False) -> _CountingResourcesServer:
    return _CountingResourcesServer(
        config=StatefulCounterResourcesServerConfig(
            host="resources.test",
            port=80,
            entrypoint="app.py",
            name="resources",
            checkpoint_restore_expected=restore_expected,
        ),
        server_client=_server_client(),
    )


def _model_response(
    *, response_id: str, output: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1.0,
        "model": "deterministic-policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _model_app(
    ledger_root: Path,
    *,
    first_call_returned: asyncio.Event | None = None,
    terminal_only: bool = False,
) -> tuple[FastAPI, list[dict[str, Any]]]:
    app = FastAPI()
    fence = ControlFence()
    limiter = AdmissionLimiter()
    requests: list[dict[str, Any]] = []
    ledger = FileLineageStore(ledger_root)
    install_control_plane(
        app,
        capabilities=ControlCapabilities(
            component="responses_api_models",
            name="policy",
            instance_role="policy",
            multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
        ),
        fence=fence,
    )
    install_model_admission(
        app,
        limiter=limiter,
        fence=fence,
        instance_role="policy",
        auth_token=AUTH_TOKEN,
    )
    install_model_checkpoint(
        app,
        fence=fence,
        limiter=limiter,
        ledger_provider=lambda: ledger,
        file_ledger_root_provider=lambda: ledger.checkpoint_root,
        instance_role="policy",
        server_name="policy",
        auth_token=AUTH_TOKEN,
    )

    @app.post("/v1/responses")
    async def responses(request: Request) -> JSONResponse:
        body = await request.json()
        rollout_id = current_logical_rollout_id()
        attempt_index = current_attempt_index()
        assert rollout_id is not None and attempt_index is not None
        input_items = body["input"]
        has_tool_result = any(
            item.get("type") == "function_call_output" for item in input_items
        )
        if terminal_only or has_tool_result:
            output = [
                {
                    "id": f"message-{rollout_id}-a{attempt_index}",
                    "content": [
                        {"annotations": [], "text": "done", "type": "output_text"}
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        else:
            output = [
                {
                    "id": "function-call-1",
                    "call_id": "counter-call-1",
                    "name": "increment_counter",
                    "arguments": '{"count":2}',
                    "type": "function_call",
                    "status": "completed",
                }
            ]

        call_id = f"{rollout_id}-a{attempt_index}-call-{len(requests) + 1}"
        requests.append(
            {"body": body, "headers": dict(request.headers), "call_id": call_id}
        )
        capture_key = (
            f"{rollout_id}{'' if attempt_index == 0 else f'-a{attempt_index}'}"
        )
        staging_key = f"stage/{rollout_id}/a{attempt_index}/{call_id}"
        parent_call_id = request.headers.get(PARENT_MODEL_CALL_ID_HEADER)
        parent_match = None
        if parent_call_id is not None:
            source_capture_key = request.headers[SOURCE_CAPTURE_KEY_HEADER]
            parent_match = (
                await ledger.resolve_explicit(
                    source_capture_key,
                    parent_call_id,
                    input_items,
                )
            ).match
            assert parent_match is not None
        prev_len = parent_match.prev_len if parent_match is not None else 0
        staging_chain = (
            (*parent_match.staging_chain, staging_key)
            if parent_match is not None
            else (staging_key,)
        )
        await ledger.record(
            CaptureLedgerCommit(
                rollout_id=capture_key,
                record=CallRecord(
                    model_call_id=call_id,
                    parent_call_id=parent_call_id,
                    staging_key=staging_key,
                    weight_version=7,
                    prev_len=prev_len,
                    delta_len=1,
                    cum_len=prev_len + 1,
                    digest="2" * 64,
                    extras_digest="3" * 64,
                    mode="text" if parent_call_id is None else "token_in",
                    admitted_at=1.0,
                    chain_hash="4" * 64,
                    cumulative_hash="5" * 64,
                    response_id=f"response-{call_id}",
                    output_fingerprint="6" * 64,
                    continuation_fingerprint="7" * 64,
                    fingerprint_version=1,
                ),
                staging_chain=staging_chain,
                parent_manifest=(
                    parent_match.parent_manifest if parent_match is not None else ()
                ),
                request_items=input_items,
                response_items=output,
            )
        )
        if first_call_returned is not None and not has_tool_result:
            first_call_returned.set()
        return JSONResponse(
            _model_response(response_id=f"response-{call_id}", output=output),
            headers={MODEL_CALL_ID_HEADER: call_id},
        )

    app.add_middleware(
        AdmissionMiddleware,
        limiter=limiter,
        gated_suffixes=GATED_MODEL_ROUTE_SUFFIXES,
    )
    app.add_middleware(RolloutContextMiddleware)
    return app, requests


def _run_body(attempt_index: int) -> dict[str, Any]:
    return {
        "_ng_rollout_id": ROLLOUT_ID,
        "_ng_attempt_index": attempt_index,
        "initial_count": 3,
        "expected_count": 5,
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "increment the counter by two, then finish",
                }
            ],
            "tools": [
                {
                    "name": "increment_counter",
                    "parameters": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                    "type": "function",
                }
            ],
        },
    }


def _genrm() -> GenRMCompareResourcesServer:
    return GenRMCompareResourcesServer(
        config=GenRMCompareConfig(
            host="genrm.test",
            port=80,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm",
            genrm_model_server=ModelServerRef(
                type="responses_api_models",
                name="genrm_model",
            ),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
                max_output_tokens=64,
            ),
            num_rollouts_per_prompt=2,
            cohort_collection_timeout_s=5,
        ),
        server_client=_server_client(
            resources_name="genrm",
            resources_host="genrm.test",
        ),
    )


def _genrm_run_body(
    rollout_index: int,
    attempt_index: int,
) -> dict[str, Any]:
    return {
        "_ng_rollout_id": f"genrm-sibling-{rollout_index}",
        "_ng_attempt_index": attempt_index,
        "_ng_task_index": 0,
        "_ng_group_id": "genrm-group",
        "_ng_group_attempt": 0,
        "_ng_rollout_index": rollout_index,
        "responses_create_params": {
            "input": [{"role": "user", "content": "Reply with one word."}],
            "tools": [],
            "parallel_tool_calls": False,
        },
    }


async def _wait_until(predicate: Any, *, timeout_s: float = 2) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not satisfied before the test deadline")
        await asyncio.sleep(0.01)


def _clients(apps: dict[str, FastAPI]) -> dict[str, httpx.AsyncClient]:
    return {
        host: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=f"http://{host}",
        )
        for host, app in apps.items()
    }


def _dispatcher(
    clients: dict[str, httpx.AsyncClient],
):
    async def dispatch(method: str, url: str, **kwargs: Any) -> _Response:
        parsed = urlsplit(url)
        payload = kwargs.get("json")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await clients[parsed.hostname].request(
            method,
            parsed.path,
            params=kwargs.get("params"),
            json=payload,
            cookies=kwargs.get("cookies"),
            headers=kwargs.get("headers"),
        )
        return _Response(response)

    return dispatch


async def _close_all(clients: dict[str, httpx.AsyncClient]) -> None:
    for client in clients.values():
        await client.aclose()


@pytest.mark.asyncio
async def test_rl_coordinator_restores_real_gym_turn_without_repeating_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RL and Gym protocols restore one turn together, end to end on CPU."""
    monkeypatch.setenv(CHECKPOINT_CONTROL_TOKEN_ENV, AUTH_TOKEN)
    _CountingResourcesServer.incremented = []
    checkpoint_dir = tmp_path / "checkpoint"
    first_call_returned = asyncio.Event()

    source_agent = _agent()
    source_resources = _resources()
    source_model_app, source_model_requests = _model_app(
        tmp_path / "source-lineage",
        first_call_returned=first_call_returned,
    )
    source_clients = _clients(
        {
            "agent.test": source_agent.setup_webserver(),
            "resources.test": source_resources.setup_webserver(),
            "policy.test": source_model_app,
        }
    )
    monkeypatch.setattr(
        nemo_gym.server_utils,
        "request",
        _dispatcher(source_clients),
    )
    source_env = _checkpoint_env(_server_client())

    partial_run: asyncio.Task[httpx.Response] | None = None
    try:
        topology = await source_env.discover_checkpoint_capabilities(
            ["policy", "agent", "resources"]
        )
        assert [
            item["participant"]["component"] for item in topology["participants"]
        ] == ["responses_api_agents", "responses_api_models", "resources_servers"]

        partial_run = asyncio.create_task(
            source_clients["agent.test"].post("/run", json=_run_body(0))
        )
        await asyncio.wait_for(first_call_returned.wait(), timeout=2)

        prepared = await source_env.prepare_checkpoint(SAVE_ID, time.time() + 5)
        assert all(item["ready"] for item in prepared["participants"])
        # Session creation is revision 1; the counter mutation is revision 2.
        assert (
            source_resources.checkpoint_participant().revision_for(ROLLOUT_ID, 0) == 2
        )

        committed = await source_env.commit_checkpoint(
            SAVE_ID,
            time.time() + 5,
            str(checkpoint_dir),
        )
        assert len(committed["participants"]) == 3
    finally:
        if partial_run is not None:
            partial_run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await partial_run
        await _close_all(source_clients)

    restored_agent = _agent()
    restored_resources = _resources(restore_expected=True)
    restored_model_app, restored_model_requests = _model_app(
        tmp_path / "restored-lineage"
    )
    restored_clients = _clients(
        {
            "agent.test": restored_agent.setup_webserver(),
            "resources.test": restored_resources.setup_webserver(),
            "policy.test": restored_model_app,
        }
    )
    monkeypatch.setattr(
        nemo_gym.server_utils,
        "request",
        _dispatcher(restored_clients),
    )
    restored_env = _checkpoint_env(_server_client())

    try:
        await restored_env.discover_checkpoint_capabilities(
            ["policy", "agent", "resources"]
        )
        restored = await restored_env.restore_checkpoint(
            RESTORE_ID,
            time.time() + 5,
            str(checkpoint_dir),
            source_checkpoint_id=SAVE_ID,
        )
        assert len(restored["participants"]) == 3

        resumed = await restored_env.resume_checkpoint(
            RESTORE_ID,
            time.time() + 5,
        )
        assert len(resumed["participants"]) == 3

        result = await restored_clients["agent.test"].post(
            "/run",
            json=_run_body(1),
        )
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1.0
    finally:
        await _close_all(restored_clients)

    # The restored agent starts after the tool boundary instead of applying the
    # already-checkpointed mutation a second time.
    assert _CountingResourcesServer.incremented == [(ROLLOUT_ID, 0)]
    assert len(source_model_requests) == 1
    assert len(restored_model_requests) == 1
    restored_request = restored_model_requests[0]
    assert [item["type"] for item in restored_request["body"]["input"][-2:]] == [
        "function_call",
        "function_call_output",
    ]
    assert restored_request["headers"][ROLLOUT_ID_HEADER] == ROLLOUT_ID
    assert restored_request["headers"][ATTEMPT_INDEX_HEADER] == "1"
    assert restored_request["headers"][SOURCE_CAPTURE_KEY_HEADER] == ROLLOUT_ID
    assert (
        restored_request["headers"][PARENT_MODEL_CALL_ID_HEADER]
        == source_model_requests[0]["call_id"]
    )


@pytest.mark.asyncio
async def test_rl_coordinator_rebuilds_real_genrm_group_after_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen verifier and parked sibling rebuild one stateless GenRM cohort."""
    monkeypatch.setenv(CHECKPOINT_CONTROL_TOKEN_ENV, AUTH_TOKEN)
    monkeypatch.setenv("NEMO_GYM_TEST_HOLD_SECOND_TERMINAL_BOUNDARY", "1")
    checkpoint_dir = tmp_path / "genrm-checkpoint"

    source_agent = _agent(
        resources_name="genrm",
        resources_host="genrm.test",
        checkpoint_replayable_verify=True,
        agent_cls=CheckpointTestAgent,
    )
    source_genrm = _genrm()
    source_compare = AsyncMock(return_value=([0.25, 0.75], None, None, None))
    monkeypatch.setattr(source_genrm, "_run_compare", source_compare)
    source_model_app, source_model_requests = _model_app(
        tmp_path / "source-genrm-lineage",
        terminal_only=True,
    )
    source_clients = _clients(
        {
            "agent.test": source_agent.setup_webserver(),
            "genrm.test": source_genrm.setup_webserver(),
            "policy.test": source_model_app,
        }
    )
    monkeypatch.setattr(nemo_gym.server_utils, "request", _dispatcher(source_clients))
    source_env = _checkpoint_env(
        _server_client(resources_name="genrm", resources_host="genrm.test")
    )

    source_runs: list[asyncio.Task[httpx.Response]] = []
    try:
        topology = await source_env.discover_checkpoint_capabilities(
            ["policy", "agent", "genrm"]
        )
        genrm_capability = next(
            item
            for item in topology["participants"]
            if item["participant"]["server_name"] == "genrm"
        )
        assert genrm_capability["capabilities"]["checkpoint_mode"] == "stateless"
        assert genrm_capability["capabilities"]["group_scoring"] == {
            "expected_group_size": 2,
            "verification_replayable": True,
            "collection_timeout_s": 5.0,
        }

        source_runs = [
            asyncio.create_task(
                source_clients["agent.test"].post(
                    "/run",
                    json=_genrm_run_body(index, 0),
                )
            )
            for index in range(2)
        ]
        await _wait_until(
            lambda: any(
                len(cohort.members) == 1
                for cohort in source_genrm._verify_cohorts.values()
            )
        )

        prepared = await source_env.prepare_checkpoint(SAVE_ID, time.time() + 5)
        assert all(item["ready"] for item in prepared["participants"])
        committed = await source_env.commit_checkpoint(
            SAVE_ID,
            time.time() + 5,
            str(checkpoint_dir),
        )
        # GenRM is deliberately stateless. Gym saves the two continuations and
        # model lineage, then rebuilds the verifier cohort by replaying /verify.
        assert {
            item["participant"]["component"] for item in committed["participants"]
        } == {"responses_api_agents", "responses_api_models"}
        source_compare.assert_not_awaited()
    finally:
        for task in source_runs:
            task.cancel()
        if source_runs:
            await asyncio.gather(*source_runs, return_exceptions=True)
        await _close_all(source_clients)

    monkeypatch.delenv("NEMO_GYM_TEST_HOLD_SECOND_TERMINAL_BOUNDARY")
    restored_agent = _agent(
        resources_name="genrm",
        resources_host="genrm.test",
        checkpoint_replayable_verify=True,
        agent_cls=CheckpointTestAgent,
    )
    restored_genrm = _genrm()
    restored_compare = AsyncMock(return_value=([0.25, 0.75], None, None, None))
    monkeypatch.setattr(restored_genrm, "_run_compare", restored_compare)
    restored_model_app, restored_model_requests = _model_app(
        tmp_path / "restored-genrm-lineage",
        terminal_only=True,
    )
    restored_clients = _clients(
        {
            "agent.test": restored_agent.setup_webserver(),
            "genrm.test": restored_genrm.setup_webserver(),
            "policy.test": restored_model_app,
        }
    )
    monkeypatch.setattr(
        nemo_gym.server_utils,
        "request",
        _dispatcher(restored_clients),
    )
    restored_env = _checkpoint_env(
        _server_client(resources_name="genrm", resources_host="genrm.test")
    )

    try:
        await restored_env.discover_checkpoint_capabilities(
            ["policy", "agent", "genrm"]
        )
        restored = await restored_env.restore_checkpoint(
            RESTORE_ID,
            time.time() + 5,
            str(checkpoint_dir),
            source_checkpoint_id=SAVE_ID,
        )
        assert {
            item["participant"]["component"] for item in restored["participants"]
        } == {"responses_api_agents", "responses_api_models"}
        await restored_env.resume_checkpoint(RESTORE_ID, time.time() + 5)

        responses = await asyncio.wait_for(
            asyncio.gather(
                *(
                    restored_clients["agent.test"].post(
                        "/run",
                        json=_genrm_run_body(index, 1),
                    )
                    for index in range(2)
                )
            ),
            timeout=5,
        )
        assert [response.status_code for response in responses] == [200, 200]
        assert sorted(response.json()["reward"] for response in responses) == [
            0.25,
            0.75,
        ]
    finally:
        await _close_all(restored_clients)

    restored_compare.assert_awaited_once()
    assert len(source_model_requests) == 2
    # Both restored siblings resume at the terminal boundary and replay only
    # their stateless verification calls; neither regenerates model output.
    assert restored_model_requests == []
