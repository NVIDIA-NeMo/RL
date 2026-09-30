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

"""Contract parity tests against the pinned NeMo-Gym checkpoint package."""

from __future__ import annotations

import time
import types
from enum import Enum
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym._checkpoint import (
    AGENT_CHECKPOINT_URL_PREFIX,
    MODEL_ADMISSION_URL_PREFIX,
    MODEL_CHECKPOINT_URL_PREFIX,
    RESOURCES_CHECKPOINT_URL_PREFIX,
    AdmissionLimiter,
    AgentAcknowledgeRequest,
    AgentCheckpointParticipant,
    CaptureLedgerCommitResult,
    CaptureLedgerRestoreResult,
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
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN, capture_key_for
from nemo_gym.token_id_capture.lineage import FileLineageStore

from nemo_rl.environments import gym_checkpoint as rl_contract
from nemo_rl.environments.gym_checkpoint import (
    GymAgentCommitResponse,
    GymAgentPrepareResponse,
    GymAgentRestoreResponse,
    GymAgentResumeResponse,
    GymCheckpointParticipantContract,
    GymCompletionReceipt,
    GymModelCommitResponse,
    GymModelPrepareResponse,
    GymModelRestoreResponse,
    GymModelResumeResponse,
    GymMultiProcessCapability,
    GymParticipantIdentity,
    GymResourcesCommitResponse,
    GymResourcesPrepareResponse,
    GymResourcesRestoreResponse,
    GymResourcesResumeResponse,
    GymSingleWorkerModelStatusResponse,
)

pytestmark = pytest.mark.nemo_gym

_AUTH_TOKEN = "checkpoint-token"
_AUTH_HEADERS = {"authorization": f"Bearer {_AUTH_TOKEN}"}


def _literal_values(annotation: object) -> set[object]:
    origin = get_origin(annotation)
    if origin is Literal:
        return set(get_args(annotation))
    if origin in {list, set, tuple, types.UnionType, Union}:
        values: set[object] = set()
        for argument in get_args(annotation):
            values.update(_literal_values(argument))
        return values
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {member.value for member in annotation}
    return set()


def _assert_model_shape_matches(
    rl_model: type[BaseModel],
    gym_model: type[BaseModel],
) -> None:
    assert set(rl_model.model_fields) == set(gym_model.model_fields)
    for name, gym_field in gym_model.model_fields.items():
        rl_field = rl_model.model_fields[name]
        assert rl_field.is_required() == gym_field.is_required(), name
        if not gym_field.is_required():
            assert rl_field.get_default(call_default_factory=True) == (
                gym_field.get_default(call_default_factory=True)
            ), name


def _capabilities(
    component: Literal[
        "responses_api_models",
        "responses_api_agents",
        "resources_servers",
    ],
    name: str,
    *,
    instance_role: Literal["policy", "auxiliary"] | None = None,
) -> ControlCapabilities:
    return ControlCapabilities(
        component=component,
        name=name,
        admission_states=["accepting", "draining", "paused"],
        checkpoint_mode="export_restore",
        concurrency_contract="stateless",
        multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
        instance_role=instance_role,
    )


def _model_app(ledger_root: Path) -> FastAPI:
    app = FastAPI()
    fence = ControlFence()
    limiter = AdmissionLimiter()
    ledger = FileLineageStore(ledger_root)
    install_control_plane(
        app,
        capabilities=_capabilities(
            "responses_api_models",
            "policy",
            instance_role="policy",
        ),
        fence=fence,
    )
    install_model_admission(
        app,
        limiter=limiter,
        fence=fence,
        instance_role="policy",
        auth_token=_AUTH_TOKEN,
    )
    install_model_checkpoint(
        app,
        fence=fence,
        limiter=limiter,
        ledger_provider=lambda: ledger,
        file_ledger_root_provider=lambda: ledger.checkpoint_root,
        instance_role="policy",
        server_name="policy",
        auth_token=_AUTH_TOKEN,
    )
    return app


def _agent_app() -> FastAPI:
    app = FastAPI()
    fence = ControlFence()
    install_control_plane(
        app,
        capabilities=_capabilities("responses_api_agents", "agent"),
        fence=fence,
    )
    install_agent_checkpoint(
        app,
        participant=AgentCheckpointParticipant("agent"),
        fence=fence,
        auth_token=_AUTH_TOKEN,
    )
    return app


def _resources_app(*, restore_expected: bool = False) -> FastAPI:
    async def export_state(_rollout_id: str, _attempt_index: int) -> dict[str, object]:
        return {}

    async def restore_states(_snapshots: list[ResourceSnapshot]) -> None:
        return None

    app = FastAPI()
    fence = ControlFence()
    install_control_plane(
        app,
        capabilities=_capabilities("resources_servers", "resources"),
        fence=fence,
    )
    install_resources_checkpoint(
        app,
        participant=ResourcesCheckpointParticipant(
            export_state=export_state,
            restore_states=restore_states,
            restore_expected=restore_expected,
        ),
        fence=fence,
        auth_token=_AUTH_TOKEN,
        server_name="resources",
        route_kind=lambda _path, _method: None,
    )
    return app


async def _post(
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, object],
) -> dict[str, object]:
    response = await client.post(path, json=payload, headers=_AUTH_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


async def _get(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, object],
) -> dict[str, object]:
    response = await client.get(path, params=params, headers=_AUTH_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def test_capability_contract_matches_pinned_gym() -> None:
    field_pairs = {
        "admission_states": "admission_states",
        "checkpoint_mode": "checkpoint_mode",
        "concurrency_contract": "concurrency_contract",
        "instance_role": "instance_role",
    }
    for rl_name, gym_name in field_pairs.items():
        assert _literal_values(
            GymCheckpointParticipantContract.model_fields[rl_name].annotation
        ) == _literal_values(ControlCapabilities.model_fields[gym_name].annotation)

    assert GymCheckpointParticipantContract.model_fields["schema_version"].default == (
        ControlCapabilities.model_fields["schema_version"].default
    )

    assert _literal_values(
        GymParticipantIdentity.model_fields["component"].annotation
    ) == _literal_values(ControlCapabilities.model_fields["component"].annotation)
    assert _literal_values(
        GymMultiProcessCapability.model_fields["mode"].annotation
    ) == _literal_values(MultiProcessCapability.model_fields["mode"].annotation)


def test_identity_and_completion_receipt_contract_matches_pinned_gym() -> None:
    assert rl_contract._IDENTITY_PATTERN == ROLLOUT_ID_PATTERN.pattern
    for rollout_id, attempt_index in (("rollout", 0), ("rollout", 3)):
        assert rl_contract.gym_capture_key(
            rollout_id, attempt_index
        ) == capture_key_for(rollout_id, attempt_index)

    _assert_model_shape_matches(GymCompletionReceipt, AgentAcknowledgeRequest)
    payload = {
        "rollout_id": "rollout",
        "attempt_index": 1,
        "execution_generation": 2,
        "result_identity": "result-1",
        "result_digest": "a" * 64,
        "manifest_capture_key": "rollout-a1",
        "terminal_model_call_id": "call-1",
    }
    assert GymCompletionReceipt.model_validate(payload).model_dump(mode="json") == (
        AgentAcknowledgeRequest.model_validate(payload).model_dump(mode="json")
    )


def test_model_ledger_response_contract_matches_pinned_gym() -> None:
    _assert_model_shape_matches(GymModelCommitResponse, CaptureLedgerCommitResult)
    _assert_model_shape_matches(GymModelRestoreResponse, CaptureLedgerRestoreResult)


@pytest.mark.asyncio
async def test_real_gym_routes_produce_rl_compatible_checkpoint_replies(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_id = "checkpoint-1"
    deadline = time.time() + 10
    prepare = {"checkpoint_id": checkpoint_id, "deadline_ts": deadline}

    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_model_app(tmp_path / "source-ledger")),
            base_url="http://model",
        ) as model,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_agent_app()),
            base_url="http://agent",
        ) as agent,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_resources_app()),
            base_url="http://resources",
        ) as resources,
    ):
        GymModelPrepareResponse.model_validate(
            await _post(model, f"{MODEL_ADMISSION_URL_PREFIX}/pause", prepare)
        )
        GymAgentPrepareResponse.model_validate(
            await _post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/prepare", prepare)
        )
        GymResourcesPrepareResponse.model_validate(
            await _post(
                resources, f"{RESOURCES_CHECKPOINT_URL_PREFIX}/prepare", prepare
            )
        )
        GymSingleWorkerModelStatusResponse.model_validate(
            await _get(
                model,
                f"{MODEL_ADMISSION_URL_PREFIX}/status",
                {"checkpoint_id": checkpoint_id, "deadline_ts": deadline},
            )
        )
        commit = {**prepare, "checkpoint_dir": str(checkpoint_dir)}
        agent_commit_payload = await _post(
            agent,
            f"{AGENT_CHECKPOINT_URL_PREFIX}/commit",
            commit,
        )
        GymAgentCommitResponse.model_validate(agent_commit_payload)
        GymModelCommitResponse.model_validate(
            await _post(
                model,
                f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
                {
                    **commit,
                    "continuation_indexes": [
                        agent_commit_payload["continuation_index"]
                    ],
                },
            )
        )
        GymResourcesCommitResponse.model_validate(
            await _post(
                resources,
                f"{RESOURCES_CHECKPOINT_URL_PREFIX}/commit",
                commit,
            )
        )

        GymResourcesResumeResponse.model_validate(
            await _post(
                resources,
                f"{RESOURCES_CHECKPOINT_URL_PREFIX}/resume",
                prepare,
            )
        )
        GymModelResumeResponse.model_validate(
            await _post(model, f"{MODEL_ADMISSION_URL_PREFIX}/resume", prepare)
        )
        GymAgentResumeResponse.model_validate(
            await _post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/resume", prepare)
        )

    restore_id = "restore-1"
    restore = {
        "checkpoint_id": restore_id,
        "deadline_ts": time.time() + 10,
        "checkpoint_dir": str(checkpoint_dir),
    }
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_model_app(tmp_path / "restored-ledger")),
            base_url="http://model",
        ) as model,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_agent_app()),
            base_url="http://agent",
        ) as agent,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_resources_app(restore_expected=True)),
            base_url="http://resources",
        ) as resources,
    ):
        GymModelRestoreResponse.model_validate(
            await _post(model, f"{MODEL_CHECKPOINT_URL_PREFIX}/restore", restore)
        )
        GymAgentRestoreResponse.model_validate(
            await _post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/restore", restore)
        )
        GymResourcesRestoreResponse.model_validate(
            await _post(
                resources,
                f"{RESOURCES_CHECKPOINT_URL_PREFIX}/restore",
                restore,
            )
        )

        resume = {"checkpoint_id": restore_id, "deadline_ts": time.time() + 10}
        GymResourcesResumeResponse.model_validate(
            await _post(
                resources,
                f"{RESOURCES_CHECKPOINT_URL_PREFIX}/resume",
                resume,
            )
        )
        GymModelResumeResponse.model_validate(
            await _post(model, f"{MODEL_ADMISSION_URL_PREFIX}/resume", resume)
        )
        GymAgentResumeResponse.model_validate(
            await _post(agent, f"{AGENT_CHECKPOINT_URL_PREFIX}/resume", resume)
        )
