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

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nemo_rl.environments.gym_checkpoint import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymCheckpointTopology,
    GymCheckpointPhase,
    GymDiscoveredParticipant,
    GymExecutionIdentity,
    GymModelPrepareResponse,
    GymMultiProcessCapability,
    GymParticipantIdentity,
    checkpoint_server_names,
    gym_capture_key,
    participates_in_checkpoint_phase,
)


def _capabilities(**overrides):
    payload = {
        "component": "responses_api_models",
        "name": "policy_model",
        "schema_version": GYM_CHECKPOINT_SCHEMA_VERSION,
        "admission_states": ["accepting", "draining", "paused"],
        "checkpoint_mode": "export_restore",
        "concurrency_contract": "stateless",
        "multi_process": {"mode": "single_worker", "num_workers": 1},
        "instance_role": "policy",
        "features": [],
        "phase": "idle",
        "active_checkpoint_id": None,
        "deadline_ts": None,
    }
    payload.update(overrides)
    multi_process = GymMultiProcessCapability.model_validate(
        payload.pop("multi_process")
    )
    return SimpleNamespace(**payload, multi_process=multi_process)


def _discovered(**overrides) -> GymDiscoveredParticipant:
    capabilities = _capabilities(**overrides)
    return GymDiscoveredParticipant(
        participant=GymParticipantIdentity(
            server_name="policy-route",
            component=capabilities.component,
            participant_name=capabilities.name,
        ),
        capabilities=capabilities,
    )


def test_gym_execution_identity_separates_logical_id_from_capture_key() -> None:
    first = GymExecutionIdentity(rollout_id="group-7_g0", attempt_index=0)
    retry = GymExecutionIdentity(rollout_id="group-7_g0", attempt_index=2)

    assert first.rollout_id == retry.rollout_id
    assert first.capture_key == "group-7_g0"
    assert retry.capture_key == "group-7_g0-a2"
    assert gym_capture_key("group-7_g0", 2) == retry.capture_key


@pytest.mark.parametrize(
    ("rollout_id", "attempt_index"),
    [
        ("../escape", 0),
        ("rollout", -1),
        ("rollout", True),
    ],
)
def test_gym_execution_identity_rejects_invalid_values(
    rollout_id: str,
    attempt_index: int,
) -> None:
    with pytest.raises(ValidationError):
        GymExecutionIdentity(
            rollout_id=rollout_id,
            attempt_index=attempt_index,
        )


def test_checkpoint_server_names_selects_only_server_entries() -> None:
    assert checkpoint_server_names(
        {
            "policy": {"responses_api_models": {}},
            "agent": {"responses_api_agents": {}},
            "global": {"policy_base_url": "http://policy"},
            "ambiguous": {
                "responses_api_agents": {},
                "resources_servers": {},
            },
        }
    ) == ["agent", "policy"]


def test_checkpoint_phase_participation_distinguishes_policy_and_state() -> None:
    policy = _discovered(checkpoint_mode="stateless")
    auxiliary = _discovered(instance_role="auxiliary")
    agent = _discovered(
        component="responses_api_agents",
        name="agent",
        instance_role=None,
    )

    assert participates_in_checkpoint_phase(policy, GymCheckpointPhase.PREPARE)
    assert participates_in_checkpoint_phase(policy, GymCheckpointPhase.RESUME)
    assert not participates_in_checkpoint_phase(policy, GymCheckpointPhase.COMMIT)
    assert not participates_in_checkpoint_phase(auxiliary, GymCheckpointPhase.PREPARE)
    assert participates_in_checkpoint_phase(agent, GymCheckpointPhase.COMMIT)


def test_topology_fingerprint_canonicalizes_capability_ordering() -> None:
    first = _discovered()
    second = _discovered(
        admission_states=["paused", "accepting", "draining"],
    )

    first_topology = GymCheckpointTopology.from_discovered([first])
    second_topology = GymCheckpointTopology.from_discovered([second])

    assert first_topology.fingerprint() == second_topology.fingerprint()


def test_model_prepare_accepts_additive_coordinator_evidence() -> None:
    response = GymModelPrepareResponse.model_validate(
        {
            "state": "paused",
            "workers": {"acknowledged": 2, "expected": 2},
            "inflight_total": 0,
            "response_inflight_total": 0,
            "generation_pending_total": 0,
            "generation_cut_proof": {"checkpoint_id": "snapshot-1"},
            "waiters_total": 0,
        }
    )

    assert response.state == "paused"
    assert response.workers.acknowledged == 2
