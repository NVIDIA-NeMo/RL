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

import pytest
from pydantic import ValidationError

from nemo_rl.environments.gym_checkpoint import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymCheckpointTopology,
    GymControlCapabilities,
    GymDiscoveredParticipant,
    GymExecutionIdentity,
    gym_capture_key,
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
        "phase": "idle",
        "active_checkpoint_id": None,
        "deadline_ts": None,
    }
    payload.update(overrides)
    return payload


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


def test_capability_contract_rejects_unknown_fields_and_schema_drift() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GymControlCapabilities.model_validate(_capabilities(unexpected=True))

    with pytest.raises(ValidationError, match="Input should be 1"):
        GymControlCapabilities.model_validate(_capabilities(schema_version=2))


def test_capability_contract_binds_routing_and_participant_identity() -> None:
    capabilities = GymControlCapabilities.model_validate(_capabilities())

    participant = capabilities.participant("policy_model_route")

    assert participant.model_dump() == {
        "server_name": "policy_model_route",
        "component": "responses_api_models",
        "participant_name": "policy_model",
    }


def test_topology_fingerprint_excludes_dynamic_checkpoint_phase() -> None:
    first = GymControlCapabilities.model_validate(_capabilities())
    second = GymControlCapabilities.model_validate(
        _capabilities(
            admission_states=["paused", "accepting", "draining"],
            phase="preparing",
            active_checkpoint_id="snapshot-7",
            deadline_ts=123.0,
        )
    )

    first_topology = GymCheckpointTopology.from_discovered(
        [
            GymDiscoveredParticipant(
                participant=first.participant("policy-route"),
                capabilities=first,
            )
        ]
    )
    second_topology = GymCheckpointTopology.from_discovered(
        [
            GymDiscoveredParticipant(
                participant=second.participant("policy-route"),
                capabilities=second,
            )
        ]
    )

    assert first_topology.fingerprint() == second_topology.fingerprint()
