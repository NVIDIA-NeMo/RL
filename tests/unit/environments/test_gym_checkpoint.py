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

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nemo_rl.environments.gym_checkpoint import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymAgentRetireResponse,
    GymAgentCommitResponse,
    GymAgentRestoreResponse,
    GymCheckpointCommitResult,
    GymCheckpointRestoreResult,
    GymCheckpointTopology,
    GymCheckpointPhase,
    GymDiscoveredParticipant,
    GymExecutionIdentity,
    GymModelPrepareResponse,
    GymMultiProcessCapability,
    GymParticipantIdentity,
    checkpoint_server_names,
    gym_capture_key,
    rebase_gym_checkpoint_commit_result,
    rebase_gym_checkpoint_restore_result,
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
    "payload",
    [
        {"retired": False, "tombstoned": False, "completed_unacknowledged": True},
        {"retired": False, "tombstoned": True},
        {"retired": True, "tombstoned": True},
    ],
)
def test_agent_retire_response_accepts_only_safe_dispositions(payload: dict) -> None:
    GymAgentRetireResponse.model_validate(payload)


def test_agent_retire_response_rejects_ambiguous_disposition() -> None:
    with pytest.raises(ValidationError, match="invalid Gym agent retirement"):
        GymAgentRetireResponse.model_validate(
            {
                "retired": False,
                "tombstoned": False,
                "completed_unacknowledged": False,
            }
        )


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


def test_shard_artifact_rebase_updates_commit_and_restore_coordinates() -> None:
    participant = {
        "server_name": "agent-route",
        "component": "responses_api_agents",
        "participant_name": "agent",
    }
    artifact = {
        "schema_version": 1,
        "relative_path": "agent/continuations.jsonl",
        "sha256": "a" * 64,
        "records": 1,
        "bytes": 10,
    }
    committed = GymCheckpointCommitResult.model_validate(
        {
            "checkpoint_id": "checkpoint-1",
            "participants": [
                {
                    "participant": participant,
                    "payload": {
                        "records": 1,
                        "manifest_digest": "b" * 64,
                        "continuation_index": artifact,
                    },
                    "manifest": {
                        "participant": participant,
                        "relative_path": "agent/manifest.json",
                        "manifest_digest": "b" * 64,
                    },
                }
            ],
        }
    )
    restored = GymCheckpointRestoreResult.model_validate(
        {
            "checkpoint_id": "restore-1",
            "participants": [
                {
                    "participant": participant,
                    "payload": {
                        "records": 1,
                        "source_checkpoint_id": "checkpoint-1",
                        "continuation_index": artifact,
                    },
                }
            ],
        }
    )

    rebased_commit = rebase_gym_checkpoint_commit_result(
        committed, Path("gym-shards/first")
    )
    rebased_restore = rebase_gym_checkpoint_restore_result(
        restored, Path("gym-shards/first")
    )

    committed_payload = rebased_commit.participants[0].payload
    restored_payload = rebased_restore.participants[0].payload
    assert isinstance(committed_payload, GymAgentCommitResponse)
    assert isinstance(restored_payload, GymAgentRestoreResponse)
    assert (
        committed_payload.continuation_index.relative_path
        == "gym-shards/first/agent/continuations.jsonl"
    )
    assert (
        rebased_commit.participants[0].manifest.relative_path
        == "gym-shards/first/agent/manifest.json"
    )
    assert (
        restored_payload.continuation_index.relative_path
        == "gym-shards/first/agent/continuations.jsonl"
    )


def test_topology_fingerprint_includes_shard_ownership() -> None:
    participant = {
        "server_name": "agent-route",
        "component": "responses_api_agents",
        "participant_name": "agent",
    }
    contract = {
        "participant": participant,
        "schema_version": 1,
        "admission_states": ["accepting"],
        "checkpoint_mode": "export_restore",
        "concurrency_contract": "serialized_per_session",
        "multi_process": {"mode": "single_worker", "num_workers": 1},
        "instance_role": None,
        "features": [],
    }
    identity = GymCheckpointTopology.participant_identity_key(
        GymParticipantIdentity.model_validate(participant)
    )
    first = GymCheckpointTopology.model_validate(
        {"participants": [contract], "participant_owners": {identity: ["first"]}}
    )
    second = GymCheckpointTopology.model_validate(
        {"participants": [contract], "participant_owners": {identity: ["second"]}}
    )

    assert first.fingerprint() != second.fingerprint()


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


def test_prefix_recovery_requires_generation_cut_lineage_capability() -> None:
    model = _discovered(features=["external_storage_reference_index_v1"])
    agent = _discovered(
        component="responses_api_agents",
        name="agent",
        admission_states=["accepting"],
        concurrency_contract="serialized_per_session",
        instance_role=None,
        features=[
            "agent_continuation_index_v1",
            "completed_result_acknowledgement",
        ],
    )
    topology = GymCheckpointTopology.from_discovered([model, agent])

    with pytest.raises(RuntimeError, match="durable lineage cuts"):
        topology.validate_turn_recovery_capabilities(
            generation_prefix_cuts_enabled=True
        )

    model.capabilities.features.append("generation_cut_lineage_v1")
    GymCheckpointTopology.from_discovered(
        [model, agent]
    ).validate_turn_recovery_capabilities(generation_prefix_cuts_enabled=True)
