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

import hashlib
import json

import pytest
from pydantic import ValidationError

from nemo_rl.environments.gym_checkpoint import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymAgentCheckpointDirectoryRequest,
    GymAgentCommitResponse,
    GymAgentRestoreResponse,
    GymCheckpointCommitResult,
    GymCheckpointContinuation,
    GymCheckpointRestoreResult,
    GymCheckpointTopology,
    GymControlCapabilities,
    GymDiscoveredParticipant,
    GymExecutionIdentity,
    GymModelCheckpointCommitRequest,
    GymModelCheckpointRestoreRequest,
    GymModelCommitResponse,
    GymCoordinatorModelStatusResponse,
    GymModelRestoreResponse,
    GymSingleWorkerModelStatusResponse,
    gym_capture_key,
    gym_checkpoint_continuations,
    gym_checkpoint_staging_keys,
    validate_gym_checkpoint_manifests,
    validate_gym_checkpoint_restore_artifacts,
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


def test_checkpoint_requests_use_required_new_only_artifact_contract() -> None:
    common = {
        "checkpoint_id": "checkpoint-1",
        "deadline_ts": 123.0,
        "checkpoint_dir": "/checkpoint",
    }
    expected_common = {"schema_version": GYM_CHECKPOINT_SCHEMA_VERSION, **common}

    assert GymAgentCheckpointDirectoryRequest(**common).model_dump() == expected_common
    assert GymModelCheckpointRestoreRequest(**common).model_dump() == expected_common
    assert GymModelCheckpointCommitRequest(
        **common,
        continuation_indexes=[],
    ).model_dump() == {**expected_common, "continuation_indexes": []}

    with pytest.raises(ValidationError, match="continuation_indexes"):
        GymModelCheckpointCommitRequest(**common)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GymAgentCheckpointDirectoryRequest(
            **common,
            include_continuation_index=True,
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GymModelCheckpointRestoreRequest(
            **common,
            include_storage_reference_index=True,
        )


def test_single_worker_model_status_accepts_current_and_additive_fields() -> None:
    status = GymSingleWorkerModelStatusResponse.model_validate(
        {
            "checkpoint_id": "checkpoint-1",
            "state": "paused",
            "per_worker": {
                "0": {
                    "state": "paused",
                    "inflight": 0,
                    "future_worker_metric": 7,
                }
            },
            "inflight_total": 0,
            "response_inflight_total": 0,
            "generation_pending_total": 0,
            "waiters_total": 0,
            "inflight": [
                {
                    "rollout_id": "rollout-1",
                    "attempt_index": 0,
                    "plane": "policy",
                    "age_seconds": 1.0,
                    "future_request_metric": 5,
                }
            ],
            "tombstones": [],
            "future_status_metric": 9,
        }
    )

    assert status.response_inflight_total == 0
    assert status.generation_pending_total == 0


def test_coordinator_model_status_accepts_current_and_additive_fields() -> None:
    status = GymCoordinatorModelStatusResponse.model_validate(
        {
            "state": "paused",
            "workers": {
                "acknowledged": 1,
                "expected": 1,
                "live": 1,
                "future_worker_summary": 3,
            },
            "missing_workers": 0,
            "inflight_total": 0,
            "response_inflight_total": 0,
            "generation_pending_total": 0,
            "waiters_total": 0,
            "per_worker": {
                "worker-0": {
                    "acked_seq": 1,
                    "inflight": 0,
                    "generation_pending": 0,
                    "generation_cut_proof": None,
                    "proof_error": None,
                    "connected": True,
                    "future_worker_metric": 7,
                }
            },
            "future_status_metric": 9,
        }
    )

    assert status.response_inflight_total == 0
    assert status.generation_pending_total == 0
    assert status.per_worker["worker-0"].generation_pending == 0


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


def test_capability_contract_accepts_additive_unknown_features() -> None:
    capabilities = GymControlCapabilities.model_validate(
        _capabilities(
            features=[
                "completed_result_acknowledgement",
                "future_prefix_recovery",
            ]
        )
    )

    assert capabilities.features == [
        "completed_result_acknowledgement",
        "future_prefix_recovery",
    ]


def test_topology_fingerprint_excludes_dynamic_checkpoint_phase() -> None:
    first = GymControlCapabilities.model_validate(_capabilities())
    second = GymControlCapabilities.model_validate(
        _capabilities(
            admission_states=["paused", "accepting", "draining"],
            phase="preparing",
            active_checkpoint_id="snapshot-7",
            deadline_ts=123.0,
            features=["future_prefix_recovery"],
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


@pytest.mark.parametrize(
    ("agent_features", "model_features", "error"),
    [
        (
            ["agent_continuation_index_v1"],
            ["external_storage_reference_index_v1"],
            "completed-result acknowledgement",
        ),
        (
            ["completed_result_acknowledgement"],
            ["external_storage_reference_index_v1"],
            "continuation-index support",
        ),
        (
            [
                "agent_continuation_index_v1",
                "completed_result_acknowledgement",
            ],
            [],
            "external-storage reference indexes",
        ),
    ],
)
def test_turn_recovery_capability_guardrails(
    agent_features: list[str],
    model_features: list[str],
    error: str,
) -> None:
    model = GymControlCapabilities.model_validate(
        _capabilities(features=model_features)
    )
    agent = GymControlCapabilities.model_validate(
        _capabilities(
            component="responses_api_agents",
            name="agent",
            admission_states=["accepting"],
            concurrency_contract="serialized_per_session",
            instance_role=None,
            features=agent_features,
        )
    )
    topology = GymCheckpointTopology.from_discovered(
        [
            GymDiscoveredParticipant(
                participant=model.participant("policy-route"),
                capabilities=model,
            ),
            GymDiscoveredParticipant(
                participant=agent.participant("agent-route"),
                capabilities=agent,
            ),
        ]
    )

    with pytest.raises(RuntimeError, match=error):
        topology.validate_turn_recovery_capabilities()


def test_restart_only_resource_requires_agent_fresh_restart_support() -> None:
    model = GymControlCapabilities.model_validate(
        _capabilities(features=["external_storage_reference_index_v1"])
    )
    agent = GymControlCapabilities.model_validate(
        _capabilities(
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
    )
    resources = GymControlCapabilities.model_validate(
        _capabilities(
            component="resources_servers",
            name="tools",
            admission_states=["accepting"],
            checkpoint_mode="restart_only",
            concurrency_contract="stateless",
            instance_role=None,
        )
    )

    def topology_for(agent_capabilities: GymControlCapabilities):
        return GymCheckpointTopology.from_discovered(
            [
                GymDiscoveredParticipant(
                    participant=model.participant("policy-route"),
                    capabilities=model,
                ),
                GymDiscoveredParticipant(
                    participant=agent_capabilities.participant("agent-route"),
                    capabilities=agent_capabilities,
                ),
                GymDiscoveredParticipant(
                    participant=resources.participant("tools-route"),
                    capabilities=resources,
                ),
            ]
        )

    with pytest.raises(RuntimeError, match="restored-continuation discard"):
        topology_for(agent).validate_turn_recovery_capabilities()

    discard_only = agent.model_copy(
        update={
            "features": [
                *agent.features,
                "discard_restored_continuation_v1",
            ]
        }
    )
    with pytest.raises(RuntimeError, match="resource dependency indexes"):
        topology_for(discard_only).validate_turn_recovery_capabilities()

    supported = discard_only.model_copy(
        update={
            "features": [
                *discard_only.features,
                "agent_resource_dependency_index_v1",
            ]
        }
    )
    topology_for(supported).validate_turn_recovery_capabilities()


def test_participant_manifest_digest_is_verified_before_publication(tmp_path) -> None:
    manifest_path = tmp_path / "resources" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checkpoint = GymCheckpointCommitResult.model_validate(
        {
            "checkpoint_id": "snapshot-7",
            "participants": [
                {
                    "participant": {
                        "server_name": "resources-route",
                        "component": "resources_servers",
                        "participant_name": "resources",
                    },
                    "payload": {"sessions": 1, "manifest_digest": digest},
                    "manifest": {
                        "participant": {
                            "server_name": "resources-route",
                            "component": "resources_servers",
                            "participant_name": "resources",
                        },
                        "relative_path": "resources/manifest.json",
                        "manifest_digest": digest,
                    },
                }
            ],
        }
    )

    validate_gym_checkpoint_manifests(tmp_path, checkpoint)
    manifest_path.write_text('{"corrupt": true}')

    with pytest.raises(ValueError, match="manifest digest mismatch"):
        validate_gym_checkpoint_manifests(tmp_path, checkpoint)


def test_private_lineage_is_not_scanned_for_tq_staging_ownership(tmp_path) -> None:
    agent_dir = tmp_path / "agent" / "instance-test"
    agent_dir.mkdir(parents=True)
    agent_record_path = agent_dir / "group-7_g0.a0.json"
    agent_record_path.write_text(
        json.dumps(
            {
                "rollout_id": "group-7_g0",
                "attempt_index": 0,
                "boundary_index": 1,
                "last_committed_model_call_id": "call-1",
            }
        )
    )
    agent_record_digest = hashlib.sha256(agent_record_path.read_bytes()).hexdigest()
    continuation_path = agent_dir / "continuations.jsonl"
    continuation_payload = (
        json.dumps(
            {
                "schema_version": 1,
                "rollout_id": "group-7_g0",
                "attempt_index": 0,
                "capture_key": "group-7_g0",
                "last_committed_model_call_id": "call-1",
                "resource_state_revisions": {"durable-tools": 2},
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    continuation_path.write_bytes(continuation_payload)
    continuation_reference = {
        "schema_version": 1,
        "relative_path": "agent/instance-test/continuations.jsonl",
        "sha256": hashlib.sha256(continuation_payload).hexdigest(),
        "records": 1,
        "bytes": len(continuation_payload),
    }
    agent_manifest_path = agent_dir / "manifest.json"
    agent_manifest_path.write_text(
        json.dumps(
            {
                "files": {agent_record_path.name: agent_record_digest},
                "continuation_index": continuation_reference,
            }
        )
    )
    agent_manifest_digest = hashlib.sha256(agent_manifest_path.read_bytes()).hexdigest()

    ledger_dir = tmp_path / "model-ledger" / "policy_model"
    ledger_dir.mkdir(parents=True)
    lineage_path = ledger_dir / "group-7_g0.lineage.jsonl"
    lineage_path.write_text(
        json.dumps({"kind": "request", "staging_key": None})
        + "\n"
        + json.dumps(
            {
                "kind": "response",
                "model_call_id": "call-1",
                "staging_key": "group-7_g0/call-1",
                "staging_chain": [
                    "group-7_g0/source-call",
                    "group-7_g0/call-1",
                ],
            }
        )
        + "\n"
    )
    lineage_digest = hashlib.sha256(lineage_path.read_bytes()).hexdigest()
    completed_lineage_path = ledger_dir / "completed_g1.lineage.jsonl"
    completed_lineage_path.write_text(
        json.dumps(
            {
                "kind": "response",
                "staging_key": "completed_g1/call-1",
            }
        )
        + "\n"
    )
    completed_lineage_digest = hashlib.sha256(
        completed_lineage_path.read_bytes()
    ).hexdigest()
    manifest_path = ledger_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_id": "snapshot-7",
                "server_name": "policy_model",
                "rollouts": {
                    "group-7_g0": {
                        "files": {lineage_path.name: lineage_digest},
                        "rows": 2,
                    },
                    # Model lineage can outlive an acknowledged completed
                    # execution. It must not keep that execution's old staging
                    # tensors alive when no parked agent boundary references it.
                    "completed_g1": {
                        "files": {
                            completed_lineage_path.name: completed_lineage_digest
                        },
                        "rows": 1,
                    },
                },
                "tombstones": [],
                "source_attempts": [],
            }
        )
    )
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    reference_path = ledger_dir / "storage-references.jsonl"
    reference_rows = [
        {
            "schema_version": 1,
            "capture_key": "group-7_g0",
            "boundary_model_call_id": "call-1",
            "kind": "token_capture_staging",
            "key": key,
        }
        for key in ("group-7_g0/source-call", "group-7_g0/call-1")
    ]
    reference_payload = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n"
        for row in reference_rows
    )
    reference_path.write_bytes(reference_payload)
    storage_reference = {
        "schema_version": 1,
        "relative_path": "model-ledger/policy_model/storage-references.jsonl",
        "sha256": hashlib.sha256(reference_payload).hexdigest(),
        "records": len(reference_rows),
        "bytes": len(reference_payload),
    }
    checkpoint = GymCheckpointCommitResult.model_validate(
        {
            "checkpoint_id": "snapshot-7",
            "participants": [
                {
                    "participant": {
                        "server_name": "policy-route",
                        "component": "responses_api_models",
                        "participant_name": "policy_model",
                    },
                    "payload": {
                        "rollouts": 2,
                        "rows": 3,
                        "excluded_tombstoned": 0,
                        "manifest_digest": manifest_digest,
                        "storage_reference_index": storage_reference,
                    },
                    "manifest": {
                        "participant": {
                            "server_name": "policy-route",
                            "component": "responses_api_models",
                            "participant_name": "policy_model",
                        },
                        "relative_path": "model-ledger/policy_model/manifest.json",
                        "manifest_digest": manifest_digest,
                    },
                },
                {
                    "participant": {
                        "server_name": "agent-route",
                        "component": "responses_api_agents",
                        "participant_name": "agent",
                    },
                    "payload": {
                        "records": 1,
                        "manifest_digest": agent_manifest_digest,
                        "continuation_index": continuation_reference,
                    },
                    "manifest": {
                        "participant": {
                            "server_name": "agent-route",
                            "component": "responses_api_agents",
                            "participant_name": "agent",
                        },
                        "relative_path": "agent/instance-test/manifest.json",
                        "manifest_digest": agent_manifest_digest,
                    },
                },
            ],
        }
    )

    assert gym_checkpoint_staging_keys(tmp_path, checkpoint) == {
        "group-7_g0/source-call",
        "group-7_g0/call-1",
    }
    assert gym_checkpoint_continuations(tmp_path, checkpoint) == (
        GymCheckpointContinuation(
            rollout_id="group-7_g0",
            source_attempt_index=0,
            capture_key="group-7_g0",
            resource_state_revisions=(("durable-tools", 2),),
            staging_keys=("group-7_g0/call-1", "group-7_g0/source-call"),
        ),
    )

    lineage_path.write_text("{}\n")
    assert gym_checkpoint_staging_keys(tmp_path, checkpoint) == {
        "group-7_g0/source-call",
        "group-7_g0/call-1",
    }


@pytest.mark.parametrize(
    ("response_type", "payload", "missing_field"),
    [
        (
            GymModelCommitResponse,
            {
                "rollouts": 0,
                "rows": 0,
                "excluded_tombstoned": 0,
                "manifest_digest": "a" * 64,
            },
            "storage_reference_index",
        ),
        (
            GymAgentCommitResponse,
            {"records": 0, "manifest_digest": "a" * 64},
            "continuation_index",
        ),
        (
            GymModelRestoreResponse,
            {
                "rollouts": 0,
                "rows": 0,
                "tombstones": [],
                "source_attempts": [],
            },
            "storage_reference_index",
        ),
        (
            GymAgentRestoreResponse,
            {"records": 0, "source_checkpoint_id": "snapshot-7"},
            "continuation_index",
        ),
    ],
)
def test_checkpoint_wire_contract_requires_artifact_indexes(
    response_type, payload, missing_field
) -> None:
    with pytest.raises(ValidationError, match=missing_field):
        response_type.model_validate(payload)


def test_storage_reference_index_avoids_private_lineage_scan(tmp_path) -> None:
    ledger_dir = tmp_path / "model-ledger" / "policy_model"
    ledger_dir.mkdir(parents=True)
    reference_path = ledger_dir / "storage-references.jsonl"
    rows = [
        {
            "schema_version": 1,
            "capture_key": "group-7_g0",
            "boundary_model_call_id": "call-1",
            "kind": "token_capture_staging",
            "key": "group-7_g0/source-call",
        },
        {
            "schema_version": 1,
            "capture_key": "group-7_g0",
            "boundary_model_call_id": "call-1",
            "kind": "token_capture_staging",
            "key": "group-7_g0/call-1",
        },
    ]
    reference_payload = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    reference_path.write_bytes(reference_payload)
    reference = {
        "schema_version": 1,
        "relative_path": "model-ledger/policy_model/storage-references.jsonl",
        "sha256": hashlib.sha256(reference_payload).hexdigest(),
        "records": len(rows),
        "bytes": len(reference_payload),
    }
    # Deliberately omit Gym's private rollout/file inventory. RL should need
    # only the digest-bound public sidecar for external TQ ownership.
    manifest_path = ledger_dir / "manifest.json"
    manifest_path.write_text("{}")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    participant = {
        "server_name": "policy-route",
        "component": "responses_api_models",
        "participant_name": "policy_model",
    }
    checkpoint = GymCheckpointCommitResult.model_validate(
        {
            "checkpoint_id": "snapshot-7",
            "participants": [
                {
                    "participant": participant,
                    "payload": {
                        "rollouts": 1,
                        "rows": 2,
                        "excluded_tombstoned": 0,
                        "excluded_inactive": 3,
                        "manifest_digest": manifest_digest,
                        "storage_reference_index": reference,
                    },
                    "manifest": {
                        "participant": participant,
                        "relative_path": "model-ledger/policy_model/manifest.json",
                        "manifest_digest": manifest_digest,
                    },
                }
            ],
        }
    )

    assert gym_checkpoint_staging_keys(tmp_path, checkpoint) == {
        "group-7_g0/source-call",
        "group-7_g0/call-1",
    }

    reference_path.write_bytes(reference_payload + b"\n")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        gym_checkpoint_staging_keys(tmp_path, checkpoint)


def test_restore_must_report_the_committed_artifact_coordinates() -> None:
    participant = {
        "server_name": "policy-route",
        "component": "responses_api_models",
        "participant_name": "policy_model",
    }
    storage_reference = {
        "schema_version": 1,
        "relative_path": "model-ledger/policy_model/storage-references.jsonl",
        "sha256": "d" * 64,
        "records": 2,
        "bytes": 128,
    }
    committed = GymCheckpointCommitResult.model_validate(
        {
            "checkpoint_id": "snapshot-7",
            "participants": [
                {
                    "participant": participant,
                    "payload": {
                        "rollouts": 1,
                        "rows": 2,
                        "excluded_tombstoned": 0,
                        "manifest_digest": "a" * 64,
                        "storage_reference_index": storage_reference,
                    },
                    "manifest": {
                        "participant": participant,
                        "relative_path": "model-ledger/policy_model/manifest.json",
                        "manifest_digest": "a" * 64,
                    },
                }
            ],
        }
    )
    restored = GymCheckpointRestoreResult.model_validate(
        {
            "checkpoint_id": "restore-7",
            "participants": [
                {
                    "participant": participant,
                    "payload": {
                        "rollouts": 1,
                        "rows": 2,
                        "checkpoint_id": "snapshot-7",
                        "tombstones": [],
                        "source_attempts": [],
                        "storage_reference_index": storage_reference,
                    },
                }
            ],
        }
    )

    validate_gym_checkpoint_restore_artifacts(committed, restored)
    mismatched = GymCheckpointRestoreResult.model_validate(
        {
            "checkpoint_id": "restore-7",
            "participants": [
                {
                    "participant": participant,
                    "payload": {
                        "rollouts": 1,
                        "rows": 2,
                        "checkpoint_id": "snapshot-7",
                        "tombstones": [],
                        "source_attempts": [],
                        "storage_reference_index": {
                            **storage_reference,
                            "sha256": "e" * 64,
                        },
                    },
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="different checkpoint artifact"):
        validate_gym_checkpoint_restore_artifacts(committed, mismatched)


def test_checkpoint_commit_rejects_mismatched_manifest_identity() -> None:
    with pytest.raises(ValidationError, match="manifest identity does not match"):
        GymCheckpointCommitResult.model_validate(
            {
                "checkpoint_id": "snapshot-7",
                "participants": [
                    {
                        "participant": {
                            "server_name": "agent-route",
                            "component": "responses_api_agents",
                            "participant_name": "agent-a",
                        },
                        "payload": {
                            "records": 1,
                            "manifest_digest": "a" * 64,
                            "continuation_index": {
                                "schema_version": 1,
                                "relative_path": "agent/continuations.jsonl",
                                "sha256": "b" * 64,
                                "records": 1,
                                "bytes": 64,
                            },
                        },
                        "manifest": {
                            "participant": {
                                "server_name": "agent-route",
                                "component": "responses_api_agents",
                                "participant_name": "agent-b",
                            },
                            "relative_path": "agent/manifest.json",
                            "manifest_digest": "a" * 64,
                        },
                    }
                ],
            }
        )


def test_topology_requires_every_stateful_checkpoint_participant() -> None:
    topology = GymCheckpointTopology.model_validate(
        {
            "participants": [
                {
                    "participant": {
                        "server_name": "policy-route",
                        "component": "responses_api_models",
                        "participant_name": "policy",
                    },
                    "schema_version": 1,
                    "admission_states": ["accepting", "draining", "paused"],
                    "checkpoint_mode": "export_restore",
                    "concurrency_contract": "stateless",
                    "multi_process": {"mode": "single_worker", "num_workers": 1},
                    "instance_role": "policy",
                },
                {
                    "participant": {
                        "server_name": "judge-route",
                        "component": "responses_api_models",
                        "participant_name": "judge",
                    },
                    "schema_version": 1,
                    "admission_states": ["accepting"],
                    "checkpoint_mode": "stateless",
                    "concurrency_contract": "stateless",
                    "multi_process": {"mode": "single_worker", "num_workers": 1},
                    "instance_role": "auxiliary",
                },
            ]
        }
    )
    checkpoint = GymCheckpointCommitResult.model_validate(
        {"checkpoint_id": "checkpoint-7", "participants": []}
    )

    with pytest.raises(ValueError, match="missing=.*policy-route"):
        topology.validate_checkpoint_participants(checkpoint)


def test_topology_reports_restart_only_resources() -> None:
    topology = GymCheckpointTopology.model_validate(
        {
            "participants": [
                {
                    "participant": {
                        "server_name": "stateful-tools",
                        "component": "resources_servers",
                        "participant_name": "stateful-tools",
                    },
                    "schema_version": 1,
                    "admission_states": ["accepting"],
                    "checkpoint_mode": "restart_only",
                    "concurrency_contract": "stateless",
                    "multi_process": {"mode": "single_worker", "num_workers": 1},
                },
                {
                    "participant": {
                        "server_name": "stateless-tools",
                        "component": "resources_servers",
                        "participant_name": "stateless-tools",
                    },
                    "schema_version": 1,
                    "admission_states": ["accepting"],
                    "checkpoint_mode": "stateless",
                    "concurrency_contract": "stateless",
                    "multi_process": {"mode": "single_worker", "num_workers": 1},
                },
            ]
        }
    )

    assert topology.restart_only_resources() == ["stateful-tools"]
