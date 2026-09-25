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

"""Versioned NeMo-Gym checkpoint wire contracts owned by NeMo-RL.

NeMo-Gym's checkpoint package is currently experimental.  Keeping these
models in NeMo-RL makes the HTTP boundary explicit and prevents a Gym package
refactor from silently changing a durable RL checkpoint protocol.

Models described as persisted are serialized into the rollout snapshot.
Changing their fields or invariants requires reviewing, and normally bumping,
the outer rollout-checkpoint schema version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

GYM_CHECKPOINT_SCHEMA_VERSION = 1
GYM_CHECKPOINT_CONTROL_PREFIX = "/ng-control/v1"
GYM_CHECKPOINT_CAPABILITIES_PATH = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/capabilities"
GYM_MODEL_ADMISSION_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-admission"
GYM_MODEL_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-checkpoint"
GYM_AGENT_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/agent-checkpoint"
GYM_RESOURCES_CHECKPOINT_PREFIX = (
    f"{GYM_CHECKPOINT_CONTROL_PREFIX}/resources-checkpoint"
)
GYM_AGENT_CONTINUATION_INDEX_FEATURE = "agent_continuation_index_v1"
GYM_EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE = "external_storage_reference_index_v1"

_IDENTITY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

GymComponent: TypeAlias = Literal[
    "responses_api_models",
    "responses_api_agents",
    "resources_servers",
]
NonNegativeInt: TypeAlias = Annotated[int, Field(strict=True, ge=0)]
PositiveInt: TypeAlias = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0)]
Sha256Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _StrictWireModel(BaseModel):
    """Reject protocol drift instead of accepting an ambiguous checkpoint."""

    model_config = ConfigDict(extra="forbid")


class _LiveResponseWireModel(BaseModel):
    """Validate required live fields while tolerating additive telemetry."""

    model_config = ConfigDict(extra="ignore")


class GymExecutionIdentity(_StrictWireModel):
    """Stable logical rollout identity plus one physical execution number."""

    rollout_id: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    attempt_index: NonNegativeInt

    @property
    def capture_key(self) -> str:
        """Return Gym's attempt-qualified token-capture and routing key."""
        return gym_capture_key(self.rollout_id, self.attempt_index)


class GymCompletionReceipt(GymExecutionIdentity):
    """Exact Gym-issued proof naming one retained terminal result."""

    execution_generation: PositiveInt
    result_identity: str = Field(min_length=1, max_length=512)
    result_digest: Sha256Digest
    manifest_capture_key: str | None = Field(
        default=None,
        min_length=1,
        pattern=_IDENTITY_PATTERN,
    )
    terminal_model_call_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_model_lineage_coordinate(self) -> "GymCompletionReceipt":
        if (self.manifest_capture_key is None) != (self.terminal_model_call_id is None):
            raise ValueError(
                "completion receipt model-lineage capture key and terminal call "
                "id must be supplied together"
            )
        return self


def gym_capture_key(logical_rollout_id: str, attempt_index: int) -> str:
    """Derive the capture key defined by Gym's stable-identity protocol."""
    identity = GymExecutionIdentity(
        rollout_id=logical_rollout_id,
        attempt_index=attempt_index,
    )
    if identity.attempt_index == 0:
        return identity.rollout_id
    return f"{identity.rollout_id}-a{identity.attempt_index}"


class GymMultiProcessCapability(_StrictWireModel):
    """How one Gym service coordinates checkpoint state across workers."""

    mode: Literal["single_worker", "coordinator", "unmanaged"]
    num_workers: PositiveInt


class GymParticipantIdentity(_StrictWireModel):
    """NeMo-RL routing name and Gym-reported participant identity."""

    server_name: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    component: GymComponent
    participant_name: str = Field(min_length=1)


class GymControlCapabilities(_StrictWireModel):
    """Response from ``GET /ng-control/v1/capabilities``."""

    component: GymComponent
    name: str = Field(min_length=1)
    schema_version: Literal[1]
    admission_states: list[Literal["accepting", "draining", "paused"]]
    checkpoint_mode: Literal["stateless", "restart_only", "export_restore"]
    concurrency_contract: Literal[
        "stateless",
        "serialized_per_session",
        "transactional_parallel",
    ]
    multi_process: GymMultiProcessCapability
    instance_role: Literal["policy", "auxiliary"] | None = None
    phase: Literal[
        "idle",
        "preparing",
        "prepared",
        "committing",
        "committed_paused",
        "restoring",
        "restore_failed_paused",
        "restored_paused",
    ]
    active_checkpoint_id: str | None = None
    deadline_ts: FiniteFloat | None = None
    # Capabilities are additive. Older NeMo-RL clients must tolerate features
    # advertised by a newer Gym and explicitly check only the ones they require.
    features: list[str] = Field(default_factory=list)

    def participant(self, server_name: str) -> GymParticipantIdentity:
        """Bind Gym's reported identity to its NeMo-RL routing name."""
        return GymParticipantIdentity(
            server_name=server_name,
            component=self.component,
            participant_name=self.name,
        )


class GymDiscoveredParticipant(_StrictWireModel):
    """One routable Gym participant and its validated capabilities."""

    participant: GymParticipantIdentity
    capabilities: GymControlCapabilities


class GymCheckpointParticipantContract(_StrictWireModel):
    """Credential-free participant properties that must match on restore."""

    participant: GymParticipantIdentity
    schema_version: Literal[1]
    admission_states: list[Literal["accepting", "draining", "paused"]]
    checkpoint_mode: Literal["stateless", "restart_only", "export_restore"]
    concurrency_contract: Literal[
        "stateless",
        "serialized_per_session",
        "transactional_parallel",
    ]
    multi_process: GymMultiProcessCapability
    instance_role: Literal["policy", "auxiliary"] | None = None
    features: list[str] = Field(default_factory=list)

    @classmethod
    def from_discovered(
        cls,
        discovered: GymDiscoveredParticipant,
    ) -> "GymCheckpointParticipantContract":
        """Project one dynamic capability response onto restore semantics."""
        capabilities = discovered.capabilities
        return cls(
            participant=discovered.participant,
            schema_version=capabilities.schema_version,
            admission_states=cast(
                list[Literal["accepting", "draining", "paused"]],
                sorted(capabilities.admission_states),
            ),
            checkpoint_mode=capabilities.checkpoint_mode,
            concurrency_contract=capabilities.concurrency_contract,
            multi_process=capabilities.multi_process,
            instance_role=capabilities.instance_role,
            features=sorted(capabilities.features),
        )


class GymCheckpointTopology(_StrictWireModel):
    """Stable participant topology cached by setup and bound to snapshots."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    participants: list[GymCheckpointParticipantContract]

    @classmethod
    def from_discovered(
        cls,
        participants: list[GymDiscoveredParticipant],
    ) -> "GymCheckpointTopology":
        """Build a deterministically ordered topology from discovery results."""
        component_order = {
            "responses_api_agents": 0,
            "responses_api_models": 1,
            "resources_servers": 2,
        }
        contracts = [
            GymCheckpointParticipantContract.from_discovered(participant)
            for participant in participants
        ]
        contracts.sort(
            key=lambda item: (
                component_order[item.participant.component],
                item.participant.server_name,
                item.participant.participant_name,
            )
        )
        return cls(participants=contracts)

    def fingerprint(self) -> str:
        """Return a canonical digest without runtime or additive capabilities."""
        compatibility_identity = {
            "schema_version": self.schema_version,
            "participants": [
                participant.model_dump(mode="json", exclude={"features"})
                for participant in self.participants
            ],
        }
        payload = json.dumps(
            compatibility_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class GymCheckpointControlRequest(_StrictWireModel):
    """Fields shared by all Gym checkpoint control requests."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTITY_PATTERN,
    )
    deadline_ts: FiniteFloat


class GymCheckpointDirectoryRequest(GymCheckpointControlRequest):
    """Checkpoint request that reads or writes one shared snapshot directory."""

    checkpoint_dir: str = Field(min_length=1)


class GymCheckpointArtifactReference(_StrictWireModel):
    """Digest-bound coordinate for a Gym-owned checkpoint sidecar."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    relative_path: str = Field(min_length=1)
    sha256: Sha256Digest
    records: NonNegativeInt
    bytes: NonNegativeInt

    @model_validator(mode="after")
    def validate_relative_path(self) -> "GymCheckpointArtifactReference":
        path = Path(self.relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("Gym checkpoint artifact path must be safely relative")
        return self


class GymAgentCheckpointDirectoryRequest(GymCheckpointDirectoryRequest):
    """Agent commit/restore request returning continuation coordinates."""


class GymModelCheckpointCommitRequest(GymCheckpointDirectoryRequest):
    """Model commit request scoped to agent-owned continuation roots."""

    continuation_indexes: list[GymCheckpointArtifactReference]


class GymModelCheckpointRestoreRequest(GymCheckpointDirectoryRequest):
    """Model restore request returning its external-storage index."""


class GymWorkerAcknowledgements(_StrictWireModel):
    """Number of model workers that acknowledged a control operation."""

    acknowledged: NonNegativeInt
    expected: NonNegativeInt


class GymModelPrepareResponse(_LiveResponseWireModel):
    """Policy-model admission state after a checkpoint prepare request."""

    state: Literal["accepting", "draining", "paused"]
    workers: GymWorkerAcknowledgements
    inflight_total: NonNegativeInt
    response_inflight_total: NonNegativeInt | None = None
    generation_pending_total: NonNegativeInt | None = None
    waiters_total: NonNegativeInt


class GymModelInflightRequest(_LiveResponseWireModel):
    """Diagnostic identity and age of one live policy-model request."""

    rollout_id: str | None = Field(default=None, pattern=_IDENTITY_PATTERN)
    attempt_index: NonNegativeInt | None = None
    plane: str | None
    age_seconds: NonNegativeFloat


class GymSingleWorkerModelStatus(_LiveResponseWireModel):
    """Live admission and request counts for one policy-model worker."""

    state: Literal["accepting", "draining", "paused"]
    inflight: NonNegativeInt


class GymSingleWorkerModelStatusResponse(_LiveResponseWireModel):
    """Read-only checkpoint status for a single-worker policy service."""

    checkpoint_id: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    state: Literal["accepting", "draining", "paused"]
    per_worker: dict[str, GymSingleWorkerModelStatus]
    inflight_total: NonNegativeInt
    response_inflight_total: NonNegativeInt | None = None
    generation_pending_total: NonNegativeInt | None = None
    waiters_total: NonNegativeInt
    inflight: list[GymModelInflightRequest]
    tombstones: list[GymExecutionIdentity]


class GymCoordinatorWorkers(_LiveResponseWireModel):
    """Live acknowledgement counts for a coordinated policy service."""

    acknowledged: NonNegativeInt
    expected: PositiveInt
    live: NonNegativeInt


class GymCoordinatorWorkerStatus(_LiveResponseWireModel):
    """Read-only checkpoint status reported for one coordinated worker."""

    acked_seq: NonNegativeInt
    inflight: NonNegativeInt
    generation_pending: NonNegativeInt | None = None
    # RL only observes proof presence here; Gym owns the nested proof schema.
    generation_cut_proof: dict[str, object] | None = None
    proof_error: str | None = None
    connected: bool


class GymCoordinatorModelStatusResponse(_LiveResponseWireModel):
    """Read-only aggregate checkpoint status for a coordinated model."""

    state: Literal["accepting", "draining", "paused"]
    workers: GymCoordinatorWorkers
    missing_workers: NonNegativeInt
    inflight_total: NonNegativeInt
    response_inflight_total: NonNegativeInt | None = None
    generation_pending_total: NonNegativeInt | None = None
    waiters_total: NonNegativeInt
    per_worker: dict[str, GymCoordinatorWorkerStatus]


class GymAgentExecutionStatus(GymExecutionIdentity):
    """Live checkpoint state for one agent execution."""

    generation: PositiveInt
    state: Literal[
        "running",
        "park_requested",
        "parked",
        "external_wait_frozen",
        "completed",
        "retired",
    ]
    parked_boundary_state: (
        Literal[
            "parked_with_boundary",
            "parked_without_boundary",
            "external_wait_frozen",
        ]
        | None
    )
    boundary_index: NonNegativeInt | None = None
    turn_index: NonNegativeInt | None = None
    boundary_kind: Literal["pending_model", "turn_complete"] | None = None
    resource_state_revisions: dict[str, NonNegativeInt] = Field(default_factory=dict)
    completion_receipt: GymCompletionReceipt | None = None
    age_seconds: NonNegativeFloat


class GymAgentSelectedBoundary(GymExecutionIdentity):
    """One agent boundary selected into the current checkpoint cut."""

    boundary_index: NonNegativeInt
    turn_index: NonNegativeInt
    boundary_kind: Literal["pending_model", "turn_complete"]
    resource_state_revisions: dict[str, NonNegativeInt]


class GymAgentPrepareResponse(_StrictWireModel):
    """Agent inventory proving whether its checkpoint cut is safe to commit."""

    state: Literal["accepting", "preparing"]
    ready_to_commit: bool
    running: NonNegativeInt
    parked: NonNegativeInt
    parked_with_boundary: NonNegativeInt
    parked_without_boundary: NonNegativeInt
    completed_unacknowledged: NonNegativeInt
    acknowledged_completed: NonNegativeInt
    active: NonNegativeInt
    blocking_attempts: list[GymAgentExecutionStatus]
    completed_unacknowledged_attempts: list[GymAgentExecutionStatus]
    selected_boundaries: list[GymAgentSelectedBoundary]
    executions: list[GymAgentExecutionStatus]


class GymAgentStatusResponse(GymAgentPrepareResponse):
    """Agent prepare state returned by the read-only status route."""

    checkpoint_id: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)


class GymResourcesPrepareInventoryEntry(GymExecutionIdentity):
    """One resources session selected into the current checkpoint cut."""

    revision: NonNegativeInt
    mutation_receipts: NonNegativeInt


class GymResourcesPrepareResponse(_StrictWireModel):
    """Resource-session inventory frozen by checkpoint preparation."""

    sessions: NonNegativeInt
    state: Literal["prepared"]
    inventory: list[GymResourcesPrepareInventoryEntry]

    @model_validator(mode="after")
    def validate_inventory_count(self) -> "GymResourcesPrepareResponse":
        if self.sessions != len(self.inventory):
            raise ValueError(
                "resources checkpoint session count does not match inventory: "
                f"sessions={self.sessions}, inventory={len(self.inventory)}"
            )
        return self


GymPreparePayload: TypeAlias = Annotated[
    GymModelPrepareResponse | GymAgentPrepareResponse | GymResourcesPrepareResponse,
    Field(union_mode="left_to_right"),
]


class GymParticipantPrepareResult(_StrictWireModel):
    """Normalized prepare result for one discovered Gym participant."""

    participant: GymParticipantIdentity
    ready: bool
    payload: GymPreparePayload


class GymCheckpointPrepareResult(_StrictWireModel):
    """One complete, safe checkpoint boundary across discovered participants."""

    checkpoint_id: str
    ready: bool
    participants: list[GymParticipantPrepareResult]


class GymModelCommitResponse(_StrictWireModel):
    """Policy lineage and external-storage index written during commit."""

    rollouts: NonNegativeInt
    rows: NonNegativeInt
    excluded_tombstoned: NonNegativeInt
    excluded_inactive: NonNegativeInt = 0
    manifest_digest: Sha256Digest
    storage_reference_index: GymCheckpointArtifactReference


class GymAgentCommitResponse(_StrictWireModel):
    """Agent continuation index written during checkpoint commit."""

    records: NonNegativeInt
    manifest_digest: Sha256Digest
    continuation_index: GymCheckpointArtifactReference


class GymResourcesCommitResponse(_StrictWireModel):
    """Resource-session state written during checkpoint commit."""

    sessions: NonNegativeInt
    manifest_digest: Sha256Digest


GymCommitPayload: TypeAlias = Annotated[
    GymModelCommitResponse | GymAgentCommitResponse | GymResourcesCommitResponse,
    Field(union_mode="left_to_right"),
]


class GymParticipantManifestReference(_StrictWireModel):
    """Digest-bound participant output included by a future outer manifest."""

    participant: GymParticipantIdentity
    relative_path: str = Field(min_length=1)
    manifest_digest: Sha256Digest


class GymParticipantCommitResult(_StrictWireModel):
    """Persisted participant payload bound to its digest-checked manifest."""

    participant: GymParticipantIdentity
    payload: GymCommitPayload
    manifest: GymParticipantManifestReference


class GymCheckpointCommitResult(_StrictWireModel):
    """Persisted Gym commit included in the outer rollout snapshot manifest.

    Participant identities must be unique and must match their manifest
    identities. Field changes require rollout-checkpoint schema review.
    """

    checkpoint_id: str
    participants: list[GymParticipantCommitResult]

    @model_validator(mode="after")
    def validate_participants(self) -> "GymCheckpointCommitResult":
        identities: list[tuple[str, str, str]] = []
        for result in self.participants:
            if result.manifest.participant != result.participant:
                raise ValueError(
                    "Gym participant manifest identity does not match its commit "
                    f"result: participant={result.participant!r}, "
                    f"manifest={result.manifest.participant!r}"
                )
            participant = result.participant
            identities.append(
                (
                    participant.server_name,
                    participant.component,
                    participant.participant_name,
                )
            )
        if len(identities) != len(set(identities)):
            raise ValueError("Gym checkpoint commit contains duplicate participants")
        return self


class GymModelRestoreResponse(_StrictWireModel):
    """Policy lineage and external-storage index installed during restore."""

    rollouts: NonNegativeInt
    rows: NonNegativeInt
    checkpoint_id: str | None = None
    tombstones: list[GymExecutionIdentity]
    source_attempts: list[GymExecutionIdentity]
    storage_reference_index: GymCheckpointArtifactReference


class GymAgentRestoreResponse(_StrictWireModel):
    """Agent continuation index installed from a saved checkpoint."""

    records: NonNegativeInt
    source_checkpoint_id: str = Field(min_length=1)
    continuation_index: GymCheckpointArtifactReference


class GymResourcesRestoreResponse(_StrictWireModel):
    """Resource sessions installed from a saved checkpoint."""

    sessions: NonNegativeInt
    source_checkpoint_id: str = Field(min_length=1)


GymRestorePayload: TypeAlias = Annotated[
    GymModelRestoreResponse | GymAgentRestoreResponse | GymResourcesRestoreResponse,
    Field(union_mode="left_to_right"),
]


class GymParticipantRestoreResult(_StrictWireModel):
    """Normalized restore result for one discovered Gym participant."""

    participant: GymParticipantIdentity
    payload: GymRestorePayload


class GymCheckpointRestoreResult(_StrictWireModel):
    """Aggregate runtime proof that every persisted Gym participant restored."""

    checkpoint_id: str
    participants: list[GymParticipantRestoreResult]

    @model_validator(mode="after")
    def validate_participants(self) -> "GymCheckpointRestoreResult":
        identities = [
            (
                result.participant.server_name,
                result.participant.component,
                result.participant.participant_name,
            )
            for result in self.participants
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Gym checkpoint restore contains duplicate participants")
        return self


class GymModelResumeResponse(_StrictWireModel):
    """Policy-model admission state after checkpoint release."""

    state: Literal["accepting"]
    workers: GymWorkerAcknowledgements
    released_waiters: NonNegativeInt


class GymAgentResumeResponse(_StrictWireModel):
    """Number of parked agent executions released after checkpointing."""

    state: Literal["accepting"]
    released: NonNegativeInt


class GymResourcesResumeResponse(_StrictWireModel):
    """Resource-server admission state after checkpoint release."""

    state: Literal["accepting"]


GymResumePayload: TypeAlias = Annotated[
    GymModelResumeResponse | GymAgentResumeResponse | GymResourcesResumeResponse,
    Field(union_mode="left_to_right"),
]


class GymParticipantResumeResult(_StrictWireModel):
    """Normalized release result for one Gym participant."""

    participant: GymParticipantIdentity
    payload: GymResumePayload


class GymCheckpointResumeResult(_StrictWireModel):
    """Aggregate runtime proof that the Gym checkpoint fence was released."""

    checkpoint_id: str
    participants: list[GymParticipantResumeResult]
