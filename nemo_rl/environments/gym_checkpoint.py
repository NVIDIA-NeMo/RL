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
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

GYM_CHECKPOINT_SCHEMA_VERSION = 1
GYM_CHECKPOINT_CONTROL_PREFIX = "/ng-control/v1"
GYM_CHECKPOINT_CAPABILITIES_PATH = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/capabilities"
GYM_MODEL_ADMISSION_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-admission"
GYM_MODEL_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-checkpoint"
GYM_AGENT_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/agent-checkpoint"
GYM_RESOURCES_CHECKPOINT_PREFIX = (
    f"{GYM_CHECKPOINT_CONTROL_PREFIX}/resources-checkpoint"
)

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


class GymExecutionIdentity(_StrictWireModel):
    """Stable logical rollout identity plus one physical execution number."""

    rollout_id: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    attempt_index: NonNegativeInt

    @property
    def capture_key(self) -> str:
        """Return Gym's attempt-qualified token-capture and routing key."""
        return gym_capture_key(self.rollout_id, self.attempt_index)


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
    checkpoint_mode: Literal["stateless", "export_restore"]
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
    checkpoint_mode: Literal["stateless", "export_restore"]
    concurrency_contract: Literal[
        "stateless",
        "serialized_per_session",
        "transactional_parallel",
    ]
    multi_process: GymMultiProcessCapability
    instance_role: Literal["policy", "auxiliary"] | None = None

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
            admission_states=sorted(capabilities.admission_states),
            checkpoint_mode=capabilities.checkpoint_mode,
            concurrency_contract=capabilities.concurrency_contract,
            multi_process=capabilities.multi_process,
            instance_role=capabilities.instance_role,
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
        contracts = [
            GymCheckpointParticipantContract.from_discovered(participant)
            for participant in participants
        ]
        contracts.sort(
            key=lambda item: (
                item.participant.component,
                item.participant.server_name,
                item.participant.participant_name,
            )
        )
        return cls(participants=contracts)

    def fingerprint(self) -> str:
        """Return a canonical digest without runtime routing or credentials."""
        payload = json.dumps(
            self.model_dump(mode="json"),
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


class GymWorkerAcknowledgements(_StrictWireModel):
    acknowledged: NonNegativeInt
    expected: NonNegativeInt


class GymModelPrepareResponse(_StrictWireModel):
    state: Literal["accepting", "draining", "paused"]
    workers: GymWorkerAcknowledgements
    inflight_total: NonNegativeInt
    waiters_total: NonNegativeInt


class GymModelInflightRequest(_StrictWireModel):
    rollout_id: str | None = Field(default=None, pattern=_IDENTITY_PATTERN)
    attempt_index: NonNegativeInt | None = None
    plane: str | None
    age_seconds: NonNegativeFloat


class GymSingleWorkerModelStatus(_StrictWireModel):
    state: Literal["accepting", "draining", "paused"]
    inflight: NonNegativeInt


class GymSingleWorkerModelStatusResponse(_StrictWireModel):
    checkpoint_id: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    state: Literal["accepting", "draining", "paused"]
    per_worker: dict[str, GymSingleWorkerModelStatus]
    inflight_total: NonNegativeInt
    waiters_total: NonNegativeInt
    inflight: list[GymModelInflightRequest]
    tombstones: list[GymExecutionIdentity]


class GymCoordinatorWorkers(_StrictWireModel):
    acknowledged: NonNegativeInt
    expected: PositiveInt
    live: NonNegativeInt


class GymCoordinatorWorkerStatus(_StrictWireModel):
    acked_seq: NonNegativeInt
    inflight: NonNegativeInt
    connected: bool


class GymCoordinatorModelStatusResponse(_StrictWireModel):
    state: Literal["accepting", "draining", "paused"]
    workers: GymCoordinatorWorkers
    missing_workers: NonNegativeInt
    inflight_total: NonNegativeInt
    waiters_total: NonNegativeInt
    per_worker: dict[str, GymCoordinatorWorkerStatus]


class GymAgentExecutionStatus(GymExecutionIdentity):
    generation: PositiveInt
    state: Literal[
        "running",
        "park_requested",
        "parked",
        "completed",
        "retired",
    ]
    parked_boundary_state: (
        Literal[
            "parked_with_boundary",
            "parked_without_boundary",
        ]
        | None
    )
    boundary_index: NonNegativeInt | None = None
    age_seconds: NonNegativeFloat


class GymAgentPrepareResponse(_StrictWireModel):
    state: Literal["accepting", "preparing"]
    ready_to_commit: bool
    running: NonNegativeInt
    parked: NonNegativeInt
    parked_with_boundary: NonNegativeInt
    parked_without_boundary: NonNegativeInt
    completed_unacknowledged: NonNegativeInt
    active: NonNegativeInt
    blocking_attempts: list[GymAgentExecutionStatus]
    completed_unacknowledged_attempts: list[GymAgentExecutionStatus]
    executions: list[GymAgentExecutionStatus]


class GymResourcesPrepareResponse(_StrictWireModel):
    sessions: NonNegativeInt
    state: Literal["prepared"]


GymPreparePayload: TypeAlias = Annotated[
    GymModelPrepareResponse | GymAgentPrepareResponse | GymResourcesPrepareResponse,
    Field(union_mode="left_to_right"),
]


class GymParticipantPrepareResult(_StrictWireModel):
    participant: GymParticipantIdentity
    ready: bool
    payload: GymPreparePayload


class GymCheckpointPrepareResult(_StrictWireModel):
    """One complete, safe checkpoint boundary across discovered participants."""

    checkpoint_id: str
    ready: bool
    participants: list[GymParticipantPrepareResult]


class GymModelCommitResponse(_StrictWireModel):
    rollouts: NonNegativeInt
    rows: NonNegativeInt
    excluded_tombstoned: NonNegativeInt
    manifest_digest: Sha256Digest


class GymAgentCommitResponse(_StrictWireModel):
    records: NonNegativeInt
    manifest_digest: Sha256Digest


class GymResourcesCommitResponse(_StrictWireModel):
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
    participant: GymParticipantIdentity
    payload: GymCommitPayload
    manifest: GymParticipantManifestReference


class GymCheckpointCommitResult(_StrictWireModel):
    checkpoint_id: str
    participants: list[GymParticipantCommitResult]


class GymModelRestoreResponse(_StrictWireModel):
    rollouts: NonNegativeInt
    rows: NonNegativeInt
    checkpoint_id: str | None = None
    tombstones: list[GymExecutionIdentity]
    source_attempts: list[GymExecutionIdentity]


class GymAgentRestoreResponse(_StrictWireModel):
    records: NonNegativeInt
    source_checkpoint_id: str = Field(min_length=1)


class GymResourcesRestoreResponse(_StrictWireModel):
    sessions: NonNegativeInt
    source_checkpoint_id: str = Field(min_length=1)


GymRestorePayload: TypeAlias = Annotated[
    GymModelRestoreResponse | GymAgentRestoreResponse | GymResourcesRestoreResponse,
    Field(union_mode="left_to_right"),
]


class GymParticipantRestoreResult(_StrictWireModel):
    participant: GymParticipantIdentity
    payload: GymRestorePayload


class GymCheckpointRestoreResult(_StrictWireModel):
    checkpoint_id: str
    participants: list[GymParticipantRestoreResult]


class GymModelResumeResponse(_StrictWireModel):
    state: Literal["accepting"]
    workers: GymWorkerAcknowledgements
    released_waiters: NonNegativeInt


class GymAgentResumeResponse(_StrictWireModel):
    state: Literal["accepting"]
    released: NonNegativeInt


class GymResourcesResumeResponse(_StrictWireModel):
    state: Literal["accepting"]


GymResumePayload: TypeAlias = Annotated[
    GymModelResumeResponse | GymAgentResumeResponse | GymResourcesResumeResponse,
    Field(union_mode="left_to_right"),
]


class GymParticipantResumeResult(_StrictWireModel):
    participant: GymParticipantIdentity
    payload: GymResumePayload


class GymCheckpointResumeResult(_StrictWireModel):
    checkpoint_id: str
    participants: list[GymParticipantResumeResult]
