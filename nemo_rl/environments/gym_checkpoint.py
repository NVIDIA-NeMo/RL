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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Mapping, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

GYM_CHECKPOINT_SCHEMA_VERSION = 1
GYM_CHECKPOINT_CONTROL_PREFIX = "/ng-control/v1"
GYM_CHECKPOINT_CAPABILITIES_PATH = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/capabilities"
GYM_MODEL_ADMISSION_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-admission"
GYM_MODEL_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-checkpoint"
GYM_AGENT_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/agent-checkpoint"
GYM_AGENT_COMPLETION_RECEIPT_PATH = f"{GYM_AGENT_CHECKPOINT_PREFIX}/completion-receipt"
GYM_AGENT_COMPLETION_ACK_PATH = f"{GYM_AGENT_CHECKPOINT_PREFIX}/acknowledge-completed"
GYM_AGENT_DISCARD_RESTORED_CONTINUATION_PATH = (
    f"{GYM_AGENT_CHECKPOINT_PREFIX}/discard-restored-continuation"
)
GYM_RESOURCES_CHECKPOINT_PREFIX = (
    f"{GYM_CHECKPOINT_CONTROL_PREFIX}/resources-checkpoint"
)
GYM_AGENT_CONTINUATION_INDEX_FEATURE = "agent_continuation_index_v1"
GYM_AGENT_DISCARD_RESTORED_CONTINUATION_FEATURE = "discard_restored_continuation_v1"
GYM_AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE = "agent_resource_dependency_index_v1"
GYM_EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE = "external_storage_reference_index_v1"
GYM_GENERATION_CUT_LINEAGE_FEATURE = "generation_cut_lineage_v1"

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


class GymActorExecutionState(str, Enum):
    """Publication state of one rollout invocation owned by the Gym actor."""

    RUNNING = "running"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class GymActorExecution:
    """One actor-local rollout execution tracked across a checkpoint fence."""

    identity: GymExecutionIdentity
    state: GymActorExecutionState


class GymActorExecutionRegistry:
    """Fence actor dispatch and retain the exact membership of a checkpoint cut."""

    _MAX_RETIRED_CHECKPOINTS = 256

    def __init__(self) -> None:
        self._live: dict[tuple[str, int], GymActorExecution] = {}
        self._frozen_checkpoint_id: str | None = None
        self._frozen_membership: tuple[GymActorExecution, ...] = ()
        self._retired_checkpoint_ids: list[str] = []

    @staticmethod
    def _key(identity: GymExecutionIdentity) -> tuple[str, int]:
        return identity.rollout_id, identity.attempt_index

    def register(self, identity: GymExecutionIdentity) -> None:
        """Register one controller-assigned attempt unless checkpointing is active."""
        if self._frozen_checkpoint_id is not None:
            raise RuntimeError(
                "NeMo-Gym dispatch is frozen for checkpoint "
                f"{self._frozen_checkpoint_id!r}"
            )
        key = self._key(identity)
        if key in self._live:
            raise ValueError(f"Gym rollout execution {key!r} is already live")
        self._live[key] = GymActorExecution(
            identity=identity,
            state=GymActorExecutionState.RUNNING,
        )

    def mark_terminal(self, identity: GymExecutionIdentity) -> None:
        """Retain a completed invocation until it crosses the actor boundary."""
        key = self._key(identity)
        execution = self._live.get(key)
        if execution is None:
            raise KeyError(f"Gym rollout execution {key!r} is not live")
        self._live[key] = GymActorExecution(
            identity=identity,
            state=GymActorExecutionState.TERMINAL,
        )

    def release(self, identity: GymExecutionIdentity) -> None:
        """Release an invocation after its result crosses the actor boundary."""
        self._live.pop(self._key(identity), None)

    def freeze(self, checkpoint_id: str) -> tuple[GymActorExecution, ...]:
        """Close actor admission and snapshot the live source-cut membership."""
        if checkpoint_id in self._retired_checkpoint_ids:
            raise RuntimeError(f"checkpoint {checkpoint_id!r} is already retired")
        if self._frozen_checkpoint_id is not None:
            if self._frozen_checkpoint_id != checkpoint_id:
                raise RuntimeError(
                    f"checkpoint {self._frozen_checkpoint_id!r} already owns "
                    "the Gym actor dispatch fence"
                )
            return self._frozen_membership
        self._frozen_checkpoint_id = checkpoint_id
        self._frozen_membership = tuple(self._live[key] for key in sorted(self._live))
        return self._frozen_membership

    def unfreeze(self, checkpoint_id: str) -> None:
        """Reopen actor admission for the transaction that owns the fence."""
        if (
            self._frozen_checkpoint_id is None
            and checkpoint_id in self._retired_checkpoint_ids
        ):
            return
        if self._frozen_checkpoint_id != checkpoint_id:
            raise RuntimeError(
                f"checkpoint {checkpoint_id!r} does not own the Gym actor "
                f"dispatch fence (owner={self._frozen_checkpoint_id!r})"
            )
        self._frozen_checkpoint_id = None
        self._frozen_membership = ()
        self._retired_checkpoint_ids.append(checkpoint_id)
        del self._retired_checkpoint_ids[: -self._MAX_RETIRED_CHECKPOINTS]

    def status(self) -> dict[str, int | str | None]:
        """Return bounded diagnostics for tests and checkpoint failures."""
        running = sum(
            execution.state is GymActorExecutionState.RUNNING
            for execution in self._live.values()
        )
        return {
            "frozen_checkpoint_id": self._frozen_checkpoint_id,
            "live": len(self._live),
            "running": running,
            "terminal_unreleased": len(self._live) - running,
            "frozen_membership": len(self._frozen_membership),
        }


class GymCompletionReceipt(GymExecutionIdentity):
    """Exact Gym-issued proof naming one retained terminal result."""

    execution_generation: PositiveInt
    result_identity: str = Field(min_length=1, max_length=512)
    result_digest: Sha256Digest


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

    def restart_only_resources(self) -> list[str]:
        """Return resources whose unfinished rollouts cannot resume in place."""
        return sorted(
            contract.participant.server_name
            for contract in self.participants
            if contract.participant.component == "resources_servers"
            and contract.checkpoint_mode == "restart_only"
        )

    def validate_checkpoint_participants(
        self,
        checkpoint: "GymCheckpointCommitResult",
    ) -> None:
        """Require stateful discovered participants to match the saved export."""
        expected = {
            (
                contract.participant.server_name,
                contract.participant.component,
                contract.participant.participant_name,
            )
            for contract in self.participants
            if contract.checkpoint_mode == "export_restore"
        }
        actual = {
            (
                result.participant.server_name,
                result.participant.component,
                result.participant.participant_name,
            )
            for result in checkpoint.participants
        }
        if actual != expected:
            raise ValueError(
                "Gym checkpoint participants do not match the discovered stateful "
                f"topology: missing={sorted(expected - actual)!r}, "
                f"unexpected={sorted(actual - expected)!r}"
            )

    def validate_turn_recovery_capabilities(
        self,
        *,
        generation_prefix_cuts_enabled: bool = False,
    ) -> None:
        """Require the Gym features used by coordinated turn recovery."""
        missing_acknowledgement: list[str] = []
        missing_continuation_index: list[str] = []
        missing_fresh_restart: list[str] = []
        missing_resource_dependencies: list[str] = []
        missing_storage_reference_index: list[str] = []
        missing_generation_cut_lineage: list[str] = []
        requires_fresh_restart = bool(self.restart_only_resources())
        for contract in self.participants:
            if contract.checkpoint_mode != "export_restore":
                continue
            if contract.participant.component == "responses_api_agents":
                if "completed_result_acknowledgement" not in contract.features:
                    missing_acknowledgement.append(
                        contract.participant.participant_name
                    )
                if GYM_AGENT_CONTINUATION_INDEX_FEATURE not in contract.features:
                    missing_continuation_index.append(
                        contract.participant.participant_name
                    )
                if (
                    requires_fresh_restart
                    and GYM_AGENT_DISCARD_RESTORED_CONTINUATION_FEATURE
                    not in contract.features
                ):
                    missing_fresh_restart.append(contract.participant.participant_name)
                if (
                    requires_fresh_restart
                    and GYM_AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE
                    not in contract.features
                ):
                    missing_resource_dependencies.append(
                        contract.participant.participant_name
                    )
            elif (
                contract.participant.component == "responses_api_models"
                and contract.instance_role == "policy"
            ):
                if (
                    GYM_EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE
                    not in contract.features
                ):
                    missing_storage_reference_index.append(
                        contract.participant.participant_name
                    )
                if (
                    generation_prefix_cuts_enabled
                    and GYM_GENERATION_CUT_LINEAGE_FEATURE not in contract.features
                ):
                    missing_generation_cut_lineage.append(
                        contract.participant.participant_name
                    )

        if missing_acknowledgement:
            raise RuntimeError(
                "Gym participant checkpointing requires completed-result "
                "acknowledgement support from every stateful agent participant; "
                f"missing={missing_acknowledgement!r}"
            )
        if missing_continuation_index:
            raise RuntimeError(
                "Gym participant checkpointing requires continuation-index "
                "support from every stateful agent participant; "
                f"missing={missing_continuation_index!r}"
            )
        if missing_fresh_restart:
            raise RuntimeError(
                "Gym restart-only resources require restored-continuation "
                "discard support from every stateful agent participant; "
                f"missing={missing_fresh_restart!r}"
            )
        if missing_resource_dependencies:
            raise RuntimeError(
                "Gym restart-only resources require per-continuation resource "
                "dependency indexes from every stateful agent participant; "
                f"missing={missing_resource_dependencies!r}"
            )
        if missing_storage_reference_index:
            raise RuntimeError(
                "Gym participant checkpointing requires external-storage "
                "reference indexes from every stateful policy model; "
                f"missing={missing_storage_reference_index!r}"
            )
        if missing_generation_cut_lineage:
            raise RuntimeError(
                "Gym generation-prefix recovery requires durable lineage cuts "
                "from every stateful policy model; "
                f"missing={missing_generation_cut_lineage!r}"
            )


class GymCheckpointControlRequest(_StrictWireModel):
    """Fields shared by all Gym checkpoint control requests."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTITY_PATTERN,
    )
    deadline_ts: FiniteFloat


class GymCompletedExecution(_StrictWireModel):
    """A completed Gym execution plus the agent participant that owns it."""

    receipt: GymCompletionReceipt
    agent_name: str = Field(min_length=1)


class GymCompletedExecutionAcknowledgementRequest(_StrictWireModel):
    """Idempotent batch release of terminal results owned durably by RL."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    executions: list[GymCompletionReceipt] = Field(min_length=1)


class GymCompletedExecutionAcknowledgementResponse(_StrictWireModel):
    """Every requested identity the agent now considers acknowledged."""

    acknowledged: list[GymCompletionReceipt]

    @model_validator(mode="after")
    def validate_unique_identities(
        self,
    ) -> "GymCompletedExecutionAcknowledgementResponse":
        keys = [
            (identity.rollout_id, identity.attempt_index)
            for identity in self.acknowledged
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("acknowledged Gym execution identities must be unique")
        return self


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


class GymAgentDiscardRestoredContinuationRequest(
    GymCheckpointControlRequest,
    GymExecutionIdentity,
):
    """Discard one restored continuation before replacement admission opens."""


class GymAgentDiscardRestoredContinuationResponse(_StrictWireModel):
    """Result of removing saved turn state for one replacement attempt."""

    discarded: bool


class GymModelCheckpointCommitRequest(GymCheckpointDirectoryRequest):
    """Model commit request scoped to agent-owned continuation roots."""

    continuation_indexes: list[GymCheckpointArtifactReference]


class GymModelCheckpointRestoreRequest(GymCheckpointDirectoryRequest):
    """Model restore request returning its external-storage index."""

    generation_cut_receipts: list[dict[str, object]] = Field(default_factory=list)
    generation_cut_exclusions: list[GymExecutionIdentity] = Field(default_factory=list)


class GymWorkerAcknowledgements(_StrictWireModel):
    acknowledged: NonNegativeInt
    expected: NonNegativeInt


class GymModelPrepareResponse(_StrictWireModel):
    state: Literal["accepting", "draining", "paused"]
    workers: GymWorkerAcknowledgements
    inflight_total: NonNegativeInt
    response_inflight_total: NonNegativeInt | None = None
    generation_pending_total: NonNegativeInt | None = None
    # Gym owns the nested generation-cut proof schema. RL persists the opaque
    # validated payload and extracts only its durable TQ staging keys.
    generation_cut_proof: dict[str, object] | None = None
    waiters_total: NonNegativeInt


class GymModelInflightRequest(_LiveResponseWireModel):
    rollout_id: str | None = Field(default=None, pattern=_IDENTITY_PATTERN)
    attempt_index: NonNegativeInt | None = None
    plane: str | None
    age_seconds: NonNegativeFloat


class GymSingleWorkerModelStatus(_LiveResponseWireModel):
    state: Literal["accepting", "draining", "paused"]
    inflight: NonNegativeInt


class GymSingleWorkerModelStatusResponse(_LiveResponseWireModel):
    checkpoint_id: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    state: Literal["accepting", "draining", "paused"]
    per_worker: dict[str, GymSingleWorkerModelStatus]
    inflight_total: NonNegativeInt
    response_inflight_total: NonNegativeInt | None = None
    generation_pending_total: NonNegativeInt | None = None
    generation_cut_proof: dict[str, object] | None = None
    waiters_total: NonNegativeInt
    inflight: list[GymModelInflightRequest]
    tombstones: list[GymExecutionIdentity]


class GymCoordinatorWorkers(_LiveResponseWireModel):
    acknowledged: NonNegativeInt
    expected: PositiveInt
    live: NonNegativeInt


class GymCoordinatorWorkerStatus(_LiveResponseWireModel):
    acked_seq: NonNegativeInt
    inflight: NonNegativeInt
    generation_pending: NonNegativeInt | None = None
    # RL only observes proof presence here; Gym owns the nested proof schema.
    generation_cut_proof: dict[str, object] | None = None
    proof_error: str | None = None
    connected: bool


class GymCoordinatorModelStatusResponse(_LiveResponseWireModel):
    state: Literal["accepting", "draining", "paused"]
    workers: GymCoordinatorWorkers
    missing_workers: NonNegativeInt
    inflight_total: NonNegativeInt
    response_inflight_total: NonNegativeInt | None = None
    generation_pending_total: NonNegativeInt | None = None
    generation_cut_proof: dict[str, object] | None = None
    waiters_total: NonNegativeInt
    per_worker: dict[str, GymCoordinatorWorkerStatus]


class GymAgentExecutionStatus(GymExecutionIdentity):
    generation: PositiveInt
    state: Literal[
        "running",
        "park_requested",
        "parked",
        "model_wait_frozen",
        "completed",
        "retired",
    ]
    parked_boundary_state: (
        Literal[
            "parked_with_boundary",
            "parked_without_boundary",
            "model_wait_frozen",
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
    participant: GymParticipantIdentity
    ready: bool
    payload: GymPreparePayload


class GymCheckpointPrepareResult(_StrictWireModel):
    """One complete, safe checkpoint boundary across discovered participants."""

    checkpoint_id: str
    ready: bool
    participants: list[GymParticipantPrepareResult]


def gym_generation_cut_proofs(
    prepare: GymCheckpointPrepareResult,
) -> tuple[dict[str, object], ...]:
    """Return opaque policy-model cut proofs in deterministic participant order."""
    proofs = [
        dict(result.payload.generation_cut_proof)
        for result in prepare.participants
        if isinstance(result.payload, GymModelPrepareResponse)
        and result.payload.generation_cut_proof is not None
    ]
    return tuple(proofs)


def gym_generation_cut_receipts(
    proofs: tuple[dict[str, object], ...],
    *,
    server_name: str,
) -> list[dict[str, object]]:
    """Extract opaque cut receipts owned by one policy model server."""
    found: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for proof in proofs:
        candidates: list[object] = [proof.get("generation_cut_receipt")]
        workers = proof.get("workers")
        if workers is not None:
            if not isinstance(workers, list):
                raise ValueError("Gym generation-cut proof workers must be a list")
            for worker in workers:
                if not isinstance(worker, Mapping):
                    raise ValueError(
                        "Gym generation-cut worker proof must be an object"
                    )
                candidates.append(worker.get("generation_cut_receipt"))
        for candidate in candidates:
            if candidate is None:
                continue
            if not isinstance(candidate, Mapping):
                raise ValueError("Gym generation-cut receipt must be an object")
            inventory = candidate.get("inventory")
            if not isinstance(inventory, Mapping):
                raise ValueError(
                    "Gym generation-cut receipt inventory must be an object"
                )
            if inventory.get("server_name") != server_name:
                continue
            checkpoint_id = candidate.get("checkpoint_id")
            cut_id = candidate.get("cut_id")
            if not isinstance(checkpoint_id, str) or not isinstance(cut_id, str):
                raise ValueError(
                    "Gym generation-cut receipt requires checkpoint_id and cut_id"
                )
            identity = (checkpoint_id, cut_id)
            if identity in seen:
                continue
            seen.add(identity)
            found.append(dict(candidate))
    return found


def gym_generation_cut_staging_keys(
    prepare: GymCheckpointPrepareResult | tuple[dict[str, object], ...],
    *,
    excluded_replacements: set[tuple[str, int]] | None = None,
) -> set[str]:
    """Extract every durable-prefix TQ key named by Gym's cut proofs."""

    def receipts(proof: Mapping[str, object]) -> list[Mapping[str, object]]:
        direct = proof.get("generation_cut_receipt")
        if isinstance(direct, Mapping):
            return [direct]
        workers = proof.get("workers")
        if workers is None:
            return []
        if not isinstance(workers, list):
            raise ValueError("Gym generation-cut proof workers must be a list")
        found: list[Mapping[str, object]] = []
        for worker in workers:
            if not isinstance(worker, Mapping):
                raise ValueError("Gym generation-cut worker proof must be an object")
            receipt = worker.get("generation_cut_receipt")
            if receipt is not None:
                if not isinstance(receipt, Mapping):
                    raise ValueError("Gym generation-cut receipt must be an object")
                found.append(receipt)
        return found

    keys: set[str] = set()
    excluded = excluded_replacements or set()
    proofs = (
        gym_generation_cut_proofs(prepare)
        if isinstance(prepare, GymCheckpointPrepareResult)
        else prepare
    )
    for proof in proofs:
        for receipt in receipts(proof):
            prefixes = receipt.get("prefixes")
            if not isinstance(prefixes, list):
                raise ValueError("Gym generation-cut receipt prefixes must be a list")
            for prefix in prefixes:
                if not isinstance(prefix, Mapping):
                    raise ValueError(
                        "Gym generation-cut prefix acknowledgement must be an object"
                    )
                disposition = prefix.get("disposition")
                if disposition == "durable_failure":
                    continue
                if disposition != "durable_prefix":
                    raise ValueError(
                        f"unknown Gym generation-cut disposition {disposition!r}"
                    )
                rollout_id = prefix.get("rollout_id")
                attempt_index = prefix.get("attempt_index")
                if (
                    isinstance(rollout_id, str)
                    and isinstance(attempt_index, int)
                    and (rollout_id, attempt_index + 1) in excluded
                ):
                    continue
                staging_keys = prefix.get("staging_keys")
                if not isinstance(staging_keys, list) or not staging_keys:
                    raise ValueError(
                        "durable Gym generation-cut prefix requires staging keys"
                    )
                if any(not isinstance(key, str) or not key for key in staging_keys):
                    raise ValueError(
                        "durable Gym generation-cut prefix contains an invalid staging key"
                    )
                if len(staging_keys) != len(set(staging_keys)):
                    raise ValueError(
                        "durable Gym generation-cut prefix contains duplicate staging keys"
                    )
                keys.update(staging_keys)
    return keys


class GymModelCommitResponse(_StrictWireModel):
    rollouts: NonNegativeInt
    rows: NonNegativeInt
    excluded_tombstoned: NonNegativeInt
    excluded_inactive: NonNegativeInt = 0
    generation_cut_records: NonNegativeInt = 0
    manifest_digest: Sha256Digest
    storage_reference_index: GymCheckpointArtifactReference


class GymAgentCommitResponse(_StrictWireModel):
    records: NonNegativeInt
    manifest_digest: Sha256Digest
    continuation_index: GymCheckpointArtifactReference


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


def validate_gym_checkpoint_manifests(
    checkpoint_dir: Path,
    checkpoint: GymCheckpointCommitResult,
) -> None:
    """Verify every participant manifest before the outer snapshot publishes."""
    root = checkpoint_dir.resolve()
    seen_paths: set[Path] = set()
    for result in checkpoint.participants:
        relative_path = Path(result.manifest.relative_path)
        if relative_path.is_absolute():
            raise ValueError(
                f"Gym participant manifest path must be relative: {relative_path}"
            )
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Gym participant manifest escapes the checkpoint directory: "
                f"{relative_path}"
            ) from error
        if path in seen_paths:
            raise ValueError(
                f"duplicate Gym participant manifest path: {relative_path}"
            )
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Gym participant manifest is missing: {relative_path}"
            )
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != result.manifest.manifest_digest:
            raise ValueError(
                "Gym participant manifest digest mismatch: "
                f"path={relative_path}, "
                f"expected={result.manifest.manifest_digest}, "
                f"actual={actual_digest}"
            )


class GymExternalStorageReference(_StrictWireModel):
    """One TQ staging row required by a Gym recovery point."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    capture_key: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    boundary_model_call_id: str = Field(min_length=1)
    kind: Literal["token_capture_staging", "generation_prefix_cut"] = (
        "token_capture_staging"
    )
    key: str = Field(min_length=1)


class GymAgentContinuationRoot(_StrictWireModel):
    """Public agent boundary coordinate used for selective recovery."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    rollout_id: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    attempt_index: NonNegativeInt
    capture_key: str = Field(min_length=1, pattern=_IDENTITY_PATTERN)
    last_committed_model_call_id: str = Field(min_length=1)
    # None identifies a legacy index that predates per-execution dependency
    # metadata.  An empty mapping explicitly means no resources were used.
    resource_state_revisions: dict[str, NonNegativeInt] | None = None

    @model_validator(mode="after")
    def validate_capture_key(self) -> "GymAgentContinuationRoot":
        expected = gym_capture_key(self.rollout_id, self.attempt_index)
        if self.capture_key != expected:
            raise ValueError(
                "Gym continuation capture key does not match its execution: "
                f"expected={expected!r}, actual={self.capture_key!r}"
            )
        return self


@dataclass(frozen=True)
class GymCheckpointContinuation:
    """One restored continuation and the external rows that keep it alive."""

    rollout_id: str
    source_attempt_index: int
    capture_key: str
    resource_state_revisions: tuple[tuple[str, int], ...] | None
    staging_keys: tuple[str, ...]

    @property
    def replacement_attempt_index(self) -> int:
        """Return the physical attempt into which Gym installs the boundary."""
        return self.source_attempt_index + 1


_ArtifactRecord = TypeVar("_ArtifactRecord", bound=_StrictWireModel)


def _read_jsonl_artifact(
    checkpoint_dir: Path,
    reference: GymCheckpointArtifactReference,
    record_type: type[_ArtifactRecord],
) -> list[_ArtifactRecord]:
    """Validate and parse a Gym-owned JSONL sidecar in one pass."""
    root = checkpoint_dir.resolve()
    path = (root / reference.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Gym checkpoint artifact escapes the checkpoint directory: "
            f"{reference.relative_path!r}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(
            f"Gym checkpoint artifact is missing: {reference.relative_path!r}"
        )

    digest = hashlib.sha256()
    byte_count = 0
    records: list[_ArtifactRecord] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                continue
            try:
                records.append(record_type.model_validate_json(line))
            except Exception as error:
                raise ValueError(
                    "invalid Gym checkpoint artifact row: "
                    f"path={reference.relative_path!r}, line={line_number}"
                ) from error

    actual_digest = digest.hexdigest()
    if actual_digest != reference.sha256:
        raise ValueError(
            "Gym checkpoint artifact digest mismatch: "
            f"path={reference.relative_path!r}, expected={reference.sha256}, "
            f"actual={actual_digest}"
        )
    if byte_count != reference.bytes:
        raise ValueError(
            "Gym checkpoint artifact byte count mismatch: "
            f"path={reference.relative_path!r}, expected={reference.bytes}, "
            f"actual={byte_count}"
        )
    if len(records) != reference.records:
        raise ValueError(
            "Gym checkpoint artifact record count mismatch: "
            f"path={reference.relative_path!r}, expected={reference.records}, "
            f"actual={len(records)}"
        )
    return records


def _gym_checkpoint_external_storage_references(
    checkpoint_dir: Path,
    checkpoint: GymCheckpointCommitResult,
) -> dict[str, GymExternalStorageReference]:
    """Load each unique external TQ reference from public model indexes."""
    model_results = [
        result
        for result in checkpoint.participants
        if result.participant.component == "responses_api_models"
    ]
    references_by_key: dict[str, GymExternalStorageReference] = {}
    for result in model_results:
        payload = result.payload
        if not isinstance(payload, GymModelCommitResponse):
            raise TypeError(
                "Gym model participant returned a non-model checkpoint payload"
            )
        reference = payload.storage_reference_index
        records = _read_jsonl_artifact(
            checkpoint_dir,
            reference,
            GymExternalStorageReference,
        )
        for raw_record in records:
            if raw_record.key in references_by_key:
                raise ValueError(
                    "Gym checkpoint repeats an external storage key: "
                    f"key={raw_record.key!r}"
                )
            references_by_key[raw_record.key] = raw_record
    return references_by_key


def gym_checkpoint_staging_keys(
    checkpoint_dir: Path,
    checkpoint: GymCheckpointCommitResult,
) -> set[str]:
    """Return the TQ staging rows required by committed Gym continuations.

    Gym owns its private agent and model-lineage formats. RL consumes only the
    required, digest-bound external-storage reference index.
    """
    validate_gym_checkpoint_manifests(checkpoint_dir, checkpoint)
    return set(_gym_checkpoint_external_storage_references(checkpoint_dir, checkpoint))


def gym_checkpoint_generation_cut_records(
    checkpoint: GymCheckpointCommitResult,
) -> int:
    """Return the number of prefix cuts embedded in Gym model lineage."""
    return sum(
        result.payload.generation_cut_records
        for result in checkpoint.participants
        if isinstance(result.payload, GymModelCommitResponse)
    )


def gym_checkpoint_continuations(
    checkpoint_dir: Path,
    checkpoint: GymCheckpointCommitResult,
) -> tuple[GymCheckpointContinuation, ...]:
    """Join public agent dependencies with model-owned TQ references."""
    validate_gym_checkpoint_manifests(checkpoint_dir, checkpoint)
    roots_by_capture_key: dict[str, GymAgentContinuationRoot] = {}
    for result in checkpoint.participants:
        if result.participant.component != "responses_api_agents":
            continue
        payload = result.payload
        if not isinstance(payload, GymAgentCommitResponse):
            raise TypeError(
                "Gym agent participant returned a non-agent checkpoint payload"
            )
        roots = _read_jsonl_artifact(
            checkpoint_dir,
            payload.continuation_index,
            GymAgentContinuationRoot,
        )
        for root in roots:
            if root.capture_key in roots_by_capture_key:
                raise ValueError(
                    "Gym checkpoint repeats an agent continuation capture key: "
                    f"capture_key={root.capture_key!r}"
                )
            roots_by_capture_key[root.capture_key] = root

    references = {
        key: reference
        for key, reference in _gym_checkpoint_external_storage_references(
            checkpoint_dir,
            checkpoint,
        ).items()
        if reference.kind == "token_capture_staging"
    }
    keys_by_capture_key: dict[str, list[str]] = {}
    for key, reference in references.items():
        root = roots_by_capture_key.get(reference.capture_key)
        if (
            root is not None
            and reference.boundary_model_call_id != root.last_committed_model_call_id
        ):
            raise ValueError(
                "Gym external storage reference does not match its agent "
                "continuation boundary: "
                f"capture_key={reference.capture_key!r}"
            )
        keys_by_capture_key.setdefault(reference.capture_key, []).append(key)
    unknown_capture_keys = set(keys_by_capture_key) - set(roots_by_capture_key)
    if unknown_capture_keys:
        raise ValueError(
            "Gym external storage references have no agent continuation: "
            f"capture_keys={sorted(unknown_capture_keys)!r}"
        )

    return tuple(
        GymCheckpointContinuation(
            rollout_id=root.rollout_id,
            source_attempt_index=root.attempt_index,
            capture_key=root.capture_key,
            resource_state_revisions=(
                tuple(sorted(root.resource_state_revisions.items()))
                if root.resource_state_revisions is not None
                else None
            ),
            staging_keys=tuple(sorted(keys_by_capture_key.get(root.capture_key, []))),
        )
        for root in sorted(
            roots_by_capture_key.values(),
            key=lambda item: (item.rollout_id, item.attempt_index),
        )
    )


class GymModelRestoreResponse(_StrictWireModel):
    rollouts: NonNegativeInt
    rows: NonNegativeInt
    checkpoint_id: str | None = None
    tombstones: list[GymExecutionIdentity]
    source_attempts: list[GymExecutionIdentity]
    storage_reference_index: GymCheckpointArtifactReference
    generation_cuts_restored: NonNegativeInt = 0


class GymAgentRestoreResponse(_StrictWireModel):
    records: NonNegativeInt
    source_checkpoint_id: str = Field(min_length=1)
    continuation_index: GymCheckpointArtifactReference


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


def validate_gym_checkpoint_restore_artifacts(
    committed: GymCheckpointCommitResult,
    restored: GymCheckpointRestoreResult,
) -> None:
    """Require Gym restore to report the exact sidecars saved in the snapshot."""

    def identity_key(
        participant: GymParticipantIdentity,
    ) -> tuple[str, str, str]:
        return (
            participant.server_name,
            participant.component,
            participant.participant_name,
        )

    restored_by_participant = {
        identity_key(result.participant): result for result in restored.participants
    }
    for committed_result in committed.participants:
        expected: GymCheckpointArtifactReference | None = None
        if isinstance(committed_result.payload, GymAgentCommitResponse):
            expected = committed_result.payload.continuation_index
        elif isinstance(committed_result.payload, GymModelCommitResponse):
            expected = committed_result.payload.storage_reference_index
        else:
            continue

        restored_result = restored_by_participant.get(
            identity_key(committed_result.participant)
        )
        if restored_result is None:
            raise ValueError(
                "Gym restore omitted a participant with checkpoint artifacts: "
                f"participant={committed_result.participant!r}"
            )
        actual: GymCheckpointArtifactReference | None = None
        if isinstance(restored_result.payload, GymAgentRestoreResponse):
            actual = restored_result.payload.continuation_index
        elif isinstance(restored_result.payload, GymModelRestoreResponse):
            actual = restored_result.payload.storage_reference_index
        if actual != expected:
            raise ValueError(
                "Gym restore reported a different checkpoint artifact: "
                f"participant={committed_result.participant!r}, "
                f"expected={expected!r}, actual={actual!r}"
            )


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
