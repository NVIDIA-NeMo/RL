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
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

GYM_CHECKPOINT_SCHEMA_VERSION = 1
GYM_CHECKPOINT_CONTROL_PREFIX = "/ng-control/v1"
GYM_CHECKPOINT_CAPABILITIES_PATH = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/capabilities"
GYM_MODEL_ADMISSION_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-admission"
GYM_MODEL_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/model-checkpoint"
GYM_AGENT_CHECKPOINT_PREFIX = f"{GYM_CHECKPOINT_CONTROL_PREFIX}/agent-checkpoint"
GYM_AGENT_COMPLETION_ACK_PATH = f"{GYM_AGENT_CHECKPOINT_PREFIX}/acknowledge-completed"
GYM_RESOURCES_CHECKPOINT_PREFIX = (
    f"{GYM_CHECKPOINT_CONTROL_PREFIX}/resources-checkpoint"
)

_IDENTITY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

GymComponent: TypeAlias = Literal[
    "responses_api_models",
    "responses_api_agents",
    "resources_servers",
]
GymCheckpointFeature: TypeAlias = Literal["completed_result_acknowledgement"]
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
    features: list[GymCheckpointFeature] = Field(default_factory=list)

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
    features: list[GymCheckpointFeature] = Field(default_factory=list)

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
        """Return a canonical digest without runtime routing or credentials."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

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

    execution: GymExecutionIdentity
    agent_name: str = Field(min_length=1)


class GymCompletedExecutionAcknowledgementRequest(_StrictWireModel):
    """Idempotent batch release of terminal results owned durably by RL."""

    schema_version: Literal[1] = GYM_CHECKPOINT_SCHEMA_VERSION
    executions: list[GymExecutionIdentity] = Field(min_length=1)


class GymCompletedExecutionAcknowledgementResponse(_StrictWireModel):
    """Every requested identity the agent now considers acknowledged."""

    acknowledged: list[GymExecutionIdentity]

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


def gym_checkpoint_staging_keys(
    checkpoint_dir: Path,
    checkpoint: GymCheckpointCommitResult,
) -> set[str]:
    """Return TQ staging keys required by committed Gym continuations.

    Gym owns token-free call lineage while TQ owns the referenced tensors. RL
    must treat both as one checkpoint ownership graph; otherwise restore-time
    orphan cleanup can delete an unfinished turn's staged model calls before
    the restored agent continues from its boundary. Completed, acknowledged
    executions are deliberately excluded: their canonical training rows own the
    durable result and their old model lineage no longer owns staging tensors.
    """
    validate_gym_checkpoint_manifests(checkpoint_dir, checkpoint)
    root = checkpoint_dir.resolve()
    continuation_model_calls: dict[str, str] = {}
    for result in checkpoint.participants:
        if result.participant.component != "responses_api_agents":
            continue
        if not isinstance(result.payload, GymAgentCommitResponse):
            raise TypeError(
                "Gym agent participant returned a non-agent checkpoint payload"
            )
        manifest_path = (root / result.manifest.relative_path).resolve()
        raw_manifest = json.loads(manifest_path.read_text())
        if not isinstance(raw_manifest, Mapping):
            raise TypeError(f"Gym agent manifest must be an object: {manifest_path}")
        raw_files = raw_manifest.get("files")
        if not isinstance(raw_files, Mapping):
            raise TypeError(
                f"Gym agent manifest files must be an object: {manifest_path}"
            )
        if len(raw_files) != result.payload.records:
            raise ValueError(
                "Gym agent checkpoint record count does not match its manifest: "
                f"payload={result.payload.records}, actual={len(raw_files)}"
            )
        for name, expected_digest in raw_files.items():
            if not isinstance(name, str) or not isinstance(expected_digest, str):
                raise TypeError("Gym agent manifest file entries are malformed")
            record_path = (manifest_path.parent / name).resolve()
            try:
                record_path.relative_to(manifest_path.parent)
            except ValueError as error:
                raise ValueError(
                    f"Gym agent record escapes its manifest directory: {name}"
                ) from error
            if not record_path.is_file():
                raise FileNotFoundError(f"Gym agent record is missing: {record_path}")
            record_payload = record_path.read_bytes()
            actual_digest = hashlib.sha256(record_payload).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(
                    "Gym agent record digest mismatch: "
                    f"path={record_path}, expected={expected_digest}, "
                    f"actual={actual_digest}"
                )
            record: Any = json.loads(record_payload)
            if not isinstance(record, Mapping):
                raise TypeError(f"Gym agent record must be an object: {record_path}")
            rollout_id = record.get("rollout_id")
            attempt_index = record.get("attempt_index")
            last_committed_model_call_id = record.get("last_committed_model_call_id")
            if (
                not isinstance(rollout_id, str)
                or not rollout_id
                or isinstance(attempt_index, bool)
                or not isinstance(attempt_index, int)
                or attempt_index < 0
                or not isinstance(last_committed_model_call_id, str)
                or not last_committed_model_call_id
            ):
                raise TypeError(
                    "Gym agent record has invalid execution identity or model-call "
                    f"boundary: {record_path}"
                )
            capture_key = gym_capture_key(rollout_id, attempt_index)
            if capture_key in continuation_model_calls:
                raise ValueError(
                    f"duplicate Gym continuation capture key {capture_key!r}"
                )
            continuation_model_calls[capture_key] = last_committed_model_call_id

    staging_keys: set[str] = set()
    found_capture_keys: set[str] = set()
    for result in checkpoint.participants:
        if result.participant.component != "responses_api_models":
            continue
        if not isinstance(result.payload, GymModelCommitResponse):
            raise TypeError(
                "Gym model participant returned a non-model checkpoint payload"
            )
        manifest_path = (root / result.manifest.relative_path).resolve()
        ledger_root = manifest_path.parent
        raw_manifest = json.loads(manifest_path.read_text())
        if not isinstance(raw_manifest, Mapping):
            raise TypeError(f"Gym model manifest must be an object: {manifest_path}")
        raw_rollouts = raw_manifest.get("rollouts")
        if not isinstance(raw_rollouts, Mapping):
            raise TypeError(
                f"Gym model manifest rollouts must be an object: {manifest_path}"
            )

        observed_rows = 0
        for rollout_id, raw_rollout in raw_rollouts.items():
            if not isinstance(rollout_id, str) or not isinstance(raw_rollout, Mapping):
                raise TypeError("Gym model manifest rollout entries are malformed")
            raw_files = raw_rollout.get("files")
            if not isinstance(raw_files, Mapping):
                raise TypeError(
                    f"Gym model manifest files for {rollout_id!r} must be an object"
                )
            rollout_rows = 0
            for name, expected_digest in raw_files.items():
                if not isinstance(name, str) or not isinstance(expected_digest, str):
                    raise TypeError("Gym model manifest file entries are malformed")
                lineage_path = (ledger_root / name).resolve()
                try:
                    lineage_path.relative_to(ledger_root)
                except ValueError as error:
                    raise ValueError(
                        f"Gym lineage path escapes its manifest directory: {name}"
                    ) from error
                if not lineage_path.is_file():
                    raise FileNotFoundError(
                        f"Gym lineage file is missing: {lineage_path}"
                    )
                lineage_payload = lineage_path.read_bytes()
                actual_digest = hashlib.sha256(lineage_payload).hexdigest()
                if actual_digest != expected_digest:
                    raise ValueError(
                        "Gym lineage digest mismatch: "
                        f"path={lineage_path}, expected={expected_digest}, "
                        f"actual={actual_digest}"
                    )
                for line_number, line in enumerate(
                    lineage_payload.splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    rollout_rows += 1
                    row: Any = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise TypeError(
                            "Gym lineage row must be an object: "
                            f"path={lineage_path}, line={line_number}"
                        )
                    if rollout_id not in continuation_model_calls:
                        continue
                    raw_staging_keys: list[Any] = []
                    staging_key = row.get("staging_key")
                    if staging_key is not None:
                        raw_staging_keys.append(staging_key)
                    staging_chain = row.get("staging_chain")
                    if staging_chain is not None:
                        if not isinstance(staging_chain, list):
                            raise TypeError(
                                "Gym lineage staging_chain must be a list: "
                                f"path={lineage_path}, line={line_number}"
                            )
                        raw_staging_keys.extend(staging_chain)
                    for owned_staging_key in raw_staging_keys:
                        if (
                            not isinstance(owned_staging_key, str)
                            or not owned_staging_key
                        ):
                            raise TypeError(
                                "Gym lineage staging keys must be non-empty strings: "
                                f"path={lineage_path}, line={line_number}"
                            )
                        # Cumulative staging chains deliberately repeat parent keys
                        # across later model calls. The ownership inventory is a set.
                        staging_keys.add(owned_staging_key)
                    if (
                        row.get("model_call_id") == continuation_model_calls[rollout_id]
                        and staging_key is not None
                    ):
                        found_capture_keys.add(rollout_id)
            expected_rows = raw_rollout.get("rows")
            if expected_rows != rollout_rows:
                raise ValueError(
                    f"Gym lineage row count mismatch for {rollout_id!r}: "
                    f"manifest={expected_rows!r}, actual={rollout_rows}"
                )
            observed_rows += rollout_rows
        if observed_rows != result.payload.rows:
            raise ValueError(
                "Gym model checkpoint row count does not match its manifest: "
                f"payload={result.payload.rows}, actual={observed_rows}"
            )
    missing_capture_keys = set(continuation_model_calls) - found_capture_keys
    if missing_capture_keys:
        raise ValueError(
            "Gym agent continuations are missing model lineage: "
            f"capture_keys={sorted(missing_capture_keys)!r}"
        )
    return staging_keys


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
