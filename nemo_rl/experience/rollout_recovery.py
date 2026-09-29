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

"""Controller-owned lineage for recoverable token-capture prompt groups.

The ledger deliberately contains control-plane metadata only. Token tensors and
router-replay payloads remain in TQ. The versioned ``state_dict`` boundary here is
what the controller writes into ``rollout_recovery.pt`` at each checkpoint and
reads back on restore. It also retains completed-result acknowledgement
obligations until Gym confirms them. Those obligations are retried while the
owning Gym deployment remains live; an exact Gym-aware snapshot is published
only after its acknowledgement outbox is empty.
"""

from __future__ import annotations

import copy
import dataclasses
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, TYPE_CHECKING, Any, Literal, Optional, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from nemo_rl.environments.gym_checkpoint import (
    GymCompletedExecution,
    GymCompletionReceipt,
    gym_capture_key,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneMutationCut
    from nemo_rl.data.interfaces import DatumSpec

# Version of the rollout recovery ledger (rollout_recovery.pt). Readers accept
# only this version, so bump it whenever a saved field is added, removed, renamed,
# or changes meaning, then regenerate
# tests/unit/single_controller/checkpoint_schema_lock.json.
ROLLOUT_RECOVERY_SCHEMA_VERSION = 3
_SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS = {ROLLOUT_RECOVERY_SCHEMA_VERSION}
ROLLOUT_RECOVERY_STATE_FILENAME = "rollout_recovery.pt"
RolloutRecoveryState: TypeAlias = dict[str, Any]


def _unsupported_schema_message(schema_version: object) -> str:
    return (
        "Unsupported rollout-recovery schema version: "
        f"{schema_version!r}; this build supports only schema version "
        f"{ROLLOUT_RECOVERY_SCHEMA_VERSION}. Backward restore is intentionally "
        "unsupported; start from a fresh checkpoint or use a compatible build."
    )


class PromptGroupPhase(StrEnum):
    """Durable sampler-admission phase for an unfinished prompt group."""

    RESERVED = "reserved"
    ADMITTED = "admitted"


class RecoveryGranularity(StrEnum):
    """Unit of completed work reused after a live failure or process restart."""

    SIBLING = "sibling"
    PROMPT_GROUP = "prompt_group"


class RolloutAttemptStatus(StrEnum):
    """Lifecycle of one physical Gate execution attempt."""

    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    SEALED = "sealed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class PromptGroupStatus(StrEnum):
    """Ownership lifecycle of one logical prompt group."""

    GENERATING = "generating"
    READY_TO_FINALIZE = "ready_to_finalize"
    FINALIZING = "finalizing"
    FINALIZATION_UNKNOWN = "finalization_unknown"


NonNegativeInt: TypeAlias = Annotated[int, Field(strict=True, ge=0)]
PositiveInt: TypeAlias = Annotated[int, Field(strict=True, ge=1)]
StrictBool: TypeAlias = Annotated[bool, Field(strict=True)]


class _SavedState(BaseModel):
    """Immutable exact model for data written into rollout checkpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptRefState(_SavedState):
    sample_id: str = Field(min_length=1)
    # Required-but-nullable fields intentionally have no default. Omitting one
    # is a schema error rather than an implicit request to use ``None``.
    task_name: Optional[str]


class RolloutAttemptState(_SavedState):
    attempt_index: NonNegativeInt
    status: RolloutAttemptStatus
    receipt: Optional[dict[str, Any]]
    completion_receipt: Optional[GymCompletionReceipt]
    reward: Optional[FiniteFloat]
    mask_sample: Optional[StrictBool]
    staging_keys: list[str]


class RolloutSiblingState(_SavedState):
    generation_index: NonNegativeInt
    attempts: list[RolloutAttemptState]


class PromptGroupRecoveryState(_SavedState):
    group_id: str = Field(min_length=1)
    admission_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_ref: PromptRefState
    task_source: Optional[str]
    resolved_agent_name: Optional[str]
    recovery_granularity: RecoveryGranularity
    expected_generations: PositiveInt
    target_step: Optional[NonNegativeInt]
    start_weight_version: NonNegativeInt
    status: PromptGroupStatus
    phase: PromptGroupPhase
    siblings: list[RolloutSiblingState]


class RolloutRecoveryLedgerState(_SavedState):
    """Exact persisted schema owned by :class:`RolloutRecoveryLedger`."""

    schema_version: Literal[3]
    groups: list[PromptGroupRecoveryState]
    pending_completed_execution_acknowledgements: list[GymCompletedExecution]


class RolloutRecoverySidecarState(RolloutRecoveryLedgerState):
    """Complete controller sidecar stored in ``rollout_recovery.pt``."""

    batch_shortfall: dict[NonNegativeInt, NonNegativeInt]
    sampler_stamps_target_steps: StrictBool


@dataclass(frozen=True)
class PromptRef:
    """Small durable locator for a prompt owned by the input dataset.

    The persistence layer will resolve this reference and validate the dataset
    identity before redispatch. The full ``DatumSpec`` is runtime-only state and
    is deliberately excluded from the serialized ledger.
    """

    sample_id: str
    task_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("prompt sample_id must not be empty")


def _validate_prompt_identity(
    prompt_ref: PromptRef,
    prompt_payload: DatumSpec,
    *,
    group_id: str,
) -> None:
    """Require a runtime prompt to resolve the ledger's durable dataset key."""
    sample_id = prompt_payload.get("idx")
    if isinstance(sample_id, bool) or not isinstance(sample_id, int):
        raise ValueError(
            f"recovery group {group_id!r} prompt payload must contain an integer idx"
        )
    if str(sample_id) != prompt_ref.sample_id:
        raise ValueError(
            f"recovery group {group_id!r} resolved sample_id={sample_id!r}; "
            f"expected {prompt_ref.sample_id!r}"
        )
    task_name = prompt_payload.get("task_name")
    if task_name is not None and not isinstance(task_name, str):
        raise TypeError("prompt task_name must be a string or None")
    if task_name != prompt_ref.task_name:
        raise ValueError(
            f"recovery group {group_id!r} resolved task_name={task_name!r}; "
            f"expected {prompt_ref.task_name!r}"
        )


@dataclass
class RolloutAttemptRecord:
    """One physical attempt for a stable logical sibling."""

    attempt_index: int
    status: RolloutAttemptStatus
    receipt: Optional[dict[str, Any]] = None
    completion_receipt: Optional[dict[str, Any]] = None
    reward: Optional[float] = None
    mask_sample: Optional[bool] = None
    staging_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutSiblingRecord:
    """One stable GRPO generation slot and its physical attempts."""

    generation_index: int
    attempts: list[RolloutAttemptRecord]

    @property
    def current_attempt(self) -> RolloutAttemptRecord:
        if not self.attempts:
            raise RuntimeError(
                f"generation index {self.generation_index} has no attempts"
            )
        return self.attempts[-1]


@dataclass
class PromptGroupRecoveryRecord:
    """Lineage and ownership for one logical prompt group."""

    group_id: str
    admission_id: str
    prompt_id: str
    prompt_ref: PromptRef
    task_source: Optional[str]
    recovery_granularity: RecoveryGranularity
    runtime_prompt_payload: Optional[DatumSpec]
    expected_generations: int
    target_step: Optional[int]
    start_weight_version: int
    siblings: list[RolloutSiblingRecord]
    phase: PromptGroupPhase
    # Gym resolves task_source to the concrete agent at execution time. Retain
    # that identity so the controller can acknowledge the agent's cached
    # terminal result once the seal and ACK obligation are recorded together.
    resolved_agent_name: Optional[str] = None
    status: PromptGroupStatus = PromptGroupStatus.GENERATING

    @property
    def prompt_payload(self) -> DatumSpec:
        """Return the runtime prompt required to redispatch unfinished work."""
        if self.runtime_prompt_payload is None:
            raise RuntimeError(
                f"recovery group {self.group_id!r} has not rehydrated prompt "
                f"sample_id={self.prompt_ref.sample_id!r}"
            )
        return self.runtime_prompt_payload

    @property
    def logical_rollout_ids(self) -> list[str]:
        return [
            self.logical_rollout_id(sibling.generation_index)
            for sibling in self.siblings
        ]

    @property
    def gate_rollout_ids(self) -> list[str]:
        return [
            self.gate_rollout_id(sibling.generation_index) for sibling in self.siblings
        ]

    def logical_rollout_id(self, generation_index: int) -> str:
        """Derive the stable sibling ID instead of storing another UUID string."""
        return f"{self.group_id}_g{generation_index}"

    def gate_rollout_id(self, generation_index: int) -> str:
        """Derive Gym's attempt-qualified capture key for this sibling."""
        sibling = self.siblings[generation_index]
        logical_rollout_id = self.logical_rollout_id(generation_index)
        attempt_index = sibling.current_attempt.attempt_index
        return gym_capture_key(logical_rollout_id, attempt_index)

    @property
    def current_attempt_indices(self) -> list[int]:
        """Return the numeric Gym execution attempt for each logical sibling."""
        return [sibling.current_attempt.attempt_index for sibling in self.siblings]

    @property
    def sealed_generation_indices(self) -> list[int]:
        return [
            sibling.generation_index
            for sibling in self.siblings
            if sibling.current_attempt.status == RolloutAttemptStatus.SEALED
        ]


@dataclass(frozen=True)
class ParsedRolloutRecoveryState:
    """Validated controller and ledger state loaded from one checkpoint sidecar."""

    ledger_state: RolloutRecoveryState
    batch_shortfall: dict[int, int]
    sampler_stamps_target_steps: Optional[bool]


@dataclass(frozen=True)
class SiblingSealResult:
    """One terminal sibling result waiting for an atomic prompt-group seal."""

    gate_rollout_id: str
    # None is an explicit terminal capture failure. The finalizer turns it
    # into a masked placeholder, matching the base token-capture contract.
    receipt: Optional[dict[str, Any]]
    reward: float
    mask_sample: bool
    resolved_agent_name: str
    completion_receipt: Optional[GymCompletionReceipt] = None


def _new_attempt(attempt_index: int) -> RolloutAttemptRecord:
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    return RolloutAttemptRecord(
        attempt_index=attempt_index,
        status=RolloutAttemptStatus.RESERVED,
    )


def _receipt_staging_keys(receipt: Optional[dict[str, Any]]) -> list[str]:
    """Validate a terminal Gate receipt and return its ordered staging keys."""
    if receipt is None:
        return []
    manifest = receipt.get("manifest")
    if not isinstance(manifest, list):
        raise ValueError("sealed rollout receipt must contain a manifest list")
    staging_keys: list[str] = []
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("staging_key"), str):
            raise ValueError(
                "sealed rollout receipt manifest entries must contain string "
                "staging_key values"
            )
        staging_keys.append(entry["staging_key"])
    return staging_keys


def _completed_execution_differences(
    expected: GymCompletedExecution,
    actual: GymCompletedExecution,
) -> str:
    """Name every receipt field that differs for one ACK identity."""
    differences: list[str] = []
    if expected.agent_name != actual.agent_name:
        differences.append(f"agent_name={expected.agent_name!r}->{actual.agent_name!r}")
    for field_name in GymCompletionReceipt.model_fields:
        expected_value = getattr(expected.receipt, field_name)
        actual_value = getattr(actual.receipt, field_name)
        if expected_value != actual_value:
            differences.append(
                f"receipt.{field_name}={expected_value!r}->{actual_value!r}"
            )
    return ", ".join(differences) or "unknown difference"


class RolloutRecoveryLedger:
    """In-memory source of truth for token-capture rollout ownership.

    Controller-owned mutations require a live data-plane cut so lineage cannot
    change outside the checkpoint barrier's consistent snapshot boundary.
    """

    def __init__(self) -> None:
        self._groups: dict[str, PromptGroupRecoveryRecord] = {}
        self._pending_completed_execution_acknowledgements: dict[
            tuple[str, int], GymCompletedExecution
        ] = {}

    def groups(self) -> list[PromptGroupRecoveryRecord]:
        return [self._copy_group(group) for group in self._groups.values()]

    def reserve_group(
        self,
        cut: DataPlaneMutationCut,
        *,
        prompt_id: str,
        prompt_payload: DatumSpec,
        expected_generations: int,
        target_step: Optional[int],
        start_weight_version: int,
        task_source: Optional[str] = None,
        recovery_granularity: RecoveryGranularity = RecoveryGranularity.SIBLING,
        admitted: bool = True,
        group_id: Optional[str] = None,
        admission_id: Optional[str] = None,
        prompt_ref: Optional[PromptRef] = None,
    ) -> PromptGroupRecoveryRecord:
        """Create one logical group and its first physical sibling attempts."""
        cut.require_live()
        if not prompt_id:
            raise ValueError("prompt_id must not be empty")
        if prompt_ref is None:
            task_name = prompt_payload.get("task_name")
            if task_name is not None and not isinstance(task_name, str):
                raise TypeError("prompt task_name must be a string or None")
            prompt_ref = PromptRef(sample_id=prompt_id, task_name=task_name)
        if prompt_ref.sample_id != prompt_id:
            raise ValueError(
                "dataset prompt reference must match prompt_id: "
                f"{prompt_ref.sample_id!r} != {prompt_id!r}"
            )
        if expected_generations < 1:
            raise ValueError("expected_generations must be at least one")
        group_id = group_id or str(uuid.uuid4())
        if group_id in self._groups:
            raise ValueError(f"duplicate recovery group_id={group_id!r}")
        admission_id = admission_id or group_id
        if not admission_id:
            raise ValueError("admission_id must not be empty")

        siblings = []
        for generation_index in range(expected_generations):
            siblings.append(
                RolloutSiblingRecord(
                    generation_index=generation_index,
                    attempts=[_new_attempt(0)],
                )
            )
        record = PromptGroupRecoveryRecord(
            group_id=group_id,
            admission_id=admission_id,
            prompt_id=prompt_id,
            prompt_ref=prompt_ref,
            task_source=task_source,
            resolved_agent_name=None,
            recovery_granularity=recovery_granularity,
            # Retain the immutable dataloader sample by reference instead of copying
            # a potentially 131k-token payload. This cache is never serialized and
            # is released as soon as canonical rows take over recovery ownership.
            runtime_prompt_payload=prompt_payload,
            expected_generations=expected_generations,
            target_step=target_step,
            start_weight_version=start_weight_version,
            siblings=siblings,
            phase=(
                PromptGroupPhase.ADMITTED if admitted else PromptGroupPhase.RESERVED
            ),
        )
        self._groups[group_id] = record
        return self._copy_group(record)

    def mark_group_admitted(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        *,
        target_step: Optional[int],
        start_weight_version: int,
    ) -> None:
        """Commit sampler admission without replacing sibling lineage."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.phase is not PromptGroupPhase.RESERVED:
            raise ValueError(
                f"recovery group {group_id!r} is already {record.phase.value}"
            )
        record.target_step = target_step
        record.start_weight_version = start_weight_version
        record.phase = PromptGroupPhase.ADMITTED

    def bind_runtime_prompt(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        prompt_payload: DatumSpec,
    ) -> None:
        """Attach a dataset-reconstructed prompt after identity validation."""
        cut.require_live()
        record = self._require_group(group_id)
        _validate_prompt_identity(
            record.prompt_ref,
            prompt_payload,
            group_id=group_id,
        )
        record.runtime_prompt_payload = prompt_payload

    def prepare_for_restart(self, cut: DataPlaneMutationCut) -> None:
        """Apply each group's persisted restore policy to interrupted attempts."""
        cut.require_live()
        self.assert_checkpoint_safe()
        for record in self._groups.values():
            if record.status is PromptGroupStatus.GENERATING:
                if record.recovery_granularity is RecoveryGranularity.PROMPT_GROUP:
                    self._abandon_entire_group(record)
                else:
                    self.abandon_unsealed(cut, record.group_id)

    def _abandon_entire_group(self, record: PromptGroupRecoveryRecord) -> None:
        """Discard every current sibling when an incomplete group is atomic.

        Sealed staging rows become unreferenced here. The controller's restore
        inventory pass removes those rows from TQ before redispatch.
        """
        for sibling in record.siblings:
            attempt = sibling.current_attempt
            attempt.status = RolloutAttemptStatus.ABANDONED
            attempt.receipt = None
            attempt.completion_receipt = None
            attempt.reward = None
            attempt.mask_sample = None
            attempt.staging_keys.clear()
        record.resolved_agent_name = None
        record.status = PromptGroupStatus.GENERATING

    def assert_checkpoint_safe(self) -> None:
        """Reject states whose canonical publication outcome is ambiguous."""
        unsafe = [
            record.group_id
            for record in self._groups.values()
            if record.status
            in {
                PromptGroupStatus.FINALIZING,
                PromptGroupStatus.FINALIZATION_UNKNOWN,
            }
        ]
        if unsafe:
            raise RuntimeError(
                "rollout recovery contains checkpoint-unsafe group states: "
                f"groups={unsafe!r}"
            )

    def expected_staging_keys(self) -> set[str]:
        """Return staged token rows still owned by sealed sibling attempts."""
        return {
            staging_key
            for record in self._groups.values()
            for sibling in record.siblings
            for attempt in sibling.attempts[-1:]
            if attempt.status is RolloutAttemptStatus.SEALED
            for staging_key in attempt.staging_keys
        }

    def prepare_incomplete_retry(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> PromptGroupRecoveryRecord:
        """Mint fresh physical attempts according to the persisted granularity."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.status != PromptGroupStatus.GENERATING:
            raise ValueError(
                f"cannot retry group {group_id!r} from status {record.status.value!r}"
            )
        current_statuses = [
            sibling.current_attempt.status for sibling in record.siblings
        ]
        retry_prompt_group = (
            record.recovery_granularity is RecoveryGranularity.PROMPT_GROUP
            and any(
                status
                in {
                    RolloutAttemptStatus.ABANDONED,
                    RolloutAttemptStatus.FAILED,
                }
                for status in current_statuses
            )
        )
        if retry_prompt_group and any(
            status
            not in {
                RolloutAttemptStatus.ABANDONED,
                RolloutAttemptStatus.FAILED,
            }
            for status in current_statuses
        ):
            raise ValueError(
                "prompt-group retry requires every sibling attempt to be abandoned "
                "or failed together"
            )

        for sibling in record.siblings:
            attempt = sibling.current_attempt
            if (
                record.recovery_granularity is RecoveryGranularity.SIBLING
                and attempt.status == RolloutAttemptStatus.SEALED
            ):
                continue
            if attempt.status == RolloutAttemptStatus.RESERVED:
                continue
            if attempt.status not in {
                RolloutAttemptStatus.ABANDONED,
                RolloutAttemptStatus.FAILED,
            }:
                raise ValueError(
                    "cannot retry logical rollout "
                    f"{record.logical_rollout_id(sibling.generation_index)!r} "
                    f"from status {attempt.status.value!r}"
                )
            sibling.attempts.append(
                _new_attempt(sibling.current_attempt.attempt_index + 1)
            )
        return self._copy_group(record)

    def mark_group_dispatched(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        *,
        generation_indices: Optional[list[int]] = None,
    ) -> None:
        """Move the selected current sibling attempts to dispatched."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.phase is not PromptGroupPhase.ADMITTED:
            raise ValueError(f"cannot dispatch unadmitted recovery group {group_id!r}")
        if record.status != PromptGroupStatus.GENERATING:
            raise ValueError(
                f"cannot dispatch group {group_id!r} from {record.status.value!r}"
            )
        indices = (
            generation_indices
            if generation_indices is not None
            else list(range(record.expected_generations))
        )
        attempts = [
            self._require_sibling(record, index).current_attempt for index in indices
        ]
        if any(attempt.status != RolloutAttemptStatus.RESERVED for attempt in attempts):
            raise ValueError("only reserved rollout attempts may be dispatched")
        for attempt in attempts:
            attempt.status = RolloutAttemptStatus.DISPATCHED

    def mark_sibling_sealed(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        *,
        generation_index: int,
        gate_rollout_id: str,
        receipt: Optional[dict[str, Any]],
        completion_receipt: Optional[GymCompletionReceipt] = None,
        reward: float,
        mask_sample: bool,
        resolved_agent_name: str,
    ) -> None:
        """Record one streamed sibling receipt as soon as the row arrives."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.recovery_granularity is RecoveryGranularity.PROMPT_GROUP:
            raise ValueError("prompt-group recovery must seal every sibling atomically")
        sibling = self._require_sibling(record, generation_index)
        attempt = sibling.current_attempt
        expected_gate_rollout_id = record.gate_rollout_id(generation_index)
        staging_keys = _receipt_staging_keys(receipt)
        if not isinstance(mask_sample, bool):
            raise TypeError("mask_sample must be a bool")
        self._validate_resolved_agent_name(record, resolved_agent_name)
        if gate_rollout_id != expected_gate_rollout_id:
            raise ValueError(
                "streamed rollout identity mismatch: "
                f"result={gate_rollout_id!r}, expected={expected_gate_rollout_id!r}"
            )
        if receipt is not None and receipt.get("rollout_id") != gate_rollout_id:
            raise ValueError(
                "receipt rollout identity mismatch: "
                f"receipt={receipt.get('rollout_id')!r}, expected={gate_rollout_id!r}"
            )
        logical_rollout_id = record.logical_rollout_id(generation_index)
        if completion_receipt is not None and (
            completion_receipt.rollout_id != logical_rollout_id
            or completion_receipt.attempt_index != attempt.attempt_index
        ):
            raise ValueError(
                "Gym completion receipt identity mismatch: "
                f"receipt={(completion_receipt.rollout_id, completion_receipt.attempt_index)!r}, "
                f"expected={(logical_rollout_id, attempt.attempt_index)!r}"
            )
        if attempt.status == RolloutAttemptStatus.SEALED:
            if (
                attempt.receipt == receipt
                and attempt.completion_receipt
                == (
                    completion_receipt.model_dump(mode="json")
                    if completion_receipt is not None
                    else None
                )
                and attempt.reward == float(reward)
                and attempt.mask_sample is mask_sample
                and attempt.staging_keys == staging_keys
            ):
                return
            raise ValueError(
                "conflicting duplicate seal for "
                f"{record.logical_rollout_id(generation_index)!r}"
            )
        if attempt.status != RolloutAttemptStatus.DISPATCHED:
            raise ValueError(
                "cannot seal logical rollout "
                f"{record.logical_rollout_id(generation_index)!r} "
                f"from status {attempt.status.value!r}"
            )

        attempt.receipt = copy.deepcopy(receipt)
        attempt.completion_receipt = (
            completion_receipt.model_dump(mode="json")
            if completion_receipt is not None
            else None
        )
        attempt.reward = float(reward)
        attempt.mask_sample = mask_sample
        attempt.staging_keys = staging_keys
        attempt.status = RolloutAttemptStatus.SEALED
        record.resolved_agent_name = resolved_agent_name
        if all(
            item.current_attempt.status == RolloutAttemptStatus.SEALED
            for item in record.siblings
        ):
            record.status = PromptGroupStatus.READY_TO_FINALIZE

    def mark_group_sealed(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        results: dict[int, SiblingSealResult],
    ) -> None:
        """Atomically seal one complete prompt-group-scoped physical cohort."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.recovery_granularity is not RecoveryGranularity.PROMPT_GROUP:
            raise ValueError(
                "atomic group sealing requires prompt-group recovery granularity"
            )
        if record.status is not PromptGroupStatus.GENERATING:
            raise ValueError(
                f"cannot seal group {group_id!r} from status {record.status.value!r}"
            )
        expected_indices = set(range(record.expected_generations))
        if set(results) != expected_indices:
            raise ValueError(
                "prompt-group seal requires every logical sibling exactly once: "
                f"expected={sorted(expected_indices)}, actual={sorted(results)}"
            )

        validated: list[tuple[RolloutAttemptRecord, SiblingSealResult, list[str]]] = []
        resolved_agent_names = {
            result.resolved_agent_name for result in results.values()
        }
        if len(resolved_agent_names) != 1:
            raise ValueError(
                "prompt-group completions resolved to inconsistent Gym agents: "
                f"{sorted(resolved_agent_names)!r}"
            )
        resolved_agent_name = next(iter(resolved_agent_names))
        self._validate_resolved_agent_name(record, resolved_agent_name)
        for generation_index in range(record.expected_generations):
            result = results[generation_index]
            sibling = self._require_sibling(record, generation_index)
            attempt = sibling.current_attempt
            expected_gate_rollout_id = record.gate_rollout_id(generation_index)
            if attempt.status is not RolloutAttemptStatus.DISPATCHED:
                raise ValueError(
                    "cannot seal logical rollout "
                    f"{record.logical_rollout_id(generation_index)!r} "
                    f"from status {attempt.status.value!r}"
                )
            if result.gate_rollout_id != expected_gate_rollout_id:
                raise ValueError(
                    "streamed rollout identity mismatch: "
                    f"result={result.gate_rollout_id!r}, "
                    f"expected={expected_gate_rollout_id!r}"
                )
            if (
                result.receipt is not None
                and result.receipt.get("rollout_id") != expected_gate_rollout_id
            ):
                raise ValueError(
                    "receipt rollout identity mismatch: "
                    f"receipt={result.receipt.get('rollout_id')!r}, "
                    f"expected={expected_gate_rollout_id!r}"
                )
            logical_rollout_id = record.logical_rollout_id(generation_index)
            if result.completion_receipt is not None and (
                result.completion_receipt.rollout_id != logical_rollout_id
                or result.completion_receipt.attempt_index != attempt.attempt_index
            ):
                raise ValueError(
                    "Gym completion receipt identity mismatch: "
                    f"receipt={(result.completion_receipt.rollout_id, result.completion_receipt.attempt_index)!r}, "
                    f"expected={(logical_rollout_id, attempt.attempt_index)!r}"
                )
            if not isinstance(result.mask_sample, bool):
                raise TypeError("mask_sample must be a bool")
            validated.append((attempt, result, _receipt_staging_keys(result.receipt)))

        # Validate the complete cohort before changing any sibling. A checkpoint
        # therefore observes either no committed siblings or the complete group.
        for attempt, result, staging_keys in validated:
            attempt.receipt = copy.deepcopy(result.receipt)
            attempt.completion_receipt = (
                result.completion_receipt.model_dump(mode="json")
                if result.completion_receipt is not None
                else None
            )
            attempt.reward = float(result.reward)
            attempt.mask_sample = result.mask_sample
            attempt.staging_keys = staging_keys
            attempt.status = RolloutAttemptStatus.SEALED
        record.resolved_agent_name = resolved_agent_name
        record.status = PromptGroupStatus.READY_TO_FINALIZE

    @staticmethod
    def _validate_resolved_agent_name(
        record: PromptGroupRecoveryRecord,
        resolved_agent_name: str,
    ) -> None:
        if not isinstance(resolved_agent_name, str) or not resolved_agent_name:
            raise ValueError("resolved_agent_name must be a non-empty string")
        if (
            record.resolved_agent_name is not None
            and record.resolved_agent_name != resolved_agent_name
        ):
            raise ValueError(
                "prompt group resolved to inconsistent Gym agents: "
                f"existing={record.resolved_agent_name!r}, "
                f"new={resolved_agent_name!r}"
            )

    def abandon_unsealed(self, cut: DataPlaneMutationCut, group_id: str) -> None:
        """Abandon failed work at the group's persisted recovery granularity."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.status not in {
            PromptGroupStatus.GENERATING,
            PromptGroupStatus.READY_TO_FINALIZE,
        }:
            raise ValueError(
                f"cannot abandon group {group_id!r} from {record.status.value!r}"
            )
        if (
            record.recovery_granularity is RecoveryGranularity.PROMPT_GROUP
            and record.status is PromptGroupStatus.GENERATING
        ):
            self._abandon_entire_group(record)
            return
        for sibling in record.siblings:
            attempt = sibling.current_attempt
            if attempt.status == RolloutAttemptStatus.SEALED:
                continue
            attempt.status = RolloutAttemptStatus.ABANDONED
        record.status = (
            PromptGroupStatus.READY_TO_FINALIZE
            if all(
                sibling.current_attempt.status == RolloutAttemptStatus.SEALED
                for sibling in record.siblings
            )
            else PromptGroupStatus.GENERATING
        )

    def finalization_inputs(
        self, group_id: str
    ) -> tuple[
        list[str],
        list[str],
        list[Optional[dict[str, Any]]],
        list[float],
        list[bool],
    ]:
        """Return sealed finalization inputs in stable sibling order."""
        record = self._require_group(group_id)
        if record.status != PromptGroupStatus.READY_TO_FINALIZE:
            raise ValueError(
                f"group {group_id!r} is not ready to finalize: {record.status.value!r}"
            )
        receipts: list[Optional[dict[str, Any]]] = []
        rewards: list[float] = []
        mask_sample: list[bool] = []
        for sibling in record.siblings:
            attempt = sibling.current_attempt
            if (
                attempt.status != RolloutAttemptStatus.SEALED
                or attempt.reward is None
                or attempt.mask_sample is None
            ):
                raise ValueError(
                    "logical rollout "
                    f"{record.logical_rollout_id(sibling.generation_index)!r} "
                    "is not sealed"
                )
            receipts.append(copy.deepcopy(attempt.receipt))
            rewards.append(attempt.reward)
            mask_sample.append(attempt.mask_sample)
        return (
            record.gate_rollout_ids,
            record.logical_rollout_ids,
            receipts,
            rewards,
            mask_sample,
        )

    def record_sealed_sibling_acknowledgement(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        generation_index: int,
    ) -> GymCompletedExecution:
        """Persist one sibling-scoped ACK obligation alongside its seal."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.recovery_granularity is not RecoveryGranularity.SIBLING:
            raise ValueError(
                "individual Gym acknowledgement requires sibling recovery granularity"
            )
        sibling = self._require_sibling(record, generation_index)
        acknowledgement = self._acknowledgement_for_sibling(record, sibling)
        self._record_completed_execution_acknowledgements([acknowledgement])
        return acknowledgement

    def record_partial_group_acknowledgement(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        generation_index: int,
        result: SiblingSealResult,
    ) -> GymCompletedExecution:
        """Persist an ACK for a partial prompt-group completion.

        Prompt-group recovery keeps the group seal atomic, so an individual
        completed sibling remains DISPATCHED until the complete cohort arrives.
        Its Gym terminal result can still be released once the exact completion
        receipt has been durably added to the independent ACK outbox.
        """
        cut.require_live()
        record = self._require_group(group_id)
        if record.recovery_granularity is not RecoveryGranularity.PROMPT_GROUP:
            raise ValueError(
                "partial Gym group acknowledgement requires prompt-group "
                "recovery granularity"
            )
        if record.status is not PromptGroupStatus.GENERATING:
            raise ValueError(
                f"cannot record a partial completion for group {group_id!r} "
                f"from status {record.status.value!r}"
            )
        sibling = self._require_sibling(record, generation_index)
        attempt = sibling.current_attempt
        if attempt.status is not RolloutAttemptStatus.DISPATCHED:
            raise ValueError(
                "cannot acknowledge logical rollout "
                f"{record.logical_rollout_id(generation_index)!r} "
                f"from status {attempt.status.value!r}"
            )
        expected_gate_rollout_id = record.gate_rollout_id(generation_index)
        if result.gate_rollout_id != expected_gate_rollout_id:
            raise ValueError(
                "streamed rollout identity mismatch: "
                f"result={result.gate_rollout_id!r}, "
                f"expected={expected_gate_rollout_id!r}"
            )
        self._validate_resolved_agent_name(record, result.resolved_agent_name)
        completion_receipt = result.completion_receipt
        if completion_receipt is None:
            raise RuntimeError(
                "cannot acknowledge a completed Gym execution without its exact "
                "completion receipt"
            )
        logical_rollout_id = record.logical_rollout_id(generation_index)
        receipt_identity = (
            completion_receipt.rollout_id,
            completion_receipt.attempt_index,
        )
        expected_identity = (logical_rollout_id, attempt.attempt_index)
        if receipt_identity != expected_identity:
            raise ValueError(
                "Gym completion receipt identity mismatch: "
                f"receipt={receipt_identity!r}, expected={expected_identity!r}"
            )
        acknowledgement = self._acknowledgement_from_completion_receipt(
            rollout_id=logical_rollout_id,
            attempt_index=attempt.attempt_index,
            agent_name=result.resolved_agent_name,
            completion_receipt=completion_receipt,
        )
        self._record_completed_execution_acknowledgements([acknowledgement])
        return acknowledgement

    def record_sealed_group_acknowledgements(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> list[GymCompletedExecution]:
        """Persist every prompt-group-scoped ACK after its atomic seal."""
        cut.require_live()
        record = self._require_group(group_id)
        if record.recovery_granularity is not RecoveryGranularity.PROMPT_GROUP:
            raise ValueError(
                "atomic Gym group acknowledgement requires prompt-group recovery "
                "granularity"
            )
        acknowledgements = [
            self._acknowledgement_for_sibling(record, sibling)
            for sibling in record.siblings
        ]
        self._record_completed_execution_acknowledgements(acknowledgements)
        return acknowledgements

    def pending_completed_execution_acknowledgements(
        self,
    ) -> list[GymCompletedExecution]:
        """Return a stable copy of Gym ACK obligations not confirmed remotely."""
        return sorted(
            self._pending_completed_execution_acknowledgements.values(),
            key=lambda item: (
                item.agent_name,
                item.receipt.rollout_id,
                item.receipt.attempt_index,
            ),
        )

    def pending_completed_execution_acknowledgement_count(self) -> int:
        """Return the pending Gym ACK count without copying or sorting the outbox."""
        return len(self._pending_completed_execution_acknowledgements)

    def discard_completed_execution_acknowledgements(
        self,
        cut: DataPlaneMutationCut,
    ) -> int:
        """Drop ACKs whose owning Gym process was not restored.

        A trainer-only fallback deliberately starts a fresh Gym deployment, so
        acknowledgements addressed to executions in the pre-crash process can
        never be satisfied. The corresponding rollout results are already
        durable in TQ; only the obsolete remote-cleanup obligations are removed.
        """
        cut.require_live()
        count = len(self._pending_completed_execution_acknowledgements)
        self._pending_completed_execution_acknowledgements.clear()
        return count

    def mark_completed_executions_acknowledged(
        self,
        cut: DataPlaneMutationCut,
        acknowledgements: list[GymCompletedExecution],
    ) -> None:
        """Remove only ACK obligations confirmed by Gym's idempotent endpoint."""
        cut.require_live()
        seen: set[tuple[str, int]] = set()
        for acknowledgement in acknowledgements:
            if acknowledgement.identity in seen:
                raise ValueError("completed execution acknowledgements must be unique")
            seen.add(acknowledgement.identity)
            pending = self._pending_completed_execution_acknowledgements.get(
                acknowledgement.identity
            )
            if pending is None:
                continue
            if pending != acknowledgement:
                raise RuntimeError(
                    "Gym acknowledgement changed for "
                    f"{acknowledgement.identity!r}: "
                    f"{_completed_execution_differences(pending, acknowledgement)}"
                )
            del self._pending_completed_execution_acknowledgements[
                acknowledgement.identity
            ]

    @staticmethod
    def _acknowledgement_for_sibling(
        record: PromptGroupRecoveryRecord,
        sibling: RolloutSiblingRecord,
    ) -> GymCompletedExecution:
        if record.resolved_agent_name is None:
            raise RuntimeError(
                f"group {record.group_id!r} has no resolved Gym agent identity"
            )
        attempt = sibling.current_attempt
        if attempt.status is not RolloutAttemptStatus.SEALED:
            raise RuntimeError(
                "cannot acknowledge unsealed logical rollout "
                f"{record.logical_rollout_id(sibling.generation_index)!r}"
            )
        if attempt.completion_receipt is None:
            raise RuntimeError(
                "cannot acknowledge a sealed Gym execution without its exact "
                f"completion receipt: rollout={record.logical_rollout_id(sibling.generation_index)!r}, "
                f"attempt_index={attempt.attempt_index}"
            )
        completion_receipt = GymCompletionReceipt.model_validate(
            attempt.completion_receipt
        )
        return RolloutRecoveryLedger._acknowledgement_from_completion_receipt(
            rollout_id=record.logical_rollout_id(sibling.generation_index),
            attempt_index=attempt.attempt_index,
            agent_name=record.resolved_agent_name,
            completion_receipt=completion_receipt,
        )

    @staticmethod
    def _acknowledgement_from_completion_receipt(
        *,
        rollout_id: str,
        attempt_index: int,
        agent_name: str,
        completion_receipt: GymCompletionReceipt,
    ) -> GymCompletedExecution:
        expected_identity = (rollout_id, attempt_index)
        if completion_receipt.identity != expected_identity:
            raise ValueError(
                "Gym completion receipt identity mismatch: "
                f"receipt={completion_receipt.identity!r}, "
                f"expected={expected_identity!r}"
            )
        return GymCompletedExecution(
            receipt=completion_receipt,
            agent_name=agent_name,
        )

    def _record_completed_execution_acknowledgements(
        self,
        acknowledgements: list[GymCompletedExecution],
    ) -> None:
        """Validate the complete batch before publishing any obligation."""
        by_identity: dict[tuple[str, int], GymCompletedExecution] = {}
        for acknowledgement in acknowledgements:
            batch_previous = by_identity.setdefault(
                acknowledgement.identity,
                acknowledgement,
            )
            if batch_previous != acknowledgement:
                raise RuntimeError(
                    "conflicting Gym acknowledgements in one batch for "
                    f"{acknowledgement.identity!r}: "
                    f"{_completed_execution_differences(batch_previous, acknowledgement)}"
                )
            previous = self._pending_completed_execution_acknowledgements.get(
                acknowledgement.identity
            )
            if previous is not None and previous != acknowledgement:
                raise RuntimeError(
                    "conflicting Gym acknowledgement for "
                    f"{acknowledgement.identity!r}: "
                    f"{_completed_execution_differences(previous, acknowledgement)}"
                )
        self._pending_completed_execution_acknowledgements.update(by_identity)

    def mark_finalization_started(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> None:
        cut.require_live()
        record = self._require_group(group_id)
        self._require_group_status(
            record,
            allowed={PromptGroupStatus.READY_TO_FINALIZE},
            transition="start finalization",
        )
        record.status = PromptGroupStatus.FINALIZING

    def mark_finalization_unknown(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> None:
        cut.require_live()
        record = self._require_group(group_id)
        self._require_group_status(
            record,
            allowed={PromptGroupStatus.FINALIZING},
            transition="mark finalization unknown",
        )
        record.status = PromptGroupStatus.FINALIZATION_UNKNOWN

    def discard_group(self, cut: DataPlaneMutationCut, group_id: str) -> None:
        """Drop a group only after its external TQ/Gate ownership is cleaned."""
        cut.require_live()
        self._require_group(group_id)
        del self._groups[group_id]

    def discard_canonical_groups(
        self,
        cut: DataPlaneMutationCut,
        group_ids: set[str],
    ) -> int:
        """Prefer canonical TQ ownership over a stale unfinished sidecar row."""
        cut.require_live()
        discarded = 0
        for group_id in list(self._groups):
            if group_id in group_ids:
                del self._groups[group_id]
                discarded += 1
        return discarded

    def get_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        return self._copy_group(self._require_group(group_id))

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, group_id: object) -> bool:
        return isinstance(group_id, str) and group_id in self._groups

    def state_dict(self) -> dict[str, Any]:
        """Return the versioned metadata persisted in ``rollout_recovery.pt``."""
        self.assert_checkpoint_safe()
        groups = []
        for record in self._groups.values():
            prompt_payload = record.runtime_prompt_payload
            if prompt_payload is None:
                raise RuntimeError(
                    f"cannot checkpoint recovery group {record.group_id!r} before "
                    "its prompt is rehydrated"
                )
            _validate_prompt_identity(
                record.prompt_ref,
                prompt_payload,
                group_id=record.group_id,
            )
            groups.append(
                {
                    "group_id": record.group_id,
                    "admission_id": record.admission_id,
                    "prompt_id": record.prompt_id,
                    "prompt_ref": {
                        "sample_id": record.prompt_ref.sample_id,
                        "task_name": record.prompt_ref.task_name,
                    },
                    "task_source": record.task_source,
                    "resolved_agent_name": record.resolved_agent_name,
                    "recovery_granularity": record.recovery_granularity.value,
                    "expected_generations": record.expected_generations,
                    "target_step": record.target_step,
                    "start_weight_version": record.start_weight_version,
                    "status": record.status.value,
                    "phase": record.phase.value,
                    "siblings": [
                        {
                            "generation_index": sibling.generation_index,
                            "attempts": [
                                {
                                    "attempt_index": attempt.attempt_index,
                                    "status": attempt.status.value,
                                    "receipt": copy.deepcopy(attempt.receipt),
                                    "completion_receipt": copy.deepcopy(
                                        attempt.completion_receipt
                                    ),
                                    "reward": attempt.reward,
                                    "mask_sample": attempt.mask_sample,
                                    "staging_keys": list(attempt.staging_keys),
                                }
                                for attempt in sibling.attempts
                            ],
                        }
                        for sibling in record.siblings
                    ],
                }
            )
        state = {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": groups,
            "pending_completed_execution_acknowledgements": [
                acknowledgement.model_dump(mode="json")
                for acknowledgement in sorted(
                    self._pending_completed_execution_acknowledgements.values(),
                    key=lambda item: (
                        item.agent_name,
                        item.receipt.rollout_id,
                        item.receipt.attempt_index,
                    ),
                )
            ],
        }
        return RolloutRecoveryLedgerState.model_validate(state).model_dump(mode="json")

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Self:
        """Restore and validate a ledger metadata envelope."""
        if not isinstance(state, dict):
            raise TypeError(
                "rollout recovery state must be a dictionary, got "
                f"{type(state).__name__}"
            )
        schema_version = state.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in _SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS
        ):
            raise ValueError(_unsupported_schema_message(schema_version))
        validated = RolloutRecoveryLedgerState.model_validate(state)

        ledger = cls()
        for acknowledgement in validated.pending_completed_execution_acknowledgements:
            if (
                acknowledgement.identity
                in ledger._pending_completed_execution_acknowledgements
            ):
                raise ValueError(
                    "duplicate completed execution acknowledgement identity="
                    f"{acknowledgement.identity!r}"
                )
            ledger._pending_completed_execution_acknowledgements[
                acknowledgement.identity
            ] = acknowledgement
        for group_state in validated.groups:
            record = cls._group_from_state(group_state)
            if record.group_id in ledger._groups:
                raise ValueError(f"duplicate recovery group_id={record.group_id!r}")
            ledger._groups[record.group_id] = record

        admission_states: dict[str, tuple[PromptGroupPhase, Optional[int]]] = {}
        for record in ledger._groups.values():
            signature = (record.phase, record.target_step)
            previous = admission_states.setdefault(record.admission_id, signature)
            if previous != signature:
                raise ValueError(
                    "rollout recovery groups sharing admission_id="
                    f"{record.admission_id!r} disagree on phase or target_step"
                )
        # The procedural checks above enforce relationships between records.
        # This final model validation independently enforces the complete saved
        # shape, including required nullable fields that ``dict.get`` cannot
        # distinguish from omitted fields.
        RolloutRecoveryLedgerState.model_validate(state)
        return ledger

    def load_state_dict(
        self,
        cut: DataPlaneMutationCut,
        state: RolloutRecoveryState,
    ) -> None:
        """Replace this empty ledger from a validated checkpoint envelope."""
        cut.require_live()
        if self._groups or self._pending_completed_execution_acknowledgements:
            raise RuntimeError(
                "cannot restore into a non-empty rollout recovery ledger"
            )
        restored = self.from_state_dict(state)
        self._groups = restored._groups
        self._pending_completed_execution_acknowledgements = (
            restored._pending_completed_execution_acknowledgements
        )

    @staticmethod
    def _group_from_state(
        group: PromptGroupRecoveryState,
    ) -> PromptGroupRecoveryRecord:
        if group.resolved_agent_name is not None and not group.resolved_agent_name:
            raise ValueError("resolved_agent_name must be a non-empty string or None")
        if len(group.siblings) != group.expected_generations:
            raise ValueError(
                f"recovery group {group.group_id!r} must contain "
                f"{group.expected_generations} siblings"
            )

        siblings: list[RolloutSiblingRecord] = []
        for generation_index, sibling in enumerate(group.siblings):
            if sibling.generation_index != generation_index:
                raise ValueError("generation indices must be contiguous")
            logical_id = f"{group.group_id}_g{generation_index}"
            if not sibling.attempts:
                raise ValueError(f"logical rollout {logical_id!r} has no attempts")
            attempts: list[RolloutAttemptRecord] = []
            for expected_attempt_index, attempt in enumerate(sibling.attempts):
                if attempt.attempt_index != expected_attempt_index:
                    raise ValueError(
                        "attempt indices must be contiguous non-negative integers"
                    )
                gate_id = gym_capture_key(logical_id, attempt.attempt_index)
                if attempt.status is RolloutAttemptStatus.SEALED:
                    if attempt.reward is None:
                        raise ValueError("sealed attempts require a reward")
                    if attempt.mask_sample is None:
                        raise ValueError(
                            "sealed attempts require a boolean mask_sample"
                        )
                    if attempt.receipt is None:
                        if attempt.staging_keys:
                            raise ValueError(
                                "sealed missing-receipt attempt cannot own staging keys"
                            )
                    else:
                        if attempt.receipt.get("rollout_id") != gate_id:
                            raise ValueError("sealed receipt identity mismatch")
                        if (
                            _receipt_staging_keys(attempt.receipt)
                            != attempt.staging_keys
                        ):
                            raise ValueError("sealed receipt staging manifest mismatch")
                    if attempt.completion_receipt is not None:
                        if (
                            attempt.completion_receipt.rollout_id != logical_id
                            or attempt.completion_receipt.attempt_index
                            != attempt.attempt_index
                        ):
                            raise ValueError(
                                "sealed Gym completion receipt identity mismatch"
                            )
                elif (
                    attempt.receipt is not None
                    or attempt.completion_receipt is not None
                    or attempt.reward is not None
                    or attempt.mask_sample is not None
                    or attempt.staging_keys
                ):
                    raise ValueError("only sealed attempts may retain receipt data")
                attempts.append(
                    RolloutAttemptRecord(
                        attempt_index=attempt.attempt_index,
                        status=attempt.status,
                        receipt=copy.deepcopy(attempt.receipt),
                        completion_receipt=(
                            attempt.completion_receipt.model_dump(mode="json")
                            if attempt.completion_receipt is not None
                            else None
                        ),
                        reward=(
                            float(attempt.reward)
                            if attempt.reward is not None
                            else None
                        ),
                        mask_sample=attempt.mask_sample,
                        staging_keys=list(attempt.staging_keys),
                    )
                )
            siblings.append(
                RolloutSiblingRecord(
                    generation_index=generation_index,
                    attempts=attempts,
                )
            )

        if group.prompt_ref.sample_id != group.prompt_id:
            raise ValueError("prompt_ref sample_id must match prompt_id")
        prefinalization_sealed_states = {
            PromptGroupStatus.READY_TO_FINALIZE,
            PromptGroupStatus.FINALIZING,
            PromptGroupStatus.FINALIZATION_UNKNOWN,
        }
        all_current_attempts_sealed = all(
            sibling.current_attempt.status == RolloutAttemptStatus.SEALED
            for sibling in siblings
        )
        if group.status is PromptGroupStatus.GENERATING and all_current_attempts_sealed:
            raise ValueError("generating group must retain an unfinished sibling")
        if (
            group.status in prefinalization_sealed_states
            and not all_current_attempts_sealed
        ):
            raise ValueError(
                f"group state {group.status.value!r} requires every sibling to be sealed"
            )
        if all_current_attempts_sealed and group.resolved_agent_name is None:
            raise ValueError("sealed recovery groups require resolved_agent_name")
        if not all_current_attempts_sealed and group.resolved_agent_name is not None:
            # A sibling-scoped group may have a mix of sealed and unsealed
            # attempts, so retaining the resolved agent remains valid there.
            if not any(
                sibling.current_attempt.status is RolloutAttemptStatus.SEALED
                for sibling in siblings
            ):
                raise ValueError(
                    "resolved_agent_name requires at least one sealed sibling"
                )

        return PromptGroupRecoveryRecord(
            group_id=group.group_id,
            admission_id=group.admission_id,
            prompt_id=group.prompt_id,
            prompt_ref=PromptRef(
                sample_id=group.prompt_ref.sample_id,
                task_name=group.prompt_ref.task_name,
            ),
            task_source=group.task_source,
            resolved_agent_name=group.resolved_agent_name,
            recovery_granularity=group.recovery_granularity,
            runtime_prompt_payload=None,
            expected_generations=group.expected_generations,
            target_step=group.target_step,
            start_weight_version=group.start_weight_version,
            siblings=siblings,
            phase=group.phase,
            status=group.status,
        )

    def _require_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        try:
            return self._groups[group_id]
        except KeyError as error:
            raise ValueError(f"unknown recovery group_id={group_id!r}") from error

    @staticmethod
    def _copy_group(record: PromptGroupRecoveryRecord) -> PromptGroupRecoveryRecord:
        """Copy mutable lineage metadata without duplicating the prompt payload."""
        return dataclasses.replace(
            record,
            siblings=copy.deepcopy(record.siblings),
        )

    @staticmethod
    def _require_sibling(
        record: PromptGroupRecoveryRecord, generation_index: int
    ) -> RolloutSiblingRecord:
        if not 0 <= generation_index < len(record.siblings):
            raise ValueError(
                f"generation_index={generation_index} is outside group "
                f"{record.group_id!r}"
            )
        return record.siblings[generation_index]

    @staticmethod
    def _require_group_status(
        record: PromptGroupRecoveryRecord,
        *,
        allowed: set[PromptGroupStatus],
        transition: str,
    ) -> None:
        if record.status not in allowed:
            raise ValueError(
                f"cannot {transition} group {record.group_id!r} from "
                f"{record.status.value!r}"
            )


def build_rollout_recovery_state(
    ledger: RolloutRecoveryLedger,
    *,
    batch_shortfall: dict[int, int],
    sampler_stamps_target_steps: bool,
) -> RolloutRecoveryState:
    """Build the complete versioned sidecar from ledger and controller state."""
    if not isinstance(sampler_stamps_target_steps, bool):
        raise TypeError(
            "rollout recovery sampler_stamps_target_steps must be a boolean"
        )
    state = ledger.state_dict()
    state["batch_shortfall"] = dict(batch_shortfall)
    state["sampler_stamps_target_steps"] = sampler_stamps_target_steps
    validated = RolloutRecoverySidecarState.model_validate(state)
    serialized = validated.model_dump(mode="json")
    # JSON-mode model dumps are required for UUIDs and enums, but Pydantic also
    # stringifies integer mapping keys. ``torch.save`` supports integer keys,
    # so retain the established batch-shortfall representation on disk.
    serialized["batch_shortfall"] = dict(validated.batch_shortfall)
    return serialized


def parse_rollout_recovery_state(state: object) -> ParsedRolloutRecoveryState:
    """Validate and split a complete checkpoint sidecar by runtime owner."""
    if not isinstance(state, dict):
        raise TypeError(
            "rollout recovery sidecar must contain a dictionary, got "
            f"{type(state).__name__}"
        )
    schema_version = state.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in _SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS
    ):
        raise ValueError(_unsupported_schema_message(schema_version))
    # Unlike the runtime dataclasses, the persisted model has no implicit
    # defaults. A nullable value may be null, but every field must be present.
    # This one validation owns the complete persisted shape. The reconstruction
    # layer below enforces relationships between otherwise valid records.
    validated = RolloutRecoverySidecarState.model_validate(state)
    serialized = validated.model_dump(mode="json")

    ledger_state: RolloutRecoveryState = {
        "schema_version": serialized["schema_version"],
        "groups": serialized["groups"],
        "pending_completed_execution_acknowledgements": serialized[
            "pending_completed_execution_acknowledgements"
        ],
    }
    return ParsedRolloutRecoveryState(
        ledger_state=ledger_state,
        batch_shortfall=dict(validated.batch_shortfall),
        sampler_stamps_target_steps=validated.sampler_stamps_target_steps,
    )
