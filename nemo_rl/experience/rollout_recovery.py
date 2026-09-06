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
reads back on restore.
"""

from __future__ import annotations

import copy
import dataclasses
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional, Self, TypeAlias

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneMutationCut
    from nemo_rl.data.interfaces import DatumSpec

ROLLOUT_RECOVERY_SCHEMA_VERSION = 5
_SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS = {ROLLOUT_RECOVERY_SCHEMA_VERSION}
ROLLOUT_RECOVERY_STATE_FILENAME = "rollout_recovery.pt"
RolloutRecoveryState: TypeAlias = dict[str, Any]

_LEDGER_STATE_FIELDS = frozenset({"schema_version", "groups"})
_SIDECAR_STATE_FIELDS = frozenset(
    {
        *_LEDGER_STATE_FIELDS,
        "batch_shortfall",
        "sampler_stamps_target_steps",
    }
)
_GROUP_STATE_FIELDS = frozenset(
    {
        "group_id",
        "admission_id",
        "prompt_id",
        "prompt_ref",
        "agent_name",
        "recovery_granularity",
        "expected_generations",
        "target_step",
        "start_weight_version",
        "status",
        "phase",
        "siblings",
    }
)
_PROMPT_REF_STATE_FIELDS = frozenset({"sample_id", "task_name"})
_SIBLING_STATE_FIELDS = frozenset({"generation_index", "attempts"})
_ATTEMPT_STATE_FIELDS = frozenset(
    {
        "attempt_uuid",
        "status",
        "receipt",
        "reward",
        "mask_sample",
        "staging_keys",
    }
)


def _reject_unknown_fields(
    mapping: Mapping[Any, Any],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    """Reject fields that are not part of the current versioned schema."""
    unknown = set(mapping) - expected
    if unknown:
        raise ValueError(
            f"{context} contains unknown fields: {sorted(unknown, key=repr)!r}"
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

    attempt_uuid: uuid.UUID
    status: RolloutAttemptStatus
    receipt: Optional[dict[str, Any]] = None
    reward: Optional[float] = None
    mask_sample: Optional[bool] = None
    staging_keys: list[str] = field(default_factory=list)

    @property
    def attempt_id(self) -> str:
        """Return the compact external representation of this attempt UUID."""
        return self.attempt_uuid.hex


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
    agent_name: Optional[str]
    recovery_granularity: RecoveryGranularity
    runtime_prompt_payload: Optional[DatumSpec]
    expected_generations: int
    target_step: Optional[int]
    start_weight_version: int
    siblings: list[RolloutSiblingRecord]
    phase: PromptGroupPhase
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
        """Derive the physical Gate ID from group, sibling, and attempt UUID."""
        sibling = self.siblings[generation_index]
        return (
            f"{self.logical_rollout_id(generation_index)}"
            f"_a{sibling.current_attempt.attempt_id}"
        )

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
    mask_sample: bool = False


def _new_attempt() -> RolloutAttemptRecord:
    return RolloutAttemptRecord(
        attempt_uuid=uuid.uuid4(),
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


class RolloutRecoveryLedger:
    """In-memory source of truth for token-capture rollout ownership.

    Controller-owned mutations require a live data-plane cut so lineage cannot
    change outside the checkpoint barrier's consistent snapshot boundary.
    """

    def __init__(self) -> None:
        self._groups: dict[str, PromptGroupRecoveryRecord] = {}

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
        agent_name: Optional[str] = None,
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
                    attempts=[_new_attempt()],
                )
            )
        record = PromptGroupRecoveryRecord(
            group_id=group_id,
            admission_id=admission_id,
            prompt_id=prompt_id,
            prompt_ref=prompt_ref,
            agent_name=agent_name,
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
            attempt.reward = None
            attempt.staging_keys.clear()
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
            sibling.attempts.append(_new_attempt())
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
        reward: float,
        mask_sample: bool = False,
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
        if attempt.status == RolloutAttemptStatus.SEALED:
            if (
                attempt.receipt == receipt
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
        attempt.reward = float(reward)
        attempt.mask_sample = mask_sample
        attempt.staging_keys = staging_keys
        attempt.status = RolloutAttemptStatus.SEALED
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
            if not isinstance(result.mask_sample, bool):
                raise TypeError("mask_sample must be a bool")
            validated.append((attempt, result, _receipt_staging_keys(result.receipt)))

        # Validate the complete cohort before changing any sibling. A checkpoint
        # therefore observes either no committed siblings or the complete group.
        for attempt, result, staging_keys in validated:
            attempt.receipt = copy.deepcopy(result.receipt)
            attempt.reward = float(result.reward)
            attempt.mask_sample = result.mask_sample
            attempt.staging_keys = staging_keys
            attempt.status = RolloutAttemptStatus.SEALED
        record.status = PromptGroupStatus.READY_TO_FINALIZE

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
                    "agent_name": record.agent_name,
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
                                    "attempt_uuid": attempt.attempt_uuid.bytes,
                                    "status": attempt.status.value,
                                    "receipt": copy.deepcopy(attempt.receipt),
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
        return {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": groups,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Self:
        """Restore and validate a ledger metadata envelope."""
        if not isinstance(state, dict):
            raise TypeError(
                "rollout recovery state must be a dictionary, got "
                f"{type(state).__name__}"
            )
        _reject_unknown_fields(
            state,
            expected=_LEDGER_STATE_FIELDS,
            context="rollout recovery state",
        )
        schema_version = state.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in _SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS
        ):
            raise ValueError(
                f"Unsupported rollout-recovery schema version: {schema_version!r}"
            )
        raw_groups = state.get("groups")
        if not isinstance(raw_groups, list):
            raise ValueError("rollout-recovery state must contain a groups list")

        ledger = cls()
        seen_attempt_uuids: set[uuid.UUID] = set()
        for raw_group in raw_groups:
            record = cls._group_from_state(
                raw_group,
                seen_attempt_uuids=seen_attempt_uuids,
            )
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
        return ledger

    def load_state_dict(
        self,
        cut: DataPlaneMutationCut,
        state: RolloutRecoveryState,
    ) -> None:
        """Replace this empty ledger from a validated checkpoint envelope."""
        cut.require_live()
        if self._groups:
            raise RuntimeError(
                "cannot restore into a non-empty rollout recovery ledger"
            )
        restored = self.from_state_dict(state)
        self._groups = restored._groups

    @classmethod
    def _group_from_state(
        cls,
        raw_group: Any,
        *,
        seen_attempt_uuids: set[uuid.UUID],
    ) -> PromptGroupRecoveryRecord:
        if not isinstance(raw_group, dict):
            raise ValueError("rollout-recovery group must be a mapping")
        _reject_unknown_fields(
            raw_group,
            expected=_GROUP_STATE_FIELDS,
            context="rollout-recovery group",
        )
        group_id = raw_group.get("group_id")
        admission_id = raw_group.get("admission_id")
        prompt_id = raw_group.get("prompt_id")
        agent_name = raw_group.get("agent_name")
        raw_recovery_granularity = raw_group.get("recovery_granularity")
        expected_generations = raw_group.get("expected_generations")
        siblings_state = raw_group.get("siblings")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        if not isinstance(admission_id, str) or not admission_id:
            raise ValueError("admission_id must be a non-empty string")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("prompt_id must be a non-empty string")
        if agent_name is not None and not isinstance(agent_name, str):
            raise ValueError("agent_name must be a string or None")
        if not isinstance(raw_recovery_granularity, str):
            raise ValueError("recovery_granularity must be a string")
        try:
            recovery_granularity = RecoveryGranularity(raw_recovery_granularity)
        except ValueError as error:
            raise ValueError(
                f"invalid recovery_granularity={raw_recovery_granularity!r}"
            ) from error
        if not isinstance(expected_generations, int) or expected_generations < 1:
            raise ValueError("expected_generations must be a positive integer")
        if (
            not isinstance(siblings_state, list)
            or len(siblings_state) != expected_generations
        ):
            raise ValueError(
                f"recovery group {group_id!r} must contain "
                f"{expected_generations} siblings"
            )
        raw_status = raw_group.get("status")
        if not isinstance(raw_status, str):
            raise ValueError(f"invalid prompt group status={raw_status!r}")
        try:
            status = PromptGroupStatus(raw_status)
        except ValueError as error:
            raise ValueError(f"invalid prompt group status={raw_status!r}") from error
        raw_phase = raw_group.get("phase")
        if not isinstance(raw_phase, str):
            raise ValueError(f"invalid prompt group phase={raw_phase!r}")
        try:
            phase = PromptGroupPhase(raw_phase)
        except ValueError as error:
            raise ValueError(f"invalid prompt group phase={raw_phase!r}") from error

        siblings: list[RolloutSiblingRecord] = []
        for generation_index, sibling_state in enumerate(siblings_state):
            if not isinstance(sibling_state, dict):
                raise ValueError("rollout-recovery sibling must be a mapping")
            _reject_unknown_fields(
                sibling_state,
                expected=_SIBLING_STATE_FIELDS,
                context="rollout-recovery sibling",
            )
            if sibling_state.get("generation_index") != generation_index:
                raise ValueError("generation indices must be contiguous")
            logical_id = f"{group_id}_g{generation_index}"
            attempts_state = sibling_state.get("attempts")
            if not isinstance(attempts_state, list) or not attempts_state:
                raise ValueError(f"logical rollout {logical_id!r} has no attempts")
            attempts: list[RolloutAttemptRecord] = []
            for attempt_state in attempts_state:
                if not isinstance(attempt_state, dict):
                    raise ValueError("rollout-recovery attempt must be a mapping")
                _reject_unknown_fields(
                    attempt_state,
                    expected=_ATTEMPT_STATE_FIELDS,
                    context="rollout-recovery attempt",
                )
                raw_attempt_uuid = attempt_state.get("attempt_uuid")
                if (
                    not isinstance(raw_attempt_uuid, bytes)
                    or len(raw_attempt_uuid) != 16
                ):
                    raise ValueError("attempt_uuid must contain exactly 16 bytes")
                attempt_uuid = uuid.UUID(bytes=raw_attempt_uuid)
                if attempt_uuid in seen_attempt_uuids:
                    raise ValueError("duplicate rollout attempt identity")
                seen_attempt_uuids.add(attempt_uuid)
                gate_id = f"{logical_id}_a{attempt_uuid.hex}"
                raw_attempt_status = attempt_state.get("status")
                if not isinstance(raw_attempt_status, str):
                    raise ValueError(
                        f"invalid rollout attempt status={raw_attempt_status!r}"
                    )
                try:
                    attempt_status = RolloutAttemptStatus(raw_attempt_status)
                except ValueError as error:
                    raise ValueError(
                        f"invalid rollout attempt status={raw_attempt_status!r}"
                    ) from error
                receipt = attempt_state.get("receipt")
                reward = attempt_state.get("reward")
                mask_sample = attempt_state.get("mask_sample")
                staging_keys = attempt_state.get("staging_keys")
                if not isinstance(staging_keys, list) or not all(
                    isinstance(key, str) for key in staging_keys
                ):
                    raise ValueError("staging_keys must be a list of strings")
                if attempt_status == RolloutAttemptStatus.SEALED:
                    if not isinstance(reward, (int, float)):
                        raise ValueError("sealed attempts require a reward")
                    if not isinstance(mask_sample, bool):
                        raise ValueError("sealed attempts require a boolean mask_sample")
                    if receipt is None:
                        if staging_keys:
                            raise ValueError(
                                "sealed missing-receipt attempt cannot own staging keys"
                            )
                    elif isinstance(receipt, dict):
                        if receipt.get("rollout_id") != gate_id:
                            raise ValueError("sealed receipt identity mismatch")
                        if _receipt_staging_keys(receipt) != staging_keys:
                            raise ValueError("sealed receipt staging manifest mismatch")
                    else:
                        raise ValueError(
                            "sealed attempt receipt must be a mapping or None"
                        )
                elif (
                    receipt is not None
                    or reward is not None
                    or mask_sample is not None
                    or staging_keys
                ):
                    raise ValueError("only sealed attempts may retain receipt data")
                attempts.append(
                    RolloutAttemptRecord(
                        attempt_uuid=attempt_uuid,
                        status=attempt_status,
                        receipt=copy.deepcopy(receipt),
                        reward=float(reward) if reward is not None else None,
                        mask_sample=mask_sample,
                        staging_keys=list(staging_keys),
                    )
                )
            siblings.append(
                RolloutSiblingRecord(
                    generation_index=generation_index,
                    attempts=attempts,
                )
            )

        raw_prompt_ref = raw_group.get("prompt_ref")
        if not isinstance(raw_prompt_ref, dict):
            raise ValueError("prompt_ref must be a mapping")
        _reject_unknown_fields(
            raw_prompt_ref,
            expected=_PROMPT_REF_STATE_FIELDS,
            context="rollout-recovery prompt_ref",
        )
        sample_id = raw_prompt_ref.get("sample_id")
        task_name = raw_prompt_ref.get("task_name")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("prompt_ref sample_id must be a non-empty string")
        if task_name is not None and not isinstance(task_name, str):
            raise ValueError("prompt_ref task_name must be a string or None")
        if sample_id != prompt_id:
            raise ValueError("prompt_ref sample_id must match prompt_id")
        target_step = raw_group.get("target_step")
        if target_step is not None and not isinstance(target_step, int):
            raise ValueError("target_step must be an integer or None")
        start_weight = raw_group.get("start_weight_version")
        if not isinstance(start_weight, int):
            raise ValueError("start_weight_version must be an integer")
        prefinalization_sealed_states = {
            PromptGroupStatus.READY_TO_FINALIZE,
            PromptGroupStatus.FINALIZING,
            PromptGroupStatus.FINALIZATION_UNKNOWN,
        }
        all_current_attempts_sealed = all(
            sibling.current_attempt.status == RolloutAttemptStatus.SEALED
            for sibling in siblings
        )
        if status is PromptGroupStatus.GENERATING and all_current_attempts_sealed:
            raise ValueError("generating group must retain an unfinished sibling")
        if status in prefinalization_sealed_states and not all_current_attempts_sealed:
            raise ValueError(
                f"group state {status.value!r} requires every sibling to be sealed"
            )

        return PromptGroupRecoveryRecord(
            group_id=group_id,
            admission_id=admission_id,
            prompt_id=prompt_id,
            prompt_ref=PromptRef(sample_id=sample_id, task_name=task_name),
            agent_name=agent_name,
            recovery_granularity=recovery_granularity,
            runtime_prompt_payload=None,
            expected_generations=expected_generations,
            target_step=target_step,
            start_weight_version=start_weight,
            siblings=siblings,
            phase=phase,
            status=status,
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


def _validate_batch_shortfall(value: object) -> dict[int, int]:
    """Return a defensive copy of per-step permanent rollout losses."""
    if not isinstance(value, dict):
        raise TypeError("rollout recovery batch_shortfall must be a dictionary")
    batch_shortfall: dict[int, int] = {}
    for step, count in value.items():
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(
                "rollout recovery batch_shortfall entries must contain "
                f"non-negative integer steps and counts, got {step!r}: {count!r}"
            )
        batch_shortfall[step] = count
    return batch_shortfall


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
    state["batch_shortfall"] = _validate_batch_shortfall(batch_shortfall)
    state["sampler_stamps_target_steps"] = sampler_stamps_target_steps
    return state


def parse_rollout_recovery_state(state: object) -> ParsedRolloutRecoveryState:
    """Validate and split a complete checkpoint sidecar by runtime owner."""
    if not isinstance(state, dict):
        raise TypeError(
            "rollout recovery sidecar must contain a dictionary, got "
            f"{type(state).__name__}"
        )
    _reject_unknown_fields(
        state,
        expected=_SIDECAR_STATE_FIELDS,
        context="rollout recovery sidecar",
    )
    schema_version = state.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in _SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS
    ):
        raise ValueError(
            "unsupported rollout recovery schema_version="
            f"{schema_version!r}; supported versions are "
            f"{sorted(_SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS)}"
        )
    groups = state.get("groups")
    if not isinstance(groups, list):
        raise TypeError("rollout recovery groups must be a list")

    raw_sampler_stamps = state.get("sampler_stamps_target_steps")
    if raw_sampler_stamps is not None and not isinstance(raw_sampler_stamps, bool):
        raise TypeError(
            "rollout recovery sampler_stamps_target_steps must be a boolean"
        )

    ledger_state: RolloutRecoveryState = {
        "schema_version": schema_version,
        "groups": groups,
    }
    return ParsedRolloutRecoveryState(
        ledger_state=ledger_state,
        batch_shortfall=_validate_batch_shortfall(state.get("batch_shortfall", {})),
        sampler_stamps_target_steps=raw_sampler_stamps,
    )
