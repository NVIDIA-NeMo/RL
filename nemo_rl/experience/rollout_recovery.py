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
router-replay payloads remain in TQ. The versioned ``state_dict`` boundary is
persisted beside the native TQ snapshot.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional, Self

from nemo_rl.data_plane import KVBatchMeta

if TYPE_CHECKING:
    from nemo_rl.data.interfaces import DatumSpec

ROLLOUT_RECOVERY_SCHEMA_VERSION = 2
ROLLOUT_RECOVERY_STATE_FILENAME = "rollout_recovery.pt"


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
    FINALIZED = "finalized"
    CLAIMED_FOR_TRAINING = "claimed_for_training"
    APPLIED_UNCHECKPOINTED = "applied_uncheckpointed"


class TrainStepStatus(StrEnum):
    """State of the one optimizer step the SingleController may have open."""

    OPEN = "open"
    APPLIED_UNCHECKPOINTED = "applied_uncheckpointed"


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


@dataclass
class RolloutAttemptRecord:
    """One physical attempt for a stable logical sibling."""

    attempt_uuid: uuid.UUID
    status: RolloutAttemptStatus
    receipt: Optional[dict[str, Any]] = None
    reward: Optional[float] = None
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
    prompt_id: str
    prompt_ref: PromptRef
    runtime_prompt_payload: Optional[DatumSpec]
    expected_generations: int
    target_step: Optional[int]
    start_weight_version: int
    siblings: list[RolloutSiblingRecord]
    status: PromptGroupStatus = PromptGroupStatus.GENERATING
    canonical_meta: Optional[KVBatchMeta] = None
    group_min_weight_version: Optional[int] = None
    group_max_weight_version: Optional[int] = None
    claimed_train_step: Optional[int] = None

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
class PromptGroupRecoverySummary:
    """Small immutable view used for restore planning at large group counts."""

    group_id: str
    prompt_ref: PromptRef
    target_step: Optional[int]
    start_weight_version: int
    status: PromptGroupStatus


@dataclass
class OpenTrainStepRecord:
    """Groups contributing to the current, not-yet-durable optimizer step."""

    train_step: int
    trainer_version: int
    expected_group_count: int
    group_ids: list[str] = field(default_factory=list)
    status: TrainStepStatus = TrainStepStatus.OPEN


def _new_attempt() -> RolloutAttemptRecord:
    return RolloutAttemptRecord(
        attempt_uuid=uuid.uuid4(),
        status=RolloutAttemptStatus.RESERVED,
    )


def _receipt_staging_keys(receipt: dict[str, Any]) -> list[str]:
    """Validate a sealed Gate receipt and return its ordered staging keys."""
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
    """In-memory source of truth for token-capture rollout ownership."""

    def __init__(self) -> None:
        self._groups: dict[str, PromptGroupRecoveryRecord] = {}
        self._open_train_step: Optional[OpenTrainStepRecord] = None

    @property
    def open_train_step(self) -> Optional[OpenTrainStepRecord]:
        return copy.deepcopy(self._open_train_step)

    def groups(self) -> list[PromptGroupRecoveryRecord]:
        return [self._copy_group(group) for group in self._groups.values()]

    def summaries(self) -> list[PromptGroupRecoverySummary]:
        """Return restore-planning fields without copying receipts or metadata."""
        return [
            PromptGroupRecoverySummary(
                group_id=record.group_id,
                prompt_ref=record.prompt_ref,
                target_step=record.target_step,
                start_weight_version=record.start_weight_version,
                status=record.status,
            )
            for record in self._groups.values()
        ]

    def bind_runtime_prompt(self, group_id: str, prompt_payload: DatumSpec) -> None:
        """Attach a dataset-reconstructed prompt without serializing it.

        Only unfinished generation needs the prompt body. Finalization and
        training consume durable TQ rows referenced by the ledger instead.
        """
        record = self._require_group(group_id)
        if record.status != PromptGroupStatus.GENERATING:
            raise ValueError(
                f"cannot bind a prompt to group {group_id!r} in "
                f"state {record.status.value!r}"
            )
        sample_id = str(prompt_payload.get("idx"))
        task_name = prompt_payload.get("task_name")
        if sample_id != record.prompt_ref.sample_id:
            raise ValueError(
                "reconstructed prompt does not match its durable sample reference: "
                f"sample={sample_id!r}, expected={record.prompt_ref.sample_id!r}"
            )
        if task_name != record.prompt_ref.task_name:
            raise ValueError(
                "reconstructed prompt task does not match its durable reference: "
                f"task={task_name!r}, expected={record.prompt_ref.task_name!r}"
            )
        record.runtime_prompt_payload = prompt_payload

    def prepare_for_restart(self) -> None:
        """Convert crash-interrupted attempts into retryable ledger state.

        A valid full-step checkpoint cannot contain an open optimizer step or
        an in-progress/unknown finalizer publication. Those states need the
        periodic-checkpoint protocol and are rejected rather than guessed at.
        """
        self.assert_full_step_checkpoint_safe()
        for record in self._groups.values():
            if record.status == PromptGroupStatus.GENERATING:
                self.abandon_unsealed(record.group_id)

    def assert_full_step_checkpoint_safe(self) -> None:
        """Reject states that require the periodic mid-step protocol."""
        if self._open_train_step is not None:
            raise RuntimeError(
                "rollout recovery contains an open optimizer step; restoring "
                "mid-step snapshots is not supported by full-step recovery"
            )
        unsupported = [
            record.group_id
            for record in self._groups.values()
            if record.status
            in {
                PromptGroupStatus.FINALIZING,
                PromptGroupStatus.FINALIZATION_UNKNOWN,
                PromptGroupStatus.CLAIMED_FOR_TRAINING,
                PromptGroupStatus.APPLIED_UNCHECKPOINTED,
            }
        ]
        if unsupported:
            raise RuntimeError(
                "rollout recovery contains checkpoint-unsafe group states: "
                f"groups={unsupported!r}"
            )

    def expected_staging_keys(self) -> set[str]:
        """Return every durable staging row still owned by the ledger."""
        return {
            staging_key
            for record in self._groups.values()
            for sibling in record.siblings
            for attempt in sibling.attempts[-1:]
            if attempt.status == RolloutAttemptStatus.SEALED
            for staging_key in attempt.staging_keys
        }

    def reserve_group(
        self,
        *,
        prompt_id: str,
        prompt_ref: PromptRef,
        prompt_payload: DatumSpec,
        expected_generations: int,
        target_step: Optional[int],
        start_weight_version: int,
        group_id: Optional[str] = None,
    ) -> PromptGroupRecoveryRecord:
        """Create one logical group and its first physical sibling attempts."""
        if not prompt_id:
            raise ValueError("prompt_id must not be empty")
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
            prompt_id=prompt_id,
            prompt_ref=prompt_ref,
            # Retain the immutable dataloader sample by reference instead of copying
            # a potentially 131k-token payload. This cache is never serialized and
            # is released as soon as canonical rows take over recovery ownership.
            runtime_prompt_payload=prompt_payload,
            expected_generations=expected_generations,
            target_step=target_step,
            start_weight_version=start_weight_version,
            siblings=siblings,
        )
        self._groups[group_id] = record
        return self._copy_group(record)

    def prepare_incomplete_retry(self, group_id: str) -> PromptGroupRecoveryRecord:
        """Mint fresh physical attempts only for siblings that are not sealed."""
        record = self._require_group(group_id)
        if record.status != PromptGroupStatus.GENERATING:
            raise ValueError(
                f"cannot retry group {group_id!r} from status {record.status.value!r}"
            )
        for sibling in record.siblings:
            attempt = sibling.current_attempt
            if attempt.status == RolloutAttemptStatus.SEALED:
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
        self, group_id: str, *, generation_indices: Optional[list[int]] = None
    ) -> None:
        """Move the selected current sibling attempts to dispatched."""
        record = self._require_group(group_id)
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
        group_id: str,
        *,
        generation_index: int,
        gate_rollout_id: str,
        receipt: dict[str, Any],
        reward: float,
    ) -> None:
        """Record one streamed sibling receipt as soon as the row arrives."""
        record = self._require_group(group_id)
        sibling = self._require_sibling(record, generation_index)
        attempt = sibling.current_attempt
        expected_gate_rollout_id = record.gate_rollout_id(generation_index)
        staging_keys = _receipt_staging_keys(receipt)
        if gate_rollout_id != expected_gate_rollout_id:
            raise ValueError(
                "streamed rollout identity mismatch: "
                f"result={gate_rollout_id!r}, expected={expected_gate_rollout_id!r}"
            )
        if receipt.get("rollout_id") != gate_rollout_id:
            raise ValueError(
                "receipt rollout identity mismatch: "
                f"receipt={receipt.get('rollout_id')!r}, expected={gate_rollout_id!r}"
            )
        if attempt.status == RolloutAttemptStatus.SEALED:
            if (
                attempt.receipt == receipt
                and attempt.reward == float(reward)
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
        attempt.staging_keys = staging_keys
        attempt.status = RolloutAttemptStatus.SEALED
        if all(
            item.current_attempt.status == RolloutAttemptStatus.SEALED
            for item in record.siblings
        ):
            record.status = PromptGroupStatus.READY_TO_FINALIZE

    def abandon_unsealed(self, group_id: str) -> None:
        """Abandon failed attempts without destroying reusable sealed receipts."""
        record = self._require_group(group_id)
        if record.status not in {
            PromptGroupStatus.GENERATING,
            PromptGroupStatus.READY_TO_FINALIZE,
        }:
            raise ValueError(
                f"cannot abandon group {group_id!r} from {record.status.value!r}"
            )
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
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[float]]:
        """Return physical IDs, canonical IDs, receipts and rewards in sibling order."""
        record = self._require_group(group_id)
        if record.status != PromptGroupStatus.READY_TO_FINALIZE:
            raise ValueError(
                f"group {group_id!r} is not ready to finalize: {record.status.value!r}"
            )
        receipts: list[dict[str, Any]] = []
        rewards: list[float] = []
        for sibling in record.siblings:
            attempt = sibling.current_attempt
            if (
                attempt.status != RolloutAttemptStatus.SEALED
                or attempt.receipt is None
                or attempt.reward is None
            ):
                raise ValueError(
                    "logical rollout "
                    f"{record.logical_rollout_id(sibling.generation_index)!r} "
                    "is not sealed"
                )
            receipts.append(copy.deepcopy(attempt.receipt))
            rewards.append(attempt.reward)
        return (
            record.gate_rollout_ids,
            record.logical_rollout_ids,
            receipts,
            rewards,
        )

    def mark_finalization_started(self, group_id: str) -> None:
        record = self._require_group(group_id)
        self._require_group_status(
            record,
            allowed={PromptGroupStatus.READY_TO_FINALIZE},
            transition="start finalization",
        )
        record.status = PromptGroupStatus.FINALIZING

    def mark_finalization_unknown(self, group_id: str) -> None:
        record = self._require_group(group_id)
        self._require_group_status(
            record,
            allowed={PromptGroupStatus.FINALIZING},
            transition="mark finalization unknown",
        )
        record.status = PromptGroupStatus.FINALIZATION_UNKNOWN

    def mark_group_finalized(
        self,
        group_id: str,
        *,
        meta: KVBatchMeta,
        group_min_weight_version: int,
        group_max_weight_version: int,
    ) -> None:
        """Transfer recovery ownership from staged receipts to canonical TQ rows."""
        record = self._require_group(group_id)
        self._require_group_status(
            record,
            allowed={PromptGroupStatus.FINALIZING},
            transition="finalize",
        )
        if list(meta.sample_ids) != record.logical_rollout_ids:
            raise ValueError(
                "finalized sample IDs do not match stable logical rollout IDs: "
                f"{meta.sample_ids!r} != {record.logical_rollout_ids!r}"
            )
        record.canonical_meta = copy.deepcopy(meta)
        record.group_min_weight_version = int(group_min_weight_version)
        record.group_max_weight_version = int(group_max_weight_version)
        record.runtime_prompt_payload = None
        record.status = PromptGroupStatus.FINALIZED

    def claim_groups_for_training(
        self,
        group_ids: list[str],
        *,
        train_step: int,
        trainer_version: int,
        expected_group_count: int,
    ) -> None:
        """Move finalized groups into the controller's one open optimizer step."""
        if not group_ids:
            raise ValueError("training claim must contain at least one group")
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("training claim contains duplicate group IDs")
        if self._open_train_step is None:
            self._open_train_step = OpenTrainStepRecord(
                train_step=train_step,
                trainer_version=trainer_version,
                expected_group_count=expected_group_count,
            )
        open_step = self._open_train_step
        if (
            open_step.train_step != train_step
            or open_step.trainer_version != trainer_version
            or open_step.expected_group_count != expected_group_count
            or open_step.status != TrainStepStatus.OPEN
        ):
            raise ValueError(
                "training claim does not match the existing open train step"
            )
        records = [self._require_group(group_id) for group_id in group_ids]
        for record in records:
            if record.status != PromptGroupStatus.FINALIZED:
                raise ValueError(
                    f"cannot claim group {record.group_id!r} from "
                    f"{record.status.value!r}"
                )
        if len(open_step.group_ids) + len(group_ids) > expected_group_count:
            raise ValueError("training claim exceeds the step's expected group count")
        for record in records:
            record.status = PromptGroupStatus.CLAIMED_FOR_TRAINING
            record.claimed_train_step = train_step
            open_step.group_ids.append(record.group_id)

    def mark_train_step_applied(self, train_step: int) -> None:
        """Record optimizer success while rows are still not checkpoint-covered."""
        open_step = self._require_open_train_step(train_step)
        if open_step.status != TrainStepStatus.OPEN:
            raise ValueError(
                f"train step {train_step} is already {open_step.status.value!r}"
            )
        if len(open_step.group_ids) != open_step.expected_group_count:
            raise ValueError(
                f"train step {train_step} has {len(open_step.group_ids)} claimed "
                f"groups; expected {open_step.expected_group_count}"
            )
        for group_id in open_step.group_ids:
            record = self._require_group(group_id)
            if record.status != PromptGroupStatus.CLAIMED_FOR_TRAINING:
                raise ValueError(
                    f"train step {train_step} owns group {group_id!r} in state "
                    f"{record.status.value!r}"
                )
        for group_id in open_step.group_ids:
            self._groups[group_id].status = PromptGroupStatus.APPLIED_UNCHECKPOINTED
        open_step.status = TrainStepStatus.APPLIED_UNCHECKPOINTED

    def release_applied_train_step(self, train_step: int) -> None:
        """Drop group metadata after the caller has cleared all owned TQ rows.

        The current controller clears immediately after optimizer success. A later
        persistence change will delay this call until a trainer checkpoint covers
        the applied update.
        """
        open_step = self._require_open_train_step(train_step)
        if open_step.status != TrainStepStatus.APPLIED_UNCHECKPOINTED:
            raise ValueError(
                f"cannot release train step {train_step} from "
                f"{open_step.status.value!r}"
            )
        for group_id in open_step.group_ids:
            del self._groups[group_id]
        self._open_train_step = None

    def rollback_open_train_step(self, train_step: int) -> None:
        """Return an uncheckpointed step's groups to finalized ownership."""
        open_step = self._require_open_train_step(train_step)
        for group_id in open_step.group_ids:
            record = self._require_group(group_id)
            if record.status not in {
                PromptGroupStatus.CLAIMED_FOR_TRAINING,
                PromptGroupStatus.APPLIED_UNCHECKPOINTED,
            }:
                raise ValueError(
                    f"cannot roll back group {group_id!r} from {record.status.value!r}"
                )
            record.status = PromptGroupStatus.FINALIZED
            record.claimed_train_step = None
        self._open_train_step = None

    def discard_group(self, group_id: str) -> None:
        """Drop a group only after its external TQ/Gate ownership is cleaned."""
        record = self._require_group(group_id)
        if record.claimed_train_step is not None:
            raise ValueError(f"cannot discard training-owned group {group_id!r}")
        del self._groups[group_id]

    def get_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        return self._copy_group(self._require_group(group_id))

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, group_id: object) -> bool:
        return isinstance(group_id, str) and group_id in self._groups

    def state_dict(self) -> dict[str, Any]:
        """Return the versioned metadata envelope used by later persistence."""
        groups = []
        for record in self._groups.values():
            groups.append(
                {
                    "group_id": record.group_id,
                    "prompt_id": record.prompt_id,
                    "prompt_ref": {
                        "sample_id": record.prompt_ref.sample_id,
                        "task_name": record.prompt_ref.task_name,
                    },
                    "expected_generations": record.expected_generations,
                    "target_step": record.target_step,
                    "start_weight_version": record.start_weight_version,
                    "status": record.status.value,
                    "canonical_meta": copy.deepcopy(record.canonical_meta),
                    "group_min_weight_version": record.group_min_weight_version,
                    "group_max_weight_version": record.group_max_weight_version,
                    "claimed_train_step": record.claimed_train_step,
                    "siblings": [
                        {
                            "generation_index": sibling.generation_index,
                            "attempts": [
                                {
                                    "attempt_uuid": attempt.attempt_uuid.bytes,
                                    "status": attempt.status.value,
                                    "receipt": copy.deepcopy(attempt.receipt),
                                    "reward": attempt.reward,
                                    "staging_keys": list(attempt.staging_keys),
                                }
                                for attempt in sibling.attempts
                            ],
                        }
                        for sibling in record.siblings
                    ],
                }
            )
        open_step = self._open_train_step
        return {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": groups,
            "open_train_step": (
                None
                if open_step is None
                else {
                    "train_step": open_step.train_step,
                    "trainer_version": open_step.trainer_version,
                    "expected_group_count": open_step.expected_group_count,
                    "group_ids": list(open_step.group_ids),
                    "status": open_step.status.value,
                }
            ),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Self:
        """Restore and validate a ledger metadata envelope."""
        if state.get("schema_version") != ROLLOUT_RECOVERY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported rollout-recovery schema version: "
                f"{state.get('schema_version')!r}"
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

        raw_open_step = state.get("open_train_step")
        if raw_open_step is not None:
            if not isinstance(raw_open_step, dict):
                raise ValueError("open_train_step must be a mapping or None")
            try:
                open_step = OpenTrainStepRecord(
                    train_step=int(raw_open_step["train_step"]),
                    trainer_version=int(raw_open_step["trainer_version"]),
                    expected_group_count=int(raw_open_step["expected_group_count"]),
                    group_ids=list(raw_open_step["group_ids"]),
                    status=TrainStepStatus(raw_open_step["status"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid open_train_step state") from error
            if not all(isinstance(group_id, str) for group_id in open_step.group_ids):
                raise ValueError("open_train_step group_ids must be strings")
            if len(open_step.group_ids) != len(set(open_step.group_ids)):
                raise ValueError("open_train_step contains duplicate group IDs")
            if len(open_step.group_ids) > open_step.expected_group_count:
                raise ValueError("open_train_step exceeds expected_group_count")
            expected_group_status = (
                PromptGroupStatus.CLAIMED_FOR_TRAINING
                if open_step.status == TrainStepStatus.OPEN
                else PromptGroupStatus.APPLIED_UNCHECKPOINTED
            )
            for group_id in open_step.group_ids:
                record = ledger._require_group(group_id)
                if (
                    record.claimed_train_step != open_step.train_step
                    or record.status != expected_group_status
                ):
                    raise ValueError(
                        f"open_train_step ownership mismatch for group {group_id!r}"
                    )
            claimed_group_ids = {
                record.group_id
                for record in ledger._groups.values()
                if record.claimed_train_step is not None
            }
            if claimed_group_ids != set(open_step.group_ids):
                raise ValueError(
                    "open_train_step does not list every training-owned group"
                )
            ledger._open_train_step = open_step
        elif any(
            record.claimed_train_step is not None for record in ledger._groups.values()
        ):
            raise ValueError("claimed groups require an open_train_step record")
        return ledger

    @classmethod
    def _group_from_state(
        cls,
        raw_group: Any,
        *,
        seen_attempt_uuids: set[uuid.UUID],
    ) -> PromptGroupRecoveryRecord:
        if not isinstance(raw_group, dict):
            raise ValueError("rollout-recovery group must be a mapping")
        group_id = raw_group.get("group_id")
        prompt_id = raw_group.get("prompt_id")
        expected_generations = raw_group.get("expected_generations")
        siblings_state = raw_group.get("siblings")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("prompt_id must be a non-empty string")
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

        siblings: list[RolloutSiblingRecord] = []
        for generation_index, sibling_state in enumerate(siblings_state):
            if not isinstance(sibling_state, dict):
                raise ValueError("rollout-recovery sibling must be a mapping")
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
                staging_keys = attempt_state.get("staging_keys")
                if not isinstance(staging_keys, list) or not all(
                    isinstance(key, str) for key in staging_keys
                ):
                    raise ValueError("staging_keys must be a list of strings")
                if attempt_status == RolloutAttemptStatus.SEALED:
                    if not isinstance(receipt, dict) or not isinstance(
                        reward, (int, float)
                    ):
                        raise ValueError("sealed attempts require receipt and reward")
                    if receipt.get("rollout_id") != gate_id:
                        raise ValueError("sealed receipt identity mismatch")
                    if _receipt_staging_keys(receipt) != staging_keys:
                        raise ValueError("sealed receipt staging manifest mismatch")
                elif receipt is not None or reward is not None or staging_keys:
                    raise ValueError("only sealed attempts may retain receipt data")
                attempts.append(
                    RolloutAttemptRecord(
                        attempt_uuid=attempt_uuid,
                        status=attempt_status,
                        receipt=copy.deepcopy(receipt),
                        reward=float(reward) if reward is not None else None,
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
        sample_id = raw_prompt_ref.get("sample_id")
        task_name = raw_prompt_ref.get("task_name")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("prompt_ref sample_id must be a non-empty string")
        if task_name is not None and not isinstance(task_name, str):
            raise ValueError("prompt_ref task_name must be a string or None")
        if sample_id != prompt_id:
            raise ValueError("prompt_ref sample_id must match prompt_id")
        canonical_meta = raw_group.get("canonical_meta")
        if canonical_meta is not None and not isinstance(canonical_meta, KVBatchMeta):
            raise ValueError("canonical_meta must be KVBatchMeta or None")
        target_step = raw_group.get("target_step")
        claimed_train_step = raw_group.get("claimed_train_step")
        if target_step is not None and not isinstance(target_step, int):
            raise ValueError("target_step must be an integer or None")
        if claimed_train_step is not None and not isinstance(claimed_train_step, int):
            raise ValueError("claimed_train_step must be an integer or None")
        start_weight = raw_group.get("start_weight_version")
        if not isinstance(start_weight, int):
            raise ValueError("start_weight_version must be an integer")
        min_weight = raw_group.get("group_min_weight_version")
        max_weight = raw_group.get("group_max_weight_version")
        if min_weight is not None and not isinstance(min_weight, int):
            raise ValueError("group_min_weight_version must be an integer or None")
        if max_weight is not None and not isinstance(max_weight, int):
            raise ValueError("group_max_weight_version must be an integer or None")

        finalized_states = {
            PromptGroupStatus.FINALIZED,
            PromptGroupStatus.CLAIMED_FOR_TRAINING,
            PromptGroupStatus.APPLIED_UNCHECKPOINTED,
        }
        prefinalization_sealed_states = {
            PromptGroupStatus.READY_TO_FINALIZE,
            PromptGroupStatus.FINALIZING,
            PromptGroupStatus.FINALIZATION_UNKNOWN,
        }
        all_current_attempts_sealed = all(
            sibling.current_attempt.status == RolloutAttemptStatus.SEALED
            for sibling in siblings
        )
        if status == PromptGroupStatus.GENERATING and all_current_attempts_sealed:
            raise ValueError("generating group must retain an unfinished sibling")
        if status in prefinalization_sealed_states and not all_current_attempts_sealed:
            raise ValueError(
                f"group state {status.value!r} requires every sibling to be sealed"
            )
        if status in finalized_states:
            if not all_current_attempts_sealed:
                raise ValueError("finalized group state requires sealed siblings")
            if canonical_meta is None or min_weight is None or max_weight is None:
                raise ValueError("finalized group state requires canonical metadata")
            if list(canonical_meta.sample_ids) != [
                f"{group_id}_g{s.generation_index}" for s in siblings
            ]:
                raise ValueError("canonical sample IDs do not match logical lineage")
        elif canonical_meta is not None:
            raise ValueError("unfinished group cannot contain canonical metadata")
        claimed_states = {
            PromptGroupStatus.CLAIMED_FOR_TRAINING,
            PromptGroupStatus.APPLIED_UNCHECKPOINTED,
        }
        if (status in claimed_states) != (claimed_train_step is not None):
            raise ValueError(
                "claimed_train_step must be present exactly for training-owned groups"
            )

        return PromptGroupRecoveryRecord(
            group_id=group_id,
            prompt_id=prompt_id,
            prompt_ref=PromptRef(sample_id=sample_id, task_name=task_name),
            runtime_prompt_payload=None,
            expected_generations=expected_generations,
            target_step=target_step,
            start_weight_version=start_weight,
            siblings=siblings,
            status=status,
            canonical_meta=copy.deepcopy(canonical_meta),
            group_min_weight_version=min_weight,
            group_max_weight_version=max_weight,
            claimed_train_step=claimed_train_step,
        )

    def _require_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        try:
            return self._groups[group_id]
        except KeyError as error:
            raise ValueError(f"unknown recovery group_id={group_id!r}") from error

    @staticmethod
    def _copy_group(record: PromptGroupRecoveryRecord) -> PromptGroupRecoveryRecord:
        """Copy mutable lineage metadata without duplicating the prompt payload."""
        return PromptGroupRecoveryRecord(
            group_id=record.group_id,
            prompt_id=record.prompt_id,
            prompt_ref=record.prompt_ref,
            runtime_prompt_payload=record.runtime_prompt_payload,
            expected_generations=record.expected_generations,
            target_step=record.target_step,
            start_weight_version=record.start_weight_version,
            siblings=copy.deepcopy(record.siblings),
            status=record.status,
            canonical_meta=copy.deepcopy(record.canonical_meta),
            group_min_weight_version=record.group_min_weight_version,
            group_max_weight_version=record.group_max_weight_version,
            claimed_train_step=record.claimed_train_step,
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

    def _require_open_train_step(self, train_step: int) -> OpenTrainStepRecord:
        open_step = self._open_train_step
        if open_step is None or open_step.train_step != train_step:
            raise ValueError(f"train step {train_step} is not open")
        return open_step
