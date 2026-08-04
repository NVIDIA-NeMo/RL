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

"""In-memory identity and lineage state for recoverable prompt groups."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self, TypedDict, cast

if TYPE_CHECKING:
    from nemo_rl.data.interfaces import DatumSpec

ROLLOUT_RECOVERY_SCHEMA_VERSION = 2
ROLLOUT_RECOVERY_STATE_FILENAME = "rollout_recovery.pt"


class RolloutAttemptStatus(StrEnum):
    """Lifecycle state of one physical execution attempt."""

    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    SEALED = "sealed"
    FINALIZED = "finalized"
    FAILED = "failed"
    ABANDONED = "abandoned"


class RolloutAttemptState(TypedDict):
    """Serializable state for one execution attempt."""

    attempt_id: str
    gate_rollout_id: str
    status: str
    receipt: dict[str, Any] | None
    reward: float | None
    staging_keys: list[str]


class RolloutSiblingState(TypedDict):
    """Serializable state for one logical generation slot."""

    generation_index: int
    logical_rollout_id: str
    attempts: list[RolloutAttemptState]


class PromptGroupRecoveryState(TypedDict):
    """Serializable state for one prompt group."""

    group_id: str
    prompt_id: str
    prompt_payload: DatumSpec
    expected_generations: int
    target_step: int | None
    start_weight_version: int
    siblings: list[RolloutSiblingState]


class RolloutRecoveryState(TypedDict):
    """Versioned envelope for the complete in-memory ledger."""

    schema_version: int
    groups: list[PromptGroupRecoveryState]


@dataclass
class RolloutAttemptRecord:
    """One physical attempt for a stable logical sibling."""

    attempt_id: str
    gate_rollout_id: str
    status: RolloutAttemptStatus
    receipt: dict[str, Any] | None = None
    reward: float | None = None
    staging_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutSiblingRecord:
    """One stable generation slot and all of its execution attempts."""

    generation_index: int
    logical_rollout_id: str
    attempts: list[RolloutAttemptRecord]

    @property
    def current_attempt(self) -> RolloutAttemptRecord:
        """Return the newest execution attempt."""
        if not self.attempts:
            raise RuntimeError(
                f"logical rollout {self.logical_rollout_id!r} has no attempts"
            )
        return self.attempts[-1]


@dataclass
class PromptGroupRecoveryRecord:
    """Prompt identity and sibling lineage retained across dispatch."""

    group_id: str
    prompt_id: str
    prompt_payload: DatumSpec
    expected_generations: int
    target_step: int | None
    start_weight_version: int
    siblings: list[RolloutSiblingRecord]

    @property
    def logical_rollout_ids(self) -> list[str]:
        """Return stable canonical sample IDs in generation order."""
        return [sibling.logical_rollout_id for sibling in self.siblings]

    @property
    def gate_rollout_ids(self) -> list[str]:
        """Return current physical gate IDs in generation order."""
        return [sibling.current_attempt.gate_rollout_id for sibling in self.siblings]


def _new_attempt(logical_rollout_id: str) -> RolloutAttemptRecord:
    attempt_id = str(uuid.uuid4())
    return RolloutAttemptRecord(
        attempt_id=attempt_id,
        gate_rollout_id=f"{logical_rollout_id}_a{attempt_id}",
        status=RolloutAttemptStatus.RESERVED,
    )


def _receipt_staging_keys(receipt: dict[str, Any]) -> list[str]:
    """Validate a sealed receipt and return its ordered staging keys."""
    manifest = receipt.get("manifest")
    if not isinstance(manifest, list):
        raise ValueError("sealed rollout receipt must contain a manifest list")
    staging_keys: list[str] = []
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("staging_key"), str):
            raise ValueError(
                "sealed rollout receipt manifest entries must contain "
                "string staging_key values"
            )
        staging_keys.append(entry["staging_key"])
    return staging_keys


class RolloutRecoveryLedger:
    """Controller-owned prompt-to-attempt lineage for token-capture rollouts.

    ``state_dict`` and ``from_state_dict`` define the versioned sidecar stored
    alongside the native TQ snapshot. The sidecar contains lineage and
    receipts only; token payloads remain in TQ's staging partition.
    """

    def __init__(self) -> None:
        self._groups: dict[str, PromptGroupRecoveryRecord] = {}

    def reserve_group(
        self,
        *,
        prompt_id: str,
        prompt_payload: DatumSpec,
        expected_generations: int,
        target_step: int | None,
        start_weight_version: int,
        group_id: str | None = None,
    ) -> PromptGroupRecoveryRecord:
        """Create one logical group and its first physical attempts."""
        if not prompt_id:
            raise ValueError("prompt_id must not be empty")
        if expected_generations < 1:
            raise ValueError("expected_generations must be at least one")
        group_id = group_id or str(uuid.uuid4())
        if group_id in self._groups:
            raise ValueError(f"duplicate recovery group_id={group_id!r}")

        siblings = []
        for generation_index in range(expected_generations):
            logical_rollout_id = f"{group_id}_g{generation_index}"
            siblings.append(
                RolloutSiblingRecord(
                    generation_index=generation_index,
                    logical_rollout_id=logical_rollout_id,
                    attempts=[_new_attempt(logical_rollout_id)],
                )
            )
        record = PromptGroupRecoveryRecord(
            group_id=group_id,
            prompt_id=prompt_id,
            prompt_payload=copy.deepcopy(prompt_payload),
            expected_generations=expected_generations,
            target_step=target_step,
            start_weight_version=start_weight_version,
            siblings=siblings,
        )
        self._groups[group_id] = record
        return copy.deepcopy(record)

    def mark_group_dispatched(self, group_id: str) -> None:
        """Move every current attempt from reserved to dispatched."""
        record = self._require_group(group_id)
        self.mark_siblings_dispatched(
            group_id,
            generation_indices=[
                sibling.generation_index for sibling in record.siblings
            ],
        )

    def mark_siblings_dispatched(
        self, group_id: str, *, generation_indices: list[int]
    ) -> None:
        """Move selected current attempts from reserved to dispatched."""
        record = self._require_group(group_id)
        if not generation_indices or len(set(generation_indices)) != len(
            generation_indices
        ):
            raise ValueError("generation_indices must be non-empty and unique")
        if any(
            index < 0 or index >= len(record.siblings) for index in generation_indices
        ):
            raise ValueError(
                f"generation_indices are outside recovery group {group_id!r}"
            )
        attempts = [
            record.siblings[generation_index].current_attempt
            for generation_index in generation_indices
        ]
        self._require_statuses(
            attempts,
            allowed={RolloutAttemptStatus.RESERVED},
            transition="dispatch",
        )
        for attempt in attempts:
            attempt.status = RolloutAttemptStatus.DISPATCHED

    def mark_group_finalized(self, group_id: str) -> None:
        """Mark all logical siblings canonical after the TQ commit succeeds."""
        record = self._require_group(group_id)
        attempts = [sibling.current_attempt for sibling in record.siblings]
        self._require_statuses(
            attempts,
            allowed={RolloutAttemptStatus.SEALED},
            transition="finalize",
        )
        for attempt in attempts:
            attempt.status = RolloutAttemptStatus.FINALIZED

    def mark_sibling_sealed(
        self,
        group_id: str,
        *,
        generation_index: int,
        gate_rollout_id: str,
        receipt: dict[str, Any],
        reward: float,
    ) -> None:
        """Record one streamed sibling receipt before the group completes."""
        record = self._require_group(group_id)
        if not 0 <= generation_index < len(record.siblings):
            raise ValueError(
                f"generation_index={generation_index} is outside group {group_id!r}"
            )
        sibling = record.siblings[generation_index]
        attempt = sibling.current_attempt
        if attempt.status != RolloutAttemptStatus.DISPATCHED:
            raise ValueError(
                f"cannot seal logical rollout {sibling.logical_rollout_id!r} "
                f"from status {attempt.status.value!r}"
            )
        if gate_rollout_id != attempt.gate_rollout_id:
            raise ValueError(
                "streamed rollout identity mismatch: "
                f"result={gate_rollout_id!r}, expected={attempt.gate_rollout_id!r}"
            )
        if receipt.get("rollout_id") != gate_rollout_id:
            raise ValueError(
                "receipt rollout identity mismatch: "
                f"receipt={receipt.get('rollout_id')!r}, "
                f"expected={gate_rollout_id!r}"
            )
        staging_keys = _receipt_staging_keys(receipt)

        attempt.receipt = copy.deepcopy(receipt)
        attempt.reward = float(reward)
        attempt.staging_keys = staging_keys
        attempt.status = RolloutAttemptStatus.SEALED

    def release_finalized_group(self, group_id: str) -> None:
        """Drop a group after canonical replay metadata owns its recovery."""
        record = self._require_group(group_id)
        if any(
            sibling.current_attempt.status != RolloutAttemptStatus.FINALIZED
            for sibling in record.siblings
        ):
            raise ValueError(
                f"cannot release unfinished recovery group_id={group_id!r}"
            )
        del self._groups[group_id]

    def abandon_group(self, group_id: str) -> None:
        """Abandon unfinished attempts while preserving sealed siblings."""
        record = self._require_group(group_id)
        for sibling in record.siblings:
            attempt = sibling.current_attempt
            if attempt.status in {
                RolloutAttemptStatus.SEALED,
                RolloutAttemptStatus.FINALIZED,
            }:
                continue
            attempt.status = RolloutAttemptStatus.ABANDONED

    def discard_group(self, group_id: str) -> None:
        """Drop a group that policy deliberately rejected after finalization."""
        self._require_group(group_id)
        del self._groups[group_id]

    def retry_sibling(
        self, group_id: str, *, generation_index: int
    ) -> RolloutAttemptRecord:
        """Create a new attempt while preserving the sibling's logical ID."""
        record = self._require_group(group_id)
        if not 0 <= generation_index < len(record.siblings):
            raise ValueError(
                f"generation_index={generation_index} is outside group {group_id!r}"
            )
        sibling = record.siblings[generation_index]
        if sibling.current_attempt.status not in {
            RolloutAttemptStatus.ABANDONED,
            RolloutAttemptStatus.FAILED,
        }:
            raise ValueError(
                f"cannot retry logical rollout {sibling.logical_rollout_id!r} "
                f"from status {sibling.current_attempt.status.value!r}"
            )
        attempt = _new_attempt(sibling.logical_rollout_id)
        sibling.attempts.append(attempt)
        return copy.deepcopy(attempt)

    def get_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        """Return a defensive copy of one group."""
        return copy.deepcopy(self._require_group(group_id))

    def groups(self) -> list[PromptGroupRecoveryRecord]:
        """Return defensive copies of all groups in checkpoint order."""
        return copy.deepcopy(list(self._groups.values()))

    def discard_canonical_groups(self, group_ids: set[str]) -> int:
        """Drop groups already represented by canonical replay metadata."""
        discarded = 0
        for group_id in group_ids:
            if group_id in self._groups:
                del self._groups[group_id]
                discarded += 1
        return discarded

    def prepare_for_restart(self) -> None:
        """Mark physical attempts that cannot survive a process restart abandoned."""
        for record in self._groups.values():
            for sibling in record.siblings:
                attempt = sibling.current_attempt
                if attempt.status in {
                    RolloutAttemptStatus.RESERVED,
                    RolloutAttemptStatus.DISPATCHED,
                }:
                    attempt.status = RolloutAttemptStatus.ABANDONED

    def expected_staging_keys(self) -> set[str]:
        """Return staging rows referenced by reusable sealed receipts."""
        return {
            staging_key
            for record in self._groups.values()
            for sibling in record.siblings
            if sibling.current_attempt.status == RolloutAttemptStatus.SEALED
            for staging_key in sibling.current_attempt.staging_keys
        }

    def retryable_generation_indices(self, group_id: str) -> list[int]:
        """Return logical siblings that require a new physical attempt."""
        record = self._require_group(group_id)
        return [
            sibling.generation_index
            for sibling in record.siblings
            if sibling.current_attempt.status
            in {
                RolloutAttemptStatus.ABANDONED,
                RolloutAttemptStatus.FAILED,
            }
        ]

    def __len__(self) -> int:
        """Return the number of unfinished or retryable prompt groups."""
        return len(self._groups)

    def state_dict(self) -> RolloutRecoveryState:
        """Return a versioned, tensor-compatible checkpoint envelope."""
        groups: list[PromptGroupRecoveryState] = []
        for record in self._groups.values():
            groups.append(
                {
                    "group_id": record.group_id,
                    "prompt_id": record.prompt_id,
                    "prompt_payload": copy.deepcopy(record.prompt_payload),
                    "expected_generations": record.expected_generations,
                    "target_step": record.target_step,
                    "start_weight_version": record.start_weight_version,
                    "siblings": [
                        {
                            "generation_index": sibling.generation_index,
                            "logical_rollout_id": sibling.logical_rollout_id,
                            "attempts": [
                                {
                                    "attempt_id": attempt.attempt_id,
                                    "gate_rollout_id": attempt.gate_rollout_id,
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
        return {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": groups,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Self:
        """Validate and restore a ledger checkpoint into a fresh instance."""
        if state.get("schema_version") != ROLLOUT_RECOVERY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported rollout-recovery schema version: "
                f"{state.get('schema_version')!r}"
            )
        raw_groups = state.get("groups")
        if not isinstance(raw_groups, list):
            raise ValueError("rollout-recovery state must contain a groups list")

        ledger = cls()
        seen_logical_ids: set[str] = set()
        seen_attempt_ids: set[str] = set()
        seen_gate_ids: set[str] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                raise ValueError("rollout-recovery group must be a mapping")
            group_id = raw_group.get("group_id")
            prompt_id = raw_group.get("prompt_id")
            expected_generations = raw_group.get("expected_generations")
            siblings_state = raw_group.get("siblings")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError("rollout-recovery group_id must be a non-empty string")
            if group_id in ledger._groups:
                raise ValueError(f"duplicate recovery group_id={group_id!r}")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(
                    "rollout-recovery prompt_id must be a non-empty string"
                )
            if not isinstance(expected_generations, int) or expected_generations < 1:
                raise ValueError(
                    "rollout-recovery expected_generations must be a positive integer"
                )
            if (
                not isinstance(siblings_state, list)
                or len(siblings_state) != expected_generations
            ):
                raise ValueError(
                    f"recovery group {group_id!r} has {len(siblings_state) if isinstance(siblings_state, list) else 'invalid'} "
                    f"siblings; expected {expected_generations}"
                )

            siblings: list[RolloutSiblingRecord] = []
            for generation_index, sibling_state in enumerate(siblings_state):
                if not isinstance(sibling_state, dict):
                    raise ValueError("rollout-recovery sibling must be a mapping")
                if sibling_state.get("generation_index") != generation_index:
                    raise ValueError(
                        f"recovery group {group_id!r} has non-contiguous generation indices"
                    )
                logical_rollout_id = sibling_state.get("logical_rollout_id")
                expected_logical_id = f"{group_id}_g{generation_index}"
                if (
                    not isinstance(logical_rollout_id, str)
                    or logical_rollout_id != expected_logical_id
                ):
                    raise ValueError(
                        f"logical rollout ID mismatch: {logical_rollout_id!r} != "
                        f"{expected_logical_id!r}"
                    )
                if logical_rollout_id in seen_logical_ids:
                    raise ValueError(
                        f"duplicate logical_rollout_id={logical_rollout_id!r}"
                    )
                seen_logical_ids.add(logical_rollout_id)

                attempts_state = sibling_state.get("attempts")
                if not isinstance(attempts_state, list) or not attempts_state:
                    raise ValueError(
                        f"logical rollout {logical_rollout_id!r} has no attempts"
                    )
                attempts: list[RolloutAttemptRecord] = []
                for attempt_state in attempts_state:
                    if not isinstance(attempt_state, dict):
                        raise ValueError("rollout-recovery attempt must be a mapping")
                    attempt_id = attempt_state.get("attempt_id")
                    gate_rollout_id = attempt_state.get("gate_rollout_id")
                    if not isinstance(attempt_id, str) or not attempt_id:
                        raise ValueError("attempt_id must be a non-empty string")
                    expected_gate_id = f"{logical_rollout_id}_a{attempt_id}"
                    if (
                        not isinstance(gate_rollout_id, str)
                        or gate_rollout_id != expected_gate_id
                    ):
                        raise ValueError(
                            f"gate rollout ID mismatch: {gate_rollout_id!r} != "
                            f"{expected_gate_id!r}"
                        )
                    if attempt_id in seen_attempt_ids:
                        raise ValueError(f"duplicate attempt_id={attempt_id!r}")
                    if gate_rollout_id in seen_gate_ids:
                        raise ValueError(
                            f"duplicate gate_rollout_id={gate_rollout_id!r}"
                        )
                    seen_attempt_ids.add(attempt_id)
                    seen_gate_ids.add(gate_rollout_id)
                    try:
                        status = RolloutAttemptStatus(attempt_state.get("status"))
                    except ValueError as error:
                        raise ValueError(
                            f"invalid rollout attempt status={attempt_state.get('status')!r}"
                        ) from error
                    receipt = attempt_state.get("receipt")
                    reward = attempt_state.get("reward")
                    staging_keys = attempt_state.get("staging_keys")
                    if receipt is not None and not isinstance(receipt, dict):
                        raise ValueError("rollout receipt must be a mapping or None")
                    if reward is not None and not isinstance(reward, (int, float)):
                        raise ValueError("rollout reward must be numeric or None")
                    if not isinstance(staging_keys, list) or not all(
                        isinstance(key, str) for key in staging_keys
                    ):
                        raise ValueError("staging_keys must be a list of strings")
                    if status in {
                        RolloutAttemptStatus.SEALED,
                        RolloutAttemptStatus.FINALIZED,
                    }:
                        if receipt is None or reward is None:
                            raise ValueError(
                                f"{status.value} rollout attempt must retain its "
                                "receipt and reward"
                            )
                        if receipt.get("rollout_id") != gate_rollout_id:
                            raise ValueError(
                                "restored receipt rollout identity does not match "
                                "its gate_rollout_id"
                            )
                        receipt_staging_keys = _receipt_staging_keys(receipt)
                        if staging_keys != receipt_staging_keys:
                            raise ValueError(
                                "restored staging_keys do not match the receipt "
                                "manifest"
                            )
                    attempts.append(
                        RolloutAttemptRecord(
                            attempt_id=attempt_id,
                            gate_rollout_id=gate_rollout_id,
                            status=status,
                            receipt=copy.deepcopy(receipt),
                            reward=float(reward) if reward is not None else None,
                            staging_keys=list(staging_keys),
                        )
                    )
                siblings.append(
                    RolloutSiblingRecord(
                        generation_index=generation_index,
                        logical_rollout_id=logical_rollout_id,
                        attempts=attempts,
                    )
                )

            target_step = raw_group.get("target_step")
            start_weight_version = raw_group.get("start_weight_version")
            if target_step is not None and not isinstance(target_step, int):
                raise ValueError("target_step must be an integer or None")
            if not isinstance(start_weight_version, int):
                raise ValueError("start_weight_version must be an integer")
            prompt_payload = raw_group.get("prompt_payload")
            if not isinstance(prompt_payload, dict):
                raise ValueError("prompt_payload must be a mapping")
            ledger._groups[group_id] = PromptGroupRecoveryRecord(
                group_id=group_id,
                prompt_id=prompt_id,
                prompt_payload=copy.deepcopy(cast("DatumSpec", prompt_payload)),
                expected_generations=expected_generations,
                target_step=target_step,
                start_weight_version=start_weight_version,
                siblings=siblings,
            )
        return ledger

    def _require_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        try:
            return self._groups[group_id]
        except KeyError as error:
            raise ValueError(f"unknown recovery group_id={group_id!r}") from error

    @staticmethod
    def _require_statuses(
        attempts: list[RolloutAttemptRecord],
        *,
        allowed: set[RolloutAttemptStatus],
        transition: str,
    ) -> None:
        invalid = [
            attempt.gate_rollout_id
            for attempt in attempts
            if attempt.status not in allowed
        ]
        if invalid:
            raise ValueError(
                f"cannot {transition} rollout attempts in their current states: {invalid}"
            )
