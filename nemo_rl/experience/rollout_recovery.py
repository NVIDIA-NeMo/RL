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

"""Versioned ownership state for unfinished SingleController prompt groups."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

if TYPE_CHECKING:
    from nemo_rl.data.interfaces import DatumSpec

ROLLOUT_RECOVERY_SCHEMA_VERSION = 1
ROLLOUT_RECOVERY_STATE_FILENAME = "rollout_recovery.pt"


class PromptGroupPhase(StrEnum):
    """Durable admission phase for an unfinished prompt group."""

    RESERVED = "reserved"
    ADMITTED = "admitted"


class PromptRefState(TypedDict):
    """Serializable locator for rebuilding one prompt from the dataset."""

    sample_id: str
    task_name: str | None


class PromptGroupRecoveryState(TypedDict):
    """Serializable ownership state for one unfinished prompt group."""

    group_id: str
    admission_id: str
    prompt_id: str
    prompt_ref: PromptRefState
    expected_generations: int
    target_step: int | None
    start_weight_version: int
    phase: str


class RolloutRecoveryState(TypedDict):
    """Versioned checkpoint sidecar for unfinished prompt groups."""

    schema_version: int
    groups: list[PromptGroupRecoveryState]
    batch_shortfall: NotRequired[dict[int, int]]
    sampler_stamps_target_steps: NotRequired[bool]


@dataclass(frozen=True)
class PromptRef:
    """Stable dataset identity for rebuilding one prompt."""

    sample_id: str
    task_name: str | None


@dataclass(frozen=True)
class PromptGroupRecoveryRecord:
    """In-memory ownership record for one prompt group."""

    group_id: str
    admission_id: str
    prompt_id: str
    prompt_ref: PromptRef
    runtime_prompt_payload: DatumSpec | None
    expected_generations: int
    target_step: int | None
    start_weight_version: int
    phase: PromptGroupPhase

    @property
    def prompt_payload(self) -> DatumSpec:
        """Return the rehydrated prompt required for rollout redispatch."""
        if self.runtime_prompt_payload is None:
            raise RuntimeError(
                f"recovery group {self.group_id!r} has not rehydrated prompt "
                f"sample_id={self.prompt_ref.sample_id!r}"
            )
        return self.runtime_prompt_payload


def _require_int(value: Any, *, field: str, minimum: int) -> int:
    """Validate one integer field without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def _prompt_task_name(prompt_payload: DatumSpec) -> str | None:
    task_name = prompt_payload.get("task_name")
    if task_name is not None and not isinstance(task_name, str):
        raise TypeError(
            "prompt_payload.task_name must be a string or None, got "
            f"{type(task_name).__name__}"
        )
    return task_name


def _validate_prompt_identity(
    prompt_ref: PromptRef,
    prompt_payload: DatumSpec,
    *,
    group_id: str,
) -> None:
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
    task_name = _prompt_task_name(prompt_payload)
    if task_name != prompt_ref.task_name:
        raise ValueError(
            f"recovery group {group_id!r} resolved task_name={task_name!r}; "
            f"expected {prompt_ref.task_name!r}"
        )


class RolloutRecoveryLedger:
    """Own prompts after dataloader advance and before canonical TQ commit."""

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
        admitted: bool,
        group_id: str | None = None,
        admission_id: str | None = None,
    ) -> PromptGroupRecoveryRecord:
        """Record ownership before the prompt can disappear from the dataloader.

        Args:
            prompt_id: Dataset-level prompt identity used for diagnostics.
            prompt_payload: Runtime prompt used for whole-group regeneration. Only
                its stable dataset reference is checkpointed.
            expected_generations: Number of GRPO siblings in the prompt group.
            target_step: Original gated training step, when the sampler stamps one.
            start_weight_version: Policy version visible at reservation time.
            admitted: Whether sampler admission already completed. This is explicit
                because ``target_step=None`` is also valid for admitted ungated groups.
            group_id: Stable logical and canonical TQ group ID. Generated when absent.
            admission_id: Stable identity shared by every prompt in one sampler
                admission. Defaults to ``group_id`` for single-prompt direct callers.

        Returns:
            A defensive copy of the new record.
        """
        if not prompt_id:
            raise ValueError("prompt_id must not be empty")
        sample_id = prompt_payload.get("idx")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise ValueError("prompt_payload must contain an integer idx")
        if prompt_id != str(sample_id):
            raise ValueError(
                f"prompt_id={prompt_id!r} does not match prompt_payload idx={sample_id!r}"
            )
        _require_int(
            expected_generations,
            field="expected_generations",
            minimum=1,
        )
        _require_int(
            start_weight_version,
            field="start_weight_version",
            minimum=0,
        )
        if target_step is not None:
            _require_int(target_step, field="target_step", minimum=0)
        group_id = group_id or str(uuid.uuid4())
        if not group_id:
            raise ValueError("group_id must not be empty")
        if group_id in self._groups:
            raise ValueError(f"duplicate recovery group_id={group_id!r}")
        admission_id = admission_id or group_id
        if not admission_id:
            raise ValueError("admission_id must not be empty")

        record = PromptGroupRecoveryRecord(
            group_id=group_id,
            admission_id=admission_id,
            prompt_id=prompt_id,
            # The rollout path treats the dataloader sample as immutable and builds
            # mutable environment inputs from copies. Retaining that sample by
            # reference avoids cloning a potentially very long prompt on every
            # dispatch; state_dict() persists only its dataset locator.
            prompt_ref=PromptRef(
                sample_id=prompt_id,
                task_name=_prompt_task_name(prompt_payload),
            ),
            runtime_prompt_payload=prompt_payload,
            expected_generations=expected_generations,
            target_step=target_step,
            start_weight_version=start_weight_version,
            phase=(
                PromptGroupPhase.ADMITTED if admitted else PromptGroupPhase.RESERVED
            ),
        )
        self._groups[group_id] = record
        return copy.copy(record)

    def mark_group_admitted(
        self,
        group_id: str,
        *,
        target_step: int | None,
        start_weight_version: int,
    ) -> None:
        """Attach the sampler result to a previously reserved prompt group."""
        record = self._require_group(group_id)
        if record.phase is not PromptGroupPhase.RESERVED:
            raise ValueError(
                f"recovery group {group_id!r} is already {record.phase.value}"
            )
        if target_step is not None:
            _require_int(target_step, field="target_step", minimum=0)
        _require_int(
            start_weight_version,
            field="start_weight_version",
            minimum=0,
        )
        self._groups[group_id] = PromptGroupRecoveryRecord(
            group_id=record.group_id,
            admission_id=record.admission_id,
            prompt_id=record.prompt_id,
            prompt_ref=record.prompt_ref,
            runtime_prompt_payload=record.runtime_prompt_payload,
            expected_generations=record.expected_generations,
            target_step=target_step,
            start_weight_version=start_weight_version,
            phase=PromptGroupPhase.ADMITTED,
        )

    def bind_runtime_prompt(
        self,
        group_id: str,
        prompt_payload: DatumSpec,
    ) -> None:
        """Attach a dataset-rehydrated prompt after identity validation."""
        record = self._require_group(group_id)
        _validate_prompt_identity(
            record.prompt_ref,
            prompt_payload,
            group_id=group_id,
        )
        self._groups[group_id] = PromptGroupRecoveryRecord(
            group_id=record.group_id,
            admission_id=record.admission_id,
            prompt_id=record.prompt_id,
            prompt_ref=PromptRef(
                sample_id=record.prompt_ref.sample_id,
                task_name=record.prompt_ref.task_name,
            ),
            runtime_prompt_payload=prompt_payload,
            expected_generations=record.expected_generations,
            target_step=record.target_step,
            start_weight_version=record.start_weight_version,
            phase=record.phase,
        )

    def get_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        """Return a record copy while sharing its immutable runtime prompt."""
        return copy.copy(self._require_group(group_id))

    def groups(self) -> list[PromptGroupRecoveryRecord]:
        """Return record copies in reservation order without cloning prompts."""
        return [copy.copy(record) for record in self._groups.values()]

    def discard_group(self, group_id: str) -> None:
        """Release ownership after canonical commit or intentional discard."""
        self._require_group(group_id)
        del self._groups[group_id]

    def discard_canonical_groups(self, group_ids: set[str]) -> int:
        """Drop ledger copies already owned by canonical replay metadata."""
        discarded = 0
        for group_id in list(self._groups):
            if group_id in group_ids:
                del self._groups[group_id]
                discarded += 1
        return discarded

    def state_dict(self) -> RolloutRecoveryState:
        """Return versioned references without serializing full prompt payloads."""
        groups: list[PromptGroupRecoveryState] = []
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
                    "expected_generations": record.expected_generations,
                    "target_step": record.target_step,
                    "start_weight_version": record.start_weight_version,
                    "phase": record.phase.value,
                }
            )
        return {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": groups,
        }

    def load_state_dict(self, state: RolloutRecoveryState) -> None:
        """Replace this empty ledger from a validated checkpoint payload."""
        if self._groups:
            raise RuntimeError(
                "cannot restore into a non-empty rollout recovery ledger"
            )
        if not isinstance(state, dict):
            raise TypeError(
                "rollout recovery state must be a dictionary, got "
                f"{type(state).__name__}"
            )
        if state.get("schema_version") != ROLLOUT_RECOVERY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported rollout recovery schema_version="
                f"{state.get('schema_version')!r}; expected "
                f"{ROLLOUT_RECOVERY_SCHEMA_VERSION}"
            )
        groups = state.get("groups")
        if not isinstance(groups, list):
            raise TypeError("rollout recovery groups must be a list")

        restored: dict[str, PromptGroupRecoveryRecord] = {}
        for index, raw_group in enumerate(groups):
            if not isinstance(raw_group, dict):
                raise TypeError(
                    f"rollout recovery groups[{index}] must be a dictionary"
                )
            group_id = raw_group.get("group_id")
            prompt_id = raw_group.get("prompt_id")
            admission_id = raw_group.get("admission_id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(
                    f"rollout recovery groups[{index}].group_id must be non-empty"
                )
            if group_id in restored:
                raise ValueError(f"duplicate recovery group_id={group_id!r}")
            if not isinstance(admission_id, str) or not admission_id:
                raise ValueError(
                    f"rollout recovery groups[{index}].admission_id must be non-empty"
                )
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(
                    f"rollout recovery groups[{index}].prompt_id must be non-empty"
                )
            expected_generations = _require_int(
                raw_group.get("expected_generations"),
                field=f"groups[{index}].expected_generations",
                minimum=1,
            )
            start_weight_version = _require_int(
                raw_group.get("start_weight_version"),
                field=f"groups[{index}].start_weight_version",
                minimum=0,
            )
            target_step = raw_group.get("target_step")
            if target_step is not None:
                target_step = _require_int(
                    target_step,
                    field=f"groups[{index}].target_step",
                    minimum=0,
                )
            raw_phase = raw_group.get("phase")
            if not isinstance(raw_phase, str):
                raise ValueError(
                    f"rollout recovery groups[{index}].phase is invalid: {raw_phase!r}"
                )
            try:
                phase = PromptGroupPhase(raw_phase)
            except ValueError as error:
                raise ValueError(
                    f"rollout recovery groups[{index}].phase is invalid: {raw_phase!r}"
                ) from error
            raw_prompt_ref = raw_group.get("prompt_ref")
            if not isinstance(raw_prompt_ref, dict):
                raise TypeError(
                    f"rollout recovery groups[{index}].prompt_ref must be a dictionary"
                )
            sample_id = raw_prompt_ref.get("sample_id")
            task_name = raw_prompt_ref.get("task_name")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"rollout recovery groups[{index}].prompt_ref.sample_id "
                    "must be non-empty"
                )
            if sample_id != prompt_id:
                raise ValueError(
                    f"rollout recovery groups[{index}] prompt_id and "
                    "prompt_ref.sample_id must match"
                )
            if task_name is not None and not isinstance(task_name, str):
                raise TypeError(
                    f"rollout recovery groups[{index}].prompt_ref.task_name "
                    "must be a string or None"
                )
            restored[group_id] = PromptGroupRecoveryRecord(
                group_id=group_id,
                admission_id=admission_id,
                prompt_id=prompt_id,
                prompt_ref=PromptRef(
                    sample_id=sample_id,
                    task_name=task_name,
                ),
                runtime_prompt_payload=None,
                expected_generations=expected_generations,
                target_step=target_step,
                start_weight_version=start_weight_version,
                phase=phase,
            )

        admission_states: dict[str, tuple[PromptGroupPhase, int | None]] = {}
        for record in restored.values():
            signature = (record.phase, record.target_step)
            prior = admission_states.setdefault(record.admission_id, signature)
            if prior != signature:
                raise ValueError(
                    "rollout recovery groups sharing admission_id="
                    f"{record.admission_id!r} disagree on phase or target_step"
                )
        self._groups = restored

    def _require_group(self, group_id: str) -> PromptGroupRecoveryRecord:
        try:
            return self._groups[group_id]
        except KeyError as error:
            raise KeyError(f"unknown recovery group_id={group_id!r}") from error

    def __len__(self) -> int:
        return len(self._groups)
