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
import math
import os
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, NamedTuple, Protocol, cast

from nemo_rl.models.policy.draft_config import DraftUpdateScheduleConfig

DraftUpdatePhase = Literal[
    "monitoring", "training_burst", "awaiting_post_refit_observation"
]
DraftUpdateReason = Literal[
    "always",
    "fixed_interval",
    "adaptive_degradation",
    "adaptive_burst",
    "max_interval",
    "none",
]

_STATE_VERSION = 1
_HISTORY_LIMIT = 64
_OUTCOME_KEYS = frozenset(
    {
        "update_attempted",
        "update_successful",
        "update_skipped",
        "draft_refit_attempted",
        "draft_refit_successful",
        "draft_refit_skipped",
        "forced_update",
        "forced_refit",
    }
)


class DecisionHistoryEntry(NamedTuple):
    global_step: int
    decision_id: int
    update_requested: bool
    draft_refit_requested: bool
    reason: str
    forced: bool


@dataclass(frozen=True, slots=True)
class DraftUpdateDecision:
    global_step: int
    decision_id: int
    update_requested: bool
    draft_refit_requested: bool
    reason: DraftUpdateReason
    observed_acceptance: float | None
    forced: bool = False
    applied_draft_version: int = 0


@dataclass(frozen=True, slots=True)
class AppliedDraftSnapshot:
    """An immutable, digest-bound serving-draft artifact."""

    version: int
    path: str
    size_bytes: int
    sha256: str


@dataclass(slots=True)
class DraftUpdateScheduleState:
    version: int
    schedule_origin_step: int
    last_update_step: int | None
    last_applied_refit_step: int | None
    applied_draft_version: int
    acceptance_ewma: float | None
    reference_acceptance_ewma: float | None
    valid_observations: int
    phase: DraftUpdatePhase
    burst_updates: int
    next_decision_id: int
    last_decided_step: int
    attempted_updates: int
    successful_updates: int
    failed_updates: int
    skipped_updates: int
    attempted_refits: int
    successful_refits: int
    failed_refits: int
    skipped_refits: int
    forced_updates: int
    forced_refits: int
    decision_history: tuple[DecisionHistoryEntry, ...]


def _finite_acceptance(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint value is not canonical JSON") from error


def _new_state(origin_step: int) -> DraftUpdateScheduleState:
    return DraftUpdateScheduleState(
        version=_STATE_VERSION,
        schedule_origin_step=origin_step,
        last_update_step=None,
        last_applied_refit_step=None,
        applied_draft_version=0,
        acceptance_ewma=None,
        reference_acceptance_ewma=None,
        valid_observations=0,
        phase="monitoring",
        burst_updates=0,
        next_decision_id=1,
        last_decided_step=origin_step,
        attempted_updates=0,
        successful_updates=0,
        failed_updates=0,
        skipped_updates=0,
        attempted_refits=0,
        successful_refits=0,
        failed_refits=0,
        skipped_refits=0,
        forced_updates=0,
        forced_refits=0,
        decision_history=(),
    )


class DraftUpdateScheduler:
    def __init__(
        self,
        config: DraftUpdateScheduleConfig,
        state: DraftUpdateScheduleState,
    ) -> None:
        self.config = config
        self.state = state
        self._pending: DraftUpdateDecision | None = None

    @classmethod
    def create(
        cls,
        config: DraftUpdateScheduleConfig,
        *,
        origin_step: int,
        restored: Mapping[str, object] | None = None,
    ) -> DraftUpdateScheduler:
        if type(origin_step) is not int or origin_step < 0:
            raise ValueError("origin_step must be a nonnegative integer")
        if restored is None:
            return cls(config, _new_state(origin_step))
        if set(restored) != {"state_version", "config", "state"}:
            raise ValueError("draft update schedule checkpoint schema mismatch")
        if (
            type(restored.get("state_version")) is not int
            or restored["state_version"] != _STATE_VERSION
        ):
            raise ValueError("unsupported draft update schedule state version")
        restored_config = restored.get("config")
        expected_config = config.model_dump(mode="json")
        if type(restored_config) is not dict or _canonical_json(
            restored_config
        ) != _canonical_json(expected_config):
            raise ValueError("resolved draft update schedule does not match checkpoint")
        restored_state = restored.get("state")
        if not isinstance(restored_state, Mapping):
            raise ValueError("draft update schedule state must be a mapping")
        validate_scheduler_state_invariants(config, restored_state)
        raw_state = dict(restored_state)
        for field in ("acceptance_ewma", "reference_acceptance_ewma"):
            value = raw_state[field]
            if value is not None:
                raw_state[field] = float(cast(float, value))
        raw_history = cast(list[object], raw_state.pop("decision_history"))
        history = tuple(
            DecisionHistoryEntry(
                global_step=cast(int, cast(Mapping[str, object], entry)["global_step"]),
                decision_id=cast(int, cast(Mapping[str, object], entry)["decision_id"]),
                update_requested=cast(
                    bool, cast(Mapping[str, object], entry)["update_requested"]
                ),
                draft_refit_requested=cast(
                    bool,
                    cast(Mapping[str, object], entry)["draft_refit_requested"],
                ),
                reason=cast(str, cast(Mapping[str, object], entry)["reason"]),
                forced=cast(bool, cast(Mapping[str, object], entry)["forced"]),
            )
            for entry in raw_history
        )
        state = DraftUpdateScheduleState(
            **raw_state,
            decision_history=history,
        )
        if state.schedule_origin_step != origin_step:
            raise ValueError("restored schedule origin does not match checkpoint step")
        return cls(config, state)

    def _consume_observation(self, acceptance: float | None) -> float | None:
        observation = _finite_acceptance(acceptance)
        if observation is None or self.config.mode != "adaptive":
            return observation
        previous = self.state.acceptance_ewma
        alpha = self.config.ewma_alpha
        self.state.acceptance_ewma = (
            observation
            if previous is None
            else alpha * observation + (1.0 - alpha) * previous
        )
        self.state.valid_observations += 1
        if (
            self.state.reference_acceptance_ewma is None
            and self.state.valid_observations >= self.config.min_observations
        ):
            self.state.reference_acceptance_ewma = self.state.acceptance_ewma
        elif (
            self.state.phase == "monitoring"
            and self.state.reference_acceptance_ewma is not None
            and self.state.acceptance_ewma > self.state.reference_acceptance_ewma
        ):
            self.state.reference_acceptance_ewma = self.state.acceptance_ewma
        return observation

    def decide(
        self,
        *,
        global_step: int,
        acceptance: float | None,
    ) -> DraftUpdateDecision:
        if self._pending is not None:
            raise RuntimeError("record the outstanding draft update decision first")
        expected_step = self.state.last_decided_step + 1
        if type(global_step) is not int or global_step != expected_step:
            raise ValueError(
                f"expected global_step={expected_step}, got global_step={global_step}"
            )
        observation = self._consume_observation(acceptance)
        update = False
        refit = False
        forced = False
        reason: DraftUpdateReason = "none"
        update_age = global_step - (
            self.state.last_update_step
            if self.state.last_update_step is not None
            else self.state.schedule_origin_step
        )
        refit_age = global_step - (
            self.state.last_applied_refit_step
            if self.state.last_applied_refit_step is not None
            else self.state.schedule_origin_step
        )
        if self.config.mode == "always":
            update, refit, reason = True, True, "always"
        elif self.config.mode == "fixed":
            if self.config.action == "sparse_update":
                update = update_age >= self.config.fixed_interval
                refit = update
            else:
                update = True
                refit = refit_age >= self.config.fixed_interval
            reason = "fixed_interval" if update or refit else "none"
        elif self.state.phase == "awaiting_post_refit_observation":
            if observation is not None:
                acceptance_ewma = self.state.acceptance_ewma
                reference_ewma = self.state.reference_acceptance_ewma
                if acceptance_ewma is None:
                    raise RuntimeError("adaptive state is missing acceptance evidence")
                if reference_ewma is not None:
                    gap = reference_ewma - acceptance_ewma
                    if gap <= self.config.recovery_threshold:
                        self.state.phase = "monitoring"
                        self.state.burst_updates = 0
                    elif self.state.burst_updates >= self.config.max_burst_updates:
                        raise RuntimeError(
                            f"max_burst_updates={self.config.max_burst_updates} "
                            f"exhausted; reference={reference_ewma}; "
                            f"current={acceptance_ewma}; "
                            f"history={self.state.decision_history}"
                        )
                    else:
                        update, refit, reason = True, True, "adaptive_burst"
                        self.state.phase = "training_burst"
        elif update_age >= self.config.max_interval:
            update, refit, forced, reason = True, True, True, "max_interval"
        elif (
            update_age >= self.config.min_interval
            and self.state.reference_acceptance_ewma is not None
            and self.state.acceptance_ewma is not None
            and self.state.reference_acceptance_ewma - self.state.acceptance_ewma
            >= self.config.degradation_threshold
        ):
            update, refit, reason = True, True, "adaptive_degradation"
            self.state.phase = "training_burst"
        decision = DraftUpdateDecision(
            global_step=global_step,
            decision_id=self.state.next_decision_id,
            update_requested=update,
            draft_refit_requested=refit,
            reason=reason,
            observed_acceptance=observation,
            forced=forced,
            applied_draft_version=self.state.applied_draft_version,
        )
        self.state.next_decision_id += 1
        self.state.last_decided_step = global_step
        self._pending = decision
        return decision

    def record_outcome(
        self,
        decision: DraftUpdateDecision,
        *,
        update_attempted: bool,
        update_successful: bool,
        draft_refit_attempted: bool,
        draft_refit_successful: bool,
    ) -> None:
        if self._pending != decision:
            raise RuntimeError("stale or mismatched draft update decision outcome")
        values = (
            update_attempted,
            update_successful,
            draft_refit_attempted,
            draft_refit_successful,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("draft update outcome values must be booleans")
        if update_attempted != decision.update_requested:
            raise RuntimeError("draft update attempt does not match decision")
        if draft_refit_attempted and not decision.draft_refit_requested:
            raise RuntimeError("out-of-band draft refit attempt")
        if update_successful and not update_attempted:
            raise RuntimeError("draft update cannot succeed without an attempt")
        if draft_refit_successful and not draft_refit_attempted:
            raise RuntimeError("draft refit cannot succeed without an attempt")
        if update_attempted:
            self.state.attempted_updates += 1
            if update_successful:
                self.state.successful_updates += 1
                self.state.last_update_step = decision.global_step
            else:
                self.state.failed_updates += 1
        else:
            self.state.skipped_updates += 1
        if draft_refit_attempted:
            self.state.attempted_refits += 1
            if draft_refit_successful:
                self.state.successful_refits += 1
                self.state.last_applied_refit_step = decision.global_step
                self.state.applied_draft_version = decision.decision_id
            else:
                self.state.failed_refits += 1
        else:
            self.state.skipped_refits += 1
        if decision.forced:
            self.state.forced_updates += int(
                decision.update_requested and update_successful
            )
            self.state.forced_refits += int(
                decision.draft_refit_requested and draft_refit_successful
            )
        if (
            self.config.mode == "adaptive"
            and decision.draft_refit_requested
            and draft_refit_successful
        ):
            self.state.burst_updates += 1
            self.state.phase = "awaiting_post_refit_observation"
        history = deque(self.state.decision_history, maxlen=_HISTORY_LIMIT)
        history.append(
            DecisionHistoryEntry(
                decision.global_step,
                decision.decision_id,
                decision.update_requested,
                decision.draft_refit_requested,
                decision.reason,
                decision.forced,
            )
        )
        self.state.decision_history = tuple(history)
        self._pending = None
        if decision.update_requested and not update_successful:
            raise RuntimeError("requested draft update failed")
        if decision.draft_refit_requested and not draft_refit_successful:
            raise RuntimeError("requested draft refit failed")

    def state_dict(self) -> dict[str, object]:
        if self._pending is not None:
            raise RuntimeError("cannot checkpoint an outstanding draft update decision")
        state = asdict(self.state)
        state["decision_history"] = [
            entry._asdict() for entry in self.state.decision_history
        ]
        return {
            "state_version": _STATE_VERSION,
            "config": self.config.model_dump(mode="json"),
            "state": state,
        }

    def metrics(self, decision: DraftUpdateDecision) -> dict[str, float]:
        update_origin = (
            self.state.last_update_step
            if self.state.last_update_step is not None
            else self.state.schedule_origin_step
        )
        refit_origin = (
            self.state.last_applied_refit_step
            if self.state.last_applied_refit_step is not None
            else self.state.schedule_origin_step
        )
        return {
            "draft_schedule/applied_draft_version": float(
                decision.applied_draft_version
            ),
            "draft_schedule/update_requested": float(decision.update_requested),
            "draft_schedule/refit_requested": float(decision.draft_refit_requested),
            "draft_schedule/steps_since_update": float(
                decision.global_step - update_origin
            ),
            "draft_schedule/steps_since_refit": float(
                decision.global_step - refit_origin
            ),
            "draft_schedule/acceptance_ewma": (
                float("nan")
                if self.state.acceptance_ewma is None
                else self.state.acceptance_ewma
            ),
            "draft_schedule/reference_acceptance_ewma": (
                float("nan")
                if self.state.reference_acceptance_ewma is None
                else self.state.reference_acceptance_ewma
            ),
        }


def validate_scheduler_state_invariants(
    config: DraftUpdateScheduleConfig,
    state: Mapping[str, object],
) -> None:
    required = set(DraftUpdateScheduleState.__dataclass_fields__)
    if set(state) != required:
        raise ValueError("draft update schedule state schema mismatch")
    integer_fields = required - {
        "last_update_step",
        "last_applied_refit_step",
        "acceptance_ewma",
        "reference_acceptance_ewma",
        "phase",
        "decision_history",
    }
    for field in integer_fields:
        if type(state[field]) is not int or cast(int, state[field]) < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    for field in ("last_update_step", "last_applied_refit_step"):
        value = state[field]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{field} must be a nonnegative integer or None")
    for field in ("acceptance_ewma", "reference_acceptance_ewma"):
        value = state[field]
        if value is not None:
            if (
                type(value) not in (int, float)
                or _finite_acceptance(cast(float, value)) is None
            ):
                raise ValueError(f"{field} must be finite and within [0, 1]")
    if state["version"] != _STATE_VERSION:
        raise ValueError("unsupported scheduler state version")
    origin = cast(int, state["schedule_origin_step"])
    last_decided = cast(int, state["last_decided_step"])
    next_decision_id = cast(int, state["next_decision_id"])
    if last_decided < origin:
        raise ValueError("last_decided_step precedes schedule_origin_step")
    closed_decisions = last_decided - origin
    if next_decision_id != closed_decisions + 1:
        raise ValueError("next_decision_id does not match last_decided_step")
    attempted_updates = cast(int, state["attempted_updates"])
    successful_updates = cast(int, state["successful_updates"])
    failed_updates = cast(int, state["failed_updates"])
    skipped_updates = cast(int, state["skipped_updates"])
    attempted_refits = cast(int, state["attempted_refits"])
    successful_refits = cast(int, state["successful_refits"])
    failed_refits = cast(int, state["failed_refits"])
    skipped_refits = cast(int, state["skipped_refits"])
    if successful_updates + failed_updates != attempted_updates:
        raise ValueError(
            "successful_updates and failed_updates must equal attempted_updates"
        )
    if attempted_updates + skipped_updates != closed_decisions:
        raise ValueError("update counters do not match the closed decision count")
    if successful_refits + failed_refits != attempted_refits:
        raise ValueError(
            "successful_refits and failed_refits must equal attempted_refits"
        )
    if attempted_refits + skipped_refits != closed_decisions:
        raise ValueError("refit counters do not match the closed decision count")
    if cast(int, state["forced_updates"]) > successful_updates:
        raise ValueError("forced_updates exceeds successful_updates")
    if cast(int, state["forced_refits"]) > successful_refits:
        raise ValueError("forced_refits exceeds successful_refits")
    for field, successes in (
        ("last_update_step", successful_updates),
        ("last_applied_refit_step", successful_refits),
    ):
        value = cast(int | None, state[field])
        if (value is None) != (successes == 0):
            raise ValueError(f"{field} does not match successful outcome count")
        if value is not None and not origin < value <= last_decided:
            raise ValueError(f"{field} is outside the decided step range")
    last_refit_step = cast(int | None, state["last_applied_refit_step"])
    expected_applied_version = (
        0 if last_refit_step is None else last_refit_step - origin
    )
    if state["applied_draft_version"] != expected_applied_version:
        raise ValueError("applied_draft_version does not match last_applied_refit_step")
    phase = state["phase"]
    if phase not in (
        "monitoring",
        "training_burst",
        "awaiting_post_refit_observation",
    ):
        raise ValueError("invalid draft update schedule phase")
    if config.mode != "adaptive" and phase != "monitoring":
        raise ValueError("non-adaptive schedule must remain in monitoring phase")
    valid_observations = cast(int, state["valid_observations"])
    acceptance_ewma = cast(float | None, state["acceptance_ewma"])
    reference_ewma = cast(float | None, state["reference_acceptance_ewma"])
    burst_updates = cast(int, state["burst_updates"])
    if config.mode != "adaptive":
        if (
            valid_observations != 0
            or acceptance_ewma is not None
            or reference_ewma is not None
            or burst_updates != 0
        ):
            raise ValueError("non-adaptive schedule contains adaptive state")
    else:
        if (valid_observations == 0) != (acceptance_ewma is None):
            raise ValueError("adaptive observation count and EWMA disagree")
        if reference_ewma is not None and valid_observations < config.min_observations:
            raise ValueError("adaptive reference EWMA lacks minimum observations")
        if burst_updates > config.max_burst_updates:
            raise ValueError("adaptive burst_updates exceeds max_burst_updates")
        if phase == "awaiting_post_refit_observation" and (
            burst_updates == 0 or last_refit_step is None
        ):
            raise ValueError("adaptive awaiting phase lacks a successful refit")
    history = state["decision_history"]
    if not isinstance(history, list) or len(history) > _HISTORY_LIMIT:
        raise ValueError("draft update decision history must be a bounded list")
    if len(history) != min(_HISTORY_LIMIT, closed_decisions):
        raise ValueError("draft update decision history length is not canonical")
    expected_first_id = max(1, closed_decisions - len(history) + 1)
    for offset, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            raise ValueError("draft update decision history entry must be a mapping")
        expected_keys = set(DecisionHistoryEntry._fields)
        if set(entry) != expected_keys:
            raise ValueError("draft update decision history entry schema mismatch")
        expected_id = expected_first_id + offset
        if type(entry["decision_id"]) is not int:
            raise ValueError("history decision_id must be an integer")
        if entry["decision_id"] != expected_id:
            raise ValueError("draft update decision history is not contiguous")
        if type(entry["global_step"]) is not int:
            raise ValueError("history global_step must be an integer")
        if entry["global_step"] != origin + expected_id:
            raise ValueError("draft update decision history step is not contiguous")
        if type(entry["update_requested"]) is not bool:
            raise ValueError("history update_requested must be boolean")
        if type(entry["draft_refit_requested"]) is not bool:
            raise ValueError("history draft_refit_requested must be boolean")
        if type(entry["forced"]) is not bool:
            raise ValueError("history forced must be boolean")
        if entry["reason"] not in (
            "always",
            "fixed_interval",
            "adaptive_degradation",
            "adaptive_burst",
            "max_interval",
            "none",
        ):
            raise ValueError("history reason is invalid")
        reason = entry["reason"]
        update_requested = entry["update_requested"]
        refit_requested = entry["draft_refit_requested"]
        forced = entry["forced"]
        if config.mode == "always" and (
            reason != "always" or not update_requested or not refit_requested or forced
        ):
            raise ValueError("history entry does not match always schedule")
        if config.mode == "fixed":
            if reason not in ("fixed_interval", "none") or forced:
                raise ValueError("history entry does not match fixed schedule")
            if (reason == "fixed_interval") != (update_requested or refit_requested):
                raise ValueError("history fixed reason does not match requests")
            if config.action == "sparse_update" and (
                update_requested != refit_requested
            ):
                raise ValueError("history fixed sparse requests disagree")
            if config.action == "refit_only" and not update_requested:
                raise ValueError("history refit-only schedule skipped an update")
        if config.mode == "adaptive":
            if reason in ("always", "fixed_interval"):
                raise ValueError("history entry does not match adaptive schedule")
            if reason == "none" and (update_requested or refit_requested or forced):
                raise ValueError("history adaptive no-op has requested work")
            if reason in ("adaptive_degradation", "adaptive_burst") and (
                not update_requested or not refit_requested or forced
            ):
                raise ValueError("history adaptive trigger is inconsistent")
            if reason == "max_interval" and (
                not update_requested or not refit_requested or not forced
            ):
                raise ValueError("history adaptive forced trigger is inconsistent")


@dataclass(frozen=True, slots=True)
class DecisionLedgerReceipt:
    path: str
    size_bytes: int
    sha256: str
    first_decision_id: int | None
    last_decision_id: int
    entry_count: int


def decision_outcome_payload(
    decision: DraftUpdateDecision,
    *,
    update_attempted: bool,
    update_successful: bool,
    draft_refit_attempted: bool,
    draft_refit_successful: bool,
) -> dict[str, bool]:
    values = (
        update_attempted,
        update_successful,
        draft_refit_attempted,
        draft_refit_successful,
    )
    if any(type(value) is not bool for value in values):
        raise TypeError("decision outcome values must be booleans")
    return {
        "update_attempted": update_attempted,
        "update_successful": update_successful,
        "update_skipped": not update_attempted,
        "draft_refit_attempted": draft_refit_attempted,
        "draft_refit_successful": draft_refit_successful,
        "draft_refit_skipped": not draft_refit_attempted,
        "forced_update": decision.forced and update_successful,
        "forced_refit": decision.forced and draft_refit_successful,
    }


def _validate_outcome(
    decision: DraftUpdateDecision,
    outcome: Mapping[str, bool],
) -> None:
    if set(outcome) != _OUTCOME_KEYS or any(
        type(outcome[key]) is not bool for key in _OUTCOME_KEYS
    ):
        raise ValueError("decision-ledger outcome schema mismatch")
    if (
        outcome["update_attempted"] != decision.update_requested
        or outcome["update_skipped"] == outcome["update_attempted"]
        or outcome["update_successful"]
        and not outcome["update_attempted"]
        or outcome["draft_refit_attempted"]
        and not decision.draft_refit_requested
        or outcome["draft_refit_skipped"] == outcome["draft_refit_attempted"]
        or outcome["draft_refit_successful"]
        and not outcome["draft_refit_attempted"]
        or outcome["forced_update"]
        != (decision.forced and outcome["update_successful"])
        or outcome["forced_refit"]
        != (decision.forced and outcome["draft_refit_successful"])
    ):
        raise ValueError("decision-ledger outcome disagrees with decision")


def _decision_from_row(row: Mapping[str, object]) -> DraftUpdateDecision:
    decision_keys = {
        "global_step",
        "decision_id",
        "update_requested",
        "draft_refit_requested",
        "reason",
        "observed_acceptance",
        "forced",
        "applied_draft_version",
    }
    if set(row) != decision_keys | {"outcome"}:
        raise ValueError("decision-ledger row schema mismatch")
    for field in ("global_step", "decision_id", "applied_draft_version"):
        if type(row[field]) is not int or cast(int, row[field]) < 0:
            raise ValueError(f"decision-ledger {field} must be a nonnegative integer")
    if cast(int, row["global_step"]) == 0:
        raise ValueError("decision-ledger global_step must be positive")
    if cast(int, row["decision_id"]) == 0:
        raise ValueError("decision-ledger decision_id must be positive")
    for field in ("update_requested", "draft_refit_requested", "forced"):
        if type(row[field]) is not bool:
            raise ValueError(f"decision-ledger {field} must be boolean")
    reason = row["reason"]
    if reason not in (
        "always",
        "fixed_interval",
        "adaptive_degradation",
        "adaptive_burst",
        "max_interval",
        "none",
    ):
        raise ValueError("decision-ledger reason is invalid")
    observed = row["observed_acceptance"]
    if observed is not None:
        if (
            type(observed) not in (int, float)
            or _finite_acceptance(cast(float, observed)) is None
        ):
            raise ValueError("decision-ledger observed_acceptance is invalid")
    decision = DraftUpdateDecision(
        global_step=cast(int, row["global_step"]),
        decision_id=cast(int, row["decision_id"]),
        update_requested=cast(bool, row["update_requested"]),
        draft_refit_requested=cast(bool, row["draft_refit_requested"]),
        reason=cast(DraftUpdateReason, reason),
        observed_acceptance=(
            None if observed is None else float(cast(float, observed))
        ),
        forced=cast(bool, row["forced"]),
        applied_draft_version=cast(int, row["applied_draft_version"]),
    )
    outcome = row["outcome"]
    if not isinstance(outcome, Mapping):
        raise ValueError("decision-ledger outcome must be a mapping")
    _validate_outcome(decision, cast(Mapping[str, bool], outcome))
    return decision


def _parse_ledger(raw: bytes) -> list[dict[str, object]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise ValueError("decision-ledger must end at a complete JSONL row")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("decision-ledger is not valid UTF-8") from error
    rows: list[dict[str, object]] = []
    for line in decoded.splitlines():
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("decision-ledger contains invalid JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("decision-ledger row must be a JSON object")
        row = cast(dict[str, object], parsed)
        _decision_from_row(row)
        rows.append(row)
    return rows


def validate_decision_ledger_receipt(receipt: DecisionLedgerReceipt) -> None:
    if not isinstance(receipt, DecisionLedgerReceipt):
        raise TypeError("decision-ledger receipt has the wrong type")
    if not receipt.path:
        raise ValueError("decision-ledger receipt path must not be empty")
    for field, value in (
        ("size_bytes", receipt.size_bytes),
        ("last_decision_id", receipt.last_decision_id),
        ("entry_count", receipt.entry_count),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"decision-ledger receipt {field} is invalid")
    if receipt.entry_count <= 0:
        raise ValueError("decision-ledger receipt cannot describe an empty segment")
    if type(receipt.first_decision_id) is not int or receipt.first_decision_id <= 0:
        raise ValueError("decision-ledger first_decision_id is invalid")
    if len(receipt.sha256) != 64 or any(
        character not in "0123456789abcdef" for character in receipt.sha256
    ):
        raise ValueError("decision-ledger receipt SHA-256 is invalid")
    path = Path(receipt.path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError("decision-ledger receipt path is unreadable") from error
    if len(raw) != receipt.size_bytes:
        raise ValueError("decision-ledger receipt size does not match file")
    if hashlib.sha256(raw).hexdigest() != receipt.sha256:
        raise ValueError("decision-ledger receipt SHA-256 does not match file")
    rows = _parse_ledger(raw)
    ids = [cast(int, row["decision_id"]) for row in rows]
    expected_ids = list(range(receipt.first_decision_id, receipt.last_decision_id + 1))
    if len(rows) != receipt.entry_count or ids != expected_ids:
        raise ValueError("decision-ledger receipt range is duplicate or gapped")
    steps = [cast(int, row["global_step"]) for row in rows]
    if steps != list(range(steps[0], steps[0] + receipt.entry_count)):
        raise ValueError("decision-ledger global_step sequence is not contiguous")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def replace_bytes_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.recovery.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _encode_row(row: Mapping[str, object]) -> bytes:
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()


class DraftDecisionLedger:
    def __init__(
        self,
        path: Path,
        *,
        sealed_prefixes: tuple[DecisionLedgerReceipt, ...] = (),
    ) -> None:
        self.path = path.resolve()
        self.sealed_prefixes = sealed_prefixes
        expected_next = 1
        sealed_prefix_last_global_step: int | None = None
        for prefix in sealed_prefixes:
            validate_decision_ledger_receipt(prefix)
            if prefix.first_decision_id != expected_next:
                raise ValueError("decision-ledger prefix is not contiguous")
            rows = _parse_ledger(Path(prefix.path).read_bytes())
            first_global_step = cast(int, rows[0]["global_step"])
            if (
                sealed_prefix_last_global_step is not None
                and first_global_step != sealed_prefix_last_global_step + 1
            ):
                raise ValueError("decision-ledger prefix global_step is not contiguous")
            sealed_prefix_last_global_step = cast(int, rows[-1]["global_step"])
            expected_next = prefix.last_decision_id + 1
        self.next_decision_id = expected_next
        self.sealed_prefix_high_water = expected_next - 1
        self._entry_count = 0
        self._first_decision_id: int | None = None
        self._sealed_prefix_last_global_step = sealed_prefix_last_global_step
        self._last_global_step = sealed_prefix_last_global_step
        self._sealed_receipt: DecisionLedgerReceipt | None = None
        self._known_suffix_size = 0
        self._known_suffix_sha256 = hashlib.sha256()
        self._directory_synced = False
        if self.path.exists():
            raise FileExistsError("decision-ledger suffix path already exists")

    def _validate_append(
        self,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
    ) -> tuple[dict[str, object], bytes]:
        if self._sealed_receipt is not None:
            raise RuntimeError("cannot append to a sealed decision-ledger segment")
        if decision.decision_id != self.next_decision_id:
            raise ValueError("decision-ledger append is not contiguous")
        _validate_outcome(decision, outcome)
        row = {**asdict(decision), "outcome": dict(outcome)}
        _decision_from_row(row)
        if (
            self._last_global_step is not None
            and decision.global_step != self._last_global_step + 1
        ):
            raise ValueError("decision-ledger global_step is not contiguous")
        return row, _encode_row(row)

    def _sync_parent_directory(self) -> None:
        if not self._directory_synced:
            _fsync_directory(self.path.parent)
            self._directory_synced = True

    def _adopt_append(
        self,
        decision: DraftUpdateDecision,
        encoded: bytes,
    ) -> None:
        if self._first_decision_id is None:
            self._first_decision_id = decision.decision_id
        self._entry_count += 1
        self.next_decision_id += 1
        self._last_global_step = decision.global_step
        self._known_suffix_size += len(encoded)
        self._known_suffix_sha256.update(encoded)

    def _read_validated_known_prefix(self) -> tuple[bytes, bytes]:
        if not self.path.exists():
            if self._known_suffix_size != 0:
                raise ValueError("decision-ledger suffix disappeared after append")
            return b"", b""
        raw = self.path.read_bytes()
        if len(raw) < self._known_suffix_size:
            raise ValueError("decision-ledger suffix is shorter than committed prefix")
        known_prefix = raw[: self._known_suffix_size]
        if hashlib.sha256(known_prefix).digest() != self._known_suffix_sha256.digest():
            raise ValueError("decision-ledger committed prefix was modified")
        return known_prefix, raw[self._known_suffix_size :]

    def _durabilize_existing_suffix(self) -> None:
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sync_parent_directory()

    def append_closed(
        self,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
    ) -> None:
        _, encoded = self._validate_append(decision, outcome)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("decision-ledger append made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sync_parent_directory()
        self._adopt_append(decision, encoded)

    def append_closed_once(
        self,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
    ) -> None:
        if decision.decision_id == self.next_decision_id:
            _, encoded = self._validate_append(decision, outcome)
            known_prefix, uncommitted_tail = self._read_validated_known_prefix()
            if not uncommitted_tail:
                self.append_closed(decision, outcome)
                return
            if uncommitted_tail == encoded:
                self._durabilize_existing_suffix()
                self._adopt_append(decision, encoded)
                return
            if encoded.startswith(uncommitted_tail):
                replace_bytes_fsync(self.path, known_prefix + encoded)
                self._directory_synced = True
                self._adopt_append(decision, encoded)
                return
            raise ValueError(
                "decision-ledger partial suffix does not match retried decision"
            )
        if decision.decision_id > self.next_decision_id:
            raise ValueError("decision-ledger idempotent append has a gap")
        _validate_outcome(decision, outcome)
        expected = {**asdict(decision), "outcome": dict(outcome)}
        _decision_from_row(expected)
        matches: list[dict[str, object]] = []
        paths = [Path(receipt.path) for receipt in self.sealed_prefixes]
        if self.path.exists():
            paths.append(self.path)
        for path in paths:
            matches.extend(
                row
                for row in _parse_ledger(path.read_bytes())
                if row["decision_id"] == decision.decision_id
            )
        if matches != [expected]:
            raise ValueError("decision-ledger replay differs from closed entry")
        return

    def truncate_to(self, ledger_high_water: int) -> None:
        if self._sealed_receipt is not None:
            raise RuntimeError("truncate only the unsealed post-checkpoint suffix")
        if type(ledger_high_water) is not int:
            raise TypeError("ledger_high_water must be an integer")
        if (
            not self.sealed_prefix_high_water
            <= ledger_high_water
            < self.next_decision_id
        ):
            raise ValueError("ledger_high_water is outside the writable suffix")
        rows = _parse_ledger(self.path.read_bytes()) if self.path.exists() else []
        ids = [cast(int, row["decision_id"]) for row in rows]
        if ids != list(range(self.sealed_prefix_high_water + 1, self.next_decision_id)):
            raise ValueError("decision-ledger suffix is duplicate or gapped")
        retained = [
            row for row in rows if cast(int, row["decision_id"]) <= ledger_high_water
        ]
        expected_ids = list(
            range(self.sealed_prefix_high_water + 1, ledger_high_water + 1)
        )
        if [cast(int, row["decision_id"]) for row in retained] != expected_ids:
            raise ValueError("checkpoint-bound ledger prefix is absent or gapped")
        retained_payload = b"".join(_encode_row(row) for row in retained)
        replace_bytes_fsync(self.path, retained_payload)
        self.next_decision_id = ledger_high_water + 1
        self._entry_count = len(retained)
        self._first_decision_id = (
            cast(int, retained[0]["decision_id"]) if retained else None
        )
        self._last_global_step = (
            cast(int, retained[-1]["global_step"])
            if retained
            else self._sealed_prefix_last_global_step
        )
        self._known_suffix_size = len(retained_payload)
        self._known_suffix_sha256 = hashlib.sha256(retained_payload)
        self._directory_synced = True

    def seal_prefix(self) -> DecisionLedgerReceipt:
        if self._sealed_receipt is not None:
            return self._sealed_receipt
        if self._entry_count == 0 or self._first_decision_id is None:
            raise RuntimeError("cannot seal an empty decision-ledger segment")
        raw = self.path.read_bytes()
        receipt = DecisionLedgerReceipt(
            path=str(self.path),
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            first_decision_id=self._first_decision_id,
            last_decision_id=self.next_decision_id - 1,
            entry_count=self._entry_count,
        )
        validate_decision_ledger_receipt(receipt)
        self._sealed_receipt = receipt
        return receipt

    def open_suffix(self, path: Path) -> DraftDecisionLedger:
        receipt = self.seal_prefix()
        return DraftDecisionLedger(
            path,
            sealed_prefixes=(*self.sealed_prefixes, receipt),
        )


def close_applied_draft_snapshot(
    decision: DraftUpdateDecision,
    apply_receipt: Mapping[str, object],
    *,
    snapshot_path: Path,
) -> AppliedDraftSnapshot:
    """Validate the transfer receipt before making a refit snapshot durable."""
    raw = snapshot_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        not decision.draft_refit_requested
        or apply_receipt.get("successful") is not True
        or apply_receipt.get("version") != decision.decision_id
        or apply_receipt.get("snapshot_path") != str(snapshot_path.resolve())
        or apply_receipt.get("sha256") != digest
    ):
        raise RuntimeError("applied draft snapshot receipt mismatch")
    return AppliedDraftSnapshot(
        version=decision.decision_id,
        path=str(snapshot_path.resolve()),
        size_bytes=len(raw),
        sha256=digest,
    )


def close_initial_draft_snapshot(
    apply_receipt: Mapping[str, object], snapshot_path: Path
) -> AppliedDraftSnapshot:
    raw = snapshot_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        apply_receipt.get("successful") is not True
        or apply_receipt.get("version") != 0
        or apply_receipt.get("snapshot_path") != str(snapshot_path.resolve())
        or apply_receipt.get("sha256") != digest
    ):
        raise RuntimeError("initial serving-draft snapshot receipt mismatch")
    return AppliedDraftSnapshot(
        version=0,
        path=str(snapshot_path.resolve()),
        size_bytes=len(raw),
        sha256=digest,
    )


def validate_applied_draft_snapshot(
    scheduler: DraftUpdateScheduler, snapshot: AppliedDraftSnapshot
) -> None:
    path = Path(snapshot.path)
    raw = path.read_bytes()
    if (
        type(snapshot.version) is not int
        or snapshot.version != scheduler.state.applied_draft_version
        or type(snapshot.size_bytes) is not int
        or snapshot.size_bytes != len(raw)
        or hashlib.sha256(raw).hexdigest() != snapshot.sha256
    ):
        raise ValueError("applied draft snapshot version or digest mismatch")


def durably_install_startup_snapshot(
    save_state: Any,
    snapshot: AppliedDraftSnapshot,
    *,
    flush_save_state: Callable[[Any], Mapping[str, object]],
) -> Mapping[str, object]:
    save_state.applied_draft_snapshot = asdict(snapshot)
    receipt = flush_save_state(save_state)
    if (
        receipt.get("successful") is not True
        or receipt.get("version") != snapshot.version
        or receipt.get("sha256") != snapshot.sha256
    ):
        raise RuntimeError("serving-draft snapshot persistence receipt mismatch")
    return receipt


@dataclass(frozen=True, slots=True)
class DraftStepTransaction:
    transaction_id: str
    decision: DraftUpdateDecision
    base_checkpoint_id: str
    pre_scheduler_sha256: str
    expected_snapshot_path: str | None


@dataclass(frozen=True, slots=True)
class ResolvedDraftStepTransaction:
    transaction: DraftStepTransaction
    decision: DraftUpdateDecision
    outcome: Mapping[str, bool]
    applied_snapshot: AppliedDraftSnapshot | None


class DraftStepTransactionStore(Protocol):
    def begin(
        self,
        decision: DraftUpdateDecision,
        *,
        pre_scheduler_state: Mapping[str, object] | None = None,
        expected_snapshot_path: Path | None = None,
    ) -> DraftStepTransaction: ...
    def resolve(
        self,
        transaction: DraftStepTransaction,
        *,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
        applied_snapshot: AppliedDraftSnapshot | None,
    ) -> None: ...
    def commit_bundle_atomic(
        self,
        transaction: DraftStepTransaction,
        *,
        save_state: Any,
        ledger_high_water: int,
    ) -> Mapping[str, object]: ...
    def mark_committed(
        self, transaction: DraftStepTransaction, receipt: Mapping[str, object]
    ) -> None: ...
    def pending_intents(self) -> tuple[DraftStepTransaction, ...]: ...
    def resolutions_since(
        self, checkpoint_id: str
    ) -> tuple[ResolvedDraftStepTransaction, ...]: ...
    def resolution_for(
        self, transaction_id: str
    ) -> ResolvedDraftStepTransaction | None: ...
    def lookup_durable_apply_receipt(
        self, transaction: DraftStepTransaction
    ) -> Mapping[str, object] | None: ...
    def write_durable_apply_receipt(
        self,
        transaction: DraftStepTransaction,
        *,
        snapshot: AppliedDraftSnapshot,
    ) -> Mapping[str, object]: ...
    def resolve_intent_for_recovery(
        self,
        transaction: DraftStepTransaction,
        *,
        apply_receipt: Mapping[str, object] | None,
    ) -> ResolvedDraftStepTransaction: ...
    def validate_checkpoint_contains(
        self, checkpoint_id: str, resolved: ResolvedDraftStepTransaction
    ) -> None: ...
    def discard_after_checkpoint(
        self, *, checkpoint_id: str, ledger_high_water: int
    ) -> None: ...


def close_draft_step_transaction(
    transaction: DraftStepTransaction,
    *,
    decision: DraftUpdateDecision,
    outcome: Mapping[str, bool],
    applied_snapshot: AppliedDraftSnapshot | None,
    scheduler: DraftUpdateScheduler,
    decision_ledger: DraftDecisionLedger,
    save_state: Any,
    transaction_store: DraftStepTransactionStore,
) -> RuntimeError | None:
    """Close one decision in durable order, preserving failed-work accounting."""
    transaction_store.resolve(
        transaction,
        decision=decision,
        outcome=outcome,
        applied_snapshot=applied_snapshot,
    )
    outcome_error: RuntimeError | None = None
    try:
        scheduler.record_outcome(
            decision,
            update_attempted=outcome["update_attempted"],
            update_successful=outcome["update_successful"],
            draft_refit_attempted=outcome["draft_refit_attempted"],
            draft_refit_successful=outcome["draft_refit_successful"],
        )
    except RuntimeError as error:
        outcome_error = error
    decision_ledger.append_closed_once(decision, outcome)
    save_state.draft_update_schedule = scheduler.state_dict()
    if applied_snapshot is not None:
        save_state.applied_draft_snapshot = asdict(applied_snapshot)
    receipt = transaction_store.commit_bundle_atomic(
        transaction, save_state=save_state, ledger_high_water=decision.decision_id
    )
    if (
        receipt.get("successful") is not True
        or receipt.get("provisional") is not True
        or receipt.get("base_checkpoint_id") != transaction.base_checkpoint_id
        or receipt.get("decision_id") != decision.decision_id
        or receipt.get("scheduler_decision_id") != decision.decision_id
        or receipt.get("snapshot_version") != scheduler.state.applied_draft_version
        or receipt.get("ledger_high_water") != decision.decision_id
    ):
        raise RuntimeError("draft-step atomic bundle receipt mismatch")
    transaction_store.mark_committed(transaction, receipt)
    return outcome_error


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _transaction_from_payload(payload: Mapping[str, object]) -> DraftStepTransaction:
    raw_decision = payload.get("decision")
    if not isinstance(raw_decision, Mapping):
        raise ValueError("draft-step intent has no decision")
    decision = DraftUpdateDecision(**cast(dict[str, object], dict(raw_decision)))
    fields = ("transaction_id", "base_checkpoint_id", "pre_scheduler_sha256")
    if any(not isinstance(payload.get(field), str) for field in fields):
        raise ValueError("draft-step intent has invalid identity")
    expected_path = payload.get("expected_snapshot_path")
    if expected_path is not None and not isinstance(expected_path, str):
        raise ValueError("draft-step intent has invalid snapshot path")
    return DraftStepTransaction(
        transaction_id=cast(str, payload["transaction_id"]),
        decision=decision,
        base_checkpoint_id=cast(str, payload["base_checkpoint_id"]),
        pre_scheduler_sha256=cast(str, payload["pre_scheduler_sha256"]),
        expected_snapshot_path=cast(str | None, expected_path),
    )


class FileDraftStepTransactionStore:
    """Crash-recoverable, append-only draft-step transaction evidence store.

    The store deliberately keeps each stage in its own exclusive file.  A restart
    can therefore distinguish an unstarted resolution from a completed one without
    relying on process-local scheduler state.
    """

    def __init__(
        self,
        result_dir: Path,
        *,
        base_checkpoint_id: str = "step_0",
        crash_after: str | None = None,
    ) -> None:
        self.root = result_dir.resolve() / "draft-step-transactions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_checkpoint_id = base_checkpoint_id
        self.crash_after = crash_after

    def _path(self, transaction_id: str, kind: str) -> Path:
        return self.root / f"{transaction_id}.{kind}.json"

    def _maybe_crash(self, point: str) -> None:
        if self.crash_after == point:
            raise RuntimeError(f"injected crash after {point}")

    def begin(
        self,
        decision: DraftUpdateDecision,
        *,
        pre_scheduler_state: Mapping[str, object] | None = None,
        expected_snapshot_path: Path | None = None,
    ) -> DraftStepTransaction:
        transaction_id = f"decision_{decision.decision_id}-{uuid.uuid4().hex}"
        transaction = DraftStepTransaction(
            transaction_id=transaction_id,
            decision=decision,
            base_checkpoint_id=self.base_checkpoint_id,
            pre_scheduler_sha256=hashlib.sha256(
                _canonical_json(
                    asdict(decision)
                    if pre_scheduler_state is None
                    else pre_scheduler_state
                ).encode()
            ).hexdigest(),
            expected_snapshot_path=(
                None
                if expected_snapshot_path is None
                else str(expected_snapshot_path.resolve())
            ),
        )
        _write_json_exclusive(
            self._path(transaction_id, "intent"),
            {**asdict(transaction), "decision": asdict(decision)},
        )
        self._maybe_crash("intent")
        return transaction

    def resolve(
        self,
        transaction: DraftStepTransaction,
        *,
        decision: DraftUpdateDecision,
        outcome: Mapping[str, bool],
        applied_snapshot: AppliedDraftSnapshot | None,
    ) -> None:
        if decision != transaction.decision:
            raise ValueError("draft-step resolution decision differs from intent")
        _validate_outcome(decision, outcome)
        _write_json_exclusive(
            self._path(transaction.transaction_id, "resolution"),
            {
                "transaction_id": transaction.transaction_id,
                "decision": asdict(decision),
                "outcome": dict(outcome),
                "applied_snapshot": None
                if applied_snapshot is None
                else asdict(applied_snapshot),
            },
        )
        self._maybe_crash("resolution")

    def commit_bundle_atomic(
        self,
        transaction: DraftStepTransaction,
        *,
        save_state: Any,
        ledger_high_water: int,
    ) -> Mapping[str, object]:
        receipt: dict[str, object] = {
            "successful": True,
            "provisional": True,
            "base_checkpoint_id": transaction.base_checkpoint_id,
            "decision_id": transaction.decision.decision_id,
            "scheduler_decision_id": transaction.decision.decision_id,
            "snapshot_version": (
                save_state.draft_update_schedule["state"]["applied_draft_version"]
            ),
            "ledger_high_water": ledger_high_water,
        }
        _write_json_exclusive(self._path(transaction.transaction_id, "bundle"), receipt)
        self._maybe_crash("bundle")
        return receipt

    def mark_committed(
        self, transaction: DraftStepTransaction, receipt: Mapping[str, object]
    ) -> None:
        _write_json_exclusive(
            self._path(transaction.transaction_id, "commit"), dict(receipt)
        )
        self._maybe_crash("commit_marker")

    def _intents(self) -> dict[str, DraftStepTransaction]:
        intents: dict[str, DraftStepTransaction] = {}
        for path in self.root.glob("*.intent.json"):
            transaction = _transaction_from_payload(json.loads(path.read_text()))
            intents[transaction.transaction_id] = transaction
        return intents

    def resolution_for(
        self, transaction_id: str
    ) -> ResolvedDraftStepTransaction | None:
        intent = self._intents().get(transaction_id)
        path = self._path(transaction_id, "resolution")
        if intent is None or not path.is_file():
            return None
        payload = json.loads(path.read_text())
        if payload.get("transaction_id") != transaction_id:
            raise ValueError("draft-step resolution transaction mismatch")
        raw_decision = payload.get("decision")
        raw_outcome = payload.get("outcome")
        if not isinstance(raw_decision, Mapping) or not isinstance(
            raw_outcome, Mapping
        ):
            raise ValueError("draft-step resolution schema mismatch")
        decision = DraftUpdateDecision(**cast(dict[str, object], dict(raw_decision)))
        _validate_outcome(decision, cast(Mapping[str, bool], raw_outcome))
        raw_snapshot = payload.get("applied_snapshot")
        snapshot = (
            None
            if raw_snapshot is None
            else AppliedDraftSnapshot(
                **cast(
                    dict[str, object], dict(cast(Mapping[str, object], raw_snapshot))
                )
            )
        )
        return ResolvedDraftStepTransaction(
            intent, decision, dict(cast(Mapping[str, bool], raw_outcome)), snapshot
        )

    def pending_intents(self) -> tuple[DraftStepTransaction, ...]:
        return tuple(
            transaction
            for transaction in self._intents().values()
            if self.resolution_for(transaction.transaction_id) is None
        )

    def resolutions_since(
        self, checkpoint_id: str
    ) -> tuple[ResolvedDraftStepTransaction, ...]:
        return tuple(
            resolution
            for transaction in self._intents().values()
            if transaction.base_checkpoint_id == checkpoint_id
            if (resolution := self.resolution_for(transaction.transaction_id))
            is not None
        )

    def lookup_durable_apply_receipt(
        self, transaction: DraftStepTransaction
    ) -> Mapping[str, object] | None:
        path = self._path(transaction.transaction_id, "apply")
        return json.loads(path.read_text()) if path.is_file() else None

    def write_durable_apply_receipt(
        self,
        transaction: DraftStepTransaction,
        *,
        snapshot: AppliedDraftSnapshot,
    ) -> Mapping[str, object]:
        """Write receipt evidence which restart re-hashes independently."""
        raw = Path(snapshot.path).read_bytes()
        if (
            snapshot.version != transaction.decision.decision_id
            or (
                transaction.expected_snapshot_path is not None
                and snapshot.path != transaction.expected_snapshot_path
            )
            or snapshot.size_bytes != len(raw)
            or snapshot.sha256 != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("durable apply receipt snapshot digest mismatch")
        receipt: dict[str, object] = {
            "schema_version": 1,
            "successful": True,
            "transaction_id": transaction.transaction_id,
            "decision_id": transaction.decision.decision_id,
            "snapshot": asdict(snapshot),
        }
        _write_json_exclusive(self._path(transaction.transaction_id, "apply"), receipt)
        self._maybe_crash("outcome")
        return receipt

    def resolve_intent_for_recovery(
        self,
        transaction: DraftStepTransaction,
        *,
        apply_receipt: Mapping[str, object] | None,
    ) -> ResolvedDraftStepTransaction:
        successful = (
            apply_receipt is not None and apply_receipt.get("successful") is True
        )
        snapshot: AppliedDraftSnapshot | None = None
        if successful:
            raw_snapshot = apply_receipt.get("snapshot")
            if (
                apply_receipt.get("schema_version") != 1
                or apply_receipt.get("transaction_id") != transaction.transaction_id
                or apply_receipt.get("decision_id") != transaction.decision.decision_id
                or not isinstance(raw_snapshot, Mapping)
            ):
                raise ValueError("durable apply receipt does not match transaction")
            if not isinstance(raw_snapshot, Mapping):
                raise ValueError("durable apply receipt has no snapshot")
            snapshot = AppliedDraftSnapshot(
                **cast(dict[str, object], dict(raw_snapshot))
            )
            raw = Path(snapshot.path).read_bytes()
            if (
                snapshot.version != transaction.decision.decision_id
                or (
                    transaction.expected_snapshot_path is not None
                    and snapshot.path != transaction.expected_snapshot_path
                )
                or snapshot.size_bytes != len(raw)
                or snapshot.sha256 != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("durable apply receipt snapshot digest mismatch")
        outcome = decision_outcome_payload(
            transaction.decision,
            update_attempted=transaction.decision.update_requested,
            update_successful=successful and transaction.decision.update_requested,
            draft_refit_attempted=transaction.decision.draft_refit_requested,
            draft_refit_successful=successful
            and transaction.decision.draft_refit_requested,
        )
        self.resolve(
            transaction,
            decision=transaction.decision,
            outcome=outcome,
            applied_snapshot=snapshot,
        )
        resolved = self.resolution_for(transaction.transaction_id)
        assert resolved is not None
        return resolved

    def validate_checkpoint_contains(
        self, checkpoint_id: str, resolved: ResolvedDraftStepTransaction
    ) -> None:
        if resolved.transaction.base_checkpoint_id != checkpoint_id:
            raise ValueError("draft-step resolution is not bound to checkpoint")

    def discard_after_checkpoint(
        self, *, checkpoint_id: str, ledger_high_water: int
    ) -> None:
        for transaction in self._intents().values():
            if transaction.decision.decision_id > ledger_high_water:
                for path in self.root.glob(f"{transaction.transaction_id}.*.json"):
                    path.unlink()
