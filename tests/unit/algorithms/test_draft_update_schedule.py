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

import json
import os
import stat
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from nemo_rl.algorithms import draft_update_schedule as schedule_module
from nemo_rl.algorithms.draft_update_schedule import (
    DecisionLedgerReceipt,
    DraftDecisionLedger,
    DraftUpdateDecision,
    DraftUpdateScheduler,
    decision_outcome_payload,
    validate_decision_ledger_receipt,
)
from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


def _finish(
    scheduler: DraftUpdateScheduler,
    step: int,
    acceptance: float | None,
) -> tuple[bool, bool]:
    decision = scheduler.decide(global_step=step, acceptance=acceptance)
    scheduler.record_outcome(
        decision,
        update_attempted=decision.update_requested,
        update_successful=decision.update_requested,
        draft_refit_attempted=decision.draft_refit_requested,
        draft_refit_successful=decision.draft_refit_requested,
    )
    return decision.update_requested, decision.draft_refit_requested


def _close_and_append(
    scheduler: DraftUpdateScheduler,
    ledger: DraftDecisionLedger,
    *,
    step: int,
    acceptance: float | None,
) -> DraftUpdateDecision:
    decision = scheduler.decide(global_step=step, acceptance=acceptance)
    outcome = decision_outcome_payload(
        decision,
        update_attempted=decision.update_requested,
        update_successful=decision.update_requested,
        draft_refit_attempted=decision.draft_refit_requested,
        draft_refit_successful=decision.draft_refit_requested,
    )
    scheduler.record_outcome(
        decision,
        update_attempted=outcome["update_attempted"],
        update_successful=outcome["update_successful"],
        draft_refit_attempted=outcome["draft_refit_attempted"],
        draft_refit_successful=outcome["draft_refit_successful"],
    )
    ledger.append_closed(decision, outcome)
    return decision


def _successful_outcome(decision: DraftUpdateDecision) -> dict[str, bool]:
    return decision_outcome_payload(
        decision,
        update_attempted=decision.update_requested,
        update_successful=decision.update_requested,
        draft_refit_attempted=decision.draft_refit_requested,
        draft_refit_successful=decision.draft_refit_requested,
    )


def _encoded_ledger_row(
    decision: DraftUpdateDecision,
    outcome: dict[str, bool],
) -> bytes:
    payload = {**asdict(decision), "outcome": outcome}
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_always_requests_update_and_refit_every_step() -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=5
    )

    assert _finish(scheduler, 6, None) == (True, True)
    assert _finish(scheduler, 7, None) == (True, True)


@pytest.mark.parametrize("interval", [1, 10, 40, 100])
def test_fixed_sparse_fires_at_exact_interval(interval: int) -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=interval
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=11)

    for step in range(12, 11 + interval):
        assert _finish(scheduler, step, None) == (False, False)
    assert _finish(scheduler, 11 + interval, None) == (True, True)
    expected_next = (interval == 1, interval == 1)
    assert _finish(scheduler, 12 + interval, None) == expected_next


def test_adaptive_requests_update_after_threshold_degradation() -> None:
    config = AdaptiveDraftUpdateScheduleConfig(
        min_interval=2,
        max_interval=100,
        min_observations=1,
        ewma_alpha=1.0,
        degradation_threshold=0.1,
        recovery_threshold=0.05,
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)

    assert _finish(scheduler, 1, 0.9) == (False, False)
    assert _finish(scheduler, 2, 0.79) == (True, True)


def test_fixed_sparse_starts_after_interval_and_restores_exactly() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=7)
    assert _finish(scheduler, 8, None) == (False, False)
    assert _finish(scheduler, 9, None) == (True, True)

    restored = DraftUpdateScheduler.create(
        config, origin_step=7, restored=scheduler.state_dict()
    )

    assert restored.decide(global_step=10, acceptance=None) == scheduler.decide(
        global_step=10, acceptance=None
    )


def test_fixed_refit_only_updates_each_step_and_refits_periodically() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="refit_only", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)

    assert _finish(scheduler, 1, None) == (True, False)
    assert _finish(scheduler, 2, None) == (True, True)


def test_duplicate_decide_and_stale_outcome_fail() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    decision = scheduler.decide(global_step=1, acceptance=None)

    with pytest.raises(RuntimeError, match="outstanding"):
        scheduler.decide(global_step=1, acceptance=None)

    stale = replace(decision, decision_id=decision.decision_id + 1)
    with pytest.raises(RuntimeError, match="stale or mismatched"):
        scheduler.record_outcome(
            stale,
            update_attempted=False,
            update_successful=False,
            draft_refit_attempted=False,
            draft_refit_successful=False,
        )


def test_steps_must_be_exactly_monotonic() -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=2
        ),
        origin_step=7,
    )

    with pytest.raises(ValueError, match="expected global_step=8"):
        scheduler.decide(global_step=9, acceptance=None)


def test_failed_requested_update_counts_attempt_but_not_success() -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
        origin_step=0,
    )
    decision = scheduler.decide(global_step=1, acceptance=None)

    with pytest.raises(RuntimeError, match="requested draft update failed"):
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=False,
            draft_refit_attempted=False,
            draft_refit_successful=False,
        )

    state = scheduler.state_dict()["state"]
    assert isinstance(state, dict)
    assert state["attempted_updates"] == 1
    assert state["successful_updates"] == 0
    assert state["failed_updates"] == 1
    assert state["attempted_refits"] == 0
    assert state["successful_refits"] == 0
    assert state["failed_refits"] == 0
    assert state["skipped_refits"] == 1
    assert state["last_update_step"] is None
    assert state["last_decided_step"] == 1


def test_restore_rejects_nonmonotonic_state_and_counters() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=2
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, None)
    saved = scheduler.state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    state["successful_updates"] = 2

    with pytest.raises(ValueError, match="successful_updates.*attempted_updates"):
        DraftUpdateScheduler.create(config, origin_step=0, restored=saved)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("applied_draft_version", 99, "applied_draft_version"),
        ("phase", "invalid", "phase"),
        ("last_decided_step", True, "last_decided_step"),
    ],
)
def test_restore_rejects_corrupted_state_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, None)
    saved = scheduler.state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    state[field] = value

    with pytest.raises(ValueError, match=message):
        DraftUpdateScheduler.create(config, origin_step=0, restored=saved)


def test_restore_rejects_history_that_does_not_match_schedule() -> None:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, None)
    saved = scheduler.state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    history = state["decision_history"]
    assert isinstance(history, list)
    history[0]["reason"] = "none"

    with pytest.raises(ValueError, match="history.*always"):
        DraftUpdateScheduler.create(config, origin_step=0, restored=saved)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("acceptance_ewma", "0.5"),
        ("acceptance_ewma", True),
        ("reference_acceptance_ewma", "0.5"),
        ("reference_acceptance_ewma", False),
    ],
)
def test_restore_rejects_noncanonical_ewma_types(
    field: str,
    value: object,
) -> None:
    config = AdaptiveDraftUpdateScheduleConfig(
        min_interval=2,
        max_interval=10,
        min_observations=1,
        ewma_alpha=0.5,
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, 0.8)
    saved = scheduler.state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    state[field] = value

    with pytest.raises(ValueError, match=field):
        DraftUpdateScheduler.create(config, origin_step=0, restored=saved)


@pytest.mark.parametrize("value", ["1", 1.0, True])
@pytest.mark.parametrize("field", ["global_step", "decision_id"])
def test_restore_rejects_noncanonical_history_integer_types(
    field: str,
    value: object,
) -> None:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, None)
    saved = scheduler.state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    history = state["decision_history"]
    assert isinstance(history, list)
    history[0][field] = value

    with pytest.raises(ValueError, match=field):
        DraftUpdateScheduler.create(config, origin_step=0, restored=saved)


def test_restore_rejects_config_mismatch_and_outer_schema_corruption() -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=10
        ),
        origin_step=0,
    )
    _finish(scheduler, 1, None)
    saved = scheduler.state_dict()

    with pytest.raises(ValueError, match="does not match checkpoint"):
        DraftUpdateScheduler.create(
            FixedDraftUpdateScheduleConfig(
                mode="fixed", action="sparse_update", fixed_interval=40
            ),
            origin_step=0,
            restored=saved,
        )

    saved_with_extra = {**saved, "unexpected": True}
    with pytest.raises(ValueError, match="schema"):
        DraftUpdateScheduler.create(
            scheduler.config,
            origin_step=0,
            restored=saved_with_extra,
        )

    saved_with_bool_version = {**saved, "state_version": True}
    with pytest.raises(ValueError, match="state version"):
        DraftUpdateScheduler.create(
            scheduler.config,
            origin_step=0,
            restored=saved_with_bool_version,
        )

    interval_one_scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
        origin_step=0,
    )
    _finish(interval_one_scheduler, 1, None)
    bool_interval = interval_one_scheduler.state_dict()
    serialized_config = bool_interval["config"]
    assert isinstance(serialized_config, dict)
    serialized_config["fixed_interval"] = True
    with pytest.raises(ValueError, match="does not match checkpoint"):
        DraftUpdateScheduler.create(
            interval_one_scheduler.config,
            origin_step=0,
            restored=bool_interval,
        )


def test_restore_normalizes_valid_numeric_ewma_to_float() -> None:
    config = AdaptiveDraftUpdateScheduleConfig(
        min_interval=2,
        max_interval=10,
        min_observations=1,
        ewma_alpha=0.5,
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, 1.0)
    saved = scheduler.state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    state["acceptance_ewma"] = 1
    state["reference_acceptance_ewma"] = 1

    restored = DraftUpdateScheduler.create(config, origin_step=0, restored=saved)

    assert type(restored.state.acceptance_ewma) is float
    assert type(restored.state.reference_acceptance_ewma) is float


def test_adaptive_forces_once_then_waits_for_evidence() -> None:
    config = AdaptiveDraftUpdateScheduleConfig(
        min_interval=2,
        max_interval=3,
        min_observations=1,
        ewma_alpha=1.0,
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)

    assert _finish(scheduler, 1, 0.8) == (False, False)
    assert _finish(scheduler, 2, None) == (False, False)
    assert _finish(scheduler, 3, None) == (True, True)
    assert _finish(scheduler, 4, None) == (False, False)
    state = scheduler.state_dict()["state"]
    assert isinstance(state, dict)
    assert state["forced_updates"] == 1
    assert state["forced_refits"] == 1


def test_adaptive_smoothed_degradation_burst_and_recovery_transitions() -> None:
    scheduler = DraftUpdateScheduler.create(
        AdaptiveDraftUpdateScheduleConfig(
            min_interval=2,
            max_interval=100,
            min_observations=1,
            ewma_alpha=0.5,
            degradation_threshold=0.15,
            recovery_threshold=0.1,
            max_burst_updates=5,
        ),
        origin_step=0,
    )

    first = scheduler.decide(global_step=1, acceptance=1.0)
    scheduler.record_outcome(
        first,
        update_attempted=False,
        update_successful=False,
        draft_refit_attempted=False,
        draft_refit_successful=False,
    )
    second = scheduler.decide(global_step=2, acceptance=0.8)
    scheduler.record_outcome(
        second,
        update_attempted=False,
        update_successful=False,
        draft_refit_attempted=False,
        draft_refit_successful=False,
    )
    degradation = scheduler.decide(global_step=3, acceptance=0.6)
    assert degradation.reason == "adaptive_degradation"
    scheduler.record_outcome(
        degradation,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    first_burst = scheduler.decide(global_step=4, acceptance=0.8)
    assert first_burst.reason == "adaptive_burst"
    scheduler.record_outcome(
        first_burst,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    second_burst = scheduler.decide(global_step=5, acceptance=1.0)
    assert second_burst.reason == "adaptive_burst"
    scheduler.record_outcome(
        second_burst,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    recovered = scheduler.decide(global_step=6, acceptance=1.0)

    assert recovered.reason == "none"
    assert recovered.update_requested is False
    assert scheduler.state.phase == "monitoring"
    assert scheduler.state.burst_updates == 0
    assert scheduler.state.acceptance_ewma == pytest.approx(0.94375)
    assert scheduler.state.reference_acceptance_ewma == pytest.approx(1.0)


def test_failed_requested_refit_counts_failure_after_successful_update() -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)

    with pytest.raises(RuntimeError, match="requested draft refit failed"):
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=True,
            draft_refit_successful=False,
        )

    state = scheduler.state_dict()["state"]
    assert isinstance(state, dict)
    assert state["attempted_updates"] == 1
    assert state["successful_updates"] == 1
    assert state["failed_updates"] == 0
    assert state["attempted_refits"] == 1
    assert state["successful_refits"] == 0
    assert state["failed_refits"] == 1
    assert state["applied_draft_version"] == 0
    assert state["last_applied_refit_step"] is None


def test_history_is_bounded_and_cap_is_evaluated_after_last_refit() -> None:
    config = AdaptiveDraftUpdateScheduleConfig(
        min_interval=1,
        max_interval=100,
        min_observations=1,
        max_burst_updates=2,
        ewma_alpha=1.0,
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    _finish(scheduler, 1, 0.9)
    _finish(scheduler, 2, 0.5)
    _finish(scheduler, 3, 0.5)

    with pytest.raises(RuntimeError, match="max_burst_updates=2"):
        scheduler.decide(global_step=4, acceptance=0.5)

    state = scheduler.state_dict()["state"]
    assert isinstance(state, dict)
    assert len(state["decision_history"]) <= 64


def test_metric_applied_version_is_bound_to_decision_time() -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="refit_only", fixed_interval=2
        ),
        origin_step=0,
    )
    first = scheduler.decide(global_step=1, acceptance=None)
    scheduler.record_outcome(
        first,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=False,
        draft_refit_successful=False,
    )
    second = scheduler.decide(global_step=2, acceptance=None)
    scheduler.record_outcome(
        second,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    third = scheduler.decide(global_step=3, acceptance=None)

    assert scheduler.metrics(first)["draft_schedule/applied_draft_version"] == 0.0
    assert scheduler.metrics(second)["draft_schedule/applied_draft_version"] == 0.0
    assert scheduler.metrics(third)["draft_schedule/applied_draft_version"] == 2.0
    assert all(key.startswith("draft_schedule/") for key in scheduler.metrics(third))


def test_invalid_acceptance_is_not_adaptive_evidence() -> None:
    scheduler = DraftUpdateScheduler.create(
        AdaptiveDraftUpdateScheduleConfig(
            min_interval=1,
            max_interval=5,
            min_observations=1,
            ewma_alpha=1.0,
        ),
        origin_step=0,
    )

    for step, acceptance in enumerate((float("nan"), -0.1, 1.1, None), start=1):
        assert _finish(scheduler, step, acceptance) == (False, False)
    assert _finish(scheduler, 5, None) == (True, True)
    state = scheduler.state_dict()["state"]
    assert isinstance(state, dict)
    assert state["valid_observations"] == 0


def test_thousand_step_history_is_bounded_and_resume_is_exact(tmp_path: Path) -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=40
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    ledger = DraftDecisionLedger(tmp_path / "run.jsonl")
    saved_at_100: dict[str, object] | None = None
    decision_at_101: DraftUpdateDecision | None = None
    for step in range(1, 1001):
        decision = _close_and_append(scheduler, ledger, step=step, acceptance=None)
        if step == 100:
            saved_at_100 = scheduler.state_dict()
        elif step == 101:
            decision_at_101 = decision

    state = scheduler.state_dict()["state"]
    assert isinstance(state, dict)
    history = state["decision_history"]
    assert isinstance(history, list)
    assert [entry["decision_id"] for entry in history] == list(range(937, 1001))
    assert saved_at_100 is not None
    assert decision_at_101 is not None
    restored = DraftUpdateScheduler.create(config, origin_step=0, restored=saved_at_100)
    assert restored.decide(global_step=101, acceptance=None) == decision_at_101


def _ledger_rows(*receipts: DecisionLedgerReceipt) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for receipt in receipts:
        rows.extend(
            json.loads(line) for line in Path(receipt.path).read_text().splitlines()
        )
    return rows


def test_ledger_seals_prefix_and_continues_in_exclusive_suffix(
    tmp_path: Path,
) -> None:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    ledger = DraftDecisionLedger(tmp_path / "prefix.jsonl")
    for step in range(1, 401):
        _close_and_append(scheduler, ledger, step=step, acceptance=None)
    prefix = ledger.seal_prefix()
    suffix_ledger = ledger.open_suffix(tmp_path / "suffix.jsonl")
    for step in range(401, 1001):
        _close_and_append(scheduler, suffix_ledger, step=step, acceptance=None)
    suffix = suffix_ledger.seal_prefix()

    rows = _ledger_rows(prefix, suffix)
    assert [row["decision_id"] for row in rows] == list(range(1, 1001))
    assert prefix.entry_count == 400
    assert suffix.entry_count == 600
    assert suffix.first_decision_id == 401
    assert suffix.last_decision_id == 1000
    assert sum(row["outcome"]["update_successful"] for row in rows) == 1000
    assert sum(row["outcome"]["draft_refit_successful"] for row in rows) == 1000
    state = scheduler.state_dict()["state"]
    assert isinstance(state, dict)
    assert (
        sum(row["outcome"]["update_attempted"] for row in rows)
        == state["attempted_updates"]
    )
    assert (
        sum(row["outcome"]["forced_update"] for row in rows) == state["forced_updates"]
    )
    assert sum(row["outcome"]["forced_refit"] for row in rows) == state["forced_refits"]


def test_append_after_seal_and_duplicate_or_gap_fail(tmp_path: Path) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    decision = _close_and_append(scheduler, ledger, step=1, acceptance=None)
    outcome = decision_outcome_payload(
        decision,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    ledger.seal_prefix()

    with pytest.raises(RuntimeError, match="sealed"):
        ledger.append_closed(decision, outcome)

    suffix = ledger.open_suffix(tmp_path / "suffix.jsonl")
    with pytest.raises(ValueError, match="gap"):
        suffix.append_closed_once(replace(decision, decision_id=3), outcome)
    with pytest.raises(ValueError, match="differs"):
        suffix.append_closed_once(decision, {**outcome, "update_successful": False})


def test_corrupted_ledger_receipt_fails_closed(tmp_path: Path) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    _close_and_append(scheduler, ledger, step=1, acceptance=None)
    receipt = ledger.seal_prefix()
    Path(receipt.path).write_bytes(Path(receipt.path).read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="size|SHA-256"):
        validate_decision_ledger_receipt(receipt)


def test_unsealed_suffix_can_truncate_and_replay_exactly(tmp_path: Path) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    decisions = [
        _close_and_append(scheduler, ledger, step=step, acceptance=None)
        for step in range(1, 4)
    ]
    ledger.truncate_to(1)

    for decision in decisions[1:]:
        outcome = decision_outcome_payload(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=True,
            draft_refit_successful=True,
        )
        ledger.append_closed_once(decision, outcome)

    receipt = ledger.seal_prefix()
    assert receipt.first_decision_id == 1
    assert receipt.last_decision_id == 3
    assert receipt.entry_count == 3


def test_ledger_rejects_noncontiguous_global_steps(tmp_path: Path) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=10
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    first = _close_and_append(scheduler, ledger, step=11, acceptance=None)
    second = scheduler.decide(global_step=12, acceptance=None)
    outcome = decision_outcome_payload(
        second,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )

    with pytest.raises(ValueError, match="global_step.*contiguous"):
        ledger.append_closed(
            replace(second, global_step=first.global_step + 2), outcome
        )


def test_append_retry_repairs_matching_partial_write_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    decision = scheduler.decide(global_step=1, acceptance=None)
    outcome = _successful_outcome(decision)
    real_write = schedule_module.os.write
    write_calls = 0

    def partial_then_fail(descriptor: int, payload: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(descriptor, payload[: len(payload) // 2])
        raise OSError("injected partial write")

    monkeypatch.setattr(schedule_module.os, "write", partial_then_fail)
    with pytest.raises(OSError, match="partial write"):
        ledger.append_closed(decision, outcome)
    monkeypatch.setattr(schedule_module.os, "write", real_write)

    ledger.append_closed_once(decision, outcome)

    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert rows == [{**asdict(decision), "outcome": outcome}]
    assert ledger.next_decision_id == 2


def test_append_retry_adopts_full_row_after_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    decision = scheduler.decide(global_step=1, acceptance=None)
    outcome = _successful_outcome(decision)
    real_fsync = schedule_module.os.fsync
    failed = False

    def durable_then_fail(descriptor: int) -> None:
        nonlocal failed
        real_fsync(descriptor)
        if not failed and stat.S_ISREG(os.fstat(descriptor).st_mode):
            failed = True
            raise OSError("injected fsync failure")

    monkeypatch.setattr(schedule_module.os, "fsync", durable_then_fail)
    with pytest.raises(OSError, match="fsync failure"):
        ledger.append_closed(decision, outcome)
    monkeypatch.setattr(schedule_module.os, "fsync", real_fsync)

    ledger.append_closed_once(decision, outcome)

    assert len(ledger.path.read_text().splitlines()) == 1
    assert ledger.next_decision_id == 2


def test_append_retry_atomically_repairs_matching_partial_jsonl_tail(
    tmp_path: Path,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    _close_and_append(scheduler, ledger, step=1, acceptance=None)
    decision = scheduler.decide(global_step=2, acceptance=None)
    outcome = _successful_outcome(decision)
    encoded = _encoded_ledger_row(decision, outcome)
    with ledger.path.open("ab") as stream:
        stream.write(encoded[: len(encoded) // 3])
        stream.flush()
        os.fsync(stream.fileno())

    ledger.append_closed_once(decision, outcome)

    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert [row["decision_id"] for row in rows] == [1, 2]
    assert rows[-1] == {**asdict(decision), "outcome": outcome}


def test_append_retry_rejects_unrelated_partial_jsonl_tail(tmp_path: Path) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    decision = scheduler.decide(global_step=1, acceptance=None)
    outcome = _successful_outcome(decision)
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_bytes(b"unrelated partial bytes")

    with pytest.raises(ValueError, match="partial.*does not match"):
        ledger.append_closed_once(decision, outcome)


def test_first_ledger_creation_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    decision = scheduler.decide(global_step=1, acceptance=None)
    real_fsync = schedule_module.os.fsync
    fsync_targets: list[str] = []

    def track_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        fsync_targets.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(schedule_module.os, "fsync", track_fsync)

    ledger.append_closed(decision, _successful_outcome(decision))

    assert fsync_targets == ["file", "directory"]
