from __future__ import annotations

from copy import deepcopy

import pytest

from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
)


def _prompt() -> dict:
    return {
        "idx": 17,
        "message_log": [{"role": "user", "content": "solve"}],
        "extra_env_info": {"task": "math"},
        "task_name": "nemo_gym",
    }


def _reserve(ledger: RolloutRecoveryLedger, *, group_id: str = "group-1"):
    return ledger.reserve_group(
        group_id=group_id,
        prompt_id="17",
        prompt_payload=_prompt(),  # type: ignore[arg-type]
        expected_generations=3,
        target_step=8,
        start_weight_version=7,
    )


def test_reserve_separates_logical_ids_from_attempt_ids() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)

    assert group.logical_rollout_ids == [
        "group-1_g0",
        "group-1_g1",
        "group-1_g2",
    ]
    assert len(set(group.gate_rollout_ids)) == 3
    assert all(
        gate_id.startswith(f"{logical_id}_a")
        for gate_id, logical_id in zip(
            group.gate_rollout_ids, group.logical_rollout_ids
        )
    )
    assert all(
        sibling.current_attempt.status == RolloutAttemptStatus.RESERVED
        for sibling in group.siblings
    )


def test_retry_preserves_logical_id_and_changes_attempt_identity() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    first_attempt = group.siblings[1].current_attempt

    ledger.mark_group_dispatched(group.group_id)
    ledger.abandon_group(group.group_id)
    retry = ledger.retry_sibling(group.group_id, generation_index=1)
    restored_group = ledger.get_group(group.group_id)

    assert restored_group.siblings[1].logical_rollout_id == "group-1_g1"
    assert retry.attempt_id != first_attempt.attempt_id
    assert retry.gate_rollout_id != first_attempt.gate_rollout_id
    assert retry.gate_rollout_id.startswith("group-1_g1_a")
    assert retry.status == RolloutAttemptStatus.RESERVED


def test_prompt_payload_is_snapshotted_defensively() -> None:
    prompt = _prompt()
    ledger = RolloutRecoveryLedger()
    group = ledger.reserve_group(
        group_id="group-1",
        prompt_id="17",
        prompt_payload=prompt,  # type: ignore[arg-type]
        expected_generations=1,
        target_step=None,
        start_weight_version=0,
    )
    prompt["extra_env_info"]["task"] = "mutated"

    assert group.prompt_payload["extra_env_info"]["task"] == "math"
    assert (
        ledger.get_group("group-1").prompt_payload["extra_env_info"]["task"] == "math"
    )


def test_state_dict_round_trip_preserves_lineage() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    state = ledger.state_dict()

    restored = RolloutRecoveryLedger.from_state_dict(deepcopy(state))

    assert state["schema_version"] == ROLLOUT_RECOVERY_SCHEMA_VERSION
    assert restored.state_dict() == state
    assert all(
        sibling.current_attempt.status == RolloutAttemptStatus.DISPATCHED
        for sibling in restored.get_group(group.group_id).siblings
    )


def test_invalid_state_transition_fails_without_partial_mutation() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)

    with pytest.raises(ValueError, match="cannot finalize"):
        ledger.mark_group_finalized(group.group_id)

    assert all(
        sibling.current_attempt.status == RolloutAttemptStatus.RESERVED
        for sibling in ledger.get_group(group.group_id).siblings
    )


def test_release_requires_finalized_group() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)

    with pytest.raises(ValueError, match="cannot release unfinished"):
        ledger.release_finalized_group(group.group_id)

    ledger.mark_group_dispatched(group.group_id)
    ledger.mark_group_finalized(group.group_id)
    ledger.release_finalized_group(group.group_id)
    assert len(ledger) == 0


def test_restore_rejects_identity_mismatch() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(ledger)
    state = ledger.state_dict()
    state["groups"][0]["siblings"][0]["logical_rollout_id"] = "wrong"

    with pytest.raises(ValueError, match="logical rollout ID mismatch"):
        RolloutRecoveryLedger.from_state_dict(state)


def test_restore_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported rollout-recovery schema"):
        RolloutRecoveryLedger.from_state_dict({"schema_version": 999, "groups": []})
