# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from copy import deepcopy

import pytest

from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
)

_STAGING_PARTITION = "rollout_staging"


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


def _receipt(gate_rollout_id: str, generation_index: int) -> dict:
    return {
        "rollout_id": gate_rollout_id,
        "manifest": [{"staging_key": f"stage-{generation_index}"}],
    }


def _seal_all(ledger: RolloutRecoveryLedger, group_id: str) -> None:
    group = ledger.get_group(group_id)
    for sibling in group.siblings:
        gate_rollout_id = sibling.current_attempt.gate_rollout_id
        ledger.mark_sibling_sealed(
            group_id,
            generation_index=sibling.generation_index,
            gate_rollout_id=gate_rollout_id,
            receipt=_receipt(gate_rollout_id, sibling.generation_index),
            reward=float(sibling.generation_index),
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


def test_partial_retry_dispatches_only_missing_sibling() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    for generation_index in (0, 2):
        attempt = group.siblings[generation_index].current_attempt
        ledger.mark_sibling_sealed(
            group.group_id,
            generation_index=generation_index,
            gate_rollout_id=attempt.gate_rollout_id,
            receipt=_receipt(attempt.gate_rollout_id, generation_index),
            reward=float(generation_index),
        )
    ledger.prepare_for_restart()

    assert ledger.retryable_generation_indices(group.group_id) == [1]
    retry = ledger.retry_sibling(group.group_id, generation_index=1)
    ledger.mark_siblings_dispatched(group.group_id, generation_indices=[1])

    restored = ledger.get_group(group.group_id)
    assert restored.siblings[0].current_attempt.status == RolloutAttemptStatus.SEALED
    assert (
        restored.siblings[1].current_attempt.status == RolloutAttemptStatus.DISPATCHED
    )
    assert restored.siblings[1].current_attempt.gate_rollout_id == retry.gate_rollout_id
    assert restored.siblings[2].current_attempt.status == RolloutAttemptStatus.SEALED


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
    gate_rollout_id = group.siblings[0].current_attempt.gate_rollout_id
    ledger.mark_sibling_sealed(
        group.group_id,
        generation_index=0,
        gate_rollout_id=gate_rollout_id,
        receipt=_receipt(gate_rollout_id, 0),
        reward=0.5,
    )
    state = ledger.state_dict(staging_partition=_STAGING_PARTITION)

    restored = RolloutRecoveryLedger.from_state_dict(
        deepcopy(state),
        expected_staging_partition=_STAGING_PARTITION,
    )

    assert state["schema_version"] == ROLLOUT_RECOVERY_SCHEMA_VERSION
    assert state["staging_partition"] == _STAGING_PARTITION
    assert restored.state_dict(staging_partition=_STAGING_PARTITION) == state
    restored_group = restored.get_group(group.group_id)
    sealed_attempt = restored_group.siblings[0].current_attempt
    assert sealed_attempt.status == RolloutAttemptStatus.SEALED
    assert sealed_attempt.reward == 0.5
    assert sealed_attempt.staging_keys == ["stage-0"]
    assert sealed_attempt.receipt == _receipt(gate_rollout_id, 0)
    assert all(
        sibling.current_attempt.status == RolloutAttemptStatus.DISPATCHED
        for sibling in restored_group.siblings[1:]
    )


def test_prepare_for_restart_preserves_only_reusable_sealed_attempts() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    gate_rollout_id = group.siblings[1].current_attempt.gate_rollout_id
    ledger.mark_sibling_sealed(
        group.group_id,
        generation_index=1,
        gate_rollout_id=gate_rollout_id,
        receipt=_receipt(gate_rollout_id, 1),
        reward=1.0,
    )

    ledger.prepare_for_restart()

    restored_group = ledger.get_group(group.group_id)
    assert [sibling.current_attempt.status for sibling in restored_group.siblings] == [
        RolloutAttemptStatus.ABANDONED,
        RolloutAttemptStatus.SEALED,
        RolloutAttemptStatus.ABANDONED,
    ]
    assert ledger.expected_staging_keys() == {"stage-1"}


def test_discard_canonical_groups_reconciles_checkpoint_overlap() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(ledger, group_id="canonical")
    _reserve(ledger, group_id="unfinished")

    assert ledger.discard_canonical_groups({"canonical", "missing"}) == 1
    assert [group.group_id for group in ledger.groups()] == ["unfinished"]


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
    _seal_all(ledger, group.group_id)
    ledger.mark_group_finalized(group.group_id)
    ledger.release_finalized_group(group.group_id)
    assert len(ledger) == 0


def test_streamed_seal_preserves_completed_sibling_on_group_abort() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    gate_rollout_id = group.siblings[1].current_attempt.gate_rollout_id
    receipt = _receipt(gate_rollout_id, 1)

    ledger.mark_sibling_sealed(
        group.group_id,
        generation_index=1,
        gate_rollout_id=gate_rollout_id,
        receipt=receipt,
        reward=2.5,
    )
    receipt["manifest"][0]["staging_key"] = "mutated"
    ledger.abandon_group(group.group_id)

    restored_group = ledger.get_group(group.group_id)
    assert (
        restored_group.siblings[0].current_attempt.status
        == RolloutAttemptStatus.ABANDONED
    )
    sealed_attempt = restored_group.siblings[1].current_attempt
    assert sealed_attempt.status == RolloutAttemptStatus.SEALED
    assert sealed_attempt.receipt == _receipt(gate_rollout_id, 1)
    assert sealed_attempt.reward == 2.5
    assert sealed_attempt.staging_keys == ["stage-1"]
    assert (
        restored_group.siblings[2].current_attempt.status
        == RolloutAttemptStatus.ABANDONED
    )


def test_streamed_seal_rejects_receipt_identity_mismatch() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    gate_rollout_id = group.siblings[0].current_attempt.gate_rollout_id

    with pytest.raises(ValueError, match="receipt rollout identity mismatch"):
        ledger.mark_sibling_sealed(
            group.group_id,
            generation_index=0,
            gate_rollout_id=gate_rollout_id,
            receipt=_receipt("wrong-attempt", 0),
            reward=0.0,
        )

    assert (
        ledger.get_group(group.group_id).siblings[0].current_attempt.status
        == RolloutAttemptStatus.DISPATCHED
    )


def test_restore_rejects_identity_mismatch() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(ledger)
    state = ledger.state_dict(staging_partition=_STAGING_PARTITION)
    state["groups"][0]["siblings"][0]["logical_rollout_id"] = "wrong"

    with pytest.raises(ValueError, match="logical rollout ID mismatch"):
        RolloutRecoveryLedger.from_state_dict(
            state,
            expected_staging_partition=_STAGING_PARTITION,
        )


def test_restore_rejects_staging_keys_that_disagree_with_receipt() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    gate_rollout_id = group.siblings[0].current_attempt.gate_rollout_id
    ledger.mark_sibling_sealed(
        group.group_id,
        generation_index=0,
        gate_rollout_id=gate_rollout_id,
        receipt=_receipt(gate_rollout_id, 0),
        reward=0.5,
    )
    state = ledger.state_dict(staging_partition=_STAGING_PARTITION)
    state["groups"][0]["siblings"][0]["attempts"][0]["staging_keys"] = ["wrong-stage"]

    with pytest.raises(ValueError, match="do not match the receipt manifest"):
        RolloutRecoveryLedger.from_state_dict(
            state,
            expected_staging_partition=_STAGING_PARTITION,
        )


def test_restore_rejects_different_staging_partition() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(ledger)
    state = ledger.state_dict(staging_partition=_STAGING_PARTITION)

    with pytest.raises(ValueError, match="staging partition does not match"):
        RolloutRecoveryLedger.from_state_dict(
            state,
            expected_staging_partition="different_staging",
        )


def test_restore_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported rollout-recovery schema"):
        RolloutRecoveryLedger.from_state_dict(
            {"schema_version": 999, "groups": []},
            expected_staging_partition=_STAGING_PARTITION,
        )
