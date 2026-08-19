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

from copy import deepcopy

import pytest

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.rollout_recovery import (
    PromptGroupStatus,
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
    TrainStepStatus,
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
        expected_generations=2,
        target_step=8,
        start_weight_version=7,
    )


def _receipt(gate_rollout_id: str) -> dict:
    return {
        "rollout_id": gate_rollout_id,
        "manifest": [{"staging_key": f"{gate_rollout_id}/call"}],
    }


def _seal(ledger: RolloutRecoveryLedger, group_id: str, generation_index: int) -> None:
    group = ledger.get_group(group_id)
    gate_id = group.siblings[generation_index].current_attempt.gate_rollout_id
    ledger.mark_sibling_sealed(
        group_id,
        generation_index=generation_index,
        gate_rollout_id=gate_id,
        receipt=_receipt(gate_id),
        reward=float(generation_index),
    )


def _finalize(ledger: RolloutRecoveryLedger, group_id: str) -> KVBatchMeta:
    group = ledger.get_group(group_id)
    meta = KVBatchMeta(
        partition_id="rollout",
        task_name="train",
        sample_ids=group.logical_rollout_ids,
        fields=["input_ids"],
        sequence_lengths=[4, 5],
        tags=[{"weight_version": 7}, {"weight_version": 7}],
    )
    ledger.mark_finalization_started(group_id)
    ledger.mark_group_finalized(
        group_id,
        meta=meta,
        group_min_weight_version=7,
        group_max_weight_version=8,
    )
    return meta


def test_retry_preserves_logical_id_and_reuses_sealed_sibling() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    first_gate_ids = group.gate_rollout_ids
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)
    ledger.abandon_unsealed(group.group_id)

    retried = ledger.prepare_incomplete_retry(group.group_id)

    assert retried.logical_rollout_ids == ["group-1_g0", "group-1_g1"]
    assert retried.gate_rollout_ids[0] == first_gate_ids[0]
    assert retried.gate_rollout_ids[1] != first_gate_ids[1]
    assert retried.siblings[0].current_attempt.status == RolloutAttemptStatus.SEALED
    assert retried.siblings[1].current_attempt.status == RolloutAttemptStatus.RESERVED


def test_all_siblings_must_be_sealed_before_finalization() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)

    with pytest.raises(ValueError, match="not ready to finalize"):
        ledger.finalization_inputs(group.group_id)

    _seal(ledger, group.group_id, 1)
    physical_ids, canonical_ids, receipts, rewards = ledger.finalization_inputs(
        group.group_id
    )
    assert canonical_ids == ["group-1_g0", "group-1_g1"]
    assert [receipt["rollout_id"] for receipt in receipts] == physical_ids
    assert rewards == [0.0, 1.0]


def test_finalized_group_remains_until_training_cleanup() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)
    _seal(ledger, group.group_id, 1)
    _finalize(ledger, group.group_id)

    finalized = ledger.get_group(group.group_id)
    assert finalized.status == PromptGroupStatus.FINALIZED
    assert finalized.prompt_payload is None
    assert len(ledger) == 1

    ledger.claim_groups_for_training(
        [group.group_id],
        train_step=3,
        trainer_version=3,
        expected_group_count=1,
    )
    assert (
        ledger.get_group(group.group_id).status
        == PromptGroupStatus.CLAIMED_FOR_TRAINING
    )
    assert ledger.open_train_step is not None
    assert ledger.open_train_step.status == TrainStepStatus.OPEN

    ledger.mark_train_step_applied(3)
    assert (
        ledger.get_group(group.group_id).status
        == PromptGroupStatus.APPLIED_UNCHECKPOINTED
    )
    ledger.release_applied_train_step(3)
    assert len(ledger) == 0
    assert ledger.open_train_step is None


def test_open_train_step_rolls_back_as_one_unit() -> None:
    ledger = RolloutRecoveryLedger()
    group_ids = []
    for index in range(2):
        group = _reserve(ledger, group_id=f"group-{index}")
        ledger.mark_group_dispatched(group.group_id)
        _seal(ledger, group.group_id, 0)
        _seal(ledger, group.group_id, 1)
        _finalize(ledger, group.group_id)
        group_ids.append(group.group_id)
    ledger.claim_groups_for_training(
        [group_ids[0]],
        train_step=4,
        trainer_version=4,
        expected_group_count=2,
    )
    ledger.claim_groups_for_training(
        [group_ids[1]],
        train_step=4,
        trainer_version=4,
        expected_group_count=2,
    )

    ledger.rollback_open_train_step(4)

    assert ledger.open_train_step is None
    assert all(
        ledger.get_group(group_id).status == PromptGroupStatus.FINALIZED
        for group_id in group_ids
    )


def test_state_dict_round_trip_preserves_partial_and_claimed_lineage() -> None:
    ledger = RolloutRecoveryLedger()
    partial = _reserve(ledger, group_id="partial")
    ledger.mark_group_dispatched(partial.group_id)
    _seal(ledger, partial.group_id, 0)
    ledger.abandon_unsealed(partial.group_id)

    finalized = _reserve(ledger, group_id="finalized")
    ledger.mark_group_dispatched(finalized.group_id)
    _seal(ledger, finalized.group_id, 0)
    _seal(ledger, finalized.group_id, 1)
    _finalize(ledger, finalized.group_id)
    ledger.claim_groups_for_training(
        [finalized.group_id],
        train_step=5,
        trainer_version=5,
        expected_group_count=2,
    )

    state = ledger.state_dict()
    restored = RolloutRecoveryLedger.from_state_dict(deepcopy(state))

    assert restored.state_dict() == state
    assert restored.open_train_step is not None
    assert restored.open_train_step.group_ids == ["finalized"]


def test_finalizer_unknown_outcome_is_terminal() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)
    _seal(ledger, group.group_id, 1)
    ledger.mark_finalization_started(group.group_id)
    ledger.mark_finalization_unknown(group.group_id)

    with pytest.raises(ValueError, match="cannot retry"):
        ledger.prepare_incomplete_retry(group.group_id)
