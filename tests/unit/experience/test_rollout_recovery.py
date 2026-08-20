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
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG
from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    encode_route_plan,
)
from nemo_rl.experience.rollout_recovery import (
    PromptRef,
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


class _NoDeepcopy:
    def __deepcopy__(self, memo):
        raise AssertionError("runtime prompt payload must not be deep-copied")


def _reserve(ledger: RolloutRecoveryLedger, *, group_id: str = "group-1"):
    return ledger.reserve_group(
        group_id=group_id,
        prompt_id="17",
        prompt_ref=PromptRef(sample_id="17", task_name="nemo_gym"),
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
    gate_id = group.gate_rollout_ids[generation_index]
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
    assert finalized.runtime_prompt_payload is None
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
    assert all("prompt_payload" not in group for group in state["groups"])
    partial_state = next(
        group for group in state["groups"] if group["group_id"] == "partial"
    )
    assert partial_state["prompt_ref"] == {
        "sample_id": "17",
        "task_name": "nemo_gym",
    }
    sibling_state = partial_state["siblings"][0]
    assert "logical_rollout_id" not in sibling_state
    attempt_state = sibling_state["attempts"][0]
    assert set(attempt_state).isdisjoint({"attempt_id", "gate_rollout_id"})
    assert len(attempt_state["attempt_uuid"]) == 16
    restored = RolloutRecoveryLedger.from_state_dict(deepcopy(state))

    assert restored.state_dict() == state
    assert restored.open_train_step is not None
    assert restored.open_train_step.group_ids == ["finalized"]
    assert restored.get_group("partial").runtime_prompt_payload is None


def test_state_dict_excludes_runtime_prompt_payload() -> None:
    ledger = RolloutRecoveryLedger()
    prompt = _prompt()
    prompt["__extra__"] = _NoDeepcopy()
    ledger.reserve_group(
        group_id="partial",
        prompt_id="17",
        prompt_ref=PromptRef(sample_id="17", task_name="nemo_gym"),
        prompt_payload=prompt,  # type: ignore[arg-type]
        expected_generations=2,
        target_step=8,
        start_weight_version=7,
    )

    state = ledger.state_dict()

    assert "prompt_payload" not in state["groups"][0]
    assert state["groups"][0]["prompt_ref"]["sample_id"] == "17"


@pytest.mark.parametrize("missing_status", ["group", "attempt"])
def test_state_dict_rejects_missing_status_with_context(missing_status: str) -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(ledger)
    state = ledger.state_dict()
    group_state = state["groups"][0]
    if missing_status == "group":
        group_state.pop("status")
        expected_message = "invalid prompt group status=None"
    else:
        group_state["siblings"][0]["attempts"][0].pop("status")
        expected_message = "invalid rollout attempt status=None"

    with pytest.raises(ValueError, match=expected_message):
        RolloutRecoveryLedger.from_state_dict(state)


def test_restored_partial_group_rebinds_runtime_prompt_by_reference() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger, group_id="partial")
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)

    restored = RolloutRecoveryLedger.from_state_dict(ledger.state_dict())
    assert restored.get_group(group.group_id).runtime_prompt_payload is None

    prompt = _prompt()
    restored.bind_runtime_prompt(group.group_id, prompt)  # type: ignore[arg-type]

    rebound = restored.get_group(group.group_id)
    assert rebound.runtime_prompt_payload is prompt


@pytest.mark.parametrize(
    "prompt",
    [
        {**_prompt(), "idx": 18},
        {**_prompt(), "task_name": "different-task"},
    ],
)
def test_runtime_prompt_reference_mismatch_is_rejected(prompt: dict) -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)

    with pytest.raises(ValueError, match="durable|task"):
        ledger.bind_runtime_prompt(group.group_id, prompt)  # type: ignore[arg-type]


def test_prepare_for_restart_preserves_sealed_and_abandons_inflight() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)

    ledger.prepare_for_restart()

    restored = ledger.get_group(group.group_id)
    assert restored.status == PromptGroupStatus.GENERATING
    assert restored.siblings[0].current_attempt.status == RolloutAttemptStatus.SEALED
    assert restored.siblings[1].current_attempt.status == RolloutAttemptStatus.ABANDONED
    assert ledger.expected_staging_keys() == {f"{restored.gate_rollout_ids[0]}/call"}


def test_finalized_group_does_not_require_cleared_staging_rows() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)
    _seal(ledger, group.group_id, 1)

    _finalize(ledger, group.group_id)

    assert ledger.expected_staging_keys() == set()


def test_finalized_group_retains_deferred_router_staging_ownership() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)
    _seal(ledger, group.group_id, 1)
    staging_keys = [f"{gate_id}/call" for gate_id in group.gate_rollout_ids]
    meta = KVBatchMeta(
        partition_id="rollout",
        task_name="train",
        sample_ids=group.logical_rollout_ids,
        fields=["input_ids"],
        sequence_lengths=[4, 5],
        tags=[
            {
                "weight_version": 7,
                ROUTE_PLAN_TAG: encode_route_plan(
                    RouteAssemblyPlan(
                        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
                        staging_partition="rollout_staging",
                        spans=(),
                        cleanup_staging_keys=(staging_key,),
                        expected_token_length=sequence_length,
                    )
                ),
            }
            for staging_key, sequence_length in zip(staging_keys, [4, 5])
        ],
    )
    ledger.mark_finalization_started(group.group_id)
    ledger.mark_group_finalized(
        group.group_id,
        meta=meta,
        group_min_weight_version=7,
        group_max_weight_version=8,
    )

    assert ledger.expected_staging_keys() == set(staging_keys)


def test_full_step_checkpoint_rejects_open_train_step() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)
    _seal(ledger, group.group_id, 1)
    _finalize(ledger, group.group_id)
    ledger.claim_groups_for_training(
        [group.group_id],
        train_step=3,
        trainer_version=3,
        expected_group_count=1,
    )

    with pytest.raises(RuntimeError, match="open optimizer step"):
        ledger.assert_full_step_checkpoint_safe()


def test_full_step_checkpoint_rejects_unknown_finalizer_outcome() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(ledger)
    ledger.mark_group_dispatched(group.group_id)
    _seal(ledger, group.group_id, 0)
    _seal(ledger, group.group_id, 1)
    ledger.mark_finalization_started(group.group_id)
    ledger.mark_finalization_unknown(group.group_id)

    with pytest.raises(RuntimeError, match="checkpoint-unsafe"):
        ledger.assert_full_step_checkpoint_safe()


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
