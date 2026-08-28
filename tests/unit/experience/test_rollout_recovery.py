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

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    PromptGroupPhase,
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
)


def _prompt(idx: int = 7) -> DatumSpec:
    return {
        "idx": idx,
        "message_log": [{"role": "user", "content": f"prompt {idx}"}],
        "length": 1,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
    }


def _single_prompt_batch(batch: list[DatumSpec]) -> DatumSpec:
    assert len(batch) == 1
    return batch[0]


def _shuffled_prompt_loader(seed: int = 123) -> StatefulDataLoader:
    return StatefulDataLoader(
        [_prompt(idx) for idx in range(12)],
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_single_prompt_batch,
        num_workers=0,
    )


def _group_state(
    idx: int = 7,
    *,
    target_step: int | None = 7,
    phase: str = "admitted",
) -> dict:
    return {
        "group_id": f"g{idx}",
        "admission_id": "batch-7",
        "prompt_id": str(idx),
        "prompt_ref": {
            "sample_id": str(idx),
            "task_name": None,
        },
        "expected_generations": 2,
        "target_step": target_step,
        "start_weight_version": 7,
        "phase": phase,
    }


def test_ledger_round_trip_preserves_group_ownership() -> None:
    ledger = RolloutRecoveryLedger()
    ledger.reserve_group(
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        admitted=True,
    )

    state = ledger.state_dict()
    restored = RolloutRecoveryLedger()
    restored.load_state_dict(state)

    with pytest.raises(RuntimeError, match="has not rehydrated prompt"):
        _ = restored.get_group("g7").prompt_payload
    restored.bind_runtime_prompt("g7", _prompt())

    assert restored.state_dict() == state
    assert restored.get_group("g7").phase is PromptGroupPhase.ADMITTED


def test_target_step_none_does_not_mean_unadmitted() -> None:
    ledger = RolloutRecoveryLedger()
    record = ledger.reserve_group(
        group_id="windowed",
        admission_id="batch-windowed",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=None,
        start_weight_version=6,
        admitted=True,
    )

    assert record.phase is PromptGroupPhase.ADMITTED
    assert record.target_step is None


def test_reserved_group_can_be_admitted_exactly_once() -> None:
    ledger = RolloutRecoveryLedger()
    ledger.reserve_group(
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=None,
        start_weight_version=6,
        admitted=False,
    )

    ledger.mark_group_admitted(
        "g7",
        target_step=7,
        start_weight_version=7,
    )

    record = ledger.get_group("g7")
    assert record.phase is PromptGroupPhase.ADMITTED
    assert record.target_step == 7
    assert record.start_weight_version == 7
    with pytest.raises(ValueError, match="already admitted"):
        ledger.mark_group_admitted(
            "g7",
            target_step=8,
            start_weight_version=8,
        )


def test_canonical_groups_are_discarded_without_touching_unfinished_groups() -> None:
    ledger = RolloutRecoveryLedger()
    for idx, group_id in enumerate(("canonical", "unfinished"), start=7):
        ledger.reserve_group(
            group_id=group_id,
            admission_id="batch-7",
            prompt_id=str(idx),
            prompt_payload=_prompt(idx),
            expected_generations=2,
            target_step=7,
            start_weight_version=7,
            admitted=True,
        )

    assert ledger.discard_canonical_groups({"canonical"}) == 1
    assert [group.group_id for group in ledger.groups()] == ["unfinished"]


def test_state_dict_stores_a_prompt_ref_without_the_full_payload() -> None:
    ledger = RolloutRecoveryLedger()
    prompt = _prompt()
    ledger.reserve_group(
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=prompt,
        expected_generations=2,
        target_step=7,
        start_weight_version=7,
        admitted=True,
    )

    state = ledger.state_dict()
    group_state = state["groups"][0]
    assert "prompt_payload" not in group_state
    assert group_state["prompt_ref"] == {
        "sample_id": "7",
        "task_name": None,
    }
    group_state["prompt_ref"]["sample_id"] = "100"

    assert ledger.get_group("g7").prompt_ref.sample_id == "7"


def test_bind_runtime_prompt_accepts_changed_content_with_the_same_identity() -> None:
    ledger = RolloutRecoveryLedger()
    original = _prompt()
    ledger.reserve_group(
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=original,
        expected_generations=2,
        target_step=7,
        start_weight_version=7,
        admitted=True,
    )
    restored = RolloutRecoveryLedger()
    restored.load_state_dict(ledger.state_dict())

    changed = _prompt()
    changed["message_log"][0]["content"] = "different prompt"
    restored.bind_runtime_prompt("g7", changed)

    assert restored.get_group("g7").prompt_payload == changed


def test_bind_runtime_prompt_rejects_the_wrong_dataset_sample() -> None:
    ledger = RolloutRecoveryLedger()
    ledger.reserve_group(
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=7,
        admitted=True,
    )
    restored = RolloutRecoveryLedger()
    restored.load_state_dict(ledger.state_dict())

    with pytest.raises(ValueError, match="expected '7'"):
        restored.bind_runtime_prompt("g7", _prompt(8))


def test_prompt_ref_rehydrates_through_a_restored_shuffled_dataloader() -> None:
    """Shuffle position and prompt identity are independent recovery assets."""

    dataloader = _shuffled_prompt_loader()
    iterator = iter(dataloader)
    fetched = [next(iterator) for _ in range(3)]
    owned_prompt = fetched[-1]

    ledger = RolloutRecoveryLedger()
    ledger.reserve_group(
        group_id="unfinished",
        admission_id="shuffled-batch",
        prompt_id=str(owned_prompt["idx"]),
        prompt_payload=owned_prompt,
        expected_generations=2,
        target_step=1,
        start_weight_version=0,
        admitted=True,
    )
    ledger_state = ledger.state_dict()
    dataloader_state = dataloader.state_dict()
    expected_next_prompt = next(iterator)

    restored_dataloader = _shuffled_prompt_loader()
    restored_dataloader.load_state_dict(dataloader_state)
    assert next(iter(restored_dataloader)) == expected_next_prompt

    restored_ledger = RolloutRecoveryLedger()
    restored_ledger.load_state_dict(ledger_state)
    restored_group = restored_ledger.get_group("unfinished")
    dataset_prompt = restored_dataloader.dataset[
        int(restored_group.prompt_ref.sample_id)
    ]
    restored_ledger.bind_runtime_prompt("unfinished", dataset_prompt)

    assert restored_ledger.get_group("unfinished").prompt_payload == owned_prompt


def test_restart_preserves_sealed_sibling_and_retries_only_interrupted_one() -> None:
    ledger = RolloutRecoveryLedger()
    group = ledger.reserve_group(
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        admitted=True,
    )
    ledger.mark_group_dispatched("g7")
    sealed_attempt_id = group.siblings[0].current_attempt.attempt_id
    sealed_id = group.gate_rollout_id(0)
    ledger.mark_sibling_sealed(
        "g7",
        generation_index=0,
        gate_rollout_id=sealed_id,
        receipt={
            "rollout_id": sealed_id,
            "manifest": [{"staging_key": "g7/sibling-0/call-0"}],
        },
        reward=1.0,
    )

    restored = RolloutRecoveryLedger.from_state_dict(ledger.state_dict())
    restored.prepare_for_restart()
    recovered_group = restored.get_group("g7")

    assert (
        recovered_group.siblings[0].current_attempt.status
        is RolloutAttemptStatus.SEALED
    )
    assert (
        recovered_group.siblings[1].current_attempt.status
        is RolloutAttemptStatus.ABANDONED
    )
    assert restored.expected_staging_keys() == {"g7/sibling-0/call-0"}

    retry = restored.prepare_incomplete_retry("g7")
    assert retry.siblings[0].current_attempt.attempt_id == sealed_attempt_id
    assert retry.siblings[0].current_attempt.status is RolloutAttemptStatus.SEALED
    assert retry.siblings[1].current_attempt.status is RolloutAttemptStatus.RESERVED


@pytest.mark.parametrize(
    "state",
    [
        {"schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION + 1, "groups": []},
        {"schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION, "groups": {}},
        {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": [_group_state(phase="unknown")],
        },
        {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": [
                _group_state(idx, target_step=target_step, phase=phase)
                for idx, target_step, phase in (
                    (7, None, "reserved"),
                    (8, 7, "admitted"),
                )
            ],
        },
    ],
)
def test_restore_rejects_incompatible_or_malformed_state(state: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        RolloutRecoveryLedger().load_state_dict(state)  # type: ignore[arg-type]
