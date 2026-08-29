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

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
    DataPlaneMutationCut,
)
from nemo_rl.algorithms.single_controller_utils.config import RolloutRecoveryConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    PromptGroupPhase,
    RecoveryGranularity,
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
    build_rollout_recovery_state,
    parse_rollout_recovery_state,
)

_T = TypeVar("_T")


def _mutate(callback: Callable[[DataPlaneMutationCut], _T]) -> _T:
    async def apply() -> _T:
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            return callback(cut)

    return asyncio.run(apply())


def _reserve(ledger: RolloutRecoveryLedger, **kwargs: Any):
    return _mutate(lambda cut: ledger.reserve_group(cut, **kwargs))


def _load(ledger: RolloutRecoveryLedger, state) -> None:
    _mutate(lambda cut: ledger.load_state_dict(cut, state))


def _mark(ledger: RolloutRecoveryLedger, group_id: str, **kwargs: Any) -> None:
    _mutate(lambda cut: ledger.mark_group_admitted(cut, group_id, **kwargs))


def _bind(ledger: RolloutRecoveryLedger, group_id: str, prompt: DatumSpec) -> None:
    _mutate(lambda cut: ledger.bind_runtime_prompt(cut, group_id, prompt))


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
        "agent_name": None,
        "recovery_granularity": "sibling",
        "expected_generations": 2,
        "target_step": target_step,
        "start_weight_version": 7,
        "phase": phase,
    }


def test_ledger_round_trip_preserves_group_ownership() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )

    state = ledger.state_dict()
    restored = RolloutRecoveryLedger()
    _load(restored, state)

    with pytest.raises(RuntimeError, match="has not rehydrated prompt"):
        _ = restored.get_group("g7").prompt_payload
    _bind(restored, "g7", _prompt())

    assert restored.state_dict() == state
    assert restored.get_group("g7").phase is PromptGroupPhase.ADMITTED


def test_checkpoint_state_round_trip_preserves_controller_and_ledger_state() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        admitted=True,
    )

    state = build_rollout_recovery_state(
        ledger,
        batch_shortfall={7: 1},
        sampler_stamps_target_steps=True,
    )
    parsed = parse_rollout_recovery_state(state)
    restored = RolloutRecoveryLedger()
    _load(restored, parsed.ledger_state)

    assert [group.group_id for group in restored.groups()] == ["g7"]
    assert parsed.batch_shortfall == {7: 1}
    assert parsed.sampler_stamps_target_steps is True


def test_checkpoint_parser_defaults_fields_absent_from_older_state() -> None:
    parsed = parse_rollout_recovery_state(RolloutRecoveryLedger().state_dict())

    assert parsed.batch_shortfall == {}
    assert parsed.sampler_stamps_target_steps is None


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("batch_shortfall", [], TypeError),
        ("batch_shortfall", {True: 1}, ValueError),
        ("batch_shortfall", {7: -1}, ValueError),
        ("sampler_stamps_target_steps", "yes", TypeError),
    ],
)
def test_checkpoint_parser_rejects_malformed_controller_state(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    state: dict[str, object] = dict(RolloutRecoveryLedger().state_dict())
    state[field] = value

    with pytest.raises(error_type):
        parse_rollout_recovery_state(state)


def test_ledger_rejects_an_expired_mutation_cut() -> None:
    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            ledger.reserve_group(
                cut,
                group_id="g7",
                admission_id="batch-7",
                prompt_id="7",
                prompt_payload=_prompt(),
                expected_generations=2,
                target_step=7,
                start_weight_version=6,
                admitted=True,
            )

        with pytest.raises(RuntimeError, match="no longer active"):
            ledger.discard_group(cut, "g7")

    asyncio.run(exercise())


def test_checkpoint_cut_can_guard_a_ledger_mutation() -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().checkpoint() as cut:
            ledger.reserve_group(
                cut,
                group_id="g7",
                admission_id="batch-7",
                prompt_id="7",
                prompt_payload=_prompt(),
                expected_generations=2,
                target_step=7,
                start_weight_version=6,
                admitted=True,
            )

        assert [group.group_id for group in ledger.groups()] == ["g7"]

    asyncio.run(exercise())


def test_recovery_config_resolves_agent_then_task_then_default() -> None:
    config = RolloutRecoveryConfig(
        default_granularity=RecoveryGranularity.SIBLING,
        agent_granularity_overrides={"genrm_agent": RecoveryGranularity.PROMPT_GROUP},
        task_granularity_overrides={
            "math": RecoveryGranularity.PROMPT_GROUP,
            "agent_wins": RecoveryGranularity.SIBLING,
        },
    )

    agent_policy = config.resolve_for_prompt(
        {
            "task_name": "agent_wins",
            "extra_env_info": {"agent_ref": {"name": "genrm_agent"}},
        }
    )
    task_policy = config.resolve_for_prompt(
        {"task_name": "math", "extra_env_info": None}
    )
    default_policy = config.resolve_for_prompt(
        {"task_name": "other", "extra_env_info": None}
    )

    assert agent_policy.agent_name == "genrm_agent"
    assert agent_policy.granularity is RecoveryGranularity.PROMPT_GROUP
    assert task_policy.granularity is RecoveryGranularity.PROMPT_GROUP
    assert default_policy.granularity is RecoveryGranularity.SIBLING


@pytest.mark.parametrize(
    ("prompt", "error_fragment"),
    [
        (
            {"extra_env_info": {"agent_ref": "genrm_agent"}},
            "agent_ref must be a mapping or None",
        ),
        (
            {"extra_env_info": {"agent_ref": {"name": 7}}},
            "agent_ref.name must be a string or None",
        ),
        (
            {"task_name": 7},
            "task_name must be a string or None",
        ),
    ],
)
def test_recovery_config_rejects_malformed_prompt_identity(
    prompt: dict[str, Any], error_fragment: str
) -> None:
    with pytest.raises(TypeError, match=error_fragment):
        RolloutRecoveryConfig().resolve_for_prompt(prompt)


def test_target_step_none_does_not_mean_unadmitted() -> None:
    ledger = RolloutRecoveryLedger()
    record = _reserve(
        ledger,
        group_id="windowed",
        admission_id="batch-windowed",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=None,
        start_weight_version=6,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )

    assert record.phase is PromptGroupPhase.ADMITTED
    assert record.target_step is None


def test_reserved_group_can_be_admitted_exactly_once() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=None,
        start_weight_version=6,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=False,
    )

    _mark(
        ledger,
        "g7",
        target_step=7,
        start_weight_version=7,
    )

    record = ledger.get_group("g7")
    assert record.phase is PromptGroupPhase.ADMITTED
    assert record.target_step == 7
    assert record.start_weight_version == 7
    with pytest.raises(ValueError, match="already admitted"):
        _mark(
            ledger,
            "g7",
            target_step=8,
            start_weight_version=8,
        )


def test_canonical_groups_are_discarded_without_touching_unfinished_groups() -> None:
    ledger = RolloutRecoveryLedger()
    for idx, group_id in enumerate(("canonical", "unfinished"), start=7):
        _reserve(
            ledger,
            group_id=group_id,
            admission_id="batch-7",
            prompt_id=str(idx),
            prompt_payload=_prompt(idx),
            expected_generations=2,
            target_step=7,
            start_weight_version=7,
            agent_name=None,
            recovery_granularity=RecoveryGranularity.SIBLING,
            admitted=True,
        )

    assert _mutate(lambda cut: ledger.discard_canonical_groups(cut, {"canonical"})) == 1
    assert [group.group_id for group in ledger.groups()] == ["unfinished"]


def test_state_dict_stores_a_prompt_ref_without_the_full_payload() -> None:
    ledger = RolloutRecoveryLedger()
    prompt = _prompt()
    _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=prompt,
        expected_generations=2,
        target_step=7,
        start_weight_version=7,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
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
    _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=original,
        expected_generations=2,
        target_step=7,
        start_weight_version=7,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    restored = RolloutRecoveryLedger()
    _load(restored, ledger.state_dict())

    changed = _prompt()
    changed["message_log"][0]["content"] = "different prompt"
    _bind(restored, "g7", changed)

    assert restored.get_group("g7").prompt_payload == changed


def test_bind_runtime_prompt_rejects_the_wrong_dataset_sample() -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=7,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    restored = RolloutRecoveryLedger()
    _load(restored, ledger.state_dict())

    with pytest.raises(ValueError, match="expected '7'"):
        _bind(restored, "g7", _prompt(8))


def test_prompt_ref_rehydrates_through_a_restored_shuffled_dataloader() -> None:
    """Shuffle position and prompt identity are independent recovery assets."""

    dataloader = _shuffled_prompt_loader()
    iterator = iter(dataloader)
    fetched = [next(iterator) for _ in range(3)]
    owned_prompt = fetched[-1]

    ledger = RolloutRecoveryLedger()
    _reserve(
        ledger,
        group_id="unfinished",
        admission_id="shuffled-batch",
        prompt_id=str(owned_prompt["idx"]),
        prompt_payload=owned_prompt,
        expected_generations=2,
        target_step=1,
        start_weight_version=0,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    ledger_state = ledger.state_dict()
    dataloader_state = dataloader.state_dict()
    expected_next_prompt = next(iterator)

    restored_dataloader = _shuffled_prompt_loader()
    restored_dataloader.load_state_dict(dataloader_state)
    assert next(iter(restored_dataloader)) == expected_next_prompt

    restored_ledger = RolloutRecoveryLedger()
    _load(restored_ledger, ledger_state)
    restored_group = restored_ledger.get_group("unfinished")
    dataset_prompt = restored_dataloader.dataset[
        int(restored_group.prompt_ref.sample_id)
    ]
    _bind(restored_ledger, "unfinished", dataset_prompt)

    assert restored_ledger.get_group("unfinished").prompt_payload == owned_prompt


def test_restart_preserves_sealed_sibling_and_retries_only_interrupted_one() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    sealed_attempt_id = group.siblings[0].current_attempt.attempt_id
    sealed_id = group.gate_rollout_id(0)
    _mutate(
        lambda cut: ledger.mark_sibling_sealed(
            cut,
            "g7",
            generation_index=0,
            gate_rollout_id=sealed_id,
            receipt={
                "rollout_id": sealed_id,
                "manifest": [{"staging_key": "g7/sibling-0/call-0"}],
            },
            reward=1.0,
        )
    )

    restored = RolloutRecoveryLedger.from_state_dict(ledger.state_dict())
    _mutate(lambda cut: restored.prepare_for_restart(cut))
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

    retry = _mutate(lambda cut: restored.prepare_incomplete_retry(cut, "g7"))
    assert retry.siblings[0].current_attempt.attempt_id == sealed_attempt_id
    assert retry.siblings[0].current_attempt.status is RolloutAttemptStatus.SEALED
    assert retry.siblings[1].current_attempt.status is RolloutAttemptStatus.RESERVED


def test_prompt_group_restart_retries_every_sibling_when_one_is_unfinished() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        agent_name="genrm_agent",
        recovery_granularity=RecoveryGranularity.PROMPT_GROUP,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    sealed_id = group.gate_rollout_id(0)
    _mutate(
        lambda cut: ledger.mark_sibling_sealed(
            cut,
            "g7",
            generation_index=0,
            gate_rollout_id=sealed_id,
            receipt={
                "rollout_id": sealed_id,
                "manifest": [{"staging_key": "g7/sibling-0/call-0"}],
            },
            reward=1.0,
        )
    )

    state = ledger.state_dict()
    assert state["groups"][0]["agent_name"] == "genrm_agent"
    assert state["groups"][0]["recovery_granularity"] == "prompt_group"

    restored = RolloutRecoveryLedger.from_state_dict(state)
    _mutate(lambda cut: restored.prepare_for_restart(cut))
    recovered = restored.get_group("g7")

    assert [sibling.current_attempt.status for sibling in recovered.siblings] == [
        RolloutAttemptStatus.ABANDONED,
        RolloutAttemptStatus.ABANDONED,
    ]
    assert restored.expected_staging_keys() == set()

    retry = _mutate(lambda cut: restored.prepare_incomplete_retry(cut, "g7"))
    assert [sibling.current_attempt.status for sibling in retry.siblings] == [
        RolloutAttemptStatus.RESERVED,
        RolloutAttemptStatus.RESERVED,
    ]


def test_prompt_group_restart_keeps_a_fully_sealed_group() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        agent_name="genrm_agent",
        recovery_granularity=RecoveryGranularity.PROMPT_GROUP,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    for generation_index in range(2):
        gate_id = group.gate_rollout_id(generation_index)
        _mutate(
            lambda cut, generation_index=generation_index, gate_id=gate_id: (
                ledger.mark_sibling_sealed(
                    cut,
                    "g7",
                    generation_index=generation_index,
                    gate_rollout_id=gate_id,
                    receipt={
                        "rollout_id": gate_id,
                        "manifest": [
                            {"staging_key": (f"g7/sibling-{generation_index}/call-0")}
                        ],
                    },
                    reward=1.0,
                )
            )
        )

    restored = RolloutRecoveryLedger.from_state_dict(ledger.state_dict())
    _mutate(lambda cut: restored.prepare_for_restart(cut))

    assert restored.get_group("g7").sealed_generation_indices == [0, 1]
    assert restored.expected_staging_keys() == {
        "g7/sibling-0/call-0",
        "g7/sibling-1/call-0",
    }


@pytest.mark.parametrize("unknown_outcome", [False, True])
def test_checkpoint_rejects_ambiguous_finalization_state(
    unknown_outcome: bool,
) -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=1,
        target_step=7,
        start_weight_version=6,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    gate_id = group.gate_rollout_id(0)
    _mutate(
        lambda cut: ledger.mark_sibling_sealed(
            cut,
            "g7",
            generation_index=0,
            gate_rollout_id=gate_id,
            receipt={
                "rollout_id": gate_id,
                "manifest": [{"staging_key": "g7/sibling-0/call-0"}],
            },
            reward=1.0,
        )
    )
    _mutate(lambda cut: ledger.mark_finalization_started(cut, "g7"))
    if unknown_outcome:
        _mutate(lambda cut: ledger.mark_finalization_unknown(cut, "g7"))

    with pytest.raises(RuntimeError, match="checkpoint-unsafe group states"):
        ledger.state_dict()


def test_checkpoint_rejects_an_open_optimizer_step() -> None:
    ledger = RolloutRecoveryLedger()
    group = _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=1,
        target_step=7,
        start_weight_version=6,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    gate_id = group.gate_rollout_id(0)
    _mutate(
        lambda cut: ledger.mark_sibling_sealed(
            cut,
            "g7",
            generation_index=0,
            gate_rollout_id=gate_id,
            receipt={
                "rollout_id": gate_id,
                "manifest": [{"staging_key": "g7/sibling-0/call-0"}],
            },
            reward=1.0,
        )
    )
    _mutate(lambda cut: ledger.mark_finalization_started(cut, "g7"))
    ledger.mark_group_finalized(
        "g7",
        meta=KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["g7_g0"],
        ),
        group_min_weight_version=6,
        group_max_weight_version=6,
    )
    ledger.claim_groups_for_training(
        ["g7"],
        train_step=7,
        trainer_version=7,
        expected_group_count=1,
    )

    with pytest.raises(RuntimeError, match="open optimizer step"):
        ledger.state_dict()


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("recovery_granularity", "banana", "invalid recovery_granularity"),
        ("recovery_granularity", None, "recovery_granularity must be a string"),
        ("agent_name", 123, "agent_name must be a string or None"),
    ],
)
def test_restore_rejects_malformed_recovery_policy_fields(
    field: str, value: Any, error_fragment: str
) -> None:
    ledger = RolloutRecoveryLedger()
    _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=1,
        target_step=7,
        start_weight_version=6,
        agent_name=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    state = ledger.state_dict()
    group_state = state["groups"][0]
    if value is None:
        del group_state[field]
    else:
        group_state[field] = value

    with pytest.raises(ValueError, match=error_fragment):
        _load(RolloutRecoveryLedger(), state)


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
        _load(RolloutRecoveryLedger(), state)  # type: ignore[arg-type]
