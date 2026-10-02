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
import dataclasses
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
from nemo_rl.experience.rollout_recovery import (
    _ATTEMPT_STATE_FIELDS,
    _GROUP_STATE_FIELDS,
    _PROMPT_REF_STATE_FIELDS,
    _SIBLING_STATE_FIELDS,
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    PromptGroupPhase,
    PromptGroupRecoveryRecord,
    PromptRef,
    RecoveryGranularity,
    RecoveryTargetLevel,
    RolloutAttemptRecord,
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
    RolloutSiblingRecord,
    SiblingSealResult,
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
        task_source=None,
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )

    state = ledger.state_dict()
    assert "open_train_step" not in state
    assert {
        "canonical_meta",
        "group_min_weight_version",
        "group_max_weight_version",
        "claimed_train_step",
    }.isdisjoint(state["groups"][0])
    restored = RolloutRecoveryLedger()
    _load(restored, state)

    with pytest.raises(RuntimeError, match="has not rehydrated prompt"):
        _ = restored.get_group("g7").prompt_payload
    _bind(restored, "g7", _prompt())

    assert restored.state_dict() == state
    assert restored.get_group("g7").phase is PromptGroupPhase.ADMITTED


def test_serialized_state_fields_match_recovery_dataclasses() -> None:
    """Require every durable dataclass field to be classified explicitly."""
    assert _PROMPT_REF_STATE_FIELDS == {
        field.name for field in dataclasses.fields(PromptRef)
    }
    assert _SIBLING_STATE_FIELDS == {
        field.name for field in dataclasses.fields(RolloutSiblingRecord)
    }
    assert _ATTEMPT_STATE_FIELDS == {
        field.name for field in dataclasses.fields(RolloutAttemptRecord)
    }
    assert _GROUP_STATE_FIELDS == {
        field.name for field in dataclasses.fields(PromptGroupRecoveryRecord)
    } - {"runtime_prompt_payload"}


@pytest.mark.parametrize(
    ("path", "context"),
    [
        ((), "rollout recovery state"),
        (("groups", 0), "rollout-recovery group"),
        (("groups", 0, "prompt_ref"), "rollout-recovery prompt_ref"),
        (("groups", 0, "siblings", 0), "rollout-recovery sibling"),
        (("groups", 0, "siblings", 0, "attempts", 0), "rollout-recovery attempt"),
    ],
)
def test_ledger_restore_rejects_unknown_fields(
    path: tuple[object, ...], context: str
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
        admitted=True,
    )
    state = ledger.state_dict()
    target: Any = state
    for component in path:
        target = target[component]
    assert isinstance(target, dict)
    target["unexpected"] = True

    with pytest.raises(ValueError, match=rf"{context} contains unknown fields"):
        RolloutRecoveryLedger.from_state_dict(state)


def _sealed_attempt_state() -> dict[str, Any]:
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
            mask_sample=True,
        )
    )
    return ledger.state_dict()


@pytest.mark.parametrize(
    ("case", "error_fragment"),
    [
        ("attempt_uuid", "attempt_uuid must contain exactly 16 bytes"),
        ("status_type", "invalid rollout attempt status"),
        ("status_value", "invalid rollout attempt status"),
        ("staging_keys", "staging_keys must be a list of strings"),
        ("reward", "sealed attempts require a reward"),
        ("mask_sample", "sealed attempts require a boolean mask_sample"),
        (
            "missing_receipt_staging",
            "sealed missing-receipt attempt cannot own staging keys",
        ),
        ("receipt_type", "sealed attempt receipt must be a mapping or None"),
        ("receipt_manifest_type", "receipt must contain a manifest list"),
        ("receipt_identity", "sealed receipt identity mismatch"),
        ("receipt_manifest", "sealed receipt staging manifest mismatch"),
        ("unsealed_payload", "only sealed attempts may retain receipt data"),
    ],
)
def test_restore_rejects_malformed_attempt_fields(
    case: str, error_fragment: str
) -> None:
    state = _sealed_attempt_state()
    attempt = state["groups"][0]["siblings"][0]["attempts"][0]

    if case == "attempt_uuid":
        attempt["attempt_uuid"] = b"short"
    elif case == "status_type":
        attempt["status"] = None
    elif case == "status_value":
        attempt["status"] = "unknown"
    elif case == "staging_keys":
        attempt["staging_keys"] = ["valid", 7]
    elif case == "reward":
        attempt["reward"] = None
    elif case == "mask_sample":
        attempt["mask_sample"] = "yes"
    elif case == "missing_receipt_staging":
        attempt["receipt"] = None
    elif case == "receipt_type":
        attempt["receipt"] = []
    elif case == "receipt_manifest_type":
        attempt["receipt"]["manifest"] = None
    elif case == "receipt_identity":
        attempt["receipt"]["rollout_id"] = "wrong"
    elif case == "receipt_manifest":
        attempt["receipt"]["manifest"] = [{"staging_key": "wrong"}]
    elif case == "unsealed_payload":
        attempt["status"] = RolloutAttemptStatus.DISPATCHED.value
    else:  # pragma: no cover - the parameter table above owns the cases.
        raise AssertionError(f"unknown malformed-attempt case={case!r}")

    with pytest.raises(ValueError, match=error_fragment):
        RolloutRecoveryLedger.from_state_dict(state)


def test_restore_rejects_non_mapping_attempt() -> None:
    state = _sealed_attempt_state()
    state["groups"][0]["siblings"][0]["attempts"][0] = None

    with pytest.raises(ValueError, match="rollout-recovery attempt must be a mapping"):
        RolloutRecoveryLedger.from_state_dict(state)


def test_restore_rejects_duplicate_attempt_identity() -> None:
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
    state = ledger.state_dict()
    siblings = state["groups"][0]["siblings"]
    siblings[1]["attempts"][0]["attempt_uuid"] = siblings[0]["attempts"][0][
        "attempt_uuid"
    ]

    with pytest.raises(ValueError, match="duplicate rollout attempt identity"):
        RolloutRecoveryLedger.from_state_dict(state)


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


def test_checkpoint_parser_rejects_unknown_sidecar_fields() -> None:
    state = build_rollout_recovery_state(
        RolloutRecoveryLedger(),
        batch_shortfall={},
        sampler_stamps_target_steps=True,
    )
    state["unexpected"] = True

    with pytest.raises(
        ValueError, match="rollout recovery sidecar contains unknown fields"
    ):
        parse_rollout_recovery_state(state)


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


def test_recovery_config_resolves_agent_then_task_source_then_default() -> None:
    config = RolloutRecoveryConfig(
        target_level=RecoveryTargetLevel.TURN,
        task_source_target_level_overrides={
            "genrm_compare": RecoveryTargetLevel.PROMPT_GROUP,
        },
        agent_target_level_overrides={
            "legacy_genrm_agent": RecoveryTargetLevel.PROMPT_GROUP,
            "sibling_agent": RecoveryTargetLevel.SIBLING,
        },
    )

    source_policy = config.resolve_for_prompt(
        {
            "extra_env_info": {
                "task_source": "genrm_compare",
                "agent_ref": {"name": "unmapped_agent"},
            }
        }
    )
    agent_policy = config.resolve_for_prompt(
        {
            "extra_env_info": {
                "task_source": "genrm_compare",
                "agent_ref": {"name": "sibling_agent"},
            }
        }
    )
    default_policy = config.resolve_for_prompt(
        {
            "extra_env_info": {
                "task_source": "other",
                "agent_ref": {"name": "unmapped_agent"},
            }
        }
    )
    with pytest.warns(FutureWarning, match="legacy agent_ref"):
        legacy_policy = config.resolve_for_prompt(
            {"extra_env_info": {"agent_ref": {"name": "legacy_genrm_agent"}}}
        )

    assert source_policy.task_source == "genrm_compare"
    assert source_policy.granularity is RecoveryGranularity.PROMPT_GROUP
    assert source_policy.restore_level is RecoveryTargetLevel.PROMPT_GROUP
    assert agent_policy.task_source == "genrm_compare"
    assert agent_policy.granularity is RecoveryGranularity.SIBLING
    assert agent_policy.restore_level is RecoveryTargetLevel.SIBLING
    assert default_policy.task_source == "other"
    assert default_policy.granularity is RecoveryGranularity.SIBLING
    assert default_policy.restore_level is RecoveryTargetLevel.TURN
    assert legacy_policy.task_source is None
    assert legacy_policy.granularity is RecoveryGranularity.PROMPT_GROUP
    assert legacy_policy.restore_level is RecoveryTargetLevel.PROMPT_GROUP


def test_recovery_overrides_may_only_make_the_target_coarser() -> None:
    with pytest.raises(ValueError, match="may only force a coarser recovery level"):
        RolloutRecoveryConfig(
            target_level=RecoveryTargetLevel.SIBLING,
            agent_target_level_overrides={
                "too-fine": RecoveryTargetLevel.TURN,
            },
        )

    prefix_config = RolloutRecoveryConfig(
        target_level=RecoveryTargetLevel.PREFIX,
        task_source_target_level_overrides={
            "restart-call": RecoveryTargetLevel.TURN,
        },
    )
    policy = prefix_config.resolve_for_prompt(
        {"extra_env_info": {"task_source": "restart-call"}}
    )
    assert policy.restore_level is RecoveryTargetLevel.TURN


@pytest.mark.parametrize(
    ("prompt", "error_fragment"),
    [
        (
            {"extra_env_info": {"task_source": 7}},
            "task_source must be a string or None",
        ),
        (
            {"extra_env_info": {"agent_ref": "legacy_agent"}},
            "agent_ref must be a mapping or None",
        ),
        (
            {"extra_env_info": {"agent_ref": {"name": 7}}},
            "agent_ref.name must be a string or None",
        ),
    ],
)
def test_recovery_config_rejects_malformed_prompt_identity(
    prompt: dict[str, Any], error_fragment: str
) -> None:
    with pytest.raises(TypeError, match=error_fragment):
        RolloutRecoveryConfig().resolve_for_prompt(prompt)


def test_recovery_config_rejects_removed_task_name_override() -> None:
    with pytest.raises(ValueError, match="target_level"):
        RolloutRecoveryConfig(
            **{
                "task_granularity_overrides": {
                    "legacy": RecoveryGranularity.PROMPT_GROUP
                }
            }
        )


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
        task_source=None,
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
        task_source=None,
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
            task_source=None,
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
        task_source=None,
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
        task_source=None,
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
        task_source=None,
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
        task_source=None,
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
        task_source=None,
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
            mask_sample=True,
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


@pytest.mark.parametrize(
    "restore_level", [RecoveryTargetLevel.PREFIX, RecoveryTargetLevel.TURN]
)
def test_gym_state_restart_preserves_sealed_sibling_and_restores_other_from_gym(
    restore_level: RecoveryTargetLevel,
) -> None:
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
        recovery_granularity=RecoveryGranularity.SIBLING,
        restore_level=restore_level,
        admitted=True,
    )
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut,
            "g7",
            gym_instance_id="tools/replica-0",
        )
    )
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
            mask_sample=False,
        )
    )

    restored = RolloutRecoveryLedger.from_state_dict(ledger.state_dict())
    inventory = restored.gym_checkpoint_inventory({"tools/replica-0"})
    (unfinished_episode,) = inventory["tools/replica-0"]

    _mutate(
        lambda cut: restored.prepare_for_restart(
            cut,
            restored_gym_episodes={
                ("tools/replica-0", *unfinished_episode),
            },
        )
    )
    recovered = restored.get_group("g7")

    assert recovered.siblings[0].current_attempt.attempt_id == sealed_attempt_id
    assert recovered.siblings[0].current_attempt.status is RolloutAttemptStatus.SEALED
    resumed = recovered.siblings[1].current_attempt
    assert resumed.status is RolloutAttemptStatus.RESERVED
    assert resumed.gym_instance_id == "tools/replica-0"
    assert resumed.gym_attempt == 1
    assert restored.expected_staging_keys() == {"g7/sibling-0/call-0"}

    retry = _mutate(lambda cut: restored.prepare_incomplete_retry(cut, "g7"))
    assert retry.siblings[0].current_attempt.status is RolloutAttemptStatus.SEALED
    assert retry.siblings[1].current_attempt.status is RolloutAttemptStatus.RESERVED


@pytest.mark.parametrize(
    "recovery_granularity",
    [RecoveryGranularity.SIBLING, RecoveryGranularity.PROMPT_GROUP],
)
def test_missing_receipt_is_a_restart_safe_sealed_placeholder(
    recovery_granularity: RecoveryGranularity,
) -> None:
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
        task_source=None,
        recovery_granularity=recovery_granularity,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    gate_ids = group.gate_rollout_ids
    receipts = [
        None,
        {
            "rollout_id": gate_ids[1],
            "manifest": [{"staging_key": f"{gate_ids[1]}/call"}],
        },
    ]

    if recovery_granularity is RecoveryGranularity.SIBLING:
        for generation_index, receipt in enumerate(receipts):
            _mutate(
                lambda cut, generation_index=generation_index, receipt=receipt: (
                    ledger.mark_sibling_sealed(
                        cut,
                        "g7",
                        generation_index=generation_index,
                        gate_rollout_id=gate_ids[generation_index],
                        receipt=receipt,
                        reward=float(generation_index),
                        mask_sample=generation_index == 0,
                    )
                )
            )
    else:
        _mutate(
            lambda cut: ledger.mark_group_sealed(
                cut,
                "g7",
                {
                    generation_index: SiblingSealResult(
                        gate_rollout_id=gate_ids[generation_index],
                        receipt=receipt,
                        reward=float(generation_index),
                        mask_sample=generation_index == 0,
                    )
                    for generation_index, receipt in enumerate(receipts)
                },
            )
        )

    state = ledger.state_dict()
    restored = RolloutRecoveryLedger.from_state_dict(state)
    physical_ids, _, restored_receipts, rewards, mask_sample = (
        restored.finalization_inputs("g7")
    )

    assert physical_ids == gate_ids
    assert restored_receipts[0] is None
    assert restored_receipts[1] == receipts[1]
    assert rewards == [0.0, 1.0]
    assert mask_sample == [True, False]

    state["schema_version"] = ROLLOUT_RECOVERY_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="Unsupported rollout-recovery schema version"):
        RolloutRecoveryLedger.from_state_dict(state)


def test_prompt_group_restart_retries_every_sibling_when_one_is_unfinished() -> None:
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
        task_source="genrm_compare",
        recovery_granularity=RecoveryGranularity.PROMPT_GROUP,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))

    state = ledger.state_dict()
    assert state["groups"][0]["task_source"] == "genrm_compare"
    assert state["groups"][0]["recovery_granularity"] == "prompt_group"
    assert state["groups"][0]["restore_level"] == "prompt_group"

    restored = RolloutRecoveryLedger.from_state_dict(state)
    _mutate(lambda cut: restored.prepare_for_restart(cut))
    recovered = restored.get_group("g7")

    assert recovered.task_source == "genrm_compare"
    assert recovered.recovery_granularity is RecoveryGranularity.PROMPT_GROUP
    assert recovered.restore_level is RecoveryTargetLevel.PROMPT_GROUP
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


def test_turn_restore_target_is_persisted_separately_from_live_retry() -> None:
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
        recovery_granularity=RecoveryGranularity.SIBLING,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )

    state = ledger.state_dict()
    assert state["groups"][0]["recovery_granularity"] == "sibling"
    assert state["groups"][0]["restore_level"] == "turn"

    restored = RolloutRecoveryLedger.from_state_dict(state).get_group("g7")
    assert restored.recovery_granularity is RecoveryGranularity.SIBLING
    assert restored.restore_level is RecoveryTargetLevel.TURN


@pytest.mark.parametrize("schema_version", [2, 3])
def test_old_rollout_recovery_schemas_are_rejected(
    schema_version: int,
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
        recovery_granularity=RecoveryGranularity.SIBLING,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))

    state = ledger.state_dict()
    state["schema_version"] = schema_version

    with pytest.raises(ValueError, match="Unsupported rollout-recovery schema"):
        RolloutRecoveryLedger.from_state_dict(state)


def test_turn_restart_preserves_gym_owner_and_advances_attempt() -> None:
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
        recovery_granularity=RecoveryGranularity.SIBLING,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut,
            "g7",
            gym_instance_id="tools/replica-1",
        )
    )

    state = ledger.state_dict()
    attempt_state = state["groups"][0]["siblings"][0]["attempts"][0]
    assert attempt_state["gym_instance_id"] == "tools/replica-1"
    assert attempt_state["gym_attempt"] == 0

    restored = RolloutRecoveryLedger.from_state_dict(state)
    inventory = restored.gym_checkpoint_inventory(
        {"judge/replica-0", "tools/replica-1"}
    )
    ((rollout_id, attempt),) = inventory["tools/replica-1"]
    assert inventory["judge/replica-0"] == ()
    assert attempt == 0

    _mutate(
        lambda cut: restored.prepare_for_restart(
            cut,
            restored_gym_episodes={
                ("tools/replica-1", rollout_id, attempt),
            },
        )
    )
    recovered = restored.get_group("g7")
    current = recovered.siblings[0].current_attempt
    assert current.status is RolloutAttemptStatus.RESERVED
    assert current.gym_instance_id == "tools/replica-1"
    assert current.gym_attempt == 1
    assert recovered.gate_rollout_id(0) == f"{rollout_id}-a1"

    repeated_inventory = restored.gym_checkpoint_inventory(
        {"judge/replica-0", "tools/replica-1"}
    )
    assert repeated_inventory["tools/replica-1"] == ((rollout_id, 1),)

    _mutate(
        lambda cut: restored.prepare_for_restart(
            cut,
            restored_gym_episodes={
                ("tools/replica-1", rollout_id, 1),
            },
        )
    )
    recovered = restored.get_group("g7")
    current = recovered.siblings[0].current_attempt
    assert current.status is RolloutAttemptStatus.RESERVED
    assert current.gym_attempt == 2
    assert recovered.gate_rollout_id(0) == f"{rollout_id}-a2"

    with pytest.raises(ValueError, match="checkpoint owner"):
        _mutate(
            lambda cut: restored.mark_group_dispatched(
                cut,
                "g7",
                gym_instance_id="judge/replica-0",
            )
        )


def test_turn_restart_falls_back_to_sibling_when_gym_episode_is_absent() -> None:
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
        recovery_granularity=RecoveryGranularity.SIBLING,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut,
            "g7",
            gym_instance_id="tools/replica-0",
        )
    )

    _mutate(lambda cut: ledger.prepare_for_restart(cut))

    current = ledger.get_group("g7").siblings[0].current_attempt
    assert current.status is RolloutAttemptStatus.ABANDONED


def test_turn_dispatch_requires_a_gym_owner() -> None:
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
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )

    with pytest.raises(ValueError, match="requires a Gym checkpoint owner"):
        _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))


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
        task_source="genrm_compare",
        recovery_granularity=RecoveryGranularity.PROMPT_GROUP,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    results = {}
    for generation_index in range(2):
        gate_id = group.gate_rollout_id(generation_index)
        results[generation_index] = SiblingSealResult(
            gate_rollout_id=gate_id,
            receipt={
                "rollout_id": gate_id,
                "manifest": [{"staging_key": f"g7/sibling-{generation_index}/call-0"}],
            },
            reward=1.0,
            mask_sample=False,
        )
    _mutate(lambda cut: ledger.mark_group_sealed(cut, "g7", results))

    state = ledger.state_dict()
    restored = RolloutRecoveryLedger.from_state_dict(state)
    _bind(restored, "g7", _prompt())

    assert restored.state_dict() == state

    _mutate(lambda cut: restored.prepare_for_restart(cut))

    assert restored.get_group("g7").sealed_generation_indices == [0, 1]
    assert restored.expected_staging_keys() == {
        "g7/sibling-0/call-0",
        "g7/sibling-1/call-0",
    }


def test_prompt_group_seal_is_atomic() -> None:
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
        task_source="genrm_compare",
        recovery_granularity=RecoveryGranularity.PROMPT_GROUP,
        admitted=True,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))
    gate_id = group.gate_rollout_id(0)
    partial = {
        0: SiblingSealResult(
            gate_rollout_id=gate_id,
            receipt={
                "rollout_id": gate_id,
                "manifest": [{"staging_key": "g7/sibling-0/call-0"}],
            },
            reward=1.0,
            mask_sample=False,
        )
    }

    with pytest.raises(ValueError, match="requires every logical sibling"):
        _mutate(lambda cut: ledger.mark_group_sealed(cut, "g7", partial))

    assert [
        sibling.current_attempt.status for sibling in ledger.get_group("g7").siblings
    ] == [RolloutAttemptStatus.DISPATCHED, RolloutAttemptStatus.DISPATCHED]
    assert ledger.expected_staging_keys() == set()


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
        task_source=None,
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
            mask_sample=False,
        )
    )
    _mutate(lambda cut: ledger.mark_finalization_started(cut, "g7"))
    if unknown_outcome:
        _mutate(lambda cut: ledger.mark_finalization_unknown(cut, "g7"))

    with pytest.raises(RuntimeError, match="checkpoint-unsafe group states"):
        ledger.state_dict()


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("recovery_granularity", "banana", "invalid recovery_granularity"),
        ("recovery_granularity", None, "recovery_granularity must be a string"),
        ("restore_level", "banana", "invalid restore_level"),
        ("restore_level", None, "restore_level must be a string"),
        ("task_source", 123, "task_source must be a string or None"),
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
        task_source=None,
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


def test_restore_rejects_unsupported_schema_version() -> None:
    state = {
        "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION + 1,
        "groups": [],
    }

    with pytest.raises(ValueError, match="Unsupported rollout-recovery schema"):
        _load(RolloutRecoveryLedger(), state)  # type: ignore[arg-type]


def test_restore_rejects_non_list_groups() -> None:
    state = {
        "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
        "groups": {},
    }

    with pytest.raises(ValueError, match="must contain a groups list"):
        _load(RolloutRecoveryLedger(), state)  # type: ignore[arg-type]


def test_restore_rejects_invalid_prompt_group_phase() -> None:
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
        admitted=True,
    )
    state = ledger.state_dict()
    state["groups"][0]["phase"] = "unknown"

    with pytest.raises(ValueError, match="invalid prompt group phase"):
        _load(RolloutRecoveryLedger(), state)


def test_restore_rejects_inconsistent_shared_admission_state() -> None:
    ledger = RolloutRecoveryLedger()
    for idx in (7, 8):
        _reserve(
            ledger,
            group_id=f"g{idx}",
            admission_id="batch-7",
            prompt_id=str(idx),
            prompt_payload=_prompt(idx),
            expected_generations=1,
            target_step=7,
            start_weight_version=6,
            admitted=True,
        )
    state = ledger.state_dict()
    state["groups"][0]["target_step"] = None
    state["groups"][0]["phase"] = "reserved"

    with pytest.raises(ValueError, match="disagree on phase or target_step"):
        _load(RolloutRecoveryLedger(), state)


def _dispatched_turn_group(ledger: RolloutRecoveryLedger) -> PromptGroupRecoveryRecord:
    group = _reserve(
        ledger,
        group_id="g7",
        admission_id="batch-7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        recovery_granularity=RecoveryGranularity.SIBLING,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut, "g7", gym_instance_id="tools/replica-0"
        )
    )
    return group


@pytest.mark.parametrize(
    ("restore_level", "expected"),
    [(RecoveryTargetLevel.PREFIX, True), (RecoveryTargetLevel.TURN, False)],
)
def test_only_a_dispatched_prefix_attempt_grows_a_recoverable_prefix(
    restore_level: RecoveryTargetLevel, expected: bool
) -> None:
    """Decoding advances only dispatched prefix-level attempts between mutations."""
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
        restore_level=restore_level,
        admitted=True,
    )
    assert not ledger.has_dispatched_generation_prefix()

    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut, "g7", gym_instance_id="tools/replica-0"
        )
    )
    assert ledger.has_dispatched_generation_prefix() is expected

    _mutate(lambda cut: ledger.abandon_unsealed(cut, "g7"))
    assert not ledger.has_dispatched_generation_prefix()


def test_refused_fresh_dispatch_keeps_identity_and_leaves_gym_inventory() -> None:
    """Gym recorded nothing for a refused /run, so a checkpoint must not wait on it."""
    ledger = RolloutRecoveryLedger()
    group = _dispatched_turn_group(ledger)
    refused_attempt_id = group.siblings[1].current_attempt.attempt_id
    refused_rollout_id = group.gate_rollout_id(1)
    assert (
        len(ledger.gym_checkpoint_inventory({"tools/replica-0"})["tools/replica-0"])
        == 2
    )

    _mutate(lambda cut: ledger.release_refused_dispatch(cut, "g7", generation_index=1))

    released = ledger.get_group("g7")
    attempt = released.siblings[1].current_attempt
    assert attempt.status is RolloutAttemptStatus.RESERVED
    assert attempt.attempt_id == refused_attempt_id
    assert released.gate_rollout_id(1) == refused_rollout_id
    assert (
        released.siblings[0].current_attempt.status is RolloutAttemptStatus.DISPATCHED
    )
    (remaining,) = ledger.gym_checkpoint_inventory({"tools/replica-0"})[
        "tools/replica-0"
    ]
    assert remaining == group.gym_episode(0)

    # The same attempt is re-sent once admission reopens.
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut, "g7", generation_indices=[1], gym_instance_id="tools/replica-0"
        )
    )
    assert (
        ledger.get_group("g7").siblings[1].current_attempt.status
        is RolloutAttemptStatus.DISPATCHED
    )


def test_refused_restored_turn_stays_gym_owned() -> None:
    """A restored turn waiting to be re-sent must keep its committed continuation."""
    ledger = RolloutRecoveryLedger()
    _dispatched_turn_group(ledger)
    restored = RolloutRecoveryLedger.from_state_dict(ledger.state_dict())
    episodes = restored.gym_checkpoint_inventory({"tools/replica-0"})["tools/replica-0"]
    _mutate(
        lambda cut: restored.prepare_for_restart(
            cut,
            restored_gym_episodes={
                ("tools/replica-0", *episode) for episode in episodes
            },
        )
    )
    _mutate(
        lambda cut: restored.mark_group_dispatched(
            cut, "g7", gym_instance_id="tools/replica-0"
        )
    )

    _mutate(
        lambda cut: restored.release_refused_dispatch(cut, "g7", generation_index=1)
    )

    attempt = restored.get_group("g7").siblings[1].current_attempt
    assert attempt.status is RolloutAttemptStatus.RESERVED
    assert attempt.gym_attempt == 1
    assert (
        restored.get_group("g7").gym_episode(1)
        in restored.gym_checkpoint_inventory({"tools/replica-0"})["tools/replica-0"]
    )


def test_refusal_requires_a_dispatched_attempt() -> None:
    ledger = RolloutRecoveryLedger()
    _dispatched_turn_group(ledger)
    _mutate(lambda cut: ledger.release_refused_dispatch(cut, "g7", generation_index=0))

    with pytest.raises(ValueError, match="only a dispatched rollout attempt"):
        _mutate(
            lambda cut: ledger.release_refused_dispatch(cut, "g7", generation_index=0)
        )


def test_dropping_a_dispatched_group_records_its_gym_episodes_for_retirement() -> None:
    """Gym keeps running what it admitted, so the next checkpoint must retire it."""
    abandoned_ledger = RolloutRecoveryLedger()
    abandoned = _dispatched_turn_group(abandoned_ledger)
    _mutate(lambda cut: abandoned_ledger.abandon_unsealed(cut, "g7"))

    discarded_ledger = RolloutRecoveryLedger()
    discarded = _dispatched_turn_group(discarded_ledger)
    _mutate(lambda cut: discarded_ledger.discard_group(cut, "g7"))

    for ledger, group in ((abandoned_ledger, abandoned), (discarded_ledger, discarded)):
        assert ledger.gym_checkpoint_retirements({"tools/replica-0"}) == {
            "tools/replica-0": tuple(
                sorted(group.gym_episode(index) for index in (0, 1))
            )
        }
    # Bookkeeping for this process's Gym executions, never persisted.
    reloaded = RolloutRecoveryLedger.from_state_dict(abandoned_ledger.state_dict())
    assert reloaded.gym_checkpoint_retirements({"tools/replica-0"}) == {
        "tools/replica-0": ()
    }


def test_attempts_that_never_reached_gym_are_not_retired() -> None:
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
        recovery_granularity=RecoveryGranularity.SIBLING,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )
    _mutate(lambda cut: ledger.abandon_unsealed(cut, "g7"))

    assert ledger.gym_checkpoint_retirements({"tools/replica-0"}) == {
        "tools/replica-0": ()
    }


def test_dropping_a_restored_reserved_attempt_retires_its_restored_gym_state() -> None:
    ledger = RolloutRecoveryLedger()
    _dispatched_turn_group(ledger)
    restored = RolloutRecoveryLedger.from_state_dict(ledger.state_dict())
    episodes = restored.gym_checkpoint_inventory({"tools/replica-0"})["tools/replica-0"]
    _mutate(
        lambda cut: restored.prepare_for_restart(
            cut,
            restored_gym_episodes={
                ("tools/replica-0", *episode) for episode in episodes
            },
        )
    )
    group = restored.get_group("g7")

    _mutate(lambda cut: restored.discard_group(cut, "g7"))

    retired = restored.gym_checkpoint_retirements({"tools/replica-0"})[
        "tools/replica-0"
    ]
    assert set(retired) == {group.gym_episode(index) for index in (0, 1)}
    assert {attempt for _, attempt in retired} == {1}


def test_mark_gym_retired_keeps_drops_recorded_after_the_retire_was_sent() -> None:
    ledger = RolloutRecoveryLedger()
    first = _dispatched_turn_group(ledger)
    _mutate(lambda cut: ledger.abandon_unsealed(cut, "g7"))
    sent = ledger.gym_checkpoint_retirements({"tools/replica-0"})

    later = _mutate(
        lambda cut: ledger.reserve_group(
            cut,
            group_id="g8",
            admission_id="batch-8",
            prompt_id="8",
            prompt_payload=_prompt(8),
            expected_generations=1,
            target_step=8,
            start_weight_version=7,
            restore_level=RecoveryTargetLevel.TURN,
            admitted=True,
        )
    )
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut, "g8", gym_instance_id="tools/replica-0"
        )
    )
    later = ledger.get_group("g8")
    _mutate(lambda cut: ledger.abandon_unsealed(cut, "g8"))

    ledger.mark_gym_retired(sent)

    assert ledger.gym_checkpoint_retirements({"tools/replica-0"}) == {
        "tools/replica-0": (later.gym_episode(0),)
    }
    assert (
        first.gym_episode(0)
        not in ledger.gym_checkpoint_retirements({"tools/replica-0"})["tools/replica-0"]
    )


def test_a_dropped_dispatch_is_queued_for_retirement_and_replaced() -> None:
    """A broken stream leaves the row running in Gym; its re-send needs a new ID."""
    ledger = RolloutRecoveryLedger()
    group = _dispatched_turn_group(ledger)

    dropped = _mutate(
        lambda cut: ledger.abandon_dropped_dispatch(
            cut, "g7", generation_indices=[1], replace=True
        )
    )

    assert dropped == (group.gym_episode(1),)
    assert ledger.gym_checkpoint_retirements({"tools/replica-0"}) == {
        "tools/replica-0": (group.gym_episode(1),)
    }
    replaced = ledger.get_group("g7")
    assert replaced.siblings[1].attempts[0].status is RolloutAttemptStatus.ABANDONED
    assert replaced.siblings[1].current_attempt.status is RolloutAttemptStatus.RESERVED
    assert replaced.gate_rollout_id(1) != group.gate_rollout_id(1)
    # The sibling that returned keeps its attempt and is not Gym-retired.
    assert replaced.gate_rollout_id(0) == group.gate_rollout_id(0)
    assert ledger.gym_checkpoint_inventory({"tools/replica-0"}) == {
        "tools/replica-0": (group.gym_episode(0),)
    }


def test_a_dropped_attempt_that_never_reached_gym_keeps_its_identity() -> None:
    ledger = RolloutRecoveryLedger()
    group = _dispatched_turn_group(ledger)
    _mutate(lambda cut: ledger.release_refused_dispatch(cut, "g7", generation_index=1))

    dropped = _mutate(
        lambda cut: ledger.abandon_dropped_dispatch(
            cut, "g7", generation_indices=[1], replace=True
        )
    )

    assert dropped == ()
    assert ledger.get_group("g7").gate_rollout_id(1) == group.gate_rollout_id(1)


def test_a_prompt_group_cannot_replace_one_dropped_sibling() -> None:
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
        recovery_granularity=RecoveryGranularity.PROMPT_GROUP,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut, "g7", gym_instance_id="tools/replica-0"
        )
    )

    with pytest.raises(ValueError, match="replaces the whole cohort"):
        _mutate(
            lambda cut: ledger.abandon_dropped_dispatch(
                cut, "g7", generation_indices=[1], replace=True
            )
        )
