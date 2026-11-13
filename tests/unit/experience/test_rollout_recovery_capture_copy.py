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

"""Recovery capture ownership, format parity, and early-filter regressions."""

import asyncio
import copy
import dataclasses
import io
import itertools
from collections import defaultdict
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.experience import rollout_recovery as recovery
from nemo_rl.experience.reward_penalties import (
    FinalizedReward,
    RewardChecks,
    RewardLogContext,
)
from nemo_rl.experience.rollout_reassembler import ActionOutputFlags, RolloutSelection


@pytest.mark.parametrize(
    "value", [None, True, False, 0, -3, 1.25, 2j, "text", b"bytes"]
)
def test_copy_reuses_immutable_leaves(value: Any) -> None:
    assert recovery._copy_checkpoint_value(value) is value


def test_copy_preserves_shared_containers_and_cycles() -> None:
    shared = {"values": [1, 2]}
    source: dict[str, Any] = {"left": shared, "right": shared}
    source["cycle"] = source
    cloned = recovery._copy_checkpoint_value(source)
    assert cloned is not source
    assert cloned["left"] is cloned["right"]
    assert cloned["cycle"] is cloned
    assert cloned["left"] is not shared
    cloned["left"]["values"].append(3)
    assert shared["values"] == [1, 2]


def test_fallback_preserves_aliases_with_fast_containers() -> None:
    shared = [1]
    custom = defaultdict(list, {"shared": shared})
    source = {"first": shared, "fallback": custom, "tuple": (shared,)}
    cloned = recovery._copy_checkpoint_value(source)
    assert cloned == copy.deepcopy(source)
    assert type(cloned["fallback"]) is defaultdict
    assert cloned["fallback"].default_factory is list
    assert cloned["fallback"]["shared"] is cloned["first"]
    assert cloned["tuple"][0] is cloned["first"]
    assert cloned["first"] is not shared


def test_custom_scalar_copy_behavior_is_preserved() -> None:
    class CopyAwareInt(int):
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            return {"custom": int(self)}

    value = CopyAwareInt(7)
    assert recovery._copy_checkpoint_value({"value": value}) == {"value": {"custom": 7}}


def test_deepcopy_failures_propagate() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            raise TypeError("cannot copy receipt extension")

    with pytest.raises(TypeError, match="cannot copy receipt extension"):
        recovery._copy_checkpoint_value({"value": Uncopyable()})


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_checks_projection_matches_dataclass_schema(flags: tuple[bool, ...]) -> None:
    checks = RewardChecks(*flags)
    assert recovery._reward_checks_state(checks) == dataclasses.asdict(checks)


@pytest.mark.parametrize("full_result_json", [None, '{"answer":"ok"}'])
def test_reward_projection_matches_dataclass_schema(
    full_result_json: str | None,
) -> None:
    context = RewardLogContext("agent", full_result_json)
    assert recovery._reward_log_context_state(context) == dataclasses.asdict(context)
    for log_context in (None, context):
        reward = FinalizedReward("sample", "rollout", 1.5, log_context)
        state = recovery._finalized_reward_state(reward)
        assert state == dataclasses.asdict(reward)
        if state["log_context"] is not None:
            state["log_context"]["agent_name"] = "changed"
            assert context.agent_name == "agent"


def test_projection_keeps_dataclass_subclass_fields() -> None:
    @dataclasses.dataclass(frozen=True)
    class ExtendedContext(RewardLogContext):
        extra: tuple[list[int]] = dataclasses.field(default_factory=lambda: ([1],))

    @dataclasses.dataclass(frozen=True)
    class ExtendedChecks(RewardChecks):
        extra: list[int] = dataclasses.field(default_factory=lambda: [2])

    @dataclasses.dataclass(frozen=True)
    class ExtendedReward(FinalizedReward):
        extra: list[int] = dataclasses.field(default_factory=lambda: [3])

    context = ExtendedContext("agent", None)
    checks = ExtendedChecks(False, True, False)
    reward = ExtendedReward("sample", "rollout", 0.0, context)
    for value, project in (
        (context, recovery._reward_log_context_state),
        (checks, recovery._reward_checks_state),
        (reward, recovery._finalized_reward_state),
    ):
        assert project(value) == dataclasses.asdict(value)
    state = recovery._reward_log_context_state(context)
    state["extra"][0].append(2)
    assert context.extra == ([1],)
    assert recovery._finalized_reward_state(
        FinalizedReward("sample", "rollout", 0.0, context)
    ) == dataclasses.asdict(FinalizedReward("sample", "rollout", 0.0, context))


def test_projection_detaches_mutable_string_subclass_attributes() -> None:
    class MutableString(str):
        def __new__(cls, value: str) -> Any:
            instance = super().__new__(cls, value)
            instance.notes = [1]
            return instance

    context = RewardLogContext(MutableString("agent"), MutableString('{"ok":true}'))
    reward = FinalizedReward(
        MutableString("sample"), MutableString("rollout"), 1.0, context
    )
    state = recovery._finalized_reward_state(reward)
    assert state == dataclasses.asdict(reward)
    for value in (
        state["sample_id"],
        state["rollout_id"],
        *state["log_context"].values(),
    ):
        assert type(value) is MutableString
        value.notes.append(2)
    assert reward.sample_id.notes == [1]
    assert reward.rollout_id.notes == [1]
    assert context.agent_name.notes == [1]
    assert context.full_result_json.notes == [1]


def _populated_ledger() -> recovery.RolloutRecoveryLedger:
    ledger = recovery.RolloutRecoveryLedger()

    async def populate() -> None:
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            for idx, name in enumerate(("canonical", "pending")):
                group = ledger.reserve_group(
                    cut,
                    group_id=name,
                    admission_id=name,
                    prompt_id=str(idx),
                    prompt_payload={"idx": idx},
                    expected_generations=2,
                    target_step=1,
                    start_weight_version=0,
                    task_source="taskset",
                    restore_level=recovery.RecoveryTargetLevel.SIBLING,
                    admitted=True,
                )
                ledger.mark_group_dispatched(cut, name)
                owner = group.gate_rollout_id(0)
                ledger.mark_sibling_sealed(
                    cut,
                    name,
                    generation_index=0,
                    gate_rollout_id=owner,
                    receipt={
                        "capture_key": owner,
                        "manifest": [
                            {
                                "model_call_id": "call0",
                                "staging_key": f"{owner}/call0",
                                "metadata": {"sizes": [4]},
                            }
                        ],
                    },
                    reward=1.0,
                    mask_sample=False,
                    resolved_agent_name="agent",
                    attribution_key="taskset",
                    reward_checks=RewardChecks(True, False, True),
                    reward_log_context=RewardLogContext("taskset", '{"answer":"ok"}'),
                    logical_selection=RolloutSelection(
                        ("response0",),
                        (ActionOutputFlags(False, True),),
                        truncated=True,
                    ),
                )

    asyncio.run(populate())
    return ledger


@pytest.mark.parametrize("canonical_ids", [None, set(), {"canonical"}])
def test_capture_matches_legacy_projection(
    canonical_ids: set[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _populated_ledger()
    rewards = {
        name: [FinalizedReward(f"sample-{name}", f"rollout-{name}", 1.0, None)]
        for name in ("canonical", "pending")
    }
    original = ledger.state_dict()
    state = recovery.build_rollout_recovery_state(
        ledger,
        batch_shortfall={1: 2},
        steps_allowed_short={3, 5},
        sampler_stamps_target_steps=True,
        finalizer_metrics_by_group={name: {"metric": 1.0} for name in rewards},
        finalizer_rewards_by_group=rewards,
        canonical_group_ids=canonical_ids,
    )
    with monkeypatch.context() as legacy:
        legacy.setattr(recovery, "_copy_checkpoint_value", copy.deepcopy)
        legacy.setattr(recovery, "_reward_checks_state", dataclasses.asdict)
        legacy.setattr(recovery, "_reward_log_context_state", dataclasses.asdict)
        legacy.setattr(recovery, "_finalized_reward_state", dataclasses.asdict)
        expected = ledger.state_dict()
    expected["groups"] = [
        group
        for group in expected["groups"]
        if canonical_ids is None or group["group_id"] not in canonical_ids
    ]
    expected.update(
        batch_shortfall={1: 2},
        steps_allowed_short=[3, 5],
        sampler_stamps_target_steps=True,
        reward_settings=None,
        finalizer_metrics_by_group={
            name: {"metric": 1.0}
            for name in rewards
            if canonical_ids is None or name in canonical_ids
        },
        finalizer_rewards_by_group={
            name: [dataclasses.asdict(row) for row in rows]
            for name, rows in rewards.items()
            if canonical_ids is None or name in canonical_ids
        },
    )
    assert state == expected
    payload = io.BytesIO()
    torch.save(state, payload)
    loaded = torch.load(io.BytesIO(payload.getvalue()), weights_only=True)
    assert loaded == expected
    parsed = recovery.parse_rollout_recovery_state(loaded)
    restored = recovery.RolloutRecoveryLedger.from_state_dict(parsed.ledger_state)
    assert len(restored) == len(state["groups"])
    for group in state["groups"]:
        attempt = group["siblings"][0]["attempts"][0]
        restored_attempt = (
            restored.get_group(group["group_id"]).siblings[0].current_attempt
        )
        assert restored_attempt.receipt == attempt["receipt"]
        assert (
            dataclasses.asdict(restored_attempt.logical_selection)
            == attempt["logical_selection"]
        )
        attempt["receipt"]["manifest"][0]["metadata"]["sizes"].append(9)
        attempt["staging_keys"].append("other")
        attempt["reward_checks"]["low_effort"] = False
        attempt["reward_log_context"]["agent_name"] = "changed"
    assert ledger.state_dict() == original


def test_canonical_groups_are_filtered_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _populated_ledger()
    original = ledger.state_dict()
    skipped = ledger._groups["canonical"].siblings[0].current_attempt.receipt
    copy_value = recovery._copy_checkpoint_value

    def guarded(value: Any, memo: dict[int, Any] | None = None) -> Any:
        assert value is not skipped, (
            "Canonical receipt must not be copied and discarded"
        )
        return copy_value(value, memo)

    monkeypatch.setattr(recovery, "_copy_checkpoint_value", guarded)
    state = recovery.build_rollout_recovery_state(
        ledger,
        batch_shortfall={},
        steps_allowed_short=set(),
        sampler_stamps_target_steps=False,
        canonical_group_ids={"canonical"},
    )
    assert state["groups"] == [
        group for group in original["groups"] if group["group_id"] == "pending"
    ]
    ledger._groups["canonical"].status = recovery.PromptGroupStatus.FINALIZING
    with pytest.raises(RuntimeError, match="checkpoint-unsafe"):
        ledger.state_dict(exclude_group_ids={"canonical"})


def test_exclusion_keeps_prompt_identity_validation() -> None:
    ledger = _populated_ledger()
    ledger._groups["canonical"].runtime_prompt_payload = None
    with pytest.raises(RuntimeError, match="prompt is rehydrated"):
        ledger.state_dict(exclude_group_ids={"canonical"})


def test_discarded_finalizer_rewards_are_not_projected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = FinalizedReward("kept", "rollout-kept", 1.0, None)
    discarded = FinalizedReward("discarded", "rollout-discarded", 2.0, None)
    project = recovery._finalized_reward_state

    def guarded(reward: FinalizedReward) -> dict[str, Any]:
        assert reward is not discarded, "Discarded reward must not be projected"
        return project(reward)

    monkeypatch.setattr(recovery, "_finalized_reward_state", guarded)
    state = recovery.build_rollout_recovery_state(
        recovery.RolloutRecoveryLedger(),
        batch_shortfall={},
        steps_allowed_short=set(),
        sampler_stamps_target_steps=False,
        finalizer_metrics_by_group={"canonical": {"metric": 1.0}},
        finalizer_rewards_by_group={"canonical": [kept], "discarded": [discarded]},
        canonical_group_ids={"canonical"},
    )
    assert state["finalizer_rewards_by_group"] == {
        "canonical": [dataclasses.asdict(kept)]
    }
