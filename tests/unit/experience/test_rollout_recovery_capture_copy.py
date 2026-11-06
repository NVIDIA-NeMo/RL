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

"""Checkpoint-copy parity, ownership and restore regressions."""

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
from nemo_rl.experience.reward_penalties import (
    FinalizedReward,
    RewardChecks,
    RewardLogContext,
)
from nemo_rl.experience.rollout_recovery import (
    RecoveryTargetLevel,
    RolloutRecoveryLedger,
    _copy_checkpoint_value,
    _finalized_reward_state,
    _reward_checks_state,
    _reward_log_context_state,
    build_rollout_recovery_state,
    parse_rollout_recovery_state,
)


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, -3, 1.25, 2j, "text", b"bytes"],
)
def test_checkpoint_copy_reuses_immutable_leaves(value: Any) -> None:
    assert _copy_checkpoint_value(value) is value


def test_checkpoint_copy_preserves_shared_containers_and_cycles() -> None:
    shared = {"values": [1, 2]}
    source: dict[str, Any] = {"left": shared, "right": shared}
    source["cycle"] = source
    cloned = _copy_checkpoint_value(source)
    assert cloned is not source
    assert cloned["left"] is cloned["right"]
    assert cloned["cycle"] is cloned
    assert cloned["left"] is not shared
    cloned["left"]["values"].append(3)
    assert shared["values"] == [1, 2]


def test_checkpoint_copy_fallback_preserves_aliases_with_fast_containers() -> None:
    shared = [1]
    custom = defaultdict(list, {"shared": shared})
    source = {"first": shared, "fallback": custom, "tuple": (shared,)}
    cloned = _copy_checkpoint_value(source)
    assert cloned == copy.deepcopy(source)
    assert type(cloned["fallback"]) is defaultdict
    assert cloned["fallback"].default_factory is list
    assert cloned["fallback"]["shared"] is cloned["first"]
    assert cloned["tuple"][0] is cloned["first"]
    assert cloned["first"] is not shared


def test_checkpoint_copy_preserves_custom_scalar_copy_behavior() -> None:
    class CopyAwareInt(int):
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            return {"custom": int(self)}

    value = CopyAwareInt(7)
    assert _copy_checkpoint_value({"value": value}) == {"value": {"custom": 7}}


def test_checkpoint_copy_propagates_deepcopy_failures() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            raise TypeError("cannot copy receipt extension")

    with pytest.raises(TypeError, match="cannot copy receipt extension"):
        _copy_checkpoint_value({"value": Uncopyable()})


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_reward_checks_projection_matches_dataclass_schema(
    flags: tuple[bool, ...],
) -> None:
    checks = RewardChecks(*flags)
    assert _reward_checks_state(checks) == dataclasses.asdict(checks)


@pytest.mark.parametrize("full_result_json", [None, '{"result":{"answer":"ok"}}'])
def test_reward_projection_matches_dataclass_schema(
    full_result_json: str | None,
) -> None:
    context = RewardLogContext("agent", full_result_json)
    assert _reward_log_context_state(context) == dataclasses.asdict(context)
    for log_context in (None, context):
        reward = FinalizedReward("sample", "rollout", 1.5, log_context)
        state = _finalized_reward_state(reward)
        assert state == dataclasses.asdict(reward)
        if state["log_context"] is not None:
            state["log_context"]["agent_name"] = "changed"
            assert context.agent_name == "agent"


def test_reward_projection_keeps_dataclass_subclass_fields() -> None:
    @dataclasses.dataclass(frozen=True)
    class ExtendedContext(RewardLogContext):
        extra: tuple[list[int]] = dataclasses.field(default_factory=lambda: ([1],))

    context = ExtendedContext("agent", None)
    state = _reward_log_context_state(context)
    assert state == dataclasses.asdict(context)
    state["extra"][0].append(2)
    assert context.extra == ([1],)
    reward = FinalizedReward("sample", "rollout", 0.0, context)
    assert _finalized_reward_state(reward) == dataclasses.asdict(reward)


def test_reward_projection_detaches_mutable_string_subclass_attributes() -> None:
    class MutableString(str):
        def __new__(cls, value: str) -> Any:
            instance = super().__new__(cls, value)
            instance.notes = [1]
            return instance

    context = RewardLogContext(MutableString("agent"), MutableString('{"ok":true}'))
    reward = FinalizedReward(
        MutableString("sample"), MutableString("rollout"), 1.0, context
    )
    state = _finalized_reward_state(reward)
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


def test_checkpoint_state_is_independent_and_restores_after_serialization() -> None:
    ledger = RolloutRecoveryLedger()
    checks = RewardChecks(True, False, True)
    context = RewardLogContext("agent", '{"answer":"ok"}')

    async def populate() -> None:
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            group = ledger.reserve_group(
                cut,
                group_id="group",
                admission_id="admission",
                prompt_id="7",
                prompt_payload={"idx": 7},
                expected_generations=2,
                target_step=1,
                start_weight_version=0,
                task_source=None,
                restore_level=RecoveryTargetLevel.SIBLING,
                admitted=True,
            )
            ledger.mark_group_dispatched(cut, "group")
            rollout_id = group.gate_rollout_ids[0]
            ledger.mark_sibling_sealed(
                cut,
                "group",
                generation_index=0,
                gate_rollout_id=rollout_id,
                receipt={
                    "rollout_id": rollout_id,
                    "manifest": [{"staging_key": "row", "metadata": {"sizes": [4]}}],
                },
                reward=1.0,
                mask_sample=False,
                reward_checks=checks,
                reward_log_context=context,
            )

    asyncio.run(populate())
    reward = FinalizedReward("sample", "rollout", 1.0, context)
    state = build_rollout_recovery_state(
        ledger,
        batch_shortfall={1: 2},
        sampler_stamps_target_steps=True,
        finalizer_metrics_by_group={"canonical": {"metric": 1.0}},
        finalizer_rewards_by_group={"canonical": [reward]},
        canonical_group_ids={"canonical"},
    )
    expected_ledger = copy.deepcopy(ledger.state_dict())
    buffer = io.BytesIO()
    torch.save(state, buffer)
    loaded = torch.load(io.BytesIO(buffer.getvalue()), weights_only=False)
    assert loaded == state
    parsed = parse_rollout_recovery_state(loaded)
    assert parsed.finalizer_rewards_by_group == {"canonical": [reward]}
    restored = RolloutRecoveryLedger.from_state_dict(parsed.ledger_state)
    assert (
        restored.get_group("group").siblings[0].current_attempt.receipt
        == (expected_ledger["groups"][0]["siblings"][0]["attempts"][0]["receipt"])
    )
    attempt = state["groups"][0]["siblings"][0]["attempts"][0]
    assert attempt["reward_checks"] == dataclasses.asdict(checks)
    assert attempt["reward_log_context"] == dataclasses.asdict(context)
    attempt["receipt"]["manifest"][0]["metadata"]["sizes"].append(9)
    attempt["staging_keys"].append("other")
    attempt["reward_checks"]["low_effort"] = False
    attempt["reward_log_context"]["agent_name"] = "changed"
    assert ledger.state_dict() == expected_ledger


def test_canonical_group_receipts_are_filtered_before_copy(monkeypatch) -> None:
    from nemo_rl.experience import rollout_recovery as recovery

    ledger = RolloutRecoveryLedger()

    async def populate():
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            for idx, name in enumerate(("canonical", "pending")):
                group = ledger.reserve_group(
                    cut,
                    group_id=name,
                    admission_id=name,
                    prompt_id=str(idx),
                    prompt_payload={"idx": idx},
                    expected_generations=1,
                    target_step=1,
                    start_weight_version=0,
                    task_source=None,
                    restore_level=RecoveryTargetLevel.SIBLING,
                    admitted=True,
                )
                ledger.mark_group_dispatched(cut, name)
                rollout_id = group.gate_rollout_ids[0]
                ledger.mark_sibling_sealed(
                    cut,
                    name,
                    generation_index=0,
                    gate_rollout_id=rollout_id,
                    receipt={"rollout_id": rollout_id, "manifest": []},
                    reward=1.0,
                    mask_sample=False,
                )

    asyncio.run(populate())
    original = ledger.state_dict()
    skipped = ledger._groups["canonical"].siblings[0].current_attempt.receipt
    copy_value = recovery._copy_checkpoint_value

    def guarded(value, memo=None):
        assert value is not skipped, (
            "Canonical receipt must not be constructed and discarded"
        )
        return copy_value(value, memo)

    monkeypatch.setattr(recovery, "_copy_checkpoint_value", guarded)
    state = build_rollout_recovery_state(
        ledger,
        batch_shortfall={},
        sampler_stamps_target_steps=False,
        canonical_group_ids={"canonical"},
    )
    assert state["groups"] == [
        group for group in original["groups"] if group["group_id"] == "pending"
    ]
    parsed = parse_rollout_recovery_state(state)
    rebuilt = RolloutRecoveryLedger.from_state_dict(parsed.ledger_state)
    assert len(rebuilt) == 1
    rebuilt._groups["pending"].runtime_prompt_payload = {"idx": 1}
    assert rebuilt.state_dict()["groups"] == state["groups"]
    # Exclusion must retain the global checkpoint-safety invariant.
    ledger._groups["canonical"].status = recovery.PromptGroupStatus.FINALIZING
    with pytest.raises(RuntimeError, match="checkpoint-unsafe"):
        ledger.state_dict(exclude_group_ids={"canonical"})
