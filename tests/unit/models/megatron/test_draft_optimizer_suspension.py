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

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.models.megatron.draft.optimizer import suspend_draft_optimizer_groups


class _MutationTrackingGroups(list[dict[str, object]]):
    def __init__(self, groups: list[dict[str, object]]) -> None:
        super().__init__(groups)
        self.slice_mutations = 0

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            self.slice_mutations += 1
        super().__setitem__(key, value)


def _draft_parameter(value: float = 1.0) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.tensor([value]))
    parameter.grad_norm_group = "draft"
    return parameter


def _clone_state(state: dict[str, object]) -> dict[str, object]:
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def test_skip_preserves_draft_bytes_moments_and_step_while_scheduler_advances() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = _draft_parameter()
    optimizer = torch.optim.AdamW(
        [{"params": [policy]}, {"params": [draft]}],
        lr=0.1,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    policy.grad = torch.ones_like(policy)
    draft.grad = torch.ones_like(draft)
    optimizer.step()
    draft_before = draft.detach().clone()
    draft_state_before = _clone_state(optimizer.state[draft])
    groups_before = list(optimizer.param_groups)

    policy.grad = torch.ones_like(policy)
    draft.grad = torch.ones_like(draft)
    policy_before = policy.detach().clone()
    with suspend_draft_optimizer_groups(optimizer):
        optimizer.step()
    scheduler.step()

    assert not torch.equal(policy, policy_before)
    assert torch.equal(draft, draft_before)
    assert optimizer.param_groups == groups_before
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.05)
    for key, expected in draft_state_before.items():
        actual = optimizer.state[draft][key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected


def test_restores_groups_when_optimizer_step_raises() -> None:
    policy = torch.nn.Parameter(torch.tensor([1.0]))
    draft = _draft_parameter()
    optimizer = torch.optim.SGD([{"params": [policy]}, {"params": [draft]}])
    groups_before = list(optimizer.param_groups)

    with pytest.raises(RuntimeError, match="step failed"):
        with suspend_draft_optimizer_groups(optimizer):
            assert optimizer.param_groups == [groups_before[0]]
            raise RuntimeError("step failed")

    assert optimizer.param_groups == groups_before


def test_mcore_wrapper_and_chained_optimizers_suspend_each_draft_group() -> None:
    first_policy = torch.nn.Parameter(torch.tensor([1.0]))
    first_draft = _draft_parameter()
    second_policy = torch.nn.Parameter(torch.tensor([2.0]))
    second_draft = _draft_parameter(2.0)
    first = torch.optim.SGD(
        [{"params": [first_policy]}, {"params": [first_draft]}], lr=0.1
    )
    second = torch.optim.SGD(
        [{"params": [second_policy]}, {"params": [second_draft]}], lr=0.1
    )
    first_groups = list(first.param_groups)
    second_groups = list(second.param_groups)
    chained = SimpleNamespace(
        chained_optimizers=(
            SimpleNamespace(optimizer=first),
            SimpleNamespace(optimizer=second),
        )
    )

    with suspend_draft_optimizer_groups(chained):
        assert first.param_groups == [first_groups[0]]
        assert second.param_groups == [second_groups[0]]

    assert first.param_groups == first_groups
    assert second.param_groups == second_groups


def test_mixed_group_in_later_chained_optimizer_fails_before_any_mutation() -> None:
    first_policy = torch.nn.Parameter(torch.tensor([1.0]))
    first_draft = _draft_parameter()
    mixed_policy = torch.nn.Parameter(torch.tensor([2.0]))
    mixed_draft = _draft_parameter(2.0)
    first = torch.optim.SGD(
        [{"params": [first_policy]}, {"params": [first_draft]}], lr=0.1
    )
    second = torch.optim.SGD([{"params": [mixed_policy, mixed_draft]}], lr=0.1)
    first.param_groups = _MutationTrackingGroups(first.param_groups)
    second.param_groups = _MutationTrackingGroups(second.param_groups)
    first_groups = list(first.param_groups)
    second_groups = list(second.param_groups)
    chained = SimpleNamespace(
        chained_optimizers=(
            SimpleNamespace(optimizer=first),
            SimpleNamespace(optimizer=second),
        )
    )

    with pytest.raises(RuntimeError, match="mixes policy and draft"):
        with suspend_draft_optimizer_groups(chained):
            raise AssertionError("mixed groups must fail on context entry")

    assert first.param_groups == first_groups
    assert second.param_groups == second_groups
    assert first.param_groups.slice_mutations == 0
    assert second.param_groups.slice_mutations == 0
