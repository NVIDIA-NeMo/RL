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
from typing import Any, Callable

import pytest
import torch
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

from nemo_rl.models.megatron.draft.optimizer import suspend_draft_optimizer_groups


pytestmark = pytest.mark.mcore


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


def _run_mcore_master_parameter_skip(rank: int, world_size: int) -> None:
    from megatron.core.optimizer import optimizer as mcore_optimizer

    from nemo_rl.models.megatron.draft.utils import register_draft_grad_norm_group

    assert world_size == 1
    original_grad_norm_groups = mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS
    register_draft_grad_norm_group()
    device = torch.device("cuda", rank)
    policy = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
    )
    draft = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16, device=device))
    draft.grad_norm_group = "draft"
    base_optimizer = torch.optim.AdamW(
        [{"params": [policy]}, {"params": [draft]}],
        lr=0.1,
        weight_decay=0.1,
    )
    wrapped_optimizer = Float16OptimizerWithFloat16Params(
        base_optimizer,
        OptimizerConfig(
            optimizer="adam",
            lr=0.1,
            bf16=True,
            clip_grad=0.0,
        ),
        None,
        lambda optimizer, config: None,
    )
    wrapped_optimizer.grad_stats_parallel_group = torch.distributed.group.WORLD
    wrapped_optimizer.tp_group = torch.distributed.group.WORLD
    optimizer = ChainedOptimizer([wrapped_optimizer])

    try:
        policy.grad = torch.ones_like(policy)
        draft.grad = torch.ones_like(draft)
        update_successful, _, _ = optimizer.step()
        assert update_successful is True

        policy_main = wrapped_optimizer.fp32_from_float16_groups[0][0]
        draft_main = wrapped_optimizer.fp32_from_float16_groups[1][0]
        policy_model_before = policy.detach().clone()
        policy_before = policy_main.detach().clone()
        draft_before = draft.detach().clone()
        draft_main_before = draft_main.detach().clone()
        draft_state_before = _clone_state(base_optimizer.state[draft_main])
        groups_before = list(base_optimizer.param_groups)

        policy.grad = torch.ones_like(policy)
        draft.grad = torch.ones_like(draft)
        with suspend_draft_optimizer_groups(optimizer):
            assert len(base_optimizer.param_groups) == 1
            assert base_optimizer.param_groups[0] is groups_before[0]
            update_successful, _, _ = optimizer.step()

        assert update_successful is True
        assert not torch.equal(policy, policy_model_before)
        assert not torch.equal(policy_main, policy_before)
        assert torch.equal(draft, draft_before)
        assert torch.equal(draft_main, draft_main_before)
        assert len(base_optimizer.param_groups) == len(groups_before)
        assert all(
            restored is original
            for restored, original in zip(
                base_optimizer.param_groups, groups_before, strict=True
            )
        )
        for key in ("exp_avg", "exp_avg_sq", "step"):
            actual = base_optimizer.state[draft_main][key]
            expected = draft_state_before[key]
            assert isinstance(actual, torch.Tensor)
            assert isinstance(expected, torch.Tensor)
            assert torch.equal(actual, expected)
    finally:
        mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS = original_grad_norm_groups


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


def test_mcore_chained_skip_preserves_draft_master_state(
    distributed_test_runner: Callable[..., None],
) -> None:
    distributed_test_runner(_run_mcore_master_parameter_skip, world_size=1)


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
