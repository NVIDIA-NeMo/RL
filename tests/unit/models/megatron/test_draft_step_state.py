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

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nemo_rl.algorithms.loss.draft import DraftLossStats

_STEP_STATE_PATH = (
    Path(__file__).parents[4]
    / "nemo_rl"
    / "models"
    / "megatron"
    / "draft"
    / "step_state.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "draft_step_state_under_test", _STEP_STATE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
DraftStepState = _MODULE.DraftStepState


def _stats(numerator: float, count: float) -> DraftLossStats:
    return DraftLossStats(
        numerators=torch.tensor([numerator], requires_grad=True),
        counts=torch.tensor([count]),
        weights=torch.ones(1),
    )


def test_accumulates_detached_one_bin_payloads() -> None:
    state = DraftStepState()

    state.accumulate(state.metric_payload(_stats(6.0, 2.0)))
    state.accumulate(state.metric_payload(_stats(9.0, 3.0)))

    assert torch.equal(state.local_numerators, torch.tensor([15.0]))
    assert torch.equal(state.local_counts, torch.tensor([5.0]))
    assert state.local_numerators.requires_grad is False


def test_rejects_bin_shape_drift() -> None:
    state = DraftStepState()
    state.accumulate(state.metric_payload(_stats(1.0, 1.0)))
    two_bins = DraftLossStats(
        numerators=torch.ones(2),
        counts=torch.ones(2),
        weights=torch.ones(2),
    )

    with pytest.raises(ValueError, match="shape changed"):
        state.accumulate(state.metric_payload(two_bins))


def test_rejects_weight_drift_within_step() -> None:
    state = DraftStepState()
    state.accumulate(state.metric_payload(_stats(1.0, 1.0)))
    weighted = DraftLossStats(
        numerators=torch.ones(1),
        counts=torch.ones(1),
        weights=torch.tensor([0.5]),
    )

    with pytest.raises(ValueError, match="weights changed"):
        state.accumulate(state.metric_payload(weighted))


def test_accumulates_weighted_dflash_position_bins() -> None:
    state = DraftStepState()
    first = DraftLossStats(
        numerators=torch.tensor([2.0, 4.0, 8.0]),
        counts=torch.tensor([1.0, 2.0, 4.0]),
        weights=torch.tensor([1.0, 0.5, 0.25]),
    )
    second = DraftLossStats(
        numerators=torch.tensor([3.0, 6.0, 12.0]),
        counts=torch.tensor([2.0, 3.0, 5.0]),
        weights=first.weights.clone(),
    )

    state.accumulate(state.metric_payload(first))
    state.accumulate(state.metric_payload(second))
    state.set_global_counts(torch.tensor([6.0, 10.0, 18.0]))

    assert torch.equal(state.local_numerators, torch.tensor([5.0, 10.0, 20.0]))
    assert torch.equal(state.local_counts, torch.tensor([3.0, 5.0, 9.0]))
    assert state.normalize_metric(torch.tensor(31.0)).item() == pytest.approx(2.0)


def test_weighted_split_gradient_and_metric_match_synchronous_normalization() -> None:
    weights = torch.tensor([1.0, 0.5, 0.25])
    global_counts = torch.tensor([6.0, 10.0, 18.0])
    sync_parameter = torch.nn.Parameter(torch.tensor(2.0))
    sync_stats = DraftLossStats(
        numerators=sync_parameter * torch.tensor([5.0, 10.0, 20.0]),
        counts=global_counts,
        weights=weights,
    )
    sync_loss = sync_stats.normalized(normalization_counts=global_counts)
    sync_loss.backward()

    split_parameter = torch.nn.Parameter(torch.tensor(2.0))
    first = DraftLossStats(
        numerators=split_parameter * torch.tensor([2.0, 4.0, 8.0]),
        counts=torch.tensor([1.0, 2.0, 4.0]),
        weights=weights,
    )
    second = DraftLossStats(
        numerators=split_parameter * torch.tensor([3.0, 6.0, 12.0]),
        counts=torch.tensor([2.0, 3.0, 5.0]),
        weights=weights,
    )
    raw_split_loss = sum(
        (stats.numerators * stats.weights).sum() for stats in (first, second)
    )
    policy_count = torch.tensor(32.0)
    (raw_split_loss / policy_count).backward()
    split_parameter.grad_norm_group = "draft"
    split_parameter.main_grad = split_parameter.grad.detach().clone()
    state = DraftStepState()
    state.accumulate(state.metric_payload(first))
    state.accumulate(state.metric_payload(second))
    state.set_global_counts(global_counts)

    state.correct_main_grads(
        [split_parameter],
        policy_normalization_count=policy_count,
    )

    torch.testing.assert_close(split_parameter.main_grad, sync_parameter.grad)
    torch.testing.assert_close(
        state.normalize_metric(raw_split_loss.detach()),
        sync_loss.detach(),
    )


def test_inactive_state_contributes_no_collective_counts() -> None:
    state = DraftStepState()
    reference = torch.tensor([4.0, 16.0], dtype=torch.float64)

    counts = state.counts_for_reduction(reference)

    assert counts.shape == (0,)
    assert counts.dtype == reference.dtype
    with pytest.raises(ValueError, match="inactive draft step"):
        state.set_global_counts(reference.new_ones(1))


def test_corrects_only_draft_main_grads_relative_to_policy_scaling() -> None:
    state = DraftStepState()
    state.accumulate(state.metric_payload(_stats(12.0, 4.0)))
    state.set_global_counts(torch.tensor([8.0]))
    draft_param = torch.nn.Parameter(torch.tensor(1.0))
    draft_param.grad_norm_group = "draft"
    draft_param.main_grad = torch.tensor(3.0)
    policy_param = torch.nn.Parameter(torch.tensor(1.0))
    policy_param.main_grad = torch.tensor(5.0)

    state.correct_main_grads(
        [draft_param, policy_param], policy_normalization_count=torch.tensor(16.0)
    )

    assert draft_param.main_grad.item() == pytest.approx(6.0)
    assert policy_param.main_grad.item() == pytest.approx(5.0)


def test_zero_draft_count_has_zero_scale_and_finite_metrics() -> None:
    state = DraftStepState()
    state.accumulate(state.metric_payload(_stats(0.0, 0.0)))
    state.set_global_counts(torch.zeros(1))
    draft_param = torch.nn.Parameter(torch.tensor(1.0))
    draft_param.grad_norm_group = "draft"
    draft_param.main_grad = torch.tensor(3.0)

    state.correct_main_grads(
        [draft_param], policy_normalization_count=torch.tensor(16.0)
    )

    assert draft_param.main_grad.item() == 0.0
    assert state.normalize_metric(torch.tensor(0.0)).item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_policy_count_zeroes_draft_gradient(dtype: torch.dtype) -> None:
    state = DraftStepState()
    state.accumulate(state.metric_payload(_stats(4.0, 2.0)))
    state.set_global_counts(torch.tensor([2.0]))
    draft_param = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype))
    draft_param.grad_norm_group = "draft"
    draft_param.main_grad = torch.tensor(3.0, dtype=dtype)

    state.correct_main_grads(
        [draft_param], policy_normalization_count=torch.tensor(0.0)
    )

    assert draft_param.main_grad.item() == 0.0
