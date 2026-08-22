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


def test_rejects_non_eagle_bins_and_shape_drift() -> None:
    state = DraftStepState()
    state.accumulate(state.metric_payload(_stats(1.0, 1.0)))
    two_bins = DraftLossStats(
        numerators=torch.ones(2),
        counts=torch.ones(2),
        weights=torch.ones(2),
    )

    with pytest.raises(ValueError, match="one bin"):
        state.accumulate(state.metric_payload(two_bins))


def test_rejects_non_unit_eagle_weight() -> None:
    state = DraftStepState()
    weighted = DraftLossStats(
        numerators=torch.ones(1),
        counts=torch.ones(1),
        weights=torch.tensor([0.5]),
    )

    with pytest.raises(ValueError, match="unit weight"):
        state.accumulate(state.metric_payload(weighted))


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


def test_split_accumulation_matches_monolithic_normalization() -> None:
    """Split-vs-nonsplit parity: draft counts equal the policy's shifted-mask
    count by construction (roll_tensor zeroes the vacated slot), so accumulating
    per-microbatch stats and normalizing once must equal the monolithic path."""
    from nemo_rl.algorithms.loss.draft import streaming_vocab_parallel_soft_ce

    generator = torch.Generator().manual_seed(20260822)
    student = torch.randn(4, 5, 7, generator=generator, dtype=torch.float64)
    teacher = torch.randn(4, 5, 7, generator=generator, dtype=torch.float64)
    token_mask = torch.ones(4, 5)
    token_mask[0, 0] = 1.0  # exercise the shifted first slot explicitly
    token_mask[1, 3] = 0.0
    rolled_mask = torch.roll(token_mask, shifts=-1, dims=1)
    rolled_mask[:, -1] = 0.0

    monolithic = streaming_vocab_parallel_soft_ce(
        student_logits=student,
        teacher_logits=teacher,
        mask=rolled_mask,
        token_chunk_size=64,
        tp_group=None,
    )
    monolithic_loss = monolithic.normalized(
        normalization_counts=monolithic.counts,
    )

    state = DraftStepState()
    split_losses = []
    for half in (slice(0, 2), slice(2, 4)):
        stats = streaming_vocab_parallel_soft_ce(
            student_logits=student[half],
            teacher_logits=teacher[half],
            mask=rolled_mask[half],
            token_chunk_size=64,
            tp_group=None,
        )
        state.accumulate(state.metric_payload(stats))
        split_losses.append(stats.numerators)
    global_counts = state.local_counts
    split_loss = sum(split_losses).sum() / (global_counts.sum() + 1e-8)

    assert torch.equal(global_counts, monolithic.counts), (
        "split-accumulated counts must equal the monolithic denominator"
    )
    torch.testing.assert_close(split_loss, monolithic_loss, rtol=1e-12, atol=1e-12)
    policy_count = token_mask[:, 1:].sum()
    assert torch.equal(global_counts.sum(), policy_count), (
        "draft rolled-mask count must equal the policy shifted-mask count"
    )
