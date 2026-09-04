# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import OPDAdvantageEstimator


def _make_estimator(alpha=1.0, subtract_global_baseline=False):
    return OPDAdvantageEstimator(
        {"name": "opd"},
        {},
        proximal_teacher_alpha=alpha,
        subtract_global_baseline=subtract_global_baseline,
    )


def test_opd_basic_positive_distill_advantage():
    """teacher_lp > student_lp => positive advantages."""
    estimator = _make_estimator()
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)  # log(1) = 0
    student_lp = torch.full((B, S), -1.0)  # lower logprob
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    assert adv.shape == (B, S)
    assert (adv > 0).all(), "teacher_lp > student_lp should yield positive advantages"


def test_opd_teacher_equals_student():
    """Same logprobs => zero advantages."""
    estimator = _make_estimator()
    B, S = 2, 4
    logprobs = torch.randn(B, S)
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=logprobs, prev_logprobs=logprobs
    )

    torch.testing.assert_close(adv, torch.zeros(B, S))


def test_opd_mask_applied():
    """Masked tokens should have zero advantage."""
    estimator = _make_estimator()
    B, S = 1, 6
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -1.0)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # Masked positions must be zero
    assert (adv[:, 3:] == 0).all(), "Masked positions should be zero"
    # Unmasked positions should be positive (teacher > student)
    assert (adv[:, :3] > 0).all(), "Unmasked positions should be positive"


def test_opd_metrics_returned():
    """self.last_metrics should be populated after compute_advantage."""
    estimator = _make_estimator()
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -1.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    assert (
        "on_policy_distillation/teacher_student_logprob_gap_mean"
        in estimator.last_metrics
    )
    assert "on_policy_distillation/adv_mean" in estimator.last_metrics
    assert "on_policy_distillation/adv_std" in estimator.last_metrics
    # teacher - student = 0 - (-1) = 1.0
    assert (
        abs(
            estimator.last_metrics[
                "on_policy_distillation/teacher_student_logprob_gap_mean"
            ]
            - 1.0
        )
        < 1e-5
    )
    assert abs(estimator.last_metrics["on_policy_distillation/adv_mean"] - 1.0) < 1e-5
    assert abs(estimator.last_metrics["on_policy_distillation/adv_std"]) < 1e-5


def test_tropd_alpha_one_is_legacy_opd():
    teacher = torch.tensor([[-0.2, -1.7, -0.5]])
    student = torch.tensor([[-1.2, -0.8, -0.5]])
    mask = torch.ones_like(teacher)

    advantage = _make_estimator(alpha=1.0).compute_advantage(
        None,
        None,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=student,
    )

    torch.testing.assert_close(advantage, teacher - student)


def test_tropd_interpolates_teacher_and_student_probabilities():
    alpha = 0.2
    teacher = torch.tensor([[-0.2, -1.7, -0.5]])
    student = torch.tensor([[-1.2, -0.8, -0.5]])
    mask = torch.ones_like(teacher)

    advantage = _make_estimator(alpha=alpha).compute_advantage(
        None,
        None,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=student,
    )
    expected = (
        torch.log(alpha * torch.exp(teacher) + (1.0 - alpha) * torch.exp(student))
        - student
    )

    torch.testing.assert_close(advantage, expected)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.01])
def test_tropd_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="proximal_teacher_alpha"):
        _make_estimator(alpha=alpha)


def test_tropd_global_baseline_uses_only_valid_tokens_after_interpolation():
    teacher = torch.tensor([[0.0, -2.0, 100.0]])
    student = torch.tensor([[-1.0, -1.0, -100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    estimator = _make_estimator(alpha=0.2, subtract_global_baseline=True)

    advantage = estimator.compute_advantage(
        None,
        None,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=student,
    )
    interpolated = (
        torch.log(0.2 * torch.exp(teacher[:, :2]) + 0.8 * torch.exp(student[:, :2]))
        - student[:, :2]
    )
    expected = torch.zeros_like(teacher)
    expected[:, :2] = interpolated - interpolated.mean()

    torch.testing.assert_close(advantage, expected)
    assert estimator.last_metrics["on_policy_distillation/adv_mean"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert estimator.last_metrics[
        "on_policy_distillation/teacher_student_logprob_gap_mean"
    ] == pytest.approx(0.0)


def test_tropd_all_masked_batch_is_finite_zero():
    estimator = _make_estimator(alpha=0.2, subtract_global_baseline=True)
    advantage = estimator.compute_advantage(
        None,
        None,
        torch.zeros(2, 3),
        teacher_logprobs=torch.randn(2, 3),
        prev_logprobs=torch.randn(2, 3),
    )

    torch.testing.assert_close(advantage, torch.zeros(2, 3))
    assert all(
        torch.isfinite(torch.tensor(value)) for value in estimator.last_metrics.values()
    )
