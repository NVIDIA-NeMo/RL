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

import torch

from nemo_rl.algorithms.advantage_estimator import (
    OPDAdvantageEstimator,
    ReinforceBaselineAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig


def _make_estimator():
    return OPDAdvantageEstimator({"name": "opd"}, {})


def _make_reinforce_baseline_estimator():
    return ReinforceBaselineAdvantageEstimator({}, ClippedPGLossConfig())


def test_reinforce_baseline_uses_ordinary_group_mean_and_singleton_zero():
    estimator = _make_reinforce_baseline_estimator()
    prompt_ids = torch.tensor([[0], [0], [1], [1], [1], [2]])
    rewards = torch.tensor([0.0, 2.0, 0.0, 4.0, 8.0, 7.0])

    rollout_advantages = estimator.compute_rollout_advantages(prompt_ids, rewards)

    # These are reward minus the full group mean, not leave-one-out values.
    torch.testing.assert_close(
        rollout_advantages,
        torch.tensor([-1.0, 1.0, -4.0, 0.0, 4.0, 0.0]),
    )


def test_reinforce_baseline_whitens_over_unequal_action_lengths():
    estimator = _make_reinforce_baseline_estimator()
    mask = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    advantages = estimator.compute_advantage(
        prompt_ids=torch.tensor([[5], [5]]),
        rewards=torch.tensor([1.0, 3.0]),
        mask=mask,
    )

    # Unwhitened rollout values are [-1, 1]. Repeating them over one and three
    # action tokens gives token mean 0.5 and population variance 0.75.
    expected = torch.tensor(
        [
            [-(3.0**0.5), 0.0, 0.0],
            [1.0 / (3.0**0.5)] * 3,
        ]
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close((advantages * mask).sum() / mask.sum(), torch.tensor(0.0))
    torch.testing.assert_close(
        ((advantages.pow(2) * mask).sum() / mask.sum()),
        torch.tensor(1.0),
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
