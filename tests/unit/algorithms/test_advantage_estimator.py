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

import math

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import OPDAdvantageEstimator


def _make_estimator(**config):
    return OPDAdvantageEstimator({"name": "opd", **config}, {})


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


@pytest.mark.parametrize("all_invalid", [False, True])
def test_opd_teacher_mask_is_nonfinite_safe_and_drives_metrics(all_invalid):
    """Only the token/sample/teacher-mask intersection contributes."""
    estimator = _make_estimator()
    teacher_lp = torch.tensor([[1.0, float("nan"), float("inf"), -3.0]])
    student_lp = torch.zeros_like(teacher_lp)
    # This represents the token-mask/sample-mask intersection supplied by GRPO.
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    teacher_mask = torch.zeros_like(mask)
    if not all_invalid:
        teacher_mask[0, 0] = 1.0

    advantages = estimator.compute_advantage(
        torch.arange(1),
        torch.zeros(1),
        mask,
        teacher_logprobs=teacher_lp,
        teacher_logprobs_mask=teacher_mask,
        prev_logprobs=student_lp,
    )

    expected = torch.tensor([[0.0 if all_invalid else 1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(advantages, expected)
    assert torch.isfinite(advantages).all()
    assert estimator.last_metrics[
        "on_policy_distillation/teacher_student_logprob_gap_mean"
    ] == pytest.approx(0.0 if all_invalid else 1.0)
    assert estimator.last_metrics["on_policy_distillation/adv_mean"] == pytest.approx(
        0.0 if all_invalid else 1.0
    )


def test_opd_proximal_reward_disabled_is_bitwise_legacy_opd():
    # A disabled transform ignores its otherwise-invalid scale and remains the
    # legacy OPD calculation bit-for-bit.
    estimator = _make_estimator(proximal_reward_alpha=None, proximal_reward_scale=0.0)
    teacher_lp = torch.tensor([[0.25, -3.0, 1.5, -0.75]])
    student_lp = torch.tensor([[-0.5, -2.0, 0.25, -1.25]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    expected = (teacher_lp - student_lp).detach() * mask

    advantages = estimator.compute_advantage(
        torch.arange(1),
        torch.zeros(1),
        mask,
        teacher_logprobs=teacher_lp,
        prev_logprobs=student_lp,
    )

    torch.testing.assert_close(advantages, expected, rtol=0, atol=0)


def test_opd_proximal_reward_known_values_stability_and_masked_nonfinite():
    estimator = _make_estimator(proximal_reward_alpha=0.2, proximal_reward_scale=5.0)
    gaps = torch.tensor([[-10000.0, 0.0, 10000.0, float("nan")]], requires_grad=True)

    advantages = estimator.compute_advantage(
        torch.arange(1),
        torch.zeros(1),
        torch.ones_like(gaps),
        teacher_logprobs=gaps,
        teacher_logprobs_mask=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        prev_logprobs=torch.zeros_like(gaps),
    )

    assert torch.isfinite(advantages).all()
    assert not advantages.requires_grad
    assert advantages[0, 0].item() == pytest.approx(5.0 * math.log(0.8))
    assert advantages[0, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert advantages[0, 2].item() == pytest.approx(
        5.0 * (10000.0 + math.log(0.2)), rel=1e-6
    )
    assert advantages[0, 3].item() == 0.0
    assert estimator.last_metrics[
        "on_policy_distillation/teacher_student_logprob_gap_mean"
    ] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"proximal_reward_alpha": 0.0}, "proximal_reward_alpha"),
        ({"proximal_reward_alpha": 1.1}, "proximal_reward_alpha"),
        ({"proximal_reward_alpha": float("nan")}, "proximal_reward_alpha"),
        (
            {"proximal_reward_alpha": 0.2, "proximal_reward_scale": 0.0},
            "proximal_reward_scale",
        ),
        (
            {
                "proximal_reward_alpha": 0.2,
                "proximal_reward_scale": float("inf"),
            },
            "proximal_reward_scale",
        ),
    ],
)
def test_opd_proximal_reward_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_estimator(**overrides)
