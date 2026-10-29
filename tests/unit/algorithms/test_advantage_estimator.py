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
from pydantic import ValidationError

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GAEConfig,
    GeneralizedAdvantageEstimator,
    OPDAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig


@pytest.mark.parametrize("estimator_name", ["reinforce_plus_plus", "gae"])
@pytest.mark.parametrize("use_kl", [False, True])
@pytest.mark.parametrize(
    "kl_type,clamps,expected_kl",
    [
        ("k1", {}, [[0, 2, 5, 10], [1, 4, 10, 10]]),
        (
            "k1",
            {"kl_input_clamp_value": 2.0, "kl_output_clamp_value": None},
            [[0, 2, 2, 2], [1, 2, 2, 2]],
        ),
        (
            "k1",
            {"kl_input_clamp_value": None, "kl_output_clamp_value": 3.0},
            [[0, 2, 3, 3], [1, 3, 3, 3]],
        ),
        (
            "k1",
            {"kl_input_clamp_value": None, "kl_output_clamp_value": None},
            [[0, 2, 5, 30], [1, 4, 15, 60]],
        ),
        (
            "k2",
            {"kl_input_clamp_value": 2.0, "kl_output_clamp_value": None},
            [[0, 2, 2, 2], [0.5, 2, 2, 2]],
        ),
        (
            "k2",
            {"kl_input_clamp_value": None, "kl_output_clamp_value": 3.0},
            [[0, 2, 3, 3], [0.5, 3, 3, 3]],
        ),
        (
            "k3",
            {"kl_input_clamp_value": 2.0, "kl_output_clamp_value": None},
            [
                [0, 1.1353353, 1.1353353, 1.1353353],
                [0.36787944, 1.1353353, 1.1353353, 1.1353353],
            ],
        ),
        (
            "k3",
            {"kl_input_clamp_value": None, "kl_output_clamp_value": 3.0},
            [[0, 1.1353353, 3, 3], [0.36787944, 3, 3, 3]],
        ),
    ],
    ids=[
        "k1-defaults",
        "k1-input-clamp",
        "k1-output-clamp",
        "k1-unclamped",
        "k2-input-clamp",
        "k2-output-clamp",
        "k3-input-clamp",
        "k3-output-clamp",
    ],
)
def test_reward_kl_clamps_reach_public_advantages(
    estimator_name: str,
    use_kl: bool,
    kl_type: str,
    clamps: dict[str, float | None],
    expected_kl: list[list[float]],
) -> None:
    """Nondefault clamps must change reward KL before normalization or GAE."""
    loss_config = ClippedPGLossConfig(
        use_kl_in_reward=use_kl,
        reference_policy_kl_penalty=0.5,
        reference_policy_kl_type=kl_type,
        **clamps,
    )
    policy = -torch.ones(2, 4)
    reference = policy - torch.tensor([[0, 2, 5, 30], [1, 4, 15, 60]])
    mask = torch.ones_like(policy)
    rewards = torch.tensor([1.0, 2.0])
    # Fixed numerical expectations avoid calling calculate_kl as the oracle.
    token_rewards = -0.5 * torch.tensor(expected_kl) if use_kl else torch.zeros(2, 4)
    kwargs = dict(logprobs_policy=policy, logprobs_reference=reference)

    if estimator_name == "gae":
        estimator = GeneralizedAdvantageEstimator(
            GAEConfig(gae_lambda=1.0, gae_gamma=1.0, normalize_advantages=False),
            loss_config,
        )
        advantages, returns = estimator.compute_advantage(
            torch.arange(2), rewards, mask, values=torch.zeros(2, 4), **kwargs
        )
        token_rewards[:, -1] += rewards
        expected = token_rewards.flip(1).cumsum(1).flip(1)
        torch.testing.assert_close(returns, expected)
    else:
        estimator = ReinforcePlusPlusAdvantageEstimator(
            AdvEstimatorConfig(name="reinforce_plus_plus", minus_baseline=False),
            loss_config,
        )
        advantages = estimator.compute_advantage(
            torch.arange(2), rewards, mask, **kwargs
        )
        expected = token_rewards + rewards.unsqueeze(-1)
        expected = (expected - expected.mean()) * expected.var(unbiased=False).clamp(
            min=1e-8
        ).rsqrt()

    torch.testing.assert_close(advantages, expected)


def _make_estimator(alpha=1.0, subtract_global_baseline=False):
    return OPDAdvantageEstimator(
        AdvEstimatorConfig(
            name="opd",
            proximal_teacher_alpha=alpha,
            subtract_global_baseline=subtract_global_baseline,
        ),
        {},
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


def test_tropd_alpha_one_is_plain_opd():
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


def test_tropd_advantage_is_bounded_below_by_log_one_minus_alpha():
    alpha = 0.2
    # The teacher all but rules out every sampled token.
    teacher = torch.full((1, 3), -80.0)
    student = torch.tensor([[-0.1, -1.0, -3.0]])

    advantage = _make_estimator(alpha=alpha).compute_advantage(
        None,
        None,
        torch.ones_like(teacher),
        teacher_logprobs=teacher,
        prev_logprobs=student,
    )

    torch.testing.assert_close(
        advantage, torch.full_like(advantage, torch.log1p(torch.tensor(-alpha)).item())
    )


def test_tropd_metrics_report_raw_teacher_student_gap():
    teacher = torch.tensor([[0.0, -1.0, 5.0]])
    student = torch.tensor([[-1.0, -1.0, -5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    estimator = _make_estimator(alpha=0.2)

    advantage = estimator.compute_advantage(
        None,
        None,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=student,
    )

    # The raw gap over valid tokens is mean([1, 0]); the advantage is smaller.
    assert estimator.last_metrics[
        "on_policy_distillation/teacher_student_logprob_gap_mean"
    ] == pytest.approx(0.5)
    assert estimator.last_metrics["on_policy_distillation/adv_mean"] == pytest.approx(
        advantage[:, :2].mean().item()
    )
    assert estimator.last_metrics["on_policy_distillation/adv_mean"] < 0.5


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.01])
def test_tropd_rejects_alpha_outside_unit_interval(alpha):
    with pytest.raises(ValidationError, match="proximal_teacher_alpha"):
        _make_estimator(alpha=alpha)


def test_tropd_global_baseline_uses_only_valid_tokens_after_interpolation():
    teacher = torch.tensor([[0.0, -1.0, 100.0]])
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
    ] == pytest.approx(0.5)


def test_opd_global_baseline_without_tropd_centers_raw_gap():
    teacher = torch.tensor([[0.0, -1.0], [-2.0, -0.5]])
    student = torch.tensor([[-1.0, -1.0], [-1.0, -3.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

    advantage = _make_estimator(subtract_global_baseline=True).compute_advantage(
        None,
        None,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=student,
    )

    # Valid gaps are [1, 0, 2.5] (the masked -1 does not vote): mean 7/6.
    torch.testing.assert_close(advantage, (teacher - student - 7.0 / 6.0) * mask)


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
