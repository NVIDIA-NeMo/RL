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
from pydantic import ValidationError

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
    OPDAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig


def _make_estimator():
    return OPDAdvantageEstimator(AdvEstimatorConfig(name="opd"), ClippedPGLossConfig())


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


def _opd(**fields) -> OPDAdvantageEstimator:
    return OPDAdvantageEstimator(
        AdvEstimatorConfig(name="opd", **fields), ClippedPGLossConfig()
    )


def _grpo_advantages(prompt_ids, rewards, mask, valid_mask=None) -> torch.Tensor:
    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(name="grpo"), ClippedPGLossConfig()
    )
    return estimator.compute_advantage(prompt_ids, rewards, mask, valid_mask=valid_mask)


def test_opd_tropd_alpha_one_is_the_plain_gap():
    """proximal_teacher_alpha=1 keeps the MOPD log-ratio exactly."""
    teacher_lp = torch.tensor([[-0.5, -3.0, -0.1]])
    student_lp = torch.tensor([[-1.0, -0.2, -0.1]])
    mask = torch.ones(1, 3)

    adv = _opd(proximal_teacher_alpha=1.0).compute_advantage(
        torch.zeros(1),
        torch.zeros(1),
        mask,
        teacher_logprobs=teacher_lp,
        prev_logprobs=student_lp,
    )

    torch.testing.assert_close(adv, teacher_lp - student_lp)


def test_opd_tropd_follows_the_proximal_teacher_and_bounds_negatives():
    """A = log(1 - a + a * p_T / p_S): floor log(1 - a), about a * gap near zero."""
    alpha = 0.2
    teacher_lp = torch.tensor([[-50.0, -0.11, 0.0]])
    student_lp = torch.tensor([[-0.1, -0.1, -5.0]])
    mask = torch.ones(1, 3)

    adv = _opd(proximal_teacher_alpha=alpha).compute_advantage(
        torch.zeros(1),
        torch.zeros(1),
        mask,
        teacher_logprobs=teacher_lp,
        prev_logprobs=student_lp,
    )

    expected = torch.log(1 - alpha + alpha * (teacher_lp - student_lp).exp())
    torch.testing.assert_close(adv, expected)
    # A token the teacher rejects costs at most log(1 - alpha) ...
    assert adv[0, 0].item() == pytest.approx(math.log(1 - alpha), abs=1e-6)
    # ... a small gap is scaled by about alpha ...
    assert adv[0, 1].item() == pytest.approx(alpha * -0.01, rel=1e-2)
    # ... and a token the teacher prefers keeps a large positive advantage.
    assert adv[0, 2].item() == pytest.approx(
        math.log(1 - alpha + alpha * math.exp(5.0)), rel=1e-6
    )


def test_opd_outcome_weight_adds_the_grpo_group_advantage():
    """A = w_d * gap + w_o * GRPO advantage, zero on masked tokens."""
    generator = torch.Generator().manual_seed(0)
    prompt_ids = torch.tensor([[0], [0], [0], [1], [1], [1]])
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    mask = torch.tensor([[0.0, 1.0, 1.0]] * 6)
    teacher_lp = -torch.rand(6, 3, generator=generator)
    student_lp = -torch.rand(6, 3, generator=generator)

    adv = _opd(
        distillation_advantage_weight=0.1, outcome_advantage_weight=1.0
    ).compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    grpo = _grpo_advantages(prompt_ids, rewards, mask)
    torch.testing.assert_close(adv, (0.1 * (teacher_lp - student_lp) + grpo) * mask)
    assert (adv[:, 0] == 0).all()


def test_opd_outcome_term_keeps_invalid_rows_out_of_the_baseline():
    """valid_mask reaches the outcome term exactly as it reaches GRPO."""
    prompt_ids = torch.tensor([[0], [0], [0]])
    rewards = torch.tensor([1.0, 0.0, 1.0])
    valid_mask = torch.tensor([1.0, 1.0, 0.0])
    mask = torch.ones(3, 2)
    zeros = torch.zeros(3, 2)

    adv = _opd(
        distillation_advantage_weight=0.0, outcome_advantage_weight=1.0
    ).compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=zeros,
        prev_logprobs=zeros,
        valid_mask=valid_mask,
    )

    torch.testing.assert_close(
        adv, _grpo_advantages(prompt_ids, rewards, mask, valid_mask=valid_mask)
    )
    assert not torch.allclose(adv, _grpo_advantages(prompt_ids, rewards, mask))


def test_opd_parts_report_each_term():
    """compute_advantage_parts exposes the raw gap and both weighted terms."""
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, 2)
    teacher_lp = torch.full((2, 2), -0.5)
    student_lp = torch.full((2, 2), -1.0)

    pure = _make_estimator().compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher_lp,
        prev_logprobs=student_lp,
    )
    mixed = _opd(
        proximal_teacher_alpha=0.5,
        distillation_advantage_weight=0.25,
        outcome_advantage_weight=2.0,
    ).compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher_lp,
        prev_logprobs=student_lp,
    )

    assert pure.outcome is None
    torch.testing.assert_close(pure.advantages, pure.distillation)
    torch.testing.assert_close(mixed.teacher_student_gap, torch.full((2, 2), 0.5))
    torch.testing.assert_close(
        mixed.distillation,
        torch.full((2, 2), 0.25 * math.log(0.5 + 0.5 * math.exp(0.5))),
    )
    assert mixed.outcome is not None
    torch.testing.assert_close(
        mixed.outcome, 2.0 * _grpo_advantages(prompt_ids, rewards, mask)
    )
    torch.testing.assert_close(mixed.advantages, mixed.distillation + mixed.outcome)


def test_opd_metrics_keep_the_raw_gap_and_report_terms():
    """The gap metric stays the raw log-ratio when TR-OPD and mixing are on."""
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, 2)
    estimator = _opd(proximal_teacher_alpha=0.2, outcome_advantage_weight=1.0)

    estimator.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=torch.zeros(2, 2),
        prev_logprobs=torch.full((2, 2), -1.0),
    )

    metrics = estimator.last_metrics
    assert metrics[
        "on_policy_distillation/teacher_student_logprob_gap_mean"
    ] == pytest.approx(1.0)
    assert metrics["on_policy_distillation/distill_adv_mean"] == pytest.approx(
        math.log(0.8 + 0.2 * math.e)
    )
    assert "on_policy_distillation/outcome_adv_mean" in metrics


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proximal_teacher_alpha", 0.2),
        ("distillation_advantage_weight", 0.5),
        ("outcome_advantage_weight", 1.0),
    ],
)
def test_opd_fields_are_rejected_on_other_estimators(field, value):
    """An OPD knob on another estimator would be silently ignored."""
    with pytest.raises(ValidationError, match=field):
        AdvEstimatorConfig(name="grpo", **{field: value})


def test_opd_rejects_two_zero_weights():
    with pytest.raises(ValidationError, match="both 0"):
        AdvEstimatorConfig(
            name="opd",
            distillation_advantage_weight=0.0,
            outcome_advantage_weight=0.0,
        )


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_proximal_teacher_alpha_must_be_in_the_unit_interval(alpha):
    with pytest.raises(ValidationError):
        AdvEstimatorConfig(name="opd", proximal_teacher_alpha=alpha)


@pytest.mark.parametrize("name", ["grpo", "gdpo", "opd", "reinforce_plus_plus"])
def test_opd_defaults_are_accepted_on_every_estimator(name):
    """The exemplar YAML spells out the OPD defaults for every estimator."""
    config = AdvEstimatorConfig(
        name=name,
        proximal_teacher_alpha=1.0,
        distillation_advantage_weight=1.0,
        outcome_advantage_weight=0.0,
    )
    assert config.name == name
