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
    reasoning_span_mask,
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


def _contrast_inputs():
    prompt_ids = torch.tensor([[0], [0], [1], [1]])
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4, 3)
    teacher = torch.tensor(
        [[-1.0, -2.0, -3.0], [-1.0, -1.0, -1.0], [-0.5, -0.5, -0.5], [-2.0, -2.0, -2.0]]
    )
    negative = torch.tensor(
        [[-1.5, -1.0, -3.0], [-2.0, -2.0, -2.0], [-0.5, -0.5, -0.5], [-1.0, -1.0, -1.0]]
    )
    prev = torch.full((4, 3), -0.7)
    return prompt_ids, rewards, mask, teacher, negative, prev


def test_contrast_teacher_distills_the_teacher_minus_contrast_log_ratio():
    cfg = AdvEstimatorConfig(
        name="opd",
        contrast_teacher=True,
        contrast_alpha=0.5,
        contrast_clip=0.4,
        distillation_advantage_weight=1.0,
        outcome_advantage_weight=0.0,
    )
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, negative, prev = _contrast_inputs()
    parts = est.compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
    )
    expected = (0.5 * (teacher - negative)).clamp(-0.4, 0.4)
    torch.testing.assert_close(parts.distillation, expected)
    torch.testing.assert_close(parts.contrast_gap, teacher - negative)
    # The student does not enter the contrast term; the diagnostic gap still does.
    torch.testing.assert_close(parts.teacher_student_gap, teacher - prev)
    assert parts.outcome is None
    adv = est.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
    )
    torch.testing.assert_close(adv, expected)
    assert "on_policy_distillation/contrast_gap_mean" in est.last_metrics


def test_contrast_teacher_requires_the_negative_column():
    cfg = AdvEstimatorConfig(name="opd", contrast_teacher=True)
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, _, prev = _contrast_inputs()
    with pytest.raises(ValueError, match="negative_teacher_logprobs"):
        est.compute_advantage_parts(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            teacher_logprobs=teacher,
            prev_logprobs=prev,
        )


def test_contrast_teacher_mixes_with_the_outcome_term():
    cfg = AdvEstimatorConfig(
        name="opd",
        contrast_teacher=True,
        distillation_advantage_weight=0.1,
        outcome_advantage_weight=1.0,
    )
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, negative, prev = _contrast_inputs()
    parts = est.compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
    )
    assert parts.outcome is not None
    torch.testing.assert_close(parts.distillation, 0.1 * (teacher - negative))
    torch.testing.assert_close(parts.advantages, parts.distillation + parts.outcome)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contrast_alpha": 0.5},
        {"contrast_clip": 0.5},
        {"contrast_teacher": True, "proximal_teacher_alpha": 0.2},
    ],
)
def test_contrast_fields_are_validated(kwargs):
    with pytest.raises(ValidationError):
        AdvEstimatorConfig(name="opd", **kwargs)


def test_contrast_fields_are_rejected_on_other_estimators():
    with pytest.raises(ValidationError, match="only apply to name='opd'"):
        AdvEstimatorConfig(name="grpo", contrast_teacher=True)


# ---------------------------------------------------------------------------
# Where the distillation term applies: trajectory gate and action-token scope
# ---------------------------------------------------------------------------

THINK, END_THINK = 12, 13


def test_reasoning_span_mask_covers_markers_and_unclosed_spans():
    ids = torch.tensor(
        [
            # prompt, <think> a b </think> call, <think> c (never closed)
            [1, THINK, 5, 6, END_THINK, 7, THINK, 8],
            # stray close marker outside a span is ignored; a second span works
            [END_THINK, 9, THINK, END_THINK, 9, THINK, 5, END_THINK],
        ]
    )
    expected = torch.tensor(
        [
            [False, True, True, True, True, False, True, True],
            [False, False, True, True, False, True, True, True],
        ]
    )
    torch.testing.assert_close(reasoning_span_mask(ids, THINK, END_THINK), expected)


def _scoped_inputs():
    prompt_ids, rewards, mask, teacher, negative, prev = _contrast_inputs()
    truncated = torch.tensor([False, False, True, False])
    # Row 0: reasoning span over tokens 0-1, action token 2. Row 1: all actions.
    # Row 2: resolved but truncated. Row 3: failed, all actions.
    input_ids = torch.tensor(
        [[THINK, END_THINK, 40], [41, 42, 43], [44, 45, 46], [47, 48, 49]]
    )
    return prompt_ids, rewards, mask, teacher, negative, prev, truncated, input_ids


def test_distill_gate_keeps_the_term_on_resolved_untruncated_rollouts_only():
    cfg = AdvEstimatorConfig(
        name="opd",
        contrast_teacher=True,
        contrast_alpha=0.5,
        distillation_advantage_weight=1.0,
        outcome_advantage_weight=1.0,
        distill_gate="resolved_untruncated",
    )
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, negative, prev, truncated, _ = _scoped_inputs()
    parts = est.compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
        truncated=truncated,
    )
    gate = torch.tensor([True, False, False, False])
    torch.testing.assert_close(parts.distill_gate, gate)
    expected = 0.5 * (teacher - negative) * gate.unsqueeze(-1).float()
    torch.testing.assert_close(parts.distillation, expected)
    # The outcome term is never gated: failed and truncated rows keep theirs.
    assert parts.outcome is not None
    assert parts.outcome[1].abs().sum() > 0 and parts.outcome[3].abs().sum() > 0
    torch.testing.assert_close(parts.advantages, parts.distillation + parts.outcome)
    est.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
        truncated=truncated,
    )
    assert est.last_metrics["on_policy_distillation/distill_gated_seq_frac"] == 0.25
    assert est.last_metrics["on_policy_distillation/distill_active_token_frac"] == 0.25


def test_distill_gate_resolved_ignores_truncation():
    cfg = AdvEstimatorConfig(name="opd", distill_gate="resolved")
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, _, prev, _, _ = _scoped_inputs()
    parts = est.compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
    )
    torch.testing.assert_close(parts.distill_gate, rewards > 0)


def test_distill_gate_resolved_untruncated_requires_the_truncated_column():
    cfg = AdvEstimatorConfig(name="opd", distill_gate="resolved_untruncated")
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, _, prev, _, _ = _scoped_inputs()
    with pytest.raises(ValueError, match="truncated"):
        est.compute_advantage_parts(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            teacher_logprobs=teacher,
            prev_logprobs=prev,
        )


def test_distill_token_scope_actions_skips_reasoning_tokens():
    cfg = AdvEstimatorConfig(
        name="opd",
        contrast_teacher=True,
        distill_token_scope="actions",
        reasoning_span_token_ids=(THINK, END_THINK),
    )
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, negative, prev, _, input_ids = _scoped_inputs()
    parts = est.compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
        input_ids=input_ids,
    )
    action_mask = torch.ones(4, 3, dtype=torch.bool)
    action_mask[0, :2] = False
    torch.testing.assert_close(parts.action_mask, action_mask)
    torch.testing.assert_close(parts.distill_active, action_mask)
    torch.testing.assert_close(parts.distillation, (teacher - negative) * action_mask)
    # The diagnostic contrast gap is still reported on every valid token.
    torch.testing.assert_close(parts.contrast_gap, teacher - negative)
    est.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
        input_ids=input_ids,
    )
    metrics = est.last_metrics
    assert metrics["on_policy_distillation/action_token_frac"] == pytest.approx(10 / 12)
    assert "on_policy_distillation/contrast_gap_actions_abs_mean" in metrics
    assert "on_policy_distillation/contrast_gap_reasoning_abs_mean" in metrics


def test_distill_token_scope_actions_requires_input_ids():
    cfg = AdvEstimatorConfig(
        name="opd",
        distill_token_scope="actions",
        reasoning_span_token_ids=(THINK, END_THINK),
    )
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, _, prev, _, _ = _scoped_inputs()
    with pytest.raises(ValueError, match="input_ids"):
        est.compute_advantage_parts(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            teacher_logprobs=teacher,
            prev_logprobs=prev,
        )


def test_gate_and_scope_compose():
    cfg = AdvEstimatorConfig(
        name="opd",
        contrast_teacher=True,
        distill_gate="resolved_untruncated",
        distill_token_scope="actions",
        reasoning_span_token_ids=(THINK, END_THINK),
    )
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, negative, prev, truncated, input_ids = (
        _scoped_inputs()
    )
    parts = est.compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
        truncated=truncated,
        input_ids=input_ids,
    )
    # Only row 0 passes the gate, and only its third token is an action.
    active = torch.zeros(4, 3, dtype=torch.bool)
    active[0, 2] = True
    torch.testing.assert_close(parts.distill_active, active)
    torch.testing.assert_close(parts.distillation, (teacher - negative) * active)


def test_default_scope_leaves_the_term_everywhere():
    cfg = AdvEstimatorConfig(name="opd", contrast_teacher=True)
    est = OPDAdvantageEstimator(cfg, ClippedPGLossConfig())
    prompt_ids, rewards, mask, teacher, negative, prev = _contrast_inputs()
    mask[3, 2] = 0.0
    parts = est.compute_advantage_parts(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
    )
    # Nothing narrowed the term, so nothing extra is reported (metrics unchanged).
    assert parts.distill_active is None
    assert parts.distill_gate is None
    assert parts.action_mask is None
    torch.testing.assert_close(parts.distillation, (teacher - negative) * mask)
    est.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=teacher,
        prev_logprobs=prev,
        negative_teacher_logprobs=negative,
    )
    assert "on_policy_distillation/distill_active_token_frac" not in est.last_metrics


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distill_token_scope": "actions"},
        {"reasoning_span_token_ids": (THINK, END_THINK)},
        {
            "distill_token_scope": "actions",
            "reasoning_span_token_ids": (THINK, THINK),
        },
        {
            "distill_gate": "resolved",
            "distillation_advantage_weight": 0.0,
            "outcome_advantage_weight": 1.0,
        },
    ],
)
def test_scope_fields_are_validated(kwargs):
    with pytest.raises(ValidationError):
        AdvEstimatorConfig(name="opd", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distill_gate": "resolved"},
        {
            "distill_token_scope": "actions",
            "reasoning_span_token_ids": (THINK, END_THINK),
        },
    ],
)
def test_scope_fields_are_rejected_on_other_estimators(kwargs):
    with pytest.raises(ValidationError, match="only apply to name='opd'"):
        AdvEstimatorConfig(name="grpo", **kwargs)
