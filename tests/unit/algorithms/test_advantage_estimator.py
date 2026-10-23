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
    PositiveLengthReweightStats,
    rank_length_weights,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt


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


# ---------------------------------------------------------------------------
# GRPO positive length reweighting (grpo.adv_estimator.positive_length_reweight)
# ---------------------------------------------------------------------------

RANK = dict(enabled=True, mode="rank", rank_spread=0.5)
INVERSE = dict(enabled=True, mode="inverse_length", alpha=1.0)
EASY = ([1.0] * 7 + [0.0], [80, 120, 150, 200, 250, 300, 500, 400])
MEDIUM = ([1.0] * 4 + [0.0] * 4, [300, 400, 600, 900, 500, 700, 1200, 1500])
HARD = ([1.0] + [0.0] * 7, [1200, 800, 1000, 1500, 2000, 2000, 1800, 1600])


def _grpo_estimator(**reweight):
    """GRPO with LOO baseline and normalization; ``reweight`` fills positive_length_reweight."""
    return GRPOAdvantageEstimator(
        AdvEstimatorConfig(
            name="grpo",
            use_leave_one_out_baseline=True,
            normalize_rewards=True,
            positive_length_reweight=reweight,
        ),
        {},
    )


def _batch(groups, valid=None):
    """Inputs for ``groups`` = [(rewards, lengths), ...], one prompt id per group.

    Row i's mask holds ``lengths[i]`` ones in ``mask[:, 1:]``; position 0 is the token the loss
    never trains, as in the real ``token_mask``.
    """
    rewards, lengths, prompt_ids = [], [], []
    for group, (group_rewards, group_lengths) in enumerate(groups):
        rewards += list(group_rewards)
        lengths += list(group_lengths)
        prompt_ids += [[group]] * len(group_rewards)
    mask = torch.zeros(len(lengths), max(lengths) + 1)
    for i, n in enumerate(lengths):
        mask[i, 1 : n + 1] = 1.0
    valid_mask = None if valid is None else torch.tensor(valid, dtype=torch.float32)
    return (
        torch.tensor(prompt_ids),
        torch.tensor(rewards, dtype=torch.float32),
        mask,
        valid_mask,
    )


def _advantages(estimator, groups, valid=None):
    """Per-row advantages (the [B, S] output is constant along S)."""
    prompt_ids, rewards, mask, valid_mask = _batch(groups, valid)
    advantages = estimator.compute_advantage(
        prompt_ids, rewards, mask, valid_mask=valid_mask
    )
    assert advantages.shape == mask.shape
    return advantages[:, 0]


def _legacy_grpo_advantages(prompt_ids, rewards, mask, valid_mask):
    """GRPOAdvantageEstimator.compute_advantage as it was before positive_length_reweight."""
    baseline, std, is_trivial_distribution = calculate_baseline_and_std_per_prompt(
        prompt_ids,
        rewards,
        torch.ones_like(rewards) if valid_mask is None else valid_mask.float(),
        leave_one_out_baseline=True,
    )
    advantages = (rewards - baseline).unsqueeze(-1)
    normalize_mask = (std > 0) & ~is_trivial_distribution
    advantages[normalize_mask] = advantages[normalize_mask] / (
        std.unsqueeze(-1)[normalize_mask] + 1e-6
    )
    return advantages.expand(mask.shape)


def _close(actual, expected, atol=1e-3):
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=actual.dtype), atol=atol, rtol=0
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_length_reweight_disabled_is_bit_identical(seed):
    generator = torch.Generator().manual_seed(seed)
    groups = []
    for _ in range(5):
        rewards = torch.rand(8, generator=generator)
        if seed != 2:  # seeds 0 and 1 binary, seed 2 continuous rewards
            rewards = (rewards < 0.5).float()
        lengths = torch.randint(1, 50, (8,), generator=generator)
        groups.append((rewards.tolist(), lengths.tolist()))
    valid = (torch.rand(40, generator=generator) > 0.1).float().tolist()
    prompt_ids, rewards, mask, valid_mask = _batch(groups, valid)
    expected = _legacy_grpo_advantages(prompt_ids, rewards.clone(), mask, valid_mask)

    for estimator in (
        _grpo_estimator(),
        _grpo_estimator(enabled=False, mode="inverse_length", alpha=2.0),
    ):
        actual = estimator.compute_advantage(
            prompt_ids, rewards.clone(), mask, valid_mask=valid_mask
        )
        assert torch.equal(actual, expected)
        assert estimator.last_metrics == {}
        assert estimator.last_length_reweight_stats is None


def test_length_reweight_inverse_length_alpha_zero_matches_disabled():
    groups = [EASY, MEDIUM, HARD, ([0.3, 0.9, 0.1, 0.7], [10, 400, 30, 90])]
    disabled = _advantages(_grpo_estimator(), groups)
    alpha_zero = _advantages(
        _grpo_estimator(enabled=True, mode="inverse_length", alpha=0.0), groups
    )
    assert torch.equal(alpha_zero, disabled)


def test_length_reweight_inverse_length_alpha_one():
    groups = [([1, 1, 0, 0], [200, 600, 200, 1000])]
    _close(_advantages(_grpo_estimator(), groups), [1.1547, 1.1547, -1.1547, -1.1547])
    _close(
        _advantages(_grpo_estimator(**INVERSE), groups),
        [2.3094, 0.7698, -1.1547, -1.1547],
    )


# G = 16 with two correct responses: each correct one's leave-one-out peers hold 1 correct out
# of 15, so A = (14/15) / sqrt(1/15) = +3.6148; each wrong one's hold 2 out of 15, A = -0.3789.
G16_REWARDS = [1.0, 1.0] + [0.0] * 14
G16_WRONG = [-0.3789] * 14


def _g16(correct_lengths):
    """The G = 16 group with the two correct responses at ``correct_lengths`` (wrong: 400)."""
    return [(G16_REWARDS, list(correct_lengths) + [400] * 14)]


def test_length_reweight_g16_advantages_before_reweighting():
    _close(
        _advantages(_grpo_estimator(), _g16((500, 501))),
        [3.6148, 3.6148] + G16_WRONG,
        atol=1e-4,
    )


def test_length_reweight_inverse_length_near_tie_is_near_even():
    # 500 vs 501 tokens: per-token weights 1.001 / 0.999, so each answer keeps ~half the credit.
    out = _advantages(_grpo_estimator(**INVERSE), _g16((500, 501)))
    _close(out, [3.6184, 3.6112] + G16_WRONG, atol=1e-4)


def test_length_reweight_inverse_length_max_ratio_caps_the_per_token_ratio():
    # 100 vs 1000 tokens: the raw 10x ratio is capped at max_ratio = 3.
    out = _advantages(_grpo_estimator(**INVERSE), _g16((100, 1000)))
    _close(out, [9.1760, 3.0587] + G16_WRONG, atol=1e-4)
    assert float(out[0] / out[1]) == pytest.approx(3.0, rel=1e-6)


def test_length_reweight_rank_mode_near_tie_gets_the_full_spread():
    # Rank mode is order-only: 500 vs 501 tokens still get 1 + s / 1 - s, a 75/25 credit split.
    out = _advantages(_grpo_estimator(**RANK), _g16((500, 501)))
    _close(out, [5.4249, 1.8083] + G16_WRONG, atol=1e-4)
    credit = out[:2].double() * torch.tensor([500.0, 501.0], dtype=torch.float64)
    assert float(credit[0] / credit.sum()) == pytest.approx(0.75, abs=1e-3)


def _random_groups(seed, n_groups, size, max_length, min_correct=2):
    """Binary-reward groups with at least ``min_correct`` correct responses each."""
    generator = torch.Generator().manual_seed(seed)
    groups = []
    for _ in range(n_groups):
        rewards = (torch.rand(size, generator=generator) < 0.5).float()
        rewards[:min_correct] = 1.0
        lengths = torch.randint(1, max_length + 1, (size,), generator=generator)
        groups.append((rewards.tolist(), lengths.tolist()))
    return groups


@pytest.mark.parametrize("max_ratio", [None, 3.0])
def test_length_reweight_inverse_length_large_alpha_is_finite(max_ratio):
    groups = _random_groups(seed=3, n_groups=4, size=16, max_length=32768)
    for rewards, lengths in groups:  # the extremes of the range among the positives
        lengths[0], lengths[1] = 1, 32768
    estimator = _grpo_estimator(
        enabled=True, mode="inverse_length", alpha=50.0, max_ratio=max_ratio
    )
    out = _advantages(estimator, groups)
    before = _advantages(_grpo_estimator(), groups)
    assert torch.isfinite(out).all()
    assert all(math.isfinite(value) for value in estimator.last_metrics.values())
    positive = before > 0
    assert torch.equal(out[~positive], before[~positive])
    if max_ratio is None:
        # Unbounded, a 32768x longer positive's credit may round to 0, but never flips sign.
        assert (out[positive] >= 0).all()
    else:
        assert (out[positive] > 0).all()


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_length_reweight_inverse_length_unbounded_matches_plain_power(alpha):
    # max_ratio=None reproduces the original w = L ** -alpha (the 1 / L_min factor cancels).
    groups = _random_groups(seed=11, n_groups=5, size=8, max_length=2000)
    out = _advantages(
        _grpo_estimator(
            enabled=True, mode="inverse_length", alpha=alpha, max_ratio=None
        ),
        groups,
    ).double()
    before = _advantages(_grpo_estimator(), groups).double()
    lengths = torch.tensor([n for _, ls in groups for n in ls], dtype=torch.float64)
    expected = before.clone()
    for group in range(len(groups)):
        rows = torch.arange(8 * group, 8 * group + 8)
        positive = rows[before[rows] > 0]
        a, n = before[positive], lengths[positive]
        raw = n.pow(-alpha)
        expected[positive] = a * raw * (a * n).sum() / (a * n * raw).sum()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("max_ratio", [1.5, 3.0, 10.0])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_length_reweight_inverse_length_weights_are_bounded(max_ratio, alpha):
    groups = _random_groups(seed=5, n_groups=8, size=16, max_length=4000)
    estimator = _grpo_estimator(
        enabled=True, mode="inverse_length", alpha=alpha, max_ratio=max_ratio
    )
    out = _advantages(estimator, groups).double()
    before = _advantages(_grpo_estimator(), groups).double()
    lengths = torch.tensor([n for _, ls in groups for n in ls], dtype=torch.float64)
    tolerance = 1 + 1e-5
    for group in range(len(groups)):
        rows = torch.arange(16 * group, 16 * group + 16)
        positive = rows[before[rows] > 0]
        if positive.numel() < 2:  # all correct: no positive advantage to reweight
            continue
        weights = out[positive] / before[positive]
        assert float(weights.max()) <= max_ratio * tolerance
        assert float(weights.max() / weights.min()) <= max_ratio * tolerance
        torch.testing.assert_close(
            (out[positive] * lengths[positive]).sum(),
            (before[positive] * lengths[positive]).sum(),
            rtol=1e-5,
            atol=0.0,
        )
    assert (
        estimator.last_metrics["adv_len_reweight/weight_max"] <= max_ratio * tolerance
    )


@pytest.mark.parametrize("reweight", [RANK, INVERSE])
def test_length_reweight_single_positive_is_unchanged(reweight):
    groups = [([1, 0, 0, 0], [100, 300, 200, 400])]
    disabled = _advantages(_grpo_estimator(), groups)
    # The positive stays unnormalized: its leave-one-out peers are all 0 (trivial guard).
    _close(disabled, [1.0, -0.5773, -0.5773, -0.5773])
    assert torch.equal(_advantages(_grpo_estimator(**reweight), groups), disabled)


def test_length_reweight_rank_mode():
    groups = [([1, 1, 1, 0], [100, 200, 400, 300])]
    _close(
        rank_length_weights(torch.tensor([100.0, 200.0, 400.0]), 0.5),
        [1.5, 1.0, 0.5],
        atol=1e-12,
    )
    _close(_advantages(_grpo_estimator(), groups), [0.5773, 0.5773, 0.5773, -1.0])
    _close(_advantages(_grpo_estimator(**RANK), groups), [1.1022, 0.7348, 0.3674, -1.0])


def test_length_reweight_rank_mode_ties():
    groups = [([1, 1, 1, 1, 0], [100, 200, 200, 400, 300])]
    _close(
        rank_length_weights(torch.tensor([100.0, 200.0, 200.0, 400.0]), 0.5),
        [1.5, 1.0, 1.0, 0.5],
        atol=1e-12,
    )
    _close(_advantages(_grpo_estimator(), groups), [0.5, 0.5, 0.5, 0.5, -1.0])
    _close(_advantages(_grpo_estimator(**RANK), groups), [0.9, 0.6, 0.6, 0.3, -1.0])


def test_length_reweight_rank_weights_are_evenly_spaced():
    weights = rank_length_weights(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]), 0.5)
    _close(weights, [1.5, 1.25, 1.0, 0.75, 0.5], atol=1e-12)
    assert abs(float(weights.mean()) - 1.0) < 1e-12
    # Only the order matters: shuffled lengths get the matching weights.
    shuffled = rank_length_weights(torch.tensor([30.0, 10.0, 50.0, 20.0, 40.0]), 0.5)
    _close(shuffled, [1.0, 1.5, 0.5, 1.25, 0.75], atol=1e-12)
    with pytest.raises(ValueError):
        rank_length_weights(torch.tensor([10.0]), 0.5)


@pytest.mark.parametrize("reweight", [RANK, INVERSE])
def test_length_reweight_is_permutation_invariant(reweight):
    rewards, lengths = [1, 1, 1, 1, 0], [100, 200, 200, 400, 300]
    perm = [3, 0, 4, 2, 1]
    out = _advantages(_grpo_estimator(**reweight), [(rewards, lengths)])
    out_perm = _advantages(
        _grpo_estimator(**reweight),
        [([rewards[i] for i in perm], [lengths[i] for i in perm])],
    )
    torch.testing.assert_close(out_perm, out[perm], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("reweight", [RANK, INVERSE])
def test_length_reweight_preserves_group_mass(reweight):
    generator = torch.Generator().manual_seed(7)
    groups = [
        (
            torch.rand(8, generator=generator).tolist(),  # non-binary rewards
            torch.randint(1, 400, (8,), generator=generator).tolist(),
        )
        for _ in range(6)
    ]
    prompt_ids, rewards, mask, _ = _batch(groups)
    lengths = mask[:, 1:].sum(-1).double()
    before = _grpo_estimator().compute_advantage(prompt_ids, rewards, mask)[:, 0]
    after = _grpo_estimator(**reweight).compute_advantage(prompt_ids, rewards, mask)[
        :, 0
    ]
    assert not torch.equal(after, before)
    for group in range(len(groups)):
        rows = slice(8 * group, 8 * group + 8)
        b, a, n = before[rows].double(), after[rows].double(), lengths[rows]
        positive = b > 0
        assert int(positive.sum()) >= 2
        assert torch.equal(a[~positive], b[~positive])  # negatives untouched
        assert (a[positive] > 0).all()  # no sign flips
        torch.testing.assert_close(
            (a[positive] * n[positive]).sum(),
            (b[positive] * n[positive]).sum(),
            rtol=1e-5,
            atol=0.0,
        )
        torch.testing.assert_close(
            (a.abs() * n).sum(), (b.abs() * n).sum(), rtol=1e-5, atol=0.0
        )


@pytest.mark.parametrize("reweight", [None, RANK, INVERSE])
def test_length_reweight_keeps_each_prompts_step_share(reweight):
    groups = [EASY, MEDIUM, HARD]
    estimator = _grpo_estimator(**(reweight or {}))
    prompt_ids, rewards, mask, _ = _batch(groups)
    advantages = estimator.compute_advantage(prompt_ids, rewards, mask)[:, 0]
    mass = (advantages.double().abs() * mask[:, 1:].sum(-1).double()).view(3, 8).sum(-1)
    _close(mass / mass.sum(), [0.0787, 0.5107, 0.4107])
    disabled = _advantages(_grpo_estimator(), groups)
    assert torch.equal(advantages[16:], disabled[16:])  # hard group: single positive
    if reweight is not None:
        assert not torch.equal(advantages[:16], disabled[:16])
        metrics = estimator.last_metrics
        assert metrics["adv_len_reweight/frac_groups_reweighted"] == pytest.approx(
            2 / 3
        )
        assert metrics["adv_len_reweight/share_bucket_ge_0.75"] == pytest.approx(
            0.0787, abs=1e-3
        )
        assert metrics["adv_len_reweight/share_bucket_0.25_0.75"] == pytest.approx(
            0.5107, abs=1e-3
        )
        assert metrics["adv_len_reweight/share_bucket_le_0.25"] == pytest.approx(
            0.4107, abs=1e-3
        )
        assert metrics["adv_len_reweight/share_max_abs_diff"] < 1e-9
        assert (
            metrics["adv_len_reweight/weight_min"]
            <= metrics["adv_len_reweight/weight_mean"]
            <= metrics["adv_len_reweight/weight_max"]
        )


def test_length_reweight_ignores_invalid_rows():
    # Row 4 is the shortest correct response but invalid: it must not take part.
    groups = [([1, 1, 1, 0, 1], [100, 200, 400, 300, 50])]
    valid = [1, 1, 1, 1, 0]
    with_invalid = _advantages(_grpo_estimator(**RANK), groups, valid)
    alone = _advantages(_grpo_estimator(**RANK), [([1, 1, 1, 0], [100, 200, 400, 300])])
    torch.testing.assert_close(with_invalid[:4], alone, rtol=1e-6, atol=1e-7)
    _close(with_invalid[:4], [1.1022, 0.7348, 0.3674, -1.0])
    disabled = _advantages(_grpo_estimator(), groups, valid)
    assert torch.equal(with_invalid[4], disabled[4])


@pytest.mark.parametrize("reweight", [RANK, INVERSE])
def test_length_reweight_is_shard_invariant(reweight):
    first = ([1, 1, 1, 0], [100, 200, 400, 300])
    second = ([1, 0, 1, 1, 0], [50, 60, 70, 80, 90])
    joint_estimator = _grpo_estimator(**reweight)
    joint = _advantages(joint_estimator, [first, second])
    shard_estimator = _grpo_estimator(**reweight)
    separate = torch.cat(
        [_advantages(shard_estimator, [first]), _advantages(shard_estimator, [second])]
    )
    assert torch.equal(joint, separate)

    # The SingleController merges the per-call statistics of a step's shards.
    merged = PositiveLengthReweightStats.empty()
    for group in (first, second):
        _advantages(shard_estimator, [group])
        merged.merge(shard_estimator.last_length_reweight_stats)
    assert merged.to_metrics() == pytest.approx(joint_estimator.last_metrics)


def test_length_reweight_stats_without_reweighted_groups():
    metrics = PositiveLengthReweightStats.empty().to_metrics()
    assert metrics == {"adv_len_reweight/frac_groups_reweighted": 0.0}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="gdpo", positive_length_reweight={"enabled": True}),
        dict(name="reinforce_plus_plus", positive_length_reweight={"enabled": True}),
        dict(name="opd", positive_length_reweight={"enabled": True}),
        dict(positive_length_reweight={"rank_spread": 1.0}),
        dict(positive_length_reweight={"rank_spread": -0.1}),
        dict(positive_length_reweight={"alpha": -0.5}),
        dict(positive_length_reweight={"max_ratio": 1.0}),
        dict(positive_length_reweight={"max_ratio": 0.5}),
        # Unknown keys are rejected, so a typo cannot silently fall back to the default.
        dict(positive_length_reweight={"enabled": True, "rank_sprad": 0.25}),
    ],
)
def test_length_reweight_config_validation(kwargs):
    with pytest.raises(ValueError):
        AdvEstimatorConfig(**kwargs)


def test_length_reweight_config_disabled_is_allowed_with_any_estimator():
    config = AdvEstimatorConfig(
        name="gdpo", positive_length_reweight={"enabled": False}
    )
    assert not config.positive_length_reweight.enabled
    default = AdvEstimatorConfig().positive_length_reweight
    assert (default.mode, default.alpha, default.max_ratio) == (
        "inverse_length",
        1.0,
        3.0,
    )
    assert (
        AdvEstimatorConfig(
            positive_length_reweight={"max_ratio": None}
        ).positive_length_reweight.max_ratio
        is None
    )


def test_length_reweight_accepts_the_upstream_default_block():
    upstream_block = dict(
        enabled=True,
        mode="inverse_length",
        rank_spread=0.5,
        alpha=1.0,
        max_ratio=3.0,
        scope="trajectory",
        exclude_final_answer=False,
        turns_counted="none",
    )
    config = AdvEstimatorConfig(positive_length_reweight=upstream_block)
    assert config.positive_length_reweight.enabled


def test_length_reweight_response_lengths_override_the_mask():
    # Two correct rows whose masks say 400 / 200 tokens but whose rollouts train 200 / 400 (the
    # SingleController's rollout totals over segment rows): the override decides the weights.
    rewards, mask_lengths = [1.0, 1.0, 0.0, 0.0], [400, 200, 300, 300]
    prompt_ids, rewards_t, mask, _ = _batch([(rewards, mask_lengths)])
    estimator = _grpo_estimator(enabled=True, alpha=1.0, max_ratio=None)
    by_mask = estimator.compute_advantage(prompt_ids, rewards_t.clone(), mask)[:, 0]
    overridden = estimator.compute_advantage(
        prompt_ids,
        rewards_t.clone(),
        mask,
        response_lengths=torch.tensor([200.0, 400.0, 300.0, 300.0]),
    )[:, 0]
    assert by_mask[1] > by_mask[0]
    assert overridden[0] > overridden[1]
    torch.testing.assert_close(overridden[0] / overridden[1], torch.tensor(2.0))
    # Negatives are untouched either way.
    torch.testing.assert_close(overridden[2:], by_mask[2:])
