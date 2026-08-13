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

from nemo_rl.algorithms.advantage_estimator import (
    GeneralizedAdvantageEstimator,
    OPDAdvantageEstimator,
    ResidualBaselineEstimator,
    TurnLevelGeneralizedAdvantageEstimator,
    homogeneous_group_sample_mask,
)


def _make_estimator():
    return OPDAdvantageEstimator({"name": "opd"}, {})


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


# ===============================================================================
# Residual baseline (decomposed group baseline + residual critic)
# ===============================================================================


class _LossCfg:
    """Minimal stand-in for ClippedPGLossConfig (KL-in-reward off)."""

    use_kl_in_reward = False
    reference_policy_kl_penalty = 0.0
    reference_policy_kl_type = "low_var_kl"


def _gae_cfg(gamma=1, lam=1.0, normalize=False):
    return {
        "gae_lambda": lam,
        "gae_gamma": gamma,
        "normalize_advantages": normalize,
        "gae_lambda_value": None,
        "gae_lambda_policy": None,
        "length_adaptive_alpha": 0.0,
    }


def _make_gae(**kwargs):
    return GeneralizedAdvantageEstimator(_gae_cfg(**kwargs), _LossCfg())


def _one_group(rewards, seq_len=6):
    """One task group of ``len(rewards)`` siblings; every token is a valid response token."""
    b = len(rewards)
    prompt_ids = torch.full((b, 3), 7)  # identical prompt => a single group
    return (
        prompt_ids,
        torch.tensor(rewards, dtype=torch.float32),
        torch.ones(b, seq_len),
    )


def test_residual_loo_arithmetic():
    """R=[1,0,0,0] => Y=[1,-1/3,-1/3,-1/3], summing to zero across siblings."""
    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    prompt_ids, rewards, mask = _one_group([1.0, 0.0, 0.0, 0.0])
    values = torch.zeros_like(mask)

    _, returns = est.compute_advantage(prompt_ids, rewards, mask, values)

    expected = torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3])
    torch.testing.assert_close(returns[:, 0], expected)
    # Constant along the trajectory (sparse terminal reward, gamma = lambda = 1).
    torch.testing.assert_close(returns, expected.unsqueeze(-1).expand_as(returns))
    assert abs(returns[:, 0].sum().item()) < 1e-6


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_residual_homogeneous_group_targets_are_zero(reward):
    """All-fail and all-pass groups both give Y = 0 for every sibling.

    This is why the mixed-group fraction, not the dataset size, bounds what a
    residual critic can learn: homogeneous groups contribute exactly zero target
    variance.
    """
    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    prompt_ids, rewards, mask = _one_group([reward] * 4)
    values = torch.zeros_like(mask)

    _, returns = est.compute_advantage(prompt_ids, rewards, mask, values)

    torch.testing.assert_close(returns, torch.zeros_like(returns))
    assert est.last_metrics["residual/frac_groups_mixed"] == 0.0


def test_residual_matches_absolute_gae_shifted_by_baseline():
    """The correctness contract: residualization is a pure change of variables.

    Advantages must be IDENTICAL to running plain GAE on a critic that predicts
    ``V~ = C + B_LOO``, and the returns must be exactly that run's returns minus
    ``B_LOO``. If this holds, nothing about the advantage estimate changed --
    only which component the critic is asked to represent.
    """
    prompt_ids, rewards, mask = _one_group([1.0, 0.0, 1.0, 0.0])
    torch.manual_seed(0)
    c_values = torch.randn(4, 6) * 0.1

    residual = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    res_adv, res_returns = residual.compute_advantage(
        prompt_ids, rewards, mask, c_values
    )

    baseline = residual._leave_one_out_baseline(prompt_ids, rewards)
    abs_adv, abs_returns = _make_gae().compute_advantage(
        prompt_ids, rewards, mask, c_values + baseline.unsqueeze(-1)
    )

    torch.testing.assert_close(res_adv, abs_adv)
    torch.testing.assert_close(res_returns, abs_returns - baseline.unsqueeze(-1))


def test_residual_lambda_one_telescopes_to_r_minus_b_minus_c():
    """At gamma = lambda = 1 the advantage is exactly ``A_t = R - B_LOO - C_t``."""
    prompt_ids, rewards, mask = _one_group([1.0, 0.0, 0.0, 1.0])
    torch.manual_seed(1)
    c_values = torch.randn(4, 6) * 0.2

    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    adv, _ = est.compute_advantage(prompt_ids, rewards, mask, c_values)

    baseline = est._leave_one_out_baseline(prompt_ids, rewards)
    expected = (rewards - baseline).unsqueeze(-1) - c_values
    torch.testing.assert_close(adv, expected)


def test_residual_off_is_bitwise_identical_to_plain_gae():
    """residual_baseline=false must not perturb training at all (metrics only)."""
    prompt_ids, rewards, mask = _one_group([1.0, 0.0, 1.0, 0.0])
    torch.manual_seed(2)
    values = torch.randn(4, 6) * 0.3

    wrapped = ResidualBaselineEstimator(_make_gae(), residual_target=False)
    w_adv, w_returns = wrapped.compute_advantage(prompt_ids, rewards, mask, values)
    p_adv, p_returns = _make_gae().compute_advantage(prompt_ids, rewards, mask, values)

    assert torch.equal(w_adv, p_adv)
    assert torch.equal(w_returns, p_returns)
    # ...but the return-space offsets are still exported, so critic/ev_res can be
    # logged for an absolute critic on the same axis as a residual one.
    torch.testing.assert_close(
        wrapped.last_returns_to_res,
        -wrapped._leave_one_out_baseline(prompt_ids, rewards),
    )
    torch.testing.assert_close(wrapped.last_returns_to_abs, torch.zeros_like(rewards))


def test_residual_rejects_gamma_below_one():
    """gamma < 1 breaks the exact cancellation of B from nonterminal TD errors."""
    with pytest.raises(ValueError, match="gamma == 1"):
        ResidualBaselineEstimator(_make_gae(gamma=0.99), residual_target=True)
    # ...but an absolute run is untouched by that condition.
    ResidualBaselineEstimator(_make_gae(gamma=0.99), residual_target=False)


def test_residual_group_composition_metrics():
    """frac_groups_mixed counts GROUPS, not trajectories."""
    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    # 3 groups of 2: mixed, all-fail, all-pass.
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    mask = torch.ones(6, 4)

    est.compute_advantage(prompt_ids, rewards, mask, torch.zeros_like(mask))

    m = est.last_metrics
    assert m["residual/frac_groups_mixed"] == pytest.approx(1 / 3)
    assert m["residual/frac_groups_all_fail"] == pytest.approx(1 / 3)
    assert m["residual/frac_groups_all_pass"] == pytest.approx(1 / 3)
    assert m["residual/n_singleton_groups"] == 0.0
    assert m["residual/group_size_min"] == 2.0


def test_homogeneous_group_sample_mask_downweights_only_homogeneous():
    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])  # mixed, then all-fail
    mask = torch.ones(4, 4)
    est.compute_advantage(prompt_ids, rewards, mask, torch.zeros_like(mask))

    sample_mask = torch.ones(4)
    assert homogeneous_group_sample_mask(sample_mask, est, 1.0) is None
    scaled = homogeneous_group_sample_mask(sample_mask, est, 0.25)
    torch.testing.assert_close(scaled, torch.tensor([1.0, 1.0, 0.25, 0.25]))


def _turn_spans(b=4, seq_len=6, anchors=(0, 3)):
    from nemo_rl.algorithms.turn_level import TurnSpans

    anchor_mask = torch.zeros(b, seq_len, dtype=torch.long)
    anchor_mask[:, list(anchors)] = 1
    turn_index = torch.zeros(b, seq_len, dtype=torch.int32)
    turn_index[:, anchors[1] :] = 1
    return TurnSpans(
        anchor_mask=anchor_mask,
        turn_index=turn_index,
        anchor_pos=torch.tensor([list(anchors)] * b),
        turn_valid=torch.ones(b, len(anchors), dtype=torch.bool),
        num_turns=torch.full((b,), len(anchors)),
        turn_ntokens=torch.full((b, len(anchors)), seq_len // len(anchors)),
    )


def test_residual_turn_mode_does_not_leak_baseline_off_anchor():
    """Turn returns are anchor-layout; subtracting B unmasked would write -B everywhere.

    The critic's batch pairs these returns with ``token_mask = anchor_mask``, so
    a leaked ``-B_LOO`` at the ~270 non-anchor positions of a real rollout would
    be silently regressed against.
    """
    turn_cfg = {
        "turn_gae_gamma": 1.0,
        "turn_gae_lambda_value": 1.0,
        "turn_gae_lambda_policy": 1.0,
        "normalize_advantages": False,
    }
    inner = TurnLevelGeneralizedAdvantageEstimator(turn_cfg, _LossCfg())
    est = ResidualBaselineEstimator(inner, residual_target=True)

    prompt_ids, rewards, mask = _one_group([1.0, 0.0, 0.0, 0.0])
    spans = _turn_spans()
    _, returns = est.compute_advantage(
        prompt_ids, rewards, mask, torch.zeros_like(mask), turn_spans=spans
    )

    off_anchor = returns[spans.anchor_mask == 0]
    assert torch.equal(off_anchor, torch.zeros_like(off_anchor))
    # Every anchor of a given rollout carries that rollout's residual return.
    expected = torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3])
    torch.testing.assert_close(returns[:, 0], expected)
    torch.testing.assert_close(returns[:, 3], expected)


def test_residual_homogeneity_survives_reward_scaling():
    """Homogeneity is zero within-group reward VARIANCE, not "sum is 0 or G".

    Rewards reaching the estimator are post-scaling / shaping / penalty --
    ppo_math_1B maps [0,1] -> [-1,1] -- so a sum-based test would call an
    all-fail group (sum = -G) "mixed" and report frac_groups_mixed = 1.0 on a
    pool that is mostly homogeneous, while homogeneous_group_weight silently
    weighted nothing.
    """
    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    # DAPO-scaled rewards in [-1, 1]: all-fail, all-pass, mixed.
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
    rewards = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
    mask = torch.ones(6, 4)

    _, returns = est.compute_advantage(
        prompt_ids, rewards, mask, torch.zeros_like(mask)
    )

    m = est.last_metrics
    assert m["residual/frac_groups_mixed"] == pytest.approx(1 / 3)
    assert m["residual/frac_groups_all_fail"] == pytest.approx(1 / 3)
    assert m["residual/frac_groups_all_pass"] == pytest.approx(1 / 3)
    # The Y = 0 claim must hold for the homogeneous groups at ANY reward scale.
    torch.testing.assert_close(returns[:4], torch.zeros_like(returns[:4]))
    assert returns[4, 0].item() > 0 and returns[5, 0].item() < 0
    torch.testing.assert_close(
        est.last_group_homogeneous, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    )


def test_residual_homogeneity_with_fractional_rewards():
    """Partial-credit judge rewards: equal-but-nonbinary siblings are homogeneous."""
    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    rewards = torch.tensor([0.4, 0.4, 0.7, 0.2])
    mask = torch.ones(4, 4)

    _, returns = est.compute_advantage(
        prompt_ids, rewards, mask, torch.zeros_like(mask)
    )

    assert est.last_metrics["residual/frac_groups_mixed"] == pytest.approx(0.5)
    torch.testing.assert_close(returns[:2], torch.zeros_like(returns[:2]))


def test_raw_reward_with_residual_baseline_raises():
    """raw_reward trains no critic, so residualization is meaningless there."""
    from nemo_rl.algorithms.ppo import _create_advantage_estimator

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.ppo = {
        "adv_estimator": {
            "name": "raw_reward",
            "normalize_advantages": False,
            "residual_baseline": True,
        }
    }
    cfg.loss_fn = _LossCfg()
    with pytest.raises(ValueError, match="requires a value model"):
        _create_advantage_estimator(cfg)


def test_homogeneous_group_sample_mask_rejects_missing_group_info():
    """Fail loud rather than silently ignoring the knob on a non-residual run."""
    with pytest.raises(ValueError, match="requires the residual baseline"):
        homogeneous_group_sample_mask(torch.ones(2), object(), 0.5)
    with pytest.raises(ValueError, match="must be >= 0"):
        homogeneous_group_sample_mask(torch.ones(2), object(), -1.0)


def test_attach_value_baseline_keys_rejects_size_mismatch():
    """Per-sample offsets that do not line up would misalign EV silently."""
    from nemo_rl.algorithms.advantage_estimator import attach_value_baseline_keys

    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    prompt_ids, rewards, mask = _one_group([1.0, 0.0, 0.0, 0.0])
    est.compute_advantage(prompt_ids, rewards, mask, torch.zeros_like(mask))

    with pytest.raises(ValueError, match="misalign"):
        attach_value_baseline_keys({"returns": torch.zeros(3, 6)}, est)


def test_ev_res_mixed_group_isolates_the_homogeneous_tax():
    """Homogeneous groups have Y=0, so any prediction there only costs ev_res.

    Constructed so the critic is PERFECT on the mixed group and merely noisy on
    the homogeneous one: whole-batch ev_res must be dragged below the
    mixed-group-only number, which is what makes the pair diagnostic.
    """
    from nemo_rl.algorithms.ppo import (
        _mixed_group_mask,
        _mixed_group_value_metrics,
    )

    est = ResidualBaselineEstimator(_make_gae(), residual_target=True)
    # group 1 mixed, group 2 all-fail.
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.ones(4, 6)
    _, returns = est.compute_advantage(
        prompt_ids, rewards, mask, torch.zeros_like(mask)
    )

    # Perfect on the mixed pair, wrong on the homogeneous pair (target is 0).
    values = returns.clone()
    values[2:] = 0.4

    mixed = _mixed_group_mask(est)
    torch.testing.assert_close(mixed, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    out = _mixed_group_value_metrics(
        values, returns, mask, mixed, returns_to_res=est.last_returns_to_res
    )
    assert out["critic/ev_res_mixed_group"] == pytest.approx(1.0, abs=1e-5)
    assert out["critic/n_mixed_group_tokens"] == 12.0
    for b in ("early", "mid", "late"):
        assert out[f"critic/ev_res_mixed_group_{b}"] == pytest.approx(1.0, abs=1e-5)

    # Whole-batch ev_res over the same tensors is dragged down by the
    # homogeneous group, which is exactly the effect this metric separates out.
    err = (returns - values).var(unbiased=False)
    whole = 1.0 - err / returns.var(unbiased=False)
    assert whole < out["critic/ev_res_mixed_group"]


def test_ev_res_mixed_group_absent_without_group_info():
    """No residual estimator -> no keys, rather than a misleading zero."""
    from nemo_rl.algorithms.ppo import _mixed_group_value_metrics

    assert (
        _mixed_group_value_metrics(
            torch.zeros(2, 4), torch.zeros(2, 4), torch.ones(2, 4), None
        )
        == {}
    )
