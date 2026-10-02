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
"""Pure critic / advantage diagnostics ported from legacy async PPO."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.legacy_ppo_diagnostics import (
    calibration_ece,
    critic_gnorm_metrics,
    leave_one_out_baseline,
    mixed_group_value_metrics,
    multi_trace_composition_metrics,
    pooled_explained_var,
    positional_value_metrics,
    raw_advantage_metrics,
    residual_baseline_diagnostics,
    return_space_stats,
    trajectory_age_metrics,
    trajectory_policy_age,
)


def _ev(values, returns):
    err = (returns - values).var(unbiased=False)
    return float(1.0 - err / returns.var(unbiased=False))


def test_raw_advantage_metrics_masked_stats_and_gain():
    adv = torch.tensor([[0.0, 1.0, 3.0, 100.0], [0.0, -2.0, 0.0, 0.0]])
    mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.float32)
    m = raw_advantage_metrics(adv, mask, normalize_advantages=True)
    valid = torch.tensor([1.0, 3.0, -2.0])
    assert m["adv_raw/mean"] == pytest.approx(valid.mean().item())
    assert m["adv_raw/std"] == pytest.approx(valid.std(unbiased=False).item())
    assert m["adv_raw/abs_mean"] == pytest.approx(2.0)
    assert m["adv_raw/max_abs"] == pytest.approx(3.0)
    assert m["adv_raw/whiten_gain"] == pytest.approx(1.0 / m["adv_raw/std"], rel=1e-5)
    assert "adv_raw/whiten_gain" not in raw_advantage_metrics(adv, mask, False)
    assert raw_advantage_metrics(adv, torch.zeros_like(mask), True) == {}


def test_gae_estimator_records_pre_whitening_scale_without_changing_advantages():
    from types import SimpleNamespace

    from nemo_rl.algorithms.advantage_estimator import (
        GAEConfig,
        GeneralizedAdvantageEstimator,
    )

    loss_cfg = SimpleNamespace(
        use_kl_in_reward=False,
        reference_policy_kl_penalty=0.0,
        reference_policy_kl_type="k3",
    )
    est = GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0, gae_gamma=1.0, normalize_advantages=True), loss_cfg
    )
    values = torch.tensor([[0.0, 0.2, 0.4, 0.0], [0.0, 0.5, 0.5, 0.0]])
    mask = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=torch.float32)
    rewards = torch.tensor([1.0, 0.0])
    adv, ret = est.compute_advantage(None, rewards, mask, values)
    # lambda = gamma = 1: raw advantage is R - V(s_t) on response tokens.
    raw = torch.tensor([1.0 - 0.2, 1.0 - 0.4, 0.0 - 0.5, 0.0 - 0.5])
    assert est.last_metrics["adv_raw/mean"] == pytest.approx(raw.mean().item())
    assert est.last_metrics["adv_raw/std"] == pytest.approx(
        raw.std(unbiased=False).item()
    )
    # The diagnostic does not touch the whitened output.
    whitened = (raw - raw.mean()) / torch.sqrt(raw.var(unbiased=True) + 1e-8)
    assert adv[mask.bool()].tolist() == pytest.approx(whitened.tolist(), abs=1e-4)
    assert ret[0, 1].item() == pytest.approx(1.0)


def test_pooled_explained_var_both_spaces():
    values = torch.tensor([[0.0, 0.2, 0.6], [0.0, 0.4, 0.4], [0.0, 9.0, 9.0]])
    returns = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    token_mask = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=torch.float32)
    sample_mask = torch.tensor([1.0, 1.0, 0.0])  # row 2 excluded like the loss
    b_loo = torch.tensor([0.25, 0.75, 0.5])
    ev_abs, ev_res = pooled_explained_var(
        values, returns, token_mask, sample_mask, -b_loo
    )
    v = torch.tensor([0.2, 0.6, 0.4, 0.4])
    r = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert ev_abs == pytest.approx(_ev(v, r), rel=1e-5)
    y = r - torch.tensor([0.25, 0.25, 0.75, 0.75])
    err = (r - v).var(unbiased=False)
    assert ev_res == pytest.approx(float(1 - err / y.var(unbiased=False)), rel=1e-5)
    # No offsets: both slots collapse to the historical number.
    a, b = pooled_explained_var(values, returns, token_mask, sample_mask)
    assert a == pytest.approx(b)
    # Degenerate target variance reports 0.0, not -1e8.
    flat = torch.ones_like(returns)
    assert pooled_explained_var(values, flat, token_mask, sample_mask) == (0.0, 0.0)


def test_calibration_ece():
    v = torch.tensor([0.05, 0.05, 0.95, 0.95])
    r = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert calibration_ece(v, r) == pytest.approx(0.05)
    assert calibration_ece(torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])) == 0.0


def test_positional_value_metrics_buckets_by_relative_position():
    # One row with 6 response tokens -> 2 per third.
    values = torch.tensor([[0.0, 0.1, 0.2, 0.5, 0.5, 0.9, 0.8]])
    returns = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
    token_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    m = positional_value_metrics(values, returns, token_mask)
    assert m["critic/n_tokens_early"] == 2.0
    assert m["critic/n_tokens_late"] == 2.0
    v_e, r_e = torch.tensor([0.1, 0.2]), torch.tensor([0.0, 1.0])
    assert m["critic/ev_early"] == pytest.approx(_ev(v_e, r_e), rel=1e-5)
    assert m["critic/bias_early"] == pytest.approx((v_e - r_e).mean().item())
    assert m["critic/abs_err_mid"] == pytest.approx(0.5)
    assert m["critic/mean_v_late"] == pytest.approx(0.85)
    assert m["critic/ece_late"] == pytest.approx(
        calibration_ece(torch.tensor([0.9, 0.8]), torch.tensor([1.0, 1.0]))
    )
    # Late bucket targets are constant -> EV reported as 0.0.
    assert m["critic/ev_late"] == 0.0
    # Residual offset shifts only the denominator.
    shifted = positional_value_metrics(
        values, returns, token_mask, returns_to_res=torch.tensor([-0.5])
    )
    assert shifted["critic/ev_res_early"] == pytest.approx(m["critic/ev_early"])
    assert positional_value_metrics(values, returns, torch.zeros_like(values)) == {}


def test_mixed_group_value_metrics_restricts_to_mixed_rows():
    values = torch.tensor([[0.0, 0.3, 0.7], [0.0, 0.6, 0.2], [0.0, 0.5, 0.5]])
    returns = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    token_mask = torch.tensor([[0, 1, 1]] * 3, dtype=torch.float32)
    mixed = torch.tensor([1.0, 1.0, 0.0])
    m = mixed_group_value_metrics(values, returns, token_mask, mixed)
    assert m["critic/n_mixed_group_tokens"] == 4.0
    v = torch.tensor([0.3, 0.7, 0.6, 0.2])
    r = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert m["critic/ev_res_mixed_group"] == pytest.approx(_ev(v, r), rel=1e-5)
    assert mixed_group_value_metrics(values, returns, token_mask, None) == {}


def test_residual_baseline_diagnostics_matches_loo_and_group_composition():
    # Groups: 0 = mixed {1, 0, 1}, 1 = all-fail {0, 0}, 2 = singleton {1}.
    group_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    baseline = leave_one_out_baseline(group_ids, rewards)
    assert baseline.tolist() == pytest.approx([0.5, 1.0, 0.5, 0.0, 0.0, 1.0])
    returns = rewards.unsqueeze(-1).expand(6, 3).clone()
    mask = torch.ones(6, 3)
    sample_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    m, to_res, homogeneous = residual_baseline_diagnostics(
        group_ids, rewards, returns, mask, sample_mask
    )
    assert to_res.tolist() == pytest.approx((-baseline).tolist())
    assert homogeneous.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert m["residual/frac_groups_mixed"] == pytest.approx(1 / 3)
    assert m["residual/frac_groups_all_fail"] == pytest.approx(1 / 3)
    assert m["residual/frac_groups_all_pass"] == pytest.approx(1 / 3)
    assert m["residual/n_singleton_groups"] == 1.0
    assert (m["residual/group_size_min"], m["residual/group_size_max"]) == (1.0, 3.0)
    assert m["residual/frac_traj_sample_masked"] == pytest.approx(1 / 6)
    assert m["residual/b_loo_mean"] == pytest.approx(baseline.mean().item())
    assert m["residual/target_var"] == pytest.approx(returns.var(unbiased=False).item())


def test_return_space_stats_masked_means():
    returns = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.float32)
    to_res = torch.tensor([-0.5, -0.25])
    m = return_space_stats(returns, mask, to_res)
    assert m["critic/abs_returns_mean"] == pytest.approx(2 / 3)
    assert m["critic/abs_returns_sq_mean"] == pytest.approx(2 / 3)
    assert m["critic/res_returns_mean"] == pytest.approx((0.5 + 0.5 - 0.25) / 3)
    assert m["critic/res_returns_sq_mean"] == pytest.approx((0.25 + 0.25 + 0.0625) / 3)
    assert return_space_stats(returns, torch.zeros_like(mask), None) == {}


def test_critic_gnorm_metrics_fractions_sum_to_one():
    m = critic_gnorm_metrics({"moe": torch.tensor(3.0), "value_head": 4.0})
    assert m["critic/gnorm/moe"] == 3.0
    assert m["critic/gnorm_frac/moe"] == pytest.approx(9 / 25)
    assert m["critic/gnorm_frac/value_head"] == pytest.approx(16 / 25)
    assert critic_gnorm_metrics(None) == {}


def test_trajectory_ages_are_freeze_aware():
    # Warmup W=5: versions <= 5 are all pi_0.
    assert trajectory_policy_age(3, 6, 5) == 1
    assert trajectory_policy_age(4, 5, 5) == 0
    assert trajectory_policy_age(7, 9, 0) == 2
    m = trajectory_age_metrics([3, 5], 6, 5)
    assert m["avg_trajectory_age"] == pytest.approx(2.0)
    assert m["avg_trajectory_policy_age"] == pytest.approx(1.0)
    assert m["max_trajectory_policy_age"] == 1.0
    assert trajectory_age_metrics([], 6, 5) == {}


def test_multi_trace_composition_metrics():
    m = multi_trace_composition_metrics(
        trace_rollout_ids=[0, 0, 1, 2, 2],
        sample_mask=torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
        trace_lengths=torch.tensor([10, 20, 30, 40, 50]),
        num_rows=8,
    )
    assert m["multi_trace/num_traces"] == 5
    assert m["multi_trace/num_rollouts"] == 3
    assert m["multi_trace/traces_per_rollout_mean"] == pytest.approx(5 / 3)
    assert m["multi_trace/traces_per_rollout_max"] == 2
    assert m["multi_trace/padding_rows"] == 3
    assert m["multi_trace/masked_trace_fraction"] == pytest.approx(0.8)
    assert m["multi_trace/fully_masked_rollout_fraction"] == pytest.approx(2 / 3)
    assert m["multi_trace/mean_trace_length"] == pytest.approx(30.0)
    assert m["multi_trace/max_trace_length"] == 50


def test_seq_logprob_error_masking_tensor_out_matches_legacy_contract():
    from nemo_rl.algorithms.grpo import compute_and_apply_seq_logprob_error_masking
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    token_mask = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=torch.float32)
    gen = torch.zeros(3, 3)
    prev = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": torch.tensor([1.0, 1.0, 0.0]),
            "prev_logprobs": prev,
            "generation_logprobs": gen,
        }
    )
    out: dict = {}
    metrics = compute_and_apply_seq_logprob_error_masking(
        data, torch.tensor([1.0, 1.0, 0.0]), 2.0, tensor_out=out
    )
    assert metrics["num_masked_seqs"] == 1
    assert out["pre_seq_error_sample_loss_mask"].tolist() == [1.0, 1.0, 0.0]
    assert out["seq_mult_prob_error"][1].item() == pytest.approx(float(torch.e**2))
    assert out["seq_mult_prob_error"][2].item() == 0.0
    assert out["masked_by_seq_logprob_error"].tolist() == [False, True, False]
    # Metrics-only (no threshold): nothing masked, tensors still exported.
    data["sample_mask"] = torch.tensor([1.0, 1.0, 0.0])
    out2: dict = {}
    compute_and_apply_seq_logprob_error_masking(
        data, torch.zeros(3), None, tensor_out=out2
    )
    assert out2["masked_by_seq_logprob_error"].tolist() == [False, False, False]
