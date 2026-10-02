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
"""Pure critic / advantage diagnostics ported from legacy async PPO (jiaqiz/ppo-dev).

Everything here is diagnostics only: it reads tensors the advantage stage
already holds and never feeds back into advantages, returns or masks. Kept
torch-only (no Ray, no model code) so advantage_estimator can import
:func:`raw_advantage_metrics` without an import cycle and every function is
unit-testable on CPU.

Legacy sources (nemo_rl/algorithms/ppo.py unless noted):

* :func:`raw_advantage_metrics` -- advantage_estimator.raw_advantage_metrics
* :func:`pooled_explained_var` -- _pooled_explained_var
* :func:`calibration_ece`, :func:`positional_value_metrics` --
  _calibration_ece, _positional_value_metrics
* :func:`mixed_group_value_metrics` -- _mixed_group_value_metrics
* :func:`residual_baseline_diagnostics` -- ResidualBaselineEstimator with
  ``residual_target=False`` (B_LOO for metrics only; the SC port has no residual
  critic, so the targets are always absolute)
* :func:`critic_gnorm_metrics` -- the gnorm block of _compute_critic_metrics
* :func:`return_space_stats` -- MseValueLossFn's abs/res return statistics
* :func:`trajectory_age_metrics` -- _async_trajectory_policy_age + the replay
  buffer's avg_trajectory_age
* :func:`multi_trace_composition_metrics` -- the ``multi_trace/*`` block of
  async_ppo_train's metrics section
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch


def raw_advantage_metrics(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    normalize_advantages: bool,
) -> dict[str, float]:
    """Stats on the PRE-whitening advantages, over valid tokens.

    ``normalize_advantages`` rescales advantages to unit std every step, so the
    post-whitening spread is 1.0 by construction and carries no information.
    The pre-whitening spread does: with lambda=1 the advantage is ``R - V(s)``,
    so ``adv_raw/std`` is the critic's residual scale and should SHRINK as the
    critic improves. ``whiten_gain`` is the amplification whitening applies
    (1/std).
    """
    m = mask.bool()
    if int(m.sum()) < 2:
        return {}
    a = advantages[m].float()
    std = a.std(unbiased=False)
    metrics = {
        "adv_raw/mean": a.mean().item(),
        "adv_raw/std": std.item(),
        "adv_raw/abs_mean": a.abs().mean().item(),
        "adv_raw/max_abs": a.abs().max().item(),
    }
    if normalize_advantages:
        metrics["adv_raw/whiten_gain"] = (1.0 / (std + 1e-8)).item()
    return metrics


def _offset(offset: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if offset is None:
        return torch.zeros(
            reference.shape[0], device=reference.device, dtype=reference.dtype
        )
    return offset.to(device=reference.device, dtype=reference.dtype)


def _explained_var(err_var: torch.Tensor, target: torch.Tensor) -> float:
    """``1 - Var(err) / Var(target)``; ``Var(target) <= 1e-8`` reports 0.0."""
    var_t = target.var(unbiased=False)
    return (1.0 - err_var / var_t).item() if var_t > 1e-8 else 0.0


def _position_bins(token_mask: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """``[B, S]`` masks of the early / mid / late thirds of each response."""
    full = token_mask.bool()
    rel = (torch.cumsum(full.long(), dim=1) - 1).float() / full.sum(
        dim=1, keepdim=True
    ).clamp(min=1).float()
    names = ("early", "mid", "late")
    bins = []
    for i, name in enumerate(names):
        lo, hi = i / len(names), (i + 1) / len(names)
        upper = (rel < hi) if i < len(names) - 1 else (rel <= 1.0)
        bins.append((name, (rel >= lo) & upper))
    return bins


def pooled_explained_var(
    values: torch.Tensor,
    returns: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    returns_to_res: Optional[torch.Tensor] = None,
) -> tuple[float, float]:
    """Pooled pre-update EV in BOTH return spaces, as ``(absolute, residual)``.

    Computed from the rollout-time values (the exact tensors GAE consumed), so it
    describes the critic BEFORE any update this step. Masked like the value loss
    (``token_mask * sample_mask``). The prediction error is offset-invariant
    (``R - (B_LOO + C) == (R - B_LOO) - C``), so one numerator serves both and
    only the denominator changes. ``Var(target) <= 1e-8`` reports 0.0.
    """
    mask = (token_mask * sample_mask.unsqueeze(-1)).bool()
    if int(mask.sum()) < 2:
        return 0.0, 0.0
    err_var = (returns[mask].float() - values[mask].float()).var(unbiased=False)
    res_returns = returns + _offset(returns_to_res, returns).unsqueeze(-1)
    return (
        _explained_var(err_var, returns[mask].float()),
        _explained_var(err_var, res_returns[mask].float()),
    )


def calibration_ece(v: torch.Tensor, r: torch.Tensor) -> float:
    """Expected Calibration Error of V as an estimate of P(success).

    Tokens are binned by predicted value (clamped to [0, 1]) into 10 bins; ECE is
    the token-weighted mean |mean(V) - mean(R)| over bins.
    """
    n_conf_bins = 10
    vc = v.clamp(0.0, 1.0)
    n_total = v.numel()
    ece = 0.0
    for i in range(n_conf_bins):
        lo, hi = i / n_conf_bins, (i + 1) / n_conf_bins
        m = (vc >= lo) & ((vc < hi) if i < n_conf_bins - 1 else (vc <= 1.0))
        n = int(m.sum())
        if n == 0:
            continue
        ece += (n / n_total) * abs((v[m].mean() - r[m].mean()).item())
    return ece


def positional_value_metrics(
    values: torch.Tensor,
    returns: torch.Tensor,
    token_mask: torch.Tensor,
    returns_to_res: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Critic-quality diagnostics bucketed by relative position within each response.

    Early / mid / late thirds: explained variance (``critic/ev_*`` absolute,
    ``critic/ev_res_*`` residual space), calibration (``critic/ece_*``, scored in
    absolute space), signed bias (``critic/bias_*`` = mean(V - R)), mean abs
    error, raw mean value and token count. Legacy scores this over
    ``token_mask`` alone (no sample mask), so env-masked and pad rows count.
    """
    mask = token_mask.bool()
    if int(mask.sum()) < 2:
        return {}
    res_returns = returns + _offset(returns_to_res, returns).unsqueeze(-1)
    out: dict[str, float] = {}
    for name, position in _position_bins(token_mask):
        bmask = mask & position
        n = int(bmask.sum())
        if n < 2:
            continue
        v = values[bmask].float()
        r = returns[bmask].float()
        err_var = (r - v).var(unbiased=False)
        out[f"critic/ev_{name}"] = _explained_var(err_var, r)
        out[f"critic/ev_res_{name}"] = _explained_var(
            err_var, res_returns[bmask].float()
        )
        out[f"critic/abs_err_{name}"] = (r - v).abs().mean().item()
        out[f"critic/mean_v_{name}"] = v.mean().item()
        out[f"critic/n_tokens_{name}"] = float(n)
        out[f"critic/ece_{name}"] = calibration_ece(v, r)
        out[f"critic/bias_{name}"] = (v - r).mean().item()
    return out


def mixed_group_value_metrics(
    values: torch.Tensor,
    returns: torch.Tensor,
    token_mask: torch.Tensor,
    mixed_mask: Optional[torch.Tensor],
    returns_to_res: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Residual explained variance restricted to MIXED-outcome groups.

    Homogeneous groups have residual target ``Y = 0``; a nonzero prediction
    there is a pure penalty to the whole-batch ``critic/ev_res``. These keys
    isolate where the target is actually nonzero. Diagnostic only.
    """
    if mixed_mask is None:
        return {}
    mask = token_mask.bool() & mixed_mask.bool().unsqueeze(-1)
    if int(mask.sum()) < 2:
        return {}
    res_returns = returns + _offset(returns_to_res, returns).unsqueeze(-1)

    def _ev(m: torch.Tensor) -> Optional[float]:
        if int(m.sum()) < 2:
            return None
        err_var = (returns[m] - values[m]).float().var(unbiased=False)
        return _explained_var(err_var, res_returns[m].float())

    out: dict[str, float] = {}
    overall = _ev(mask)
    if overall is not None:
        out["critic/ev_res_mixed_group"] = overall
    out["critic/n_mixed_group_tokens"] = float(int(mask.sum()))
    for name, position in _position_bins(token_mask):
        ev = _ev(mask & position)
        if ev is not None:
            out[f"critic/ev_res_mixed_group_{name}"] = ev
    return out


def leave_one_out_baseline(
    group_ids: torch.Tensor, rewards: torch.Tensor
) -> torch.Tensor:
    """``B_LOO[i] = (sum_{k in g(i)} R_k - R_i) / (G - 1)``, fp32.

    Same call legacy made (calculate_baseline_and_std_per_prompt with an
    all-ones valid mask), so rows dropped from the loss still feed their
    siblings' baselines, and a group of one falls back to its own reward.
    """
    from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt

    rewards_f32 = rewards.float()
    result = calculate_baseline_and_std_per_prompt(
        group_ids.reshape(-1, 1),
        rewards_f32,
        torch.ones_like(rewards_f32),
        leave_one_out_baseline=True,
    )
    return result[0]


def residual_baseline_diagnostics(
    group_ids: torch.Tensor,
    rewards: torch.Tensor,
    returns: torch.Tensor,
    return_mask: torch.Tensor,
    sample_mask: Optional[torch.Tensor],
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """``residual/*`` metrics plus the per-row offset the critic metrics need.

    Port of ResidualBaselineEstimator with ``residual_target=False``: the batch
    stays in absolute space and B_LOO is computed for diagnostics only, so the
    residual-space offset is ``returns_to_res = -B_LOO``. "Homogeneous" means
    zero within-group reward variance (all-fail / all-pass under any scale).

    Args:
        group_ids: ``[B]`` sibling-group index per row (legacy: the positional
            prompt-group id, so a rollout's traces share their group).
        rewards: ``[B]`` per-row reward.
        returns: ``[B, S]`` absolute critic targets from GAE.
        return_mask: ``[B, S]`` positions the returns live at (the GAE mask).
        sample_mask: ``[B]`` loss sample mask, for ``frac_traj_sample_masked``.

    Returns:
        ``(metrics, returns_to_res, group_homogeneous)``, the last two ``[B]``.
    """
    baseline = leave_one_out_baseline(group_ids, rewards)
    returns_to_res = -baseline
    with torch.no_grad():
        _, inverse = torch.unique(group_ids.reshape(-1), return_inverse=True)
        gids = inverse.reshape(-1)
        rewards_f32 = rewards.float()
        n_groups = int(gids.max().item()) + 1 if gids.numel() else 0
        if n_groups == 0:
            return {}, returns_to_res, torch.zeros(0)
        ones = torch.ones_like(rewards_f32)
        counts = torch.zeros(n_groups).index_add_(0, gids, ones)
        sums = torch.zeros(n_groups).index_add_(0, gids, rewards_f32)
        means = sums / counts.clamp(min=1)
        sq_dev = torch.zeros(n_groups).index_add_(
            0, gids, (rewards_f32 - means[gids]) ** 2
        )
        homogeneous = (sq_dev <= 1e-12).float()
        mixed = 1.0 - homogeneous
        r_min, r_max = rewards_f32.min(), rewards_f32.max()
        all_fail = homogeneous * (means <= r_min + 1e-12).float()
        all_pass = homogeneous * (means >= r_max - 1e-12).float()
        singletons = int((counts < 2).sum().item())

        m = return_mask.bool()
        target_var = (
            returns[m].float().var(unbiased=False).item() if int(m.sum()) > 1 else 0.0
        )
        metrics = {
            "residual/b_loo_mean": baseline.mean().item(),
            "residual/b_loo_std": baseline.std(unbiased=False).item(),
            "residual/target_var": target_var,
            "residual/frac_groups_mixed": mixed.mean().item(),
            "residual/frac_groups_all_fail": all_fail.mean().item(),
            "residual/frac_groups_all_pass": all_pass.mean().item(),
            "residual/n_singleton_groups": float(singletons),
            "residual/group_size_min": counts.min().item(),
            "residual/group_size_max": counts.max().item(),
        }
        if sample_mask is not None:
            metrics["residual/frac_traj_sample_masked"] = (
                1.0 - sample_mask.float().mean().item()
            )
    return metrics, returns_to_res, homogeneous[gids]


def return_space_stats(
    returns: torch.Tensor,
    mask: torch.Tensor,
    returns_to_res: Optional[torch.Tensor],
) -> dict[str, float]:
    """``critic/{abs,res}_returns_{mean,sq_mean}`` over the value-loss mask.

    Legacy computed these inside MseValueLossFn as sums of per-microbatch
    ``masked_mean(..., global_valid_toks)``, i.e. the masked mean over the whole
    critic batch. They depend only on the returns and the mask, never on the
    critic's output, so the controller computes the identical number from the
    returns it just wrote back (``mask = token_mask * sample_mask``).
    """
    m = mask.float()
    denom = m.sum()
    if float(denom) <= 0:
        return {}
    abs_target = returns.float()
    res_target = abs_target + _offset(returns_to_res, returns).float().unsqueeze(-1)
    out: dict[str, float] = {}
    for name, target in (("abs", abs_target), ("res", res_target)):
        out[f"critic/{name}_returns_mean"] = float((target * m).sum() / denom)
        out[f"critic/{name}_returns_sq_mean"] = float((target**2 * m).sum() / denom)
    return out


def critic_gnorm_metrics(
    grad_norm_groups: Optional[Mapping[str, Any]],
) -> dict[str, float]:
    """``critic/gnorm/{group}`` and ``critic/gnorm_frac/{group}`` (share of norm^2)."""
    gn_groups = grad_norm_groups or {}
    out: dict[str, float] = {}
    if not gn_groups:
        return out
    sq = {k: float(v) ** 2 for k, v in gn_groups.items()}
    sq_total = sum(sq.values())
    for k, v in gn_groups.items():
        out[f"critic/gnorm/{k}"] = float(v)
        if sq_total > 0:
            out[f"critic/gnorm_frac/{k}"] = sq[k] / sq_total
    return out


def trajectory_policy_age(
    gen_version: int, current_weight_version: int, policy_training_start_step: int
) -> int:
    """Off-policy staleness in POLICY updates (frozen warmup versions count once)."""
    return max(0, current_weight_version - policy_training_start_step) - max(
        0, gen_version - policy_training_start_step
    )


def trajectory_age_metrics(
    gen_versions: Sequence[int],
    current_weight_version: int,
    policy_training_start_step: int,
) -> dict[str, float]:
    """``avg_trajectory_age`` / ``avg_trajectory_policy_age`` / ``max_trajectory_policy_age``.

    One entry per sampled prompt group, as legacy's replay buffer reported.
    """
    if not gen_versions:
        return {}
    avg_age = current_weight_version - sum(gen_versions) / len(gen_versions)
    policy_ages = [
        trajectory_policy_age(g, current_weight_version, policy_training_start_step)
        for g in gen_versions
    ]
    return {
        "avg_trajectory_age": float(avg_age),
        "avg_trajectory_policy_age": sum(policy_ages) / len(policy_ages),
        "max_trajectory_policy_age": float(max(policy_ages)),
    }


def multi_trace_composition_metrics(
    *,
    trace_rollout_ids: Sequence[int],
    sample_mask: torch.Tensor,
    trace_lengths: torch.Tensor,
    num_rows: int,
) -> dict[str, float]:
    """Legacy ``multi_trace/*`` batch composition, over the UNPADDED trace rows.

    Args:
        trace_rollout_ids: per unpadded row, its rollout's step-unique id.
        sample_mask: ``[n]`` loss sample mask of the unpadded rows.
        trace_lengths: ``[n]`` total token length of each unpadded row.
        num_rows: row count including DP padding.
    """
    n = len(trace_rollout_ids)
    if n == 0:
        return {}
    rollout_trace_counts: dict[int, int] = {}
    rollout_has_unmasked: dict[int, bool] = {}
    for rollout_id, is_unmasked in zip(
        trace_rollout_ids, (sample_mask[:n] > 0).tolist()
    ):
        rollout_trace_counts[rollout_id] = rollout_trace_counts.get(rollout_id, 0) + 1
        rollout_has_unmasked[rollout_id] = (
            rollout_has_unmasked.get(rollout_id, False) or is_unmasked
        )
    num_rollouts = max(len(rollout_trace_counts), 1)
    lengths = trace_lengths[:n].float()
    return {
        "multi_trace/num_traces": int(n),
        "multi_trace/num_rollouts": len(rollout_trace_counts),
        "multi_trace/traces_per_rollout_mean": n / num_rollouts,
        "multi_trace/traces_per_rollout_max": max(
            rollout_trace_counts.values(), default=0
        ),
        "multi_trace/padding_rows": int(num_rows - n),
        "multi_trace/masked_trace_fraction": float(
            (sample_mask[:n] <= 0).float().mean().item()
        ),
        "multi_trace/fully_masked_rollout_fraction": sum(
            1 for has_unmasked in rollout_has_unmasked.values() if not has_unmasked
        )
        / num_rollouts,
        "multi_trace/mean_trace_length": float(lengths.mean().item()),
        "multi_trace/max_trace_length": int(lengths.max().item()),
    }
