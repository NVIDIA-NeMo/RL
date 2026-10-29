"""Driver-side diagnostics for a scalar PPO value model."""

from collections.abc import Sequence
from typing import Hashable

import torch


def _rank_auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    """Tie-corrected pairwise AUC; undefined when only one class is present."""
    n_pos = int(positive.sum())
    n_neg = positive.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ordered_scores, order = torch.sort(scores.float())
    _, inverse, counts = torch.unique_consecutive(
        ordered_scores, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0).float()
    midranks = (ends - counts.float() + 1 + ends) / 2
    ranks = torch.empty_like(ordered_scores)
    ranks[order] = midranks[inverse]
    return float(
        ((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()
    )


def critic_position_metrics(
    values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Value accuracy/calibration over early, middle and late response tokens."""
    valid = mask.bool()
    if not valid.any():
        return {}
    positions = (valid.long().cumsum(1) - 1).float()
    lengths = valid.sum(1, keepdim=True).clamp_min(1)
    bins = (positions * 3 / lengths).long().clamp(max=2)
    metrics: dict[str, float] = {}
    for i, name in enumerate(("early", "mid", "late")):
        selected = valid & (bins == i)
        n = int(selected.sum())
        if n == 0:
            continue
        v = values[selected].float()
        r = returns[selected].float()
        err = v - r
        metrics[f"critic/n_tokens_{name}"] = float(n)
        metrics[f"critic/mean_v_{name}"] = v.mean().item()
        metrics[f"critic/bias_{name}"] = err.mean().item()
        metrics[f"critic/abs_err_{name}"] = err.abs().mean().item()
        target_var = r.var(unbiased=False)
        metrics[f"critic/ev_{name}"] = (
            (1 - err.var(unbiased=False) / target_var).item()
            if target_var > 1e-8
            else 0.0
        )
        # Match Jiaqi's 10 confidence bins and clamp only for bin assignment.
        confidence_bins = (v.clamp(0, 1) * 10).long().clamp(max=9)
        ece = 0.0
        for bucket in range(10):
            in_bucket = confidence_bins == bucket
            if in_bucket.any():
                ece += (
                    in_bucket.float().mean()
                    * (v[in_bucket].mean() - r[in_bucket].mean()).abs()
                ).item()
        metrics[f"critic/ece_{name}"] = ece
    return metrics


def critic_reward_auc_metrics(
    values: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    group_ids: Sequence[Hashable] | None = None,
    positive_threshold: float = 0.5,
) -> dict[str, float]:
    """Terminal and within-task AUC of critic values against rollout success."""
    valid = mask.bool()
    has_response = valid.any(1)
    labels = rewards.detach().float() >= positive_threshold
    last = valid.shape[1] - 1 - valid.flip(1).float().argmax(1)
    rows = torch.arange(valid.shape[0], device=valid.device)
    metrics = {
        "critic/auc_positive_samples": float(labels[has_response].sum()),
        "critic/auc_negative_samples": float((~labels[has_response]).sum()),
        "critic/terminal_auc": _rank_auc(
            values[rows[has_response], last[has_response]].float(),
            labels[has_response],
        ),
    }
    if group_ids is not None and len(group_ids) != values.shape[0]:
        raise ValueError("group_ids must have one entry per trajectory")
    groups: dict[Hashable, list[int]] = {}
    for row, group_id in enumerate(group_ids if group_ids is not None else []):
        if bool(has_response[row]):
            groups.setdefault(group_id, []).append(row)
    mixed_groups = [
        rows_in_group
        for rows_in_group in groups.values()
        if len(rows_in_group) >= 2
        and bool(labels[rows_in_group].any())
        and bool((~labels[rows_in_group]).any())
    ]
    if group_ids is not None:
        metrics["critic/n_mixed_groups"] = float(len(mixed_groups))
    positions = (valid.long().cumsum(1) - 1).float()
    lengths = valid.sum(1, keepdim=True).clamp_min(1)
    bins = (positions * 4 / lengths).long().clamp(max=3)
    for i in range(4):
        selected = valid & (bins == i)
        counts = selected.sum(1)
        scores = (values.float() * selected).sum(1) / counts.clamp_min(1)
        eligible = counts > 0
        metrics[f"critic/global_auc_q{i + 1}"] = _rank_auc(
            scores[eligible], labels[eligible]
        )
        metrics[f"critic/auc_samples_q{i + 1}"] = float(eligible.sum())
        if group_ids is None:
            continue
        aucs = []
        for group in mixed_groups:
            members = [row for row in group if bool(counts[row])]
            if len(members) >= 2:
                auc = _rank_auc(scores[members], labels[members])
                if auc == auc:
                    aucs.append(auc)
        metrics[f"critic/within_group_auc_q{i + 1}"] = (
            sum(aucs) / len(aucs) if aucs else float("nan")
        )
    return metrics
