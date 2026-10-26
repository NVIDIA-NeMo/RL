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
"""Pass-rate filter: train only on prompt groups whose pass rate lies in a band.

``<algo>.pass_rate_filter`` (``grpo`` or ``ppo``): after the advantage stage has its final advantages, every row of a
prompt group whose pass rate is outside ``[lower, upper]`` has its advantages multiplied by ``out_of_band_weight``
(0 drops the group from the policy gradient). The defaults 0.15 / 0.6 keep 3-9 correct out of 16 rollouts.

Pass rate = mean reward over the group's rollouts that vote in the baseline (the stage's baseline mask, so masked
rollouts whose reward stays in the baseline count too), one vote per rollout (its canonical row; a rollout's segment
rows follow it). With ``pass_threshold`` set, the share of those rollouts with reward >= the threshold instead.

Only advantages change: the sample / token masks are untouched, so the value model still trains on every rollout and
the rows' tokens stay in the policy loss's token normaliser (with Adam, a per-step constant scale of the gradient).

Why: docs/thinking_shrink_research_log.md section 11 (easy groups and, with them, search push the policy toward
less thinking per turn; band 3-9/16 with the search-long data leaves the push near zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel, model_validator


class PassRateFilterConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    # Groups with lower <= pass rate <= upper keep their advantages.
    lower: float = 0.15
    upper: float = 0.6
    # Multiplier on the advantages of out-of-band groups (0 = drop from the policy gradient).
    out_of_band_weight: float = 0.0
    # None: pass rate = mean reward. A number: share of rollouts with reward >= pass_threshold.
    pass_threshold: Optional[float] = None

    @model_validator(mode="after")
    def _check(self) -> "PassRateFilterConfig":
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError(f"pass_rate_filter needs 0 <= lower <= upper <= 1, got {self.lower}, {self.upper}")
        if self.out_of_band_weight < 0.0:
            raise ValueError(f"pass_rate_filter.out_of_band_weight must be >= 0, got {self.out_of_band_weight}")
        return self


def pass_rate_filter_config(algo_cfg: object) -> Optional[PassRateFilterConfig]:
    """The algorithm block's pass_rate_filter when enabled, else None (dicts from ``++`` overrides are parsed)."""
    raw = getattr(algo_cfg, "pass_rate_filter", None)
    if raw is None:
        return None
    cfg = raw if isinstance(raw, PassRateFilterConfig) else PassRateFilterConfig.model_validate(dict(raw))
    return cfg if cfg.enabled else None


@dataclass
class PassRateFilterStats:
    """Additive per-step sums; ``to_metrics`` turns them into shares."""

    groups: int = 0
    groups_below: int = 0
    groups_above: int = 0
    pass_rate_sum: float = 0.0
    trained_tokens: float = 0.0
    trained_tokens_in_band: float = 0.0

    def merge(self, other: "PassRateFilterStats") -> None:
        self.groups += other.groups
        self.groups_below += other.groups_below
        self.groups_above += other.groups_above
        self.pass_rate_sum += other.pass_rate_sum
        self.trained_tokens += other.trained_tokens
        self.trained_tokens_in_band += other.trained_tokens_in_band

    def to_metrics(self) -> dict[str, float]:
        if not self.groups:
            return {}
        in_band = self.groups - self.groups_below - self.groups_above
        return {
            "pass_rate_filter/groups": float(self.groups),
            "pass_rate_filter/group_share_in_band": in_band / self.groups,
            "pass_rate_filter/group_share_below": self.groups_below / self.groups,
            "pass_rate_filter/group_share_above": self.groups_above / self.groups,
            "pass_rate_filter/mean_group_pass_rate": self.pass_rate_sum / self.groups,
            "pass_rate_filter/trained_token_share_in_band": (
                self.trained_tokens_in_band / self.trained_tokens if self.trained_tokens else 0.0
            ),
        }


def pass_rate_filter_weights(
    *,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    vote_mask: torch.Tensor,
    canonical_of_row: torch.Tensor,
    train_mask: torch.Tensor,
    config: PassRateFilterConfig,
) -> tuple[torch.Tensor, PassRateFilterStats]:
    """Per-row advantage multipliers for the pass-rate filter.

    Args:
        prompt_ids: ``(B, ...)`` the rows' prompt ids (rows of one group are equal).
        rewards: ``(B,)`` each row's rollout reward.
        vote_mask: ``(B,)`` > 0 where the row's reward enters its group baseline.
        canonical_of_row: ``(B,)`` long, each row's canonical row (itself for a canonical row; -1 for a DP pad row).
        train_mask: ``(B, T)`` trained tokens (for the token-share metric).
        config: the filter config.

    Returns:
        ``(B,)`` float multipliers (1 in band, ``out_of_band_weight`` outside; pad rows 1) and the step statistics.
    """
    num_rows = rewards.shape[0]
    weights = torch.ones(num_rows, dtype=torch.float32)
    stats = PassRateFilterStats()
    canonical_of_row = canonical_of_row.to(torch.long).cpu()
    vote = vote_mask.detach().float().cpu() > 0
    rew = rewards.detach().float().cpu()
    canon_rows = [i for i in range(num_rows) if int(canonical_of_row[i]) == i]
    if not canon_rows:
        return weights, stats
    # A rollout votes if any of its rows does (a canonical row gated out can still train via a segment row).
    rollout_votes = {c: False for c in canon_rows}
    for i in range(num_rows):
        c = int(canonical_of_row[i])
        if c >= 0 and bool(vote[i]):
            rollout_votes[c] = True
    # Group key: the canonical row's prompt ids as bytes (exact equality; cheaper than torch.unique over long rows).
    flat = prompt_ids.detach().cpu().reshape(num_rows, -1)
    key_to_group: dict[bytes, int] = {}
    group_of_canon = []
    for c in canon_rows:
        group_of_canon.append(key_to_group.setdefault(flat[c].numpy().tobytes(), len(key_to_group)))
    group_rewards: dict[int, list[float]] = {}
    for k, c in enumerate(canon_rows):
        if rollout_votes[c]:
            group_rewards.setdefault(group_of_canon[k], []).append(float(rew[c]))
    group_weight: dict[int, float] = {}
    for g, rs in group_rewards.items():
        if config.pass_threshold is None:
            rate = sum(rs) / len(rs)
        else:
            rate = sum(1.0 for r in rs if r >= config.pass_threshold) / len(rs)
        stats.groups += 1
        stats.pass_rate_sum += rate
        if rate < config.lower:
            stats.groups_below += 1
            group_weight[g] = config.out_of_band_weight
        elif rate > config.upper:
            stats.groups_above += 1
            group_weight[g] = config.out_of_band_weight
        else:
            group_weight[g] = 1.0
    canon_weight = {c: group_weight.get(group_of_canon[k], 1.0) for k, c in enumerate(canon_rows)}
    for i in range(num_rows):
        c = int(canonical_of_row[i])
        if c >= 0:
            weights[i] = canon_weight.get(c, 1.0)
    tokens = train_mask.detach().float().cpu().reshape(num_rows, -1).sum(dim=-1)
    stats.trained_tokens = float(tokens.sum())
    stats.trained_tokens_in_band = float((tokens * (weights == 1.0).float()).sum())
    return weights, stats
