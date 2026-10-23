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

"""Advantage Estimators for RL algorithms.

This module provides different advantage estimation strategies:
- GRPOAdvantageEstimator: Standard GRPO advantage with leave-one-out baseline
- GDPOAdvantageEstimator: Multi-reward GDPO (per-component baselines, optional per-reward weights, then normalize)
- ReinforcePlusPlusAdvantageEstimator: Reinforce++ with optional baseline subtraction (minus_baseline) and KL penalty in reward
- RawRewardAdvantageEstimator: Raw reward as advantage with optional batch normalization (no baseline, no value model)
- GeneralizedAdvantageEstimator: Generalized Advantage Estimation (GAE) with temporal bootstrapping
- OPDAdvantageEstimator: Multi-Teacher On-Policy Distillation (MOPD) token-level distillation advantages

Every group-relative estimator (GRPO, GDPO, Reinforce++) accepts ``valid_mask``
and must honor it: the SingleController always passes ``final_sample_mask`` as
``valid_mask`` so token-capture placeholder rows (and sequence-logprob-error
masked rows) do not vote in their siblings' baselines.

Reference papers:
- ProRLv2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
- Reinforce++: https://arxiv.org/abs/2501.03262
- GAE: https://arxiv.org/abs/1506.02438 (High-Dimensional Continuous Control Using Generalized Advantage Estimation)
- MOPD: https://arxiv.org/abs/2601.02780
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from nemo_rl.algorithms.legacy_ppo_diagnostics import raw_advantage_metrics
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    calculate_kl,
    get_gdpo_reward_component_keys,
    masked_mean,
    masked_var,
)


class PositiveLengthReweightConfig(BaseModel, extra="forbid"):
    """Length-aware reweighting of positive GRPO advantages within each prompt group.

    See ``reweight_positive_advantages_by_length`` for the algorithm. Unknown keys are rejected
    (``extra="forbid"``): a new block has no older configs to stay compatible with, and a typo
    such as ``rank_sprad`` would otherwise run silently at the default.

    Ported from upstream NeMo RL PR #4747. On the SingleController path with token-capture
    segment rows, a rollout (its canonical row plus its subagent / compaction segment rows) is
    measured and weighted as one response.
    """

    # Off by default; disabled, the GRPO advantages are bit-identical to not having this block.
    enabled: bool = False
    # "inverse_length": weight (L / L_min) ** -alpha, floored at 1 / max_ratio;
    # "rank": weight linear in the length rank among the group's positives. Rank mode is
    # order-only, so near-tied lengths still get the full 1 + s / 1 - s spread: two positives
    # of 500 and 501 tokens split their credit 75/25 at s = 0.5.
    mode: Literal["rank", "inverse_length"] = "inverse_length"
    # Rank mode: the shortest positive gets 1 + s and the longest 1 - s, a per-token ratio of
    # (1 + s) / (1 - s) (3x at 0.5) whatever the number of positives. 0 <= s < 1.
    rank_spread: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5
    # Inverse-length mode: exponent >= 0. 0 is a no-op; 1 gives every correct response of a
    # binary-reward group the same total credit while their lengths are within max_ratio.
    alpha: Annotated[float, Field(ge=0.0)] = 1.0
    # Inverse-length mode: the largest per-token weight ratio within a group (> 1), which also
    # bounds every final weight; None leaves the weights unbounded.
    max_ratio: Optional[Annotated[float, Field(gt=1.0)]] = 3.0
    # What is measured and reweighted:
    # "trajectory": the whole response length; every trained token of a positive rollout is
    #   scaled. With segment rows the length is the rollout's trained tokens over all its rows.
    # "thinking_per_turn": thinking tokens per assistant turn; only thinking tokens are scaled.
    # "prose_per_turn": free text (thinking + visible content) per assistant turn; only those
    #   tokens are scaled, so moving text between thinking and content changes nothing.
    # The per-turn scopes (reweight_positive_text_per_turn) leave tool calls and end-of-turn
    # tokens at the unweighted advantage, so the number of turns carries no length pressure
    # (unless turns_counted is "none", which measures the rollout's total).
    # They read the assistant_turn_id / thinking_mask / action_mask columns, which the
    # SingleController token-capture path builds.
    scope: Literal["trajectory", "thinking_per_turn", "prose_per_turn"] = "trajectory"
    # prose_per_turn only: the final assistant turn's content (the answer) is neither measured
    # nor reweighted. For tasks where the answer's length is part of the job (reports, code
    # delivered as text). With segment rows only the canonical (terminal) row holds the answer.
    exclude_final_answer: bool = False
    # Per-turn scopes: the turn count T in V = P / T. "with_spans" counts the turns that have at
    # least one measured token; "all" counts every trained assistant turn, so a turn holding
    # only a tool call lowers V; "none" does not divide (V = P, the rollout's total measured
    # tokens), so of two rollouts equally verbose per turn the one with more turns weighs less.
    turns_counted: Literal["with_spans", "all", "none"] = "with_spans"

    @model_validator(mode="after")
    def _check_scope_options(self) -> "PositiveLengthReweightConfig":
        if self.exclude_final_answer and self.scope != "prose_per_turn":
            raise ValueError(
                "positive_length_reweight.exclude_final_answer applies only to "
                f"scope='prose_per_turn', got scope={self.scope!r}"
            )
        return self

    @property
    def per_turn(self) -> bool:
        """Enabled with a per-turn scope, which runs after the GRPO estimator, not inside it."""
        return self.enabled and self.scope != "trajectory"


class AdvEstimatorConfig(BaseModel, extra="allow"):
    """Configuration for advantage estimator (GRPO, GDPO, OPD, or Reinforce++)."""

    name: Literal["grpo", "gdpo", "opd", "reinforce_plus_plus"] = "grpo"
    # GRPO specific
    normalize_rewards: bool = True
    use_leave_one_out_baseline: bool = True
    # GRPO specific: shift positive credit within a prompt group from long to short correct
    # responses, preserving the group's token-weighted positive mass.
    positive_length_reweight: PositiveLengthReweightConfig = Field(
        default_factory=PositiveLengthReweightConfig
    )
    # GDPO specific: optional per-component weights w_n for the aggregation.
    reward_weights: list[float] | None = None
    # Reinforce++ specific
    minus_baseline: bool = True
    # OPD specific: TROPD proximal-teacher weight alpha in (0, 1]. The advantage
    # targets the mixture log(alpha * p_teacher + (1 - alpha) * p_student)
    # instead of the teacher itself; 1.0 is plain MOPD.
    proximal_teacher_alpha: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    # OPD specific: subtract the mean advantage over every valid token in the
    # batch the estimator sees (a whole step; see OPDAdvantageEstimator).
    subtract_global_baseline: bool = False

    @model_validator(mode="after")
    def _check_positive_length_reweight_estimator(self) -> "AdvEstimatorConfig":
        if self.positive_length_reweight.enabled and self.name != "grpo":
            raise ValueError(
                "grpo.adv_estimator.positive_length_reweight.enabled requires "
                f"grpo.adv_estimator.name == 'grpo', got {self.name!r}"
            )
        return self


def per_turn_length_reweight_config(
    algo_cfg: object,
) -> Optional[PositiveLengthReweightConfig]:
    """The algorithm block's positive_length_reweight when it is enabled with a per-turn scope.

    None otherwise, including for algorithm blocks whose estimator has no such block (PPO).
    """
    reweight = getattr(
        getattr(algo_cfg, "adv_estimator", None), "positive_length_reweight", None
    )
    if isinstance(reweight, PositiveLengthReweightConfig) and reweight.per_turn:
        return reweight
    return None


# Group pass-rate buckets for the step-share metrics: <= 0.25, (0.25, 0.75), >= 0.75.
_PASS_RATE_BUCKET_NAMES = ("le_0.25", "0.25_0.75", "ge_0.75")


def _pass_rate_bucket(pass_rate: float) -> int:
    if pass_rate <= 0.25:
        return 0
    if pass_rate >= 0.75:
        return 2
    return 1


@dataclass
class PositiveLengthReweightStats:
    """Additive statistics of one or more positive-length-reweight calls.

    Kept as sums so the SingleController can merge every call of a step (streaming chunks)
    before ``to_metrics`` turns them into ratios.
    """

    # Groups with at least one valid row, and those with >= 2 positives that were reweighted.
    groups_with_valid_rows: int
    groups_reweighted: int
    # Final (renormalized) weights applied to positive rows.
    weight_sum: float
    weight_count: int
    weight_min: float
    weight_max: float
    # sum(|A_i| * L_i) over valid rows, per group pass-rate bucket, before and after reweighting.
    # Per-turn scopes: sum(|advantage|) over the valid rows' trained tokens.
    mass_before: list[float]
    mass_after: list[float]
    # Per-turn scopes only; all zero for scope="trajectory", whose metrics are then unchanged.
    # Valid rollouts with at least one trained assistant turn, split [reward > 0, reward <= 0].
    per_turn_rows: list[int] = field(default_factory=lambda: [0, 0])
    per_turn_turns: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # Correct (reward > 0) rollouts: trained thinking / content / action tokens.
    per_turn_thinking_tokens: float = 0.0
    per_turn_content_tokens: float = 0.0
    per_turn_action_tokens: float = 0.0
    # Correct rollouts with any prose: sum of thinking / (thinking + content).
    per_turn_thinking_share_sum: float = 0.0
    per_turn_thinking_share_rows: int = 0
    # net_turn_credit = sum(A_i * T_i) / sum(|A_i| * T_i) per pass-rate bucket (T_i = trained
    # turns, A_i = the rollout's advantage before reweighting).
    net_turn_credit_num: list[float] = field(
        default_factory=lambda: [0.0] * len(_PASS_RATE_BUCKET_NAMES)
    )
    net_turn_credit_den: list[float] = field(
        default_factory=lambda: [0.0] * len(_PASS_RATE_BUCKET_NAMES)
    )

    @classmethod
    def empty(cls) -> "PositiveLengthReweightStats":
        return cls(
            groups_with_valid_rows=0,
            groups_reweighted=0,
            weight_sum=0.0,
            weight_count=0,
            weight_min=math.inf,
            weight_max=-math.inf,
            mass_before=[0.0] * len(_PASS_RATE_BUCKET_NAMES),
            mass_after=[0.0] * len(_PASS_RATE_BUCKET_NAMES),
        )

    def merge(self, other: "PositiveLengthReweightStats") -> None:
        """Add ``other`` into this one in place."""
        self.groups_with_valid_rows += other.groups_with_valid_rows
        self.groups_reweighted += other.groups_reweighted
        self.weight_sum += other.weight_sum
        self.weight_count += other.weight_count
        self.weight_min = min(self.weight_min, other.weight_min)
        self.weight_max = max(self.weight_max, other.weight_max)
        self.mass_before = [a + b for a, b in zip(self.mass_before, other.mass_before)]
        self.mass_after = [a + b for a, b in zip(self.mass_after, other.mass_after)]
        self.per_turn_rows = [
            a + b for a, b in zip(self.per_turn_rows, other.per_turn_rows)
        ]
        self.per_turn_turns = [
            a + b for a, b in zip(self.per_turn_turns, other.per_turn_turns)
        ]
        self.per_turn_thinking_tokens += other.per_turn_thinking_tokens
        self.per_turn_content_tokens += other.per_turn_content_tokens
        self.per_turn_action_tokens += other.per_turn_action_tokens
        self.per_turn_thinking_share_sum += other.per_turn_thinking_share_sum
        self.per_turn_thinking_share_rows += other.per_turn_thinking_share_rows
        self.net_turn_credit_num = [
            a + b for a, b in zip(self.net_turn_credit_num, other.net_turn_credit_num)
        ]
        self.net_turn_credit_den = [
            a + b for a, b in zip(self.net_turn_credit_den, other.net_turn_credit_den)
        ]

    def to_metrics(self) -> dict[str, float]:
        """Logging metrics under ``adv_len_reweight/``.

        The bucket shares are taken after reweighting; by construction they equal the shares
        before, and ``share_max_abs_diff`` reports the largest difference as a sanity check.
        Weight statistics are omitted when no group was reweighted.
        """
        prefix = "adv_len_reweight/"
        metrics = {
            f"{prefix}frac_groups_reweighted": (
                self.groups_reweighted / self.groups_with_valid_rows
                if self.groups_with_valid_rows
                else 0.0
            )
        }
        if self.weight_count:
            metrics[f"{prefix}weight_min"] = self.weight_min
            metrics[f"{prefix}weight_mean"] = self.weight_sum / self.weight_count
            metrics[f"{prefix}weight_max"] = self.weight_max
        total_before, total_after = sum(self.mass_before), sum(self.mass_after)
        if total_before > 0 and total_after > 0:
            diffs = []
            for name, before, after in zip(
                _PASS_RATE_BUCKET_NAMES, self.mass_before, self.mass_after
            ):
                metrics[f"{prefix}share_bucket_{name}"] = after / total_after
                diffs.append(abs(after / total_after - before / total_before))
            metrics[f"{prefix}share_max_abs_diff"] = max(diffs)
        # Per-turn scopes (absent for scope="trajectory").
        correct_rows, incorrect_rows = self.per_turn_rows
        if correct_rows:
            turns = self.per_turn_turns[0]
            metrics[f"{prefix}turns_mean_correct"] = turns / correct_rows
            if turns > 0:
                metrics[f"{prefix}thinking_per_turn_mean_correct"] = (
                    self.per_turn_thinking_tokens / turns
                )
                metrics[f"{prefix}content_per_turn_mean_correct"] = (
                    self.per_turn_content_tokens / turns
                )
                metrics[f"{prefix}action_tokens_per_turn_mean_correct"] = (
                    self.per_turn_action_tokens / turns
                )
        if incorrect_rows:
            metrics[f"{prefix}turns_mean_incorrect"] = (
                self.per_turn_turns[1] / incorrect_rows
            )
        if self.per_turn_thinking_share_rows:
            metrics[f"{prefix}thinking_share_of_prose_mean_correct"] = (
                self.per_turn_thinking_share_sum / self.per_turn_thinking_share_rows
            )
        if sum(self.net_turn_credit_den) > 0:
            metrics[f"{prefix}net_turn_credit"] = sum(self.net_turn_credit_num) / sum(
                self.net_turn_credit_den
            )
            for name, num, den in zip(
                _PASS_RATE_BUCKET_NAMES,
                self.net_turn_credit_num,
                self.net_turn_credit_den,
            ):
                if den > 0:
                    metrics[f"{prefix}net_turn_credit_bucket_{name}"] = num / den
        return metrics


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """0-based ranks of ``values`` (0 = smallest); tied values share the mean of their ranks."""
    smaller = (values.unsqueeze(0) < values.unsqueeze(1)).sum(dim=1)
    equal = (values.unsqueeze(0) == values.unsqueeze(1)).sum(dim=1)
    return smaller.to(torch.float64) + (equal.to(torch.float64) - 1.0) / 2.0


def rank_length_weights(lengths: torch.Tensor, rank_spread: float) -> torch.Tensor:
    """Raw rank-mode weights ``1 + s * (1 - 2 r / (k - 1))`` for ``k >= 2`` positive lengths.

    ``r`` is the average rank of each length (0 = shortest). The weights are linear in the rank,
    go from ``1 + s`` (shortest) to ``1 - s`` (longest), and average to 1.
    """
    k = lengths.numel()
    if k < 2:
        raise ValueError(f"rank weights need at least 2 positives, got {k}")
    ranks = _average_ranks(lengths)
    return 1.0 + rank_spread * (1.0 - 2.0 * ranks / (k - 1))


def inverse_length_weights(
    lengths: torch.Tensor, alpha: float, max_ratio: Optional[float]
) -> torch.Tensor:
    """Raw inverse-length weights ``(L / L_min) ** -alpha``, floored at ``1 / max_ratio``.

    Dividing by the shortest length keeps the weights in ``(0, 1]``, so a large ``alpha`` cannot
    underflow the renormalization to 0 / 0; the common factor cancels in the renormalization, so
    the final weights equal those of ``L ** -alpha``. The floor caps the per-token ratio between
    any two positives at ``max_ratio`` (``None``: unbounded).
    """
    raw = (lengths / lengths.min()).pow(-alpha)
    if max_ratio is not None:
        raw = raw.clamp(min=1.0 / max_ratio)
    return raw


def reweight_positive_advantages_by_length(
    advantages: torch.Tensor,
    *,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    lengths: torch.Tensor,
    valid: torch.Tensor,
    config: PositiveLengthReweightConfig,
) -> tuple[torch.Tensor, PositiveLengthReweightStats]:
    """Rescale the positive advantages of each prompt group by response length.

    Per group (rows with equal ``prompt_ids``), with positives
    ``P = {i : A_i > 0, valid_i, L_i > 0}`` and only when ``|P| >= 2``:

    * raw weights: inverse-length mode ``w_i = max((L_i / min_P L) ** -alpha, 1 / max_ratio)``;
      rank mode ``w_i = 1 + s * (1 - 2 r_i / (|P| - 1))`` with ``r_i`` the average length rank
      within P (order-only: near-tied lengths get the full spread);
    * one common factor per group restores the token-weighted positive mass:
      ``w_i *= sum_P(A_j L_j) / sum_P(A_j L_j w_j)``;
    * ``A_i *= w_i`` for i in P. Every other row is returned unchanged.

    Under ``token_level_loss`` a group's share of the step is ``sum(|A_i| L_i)``, which the
    renormalization leaves unchanged: credit moves only from long to short correct responses of
    the same group, never between prompts, and no sign flips. Only within-group statistics are
    used, so the result does not depend on how groups are split across calls (shard invariance)
    or on the row order within a group. ``grpo.advantage_clip_low/high``, applied after the
    estimator, can cap boosted short-correct advantages.

    Args:
        advantages: ``[B, 1]`` group-relative advantages (after baseline and normalization).
        prompt_ids: ``[B]`` or ``[B, S]`` rows; equal rows form a group.
        rewards: ``[B]`` rewards; only the group pass-rate bucket of the metrics reads them.
        lengths: ``[B]`` trained response tokens per row (``mask[:, 1:].sum(-1)``); rows that
            do not train have length 0 and are never reweighted.
        valid: ``[B]`` bool, rows that may be reweighted and count in the metrics.
        config: the ``positive_length_reweight`` block.

    Returns:
        The reweighted ``[B, 1]`` advantages (same dtype and device) and this call's statistics.
    """
    device = advantages.device
    out = advantages.clone()
    adv = advantages[:, 0]
    lengths = lengths.to(device=device, dtype=torch.float64)
    valid = valid.to(device=device, dtype=torch.bool)
    rewards = rewards.to(device=device, dtype=torch.float64)
    stats = PositiveLengthReweightStats.empty()
    if adv.numel() == 0:
        return out, stats
    _, group_index = torch.unique(prompt_ids, dim=0, return_inverse=True)
    group_index = group_index.to(device)
    for group in range(int(group_index.max().item()) + 1):
        rows = torch.nonzero((group_index == group) & valid, as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        stats.groups_with_valid_rows += 1
        bucket = _pass_rate_bucket(float(rewards[rows].mean()))
        row_adv = adv[rows].to(torch.float64)
        row_len = lengths[rows]
        stats.mass_before[bucket] += float((row_adv.abs() * row_len).sum())
        positive = (row_adv > 0) & (row_len > 0)
        if int(positive.sum()) >= 2:
            pos_adv, pos_len = row_adv[positive], row_len[positive]
            if config.mode == "rank":
                raw = rank_length_weights(pos_len, config.rank_spread)
            else:
                raw = inverse_length_weights(pos_len, config.alpha, config.max_ratio)
            pos_mass = pos_adv * pos_len
            weights = raw * (pos_mass.sum() / (pos_mass * raw).sum())
            out[rows[positive], 0] = (pos_adv * weights).to(out.dtype)
            stats.groups_reweighted += 1
            stats.weight_sum += float(weights.sum())
            stats.weight_count += int(weights.numel())
            stats.weight_min = min(stats.weight_min, float(weights.min()))
            stats.weight_max = max(stats.weight_max, float(weights.max()))
        stats.mass_after[bucket] += float(
            (out[rows, 0].to(torch.float64).abs() * row_len).sum()
        )
    return out, stats


def _distinct_turns(turn_id: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
    """``[B]`` number of distinct non-negative ``turn_id`` values at the ``select`` positions."""
    num_turns = int(turn_id.max().item()) + 1 if turn_id.numel() else 0
    if num_turns <= 0:
        return torch.zeros(turn_id.shape[0], dtype=torch.float64, device=turn_id.device)
    hit = torch.zeros(
        (turn_id.shape[0], num_turns), dtype=torch.float64, device=turn_id.device
    )
    keep = select & (turn_id >= 0)
    hit.scatter_add_(1, turn_id.clamp(min=0), keep.to(torch.float64))
    return (hit > 0).sum(dim=1).to(torch.float64)


def reweight_positive_text_per_turn(
    advantages: torch.Tensor,
    *,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    assistant_turn_id: torch.Tensor,
    thinking_mask: torch.Tensor,
    action_mask: torch.Tensor,
    valid: torch.Tensor,
    config: PositiveLengthReweightConfig,
    row_to_rollout: Optional[torch.Tensor] = None,
    final_answer_rows: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, PositiveLengthReweightStats]:
    """Per-turn scopes of ``positive_length_reweight``: rescale the free text of positive rollouts.

    Every trained assistant token is one of THINKING (``thinking_mask``), ACTION
    (``action_mask``: tool-call spans and the end-of-turn token) or CONTENT (the rest of its
    turn, ``assistant_turn_id >= 0``). The measured set M is the thinking tokens
    (``thinking_per_turn``) or the thinking and content tokens (``prose_per_turn``; with
    ``exclude_final_answer`` minus the final answer: the last turn's content of the rows in
    ``final_answer_rows``), restricted to trained tokens.

    A rollout is the rows that ``row_to_rollout`` maps to one index (default: every row is its
    own rollout). On the SingleController path with token-capture segment rows, a rollout's
    canonical row and its subagent / compaction segment rows share one advantage, and turns in
    different rows are different model calls, so the rollout's measured tokens and turns are
    the sums over its rows. Per group (rollouts with equal ``prompt_ids``), counted on
    ``[:, 1:]`` like the loss:

    * ``P_r`` = measured tokens, ``T_r`` = distinct turns with a measured token (``with_spans``)
      or with any trained token (``all``), or 1 (``none``: V is the total), ``V_r = P_r / T_r``;
    * positives ``S = {r : A_r > 0, valid_r, P_r > 0, T_r > 0}``; groups with ``|S| < 2`` are
      left unchanged;
    * raw weights ``u_r`` from ``V`` with the same function as ``scope="trajectory"`` uses for
      the length (inverse-length or rank);
    * ``c = sum_S(A_j P_j) / sum_S(A_j P_j u_j)`` and every measured token of every row of
      rollout ``r`` in S is multiplied by ``c * u_r``.

    Every other token (actions, excluded final content, negative / zero-advantage / invalid
    rollouts) is returned bit-identical. The sum of the group's positive advantages over measured
    tokens is unchanged, so with actions untouched each group keeps its share of the step. A
    rollout of the same per-turn verbosity gets the same weight whatever its turn count, so turns
    carry no length pressure, except under ``turns_counted="none"``, where more turns of measured
    text mean a larger V. Only within-group statistics are used (shard and row-order invariant).

    Must run on the estimator's output (after the rollout broadcast), before message-level
    penalties and advantage clipping: ``A_r`` is read as the single value over its rows' trained
    tokens, and a rollout whose trained tokens disagree raises.

    Args:
        advantages: ``[B, T]`` token-level advantages, constant over each rollout's trained tokens.
        prompt_ids: ``[B]`` or ``[B, S]`` rows; rollouts with equal ids form a group.
        rewards: ``[B]`` rewards (pass-rate buckets and the correct / incorrect metrics); the rows
            of a rollout carry its reward.
        mask: ``[B, T]`` trained-token mask (token mask times sample mask).
        assistant_turn_id: ``[B, T]`` 0-based assistant turn of each token within its row, -1
            elsewhere.
        thinking_mask: ``[B, T]`` bool, THINKING tokens.
        action_mask: ``[B, T]`` bool, ACTION tokens.
        valid: ``[B]`` bool, rows that may be reweighted and count in the metrics; a rollout is
            valid when any of its rows is.
        config: the ``positive_length_reweight`` block (a per-turn scope).
        row_to_rollout: optional ``[B]`` long, the rollout (``0..R-1``) of each row.
        final_answer_rows: optional ``[B]`` bool, the rows whose last turn's content is the final
            answer for ``exclude_final_answer`` (default: every row).

    Returns:
        The reweighted ``[B, T]`` advantages (a new tensor, same dtype and device) and this
        call's statistics.
    """
    if config.scope == "trajectory":
        raise ValueError("reweight_positive_text_per_turn needs a per-turn scope")
    device = advantages.device
    out = advantages.clone()
    stats = PositiveLengthReweightStats.empty()
    n_rows = out.shape[0]
    if n_rows == 0:
        return out, stats
    shape = tuple(advantages.shape)
    for name, tensor in (
        ("mask", mask),
        ("assistant_turn_id", assistant_turn_id),
        ("thinking_mask", thinking_mask),
        ("action_mask", action_mask),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match advantages {shape}"
            )
    if row_to_rollout is None:
        rollout = torch.arange(n_rows, device=device)
    else:
        rollout = row_to_rollout.to(device=device, dtype=torch.long).reshape(-1)
        if rollout.numel() != n_rows or bool((rollout < 0).any()):
            raise ValueError(
                "row_to_rollout must map every one of the "
                f"{n_rows} rows to a rollout index >= 0"
            )
    n_rollouts = int(rollout.max().item()) + 1

    trained = mask.to(device=device).bool()
    turn = assistant_turn_id.to(device=device, dtype=torch.int64)
    in_turn = (turn >= 0) & trained
    thinking = thinking_mask.to(device=device).bool() & in_turn
    action = action_mask.to(device=device).bool() & in_turn
    if bool((thinking & action).any()):
        raise ValueError("thinking_mask and action_mask overlap")
    content = in_turn & ~thinking & ~action
    if config.scope == "thinking_per_turn":
        measured = thinking.clone()
    else:
        measured = thinking | content
        if config.exclude_final_answer:
            last_turn = turn.masked_fill(~in_turn, -1).max(dim=1, keepdim=True).values
            final_content = content & (turn == last_turn)
            if final_answer_rows is not None:
                final_content &= final_answer_rows.to(
                    device=device, dtype=torch.bool
                ).reshape(-1, 1)
            measured &= ~final_content
    valid = valid.to(device=device, dtype=torch.bool)
    rewards = rewards.to(device=device, dtype=torch.float64)

    # Loss-aligned per-row counts: token t is trained from position t - 1, so [:, 1:].
    loss_trained = trained[:, 1:]
    loss_measured = measured[:, 1:]
    loss_turn = turn[:, 1:]
    row_measured = loss_measured.sum(dim=1).to(torch.float64)
    row_all_turns = _distinct_turns(loss_turn, in_turn[:, 1:])
    if config.turns_counted == "with_spans":
        row_counted_turns = _distinct_turns(loss_turn, loss_measured)
    else:
        row_counted_turns = row_all_turns
    row_trained = loss_trained.sum(dim=1).to(torch.float64)
    row_thinking = thinking[:, 1:].sum(dim=1).to(torch.float64)
    row_content = content[:, 1:].sum(dim=1).to(torch.float64)
    row_action = action[:, 1:].sum(dim=1).to(torch.float64)

    # A_r: the rollout's single value over its rows' trained tokens.
    adv64 = advantages.to(torch.float64)
    big = torch.finfo(torch.float64).max
    row_hi = adv64.masked_fill(~trained, -big).max(dim=1).values
    row_lo = adv64.masked_fill(~trained, big).min(dim=1).values
    has_tokens = trained.any(dim=1)
    roll_hi = torch.full((n_rollouts,), -big, dtype=torch.float64, device=device)
    roll_lo = torch.full((n_rollouts,), big, dtype=torch.float64, device=device)
    roll_hi = roll_hi.scatter_reduce(
        0, rollout[has_tokens], row_hi[has_tokens], reduce="amax"
    )
    roll_lo = roll_lo.scatter_reduce(
        0, rollout[has_tokens], row_lo[has_tokens], reduce="amin"
    )
    roll_has_tokens = torch.zeros(n_rollouts, dtype=torch.bool, device=device)
    roll_has_tokens[rollout[has_tokens]] = True
    if bool((roll_has_tokens & (roll_hi != roll_lo)).any()):
        bad = torch.nonzero(roll_has_tokens & (roll_hi != roll_lo), as_tuple=True)[0]
        raise ValueError(
            "per-turn length reweighting expects one advantage per row and per rollout over "
            "its trained tokens (apply it before message penalties and clipping); rollouts "
            f"{bad[:8].tolist()} vary"
        )
    roll_adv = torch.where(roll_has_tokens, roll_hi, torch.zeros_like(roll_hi))

    def per_rollout(values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(n_rollouts, dtype=torch.float64, device=device).index_add_(
            0, rollout, values.to(torch.float64)
        )

    measured_len = per_rollout(row_measured)
    all_turns = per_rollout(row_all_turns)
    if config.turns_counted == "none":  # V is the rollout's total measured tokens
        counted_turns = torch.ones_like(measured_len)
    else:
        counted_turns = per_rollout(row_counted_turns)
    trained_len = per_rollout(row_trained)
    thinking_len = per_rollout(row_thinking)
    content_len = per_rollout(row_content)
    action_len = per_rollout(row_action)
    roll_valid = per_rollout(valid.to(torch.float64)) > 0
    # A rollout's rows carry its reward and group; read both at its first row.
    first_row = torch.full((n_rollouts,), n_rows, dtype=torch.long, device=device)
    first_row = first_row.scatter_reduce(
        0, rollout, torch.arange(n_rows, device=device), reduce="amin"
    )
    if bool((first_row >= n_rows).any()):
        raise ValueError("row_to_rollout must use every rollout index 0..R-1")
    roll_reward = rewards[first_row]
    roll_prompt = prompt_ids.to(device=device).index_select(0, first_row)

    weight = torch.ones(n_rollouts, dtype=torch.float64, device=device)
    bucket_of = torch.full((n_rollouts,), -1, dtype=torch.long, device=device)
    _, group_index = torch.unique(roll_prompt, dim=0, return_inverse=True)
    group_index = group_index.to(device)
    for group in range(int(group_index.max().item()) + 1):
        members = torch.nonzero((group_index == group) & roll_valid, as_tuple=True)[0]
        if members.numel() == 0:
            continue
        stats.groups_with_valid_rows += 1
        bucket = _pass_rate_bucket(float(roll_reward[members].mean()))
        bucket_of[members] = bucket
        a = roll_adv[members]
        stats.mass_before[bucket] += float((a.abs() * trained_len[members]).sum())
        for k, r in enumerate(members.tolist()):
            turns_r = float(all_turns[r])
            if turns_r <= 0:
                continue
            correct = float(roll_reward[r]) > 0
            stats.per_turn_rows[0 if correct else 1] += 1
            stats.per_turn_turns[0 if correct else 1] += turns_r
            stats.net_turn_credit_num[bucket] += float(a[k]) * turns_r
            stats.net_turn_credit_den[bucket] += abs(float(a[k])) * turns_r
            if correct:
                think_n, content_n = float(thinking_len[r]), float(content_len[r])
                stats.per_turn_thinking_tokens += think_n
                stats.per_turn_content_tokens += content_n
                stats.per_turn_action_tokens += float(action_len[r])
                if think_n + content_n > 0:
                    stats.per_turn_thinking_share_sum += think_n / (think_n + content_n)
                    stats.per_turn_thinking_share_rows += 1
        p, t = measured_len[members], counted_turns[members]
        positive = (a > 0) & (p > 0) & (t > 0)
        if int(positive.sum()) >= 2:
            pos_members = members[positive]
            pos_adv, pos_p = a[positive], p[positive]
            per_turn = pos_p / t[positive]
            if config.mode == "rank":
                raw = rank_length_weights(per_turn, config.rank_spread)
            else:
                raw = inverse_length_weights(per_turn, config.alpha, config.max_ratio)
            pos_mass = pos_adv * pos_p
            weights = raw * (pos_mass.sum() / (pos_mass * raw).sum())
            weight[pos_members] = weights
            stats.groups_reweighted += 1
            stats.weight_sum += float(weights.sum())
            stats.weight_count += int(weights.numel())
            stats.weight_min = min(stats.weight_min, float(weights.min()))
            stats.weight_max = max(stats.weight_max, float(weights.max()))

    row_weight = weight[rollout]
    scaled = measured & (row_weight != 1.0).unsqueeze(1)
    if bool(scaled.any()):
        out[scaled] = (adv64 * row_weight.unsqueeze(1))[scaled].to(out.dtype)
    row_mass_after = (
        out[:, 1:].to(torch.float64).abs() * loss_trained.to(torch.float64)
    ).sum(dim=1)
    roll_mass_after = per_rollout(row_mass_after)
    for bucket in range(len(_PASS_RATE_BUCKET_NAMES)):
        stats.mass_after[bucket] += float(roll_mass_after[bucket_of == bucket].sum())
    return out, stats


class GAEConfig(BaseModel, extra="allow"):
    """Configuration for the value-model advantage estimators (PPO)."""

    name: Literal["gae", "raw_reward"] = "gae"
    gae_lambda: float = 0.95
    gae_gamma: float = 1.0
    normalize_advantages: bool = True
    # VAPO decoupled GAE (None = standard GAE, no decoupling)
    gae_lambda_value: Optional[float] = None
    gae_lambda_policy: Optional[float] = None
    # Length-adaptive λ_policy = 1 - 1/(α·l). 0 = disabled.
    length_adaptive_alpha: float = 0.0
    # Which tokens GAE runs and whitens over on the SingleController path.
    # "all_rows" (legacy async PPO): every row's response tokens, including rows
    # later dropped from the loss by sample_mask (env mask_sample, overlong,
    # sequence-logprob error, multi-trace pad rows). "kept_rows": only the rows
    # that train. Rows excluded from the loss stay excluded either way.
    normalize_over: Literal["all_rows", "kept_rows"] = "all_rows"
    # Clamp the critic's values to [lo, hi] before GAE: the range a return can take. Default (0, 1): rewards in
    # [0, 1] with gamma 1 and no KL in the reward. Clamping onto that range never moves a value farther from the
    # realised return, so no token's advantage gets noisier. Policy advantages and bootstrapped returns use the
    # clamped values; the value loss still trains the critic's raw outputs. A terminal reward outside the range
    # warns once (and is counted in adv_raw/reward_outside_clamp_frac); null turns the clamp off.
    value_clamp: Optional[tuple[float, float]] = (0.0, 1.0)

    @field_validator("value_clamp")
    @classmethod
    def _ordered_clamp(cls, v: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        if v is not None and not v[0] < v[1]:
            raise ValueError(f"value_clamp must be (lo, hi) with lo < hi, got {v}")
        return v


class GRPOAdvantageEstimator:
    """GRPO-style advantage estimator with leave-one-out baseline.

    Note: GRPO computes advantages over all responses for each prompt.

    With ``positive_length_reweight.enabled`` the positive advantages of each prompt group are
    then rescaled by response length (``reweight_positive_advantages_by_length``).
    """

    def __init__(
        self, estimator_config: AdvEstimatorConfig, loss_config: ClippedPGLossConfig
    ):
        self.use_leave_one_out_baseline = estimator_config.use_leave_one_out_baseline
        self.normalize_rewards = estimator_config.normalize_rewards
        self.positive_length_reweight = estimator_config.positive_length_reweight
        # Set by every compute_advantage call: the positive length reweighting's statistics
        # (None while disabled) and their logging metrics ({} while disabled). The
        # SingleController merges the statistics across a step's calls; the other paths log
        # last_metrics.
        self.last_length_reweight_stats: Optional[PositiveLengthReweightStats] = None
        self.last_metrics: dict[str, float] = {}

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        valid_mask=None,
        response_lengths=None,
        **kwargs,
    ):
        """Compute GRPO advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
                  Used for expanding advantages to token-level shape and, with
                  positive_length_reweight, for the response lengths ``mask[:, 1:].sum(-1)``.
            valid_mask: Optional tensor of shape [batch_size], 1.0 for samples whose
                  reward should participate in the per-prompt baseline/std. Token-capture
                  placeholder rows carry 0.0 (their sample_mask already excludes them
                  from the loss; excluding them here keeps siblings' baselines unbiased).
                  None keeps the legacy all-valid behavior.
            response_lengths: Optional tensor of shape [batch_size], the trained tokens of each
                  sample for positive_length_reweight, overriding ``mask[:, 1:].sum(-1)``. The
                  SingleController passes a rollout's total over all its rows when it calls the
                  estimator on the canonical rows only (segment rows).
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        (
            baseline,
            std,
            is_trivial_distribution,
        ) = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards,
            torch.ones_like(rewards) if valid_mask is None else valid_mask.float(),
            leave_one_out_baseline=self.use_leave_one_out_baseline,
        )
        advantages = (rewards - baseline).unsqueeze(-1)

        if self.normalize_rewards:
            # don't sharpen the ones with no variation
            epsilon = 1e-6
            normalize_mask = (std > 0) & ~is_trivial_distribution
            advantages[normalize_mask] = advantages[normalize_mask] / (
                std.unsqueeze(-1)[normalize_mask] + epsilon
            )

        # The per-turn scopes need token-class columns the estimator does not see; the caller
        # applies reweight_positive_text_per_turn to this output instead.
        if (
            self.positive_length_reweight.enabled
            and self.positive_length_reweight.scope == "trajectory"
        ):
            advantages, stats = reweight_positive_advantages_by_length(
                advantages,
                prompt_ids=prompt_ids,
                rewards=rewards,
                # Shifted like the loss: token t is predicted from position t - 1, so
                # mask[:, 1:] is exactly what global_valid_toks counts.
                lengths=(
                    mask[:, 1:].sum(-1)
                    if response_lengths is None
                    else response_lengths
                ),
                valid=(
                    torch.ones_like(rewards, dtype=torch.bool)
                    if valid_mask is None
                    else valid_mask > 0
                ),
                config=self.positive_length_reweight,
            )
            self.last_length_reweight_stats = stats
            self.last_metrics = stats.to_metrics()
        else:
            self.last_length_reweight_stats = None
            self.last_metrics = {}

        return advantages.expand(mask.shape)


class GDPOAdvantageEstimator:
    """GDPO-style advantage estimator with leave-one-out baseline.

    Note: GDPO computes advantages for each reward separately over all responses for each prompt.
    """

    def __init__(
        self, estimator_config: AdvEstimatorConfig, loss_config: ClippedPGLossConfig
    ):
        self.use_leave_one_out_baseline = estimator_config.use_leave_one_out_baseline
        self.normalize_rewards = estimator_config.normalize_rewards
        # Optional per-reward weights w_n for the aggregation A = sum_n w_n * A_n
        # (paper: https://arxiv.org/abs/2601.05242). None => equal weights (all 1.0).
        self.reward_weights = estimator_config.reward_weights

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        repeated_batch,
        valid_mask=None,
        **kwargs,
    ):
        """Compute GDPO advantages.

        Args:
            prompt_ids: Tensor identifying which prompt each sample belongs to (for per-prompt baselines).
            rewards: Unused; for interface consistency.
            repeated_batch: Batch containing named reward component keys (e.g. reward/correctness, reward/format).
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
            valid_mask: Optional tensor of shape [batch_size], 1.0 for samples whose
                  reward components should participate in the per-prompt baseline/std.
                  Token-capture placeholder rows carry 0.0 (their sample_mask already
                  excludes them from the loss; excluding them here keeps siblings'
                  baselines unbiased). None keeps the legacy all-valid behavior.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        reward_component_keys = get_gdpo_reward_component_keys(repeated_batch)
        if len(reward_component_keys) < 2:
            raise ValueError(
                f"GDPO requires multiple reward components (reward/name1, reward/name2, ...). "
                f"This batch has {len(reward_component_keys)} component(s): {reward_component_keys}. "
                "Switch to GRPO by setting grpo.adv_estimator.name to 'grpo' in your config."
            )
        # Resolve per-reward weights (ordered to match the reward/<name> components,
        # sorted alphabetically by name — same order as reward_component_keys).
        weights = self.reward_weights
        if weights is None:
            weights = [1.0] * len(reward_component_keys)
        elif len(weights) != len(reward_component_keys):
            raise ValueError(
                f"reward_weights has {len(weights)} entries but this batch has "
                f"{len(reward_component_keys)} reward components ({reward_component_keys}). "
                "Provide exactly one weight per component, ordered alphabetically by "
                "component name (matching the sorted reward/<name> keys)."
            )
        reference = repeated_batch[reward_component_keys[0]]
        valid = torch.ones_like(reference) if valid_mask is None else valid_mask.float()
        leave_one_out = self.use_leave_one_out_baseline
        assert prompt_ids.shape[0] == valid.shape[0], (
            "prompt_ids must match reward batch size; "
            f"got {prompt_ids.shape[0]} vs {valid.shape[0]}"
        )
        advantage_parts = []
        for key in reward_component_keys:
            r = repeated_batch[key]
            (
                base,
                std_k,
                is_trivial_distribution,
            ) = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                r,
                valid,
                leave_one_out_baseline=leave_one_out,
            )
            adv_k = (r - base).unsqueeze(-1)
            if self.normalize_rewards:
                epsilon = 1e-6
                normalize_mask = (std_k > 0) & ~is_trivial_distribution
                adv_k[normalize_mask] = adv_k[normalize_mask] / (
                    std_k.unsqueeze(-1)[normalize_mask] + epsilon
                )

            advantage_parts.append(adv_k)

        advantages = sum(
            weight * adv_k for weight, adv_k in zip(weights, advantage_parts)
        )
        # Normalize combined advantage to zero mean and unit std
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - advantages.mean()) / adv_std
        else:
            advantages = advantages - advantages.mean()

        return advantages.expand(mask.shape)


class ReinforcePlusPlusAdvantageEstimator:
    """Reinforce++ advantage estimator with optional baseline subtraction and KL penalty in reward.

    Args:
        minus_baseline: If True, subtract per-prompt mean baseline from rewards.
        use_kl_in_reward: If True, add KL penalty to reward instead of loss.
    """

    def __init__(
        self, estimator_config: AdvEstimatorConfig, loss_config: ClippedPGLossConfig
    ):
        self.minus_baseline = estimator_config.minus_baseline
        self.use_kl_in_reward = loss_config.use_kl_in_reward
        self.kl_coef = loss_config.reference_policy_kl_penalty
        self.kl_type = loss_config.reference_policy_kl_type

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        *,
        logprobs_policy=None,
        logprobs_reference=None,
        valid_mask=None,
        **kwargs,
    ):
        """Compute Reinforce++ advantages with optional KL penalty.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
                  Used for: (1) expanding advantages to token-level shape, (2) global normalization
                  that only considers valid tokens.
            logprobs_policy: Policy log probabilities of shape [batch_size, seq_len], required if use_kl_in_reward.
            logprobs_reference: Reference policy log probabilities of shape [batch_size, seq_len], required if use_kl_in_reward.
            valid_mask: Optional tensor of shape [batch_size], 1.0 for samples whose
                  reward should participate in the per-prompt mean baseline. Token-capture
                  placeholder rows carry 0.0 (their sample_mask already excludes them from
                  the loss; excluding them here keeps siblings' baselines unbiased).
                  None keeps the legacy all-valid behavior.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len], globally normalized across valid tokens.
        """
        # minus baseline
        if self.minus_baseline:
            mean, _, _ = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                rewards,
                torch.ones_like(rewards) if valid_mask is None else valid_mask.float(),
                leave_one_out_baseline=False,
            )
            adv = rewards - mean
        else:
            adv = rewards

        adv = adv.unsqueeze(-1)
        adv = adv.expand(mask.shape)

        # add kl penalty to reward (token-level)
        if (
            self.use_kl_in_reward
            and logprobs_policy is not None
            and logprobs_reference is not None
        ):
            kl = calculate_kl(
                logprobs_policy,
                logprobs_reference,
                kl_type=self.kl_type,
            )
            adv = adv - self.kl_coef * kl

        # global normalization across the batch
        adv_mean = (adv * mask).sum() / mask.sum()
        adv_var = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum()
        adv_rstd = adv_var.clamp(min=1e-8).rsqrt()
        adv = (adv - adv_mean) * adv_rstd

        return adv


class RawRewardAdvantageEstimator:
    """Advantage estimator that uses the raw reward directly as the advantage.

    No value model, no baselines. Optionally normalizes across the batch.
    """

    def __init__(self, estimator_config: GAEConfig, loss_config: ClippedPGLossConfig):
        self.normalize_advantages = estimator_config.normalize_advantages

    def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
        """Compute advantages as raw rewards expanded to token-level shape.

        Args:
            prompt_ids: Tensor of shape [batch_size] (unused).
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len].
            **kwargs: Additional arguments (unused).

        Returns:
            Tuple of (advantages, returns) where returns is None.
        """
        adv = rewards.unsqueeze(-1).expand(mask.shape)

        if self.normalize_advantages:
            adv_mean = (adv * mask).sum() / mask.sum().clamp(min=1)
            adv_var = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum().clamp(min=1)
            adv_rstd = adv_var.clamp(min=1e-8).rsqrt()
            adv = (adv - adv_mean) * adv_rstd

        return adv, None


class GeneralizedAdvantageEstimator:
    """Generalized Advantage Estimation (GAE) with temporal bootstrapping.

    GAE computes advantages using temporal difference (TD) and exponentially-weighted averages:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

    This is computed recursively backwards:
        A_t = δ_t + γλ * (1 - done_t) * A_{t+1}

    KL penalty is applied to token-level rewards *externally* before calling the
    pure GAE computation, following veRL's separation-of-concerns approach.  This
    keeps the core GAE loop agnostic to reward construction and makes it easy to
    swap in different reward signals (process reward models, no KL, etc.) without
    touching the advantage estimator.

    The GAE loop uses carry-forward masking: at masked positions the running
    accumulators (next_values, last_gae_lam) are preserved from the last valid
    token rather than being zeroed.  This correctly skips over non-response tokens
    (padding, separators in multi-turn) without introducing phantom TD errors.

    Args:
        gae_lambda: GAE λ parameter (decay factor for advantage estimation, typically 0.95-0.98)
        gae_gamma: Discount factor γ (typically 0.99)
        normalize_advantages: If True, normalize advantages globally across batch
    """

    def __init__(self, estimator_config: GAEConfig, loss_config: ClippedPGLossConfig):
        self.normalize_over = estimator_config.normalize_over
        self.gae_lambda = estimator_config.gae_lambda
        self.gae_gamma = estimator_config.gae_gamma
        self.normalize_advantages = estimator_config.normalize_advantages

        # VAPO decoupled GAE: separate λ for value returns vs policy advantages.
        # None for both = standard GAE (use gae_lambda everywhere, no decoupling).
        self.gae_lambda_value = estimator_config.gae_lambda_value
        self.gae_lambda_policy = estimator_config.gae_lambda_policy
        # Length-adaptive λ_policy = 1 - 1/(α·l). 0 = disabled (use fixed λ).
        self.length_adaptive_alpha = estimator_config.length_adaptive_alpha
        self.value_clamp = estimator_config.value_clamp
        self._warned_reward_outside_clamp = False
        if self.value_clamp is not None and loss_config.use_kl_in_reward and (
            loss_config.reference_policy_kl_penalty > 0
        ):
            warnings.warn(
                f"ppo.adv_estimator.value_clamp={self.value_clamp} with KL in the reward: "
                "returns can leave the clamp range, so the clamp may bias the baseline."
            )

        self.use_kl_in_reward = loss_config.use_kl_in_reward
        self.kl_coef = loss_config.reference_policy_kl_penalty
        self.kl_type = loss_config.reference_policy_kl_type

        # adv_raw/* of the last compute_advantage call (legacy async PPO parity).
        self.last_metrics: dict[str, float] = {}

    def _reward_whiten(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        shift_mean: bool = True,
    ) -> torch.Tensor:
        mean = masked_mean(rewards, mask)
        var = masked_var(rewards, mask, mean)

        whitened_rewards = (rewards - mean) * torch.rsqrt(var + 1e-8)

        if not shift_mean:
            whitened_rewards = whitened_rewards + mean
        return whitened_rewards

    def _build_token_level_rewards(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        logprobs_policy: torch.Tensor | None = None,
        logprobs_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-token reward tensor with optional KL penalty.

        Constructs token_level_rewards = -kl_coef * KL  (at every response token)
                                        + terminal_reward  (at last valid token)

        Args:
            rewards: Scalar reward per sample, shape [batch_size].
            mask: Response token mask, shape [batch_size, seq_len].
            logprobs_policy: Current policy log probs, shape [batch_size, seq_len].
            logprobs_reference: Reference policy log probs, shape [batch_size, seq_len].

        Returns:
            token_level_rewards: shape [batch_size, seq_len].
        """
        seq_len = mask.shape[1]
        token_level_rewards = torch.zeros(
            rewards.shape[0], seq_len, device=rewards.device, dtype=rewards.dtype
        )

        # Apply KL penalty at every response token (gated by use_kl_in_reward).
        if (
            self.use_kl_in_reward
            and self.kl_coef > 0
            and logprobs_policy is not None
            and logprobs_reference is not None
        ):
            kl = calculate_kl(logprobs_policy, logprobs_reference, self.kl_type)
            token_level_rewards = token_level_rewards - self.kl_coef * kl

        # Place terminal reward at the last response token (last mask=1
        # position) for each sample. Using mask (not a separate `lengths`
        # tensor) ensures the reward lands on an assistant token even in
        # multi-turn scenarios where the sequence may end with a non-assistant
        # message.
        last_response_idx = mask.shape[1] - 1 - mask.fliplr().argmax(dim=1)
        has_response = mask.any(dim=1)
        token_level_rewards[has_response, last_response_idx[has_response]] += rewards[
            has_response
        ]

        # Zero out prompt/padding positions
        token_level_rewards = token_level_rewards * mask

        return token_level_rewards

    def _resolve_lambda_policy(self, mask: torch.Tensor) -> float | torch.Tensor:
        """Return the λ to use for policy advantages.

        Priority: length_adaptive_alpha > gae_lambda_policy > gae_lambda.
        """
        if self.length_adaptive_alpha > 0:
            resp_lens = mask.sum(dim=1).float()  # [batch_size]
            lam = 1.0 - 1.0 / (self.length_adaptive_alpha * resp_lens).clamp(min=1.0)
            return lam.clamp(min=0.0, max=1.0)
        if self.gae_lambda_policy is not None:
            return self.gae_lambda_policy
        return self.gae_lambda

    def _resolve_lambda_value(self) -> float:
        """Return the λ to use for value returns.

        Returns gae_lambda_value if set, else gae_lambda (standard GAE).
        """
        if self.gae_lambda_value is not None:
            return self.gae_lambda_value
        return self.gae_lambda

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        values,
        *,
        logprobs_policy=None,
        logprobs_reference=None,
        **kwargs,
    ):
        """Compute GAE advantages with temporal bootstrapping.

        Supports VAPO-style decoupled GAE when gae_lambda_value or
        gae_lambda_policy or length_adaptive_alpha are set:
        - Value returns use gae_lambda_value (default: gae_lambda)
        - Policy advantages use gae_lambda_policy or length-adaptive λ
          (default: gae_lambda)
        When none of these are set, this is standard GAE.

        Returns:
            Tuple of (advantages, returns), each of shape [batch_size, seq_len].
        """
        token_level_rewards = self._build_token_level_rewards(
            rewards,
            mask,
            logprobs_policy,
            logprobs_reference,
        )

        clamp_metrics: dict[str, float] = {}
        if self.value_clamp is not None:
            lo, hi = self.value_clamp
            valid = mask.bool()
            n_valid = valid.sum().clamp(min=1)
            clamp_metrics = {
                "adv_raw/value_clamped_high_frac": float(((values > hi) & valid).sum() / n_valid),
                "adv_raw/value_clamped_low_frac": float(((values < lo) & valid).sum() / n_valid),
            }
            # The clamp assumes every return lies in [lo, hi]; check the terminal rewards of the rows that train.
            rows = valid.any(dim=-1)
            terminal = rewards.reshape(-1).to(values.device, torch.float32)
            outside = rows & ((terminal < lo - 1e-6) | (terminal > hi + 1e-6))
            clamp_metrics["adv_raw/reward_outside_clamp_frac"] = float(
                outside.sum() / rows.sum().clamp(min=1)
            )
            if bool(outside.any()) and not self._warned_reward_outside_clamp:
                self._warned_reward_outside_clamp = True
                bad = terminal[outside]
                warnings.warn(
                    f"ppo.adv_estimator.value_clamp={self.value_clamp}: {int(outside.sum())} terminal "
                    f"reward(s) outside the clamp range (min {float(bad.min()):.4g}, max "
                    f"{float(bad.max()):.4g}); the clamp can bias the baseline for them. Warned once; "
                    "adv_raw/reward_outside_clamp_frac counts them every step."
                )
            values = values.clamp(min=lo, max=hi)

        lam_value = self._resolve_lambda_value()
        lam_policy = self._resolve_lambda_policy(mask)

        # A tensor policy lambda is length-adaptive and may differ per sample.
        # Scalar overrides only need separate passes when their resolved values differ.
        need_decouple = isinstance(lam_policy, torch.Tensor) or (
            lam_value != lam_policy
        )
        if need_decouple:
            _, returns = self._compute_gae(
                token_level_rewards,
                values,
                mask,
                gae_lambda=lam_value,
            )
            advantages, _ = self._compute_gae(
                token_level_rewards,
                values,
                mask,
                gae_lambda=lam_policy,
            )
        else:
            advantages, returns = self._compute_gae(
                token_level_rewards,
                values,
                mask,
                gae_lambda=lam_value,
            )

        # Pre-whitening advantage scale (critic-quality diagnostic), captured
        # before normalize_advantages pins the std to 1.0.
        self.last_metrics = raw_advantage_metrics(
            advantages, mask, self.normalize_advantages
        )
        self.last_metrics.update(clamp_metrics)

        # Whiten advantages (optional) and zero out masked positions (always)
        if self.normalize_advantages:
            advantages = self._reward_whiten(advantages, mask)
        advantages = torch.masked_fill(advantages, ~(mask.bool()), 0)
        return advantages, returns

    def _compute_gae(
        self,
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        gae_lambda: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure GAE computation with carry-forward masking.

        At masked positions the running accumulators (next_values, last_gae_lam)
        are preserved from the last valid token rather than being corrupted by
        zeroed-out values.  This correctly handles non-contiguous response masks
        (multi-turn conversations, tool-use delimiters, packed sequences).

        Args:
            token_level_rewards: Per-token rewards, shape [batch_size, response_length].
            values: Value predictions, shape [batch_size, response_length].
            mask: Response token mask, shape [batch_size, response_length].
            gae_lambda: Override for self.gae_lambda.  Can be a scalar or a
                per-sample tensor of shape [batch_size] (VAPO length-adaptive).

        Returns:
            advantages: shape [batch_size, response_length].
            returns: advantages + values, shape [batch_size, response_length].
        """
        lam = gae_lambda if gae_lambda is not None else self.gae_lambda

        gen_len = token_level_rewards.shape[-1]
        if self.gae_gamma == 1.0 and not isinstance(lam, torch.Tensor) and lam == 1.0:
            print(
                f"Fast GAE compute activated for lambda={lam}, gamma={self.gae_gamma}",
                flush=True,
            )

            # With zero terminal bootstrap, the TD value terms telescope:
            # A_t = sum_{k=t}^T r_k - V_t. Scan rewards instead of running
            # one Python/PyTorch iteration per token. Keep tensor-valued lambda
            # on the general path.
            masked_rewards = token_level_rewards * mask
            reward_to_go = (
                masked_rewards.to(
                    torch.promote_types(masked_rewards.dtype, values.dtype)
                )
                .flip(-1)
                .cumsum(-1)
                .flip(-1)
            )

            # At masked positions, the loop carries the next valid token's
            # advantage. Gather that token's value to preserve this behavior
            # for both advantages and returns, including fully masked rows.
            indices = torch.arange(gen_len, device=values.device)
            next_valid = torch.where(mask.bool(), indices, gen_len)
            next_valid = next_valid.flip(-1)
            next_valid = next_valid.cummin(-1).values
            next_valid = next_valid.flip(-1)
            padded_values = torch.nn.functional.pad(values, (0, 1))
            advantages = reward_to_go - padded_values.gather(-1, next_valid)
            return advantages, advantages + values

        next_values: torch.Tensor = torch.zeros(
            values.shape[0], device=values.device, dtype=values.dtype
        )
        last_gae_lam: torch.Tensor = torch.zeros_like(next_values)
        advantages_reversed = []

        for t in reversed(range(gen_len)):
            delta = (
                token_level_rewards[:, t] + self.gae_gamma * next_values - values[:, t]
            )
            new_gae_lam = delta + self.gae_gamma * lam * last_gae_lam

            # Carry-forward: at masked positions, preserve accumulators from
            # the last valid token instead of updating them.
            m = mask[:, t]
            next_values = values[:, t] * m + (1 - m) * next_values
            last_gae_lam = new_gae_lam * m + (1 - m) * last_gae_lam

            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages, returns


class OPDAdvantageEstimator:
    """Multi-Teacher On-Policy Distillation (MOPD) advantage estimator (arXiv:2601.02780).

    Computes token-level distillation advantages:
        Â_MOPD,t = sg[log π_teacher - log π_student]

    This is Equation 8 from the MOPD paper. The IS truncation (w_t, the
    hard gate on the training-to-inference ratio) is handled separately by
    ICE-POP mode in ClippedPGLoss — not here.

    With ``proximal_teacher_alpha < 1`` (TROPD) the student is pulled toward a
    proximal teacher that stays within a trust region of the student:
        log π_prox = log(α · π_teacher + (1 − α) · π_student)
        Â_TROPD,t = sg[log π_prox − log π_student]
    The advantage is bounded below by log(1 − α), so tokens the teacher
    strongly rejects cannot dominate the update. α = 1 recovers Â_MOPD exactly.

    ``subtract_global_baseline`` then centers the advantage on the mean over
    every valid token passed to one ``compute_advantage`` call. That call
    covers a whole training step on ``run_grpo.py``; the SingleController
    calls it once per streaming chunk and so requires one chunk per step.

    The loss function should be configured with:
        disable_ppo_ratio: true               (REINFORCE, no PPO ratio)
        use_importance_sampling_correction: true
        truncated_importance_sampling_type: icepop
        truncated_importance_sampling_ratio_min: <eps_low>
        truncated_importance_sampling_ratio: <eps_high>

    Required kwargs in compute_advantage:
        teacher_logprobs: [B, S] teacher model log probabilities
        prev_logprobs: [B, S] student training-engine log probabilities
    """

    def __init__(
        self, estimator_config: AdvEstimatorConfig, loss_config: ClippedPGLossConfig
    ):
        self.proximal_teacher_alpha = estimator_config.proximal_teacher_alpha
        self.subtract_global_baseline = estimator_config.subtract_global_baseline
        self.last_metrics: dict[str, float] = {}

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=None,
        prev_logprobs=None,
        **kwargs,
    ):
        """Compute OPD distillation advantages.

        Args:
            prompt_ids: [B] prompt IDs (unused, kept for interface compatibility)
            rewards: [B] rewards (unused for pure distillation)
            mask: [B, S] token mask
            teacher_logprobs: [B, S] teacher model logprobs (required)
            prev_logprobs: [B, S] student training-engine logprobs (required)

        Returns:
            [B, S] token-level distillation advantages (stop-gradient)
        """
        if teacher_logprobs is None:
            raise ValueError("OPD requires teacher_logprobs")
        if prev_logprobs is None:
            raise ValueError("OPD requires prev_logprobs")

        # Â_MOPD,t = sg[log π_teacher - log π_student]  (Equation 8). Metrics
        # report this raw gap even when TROPD changes the training signal.
        teacher_student_gap = (teacher_logprobs - prev_logprobs).detach()

        if self.proximal_teacher_alpha == 1.0:
            distill_advantages = teacher_student_gap
        else:
            # TROPD: log(α · π_teacher + (1 − α) · π_student), in log space.
            alpha = self.proximal_teacher_alpha
            proximal_teacher_logprobs = torch.logaddexp(
                teacher_logprobs + math.log(alpha),
                prev_logprobs + math.log1p(-alpha),
            )
            distill_advantages = (proximal_teacher_logprobs - prev_logprobs).detach()

        if self.subtract_global_baseline:
            valid_advantages = torch.masked_select(distill_advantages, mask.bool())
            if valid_advantages.numel() > 0:
                distill_advantages = distill_advantages - valid_advantages.mean()

        # Apply mask
        advantages = distill_advantages * mask

        # Metrics
        self._compute_metrics(teacher_student_gap, advantages, mask)

        return advantages

    def _compute_metrics(self, teacher_student_gap, advantages, mask):
        """Compute OPD logging metrics and store in self.last_metrics."""
        valid_bool = mask.bool()
        distill_valid = torch.masked_select(teacher_student_gap, valid_bool)
        adv_valid = torch.masked_select(advantages, valid_bool)

        distill_mean = distill_valid.mean().item() if distill_valid.numel() > 0 else 0.0
        adv_mean = adv_valid.mean().item() if adv_valid.numel() > 0 else 0.0
        adv_std = adv_valid.std().item() if adv_valid.numel() > 1 else 0.0

        self.last_metrics = {
            "on_policy_distillation/teacher_student_logprob_gap_mean": distill_mean,
            "on_policy_distillation/adv_mean": adv_mean,
            "on_policy_distillation/adv_std": adv_std,
        }
