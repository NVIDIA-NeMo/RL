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
from typing import TypeVar

import torch
from pydantic import BaseModel, model_validator

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class RewardShapingConfig(BaseModel, extra="allow"):
    """Configuration for reward function processing.

    This configuration enables custom reward shaping, currently supporting DAPO-style
    penalties for responses that exceed the maximum response length threshold.
    """

    enabled: bool = False

    # The length of the buffer to penalize responses that exceed the maximum response length threshold.
    # Responses of length greater than overlong_buffer_length + max_response_length will
    # receive the maximum penalty.
    overlong_buffer_length: int | None = None

    # The penalty for responses that exceed the maximum response length threshold.
    overlong_buffer_penalty: float | None = None

    # The maximum response length threshold. Responses exceeding this length will be penalized.
    max_response_length: int | None = None

    # Stop properly penalty: scale factor for rewards of truncated responses (0-1).
    # When set to 0, truncated responses get zero reward.
    # When set to 1, no penalty is applied (default behavior).
    stop_properly_penalty_coef: float | None = None


class GroupLengthPenaltyConfig(BaseModel, extra="forbid"):
    """Group-relative length penalty on passing rollouts (MiMo-V2.6, arXiv:2610.11959, Eq. 4).

    For a prompt group with pass set P (reward >= success_threshold) and
    min_pass_rate < |P|/G <= max_pass_rate, the reference length l* is the
    reference_percentile-th percentile of the passing rollouts' generated-token counts, and
    each passing rollout i loses

        max_penalty * clip((l_i / l* - 1 - tolerance) / (saturation - tolerance), 0, 1) ** ramp_exponent

    Failed rollouts and groups outside the pass-rate window are unchanged. l_i counts every
    generated token of the rollout, summed over its turns.

    With std-normalized advantages, a group whose rollouts all pass has no reward spread except
    the penalty, so any penalty there becomes a full-size "be shorter" update whatever
    max_penalty is. max_pass_rate < 1 (e.g. 0.875 for groups of 8) leaves such groups untouched.
    """

    enabled: bool = False
    # X: largest reward deduction; keep it below (success reward - failure reward) so a
    # passing rollout always outscores a failing one.
    max_penalty: float = 0.3
    # delta: relative excess over l* that is never penalized.
    tolerance: float = 0.2
    # s: relative excess at which the full max_penalty applies (s > delta).
    saturation: float = 1.0
    # gamma: ramp exponent (>= 1).
    ramp_exponent: float = 2.0
    # A: the penalty applies only when the group's pass rate is strictly above this.
    min_pass_rate: float = 0.125
    # The penalty applies only when the group's pass rate is at most this.
    max_pass_rate: float = 1.0
    # B: percentile (0-100) of the passing rollouts' lengths used as l*.
    reference_percentile: float = 50.0
    # A rollout passes when its reward is >= this.
    success_threshold: float = 0.5

    @model_validator(mode="after")
    def _check_ranges(self) -> "GroupLengthPenaltyConfig":
        if self.max_penalty < 0:
            raise ValueError(f"max_penalty={self.max_penalty} must be >= 0")
        if not self.saturation > self.tolerance >= 0:
            raise ValueError(
                f"need saturation > tolerance >= 0, got saturation={self.saturation} "
                f"tolerance={self.tolerance}"
            )
        if self.ramp_exponent < 1:
            raise ValueError(f"ramp_exponent={self.ramp_exponent} must be >= 1")
        if not 0 <= self.min_pass_rate < self.max_pass_rate <= 1:
            raise ValueError(
                "need 0 <= min_pass_rate < max_pass_rate <= 1, got "
                f"min_pass_rate={self.min_pass_rate} max_pass_rate={self.max_pass_rate}"
            )
        if not 0 < self.reference_percentile < 100:
            raise ValueError(
                f"reference_percentile={self.reference_percentile} must be in (0, 100)"
            )
        return self


def apply_group_length_penalty(
    rewards: torch.Tensor,
    lengths: torch.Tensor,
    group_ids: torch.Tensor,
    cfg: GroupLengthPenaltyConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (penalized rewards, metrics) for one row per rollout; see GroupLengthPenaltyConfig.

    Args:
        rewards: [num_rollouts] rewards.
        lengths: [num_rollouts] generated-token counts.
        group_ids: [num_rollouts] prompt-group id of each rollout (rows of a group need not be
            contiguous).
        cfg: Penalty parameters.
    """
    if not rewards.shape == lengths.shape == group_ids.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)}, lengths {tuple(lengths.shape)} and group_ids "
            f"{tuple(group_ids.shape)} must have the same shape"
        )

    lengths = lengths.float()
    passed = rewards >= cfg.success_threshold
    penalties = torch.zeros_like(rewards, dtype=torch.float32)
    reference_lengths = []
    groups_above_max = 0
    for group_id in torch.unique(group_ids):
        group = group_ids == group_id
        group_passed = passed & group
        pass_rate = group_passed.sum().item() / group.sum().item()
        if pass_rate <= cfg.min_pass_rate:
            continue
        if pass_rate > cfg.max_pass_rate:
            groups_above_max += 1
            continue
        reference = torch.quantile(
            lengths[group_passed], cfg.reference_percentile / 100.0
        )
        reference_lengths.append(reference.item())
        excess = (lengths / reference - 1 - cfg.tolerance) / (
            cfg.saturation - cfg.tolerance
        )
        ramp = excess.clamp(0, 1) ** cfg.ramp_exponent
        penalties = torch.where(group_passed, cfg.max_penalty * ramp, penalties)

    penalized = rewards - penalties.to(rewards.dtype)
    num_passed = int(passed.sum().item())
    # Rollouts exactly at (1 + tolerance) * l* get float-rounding penalties of ~1e-15; don't
    # count them.
    hit = penalties > 1e-6
    metrics = {
        "length_penalty/groups_applied": float(len(reference_lengths)),
        "length_penalty/groups_skipped_above_max_pass_rate": float(groups_above_max),
        "length_penalty/mean_reference_length": (
            sum(reference_lengths) / len(reference_lengths)
            if reference_lengths
            else 0.0
        ),
        "length_penalty/frac_passing_penalized": (
            float(hit.sum().item()) / num_passed if num_passed else 0.0
        ),
        "length_penalty/mean_penalty_on_penalized": (
            penalties[hit].mean().item() if hit.any() else 0.0
        ),
        "length_penalty/mean_reward_unpenalized": (
            rewards.float().mean().item() if rewards.numel() else 0.0
        ),
    }
    return penalized, metrics


def apply_group_length_penalty_to_rows(
    rewards: torch.Tensor,
    token_mask: torch.Tensor,
    group_ids: torch.Tensor,
    participating: torch.Tensor,
    cfg: GroupLengthPenaltyConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the penalty to one row per rollout of a training batch.

    The trainers' entry point to ``apply_group_length_penalty``. A rollout's length is its
    generated tokens over all turns (the row's ``token_mask`` sum). Only ``participating``
    rows -- the rows that take part in the advantage statistics -- count: token-capture
    placeholders keep the environment reward but have no tokens, and counted they would be
    length-0 passing rollouts that pull their group's reference length down. Other rows keep
    their reward.

    Args:
        rewards: [B] rewards.
        token_mask: [B, T] generated-token mask.
        group_ids: [B] or [B, 1] integer prompt-group ids. A batch must hold whole groups.
        participating: [B] rows that take part in the advantage statistics (nonzero = yes).
        cfg: Penalty parameters.

    Returns:
        The rewards for the advantage estimator, and this batch's metric sums for
        ``group_length_penalty_metrics`` (sums, so callers that compute advantages in
        several calls per step can add them).
    """
    participating = participating.bool()
    adv_rewards = rewards.clone()
    metrics = {
        "length_penalty/groups_applied": 0.0,
        "length_penalty/groups_skipped_above_max_pass_rate": 0.0,
        "length_penalty/mean_reference_length": 0.0,
    }
    if participating.any():
        adv_rewards[participating], metrics = apply_group_length_penalty(
            rewards[participating],
            token_mask.sum(dim=-1)[participating],
            group_ids.reshape(-1)[participating],
            cfg,
        )
    penalties = (rewards - adv_rewards).float()
    hit = penalties > 1e-6
    groups_applied = metrics["length_penalty/groups_applied"]
    return adv_rewards, {
        "groups_applied": groups_applied,
        "groups_skipped_above_max_pass_rate": metrics[
            "length_penalty/groups_skipped_above_max_pass_rate"
        ],
        "reference_length_sum": metrics["length_penalty/mean_reference_length"]
        * groups_applied,
        "num_passed": float(
            (rewards[participating] >= cfg.success_threshold).sum().item()
        ),
        "num_penalized": float(hit.sum().item()),
        "penalty_sum": float(penalties[hit].sum().item()),
        "num_rows": float(participating.sum().item()),
        "reward_sum": float(rewards[participating].float().sum().item()),
        "reward_with_length_penalty_sum": float(
            adv_rewards[participating].float().sum().item()
        ),
    }


def group_length_penalty_metrics(partials: list[dict[str, float]]) -> dict[str, float]:
    """Turn ``apply_group_length_penalty_to_rows`` sums into step metrics.

    Rates and means are over the participating rows.
    """
    if not partials:
        return {}
    total = {key: sum(partial[key] for partial in partials) for key in partials[0]}
    groups = total["groups_applied"]
    rows = total["num_rows"]
    passed = total["num_passed"]
    penalized = total["num_penalized"]
    return {
        "length_penalty/groups_applied": groups,
        "length_penalty/groups_skipped_above_max_pass_rate": total[
            "groups_skipped_above_max_pass_rate"
        ],
        "length_penalty/mean_reference_length": (
            total["reference_length_sum"] / groups if groups else 0.0
        ),
        "length_penalty/frac_passing_penalized": penalized / passed if passed else 0.0,
        "length_penalty/mean_penalty_on_penalized": (
            total["penalty_sum"] / penalized if penalized else 0.0
        ),
        "length_penalty/mean_reward_unpenalized": (
            total["reward_sum"] / rows if rows else 0.0
        ),
        "reward_with_length_penalty": (
            total["reward_with_length_penalty_sum"] / rows if rows else 0.0
        ),
    }


def apply_reward_shaping(
    batch: BatchedDataDict, cfg: RewardShapingConfig
) -> BatchedDataDict:
    """Process rewards by applying penalties for responses exceeding max_response_length. Currently, this function only supports DAPO reward shaping as illustrated in the DAPO paper : https://arxiv.org/pdf/2503.14476.

    Nonetheless, it can be potentially extended to support any custom reward logic.
    """
    rewards = batch["total_reward"]
    if not cfg.enabled:
        return batch

    # Preserve the pre-shaping reward so downstream consumers (e.g. DAPO
    # dynamic sampling) can filter prompt groups on the raw task metric
    # rather than on length-dependent shaped rewards.
    batch["unshaped_total_reward"] = rewards.clone()

    # Apply stop properly penalty if configured
    if cfg.stop_properly_penalty_coef is not None:
        stop_properly_penalty_coef = cfg.stop_properly_penalty_coef
        assert 0 <= stop_properly_penalty_coef <= 1, (
            f"stop_properly_penalty_coef must be in [0, 1], got {stop_properly_penalty_coef}"
        )
        # Warn user that DAPO overlong parameters are ignored when stop_properly_penalty_coef is set
        ignored_params = []
        if cfg.overlong_buffer_length is not None:
            ignored_params.append("overlong_buffer_length")
        if cfg.overlong_buffer_penalty is not None:
            ignored_params.append("overlong_buffer_penalty")
        if cfg.max_response_length is not None:
            ignored_params.append("max_response_length")
        if ignored_params:
            print(
                f"[WARN] stop_properly_penalty_coef is set, so the following DAPO overlong "
                f"parameters are ignored: {', '.join(ignored_params)}. "
                f"Set stop_properly_penalty_coef=null to use DAPO overlong reward shaping instead.",
                flush=True,
            )
        truncated = batch.get("truncated")
        assert truncated is not None, "truncated field not found in batch"
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=rewards.device)
        else:
            truncated = truncated.to(device=rewards.device)

        num_truncated = truncated.sum().item()
        if num_truncated > 0:
            original_rewards = rewards.clone()
            # For truncated samples, scale the reward by stop_properly_penalty_coef
            rewards = torch.where(
                truncated, rewards * stop_properly_penalty_coef, rewards
            )
            batch["total_reward"] = rewards
            print(
                f"[INFO] stop properly penalty applied: {num_truncated}/{len(truncated)} samples truncated, "
                f"coef={stop_properly_penalty_coef}, "
                f"original_reward_mean={original_rewards[truncated].mean().item():.4f}, "
                f"shaped_reward_mean={rewards[truncated].mean().item():.4f}",
                flush=True,
            )
        else:
            print(
                "[INFO] stop properly penalty: no truncated samples (truncation_rate=0)",
                flush=True,
            )

        return batch

    # DAPO reward shaping requires overlong_buffer_length, overlong_buffer_penalty, and max_response_length to be set.
    overlong_buffer_length = cfg.overlong_buffer_length
    overlong_buffer_penalty = cfg.overlong_buffer_penalty
    max_response_length = cfg.max_response_length
    if (
        overlong_buffer_length is None
        or overlong_buffer_penalty is None
        or max_response_length is None
    ):
        raise ValueError(
            "Reward function is enabled but only DAPO reward shaping is currently supported. Please ensure overlong_buffer_length, overlong_buffer_penalty, and max_response_length are properly configured."
        )

    assert overlong_buffer_penalty >= 0, f"{overlong_buffer_penalty=} must be >=0"
    # Calculate the expected response length
    expected_response_length = max_response_length - overlong_buffer_length

    # Prefer slim per-sample tensor (data-plane path: message_log lives in
    # TQ, slice carries response_token_lengths). Fall back to scanning
    # message_log for the legacy non-data-plane caller.
    response_token_lengths = batch.get("response_token_lengths")
    if response_token_lengths is not None:
        if isinstance(response_token_lengths, torch.Tensor):
            response_lengths = response_token_lengths.tolist()
        else:
            response_lengths = list(response_token_lengths)
    else:
        # Count all generated assistant turns, excluding prompt history. Measure
        # before loss-mask preparation backfills missing generation_logprobs.
        response_lengths = []
        for message_log in batch["message_log"]:
            assistant_lengths = [
                message["token_ids"].shape[0]
                for message in message_log
                if message["role"] == "assistant" and "generation_logprobs" in message
            ]
            assert assistant_lengths, (
                "Assistant response not found during reward shaping"
            )
            response_lengths.append(sum(assistant_lengths))

    assert len(response_lengths) == len(rewards), (
        "The number of messages in the batch must match the number of rewards"
    )

    updated_rewards = torch.zeros_like(rewards)
    for i, message_response_length in enumerate(response_lengths):
        # Calculate the exceed length and the corresponding reward penalty
        exceed_length = message_response_length - expected_response_length
        overlong_reward = min(
            -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
        )
        updated_rewards[i] = rewards[i] + overlong_reward

    # Update the rewards in the batch
    batch["total_reward"] = updated_rewards

    return batch
