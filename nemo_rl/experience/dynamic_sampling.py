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

"""Shared dynamic-sampling reward participation, filtering and step metrics."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from nemo_rl.data_plane.schema import MASK_SAMPLE
from nemo_rl.experience.interfaces import PromptGroupRecord

if TYPE_CHECKING:
    from nemo_rl.experience.rollout_reassembler import FinalizedGroup


def mask_sample_flagged(extra: dict[str, Any] | None) -> bool:
    """Read both mask flags, since Gym may default the first-class flag to false."""
    extra = extra or {}
    return bool(
        extra.get(MASK_SAMPLE)
        or (extra.get("instance_config") or {}).get(MASK_SAMPLE, False)
    )


def rewards_carry_no_signal(rewards: Sequence[float]) -> bool:
    """Whether a group's voting rewards leave every row zero advantage.

    True for at most one voting reward, or when they are all exactly equal.
    Exact equality rather than a std threshold because float roundoff can give a
    uniform group a tiny nonzero std.

    ``rewards`` must already be restricted to the rows that vote in the group
    statistics. Passing every row instead keeps groups whose voters are uniform: at
    ``masked_reward_policy="exclude"`` with ``overlong_filtering``, ``[0 (truncated),
    1, 1, 1]`` has spread overall and none at all among the three rows the baseline
    actually sees.
    """
    return len(rewards) <= 1 or max(rewards) == min(rewards)


def voting_rewards(
    rewards: Sequence[float],
    *,
    mask_sample: Sequence[bool],
    truncated: Sequence[bool],
    loss_multiplier: float,
    masked_reward_policy: Optional[str],
    overlong_filtering: bool,
) -> list[float]:
    """The rewards that vote in a group's baseline and std.

    Mirrors v1's ``_dynamic_sampling_valid_mask``, which is the mask the group
    statistics are actually computed under. Filtering on the full reward list instead
    would keep groups whose voters are uniform, and drop groups whose only spread
    comes from rows that never vote.

    A zero dataset weight is excluded under either policy: ``processors.py`` cuts a
    prompt that did not fit the sequence limit down to a few tokens and gives it
    ``loss_multiplier == 0`` before any rollout, so its rewards are placeholders
    rather than measurements. That is group-level here, so such a group has no voters
    at all and is trivial by construction.

    Under ``exclude`` the environment's ``mask_sample`` rows drop out, and with
    ``overlong_filtering`` so do truncated ones -- the rows the loss drops later in
    the step. Under ``include`` both still vote: their rewards are measurements of
    the policy even though they carry no loss.

    Args:
        rewards: One Gym reward per row, before effort shaping and reward penalties.
        mask_sample: Per-row environment ``mask_sample`` flag, parallel to ``rewards``.
        truncated: Per-row truncation flag, parallel to ``rewards``.
        loss_multiplier: The group's dataset loss weight.
        masked_reward_policy: ``grpo.masked_reward_policy``; ``None`` is PPO, which
            has no group-relative baseline to mask rows out of.
        overlong_filtering: ``grpo.overlong_filtering``.
    """
    if loss_multiplier == 0:
        return []
    if masked_reward_policy != "exclude":
        return list(rewards)
    return [
        reward
        for reward, masked, is_truncated in zip(
            rewards, mask_sample, truncated, strict=True
        )
        if not masked and not (overlong_filtering and is_truncated)
    ]


def voting_rewards_from_record(
    record: PromptGroupRecord,
    *,
    masked_reward_policy: Optional[str],
    overlong_filtering: bool,
) -> list[float]:
    """The rewards that vote in a group's baseline, read from its ``Completion`` rows."""
    return voting_rewards(
        [
            completion.reward
            if completion.reward_original is None
            else completion.reward_original
            for completion in record.completions
        ],
        mask_sample=[
            mask_sample_flagged(completion.env_extras)
            for completion in record.completions
        ],
        truncated=[completion.truncated for completion in record.completions],
        loss_multiplier=record.loss_multiplier,
        masked_reward_policy=masked_reward_policy,
        overlong_filtering=overlong_filtering,
    )


def voting_rewards_from_finalized(
    finalized: "FinalizedGroup",
    *,
    masked_reward_policy: Optional[str],
    overlong_filtering: bool,
) -> list[float]:
    """The rewards that vote in a group's baseline, read from a finalized capture group."""
    return voting_rewards(
        finalized.reward_original,
        mask_sample=finalized.reward_mask_sample,
        truncated=finalized.reward_truncated,
        loss_multiplier=finalized.loss_multiplier,
        masked_reward_policy=masked_reward_policy,
        overlong_filtering=overlong_filtering,
    )


@dataclass
class StepRewardStats:
    """Reward totals for the groups generated for one training step.

    Counts every group that finished generating, whether dynamic sampling kept or
    dropped it. The kept groups all have reward spread, so their mean alone says
    little about how well the policy is doing.
    """

    groups: int = 0
    zero_variance_groups: int = 0
    reward_sum: float = 0.0
    samples: int = 0
    # First attempts only: the prompts originally scheduled, not their replacements.
    first_attempt_reward_sum: float = 0.0
    first_attempt_samples: int = 0


def record_group_rewards(
    step_rewards: dict[int, StepRewardStats],
    target_step: Optional[int],
    rewards: Sequence[float],
    *,
    replacement_index: int,
    zero_variance: bool,
) -> None:
    """Add one finished group's rewards to the tally of the step it was made for.

    Args:
        target_step: Step the group was stamped for. ``None`` (sampler does not
            stamp steps) means there is nothing to report against, so skip.
        rewards: Every row's reward, voting or not. ``unfiltered_reward`` is the
            pre-filter progress signal, so it averages the whole group.
        replacement_index: 0 for the originally scheduled prompt, N for its N-th
            replacement.
        zero_variance: Whether the group's *voting* rewards carry no signal.
            Passed in rather than recomputed here, so the count cannot disagree
            with the decision the filter actually made on the same group.
    """
    if target_step is None or not rewards:
        return
    tally = step_rewards.setdefault(target_step, StepRewardStats())
    total = float(sum(rewards))
    tally.groups += 1
    tally.zero_variance_groups += int(zero_variance)
    tally.reward_sum += total
    tally.samples += len(rewards)
    if replacement_index == 0:
        tally.first_attempt_reward_sum += total
        tally.first_attempt_samples += len(rewards)


def drain_step_reward_metrics(
    step_rewards: dict[int, StepRewardStats], step: int
) -> dict[str, float]:
    """Return the reward metrics for ``step`` and forget that step and older ones.

    Groups count toward the step they were generated for, even if a later step
    borrowed them (``promote_ready_group``). With lookahead above zero that can
    differ from the step that trained on them.

    Returns:
        ``unfiltered_reward`` (mean over every generated group, kept or dropped),
        ``first_attempt_reward`` (same, but first attempts only),
        ``generated_prompt_groups``, ``zero_variance_prompt_groups`` and
        ``zero_variance_fraction``. Empty if nothing was generated for the step.
    """
    tally = step_rewards.get(step)
    for old_step in list(step_rewards):
        if old_step <= step:
            del step_rewards[old_step]
    if tally is None or not tally.groups:
        return {}
    metrics = {
        "unfiltered_reward": tally.reward_sum / tally.samples,
        "generated_prompt_groups": float(tally.groups),
        "zero_variance_prompt_groups": float(tally.zero_variance_groups),
        "zero_variance_fraction": tally.zero_variance_groups / tally.groups,
    }
    if tally.first_attempt_samples:
        metrics["first_attempt_reward"] = (
            tally.first_attempt_reward_sum / tally.first_attempt_samples
        )
    return metrics
