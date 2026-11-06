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
from typing import Any, Optional

from nemo_rl.data_plane.schema import MASK_SAMPLE
from nemo_rl.experience.interfaces import PromptGroupRecord


def mask_sample_flagged(extra: dict[str, Any] | None) -> bool:
    """Read both mask flags, since Gym may default the first-class flag to false."""
    extra = extra or {}
    return bool(
        extra.get(MASK_SAMPLE)
        or (extra.get("instance_config") or {}).get(MASK_SAMPLE, False)
    )


def rewards_carry_no_signal(rewards: Sequence[float]) -> bool:
    """Whether a group's participating rewards leave every row zero advantage.

    True for at most one participating reward, or when they are all exactly equal.
    Exact equality rather than a std threshold because float roundoff can give a
    uniform group a tiny nonzero std.

    ``rewards`` must already be restricted to the rows that vote in the group
    statistics. Passing every row instead keeps groups whose voters are uniform: at
    ``masked_reward_policy="exclude"`` with ``overlong_filtering``, ``[0 (truncated),
    1, 1, 1]`` has spread overall and none at all among the three rows the baseline
    actually sees.
    """
    return len(rewards) <= 1 or max(rewards) == min(rewards)


def participating_rewards(
    record: PromptGroupRecord,
    *,
    masked_reward_policy: Optional[str],
    overlong_filtering: bool,
) -> list[float]:
    """The rewards that vote in this group's baseline and std.

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
    """
    if record.loss_multiplier == 0:
        return []
    # None is PPO: no group-relative baseline exists to mask rows out of.
    if masked_reward_policy != "exclude":
        return [completion.reward for completion in record.completions]
    return [
        completion.reward
        for completion in record.completions
        if not mask_sample_flagged(completion.env_extras)
        and not (overlong_filtering and completion.truncated)
    ]


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
