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

"""Per-step reward tallies that make dynamic sampling visible on the SC path."""

from types import SimpleNamespace

import pytest

from nemo_rl.data_plane.schema import MASK_SAMPLE
from nemo_rl.experience.rollout_manager import RolloutManager, rewards_carry_no_signal


def _manager(
    masked_reward_policy: str | None = None, overlong_filtering: bool = False
) -> RolloutManager:
    # The tally methods touch one dict and nothing else, so a full manager (tokenizer,
    # policy, Ray) is not needed to exercise them.
    manager = RolloutManager.__new__(RolloutManager)
    manager._step_rewards = {}
    manager._masked_reward_policy = masked_reward_policy
    manager._overlong_filtering = overlong_filtering
    return manager


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([1.0], True),
        ([0.0, 0.0, 0.0], True),
        ([1.0, 1.0], True),
        ([0.0, 1.0], False),
        ([0.5, 0.5, 0.25], False),
    ],
)
def test_rewards_carry_no_signal(rewards, expected):
    assert rewards_carry_no_signal(rewards) is expected


def _completion(reward, *, truncated=False, mask_sample=False):
    return SimpleNamespace(
        reward=reward,
        truncated=truncated,
        env_extras={MASK_SAMPLE: True} if mask_sample else None,
    )


def _record(completions, *, loss_multiplier=1.0):
    return SimpleNamespace(completions=completions, loss_multiplier=loss_multiplier)


def test_zero_variance_group_predicate_reads_completion_rewards():
    record = _record([_completion(1.0), _completion(1.0)])
    assert _manager()._is_zero_variance_group(record) is True
    record.completions.append(_completion(0.0))
    assert _manager()._is_zero_variance_group(record) is False


def test_only_the_voting_rows_decide_whether_a_group_is_trivial():
    """The bug this guards: spread that comes only from non-voting rows.

    Under exclude with overlong_filtering the truncated 0.0 does not reach the
    baseline, so the three rows that do are uniform and the group has no signal --
    even though the full list has spread.
    """
    record = _record(
        [_completion(0.0, truncated=True)] + [_completion(1.0) for _ in range(3)]
    )

    excluding = _manager("exclude", overlong_filtering=True)
    including = _manager("include", overlong_filtering=True)

    assert excluding._is_zero_variance_group(record) is True
    assert including._is_zero_variance_group(record) is False


def test_a_flagged_row_only_votes_under_include():
    record = _record([_completion(0.0, mask_sample=True), _completion(1.0)])

    assert _manager("exclude")._is_zero_variance_group(record) is True
    assert _manager("include")._is_zero_variance_group(record) is False


def test_a_zero_weight_group_is_trivial_under_either_policy():
    """processors.py gives an unusable prompt loss_multiplier=0 before any rollout,
    so its rewards are placeholders and no row votes."""
    record = _record([_completion(0.0), _completion(1.0)], loss_multiplier=0.0)

    for policy in ("include", "exclude", None):
        assert _manager(policy)._is_zero_variance_group(record) is True


def test_unfiltered_reward_counts_kept_and_dropped_groups():
    manager = _manager()
    # Two original groups: one mixed (kept), one all-zero (dropped by dynamic sampling).
    manager.record_group_rewards(
        0,
        [1.0, 0.0, 1.0, 0.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([1.0, 0.0, 1.0, 0.0]),
    )
    manager.record_group_rewards(
        0,
        [0.0, 0.0, 0.0, 0.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([0.0, 0.0, 0.0, 0.0]),
    )
    # Its replacement: mixed, and the second slot's replacement: all-one (dropped).
    manager.record_group_rewards(
        0,
        [1.0, 1.0, 1.0, 0.0],
        replacement_index=1,
        zero_variance=rewards_carry_no_signal([1.0, 1.0, 1.0, 0.0]),
    )
    manager.record_group_rewards(
        0,
        [1.0, 1.0, 1.0, 1.0],
        replacement_index=1,
        zero_variance=rewards_carry_no_signal([1.0, 1.0, 1.0, 1.0]),
    )

    metrics = manager.drain_step_reward_metrics(0)

    assert metrics["generated_prompt_groups"] == 4.0
    assert metrics["zero_variance_prompt_groups"] == 2.0
    assert metrics["zero_variance_fraction"] == pytest.approx(0.5)
    # (2 + 0 + 3 + 4) / 16 over every generated group, not just the two kept ones.
    assert metrics["unfiltered_reward"] == pytest.approx(9 / 16)
    # Only the two original groups: (2 + 0) / 8.
    assert metrics["first_attempt_reward"] == pytest.approx(2 / 8)


def test_kept_groups_alone_would_hide_the_signal():
    manager = _manager()
    manager.record_group_rewards(
        0,
        [1.0, 0.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([1.0, 0.0]),
    )
    manager.record_group_rewards(
        0,
        [1.0, 1.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([1.0, 1.0]),
    )
    manager.record_group_rewards(
        0,
        [1.0, 1.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([1.0, 1.0]),
    )
    manager.record_group_rewards(
        0,
        [0.0, 0.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([0.0, 0.0]),
    )

    metrics = manager.drain_step_reward_metrics(0)

    # The one mixed group alone averages 0.5; the whole batch is 5/8.
    assert metrics["unfiltered_reward"] == pytest.approx(5 / 8)
    assert metrics["zero_variance_fraction"] == pytest.approx(0.75)


def test_unstamped_groups_are_not_recorded():
    manager = _manager()
    manager.record_group_rewards(
        None,
        [1.0, 0.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([1.0, 0.0]),
    )
    assert manager._step_rewards == {}


def test_empty_reward_list_is_not_recorded():
    manager = _manager()
    manager.record_group_rewards(
        0,
        [],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([]),
    )
    assert manager.drain_step_reward_metrics(0) == {}


def test_replacement_only_step_has_no_first_attempt_reward():
    manager = _manager()
    manager.record_group_rewards(
        0,
        [1.0, 0.0],
        replacement_index=2,
        zero_variance=rewards_carry_no_signal([1.0, 0.0]),
    )

    metrics = manager.drain_step_reward_metrics(0)

    assert "first_attempt_reward" not in metrics
    assert metrics["unfiltered_reward"] == pytest.approx(0.5)


def test_pop_forgets_the_closed_step_and_older_ones_only():
    manager = _manager()
    manager.record_group_rewards(
        0,
        [1.0, 0.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([1.0, 0.0]),
    )
    manager.record_group_rewards(
        1,
        [0.0, 0.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([0.0, 0.0]),
    )
    manager.record_group_rewards(
        2,
        [1.0, 1.0],
        replacement_index=0,
        zero_variance=rewards_carry_no_signal([1.0, 1.0]),
    )

    assert manager.drain_step_reward_metrics(1)["unfiltered_reward"] == 0.0
    # Step 0 was older than the closed step, so it is gone without having been read.
    assert manager.drain_step_reward_metrics(0) == {}
    # Step 2 was generated ahead (lookahead) and survives until its own step closes.
    assert manager.drain_step_reward_metrics(2)["unfiltered_reward"] == 1.0
    assert manager.drain_step_reward_metrics(2) == {}


def test_step_with_nothing_generated_reports_nothing():
    assert _manager().drain_step_reward_metrics(5) == {}
