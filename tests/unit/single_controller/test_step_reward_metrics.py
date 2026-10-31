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

from nemo_rl.experience.rollout_manager import RolloutManager, rewards_carry_no_signal


def _manager() -> RolloutManager:
    # The tally methods touch one dict and nothing else, so a full manager (tokenizer,
    # policy, Ray) is not needed to exercise them.
    manager = RolloutManager.__new__(RolloutManager)
    manager._step_rewards = {}
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


def test_zero_variance_group_predicate_reads_completion_rewards():
    record = SimpleNamespace(
        completions=[SimpleNamespace(reward=1.0), SimpleNamespace(reward=1.0)]
    )
    assert RolloutManager._is_zero_variance_group(record) is True
    record.completions.append(SimpleNamespace(reward=0.0))
    assert RolloutManager._is_zero_variance_group(record) is False


def test_unfiltered_reward_counts_kept_and_dropped_groups():
    manager = _manager()
    # Two original groups: one mixed (kept), one all-zero (dropped by dynamic sampling).
    manager.record_group_rewards(0, [1.0, 0.0, 1.0, 0.0], replacement_index=0)
    manager.record_group_rewards(0, [0.0, 0.0, 0.0, 0.0], replacement_index=0)
    # Its replacement: mixed, and the second slot's replacement: all-one (dropped).
    manager.record_group_rewards(0, [1.0, 1.0, 1.0, 0.0], replacement_index=1)
    manager.record_group_rewards(0, [1.0, 1.0, 1.0, 1.0], replacement_index=1)

    metrics = manager.pop_step_reward_metrics(0)

    assert metrics["generated_prompt_groups"] == 4.0
    assert metrics["zero_variance_prompt_groups"] == 2.0
    assert metrics["zero_variance_fraction"] == pytest.approx(0.5)
    # (2 + 0 + 3 + 4) / 16 over every generated group, not just the two kept ones.
    assert metrics["unfiltered_reward"] == pytest.approx(9 / 16)
    # Only the two original groups: (2 + 0) / 8.
    assert metrics["first_attempt_reward"] == pytest.approx(2 / 8)


def test_kept_groups_alone_would_hide_the_signal():
    manager = _manager()
    manager.record_group_rewards(0, [1.0, 0.0], replacement_index=0)
    manager.record_group_rewards(0, [1.0, 1.0], replacement_index=0)
    manager.record_group_rewards(0, [1.0, 1.0], replacement_index=0)
    manager.record_group_rewards(0, [0.0, 0.0], replacement_index=0)

    metrics = manager.pop_step_reward_metrics(0)

    # The one mixed group alone averages 0.5; the whole batch is 5/8.
    assert metrics["unfiltered_reward"] == pytest.approx(5 / 8)
    assert metrics["zero_variance_fraction"] == pytest.approx(0.75)


def test_unstamped_groups_are_not_recorded():
    manager = _manager()
    manager.record_group_rewards(None, [1.0, 0.0], replacement_index=0)
    assert manager._step_rewards == {}


def test_empty_reward_list_is_not_recorded():
    manager = _manager()
    manager.record_group_rewards(0, [], replacement_index=0)
    assert manager.pop_step_reward_metrics(0) == {}


def test_replacement_only_step_has_no_first_attempt_reward():
    manager = _manager()
    manager.record_group_rewards(0, [1.0, 0.0], replacement_index=2)

    metrics = manager.pop_step_reward_metrics(0)

    assert "first_attempt_reward" not in metrics
    assert metrics["unfiltered_reward"] == pytest.approx(0.5)


def test_pop_forgets_the_closed_step_and_older_ones_only():
    manager = _manager()
    manager.record_group_rewards(0, [1.0, 0.0], replacement_index=0)
    manager.record_group_rewards(1, [0.0, 0.0], replacement_index=0)
    manager.record_group_rewards(2, [1.0, 1.0], replacement_index=0)

    assert manager.pop_step_reward_metrics(1)["unfiltered_reward"] == 0.0
    # Step 0 was older than the closed step, so it is gone without having been read.
    assert manager.pop_step_reward_metrics(0) == {}
    # Step 2 was generated ahead (lookahead) and survives until its own step closes.
    assert manager.pop_step_reward_metrics(2)["unfiltered_reward"] == 1.0
    assert manager.pop_step_reward_metrics(2) == {}


def test_step_with_nothing_generated_reports_nothing():
    assert _manager().pop_step_reward_metrics(5) == {}
