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

"""Unit tests for the wall-clock time-efficiency reward in
nemo_rl/utils/time_efficiency.py."""

import pytest
from pydantic import ValidationError

from nemo_rl.utils.time_efficiency import (
    TimeEfficiencyConfig,
    apply_time_efficiency_reward,
    rollout_minutes,
)


def make_result(reward, run_time_s, resolved=True):
    """Minimal NeMo-Gym rollout result in the shape the module consumes."""
    return {
        "full_result": {
            "reward": reward,
            "openhands_run_time": run_time_s,
            "resolved": resolved,
        }
    }


def rewards(results):
    return [r["full_result"]["reward"] for r in results]


class TestTimeEfficiencyConfig:
    def test_defaults(self):
        cfg = TimeEfficiencyConfig()
        assert cfg.enabled is False
        assert cfg.lambda_time == pytest.approx(1.0 / 60.0)
        assert cfg.apply_to == "all"
        assert cfg.floor is None

    def test_rejects_unknown_apply_to(self):
        with pytest.raises(ValidationError):
            TimeEfficiencyConfig(apply_to="solved")


class TestRolloutMinutes:
    @pytest.mark.parametrize(
        "run_time_s, expected",
        [
            (600, 10.0),
            (90.0, 1.5),
            ("120", 2.0),
            (None, 0.0),
            ("n/a", 0.0),
            (-30, 0.0),
        ],
    )
    def test_parses_and_guards_bad_values(self, run_time_s, expected):
        assert rollout_minutes(make_result(1.0, run_time_s)) == pytest.approx(expected)

    def test_missing_key_counts_as_zero(self):
        assert rollout_minutes({"full_result": {"reward": 1.0}}) == 0.0


class TestApplyTimeEfficiencyReward:
    def test_none_or_disabled_is_a_noop(self):
        for cfg in (None, TimeEfficiencyConfig(enabled=False)):
            results = [make_result(1.0, 1800.0), make_result(0.0, 3600.0)]
            assert apply_time_efficiency_reward(results, cfg) == {}
            assert rewards(results) == [1.0, 0.0]

    def test_empty_group(self):
        assert (
            apply_time_efficiency_reward([], TimeEfficiencyConfig(enabled=True)) == {}
        )

    def test_deducts_lambda_per_minute_from_every_rollout(self):
        results = [
            make_result(1.0, 1800.0, resolved=True),  # 30 min
            make_result(0.0, 3600.0, resolved=False),  # 60 min
        ]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True)
        )

        assert rewards(results) == pytest.approx([0.5, -1.0])
        assert stats == pytest.approx(
            {
                "time_efficiency/minutes_mean": 45.0,
                "time_efficiency/minutes_max": 60.0,
                "time_efficiency/deduction_mean": 0.75,
                "time_efficiency/deduction_max": 1.0,
                "time_efficiency/group_has_signal": 1.0,
            }
        )

    def test_correct_only_leaves_failures_untouched(self):
        results = [
            make_result(1.0, 1800.0, resolved=True),
            make_result(0.0, 3600.0, resolved=False),
        ]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True, apply_to="correct")
        )

        assert rewards(results) == pytest.approx([0.5, 0.0])
        # Skipped rollouts still count toward the group means with a 0 deduction.
        assert stats["time_efficiency/deduction_mean"] == pytest.approx(0.25)
        assert stats["time_efficiency/minutes_mean"] == pytest.approx(45.0)

    def test_floor_clamps_the_post_deduction_reward(self):
        results = [make_result(1.0, 5400.0), make_result(0.0, 3600.0)]  # 90 / 60 min
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True, floor=0.0)
        )

        assert rewards(results) == pytest.approx([0.0, 0.0])
        assert stats["time_efficiency/deduction_max"] == pytest.approx(1.0)
        assert stats["time_efficiency/deduction_mean"] == pytest.approx(0.5)

    def test_custom_lambda(self):
        results = [make_result(1.0, 600.0)]  # 10 min
        apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True, lambda_time=0.01)
        )
        assert rewards(results) == pytest.approx([0.9])

    def test_missing_timing_costs_nothing(self):
        results = [{"full_result": {"reward": 1.0, "resolved": True}}]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True)
        )
        assert rewards(results) == [1.0]
        assert stats["time_efficiency/deduction_max"] == 0.0

    def test_none_reward_is_treated_as_zero(self):
        results = [make_result(None, 1800.0)]
        apply_time_efficiency_reward(results, TimeEfficiencyConfig(enabled=True))
        assert rewards(results) == pytest.approx([-0.5])

    def test_group_has_signal_is_zero_for_equal_wall_times(self):
        results = [make_result(1.0, 600.0), make_result(0.0, 600.0)]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True)
        )
        assert stats["time_efficiency/group_has_signal"] == 0.0
