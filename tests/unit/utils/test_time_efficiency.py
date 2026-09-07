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
    rollout_calls,
    rollout_minutes,
)


def make_result(reward, run_time_s, resolved=True, calls=0, malformed=0):
    """Minimal NeMo-Gym rollout result in the shape the module consumes."""
    output = [{"type": "function_call", "name": "bash"} for _ in range(calls)]
    # Rejected calls surface as plain messages, never as function_call items.
    output += [{"type": "message", "content": "<tool_call>"} for _ in range(malformed)]
    return {
        "full_result": {
            "reward": reward,
            "openhands_run_time": run_time_s,
            "resolved": resolved,
            "response": {"output": output},
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
        assert cfg.lambda_call_bonus == 0.0
        assert cfg.call_bonus_ref is None
        assert cfg.floor is None

    def test_rejects_unknown_apply_to(self):
        with pytest.raises(ValidationError):
            TimeEfficiencyConfig(apply_to="solved")

    def test_rejects_negative_call_bonus(self):
        with pytest.raises(ValidationError):
            TimeEfficiencyConfig(lambda_call_bonus=-0.1, call_bonus_ref=56.0)

    @pytest.mark.parametrize("call_bonus_ref", [None, 0.0, -5.0])
    def test_call_bonus_requires_positive_ref(self, call_bonus_ref):
        with pytest.raises(ValidationError, match="call_bonus_ref must be a positive"):
            TimeEfficiencyConfig(lambda_call_bonus=0.1, call_bonus_ref=call_bonus_ref)

    def test_ref_without_bonus_is_allowed(self):
        TimeEfficiencyConfig(call_bonus_ref=56.0)


class TestRolloutCalls:
    def test_counts_only_function_call_items(self):
        assert rollout_calls(make_result(1.0, 60.0, calls=3, malformed=2)) == 3

    @pytest.mark.parametrize(
        "full_result",
        [
            {"reward": 1.0},
            {"reward": 1.0, "response": None},
            {"reward": 1.0, "response": {"output": None}},
            {"reward": 1.0, "response": {"output": ["not-a-dict"]}},
        ],
    )
    def test_missing_or_malformed_output_counts_as_zero(self, full_result):
        assert rollout_calls({"full_result": full_result}) == 0


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
                "time_efficiency/minutes/mean": 45.0,
                "time_efficiency/minutes/max": 60.0,
                "time_efficiency/deduction/mean": 0.75,
                "time_efficiency/deduction/max": 1.0,
                "time_efficiency/calls/mean": 0.0,
                "time_efficiency/bonus/mean": 0.0,
                # 90 min of wall time and no calls: max(sum(calls), 1) guards the division.
                "time_efficiency/seconds_per_call": 5400.0,
                "time_efficiency/bonus_saturated_frac": 0.0,
                "time_efficiency/floored_frac": 0.0,
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
        assert stats["time_efficiency/deduction/mean"] == pytest.approx(0.25)
        assert stats["time_efficiency/minutes/mean"] == pytest.approx(45.0)

    def test_correct_skips_resolved_rollouts_zeroed_by_a_penalty(self):
        # A reward-zeroing penalty ran before us: resolved but reward == 0.0 is
        # treated as a failure and not charged.
        results = [
            make_result(0.0, 1800.0, resolved=True),
            make_result(1.0, 1800.0, resolved=True),
        ]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True, apply_to="correct")
        )

        assert rewards(results) == pytest.approx([0.0, 0.5])
        assert stats["time_efficiency/deduction/max"] == pytest.approx(0.5)

    def test_floor_clamps_the_post_deduction_reward(self):
        results = [make_result(1.0, 5400.0), make_result(0.0, 3600.0)]  # 90 / 60 min
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True, floor=0.0)
        )

        assert rewards(results) == pytest.approx([0.0, 0.0])
        assert stats["time_efficiency/deduction/max"] == pytest.approx(1.0)
        assert stats["time_efficiency/deduction/mean"] == pytest.approx(0.5)

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
        assert stats["time_efficiency/deduction/max"] == 0.0

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

    def test_call_bonus_saturates_at_the_reference(self):
        # lambda_call_bonus 0.10, ref 50: 25 calls -> +0.05, 50 -> +0.10, 80 -> +0.10.
        results = [
            make_result(1.0, 1200.0, calls=25),  # 20 min
            make_result(1.0, 1200.0, calls=50),
            make_result(1.0, 1200.0, calls=80),
        ]
        cfg = TimeEfficiencyConfig(
            enabled=True, lambda_call_bonus=0.10, call_bonus_ref=50.0
        )
        stats = apply_time_efficiency_reward(results, cfg)

        penalty = 20.0 / 60.0
        assert rewards(results) == pytest.approx(
            [1.0 + 0.05 - penalty, 1.0 + 0.10 - penalty, 1.0 + 0.10 - penalty]
        )
        assert stats["time_efficiency/calls/mean"] == pytest.approx(155 / 3)
        assert stats["time_efficiency/bonus/mean"] == pytest.approx(0.25 / 3)
        assert stats["time_efficiency/bonus_saturated_frac"] == pytest.approx(2 / 3)
        assert stats["time_efficiency/seconds_per_call"] == pytest.approx(
            3 * 1200.0 / 155
        )
        # deduction is the realized decrease, net of the bonus.
        assert stats["time_efficiency/deduction/max"] == pytest.approx(penalty - 0.05)

    def test_call_bonus_follows_the_correct_gate(self):
        # Under "correct" a failure earns neither the bonus nor the deduction,
        # so junk calls on a failed rollout cannot be rewarded.
        results = [
            make_result(0.0, 600.0, resolved=False, calls=200),
            make_result(1.0, 600.0, resolved=True, calls=10),
        ]
        cfg = TimeEfficiencyConfig(
            enabled=True,
            apply_to="correct",
            lambda_call_bonus=0.10,
            call_bonus_ref=50.0,
        )
        apply_time_efficiency_reward(results, cfg)
        assert rewards(results) == pytest.approx([0.0, 1.0 + 0.02 - 10.0 / 60.0])

    def test_floor_counts_floored_rows_and_lifts_the_full_timeout(self):
        # A correct 60-min rollout with no calls lands on exactly 0.0 and would
        # tie with the failures; the floor keeps it above them.
        results = [
            make_result(1.0, 3600.0, resolved=True, calls=0),
            make_result(1.0, 600.0, resolved=True, calls=50),
            make_result(0.0, 600.0, resolved=False),
        ]
        cfg = TimeEfficiencyConfig(
            enabled=True,
            apply_to="correct",
            lambda_call_bonus=0.10,
            call_bonus_ref=50.0,
            floor=0.05,
        )
        stats = apply_time_efficiency_reward(results, cfg)
        assert rewards(results) == pytest.approx([0.05, 1.0 + 0.10 - 10.0 / 60.0, 0.0])
        assert stats["time_efficiency/floored_frac"] == pytest.approx(1 / 3)

    def test_zero_call_bonus_is_the_plain_deduction(self):
        results = [
            make_result(1.0, 1800.0, calls=40),
            make_result(0.0, 3600.0, calls=5),
        ]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True)
        )
        assert rewards(results) == pytest.approx([0.5, -1.0])
        assert stats["time_efficiency/bonus/mean"] == 0.0
        assert stats["time_efficiency/bonus_saturated_frac"] == 0.0
        assert stats["time_efficiency/calls/mean"] == pytest.approx(22.5)

    def test_masked_rows_are_not_charged_when_the_loss_mask_is_on(self):
        # Three trainable failures at equal wall time plus a loss-masked 60-min
        # timeout (the group's longest rollout by construction).
        def group():
            return [
                make_result(0.0, 600.0, resolved=False),
                make_result(0.0, 600.0, resolved=False),
                make_result(0.0, 600.0, resolved=False),
                make_result(0.0, 3600.0, resolved=False),
            ]

        results = group()
        stats = apply_time_efficiency_reward(
            results,
            TimeEfficiencyConfig(enabled=True),
            mask_sample=[False, False, False, True],
        )
        # Masked row keeps its raw reward; trainable rows are charged as usual.
        assert rewards(results) == pytest.approx([-1 / 6, -1 / 6, -1 / 6, 0.0])
        assert stats["time_efficiency/deduction/max"] == pytest.approx(1 / 6)
        # The trainable rows have identical deductions, so there is no signal
        # even though the masked row's 0 deduction differs from theirs.
        assert stats["time_efficiency/group_has_signal"] == 0.0

        # With the loss mask off those rows train, so they are charged.
        results = group()
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True), mask_sample=None
        )
        assert rewards(results) == pytest.approx([-1 / 6, -1 / 6, -1 / 6, -1.0])
        assert stats["time_efficiency/deduction/max"] == pytest.approx(1.0)
        assert stats["time_efficiency/group_has_signal"] == 1.0

    def test_mask_sample_length_mismatch_raises(self):
        results = [make_result(1.0, 600.0), make_result(1.0, 600.0)]
        with pytest.raises(ValueError, match="mask_sample has 1 entries"):
            apply_time_efficiency_reward(
                results, TimeEfficiencyConfig(enabled=True), mask_sample=[True]
            )

    def test_group_has_signal_follows_deductions_not_wall_times(self):
        # "correct" on an all-failed group: wall times differ, nothing deducted.
        results = [
            make_result(0.0, 600.0, resolved=False),
            make_result(0.0, 3600.0, resolved=False),
        ]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True, apply_to="correct")
        )
        assert stats["time_efficiency/group_has_signal"] == 0.0

        # floor clamps both rewards to the same value: equal deductions, no signal.
        results = [make_result(1.0, 5400.0), make_result(1.0, 7200.0)]  # 90 / 120 min
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True, floor=0.0)
        )
        assert rewards(results) == pytest.approx([0.0, 0.0])
        assert stats["time_efficiency/group_has_signal"] == 0.0

        # one second of difference is a signal.
        results = [make_result(1.0, 600.0), make_result(1.0, 601.0)]
        stats = apply_time_efficiency_reward(
            results, TimeEfficiencyConfig(enabled=True)
        )
        assert stats["time_efficiency/group_has_signal"] == 1.0
