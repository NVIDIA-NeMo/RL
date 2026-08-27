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

"""Unit tests for the profile_band multiplier and group relative-length
scaling algorithms in nemo_rl/utils/length_adjustments.py."""

import pytest

from nemo_rl.utils.length_adjustments import (
    _band_multiplier,
    apply_group_length_adjustments,
)

AGENT = "math_with_judge_simple_agent"


def make_result(reasoning: str, answer: str, reward: float, band=None):
    """Build a minimal rollout result dict in the shape the module consumes."""
    result = {
        "full_result": {
            "reward": reward,
            "response": {
                "output": [
                    {"type": "reasoning", "summary": [{"text": reasoning}]},
                    {"type": "message", "content": [{"text": answer}]},
                ]
            },
        },
        "agent_ref": {"name": AGENT},
    }
    if band is not None:
        result["profile_band"] = band
    return result


def make_config(default=None, profile_band=None, num_gens=2):
    length_bonus = {}
    if default is not None:
        length_bonus["default"] = {"length_type": "chars", **default}
    if profile_band is not None:
        length_bonus["profile_band"] = profile_band
    return {
        "grpo": {
            "num_generations_per_prompt": num_gens,
            "length_bonus": length_bonus,
        }
    }


def rewards_of(results):
    return [r["full_result"]["reward"] for r in results]


class TestBandMultiplier:
    """Direct tests of the {a, b, f} multiplier shape."""

    CH = {"a": 10, "b": 20, "f": 0.5}

    def test_at_or_below_a_is_one(self):
        assert _band_multiplier(5, self.CH) == 1.0
        assert _band_multiplier(10, self.CH) == 1.0

    def test_linear_interpolation_between_a_and_b(self):
        assert _band_multiplier(15, self.CH) == pytest.approx(0.75)

    def test_exactly_b_is_f(self):
        assert _band_multiplier(20, self.CH) == pytest.approx(0.5)

    def test_clamps_at_f_past_b(self):
        # Past b the multiplier stays at f; it must NOT keep decaying to 0.
        assert _band_multiplier(25, self.CH) == pytest.approx(0.5)
        assert _band_multiplier(30, self.CH) == pytest.approx(0.5)
        assert _band_multiplier(10_000, self.CH) == pytest.approx(0.5)

    def test_missing_or_malformed_channel_is_noop(self):
        assert _band_multiplier(100, None) == 1.0
        assert _band_multiplier(100, {}) == 1.0
        assert _band_multiplier(100, {"a": 10, "b": 20}) == 1.0  # missing f
        assert _band_multiplier(100, {"a": 20, "b": 10, "f": 0.5}) == 1.0  # b <= a


class TestProfileBandPerRow:
    """profile_band multipliers driven by per-row dataset metadata."""

    def test_total_channel_scales_correct_rollouts(self):
        band = {"total": {"a": 10, "b": 20, "f": 0.5}}
        results = [
            make_result("12345", "12345", 1.0, band=band),  # total 10 -> x1.0
            make_result("1234567890", "1234567890", 1.0, band=band),  # 20 -> x0.5
        ]
        cfg = make_config(default={"enabled": True, "profile_band_total": True})
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.5])

    def test_zero_reward_rollouts_untouched(self):
        band = {"total": {"a": 10, "b": 20, "f": 0.5}}
        results = [
            make_result("1234567890", "1234567890", 0.0, band=band),
            make_result("1234567890", "1234567890", 1.0, band=band),
        ]
        cfg = make_config(default={"enabled": True, "profile_band_total": True})
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([0.0, 0.5])

    def test_reasoning_channel_ignores_answer_length(self):
        band = {"reasoning": {"a": 10, "b": 20, "f": 0.5}}
        long_answer = "x" * 100  # must not affect the reasoning channel
        results = [
            make_result("12345", long_answer, 1.0, band=band),  # reasoning 5 -> x1.0
            make_result("123456789012345", long_answer, 1.0, band=band),  # 15 -> x0.75
        ]
        cfg = make_config(default={"enabled": True, "profile_band_reasoning": True})
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.75])

    def test_missing_row_band_is_noop(self):
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        cfg = make_config(default={"enabled": True, "profile_band_total": True})
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_channel_not_enabled_in_config_is_noop(self):
        band = {"total": {"a": 10, "b": 20, "f": 0.5}}
        results = [
            make_result("1234567890", "1234567890", 1.0, band=band),
            make_result("12345", "12345", 1.0, band=band),
        ]
        cfg = make_config(default={"enabled": True})  # no profile_band_* flag
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])


class TestProfileBandGlobalDefaults:
    """profile_band driven by config-level length_bonus.profile_band defaults."""

    def test_global_total_only(self):
        cfg = make_config(
            default={"enabled": True},
            profile_band={
                "enabled": True,
                "defaults": {"total": {"a": 10, "b": 20, "f": 0.5}},
            },
        )
        results = [
            make_result("12345", "12345", 1.0),  # total 10 -> x1.0
            make_result("1234567890123456789012345", "12345", 1.0),  # 30 -> x0.5
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.5])

    def test_global_works_without_default_block(self):
        # Channels under defaults are implicitly enabled; no length_bonus.default
        # `enabled` or profile_band_* booleans required.
        cfg = make_config(
            default={},  # only length_type
            profile_band={
                "enabled": True,
                "defaults": {"reasoning": {"a": 10, "b": 20, "f": 0.5}},
            },
        )
        results = [
            make_result("123456789012345", "xx", 1.0),  # reasoning 15 -> x0.75
            make_result("12345", "xx", 1.0),  # 5 -> x1.0
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([0.75, 1.0])

    def test_row_band_wins_over_global(self):
        cfg = make_config(
            default={"enabled": True},
            profile_band={
                "enabled": True,
                "defaults": {"total": {"a": 10, "b": 20, "f": 0.5}},
            },
        )
        generous = {"total": {"a": 100, "b": 200, "f": 0.5}}
        results = [
            make_result("12345", "12345", 1.0, band=generous),
            make_result("1234567890123456789012345", "12345", 1.0, band=generous),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_disabled_block_is_noop(self):
        cfg = make_config(
            default={"enabled": True},
            profile_band={
                "enabled": False,
                "defaults": {"total": {"a": 10, "b": 20, "f": 0.5}},
            },
        )
        results = [
            make_result("1234567890123456789012345", "12345", 1.0),
            make_result("12345", "12345", 1.0),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_malformed_global_channel_ignored(self):
        cfg = make_config(
            default={"enabled": True},
            profile_band={
                "enabled": True,
                "defaults": {"total": {"a": 20, "b": 10, "f": 0.5}},  # b <= a
            },
        )
        results = [
            make_result("1234567890123456789012345", "12345", 1.0),
            make_result("12345", "12345", 1.0),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])


class TestGroupRelativeLengthScaling:
    """Dense zero-centered group relative-length penalty."""

    def test_two_rollouts_symmetric_adjustment(self):
        # lengths 10 and 30: raw weights 1 and 0, centered +0.5/-0.5, coeff 0.1.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        results = [
            make_result("12345", "12345", 1.0),  # total 10 -> +0.05
            make_result("1234567890123456789012345", "12345", 1.0),  # 30 -> -0.05
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 0.95])

    def test_three_rollouts_zero_centered(self):
        # lengths 10/20/30 -> raw weights 1/0.5/0 -> centered +0.5/0/-0.5.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1},
            num_gens=3,
        )
        results = [
            make_result("12345", "12345", 1.0),  # 10
            make_result("1234567890", "1234567890", 1.0),  # 20
            make_result("123456789012345", "123456789012345", 1.0),  # 30
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 1.0, 0.95])
        # Zero-centered: the group's mean reward is unchanged by the adjustment.
        assert sum(rewards_of(results)) == pytest.approx(3.0)

    def test_equal_lengths_no_adjustment(self):
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        results = [
            make_result("12345", "12345", 1.0),
            make_result("12345", "12345", 1.0),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_only_positive_rollouts_participate(self):
        # The zero-reward rollout is neither adjusted nor part of min/max, so
        # the two positives (10 and 30) still get the symmetric +/-0.05.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1},
            num_gens=3,
        )
        results = [
            make_result("12345", "12345", 1.0),  # 10 -> +0.05
            make_result("1" * 1000, "1" * 1000, 0.0),  # untouched, excluded
            make_result("1234567890123456789012345", "12345", 1.0),  # 30 -> -0.05
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 0.0, 0.95])

    def test_reasoning_channel_uses_reasoning_length_only(self):
        # Same total lengths, different reasoning/answer split: only the
        # reasoning coefficient is on, so the shorter-reasoning rollout wins.
        cfg = make_config(
            default={"enabled": True, "group_reasoning_length_penalty_coeff": 0.1}
        )
        results = [
            make_result("12345", "123456789012345", 1.0),  # reasoning 5 -> +0.05
            make_result("123456789012345", "12345", 1.0),  # reasoning 15 -> -0.05
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 0.95])

    def test_zero_coefficient_is_noop(self):
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.0}
        )
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_agent_override_disables_for_agent(self):
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        cfg["grpo"]["length_bonus"]["agent_overrides"] = {AGENT: {"enabled": False}}
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])


class TestProfiledLengthPenalty:
    """Profiled-length threshold penalty and its passing-samples requirement."""

    @staticmethod
    def make_profiled(reasoning, answer, reward, p_rewards, p_lengths):
        r = make_result(reasoning, answer, reward)
        r["profiled_rewards"] = p_rewards
        r["profiled_output_lengths"] = p_lengths
        return r

    def cfg(self, min_samples=2):
        return make_config(
            default={
                "enabled": True,
                "profiled_length_penalty": 0.3,
                "profiled_length_n_std": 1.0,
                "profiled_length_min_samples": min_samples,
            }
        )

    def test_enough_passes_penalizes_over_threshold(self):
        # Passing profiled lengths 10 and 14: threshold = 12 + 1*std(=~2.83) ~ 14.83.
        p_rewards, p_lengths = [1, 1, 0], [10, 14, 100]
        results = [
            self.make_profiled("12345", "12345", 1.0, p_rewards, p_lengths),  # 10 < thr
            self.make_profiled("1234567890", "1234567890", 1.0, p_rewards, p_lengths),  # 20 >= thr
        ]
        apply_group_length_adjustments(results, self.cfg())
        assert rewards_of(results) == pytest.approx([1.0, 0.7])
        # The failing profiled length (100) must not have entered the threshold:
        # with it, mean+std would exceed 20 and nothing would be penalized.

    def test_one_pass_below_min_samples_no_penalty(self):
        # Only 1 passing profiled rollout with min_samples=2: no penalty for
        # anyone — no fallback to failing profiled lengths.
        p_rewards, p_lengths = [1, 0, 0], [10, 100, 120]
        results = [
            self.make_profiled("1234567890", "1234567890", 1.0, p_rewards, p_lengths),
            self.make_profiled("1" * 50, "1" * 50, 1.0, p_rewards, p_lengths),
        ]
        apply_group_length_adjustments(results, self.cfg())
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_zero_passes_no_penalty(self):
        p_rewards, p_lengths = [0, 0, 0], [10, 12, 14]
        results = [
            self.make_profiled("1234567890", "1234567890", 1.0, p_rewards, p_lengths),
            self.make_profiled("1" * 50, "1" * 50, 1.0, p_rewards, p_lengths),
        ]
        apply_group_length_adjustments(results, self.cfg())
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_one_pass_allowed_when_min_samples_is_one(self):
        # min_samples=1 opts in to single-pass thresholds: threshold = 10 + 0.
        p_rewards, p_lengths = [1, 0], [10, 100]
        results = [
            self.make_profiled("1234", "1234", 1.0, p_rewards, p_lengths),  # 8 < 10
            self.make_profiled("1234567890", "1234567890", 1.0, p_rewards, p_lengths),  # 20 >= 10
        ]
        apply_group_length_adjustments(results, self.cfg(min_samples=1))
        assert rewards_of(results) == pytest.approx([1.0, 0.7])


class TestReviewFixes:
    """Regression tests for review findings on PR #3852."""

    def test_flat_penalty_exceeding_reward_clamps_at_zero(self):
        # Stacked/oversized flat penalties may wipe a correct rollout's reward
        # out but never flip its sign: 1.0 - 1.5 clamps to 0.0, not -0.5.
        results = [
            make_result("1234", "1234", 1.0),  # len 8 < threshold 10: untouched
            make_result("1234567890", "1234567890", 1.0),  # len 20 >= 10: clamped
        ]
        for r in results:
            r["profiled_rewards"] = [1, 1]
            r["profiled_output_lengths"] = [10, 10]
        cfg = make_config(
            default={"enabled": True, "profiled_length_penalty": 1.5}
        )
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.0])

    def test_negative_env_reward_not_raised_by_clamp(self):
        # The clamp applies only to originally-positive rollouts: an env's own
        # negative reward must pass through untouched.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        results = [
            make_result("12345", "12345", -1.0),
            make_result("1234567890", "1234567890", 1.0),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results)[0] == pytest.approx(-1.0)

    def test_band_multiplier_never_rewards_length_on_penalized_base(self):
        # Flat penalty exceeds the reward, so bases clamp to 0; the band phase
        # must leave them at 0 (base * m - base > 0 for base < 0 would have
        # made longer rollouts score HIGHER pre-clamp).
        band = {"total": {"a": 1, "b": 2, "f": 0.1}}
        results = [
            make_result("12345", "12345", 1.0, band=band),
            make_result("1234567890", "1234567890", 1.0, band=band),
        ]
        for r in results:
            r["profiled_rewards"] = [1, 1]
            r["profiled_output_lengths"] = [5, 5]
        cfg = make_config(
            default={
                "enabled": True,
                "profiled_length_penalty": 1.5,  # threshold 5: both clamped to 0
                "profile_band_total": True,
            }
        )
        apply_group_length_adjustments(results, cfg)
        r_short, r_long = rewards_of(results)
        assert r_short == pytest.approx(0.0)
        assert r_long == pytest.approx(0.0)
        assert r_long <= r_short  # longer must never beat shorter

    def test_explicit_false_channel_not_overridden_by_global_defaults(self):
        cfg = make_config(
            default={"enabled": True, "profile_band_total": False},
            profile_band={
                "enabled": True,
                "defaults": {"total": {"a": 10, "b": 20, "f": 0.5}},
            },
        )
        results = [
            make_result("1234567890123456789012345", "12345", 1.0),  # total 30
            make_result("12345", "12345", 1.0),
        ]
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_unknown_key_raises(self):
        cfg = make_config(default={"enabled": True, "group_total_length_coeff": 0.1})
        results = [make_result("12345", "12345", 1.0), make_result("123", "123", 1.0)]
        try:
            apply_group_length_adjustments(results, cfg)
        except ValueError as e:
            assert "group_total_length_coeff" in str(e)
        else:
            raise AssertionError("expected ValueError for unknown config key")

    def test_top_percentile_default_is_half(self):
        # With 2 positive scorers (rewards 1.0 and 0.9) and default
        # top_percentile=0.5, only the best scorer is a "top scorer" — the
        # longest-total penalty needs >= 2 eligible top scorers, so it no-ops.
        # With the old effective default of 0.0 the behavior was the same, but
        # an explicit 1.0 must include both and penalize the longer one.
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890", "1234567890", 0.9),
        ]
        cfg = make_config(
            default={"enabled": True, "longest_total_penalty": 0.2, "top_percentile": 1.0}
        )
        apply_group_length_adjustments(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.7])
