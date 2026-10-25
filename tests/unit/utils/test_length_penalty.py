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
scaling algorithms in nemo_rl/utils/length_penalty.py."""

import logging

import pytest

import nemo_rl.utils.length_penalty as length_penalty_mod
from nemo_rl.utils.length_penalty import (
    LengthPenaltyConfig,
    _band_multiplier,
    apply_group_length_penalties,
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
    length_penalty = {}
    if default is not None:
        length_penalty["default"] = {"length_type": "chars", **default}
    if profile_band is not None:
        length_penalty["profile_band"] = profile_band
    return {
        "grpo": {
            "num_generations_per_prompt": num_gens,
            "length_penalty": length_penalty,
        }
    }


def apply(results, cfg, tokenizer=None):
    """Call the hook the way rollouts.py does: block + group size, not master config."""
    return apply_group_length_penalties(
        results,
        cfg["grpo"]["length_penalty"],
        cfg["grpo"]["num_generations_per_prompt"],
        tokenizer=tokenizer,
    )


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
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.5])

    def test_zero_reward_rollouts_untouched(self):
        band = {"total": {"a": 10, "b": 20, "f": 0.5}}
        results = [
            make_result("1234567890", "1234567890", 0.0, band=band),
            make_result("1234567890", "1234567890", 1.0, band=band),
        ]
        cfg = make_config(default={"enabled": True, "profile_band_total": True})
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([0.0, 0.5])

    def test_reasoning_channel_ignores_answer_length(self):
        band = {"reasoning": {"a": 10, "b": 20, "f": 0.5}}
        long_answer = "x" * 100  # must not affect the reasoning channel
        results = [
            make_result("12345", long_answer, 1.0, band=band),  # reasoning 5 -> x1.0
            make_result("123456789012345", long_answer, 1.0, band=band),  # 15 -> x0.75
        ]
        cfg = make_config(default={"enabled": True, "profile_band_reasoning": True})
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.75])

    def test_missing_row_band_is_noop(self):
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        cfg = make_config(default={"enabled": True, "profile_band_total": True})
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_channel_not_enabled_in_config_is_noop(self):
        band = {"total": {"a": 10, "b": 20, "f": 0.5}}
        results = [
            make_result("1234567890", "1234567890", 1.0, band=band),
            make_result("12345", "12345", 1.0, band=band),
        ]
        cfg = make_config(default={"enabled": True})  # no profile_band_* flag
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])


class TestProfileBandGlobalDefaults:
    """profile_band driven by config-level length_penalty.profile_band defaults."""

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
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.5])

    def test_global_works_without_default_block(self):
        # Channels under defaults are implicitly enabled; no length_penalty.default
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
        apply(results, cfg)
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
        apply(results, cfg)
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
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_malformed_global_channel_rejected(self):
        # b <= a and f outside [0, 1] are config errors, caught at validation
        # rather than silently ignoring the channel at rollout time.
        for channel in ({"a": 20, "b": 10, "f": 0.5}, {"a": 10, "b": 20, "f": 1.5}):
            with pytest.raises(ValueError):
                LengthPenaltyConfig.model_validate(
                    {"profile_band": {"enabled": True, "defaults": {"total": channel}}}
                )


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
        apply(results, cfg)
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
        apply(results, cfg)
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
        apply(results, cfg)
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
        apply(results, cfg)
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
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 0.95])

    def test_zero_coefficient_is_noop(self):
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.0}
        )
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_agent_override_disables_for_agent(self):
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        cfg["grpo"]["length_penalty"]["agent_overrides"] = {AGENT: {"enabled": False}}
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        apply(results, cfg)
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
            self.make_profiled(
                "1234567890", "1234567890", 1.0, p_rewards, p_lengths
            ),  # 20 >= thr
        ]
        apply(results, self.cfg())
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
        apply(results, self.cfg())
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_zero_passes_no_penalty(self):
        p_rewards, p_lengths = [0, 0, 0], [10, 12, 14]
        results = [
            self.make_profiled("1234567890", "1234567890", 1.0, p_rewards, p_lengths),
            self.make_profiled("1" * 50, "1" * 50, 1.0, p_rewards, p_lengths),
        ]
        apply(results, self.cfg())
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_one_pass_allowed_when_min_samples_is_one(self):
        # min_samples=1 opts in to single-pass thresholds: threshold = 10 + 0.
        p_rewards, p_lengths = [1, 0], [10, 100]
        results = [
            self.make_profiled("1234", "1234", 1.0, p_rewards, p_lengths),  # 8 < 10
            self.make_profiled(
                "1234567890", "1234567890", 1.0, p_rewards, p_lengths
            ),  # 20 >= 10
        ]
        apply(results, self.cfg(min_samples=1))
        assert rewards_of(results) == pytest.approx([1.0, 0.7])


class TestPassRateLengthPenalty:
    """MAI-style penalty: -w * rho_q * |y_i| / l_max on correct rollouts."""

    def cfg(self, w=0.2, num_gens=2):
        return make_config(
            default={"enabled": True, "pass_rate_length_penalty_weight": w},
            num_gens=num_gens,
        )

    def test_all_correct_group_scales_by_relative_length(self):
        # rho_q = 1.0, l_max = 20: penalties are w*10/20 and w*20/20.
        results = [
            make_result("12345", "12345", 1.0),  # total 10 -> -0.1
            make_result("1234567890", "1234567890", 1.0),  # total 20 -> -0.2
        ]
        apply(results, self.cfg(w=0.2))
        assert rewards_of(results) == pytest.approx([0.9, 0.8])

    def test_pass_rate_scales_penalty(self):
        # 1 of 2 correct -> rho_q = 0.5; only the correct rollout is penalized:
        # 1.0 - 0.2 * 0.5 * 20/20 = 0.9. The wrong rollout stays at 0.
        results = [
            make_result("1234567890", "1234567890", 1.0),  # l_max contributor
            make_result("12345", "12345", 0.0),
        ]
        apply(results, self.cfg(w=0.2))
        assert rewards_of(results) == pytest.approx([0.9, 0.0])

    def test_all_wrong_group_gets_zero_penalty(self):
        # rho_q = 0 -> no penalty at all, group stays variance-free.
        results = [
            make_result("12345", "12345", 0.0),
            make_result("1234567890", "1234567890", 0.0),
        ]
        apply(results, self.cfg(w=0.2))
        assert rewards_of(results) == pytest.approx([0.0, 0.0])

    def test_zero_weight_is_noop(self):
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890", "1234567890", 1.0),
        ]
        apply(results, self.cfg(w=0.0))
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_wrong_rollout_length_does_not_set_normalizer(self):
        # l_max comes from CORRECT rollouts only: the wrong rollout's total of
        # 40 is ignored, the correct rollout (total 20) is its own max and
        # loses exactly w * rho = 0.2 * 0.5. A long wrong ramble must not
        # dilute the pressure on correct rollouts.
        results = [
            make_result("1234567890", "1234567890", 1.0),
            make_result("1" * 20, "1" * 20, 0.0),
        ]
        apply(results, self.cfg(w=0.2))
        assert rewards_of(results) == pytest.approx([1.0 - 0.2 * 0.5, 0.0])


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
        cfg = make_config(default={"enabled": True, "profiled_length_penalty": 1.5})
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.0])

    def test_negative_env_reward_group_skipped(self):
        # Length adjustments require binary rewards: a group containing a
        # negative env reward is skipped wholesale — no adjustment, no clamp.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        results = [
            make_result("12345", "12345", -1.0),
            make_result("1234567890", "1234567890", 1.0),
        ]
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([-1.0, 1.0])

    def test_graded_rewards_group_skipped(self):
        # Graded (non-binary) rewards also skip the group untouched.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        results = [
            make_result("12345", "12345", 0.5),
            make_result("1234567890", "1234567890", 1.0),
        ]
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([0.5, 1.0])

    def test_all_wrong_group_stays_variance_free(self):
        # profiled_length_penalty has no positive-reward gate, but the binary
        # clamp floors penalty-created negatives at 0: an all-wrong group must
        # stay all-zero (no within-group variance -> no GRPO gradient).
        results = [
            make_result("1234", "1234", 0.0),  # len 8, under threshold
            make_result("1234567890", "1234567890", 0.0),  # len 20, over
        ]
        for r in results:
            r["profiled_rewards"] = [1, 1]
            r["profiled_output_lengths"] = [10, 10]
        cfg = make_config(default={"enabled": True, "profiled_length_penalty": 0.3})
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([0.0, 0.0])

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
        apply(results, cfg)
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
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])

    def test_default_block_without_enabled_is_active(self):
        # `enabled` defaults True consistently: a default: block that omits it
        # applies its penalties (previously the early return used default False
        # and silently no-oped).
        cfg = make_config(default={"group_total_length_penalty_coeff": 0.2})
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.1, 0.9])

    def test_behavior_independent_of_empty_agent_overrides(self):
        # The same default: block must behave identically with or without an
        # unrelated empty agent_overrides entry (previously it flipped the
        # early return and changed behavior).
        base = {"group_total_length_penalty_coeff": 0.2}
        outs = []
        for extra_overrides in (None, {"some_agent": {}}):
            cfg = make_config(default=dict(base))
            if extra_overrides is not None:
                cfg["grpo"]["length_penalty"]["agent_overrides"] = extra_overrides
            results = [
                make_result("12345", "12345", 1.0),
                make_result("1234567890123456789012345", "12345", 1.0),
            ]
            apply(results, cfg)
            outs.append(rewards_of(results))
        assert outs[0] == pytest.approx(outs[1])

    def test_unknown_key_raises(self):
        cfg = make_config(default={"enabled": True, "group_total_length_coeff": 0.1})
        results = [make_result("12345", "12345", 1.0), make_result("123", "123", 1.0)]
        with pytest.raises(ValueError, match="group_total_length_coeff"):
            apply(results, cfg)

    def test_invalid_literal_values_rejected(self):
        # Typos in Literal-typed keys used to change behavior silently
        # (`token` counted characters; a misspelled gate channel closed the
        # gate on every group).
        for bad in (
            {"length_type": "token"},
            {"group_length_penalty_profile_gate_channel": "totl"},
            {"group_length_penalty_profile_gate_field": "x"},
        ):
            with pytest.raises(ValueError):
                LengthPenaltyConfig.model_validate({"default": bad})

    def test_agent_override_replaces_only_set_keys(self):
        # An override that sets one key inherits every other value from
        # `default` (merge uses model_dump(exclude_unset=True)), including
        # non-default ones like length_type: chars.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.2}
        )
        cfg["grpo"]["length_penalty"]["agent_overrides"] = {
            AGENT: {"group_total_length_penalty_coeff": 0.1}
        }
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890123456789012345", "12345", 1.0),
        ]
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 0.95])

    def test_longest_penalty_under_binary_rewards(self):
        # Under binary rewards all correct rollouts tie at the top score, so
        # top_percentile is inert and both rollouts are eligible top scorers;
        # the longest one takes the flat penalty (default top_percentile 0.5).
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890", "1234567890", 1.0),
        ]
        cfg = make_config(default={"enabled": True, "longest_total_penalty": 0.2})
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.8])

    def test_disabled_graded_agent_is_skipped_without_warning(
        self, caplog, monkeypatch
    ):
        monkeypatch.setattr(length_penalty_mod, "_NON_BINARY_WARNED_AGENTS", set())
        results = [make_result("r", "a", 0.7), make_result("r", "aaaa", 0.3)]
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        cfg["grpo"]["length_penalty"]["agent_overrides"] = {AGENT: {"enabled": False}}
        with caplog.at_level(logging.WARNING, logger=length_penalty_mod.__name__):
            apply(results, cfg)
        assert rewards_of(results) == [0.7, 0.3]
        assert "non-binary" not in caplog.text

    def test_multi_reward_group_skipped(self):
        # A multi-reward result must keep reward == sum(reward_components);
        # adjusting only the scalar would trip the rollout's
        # validate_reward_components_match_scalar check.
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        results = [
            make_result("12345", "12345", 1.0),
            make_result("1234567890", "1234567890", 1.0),
        ]
        for r in results:
            r["full_result"]["reward_components"] = {"correct": 1.0, "format": 0.0}
        metrics = apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0])
        assert metrics["length_penalty/skipped_non_binary_frac"] == 1.0

    def test_env_observation_messages_not_counted(self):
        # Gym's gymnasium_agent appends env observations to response.output as
        # user-role messages; only model output may drive the answer length.
        cfg = make_config(
            default={"enabled": True, "group_answer_length_penalty_coeff": 0.2}
        )
        results = []
        for obs_len in (10, 200, 400, 800):
            r = make_result("think", "same answer", 1.0)
            r["full_result"]["response"]["output"].insert(
                1,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"text": "o" * obs_len}],
                },
            )
            results.append(r)
        cfg["grpo"]["num_generations_per_prompt"] = 4
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_returns_rollout_metrics(self):
        # total 10 / 20 / 20 with profiled threshold 10: rows 1-2 lose 1.5
        # each and clamp to 0; row 3 is wrong and untouched.
        cfg = make_config(
            default={"enabled": True, "profiled_length_penalty": 1.5}, num_gens=4
        )
        results = [
            make_result("1234", "1234", 1.0),
            make_result("1234567890", "1234567890", 1.0),
            make_result("1234567890", "1234567890", 1.0),
            make_result("1234567890", "1234567890", 0.0),
        ]
        for r in results:
            r["profiled_rewards"] = [1, 1]
            r["profiled_output_lengths"] = [10, 10]
        metrics = apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert metrics["length_penalty/env_reward_mean"] == pytest.approx(0.75)
        assert metrics["length_penalty/reward_delta_mean"] == pytest.approx(-0.5)
        assert metrics["length_penalty/wiped_correct_frac"] == pytest.approx(0.5)
        assert metrics["length_penalty/adjusted_frac"] == pytest.approx(0.5)
        assert metrics["length_penalty/skipped_non_binary_frac"] == 0.0

    def test_metrics_match_batched_and_per_group(self):
        # The async path applies the hook once per prompt group and averages
        # the per-group metrics; the batched paths apply it once per batch.
        # Every metric is a per-row mean so the two agree: here an all-wrong
        # group plus a group with one wiped correct rollout.
        def make_groups():
            wrong = [make_result("1234567890", "1234567890", 0.0) for _ in range(4)]
            mixed = [
                make_result("1234", "1234", 1.0),
                make_result("1234567890", "1234567890", 1.0),  # wiped
                make_result("1234", "1234", 0.0),
                make_result("1234", "1234", 0.0),
            ]
            for r in wrong + mixed:
                r["profiled_rewards"] = [1, 1]
                r["profiled_output_lengths"] = [10, 10]
            return wrong, mixed

        cfg = make_config(
            default={"enabled": True, "profiled_length_penalty": 1.5}, num_gens=4
        )
        wrong, mixed = make_groups()
        batched = apply(wrong + mixed, cfg)
        wrong, mixed = make_groups()
        per_group = [apply(wrong, cfg), apply(mixed, cfg)]
        averaged = {
            k: sum(m[k] for m in per_group) / len(per_group) for k in per_group[0]
        }
        assert batched == pytest.approx(averaged)
        assert batched["length_penalty/wiped_correct_frac"] == pytest.approx(1 / 8)

    def test_nothing_enabled_returns_no_metrics(self):
        cfg = make_config(default={"enabled": False})
        results = [make_result("12345", "12345", 1.0), make_result("123", "123", 1.0)]
        assert apply(results, cfg) == {}
        assert rewards_of(results) == [1.0, 1.0]


class TestNemoGymPostprocessHook:
    """grpo.length_penalty applied by rollouts._postprocess_single_nemo_gym_group.

    Needs torch (imported lazily) -- the rest of this file is torch-free.
    """

    @staticmethod
    def postprocess(rows, texts, grpo_config, reward_penalty_config=None):
        from types import SimpleNamespace

        torch = pytest.importorskip("torch")

        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        from nemo_rl.experience.rollouts import _postprocess_single_nemo_gym_group
        from nemo_rl.utils.timer import Timer

        prompt = {"role": "user", "content": "q", "token_ids": torch.tensor([1])}
        reply = {"role": "assistant", "content": "a", "token_ids": torch.tensor([2])}
        results = [
            {
                "full_result": make_result(reasoning, answer, 1.0)["full_result"],
                "input_message_log": [prompt],
                "message_log": [prompt, reply],
            }
            for reasoning, answer in texts
        ]
        rollout_result = _postprocess_single_nemo_gym_group(
            nemo_gym_rows=rows,
            results=results,
            timer=Timer(),
            timer_prefix="timing/test",
            policy_generation=SimpleNamespace(cfg={"max_total_sequence_length": 128}),
            input_batch=BatchedDataDict({"loss_multiplier": torch.ones(len(rows))}),
            tokenizer=SimpleNamespace(pad_token_id=0),
            log_full_result_tables=False,
            reward_penalty_config=reward_penalty_config,
            length_penalty_config=grpo_config["length_penalty"],
            group_size=grpo_config["num_generations_per_prompt"],
        )
        return rollout_result

    def test_each_prompt_group_uses_its_own_rows(self):
        # Two prompt groups in one call (the legacy sync batch). Totals 10/20
        # vs profiled threshold 12: the long rollout loses 0.3 (AGENT) or 0.6
        # (code_agent override), then its own row band scales the rest:
        # x0.5 at b=20 (group 0), x0.75 halfway to b=30 (group 1).
        rows = [
            {
                "agent_ref": {"name": agent},
                "profiled_rewards": [1, 1],
                "profiled_output_lengths": [12, 12],
                "profile_band": {"total": {"a": 10, "b": b, "f": 0.5}},
            }
            for agent, b in ((AGENT, 20), ("code_agent", 30))
            for _ in range(2)
        ]
        grpo_config = make_config(
            default={"profiled_length_penalty": 0.3, "profile_band_total": True}
        )["grpo"]
        grpo_config["length_penalty"]["agent_overrides"] = {
            AGENT: {},
            "code_agent": {"profiled_length_penalty": 0.6},
        }
        texts = [("12345", "12345"), ("1234567890", "1234567890")] * 2
        rollout_result = self.postprocess(rows, texts, grpo_config)
        rewards = rollout_result.final_batch["total_reward"].tolist()
        assert rewards == pytest.approx([1.0, 0.35, 1.0, 0.3])
        # The pre-hook env reward survives as its own column, and the hook
        # reports its metrics next to the other reward shapers.
        assert rollout_result.final_batch["env_reward"].tolist() == [1.0] * 4
        metrics = rollout_result.rollout_metrics
        assert metrics["length_penalty/env_reward_mean"] == pytest.approx(1.0)
        assert metrics["length_penalty/reward_delta_mean"] == pytest.approx(
            (0.35 - 1.0 + 0.3 - 1.0) / 4
        )

    def test_runs_after_reward_penalties(self):
        # The duplicated-reasoning penalty zeroes row 2 first, so it drops out
        # of the positive set: rows 0/1 (totals 10/30) get exactly +/-0.05.
        rollout_result = self.postprocess(
            [{"agent_ref": {"name": AGENT}} for _ in range(3)],
            [("1234", "123456"), ("1" * 20, "1" * 10), ("x" * 50, "x" * 50)],
            make_config(default={"group_total_length_penalty_coeff": 0.1}, num_gens=3)[
                "grpo"
            ],
            reward_penalty_config={"penalize_duplicated_reasoning": True},
        )
        rewards = rollout_result.final_batch["total_reward"].tolist()
        assert rewards == pytest.approx([1.05, 0.95, 0.0])
        # env_reward is the reward as handed to the length hook, i.e. after the
        # reward penalties already zeroed row 2.
        assert rollout_result.final_batch["env_reward"].tolist() == [1.0, 1.0, 0.0]
