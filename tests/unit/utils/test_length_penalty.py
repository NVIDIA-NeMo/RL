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
from types import SimpleNamespace

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


class TestProfiledLengthZeroOverMax:
    """profiled_length_penalty_type: zero_over_max (Kimi-style cap): a rollout
    strictly longer than the longest PASSING profiled rollout for its prompt
    gets exactly 0.0 reward.

    Rollouts are built with all their length in reasoning and an empty answer,
    so each ``lengths`` entry is the total length the cap compares against.
    Rewards are compared with ``==``, not ``approx``: a capped reward must be
    exactly zero, and an uncapped one exactly what the env returned.
    """

    @staticmethod
    def make_group(rewards, lengths, p_rewards, p_lengths):
        results = [
            make_result("x" * length, "", reward)
            for reward, length in zip(rewards, lengths)
        ]
        for r in results:
            r["profiled_rewards"] = p_rewards
            r["profiled_output_lengths"] = p_lengths
        return results

    @staticmethod
    def cfg(num_gens, min_samples=2, **params):
        return make_config(
            default={
                "enabled": True,
                "profiled_length_penalty_type": "zero_over_max",
                "profiled_length_min_samples": min_samples,
                **params,
            },
            num_gens=num_gens,
        )

    @pytest.mark.parametrize(
        "rewards, lengths, p_rewards, p_lengths, min_samples, expected",
        [
            pytest.param(
                [1.0, 1.0, 1.0, 1.0],
                [5, 10, 11, 40],
                [1, 1, 1],
                [4, 8, 10],
                2,
                [1.0, 1.0, 0.0, 0.0],
                id="over_max_zeroed_at_max_kept",
            ),
            pytest.param(
                [1.0, 1.0, 1.0],
                [12, 13, 50],
                [1, 1, 0],
                [10, 12, 100],
                2,
                # A failing profiled length (100) must not raise the cap.
                [1.0, 0.0, 0.0],
                id="failing_profiled_lengths_ignored",
            ),
            pytest.param(
                [1.0, 1.0],
                [20, 30],
                [1, 1],
                [10, 10],
                2,
                [0.0, 0.0],
                id="whole_group_over_max",
            ),
            pytest.param(
                [0.0, 1.0, 0.0, 1.0],
                [50, 50, 5, 5],
                [1, 1],
                [10, 10],
                2,
                [0.0, 0.0, 0.0, 1.0],
                id="wrong_rollouts_stay_zero",
            ),
            pytest.param(
                [0.0, 0.0],
                [50, 5],
                [1, 1],
                [10, 10],
                2,
                [0.0, 0.0],
                id="all_wrong_group_unchanged",
            ),
            pytest.param(
                [1.0, 1.0],
                [50, 5],
                [1, 0, 0],
                [10, 100, 100],
                2,
                [1.0, 1.0],
                id="fewer_passes_than_min_samples_no_cap",
            ),
            pytest.param(
                [1.0, 1.0],
                [50, 5],
                [0, 0],
                [10, 10],
                0,
                # min_samples=0 must not reach max([]) on an all-failed profile.
                [1.0, 1.0],
                id="no_passes_no_cap",
            ),
            pytest.param(
                [1.0, 1.0],
                [10, 11],
                [1, 0],
                [10, 100],
                1,
                [1.0, 0.0],
                id="single_pass_allowed_when_min_samples_one",
            ),
        ],
    )
    def test_rewards_in_rewards_out(
        self, rewards, lengths, p_rewards, p_lengths, min_samples, expected
    ):
        results = self.make_group(rewards, lengths, p_rewards, p_lengths)
        apply(results, self.cfg(len(rewards), min_samples=min_samples))
        assert rewards_of(results) == expected

    @pytest.mark.parametrize(
        "correct_threshold, expected",
        [
            # Correct profiled lengths -> cap, for rollouts of length 12 and 13:
            pytest.param(None, [1.0, 1.0], id="default_0"),  # all 4 -> 100
            pytest.param(0.4, [1.0, 1.0], id="0.4_counts_0.5"),  # 90, 10, 12 -> 90
            pytest.param(0.5, [1.0, 0.0], id="0.5_excludes_0.5"),  # 10, 12 -> 12
            pytest.param(0.9, [1.0, 1.0], id="too_few_correct"),  # 12 < min 2
        ],
    )
    def test_reward_correct_threshold(self, correct_threshold, expected):
        # Graded profiled rewards: only r > threshold counts as correct, so
        # the threshold picks the lengths that set the cap and the count that
        # min_samples gates on.
        results = self.make_group(
            [1.0, 1.0], [12, 13], [0.3, 0.5, 0.8, 1.0], [100, 90, 10, 12]
        )
        params = (
            {}
            if correct_threshold is None
            else {"profiled_reward_correct_threshold": correct_threshold}
        )
        apply(results, self.cfg(num_gens=2, **params))
        assert rewards_of(results) == expected

    def test_reward_correct_threshold_applies_to_constant_type(self):
        # With the default 0, the 0.3-reward length 100 counts and drags the
        # mean + std threshold (~92) past the 50-long rollout. At 0.5 only the
        # two length-10 profiles count: threshold 10, so the rollout loses 0.3.
        p_rewards, p_lengths = [0.3, 1.0, 1.0], [100, 10, 10]
        outs = []
        for correct_threshold in (0.0, 0.5):
            results = self.make_group([1.0, 1.0], [5, 50], p_rewards, p_lengths)
            cfg = make_config(
                default={
                    "profiled_length_penalty": 0.3,
                    "profiled_reward_correct_threshold": correct_threshold,
                }
            )
            apply(results, cfg)
            outs.append(rewards_of(results))
        assert outs[0] == pytest.approx([1.0, 1.0])
        assert outs[1] == pytest.approx([1.0, 0.7])

    def test_each_group_uses_its_own_profile(self):
        # Same lengths, different profiles: group 0 caps at 10, group 1 at 30.
        results = self.make_group([1.0, 1.0], [20, 20], [1, 1], [10, 10])
        results += self.make_group([1.0, 1.0], [20, 20], [1, 1], [30, 30])
        apply(results, self.cfg(num_gens=2))
        assert rewards_of(results) == [0.0, 0.0, 1.0, 1.0]

    def test_zero_survives_bonus(self):
        # total_bonus lifts the shortest correct rollout (length 20) to 1.5,
        # but it is over the cap of 10: the cap runs last, so it ends at 0.0.
        results = self.make_group([1.0, 1.0], [20, 30], [1, 1], [10, 10])
        apply(results, self.cfg(num_gens=2, total_bonus=0.5))
        assert rewards_of(results) == [0.0, 0.0]

    def test_zero_survives_group_relative_penalty(self):
        # The longer rollout (50 > cap 10) also takes the group-relative -0.05.
        # The no-double-penalty rule drops a constant_over_n_std penalty on
        # such a rollout, but it must not drop the cap: 0.0, not 0.95.
        results = self.make_group([1.0, 1.0], [5, 50], [1, 1], [10, 10])
        apply(results, self.cfg(num_gens=2, group_total_length_penalty_coeff=0.1))
        under_cap, over_cap = rewards_of(results)
        assert under_cap == pytest.approx(1.05)
        assert over_cap == 0.0

    def test_zero_survives_profile_band_multiplier(self):
        band = {"total": {"a": 1, "b": 100, "f": 0.5}}
        results = self.make_group([1.0, 1.0], [5, 20], [1, 1], [10, 10])
        for r in results:
            r["profile_band"] = band
        apply(results, self.cfg(num_gens=2, profile_band_total=True))
        under_cap, over_cap = rewards_of(results)
        assert under_cap == pytest.approx(1.0 - (5 - 1) / 99 * 0.5)
        assert over_cap == 0.0

    def test_type_picks_algorithm_on_same_profile(self):
        # Passing lengths 8 and 12 (failing 100 ignored), n_std=0. The same
        # rollouts under each type:
        #   constant_over_n_std: threshold = mean = 10, length >= 10 loses 0.3
        #   zero_over_max:       cap = max = 12, length > 12 is zeroed
        rewards, lengths = [1.0, 1.0, 1.0, 1.0], [5, 10, 12, 13]
        p_rewards, p_lengths = [1, 1, 0], [8, 12, 100]
        constant = self.make_group(rewards, lengths, p_rewards, p_lengths)
        apply(
            constant,
            make_config(
                default={"profiled_length_penalty": 0.3, "profiled_length_n_std": 0.0},
                num_gens=4,
            ),
        )
        assert rewards_of(constant) == pytest.approx([1.0, 0.7, 0.7, 0.7])
        zeroed = self.make_group(rewards, lengths, p_rewards, p_lengths)
        apply(zeroed, self.cfg(num_gens=4))
        assert rewards_of(zeroed) == [1.0, 1.0, 1.0, 0.0]

    def test_constant_with_zero_over_max_rejected(self):
        # zero_over_max ignores the constant, so setting both is a config error
        # rather than a penalty that silently does nothing.
        with pytest.raises(ValueError, match="profiled_length_penalty"):
            LengthPenaltyConfig.model_validate(
                {
                    "default": {
                        "profiled_length_penalty_type": "zero_over_max",
                        "profiled_length_penalty": 0.3,
                    }
                }
            )

    def test_counts_tokens_when_length_type_is_tokens(self):
        # Profiled lengths are token counts. "a b c" is 3 tokens but 5 chars:
        # counting chars would wrongly cap it against a 3-token profile.
        tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens: text.split()
        )
        results = [
            make_result("a b c", "", 1.0),
            make_result("a b c d", "", 1.0),
        ]
        for r in results:
            r["profiled_rewards"] = [1, 1]
            r["profiled_output_lengths"] = [3, 3]
        apply(results, self.cfg(num_gens=2, length_type="tokens"), tokenizer)
        assert rewards_of(results) == [1.0, 0.0]

    def test_off_by_default(self):
        # The default type is constant_over_n_std, which stays off at the
        # default profiled_length_penalty of 0.
        results = self.make_group([1.0, 1.0], [5, 50], [1, 1], [10, 10])
        apply(results, make_config(default={"enabled": True}))
        assert rewards_of(results) == [1.0, 1.0]

    @pytest.mark.parametrize(
        "penalty",
        [
            pytest.param({"profiled_length_penalty_type": "zero_over_max"}, id="zero"),
            pytest.param({"profiled_length_penalty": 0.3}, id="constant"),
        ],
    )
    @pytest.mark.parametrize(
        "p_rewards, p_lengths",
        [
            # rollouts.py copies a field the gym row lacks as None.
            pytest.param(None, None, id="both_missing"),
            pytest.param([1, 1], None, id="lengths_missing"),
            pytest.param(None, [10, 10], id="rewards_missing"),
            pytest.param([1, 1], "10,10", id="lengths_not_a_list"),
            pytest.param([1, 1], {"p50": 10}, id="lengths_a_stats_dict"),
            pytest.param([1, 1, 0], [10, 10], id="length_mismatch"),
        ],
    )
    def test_bad_profile_fields_raise(self, penalty, p_rewards, p_lengths):
        # A forgotten, misspelled, or malformed profile must fail the rollout,
        # not silently train without the penalty.
        results = self.make_group([1.0, 1.0], [5, 50], p_rewards, p_lengths)
        cfg = make_config(default={"enabled": True, **penalty}, num_gens=2)
        with pytest.raises(ValueError, match=f"{AGENT}.*profiled_output_lengths"):
            apply(results, cfg)

    def test_rows_without_profile_ok_when_agent_turns_it_off(self):
        # Mixed data: the profiled agent's group is capped, and another agent
        # whose rows carry no profile turns the penalty off in its override.
        profiled = self.make_group([1.0, 1.0], [5, 50], [1, 1], [10, 10])
        unprofiled = self.make_group([1.0, 1.0], [5, 50], None, None)
        for r in unprofiled:
            r["agent_ref"] = {"name": "code_agent"}
        cfg = self.cfg(num_gens=2)
        cfg["grpo"]["length_penalty"]["agent_overrides"] = {
            AGENT: {},
            "code_agent": {"profiled_length_penalty_type": "constant_over_n_std"},
        }
        apply(profiled + unprofiled, cfg)
        assert rewards_of(profiled + unprofiled) == [1.0, 0.0, 1.0, 1.0]

    def test_rows_without_profile_ok_when_penalty_off(self):
        # Only the group-relative penalty is on: no profile is needed.
        results = self.make_group([1.0, 1.0], [5, 25], None, None)
        cfg = make_config(
            default={"enabled": True, "group_total_length_penalty_coeff": 0.1}
        )
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 0.95])

    def test_agent_override_switches_type(self):
        # Default zeroes over the max (10); this agent's override switches it
        # to constant_over_n_std, so the over-length rollout loses 0.3 instead.
        cfg = self.cfg(num_gens=2)
        cfg["grpo"]["length_penalty"]["agent_overrides"] = {
            AGENT: {
                "profiled_length_penalty_type": "constant_over_n_std",
                "profiled_length_penalty": 0.3,
            }
        }
        results = self.make_group([1.0, 1.0], [5, 50], [1, 1], [10, 10])
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.0, 0.7])

    @pytest.mark.parametrize(
        "rewards",
        [
            pytest.param([-1.0, 1.0], id="negative"),
            pytest.param([0.5, 1.0], id="graded"),
        ],
    )
    def test_non_binary_group_untouched_and_warned(self, rewards, caplog, monkeypatch):
        # Default (profiled_length_allow_non_binary_rewards off): the binary
        # check skips the whole group — both rows are over the cap of 10 and
        # both keep their env reward — and warns.
        monkeypatch.setattr(length_penalty_mod, "_NON_BINARY_WARNED_AGENTS", set())
        results = self.make_group(rewards, [50, 50], [1, 1], [10, 10])
        with caplog.at_level(logging.WARNING, logger=length_penalty_mod.__name__):
            metrics = apply(results, self.cfg(num_gens=2))
        assert rewards_of(results) == rewards
        assert "non-binary" in caplog.text
        assert AGENT in caplog.text
        assert metrics["length_penalty/skipped_non_binary_frac"] == 1.0

    @pytest.mark.parametrize(
        "rewards, lengths, expected",
        [
            pytest.param(
                [0.5, 0.7, 0.3, 2.0],
                [5, 50, 50, 5],
                [0.5, 0.0, 0.0, 2.0],
                id="graded_over_cap_zeroed",
            ),
            pytest.param(
                [0.9, 0.1],
                [20, 30],
                [0.0, 0.0],
                id="graded_all_over_cap",
            ),
            pytest.param(
                [1.0, 0.0, 0.25, 1.0],
                [50, 50, 5, 5],
                [0.0, 0.0, 0.25, 1.0],
                id="binary_and_graded_mixed",
            ),
            pytest.param(
                [0.0, 0.6],
                [50, 50],
                [0.0, 0.0],
                id="zero_stays_zero",
            ),
        ],
    )
    def test_non_binary_rewards_capped_when_allowed(self, rewards, lengths, expected):
        # With the option on, every non-negative scalar reward over the cap
        # (10) becomes exactly 0; under the cap it is untouched.
        results = self.make_group(rewards, lengths, [1, 1], [10, 10])
        cfg = self.cfg(
            num_gens=len(rewards), profiled_length_allow_non_binary_rewards=True
        )
        metrics = apply(results, cfg)
        assert rewards_of(results) == expected
        assert metrics["length_penalty/skipped_non_binary_frac"] == 0.0

    @pytest.mark.parametrize(
        "rewards, lengths",
        [
            pytest.param([-1.0, 1.0], [50, 50], id="negative_over_cap"),
            # Raised even when no rollout is over the cap: whether a run fails
            # must not depend on how long the rollouts happen to be.
            pytest.param([-0.5, 0.5], [5, 5], id="negative_under_cap"),
        ],
    )
    def test_negative_reward_raises_when_allowed(self, rewards, lengths):
        results = self.make_group(rewards, lengths, [1, 1], [10, 10])
        cfg = self.cfg(num_gens=2, profiled_length_allow_non_binary_rewards=True)
        with pytest.raises(ValueError, match=f"{AGENT}.*negative reward"):
            apply(results, cfg)

    @pytest.mark.parametrize("allow_non_binary", [False, True])
    def test_reward_components_raise(self, allow_non_binary):
        # Zeroing the scalar alone would break reward == sum(reward_components),
        # so zero_over_max refuses multi-component rewards outright.
        results = self.make_group([1.0, 1.0], [5, 50], [1, 1], [10, 10])
        for r in results:
            r["full_result"]["reward_components"] = {"correct": 1.0, "format": 0.0}
        cfg = self.cfg(
            num_gens=2, profiled_length_allow_non_binary_rewards=allow_non_binary
        )
        with pytest.raises(ValueError, match=f"{AGENT}.*reward_components"):
            apply(results, cfg)

    def test_non_binary_group_gets_only_the_cap(self):
        # Group 0 is binary: group-relative +/-0.05, then the cap zeroes the
        # over-cap rollout. Group 1 is graded: the group-relative penalty is
        # binary-only, so it gets the cap alone and its under-cap rollout keeps
        # exactly 0.5.
        results = self.make_group([1.0, 1.0], [5, 50], [1, 1], [10, 10])
        results += self.make_group([0.5, 0.5], [5, 50], [1, 1], [10, 10])
        cfg = self.cfg(
            num_gens=2,
            profiled_length_allow_non_binary_rewards=True,
            group_total_length_penalty_coeff=0.1,
        )
        apply(results, cfg)
        assert rewards_of(results) == pytest.approx([1.05, 0.0, 0.5, 0.0])
        assert rewards_of(results)[1:] == [0.0, 0.5, 0.0]

    def test_low_effort_group_skipped_when_allowed(self):
        # Effort shaping already rewrote these rewards (even below 0); the
        # length penalties leave low-effort groups alone, without raising.
        results = self.make_group([-0.2, 0.5], [50, 50], [1, 1], [10, 10])
        for r in results:
            r["low_effort_applied"] = True
        cfg = self.cfg(num_gens=2, profiled_length_allow_non_binary_rewards=True)
        apply(results, cfg)
        assert rewards_of(results) == [-0.2, 0.5]

    def test_metrics_count_capped_non_binary_rollouts(self):
        results = self.make_group([0.5, 0.5], [5, 50], [1, 1], [10, 10])
        cfg = self.cfg(num_gens=2, profiled_length_allow_non_binary_rewards=True)
        metrics = apply(results, cfg)
        assert rewards_of(results) == [0.5, 0.0]
        assert metrics["length_penalty/wiped_correct_frac"] == 0.5
        assert metrics["length_penalty/adjusted_frac"] == 0.5
        assert metrics["length_penalty/reward_delta_mean"] == -0.25

    def test_allow_non_binary_requires_zero_over_max(self):
        # Every other length penalty needs binary rewards, so the option would
        # silently do nothing next to constant_over_n_std.
        with pytest.raises(ValueError, match="zero_over_max"):
            LengthPenaltyConfig.model_validate(
                {
                    "default": {
                        "profiled_length_penalty": 0.3,
                        "profiled_length_allow_non_binary_rewards": True,
                    }
                }
            )

    @pytest.mark.parametrize(
        "default, override, match",
        [
            pytest.param(
                {"profiled_length_penalty_type": "zero_over_max"},
                {"profiled_length_penalty": 0.2},
                "profiled_length_penalty=0.2",
                id="override_constant_under_default_zero_over_max",
            ),
            pytest.param(
                {"profiled_length_penalty": 0.2},
                {"profiled_length_penalty_type": "zero_over_max"},
                "profiled_length_penalty=0.2",
                id="override_zero_over_max_under_default_constant",
            ),
            pytest.param(
                {
                    "profiled_length_penalty_type": "zero_over_max",
                    "profiled_length_allow_non_binary_rewards": True,
                },
                {
                    "profiled_length_penalty_type": "constant_over_n_std",
                    "profiled_length_penalty": 0.2,
                },
                "profiled_length_allow_non_binary_rewards",
                id="override_constant_inherits_allow_non_binary",
            ),
        ],
    )
    def test_profiled_settings_rejected_when_merged(self, default, override, match):
        # Each block is fine alone; the agent runs on default merged with its
        # override, and that is not.
        with pytest.raises(ValueError, match=f"{AGENT}.*{match}"):
            LengthPenaltyConfig.model_validate(
                {"default": default, "agent_overrides": {AGENT: override}}
            )

    @pytest.mark.parametrize(
        "default, override",
        [
            pytest.param(
                {"profiled_length_penalty_type": "zero_over_max"},
                {"profiled_length_allow_non_binary_rewards": True},
                id="override_allows_non_binary_under_default_zero_over_max",
            ),
            pytest.param(
                {"profiled_length_penalty": 0.2},
                {
                    "profiled_length_penalty_type": "zero_over_max",
                    "profiled_length_penalty": 0.0,
                },
                id="override_switches_to_zero_over_max_and_clears_constant",
            ),
            pytest.param(
                {
                    "profiled_length_penalty_type": "zero_over_max",
                    "profiled_length_allow_non_binary_rewards": True,
                },
                {
                    "profiled_length_penalty_type": "constant_over_n_std",
                    "profiled_length_allow_non_binary_rewards": False,
                },
                id="documented_opt_out",
            ),
            pytest.param(
                {"profiled_length_penalty_type": "zero_over_max"},
                {"enabled": False, "profiled_length_penalty": 0.2},
                id="disabled_agent",
            ),
        ],
    )
    def test_profiled_settings_accepted_when_merged(self, default, override):
        # Checked on the merged settings the agent runs with, whatever each
        # block looks like alone.
        LengthPenaltyConfig.model_validate(
            {"default": default, "agent_overrides": {AGENT: override}}
        )

    def test_override_allows_non_binary_under_default_zero_over_max(self):
        # The type comes from default and the option from the override: the
        # agent's graded group is capped instead of skipped.
        results = self.make_group([0.5, 0.5], [5, 50], [1, 1], [10, 10])
        cfg = self.cfg(num_gens=2)
        cfg["grpo"]["length_penalty"]["agent_overrides"] = {
            AGENT: {"profiled_length_allow_non_binary_rewards": True}
        }
        apply(results, cfg)
        assert rewards_of(results) == [0.5, 0.0]

    def test_metrics_count_zeroed_rollouts(self):
        results = self.make_group(
            [1.0, 1.0, 0.0, 0.0], [5, 50, 50, 5], [1, 1], [10, 10]
        )
        metrics = apply(results, self.cfg(num_gens=4))
        assert rewards_of(results) == [1.0, 0.0, 0.0, 0.0]
        assert metrics["length_penalty/env_reward_mean"] == 0.5
        assert metrics["length_penalty/reward_delta_mean"] == -0.25
        assert metrics["length_penalty/wiped_correct_frac"] == 0.25
        assert metrics["length_penalty/adjusted_frac"] == 0.25


class TestAllGeneratedTokens:
    """length_type: all_generated_tokens counts every token id in the
    assistant messages of message_log instead of re-tokenizing text."""

    @staticmethod
    def make_rollout(reward, generated_per_turn):
        # 2000 chars of text: were the text measured, every rollout would be
        # far over any cap below. Each turn also has 50 prompt/observation
        # tokens in a user message, which must not count.
        result = make_result("x" * 1000, "y" * 1000, reward)
        result["message_log"] = []
        for n_generated in generated_per_turn:
            result["message_log"].append({"role": "user", "token_ids": [0] * 50})
            result["message_log"].append(
                {"role": "assistant", "token_ids": [1] * n_generated}
            )
        result["profiled_rewards"] = [1, 1]
        result["profiled_output_lengths"] = [10, 10]
        return result

    @staticmethod
    def cfg(num_gens, **params):
        return make_config(
            default={"length_type": "all_generated_tokens", **params},
            num_gens=num_gens,
        )

    def test_cap_counts_generated_tokens_of_all_turns(self):
        # Cap 10. Generated tokens: 10, 4 + 6 = 10, 4 + 7 = 11, 12.
        results = [
            self.make_rollout(1.0, turns) for turns in ([10], [4, 6], [4, 7], [12])
        ]
        apply(results, self.cfg(4, profiled_length_penalty_type="zero_over_max"))
        assert rewards_of(results) == [1.0, 1.0, 0.0, 0.0]

    def test_total_length_settings_use_generated_tokens(self):
        # 5 vs 15 generated tokens: the group-relative total channel gives the
        # same +/-0.05 it gives any two-length group.
        results = [self.make_rollout(1.0, [5]), self.make_rollout(1.0, [15])]
        apply(results, self.cfg(2, group_total_length_penalty_coeff=0.1))
        assert rewards_of(results) == pytest.approx([1.05, 0.95])

    @pytest.mark.parametrize(
        "setting",
        [
            {"reasoning_bonus": 0.1},
            {"answer_bonus": 0.1},
            {"longest_reasoning_penalty": 0.1},
            {"longest_answer_penalty": 0.1},
            {"group_reasoning_length_penalty_coeff": 0.1},
            {"group_answer_length_penalty_coeff": 0.1},
            {"reasoning_zmad_threshold": 2.0, "reasoning_zmad_penalty": 0.1},
            {"answer_zmad_threshold": 2.0, "answer_zmad_penalty": 0.1},
            {"profile_band_reasoning": True},
            {"profile_band_answer": True},
            {
                "group_length_penalty_profile_gate": True,
                "group_length_penalty_profile_gate_channel": "reasoning",
            },
        ],
        ids=lambda setting: next(iter(setting)),
    )
    def test_reasoning_or_answer_setting_rejected(self, setting):
        # Generated token ids have no reasoning/answer split: these settings
        # would silently see length 0.
        with pytest.raises(ValueError, match=next(iter(setting))):
            LengthPenaltyConfig.model_validate(
                {"default": {"length_type": "all_generated_tokens", **setting}}
            )

    @pytest.mark.parametrize(
        "default, override",
        [
            pytest.param(
                {"length_type": "all_generated_tokens"},
                {"group_reasoning_length_penalty_coeff": 0.1},
                id="override_adds_reasoning_setting",
            ),
            pytest.param(
                {"group_reasoning_length_penalty_coeff": 0.1},
                {"length_type": "all_generated_tokens"},
                id="override_switches_length_type",
            ),
        ],
    )
    def test_rejected_on_merged_agent_settings(self, default, override):
        # Each block is fine alone; the agent's merged settings are not.
        with pytest.raises(ValueError, match=AGENT):
            LengthPenaltyConfig.model_validate(
                {"default": default, "agent_overrides": {AGENT: override}}
            )

    def test_global_reasoning_band_rejected_unless_turned_off(self):
        band = {"enabled": True, "defaults": {"reasoning": {"a": 1, "b": 2, "f": 0.5}}}
        with pytest.raises(ValueError, match="profile_band_reasoning"):
            LengthPenaltyConfig.model_validate(
                {
                    "default": {"length_type": "all_generated_tokens"},
                    "profile_band": band,
                }
            )
        LengthPenaltyConfig.model_validate(
            {
                "default": {
                    "length_type": "all_generated_tokens",
                    "profile_band_reasoning": False,
                },
                "profile_band": band,
            }
        )

    def test_total_only_and_disabled_agents_accepted(self):
        LengthPenaltyConfig.model_validate(
            {
                "default": {
                    "length_type": "all_generated_tokens",
                    "profiled_length_penalty_type": "zero_over_max",
                    "group_total_length_penalty_coeff": 0.1,
                    "total_zmad_threshold": 2.0,
                    "total_zmad_penalty": 0.1,
                    "profile_band_total": True,
                },
                "agent_overrides": {
                    "code_agent": {
                        "enabled": False,
                        "group_reasoning_length_penalty_coeff": 0.1,
                    }
                },
            }
        )


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
            {"profiled_length_penalty_type": "zero_over_mean"},
            {"length_type": "generated_tokens"},
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

    def test_profiled_zero_over_max_reaches_total_reward(self):
        # The profiled fields are copied from the gym rows onto the results,
        # and the capped rollout (total 11 > max 10) trains on exactly 0.
        rows = [
            {
                "agent_ref": {"name": AGENT},
                "profiled_rewards": [1, 1],
                "profiled_output_lengths": [10, 10],
            }
            for _ in range(2)
        ]
        grpo_config = make_config(
            default={"profiled_length_penalty_type": "zero_over_max"}
        )["grpo"]
        texts = [("12345", "12345"), ("123456", "12345")]
        rollout_result = self.postprocess(rows, texts, grpo_config)
        assert rollout_result.final_batch["total_reward"].tolist() == [1.0, 0.0]
        assert rollout_result.final_batch["env_reward"].tolist() == [1.0, 1.0]

    def test_all_generated_tokens_reads_gym_message_log(self):
        # Each rollout generated 1 token (the tensor-valued assistant message
        # the Gym path builds) but has 10 chars of text. Group 0's cap is 0, so
        # it is zeroed; group 1's cap is 1, so it is kept, which measuring the
        # text would not do.
        rows = [
            {
                "agent_ref": {"name": AGENT},
                "profiled_rewards": [1, 1],
                "profiled_output_lengths": [cap, cap],
            }
            for cap in (0, 1)
            for _ in range(2)
        ]
        grpo_config = make_config(
            default={
                "length_type": "all_generated_tokens",
                "profiled_length_penalty_type": "zero_over_max",
            }
        )["grpo"]
        texts = [("12345", "12345")] * 4
        rollout_result = self.postprocess(rows, texts, grpo_config)
        rewards = rollout_result.final_batch["total_reward"].tolist()
        assert rewards == [0.0, 0.0, 1.0, 1.0]

    def test_gym_row_without_profile_raises(self):
        # The user's mistake end to end: profiled penalty on, but the gym rows
        # nest the profile under extra_env_info instead of the top level.
        rows = [
            {
                "agent_ref": {"name": AGENT},
                "extra_env_info": {
                    "profiled_rewards": [1, 1],
                    "profiled_output_lengths": [10, 10],
                },
            }
            for _ in range(2)
        ]
        grpo_config = make_config(
            default={"profiled_length_penalty_type": "zero_over_max"}
        )["grpo"]
        texts = [("12345", "12345"), ("123456", "12345")]
        with pytest.raises(ValueError, match="profiled_rewards=None"):
            self.postprocess(rows, texts, grpo_config)

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
