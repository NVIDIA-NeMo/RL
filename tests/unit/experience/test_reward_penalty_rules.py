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
from __future__ import annotations

import pytest

from nemo_rl.experience.agent_step_metrics import AgentStepStats
from nemo_rl.experience.mask_sample_rules import (
    apply_mask_sample_rules,
    parse_mask_sample_rules,
)
from nemo_rl.experience.reward_penalty_rules import (
    RAW_REWARD_KEY,
    PenaltyOutcome,
    apply_reward_penalty_rules,
    is_env_masked,
    parse_reward_penalty_rules,
)

MASK_RULES = parse_mask_sample_rules(
    {
        "mask_sample_rules": [
            {
                "name": "context_limited",
                "field": "opencode_context_limited",
                "equals": True,
            },
            {"name": "agent_oom", "field": "opencode_exit_code", "equals": 137},
        ]
    }
)
MASK_RULE_NAMES = [rule.name for rule in MASK_RULES]
RULES = parse_reward_penalty_rules(
    {
        "reward_penalty_rules": [
            {
                "name": "capped_call",
                "field": "opencode_max_call_output_tokens",
                "at_least": 65000,
                "reward": -1.0,
                "unmask_rules": ["context_limited"],
            },
            {
                "name": "repeated_tool_call",
                "field": "opencode_max_tool_call_repeat",
                "at_least": 30,
                "reward": -0.5,
                "agents": ["swe_agent"],
            },
        ]
    },
    mask_rule_names=MASK_RULE_NAMES,
)


def _convert(full_result: dict, agent: str = "swe_agent") -> PenaltyOutcome:
    """The order the rollout manager uses: env flag, then mask rules, then penalties."""
    env_masked = is_env_masked(full_result)
    matched = apply_mask_sample_rules(full_result, MASK_RULES)
    return apply_reward_penalty_rules(
        full_result,
        RULES,
        agent_name=agent,
        env_masked=env_masked,
        matched_mask_rules=matched,
    )


def test_parse_absent_none_or_empty_means_no_rules():
    assert parse_reward_penalty_rules(None) == ()
    assert parse_reward_penalty_rules({}) == ()
    assert parse_reward_penalty_rules({"reward_penalty_rules": None}) == ()
    assert parse_reward_penalty_rules({"reward_penalty_rules": []}) == ()


def test_parse_reads_rules():
    capped, repeated = RULES
    assert capped.name == "capped_call" and capped.at_least == 65000.0
    assert capped.reward == -1.0 and capped.unmask_rules == {"context_limited"}
    assert capped.agents is None
    assert repeated.agents == {"swe_agent"} and repeated.unmask_rules == frozenset()


@pytest.mark.parametrize(
    "bad",
    [
        "not a list",
        [["not a mapping"]],
        [{"name": "x", "field": "f", "at_least": 1}],  # no reward
        [{"name": "x", "field": "f", "at_least": True, "reward": -1}],
        [{"name": "x", "field": " f", "at_least": 1, "reward": -1}],
        [{"name": "x", "field": "f", "at_least": 1, "reward": -1, "extra": 1}],
        [{"name": "x", "field": "f", "at_least": 1, "reward": -1, "agents": "a"}],
        # A typo in unmask_rules would leave every penalized rollout masked.
        [
            {
                "name": "x",
                "field": "f",
                "at_least": 1,
                "reward": -1,
                "unmask_rules": ["context_limitd"],
            }
        ],
        [
            {"name": "x", "field": "f", "at_least": 1, "reward": -1},
            {"name": "x", "field": "g", "at_least": 1, "reward": -1},
        ],
    ],
)
def test_parse_rejects_malformed_rules(bad):
    with pytest.raises(ValueError):
        parse_reward_penalty_rules(
            {"reward_penalty_rules": bad}, mask_rule_names=MASK_RULE_NAMES
        )


def test_capped_call_is_penalized_and_trained_instead_of_masked():
    result = {
        "reward": 0.0,
        "opencode_context_limited": True,
        "opencode_max_call_output_tokens": 65536,
    }
    assert _convert(result) == PenaltyOutcome(rule="capped_call", applied=True)
    assert result["reward"] == -1.0
    assert result[RAW_REWARD_KEY] == 0.0
    assert result["instance_config"]["mask_sample"] is False


def test_a_solved_rollout_with_a_repeated_tool_call_is_penalized():
    result = {"reward": 1.0, "opencode_max_tool_call_repeat": 31}
    assert _convert(result) == PenaltyOutcome(rule="repeated_tool_call", applied=True)
    assert result["reward"] == -0.5
    assert result[RAW_REWARD_KEY] == 1.0
    assert "instance_config" not in result  # never masked, nothing to unmask


def test_infrastructure_masks_keep_the_rollout_masked_and_its_reward():
    oom = {
        "reward": 0.0,
        "opencode_context_limited": True,
        "opencode_exit_code": 137,
        "opencode_max_call_output_tokens": 65536,
    }
    assert _convert(oom) == PenaltyOutcome(rule="capped_call", applied=False)
    assert oom["reward"] == 0.0 and RAW_REWARD_KEY not in oom
    assert oom["instance_config"]["mask_sample"] is True

    env_masked = {
        "reward": 0.0,
        "mask_sample": True,
        "opencode_max_call_output_tokens": 65536,
    }
    assert _convert(env_masked).applied is False
    assert env_masked["reward"] == 0.0


def test_no_match_below_threshold_for_other_agents_or_without_the_field():
    for result, agent in (
        ({"reward": 1.0, "opencode_max_call_output_tokens": 64999}, "swe_agent"),
        ({"reward": 1.0, "opencode_max_tool_call_repeat": 99}, "search_agent"),
        ({"reward": 1.0}, "swe_agent"),
        ({"reward": 1.0, "opencode_max_call_output_tokens": None}, "swe_agent"),
        ({"reward": 1.0, "opencode_max_tool_call_repeat": True}, "swe_agent"),
    ):
        assert _convert(result, agent) == PenaltyOutcome()
        assert result["reward"] == 1.0


def test_first_matching_rule_wins():
    result = {
        "reward": 1.0,
        "opencode_max_call_output_tokens": 70000,
        "opencode_max_tool_call_repeat": 40,
    }
    assert _convert(result).rule == "capped_call"
    assert result["reward"] == -1.0


def test_applying_twice_is_idempotent():
    # The rollout manager converts a streamed result once per row and again per group.
    result = {
        "reward": 0.0,
        "opencode_context_limited": True,
        "opencode_max_call_output_tokens": 65536,
    }
    first = _convert(result)
    second = _convert(result)
    assert first == second == PenaltyOutcome(rule="capped_call", applied=True)
    assert result["reward"] == -1.0 and result[RAW_REWARD_KEY] == 0.0
    assert result["instance_config"]["mask_sample"] is False


FAIL_RULES = parse_reward_penalty_rules(
    {
        "reward_penalty_rules": [
            {
                "name": "unfinished_fails",
                "masked_by": ["context_limited"],
                "reward": 0.0,
                "unmask_rules": ["context_limited"],
            }
        ]
    },
    mask_rule_names=MASK_RULE_NAMES,
)


def _convert_with(rules, full_result: dict, agent: str = "swe_agent") -> PenaltyOutcome:
    env_masked = is_env_masked(full_result)
    matched = apply_mask_sample_rules(full_result, MASK_RULES)
    return apply_reward_penalty_rules(
        full_result,
        rules,
        agent_name=agent,
        env_masked=env_masked,
        matched_mask_rules=matched,
    )


def test_parse_reads_a_masked_by_rule():
    (rule,) = FAIL_RULES
    assert rule.masked_by == {"context_limited"} and rule.unmask_rules == {
        "context_limited"
    }
    assert rule.field is None and rule.at_least is None and rule.reward == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        [{"name": "x", "masked_by": ["context_limited"], "field": "f", "reward": 0}],
        [{"name": "x", "masked_by": ["context_limited"], "at_least": 1, "reward": 0}],
        [{"name": "x", "masked_by": [], "reward": 0}],
        [{"name": "x", "masked_by": "context_limited", "reward": 0}],
        # A typo in masked_by would never match.
        [{"name": "x", "masked_by": ["context_limitd"], "reward": 0}],
    ],
)
def test_parse_rejects_malformed_masked_by_rules(bad):
    with pytest.raises(ValueError):
        parse_reward_penalty_rules(
            {"reward_penalty_rules": bad}, mask_rule_names=MASK_RULE_NAMES
        )


def test_a_context_limited_rollout_trains_as_a_failure_even_when_it_passed():
    result = {"reward": 1.0, "opencode_context_limited": True}
    assert _convert_with(FAIL_RULES, result) == PenaltyOutcome(
        rule="unfinished_fails", applied=True
    )
    assert result["reward"] == 0.0 and result[RAW_REWARD_KEY] == 1.0
    assert result["instance_config"]["mask_sample"] is False


def test_masked_by_leaves_infrastructure_masks_and_unmasked_rollouts_alone():
    # Also masked by another rule (an OOM): kept masked with its reward.
    oom = {"reward": 1.0, "opencode_context_limited": True, "opencode_exit_code": 137}
    assert _convert_with(FAIL_RULES, oom) == PenaltyOutcome(
        rule="unfinished_fails", applied=False
    )
    assert oom["reward"] == 1.0 and oom["instance_config"]["mask_sample"] is True
    # Masked by the environment itself: kept.
    env_masked = {"reward": 1.0, "mask_sample": True, "opencode_context_limited": True}
    assert _convert_with(FAIL_RULES, env_masked).applied is False
    assert env_masked["reward"] == 1.0
    # Masked only by a rule the penalty does not name, or not masked: no match.
    for result in ({"reward": 1.0, "opencode_exit_code": 137}, {"reward": 1.0}):
        assert _convert_with(FAIL_RULES, result) == PenaltyOutcome()
        assert result["reward"] == 1.0


def test_agent_step_stats_accept_a_rule_without_a_field():
    stats = AgentStepStats(penalty_rules=FAIL_RULES)
    result = {"reward": 1.0, "opencode_context_limited": True}
    stats.record("swe_agent", result, _convert_with(FAIL_RULES, result))
    metrics = stats.pop_metrics()
    assert metrics["swe_agent/unfinished_fails_frac"] == 1.0


def test_a_masked_by_rule_with_a_null_reward_unmasks_and_keeps_the_verifier_reward():
    rules = parse_reward_penalty_rules(
        {
            "reward_penalty_rules": [
                {
                    "name": "unfinished_trains",
                    "masked_by": ["context_limited"],
                    "reward": None,
                    "unmask_rules": ["context_limited"],
                }
            ]
        },
        mask_rule_names=MASK_RULE_NAMES,
    )
    assert rules[0].reward is None
    for reward in (1.0, 0.0):
        result = {"reward": reward, "opencode_context_limited": True}
        assert _convert_with(rules, result) == PenaltyOutcome(
            rule="unfinished_trains", applied=True
        )
        assert result["reward"] == reward and RAW_REWARD_KEY not in result
        assert result["instance_config"]["mask_sample"] is False


def test_context_limited_trains_with_its_reward_while_a_timeout_alone_stays_masked():
    mask_rules = parse_mask_sample_rules(
        {
            "mask_sample_rules": [
                {
                    "name": "agent_timeout",
                    "field": "opencode_error_type",
                    "equals": "timeout",
                },
                {
                    "name": "context_limited",
                    "field": "opencode_context_limited",
                    "equals": True,
                },
            ]
        }
    )
    rules = parse_reward_penalty_rules(
        {
            "reward_penalty_rules": [
                {
                    "name": "context_limited_trains",
                    "masked_by": ["context_limited"],
                    "reward": None,
                    "unmask_rules": ["context_limited", "agent_timeout"],
                }
            ]
        },
        mask_rule_names=[rule.name for rule in mask_rules],
    )

    def convert(result):
        env_masked = is_env_masked(result)
        matched = apply_mask_sample_rules(result, mask_rules)
        return apply_reward_penalty_rules(
            result,
            rules,
            agent_name="swe_agent",
            env_masked=env_masked,
            matched_mask_rules=matched,
        )

    both = {
        "reward": 1.0,
        "opencode_error_type": "timeout",
        "opencode_context_limited": True,
    }
    assert convert(both).applied is True
    assert both["reward"] == 1.0 and both["instance_config"]["mask_sample"] is False
    timeout_only = {"reward": 1.0, "opencode_error_type": "timeout"}
    assert convert(timeout_only) == PenaltyOutcome()
    assert (
        timeout_only["reward"] == 1.0
        and timeout_only["instance_config"]["mask_sample"] is True
    )
