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
PENALTY_RULES = parse_reward_penalty_rules(
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
    mask_rule_names=[rule.name for rule in MASK_RULES],
)


def _convert(full_result: dict, agent: str, penalty_rules=()) -> PenaltyOutcome:
    """The order the rollout manager uses: env flag, then mask rules, then penalties."""
    env_masked = is_env_masked(full_result)
    matched = apply_mask_sample_rules(full_result, MASK_RULES)
    return apply_reward_penalty_rules(
        full_result,
        penalty_rules,
        agent_name=agent,
        env_masked=env_masked,
        matched_mask_rules=matched,
    )


def test_per_agent_metrics_are_reported_without_penalty_rules():
    stats = AgentStepStats()
    rollouts = [
        (
            "swe_agent",
            {
                "reward": 1.0,
                "opencode_max_call_output_tokens": 1000,
                "opencode_max_tool_call_repeat": 3,
                "opencode_max_tool_call_streak": 1,
            },
        ),
        (
            "swe_agent",
            {
                "reward": 0.0,
                "opencode_context_limited": True,
                "opencode_max_call_output_tokens": 90000,
                "opencode_max_tool_call_repeat": 7,
                "opencode_max_tool_call_streak": 2,
            },
        ),
        ("swe_agent", {"reward": 0.0, "opencode_max_call_output_tokens": 2000}),
        ("search_agent", {"reward": 0.5}),
        ("search_agent", {"reward": 1.0, "instance_config": {"mask_sample": True}}),
    ]
    for agent, result in rollouts:
        stats.record(agent, result, _convert(result, agent))
    metrics = stats.pop_metrics()

    assert metrics["swe_agent/rollouts"] == 3.0
    assert metrics["swe_agent/reward_mean"] == pytest.approx(1 / 3)
    # The context-limited rollout is masked by its mask rule.
    assert metrics["swe_agent/env_masked_frac"] == pytest.approx(1 / 3)
    assert metrics["swe_agent/unmasked_reward_mean"] == pytest.approx(0.5)
    assert metrics["swe_agent/opencode_max_call_output_tokens_max"] == 90000.0
    assert metrics["swe_agent/opencode_max_call_output_tokens_mean"] == pytest.approx(
        93000 / 3
    )
    assert metrics["swe_agent/opencode_max_tool_call_repeat_max"] == 7.0
    assert metrics["swe_agent/opencode_max_tool_call_streak_mean"] == 1.5

    assert metrics["search_agent/rollouts"] == 2.0
    assert metrics["search_agent/reward_mean"] == pytest.approx(0.75)
    assert metrics["search_agent/env_masked_frac"] == 0.5
    assert metrics["search_agent/unmasked_reward_mean"] == 0.5
    # Fields an agent's results do not carry, and rule metrics without rules, are absent.
    assert not any(key.startswith("search_agent/opencode_") for key in metrics)
    assert not any(
        key.endswith(("_frac", "raw_reward_mean"))
        and not key.endswith("env_masked_frac")
        for key in metrics
    )
    assert not any(key.startswith("penalty_rules/") for key in metrics)
    assert stats.pop_metrics() == {}


def test_all_masked_agent_has_no_unmasked_reward():
    stats = AgentStepStats()
    result = {"reward": 0.0, "instance_config": {"mask_sample": True}}
    stats.record("swe_agent", result)
    metrics = stats.pop_metrics()
    assert metrics["swe_agent/env_masked_frac"] == 1.0
    assert "swe_agent/unmasked_reward_mean" not in metrics


def test_penalty_rules_add_rule_rates_and_the_raw_reward():
    stats = AgentStepStats(PENALTY_RULES)
    rollouts = [
        (
            "swe_agent",
            {
                "reward": 0.0,
                "opencode_context_limited": True,
                "opencode_max_call_output_tokens": 65536,
            },
        ),
        (
            "swe_agent",
            {
                "reward": 1.0,
                "opencode_max_call_output_tokens": 1000,
                "opencode_max_tool_call_repeat": 2,
            },
        ),
        (
            "swe_agent",
            {
                "reward": 0.0,
                "opencode_exit_code": 137,
                "opencode_context_limited": True,
                "opencode_max_call_output_tokens": 65536,
            },
        ),
        ("search_agent", {"reward": 1.0}),
    ]
    for agent, result in rollouts:
        stats.record(agent, result, _convert(result, agent, PENALTY_RULES))
    metrics = stats.pop_metrics()
    assert metrics["swe_agent/rollouts"] == 3.0
    assert metrics["swe_agent/capped_call_frac"] == pytest.approx(1 / 3)
    assert metrics["swe_agent/capped_call_kept_masked_frac"] == pytest.approx(1 / 3)
    assert metrics["swe_agent/repeated_tool_call_frac"] == 0.0
    assert metrics["swe_agent/reward_mean"] == pytest.approx((-1.0 + 1.0 + 0.0) / 3)
    assert metrics["swe_agent/raw_reward_mean"] == pytest.approx(1 / 3)
    # The penalized rollout is trained; the OOM one keeps its mask.
    assert metrics["swe_agent/env_masked_frac"] == pytest.approx(1 / 3)
    assert metrics["swe_agent/opencode_max_call_output_tokens_max"] == 65536.0
    assert metrics["swe_agent/opencode_max_tool_call_repeat_mean"] == 2.0
    assert metrics["search_agent/rollouts"] == 1.0
    assert metrics["search_agent/capped_call_frac"] == 0.0
    # The repeated-tool-call rule is scoped to swe_agent.
    assert "search_agent/repeated_tool_call_frac" not in metrics
    assert stats.pop_metrics() == {}
