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

"""Capture reward parity with the standard path, without token decoding."""

from itertools import product

import pytest

from nemo_rl.algorithms.grpo import RewardPenaltyConfig
from nemo_rl.experience.reward_penalties import (
    aggregate_capture_reward_metrics,
    compute_reward_checks,
    finalize_capture_reward,
    has_duplicated_reasoning,
    has_empty_final_answer,
)
from nemo_rl.experience.rollouts import (
    EffortLevelsConfig,
    _apply_effort_shaping,
    apply_reward_penalties,
    resolve_reward_penalty_config,
)

OUTPUTS = [
    ([], False, True),
    ([{"type": "message", "content": "  answer "}], False, False),
    ([{"type": "message", "content": [{"text": "  "}]}], False, True),
    (
        [{"type": "reasoning", "summary": [{"text": " a "}]}, {"content": "a"}],
        True,
        False,
    ),
    (
        [
            {"type": "reasoning", "summary": [{"text": " a "}]},
            {"content": [{"text": " a "}]},
        ],
        True,
        False,
    ),
    (
        [{"type": "reasoning", "summary": [{"text": " "}]}, {"content": " "}],
        False,
        True,
    ),
    ([{"type": "reasoning", "summary": []}, {"content": "a"}], False, False),
    (
        [
            {"type": "reasoning", "summary": [{"text": "a"}]},
            {"type": "function_call"},
            {"content": "a"},
        ],
        False,
        False,
    ),
    ([{"content": "a"}, {"content": []}], False, False),
    ([{"content": "a"}, {"content": [{}]}], False, False),
    ([{"content": {"text": "a"}}], False, True),
    (
        [{"type": "reasoning", "summary": [{"text": "a"}]}, {"type": "function_call"}],
        False,
        False,
    ),
]


@pytest.mark.parametrize("output,duplicated,empty", OUTPUTS)
def test_text_detector_contract(output, duplicated, empty):
    assert has_duplicated_reasoning(output) is duplicated
    assert has_empty_final_answer(output) is empty


@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("reward", [-2.0, 0.0, 1.5])
@pytest.mark.parametrize("output,duplicated,empty", OUTPUTS)
def test_inline_and_capture_reward_parity(flags, reward, output, duplicated, empty):
    resolved = resolve_reward_penalty_config(
        {
            "penalize_duplicated_reasoning": flags[0],
            "penalize_empty_final_answer": flags[1],
            "penalize_unwanted_tokens": flags[2],
            "token_ids": {"unwanted": [99]},
        },
        None,
    )
    config = RewardPenaltyConfig.model_validate(resolved)
    result = {
        "full_result": {"reward": reward, "response": {"output": output}},
        "message_log": [
            {"role": "user", "token_ids": [99, 10]},
            {"role": "assistant", "token_ids": [11]},
            {"role": "user", "token_ids": [99]},
            {"role": "assistant", "token_ids": [12, 99]},
        ],
    }
    expected = apply_reward_penalties([result], resolved)
    checks = compute_reward_checks(result["full_result"], {}, None)
    actual_reward, counts, _ = finalize_capture_reward(
        reward,
        checks=checks,
        penalty_config=config,
        effort_config=None,
        token_ids=[99, 10, 11, 99, 12, 99],
        link_spans=[("c1", 2, 1), ("c2", 1, 2)],
    )
    assert actual_reward == result["full_result"]["reward"]
    assert counts == {k: expected[k] for k in counts}


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([99, 10, 11, 99, 12, 13], False),
        ([10, 11, 99, 20, 12, 13], True),
        ([10, 11, 12, 20, 13, 99], True),
    ],
)
def test_generated_spans_only_including_each_terminal(ids, expected):
    reward, counts, _ = finalize_capture_reward(
        -1.0,
        checks=None,
        effort_config=None,
        penalty_config=RewardPenaltyConfig(
            penalize_unwanted_tokens=True, token_ids={"unwanted": [99]}
        ),
        token_ids=ids,
        link_spans=[("c1", 2, 1), ("c2", 1, 2)],
    )
    assert reward == (0.0 if expected else -1.0)
    assert counts["unwanted_token"] == int(expected)


def test_metric_pooling_uses_valid_rollouts_and_legacy_names():
    metrics = aggregate_capture_reward_metrics(
        {
            "finalize/reward_count": [1, 3, 0],
            "finalize/reward_sum": [0, 6],
            "finalize/reward_sumsq": [0, 12],
            "finalize/reward_min": [0, 2],
            "finalize/reward_max": [0, 2],
            "finalize/penalty_count/empty_final_answer": [1, 0, 0],
        }
    )
    assert metrics == {
        "finalize/reward_count": 4,
        "finalize/penalty_count/empty_final_answer": 1,
        "empty_final_answer_rate": 0.25,
        "total_reward/mean": 1.5,
        "total_reward/stddev": 1.0,
        "total_reward/min": 0,
        "total_reward/max": 2,
    }
    assert aggregate_capture_reward_metrics({"finalize/reward_count": [0]}) == {}


def _input(*messages):
    return {"responses_create_params": {"input": list(messages)}}


LOW = _input({"role": "user", "content": "budget"})
CONFIG = EffortLevelsConfig(
    low_weight=1, low_penalty=1, low_ub=1000, low_string="budget"
)


@pytest.mark.parametrize("reward", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("length", [1, 100, 900, 1000, 1200, 2500])
@pytest.mark.parametrize("weight", [0.0, -1.0, 1.0, 3.0])
@pytest.mark.parametrize("low", [True, False])
def test_finalized_effort_matches_inline_formula_and_metrics(
    reward, length, weight, low
):
    config = CONFIG.model_copy(update={"low_weight": weight})
    prompt = LOW if low else _input({"role": "user", "content": "explain fully"})
    inline = {
        "message_log": [
            {"role": "assistant", "token_ids": [9] * 900},
            {"role": "user", "token_ids": [4] * 17},
            {"role": "assistant", "token_ids": [8] * length},
        ],
        "full_result": {"reward": reward},
    }
    expected_metrics = _apply_effort_shaping([inline], [prompt], config)
    checks = compute_reward_checks({}, prompt, config)
    actual, _, metrics = finalize_capture_reward(
        reward,
        checks=checks,
        effort_config=config,
        penalty_config=None,
        token_ids=[9] * 900 + [8] * length,
        link_spans=[("c1", 0, 900), ("c2", 0, length)],
    )
    assert actual == inline["full_result"]["reward"]
    if weight > 0:
        bucket = "low" if low else "high"
        assert metrics[f"finalize/effort/{bucket}/{length}"] == 1
        if low:
            assert (
                metrics["finalize/effort/reward_sum"] == expected_metrics.rewards_low[0]
            )
            assert (
                metrics["finalize/effort/length_reward_sum"]
                == expected_metrics.length_rewards_low[0]
            )
    else:
        assert metrics == {}
    if weight > 0 and low:
        term = min(1, weight * (1 - length / 1000))
        assert actual == reward + reward * max(term, 0) + min(term, 0)


@pytest.mark.parametrize(
    "messages,expected",
    [
        (
            [
                {"role": "user", "content": "budget"},
                {"role": "assistant", "content": "tools"},
            ],
            True,
        ),
        (
            [
                {"role": "user", "content": "budget"},
                {"role": "user", "content": "long"},
            ],
            False,
        ),
        ([{"role": "user", "content": "budget"}, {"role": "user"}], True),
        ([{"role": "system", "content": "budget"}], False),
        (
            [{"role": "user", "content": [{"type": "input_text", "text": "budget"}]}],
            False,
        ),
    ],
)
def test_classification_retains_original_last_user_membership_rule(messages, expected):
    assert compute_reward_checks({}, _input(*messages), CONFIG).low_effort is expected


@pytest.mark.parametrize("bound", [0, -1])
def test_active_effort_requires_positive_bound(bound):
    with pytest.raises(ValueError, match="low_ub"):
        EffortLevelsConfig(low_weight=1, low_string="budget", low_ub=bound)
    assert (
        EffortLevelsConfig(low_weight=0, low_string="budget", low_ub=bound).low_ub
        == bound
    )
