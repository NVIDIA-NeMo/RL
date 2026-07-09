# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_rl.experience.rollouts import EffortLevelsConfig, _apply_effort_shaping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(reward: float, response_tokens: int) -> dict:
    """Build a minimal result dict with one assistant message."""
    return {
        "full_result": {"reward": reward},
        "message_log": [
            {"role": "assistant", "token_ids": list(range(response_tokens))}
        ],
    }


def _make_row(prompt: str) -> dict:
    """Build a minimal nemo_gym_rows entry whose last user turn is ``prompt``."""
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": prompt},
            ]
        }
    }


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_effort_levels_config_defaults():
    cfg = EffortLevelsConfig()
    assert cfg.low_weight == 0.0
    assert cfg.low_penalty == 1.0
    assert cfg.low_ub == 64000
    assert cfg.low_string == ""


# ---------------------------------------------------------------------------
# No-op paths
# ---------------------------------------------------------------------------


def test_no_shaping_when_config_is_none():
    results = [_make_result(reward=1.0, response_tokens=100)]
    rows = [_make_row("think carefully")]
    metrics = _apply_effort_shaping(results, rows, effort_config=None)
    assert metrics.length_rewards_low == []
    assert metrics.rewards_low == []
    assert metrics.low_lengths == []
    assert metrics.high_lengths == []
    assert results[0]["full_result"]["reward"] == 1.0


def test_no_shaping_when_low_weight_is_zero():
    cfg = EffortLevelsConfig(low_weight=0.0, low_string="<think>")
    results = [_make_result(reward=1.0, response_tokens=100)]
    rows = [_make_row("<think> solve this")]
    metrics = _apply_effort_shaping(results, rows, effort_config=cfg)
    assert metrics.length_rewards_low == []
    assert results[0]["full_result"]["reward"] == 1.0


def test_no_shaping_when_low_string_is_empty():
    cfg = EffortLevelsConfig(low_weight=1.0, low_string="")
    results = [_make_result(reward=1.0, response_tokens=100)]
    rows = [_make_row("any prompt")]
    metrics = _apply_effort_shaping(results, rows, effort_config=cfg)
    assert metrics.length_rewards_low == []
    assert results[0]["full_result"]["reward"] == 1.0


# ---------------------------------------------------------------------------
# Reward formula
# ---------------------------------------------------------------------------


def test_short_response_amplifies_reward():
    """A response well under low_ub → positive length_reward → reward increases."""
    cfg = EffortLevelsConfig(
        low_weight=1.0, low_penalty=1.0, low_ub=1000, low_string="<budget>"
    )
    # 100 tokens out of 1000 → length_reward = min(1, 1*(1 - 0.1)) = 0.9
    results = [_make_result(reward=1.0, response_tokens=100)]
    rows = [_make_row("<budget> solve this")]

    metrics = _apply_effort_shaping(results, rows, effort_config=cfg)

    expected_length_reward = min(1.0, 1.0 * (1.0 - 100 / 1000))  # 0.9
    expected_reward = 1.0 + 1.0 * max(expected_length_reward, 0.0)  # 1.9
    assert metrics.length_rewards_low == pytest.approx([expected_length_reward])
    assert metrics.rewards_low == pytest.approx([expected_reward])
    assert results[0]["full_result"]["reward"] == pytest.approx(expected_reward)
    assert metrics.low_lengths == [100]
    assert metrics.high_lengths == []


def test_long_response_applies_penalty():
    """A response over low_ub → negative length_reward → low_penalty is applied."""
    cfg = EffortLevelsConfig(
        low_weight=1.0, low_penalty=2.0, low_ub=100, low_string="<budget>"
    )
    # 200 tokens, low_ub=100 → raw = 1*(1 - 2.0) = -1.0 → clamped to min with 1 → -1.0
    results = [_make_result(reward=1.0, response_tokens=200)]
    rows = [_make_row("<budget> solve this")]

    metrics = _apply_effort_shaping(results, rows, effort_config=cfg)

    expected_length_reward = min(1.0, 1.0 * (1.0 - 200 / 100))  # -1.0
    expected_reward = (
        1.0
        + 1.0 * max(expected_length_reward, 0.0)
        + 2.0 * min(expected_length_reward, 0.0)
    )
    # = 1.0 + 0 + 2.0*(-1.0) = -1.0
    assert metrics.length_rewards_low == pytest.approx([expected_length_reward])
    assert metrics.rewards_low == pytest.approx([expected_reward])
    assert results[0]["full_result"]["reward"] == pytest.approx(expected_reward)


def test_length_reward_capped_at_one():
    """length_reward is capped at 1.0 even with a large low_weight."""
    cfg = EffortLevelsConfig(
        low_weight=100.0, low_penalty=1.0, low_ub=1000, low_string="<budget>"
    )
    results = [_make_result(reward=1.0, response_tokens=1)]
    rows = [_make_row("<budget> go")]

    metrics = _apply_effort_shaping(results, rows, effort_config=cfg)

    assert metrics.length_rewards_low[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Routing: low vs. high buckets
# ---------------------------------------------------------------------------


def test_prompt_without_low_string_goes_to_high_bucket():
    cfg = EffortLevelsConfig(low_weight=1.0, low_string="<budget>")
    results = [_make_result(reward=1.0, response_tokens=50)]
    rows = [_make_row("ordinary prompt without the trigger")]

    metrics = _apply_effort_shaping(results, rows, effort_config=cfg)

    assert metrics.low_lengths == []
    assert metrics.high_lengths == [50]
    assert results[0]["full_result"]["reward"] == 1.0  # unchanged


def test_mixed_batch_routes_correctly():
    """Two samples: one with low_string, one without."""
    cfg = EffortLevelsConfig(
        low_weight=1.0, low_penalty=1.0, low_ub=1000, low_string="<budget>"
    )
    results = [
        _make_result(reward=1.0, response_tokens=100),  # low-effort prompt
        _make_result(reward=0.5, response_tokens=200),  # high-effort prompt
    ]
    rows = [
        _make_row("<budget> be concise"),
        _make_row("explain everything in detail"),
    ]

    metrics = _apply_effort_shaping(results, rows, effort_config=cfg)

    assert len(metrics.low_lengths) == 1
    assert len(metrics.high_lengths) == 1
    assert metrics.low_lengths == [100]
    assert metrics.high_lengths == [200]
    # high-effort sample reward must be unchanged
    assert results[1]["full_result"]["reward"] == pytest.approx(0.5)
