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

import asyncio
from typing import Optional

import pytest

from nemo_rl.experience.rollout_manager import (
    AsyncNemoGymRolloutImpl,
    RolloutManager,
)
from nemo_rl.experience.rollouts import EffortLevelsConfig, _apply_effort_shaping
from nemo_rl.utils.timer import Timer

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


# ---------------------------------------------------------------------------
# Wiring: RolloutManager -> AsyncNemoGymRolloutImpl -> Completion + metrics
#
# The cases above exercise `_apply_effort_shaping` in isolation. These drive the
# real `_run_rollouts` so they also pin down *where* the shaping happens: it must
# land before `_results_to_completions`, or `Completion.reward` -- the value that
# becomes `total_reward` in the train batch -- keeps the env's raw reward.
# ---------------------------------------------------------------------------

_GENERATION_CONFIG = {
    "stop_strings": None,
    "stop_token_ids": None,
    "top_k": None,
}

_LOW_EFFORT_CONFIG = EffortLevelsConfig(
    low_weight=1.0, low_penalty=1.0, low_ub=1000, low_string="<budget>"
)

_SHAPING_METRIC_KEYS = (
    "mean_length_reward_low",
    "mean_reward_low",
    "mean_length_low",
    "median_length_low",
    "mean_length_high",
    "median_length_high",
)


class _ReadyRef:
    """Stands in for the ObjectRef each streamed NeMo-Gym row resolves to."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def _resolve():
            return self.value

        return _resolve().__await__()


class _Stream:
    def __init__(self, refs):
        self._refs = iter(refs)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._refs)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _RunRolloutsRemote:
    """The Ray actor handle is the only seam that has to be faked."""

    def __init__(self, results):
        self._results = results

    def options(self, *, num_returns):
        assert num_returns == "streaming"
        return self

    def remote(self, inputs, timer_prefix):
        del inputs, timer_prefix
        return _Stream([_ReadyRef((i, r, None)) for i, r in enumerate(self._results)])


def _gym_result(reward: float, response_tokens: int) -> dict:
    """Build one NeMo-Gym result row, pre-tensorization."""
    return {
        "input_message_log": [{"role": "user", "token_ids": [1, 2]}],
        "message_log": [
            {"role": "user", "token_ids": [1, 2]},
            {
                "role": "assistant",
                "token_ids": list(range(response_tokens)),
                "generation_logprobs": [0.0] * response_tokens,
            },
        ],
        "full_result": {"reward": reward},
    }


def _gym_input(rowidx: int, prompt: str) -> dict:
    return {
        "_rowidx": rowidx,
        "agent_ref": {"name": "agent"},
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
    }


def _run_gym_rollouts(
    effort_config: Optional[EffortLevelsConfig],
    prompt: str,
    results: list[dict],
):
    """Drive the real _run_rollouts against a fake NeMo-Gym stream."""
    impl = AsyncNemoGymRolloutImpl(
        tokenizer=None,
        task_to_env={},
        num_generations_per_prompt=len(results),
        max_seq_len=100_000,
        max_rollout_turns=1,
        generation_config=_GENERATION_CONFIG,
        effort_config=effort_config,
    )
    impl._task_to_env = {
        "nemo_gym": type(
            "_Environment", (), {"run_rollouts": _RunRolloutsRemote(results)}
        )()
    }
    inputs = [_gym_input(i, prompt) for i in range(len(results))]
    return asyncio.run(impl._run_rollouts(inputs, Timer(), "timing/test"))


def test_rollout_manager_forwards_effort_config():
    """effort_config reaches the NeMo-Gym impl through RolloutManager."""
    common = {
        "tokenizer": None,
        "task_to_env": {},
        "num_generations_per_prompt": 1,
        "max_seq_len": 1,
        "generation_config": _GENERATION_CONFIG,
        "use_nemo_gym": True,
    }

    assert RolloutManager(**common)._impl._effort_config is None
    manager = RolloutManager(**common, effort_config=_LOW_EFFORT_CONFIG)
    assert manager._impl._effort_config is _LOW_EFFORT_CONFIG


def test_run_rollouts_shapes_completion_reward_and_emits_low_metrics():
    """The Completion carries the shaped reward, not the env's raw reward."""
    completions, _, metrics = _run_gym_rollouts(
        _LOW_EFFORT_CONFIG,
        "<budget> be concise",
        [_gym_result(reward=1.0, response_tokens=100)],
    )

    # length_reward = min(1, 1.0 * (1 - 100/1000)) = 0.9 -> 1.0 + 1.0 * 0.9 = 1.9
    assert completions[0].reward == pytest.approx(1.9)
    # Downstream reward metrics are computed from the shaped Completion.
    assert metrics["total_reward/mean"] == pytest.approx(1.9)
    assert metrics["mean_length_reward_low"] == pytest.approx(0.9)
    assert metrics["mean_reward_low"] == pytest.approx(1.9)
    assert metrics["mean_length_low"] == pytest.approx(100)
    assert metrics["median_length_low"] == pytest.approx(100.0)
    assert "mean_length_high" not in metrics
    assert "median_length_high" not in metrics


def test_run_rollouts_leaves_high_effort_prompt_reward_untouched():
    """A prompt without low_string is only counted, never re-scored."""
    completions, _, metrics = _run_gym_rollouts(
        _LOW_EFFORT_CONFIG,
        "explain everything in detail",
        [_gym_result(reward=1.0, response_tokens=200)],
    )

    assert completions[0].reward == pytest.approx(1.0)
    assert metrics["mean_length_high"] == pytest.approx(200)
    assert metrics["median_length_high"] == pytest.approx(200.0)
    assert "mean_length_low" not in metrics
    assert "mean_reward_low" not in metrics


def test_run_rollouts_without_effort_config_emits_no_shaping_metrics():
    """The default (effort_levels unset) path is a no-op end to end."""
    completions, _, metrics = _run_gym_rollouts(
        None,
        "<budget> be concise",
        [_gym_result(reward=1.0, response_tokens=100)],
    )

    assert completions[0].reward == pytest.approx(1.0)
    assert not any(key in metrics for key in _SHAPING_METRIC_KEYS)
