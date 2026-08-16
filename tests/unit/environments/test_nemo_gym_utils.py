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
"""Pure-Python (vllm-free) unit tests for NeMo-Gym helpers.

These run in the default L0 suite. Keep this module free of heavy imports
(e.g. vllm) so the fast detector tests are not gated behind the nemo_gym extra.
"""

import pytest

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    NEMO_GYM_CONTROL_CONCURRENCY_HEADROOM,
    RAY_DEFAULT_ASYNC_ACTOR_MAX_CONCURRENCY,
    _detect_invalid_tool_call_and_malformed_thinking,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
    resolve_nemo_gym_max_concurrency,
    validate_nemo_gym_actor_concurrency,
)


@pytest.mark.parametrize(
    ("output_item_dict", "expected_invalid_tool_call", "expected_malformed_thinking"),
    [
        (
            {"content": [{"text": "use <tool_call>{}</tool_call>"}]},
            True,
            False,
        ),
        (
            {"content": [{"text": "final answer leaked <think>reasoning</think>"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think>"}]},
            False,
            False,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think><think>b"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "bad <function_call>{}"}]},
            True,
            False,
        ),
    ],
)
def test_detect_invalid_tool_call_and_malformed_thinking(
    output_item_dict,
    expected_invalid_tool_call,
    expected_malformed_thinking,
):
    assert _detect_invalid_tool_call_and_malformed_thinking(output_item_dict) == (
        expected_invalid_tool_call,
        expected_malformed_thinking,
    )


def test_get_nemo_gym_venv_dir_returns_env_value(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    assert get_nemo_gym_venv_dir() == "/opt/gym_venvs"


def test_get_nemo_gym_venv_dir_none_when_unset(monkeypatch):
    monkeypatch.delenv("NEMO_GYM_VENV_DIR", raising=False)
    assert get_nemo_gym_venv_dir() is None


def test_get_nemo_gym_uv_cache_dir_none_outside_container(monkeypatch):
    # Outside a container the caller should omit the arg; uv must not be invoked.
    monkeypatch.delenv("NRL_CONTAINER", raising=False)

    def _fail(*args, **kwargs):
        raise AssertionError("uv should not be invoked outside a container")

    monkeypatch.setattr(nemo_gym_mod.subprocess, "check_output", _fail)
    assert get_nemo_gym_uv_cache_dir() is None


def test_get_nemo_gym_uv_cache_dir_uses_uv_inside_container(monkeypatch):
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setattr(
        nemo_gym_mod.subprocess,
        "check_output",
        lambda *args, **kwargs: b"  /root/.cache/uv\n",
    )
    assert get_nemo_gym_uv_cache_dir() == "/root/.cache/uv"


class TestResolveNemoGymMaxConcurrency:
    """max_concurrency sizing for the single NemoGym rollout actor."""

    def test_derives_from_fan_in_when_unset(self):
        assert (
            resolve_nemo_gym_max_concurrency(None, rollout_fan_in=5120)
            == 5120 + NEMO_GYM_CONTROL_CONCURRENCY_HEADROOM
        )

    def test_derived_default_admits_a_fan_in_past_rays_default(self):
        """The 6K shape: Ray's 1000 default is what stalled it, so never return it."""
        resolved = resolve_nemo_gym_max_concurrency(None, rollout_fan_in=5120)

        assert resolved > RAY_DEFAULT_ASYNC_ACTOR_MAX_CONCURRENCY

    def test_explicit_value_wins(self):
        assert resolve_nemo_gym_max_concurrency(9000, rollout_fan_in=5120) == 9000

    def test_explicit_value_equal_to_fan_in_is_allowed(self):
        assert resolve_nemo_gym_max_concurrency(640, rollout_fan_in=640) == 640

    def test_rejects_explicit_value_below_fan_in(self):
        with pytest.raises(ValueError, match="below the rollout fan-in"):
            resolve_nemo_gym_max_concurrency(1000, rollout_fan_in=5120)

    def test_rejects_non_positive_fan_in(self):
        with pytest.raises(ValueError, match="rollout_fan_in must be positive"):
            resolve_nemo_gym_max_concurrency(None, rollout_fan_in=0)


class TestValidateNemoGymActorConcurrency:
    def test_unset_is_a_no_op(self):
        validate_nemo_gym_actor_concurrency(None, rollout_fan_in=5120)

    def test_error_names_both_knobs(self):
        with pytest.raises(ValueError) as excinfo:
            validate_nemo_gym_actor_concurrency(1000, rollout_fan_in=5120)

        message = str(excinfo.value)
        assert "env.nemo_gym.max_concurrency" in message
        assert "async_rl.max_inflight_prompts" in message
