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

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    NEMO_GYM_CONTROL_CONCURRENCY_HEADROOM,
    RAY_DEFAULT_ASYNC_ACTOR_MAX_CONCURRENCY,
    _detect_invalid_tool_call_and_malformed_thinking,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
    resolve_nemo_gym_max_concurrency,
    spinup_nemo_gym_actor,
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
        ({"content": None}, False, False),
        ({"content": []}, False, False),
        ({"content": [None]}, False, False),
        ({"content": [{"text": None}]}, False, False),
        ({"type": "reasoning", "summary": None}, False, False),
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

    def test_rejects_explicit_value_equal_to_fan_in(self):
        """Zero slots left over is what locks the control plane out."""
        with pytest.raises(ValueError, match="below the rollout fan-in"):
            resolve_nemo_gym_max_concurrency(640, rollout_fan_in=640)

    def test_accepts_explicit_value_that_clears_the_headroom(self):
        required = 640 + NEMO_GYM_CONTROL_CONCURRENCY_HEADROOM

        assert (
            resolve_nemo_gym_max_concurrency(required, rollout_fan_in=640) == required
        )

    def test_rejects_explicit_value_below_fan_in(self):
        with pytest.raises(ValueError, match="below the rollout fan-in"):
            resolve_nemo_gym_max_concurrency(1000, rollout_fan_in=5120)

    def test_rejects_non_positive_fan_in(self):
        with pytest.raises(ValueError, match="rollout_fan_in must be positive"):
            resolve_nemo_gym_max_concurrency(None, rollout_fan_in=0)

    def test_no_fan_in_leaves_the_option_unset(self):
        """The v1 shape: no config knob to size from, so Ray's default stands."""
        assert resolve_nemo_gym_max_concurrency(None, rollout_fan_in=None) is None

    def test_no_fan_in_still_honours_an_explicit_value(self):
        """There is no floor to hold it to, so a small v1 value is not rejected."""
        assert resolve_nemo_gym_max_concurrency(64, rollout_fan_in=None) == 64


class TestValidateNemoGymActorConcurrency:
    def test_unset_is_a_no_op(self):
        validate_nemo_gym_actor_concurrency(None, rollout_fan_in=5120)

    def test_absent_fan_in_is_a_no_op(self):
        validate_nemo_gym_actor_concurrency(1, rollout_fan_in=None)

    def test_error_names_both_knobs(self):
        with pytest.raises(ValueError) as excinfo:
            validate_nemo_gym_actor_concurrency(1000, rollout_fan_in=5120)

        message = str(excinfo.value)
        assert "env.nemo_gym.max_concurrency" in message
        assert "async_rl.max_inflight_prompts" in message


def _spinup_with_stubbed_ray(nemo_gym_config, *, rollout_fan_in):
    """Run spinup_nemo_gym_actor with every out-of-process dependency stubbed."""
    fake_nemo_gym = MagicMock(name="NemoGym")
    with (
        patch.object(nemo_gym_mod, "NemoGym", fake_nemo_gym),
        patch.object(nemo_gym_mod, "ray", MagicMock(name="ray")),
        patch.object(
            nemo_gym_mod, "get_actor_python_env", return_value="/usr/bin/python"
        ),
        patch.object(nemo_gym_mod, "get_nemo_gym_uv_cache_dir", return_value=None),
        patch.object(nemo_gym_mod, "get_nemo_gym_venv_dir", return_value=None),
    ):
        spinup_nemo_gym_actor(
            env_configs={"nemo_gym": nemo_gym_config},
            base_urls=["http://127.0.0.1:8000"],
            model_name="test-model",
            tokenizer=MagicMock(name="tokenizer"),
            enable_router_replay=False,
            routed_experts_dtype="int16",
            use_fastokens=False,
            rollout_fan_in=rollout_fan_in,
        )
    options_kwargs = fake_nemo_gym.options.call_args.kwargs
    (built_config,), _ = fake_nemo_gym.options.return_value.remote.call_args
    return options_kwargs, built_config


class TestSpinupNemoGymActorConcurrency:
    """max_concurrency is a Ray option, so it must reach .options() and only there."""

    def test_derived_value_reaches_ray_options(self):
        options_kwargs, _ = _spinup_with_stubbed_ray({}, rollout_fan_in=5120)

        assert options_kwargs["max_concurrency"] == (
            5120 + NEMO_GYM_CONTROL_CONCURRENCY_HEADROOM
        )

    def test_explicit_value_is_moved_out_of_the_gym_global_config(self):
        options_kwargs, built_config = _spinup_with_stubbed_ray(
            {"max_concurrency": 6144, "some_gym_setting": 1}, rollout_fan_in=5120
        )

        assert options_kwargs["max_concurrency"] == 6144
        assert "max_concurrency" not in built_config["initial_global_config_dict"]
        assert built_config["initial_global_config_dict"]["some_gym_setting"] == 1

    def test_option_is_omitted_without_a_fan_in(self):
        """The v1 actor is configured exactly as it was before this option existed."""
        options_kwargs, _ = _spinup_with_stubbed_ray({}, rollout_fan_in=None)

        assert "max_concurrency" not in options_kwargs
