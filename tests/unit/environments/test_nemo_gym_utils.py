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

from unittest.mock import MagicMock

import pytest

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    _detect_invalid_tool_call_and_malformed_thinking,
    create_nemo_gym_actor,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
    setup_nemo_gym_generation_config,
)


def test_create_nemo_gym_actor_starts_with_generation_endpoints(monkeypatch) -> None:
    actor = MagicMock()
    actor._spinup.remote.return_value = "started"
    actor_class = MagicMock()
    actor_class.options.return_value.remote.return_value = actor

    monkeypatch.setattr(nemo_gym_mod, "NemoGym", actor_class)
    monkeypatch.setattr(nemo_gym_mod, "get_actor_python_env", lambda _: "/gym/python")
    monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: None)
    monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: None)
    ray_get = MagicMock(return_value=None)
    monkeypatch.setattr(nemo_gym_mod.ray, "get", ray_get)

    result = create_nemo_gym_actor(
        model_name="test-model",
        base_urls=["http://worker-0:8000/v1"],
        nemo_gym_config={
            "config_paths": ["gym.yaml"],
            "invalid_tool_call_patterns": ["<bad>"],
        },
    )

    assert result is actor
    actor_options = actor_class.options.call_args.kwargs
    assert actor_options["runtime_env"]["py_executable"] == "/gym/python"
    nemo_gym_config = actor_class.options.return_value.remote.call_args.args[0]
    assert nemo_gym_config["model_name"] == "test-model"
    assert nemo_gym_config["base_urls"] == ["http://worker-0:8000/v1"]
    assert nemo_gym_config["invalid_tool_call_patterns"] == ["<bad>"]
    assert nemo_gym_config["initial_global_config_dict"] == {
        "config_paths": ["gym.yaml"]
    }
    ray_get.assert_called_once_with("started")


def test_setup_nemo_gym_generation_config_enables_http_rollouts() -> None:
    generation_config = {
        "backend": "vllm",
        "stop_strings": ["stop"],
        "stop_token_ids": [1],
        "vllm_cfg": {"async_engine": False, "expose_http_server": False},
    }

    setup_nemo_gym_generation_config(generation_config)

    assert generation_config["vllm_cfg"]["async_engine"] is True
    assert generation_config["vllm_cfg"]["expose_http_server"] is True
    assert generation_config["stop_strings"] is None
    assert generation_config["stop_token_ids"] is None


def test_setup_nemo_gym_generation_config_enables_megatron_http_rollouts() -> None:
    generation_config = {
        "backend": "megatron",
        "stop_strings": ["stop"],
        "stop_token_ids": [1],
        "mcore_generation_config": {
            "async_engine": False,
            "expose_http_server": False,
        },
    }

    setup_nemo_gym_generation_config(generation_config)

    assert generation_config["mcore_generation_config"]["async_engine"] is True
    assert generation_config["mcore_generation_config"]["expose_http_server"] is True
    assert generation_config["stop_strings"] is None
    assert generation_config["stop_token_ids"] is None


def test_setup_nemo_gym_generation_config_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="backend=vllm or backend=megatron"):
        setup_nemo_gym_generation_config({"backend": "sglang"})


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
