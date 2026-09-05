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

import copy
from unittest.mock import MagicMock, call, patch

import pytest

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    NEMO_GYM_ACTOR_FQN,
    _detect_invalid_tool_call_and_malformed_thinking,
    build_nemo_gym_config,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
    spinup_nemo_gym_actor,
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


def _env_configs(**overrides):
    nemo_gym = {
        "num_gpu_nodes": 1,
        "invalid_tool_call_patterns": ["bad_call"],
        "thinking_tags": ["<think>"],
        "tokenizer_config": {"name": "test-tokenizer"},
        "pad_dynamic_image_shapes": True,
        "config_paths": ["gym.yaml"],
    }
    nemo_gym.update(overrides)
    return {"nemo_gym": nemo_gym}


@pytest.fixture
def detected_uv_dirs(monkeypatch):
    """Pretend we are in a container with image-baked uv cache + venv dirs."""
    monkeypatch.setattr(
        nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: "/opt/nemo-gym/.uv-cache"
    )
    monkeypatch.setattr(
        nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: "/opt/nemo-gym/venvs"
    )


def test_build_nemo_gym_config_splits_nemo_rl_keys(detected_uv_dirs):
    """NeMo-RL-only knobs become top-level fields; the rest is Gym's global config."""
    env_configs = _env_configs()
    env_configs_before = copy.deepcopy(env_configs)

    cfg = build_nemo_gym_config(
        env_configs,
        base_urls=["http://vllm-0"],
        model_name="test-model",
        enable_router_replay=False,
        use_fastokens=False,
    )

    assert cfg["model_name"] == "test-model"
    assert cfg["base_urls"] == ["http://vllm-0"]
    assert cfg["invalid_tool_call_patterns"] == ["bad_call"]
    assert cfg["thinking_tags"] == ["<think>"]
    assert cfg["tokenizer_config"] == {"name": "test-tokenizer"}
    assert cfg["pad_dynamic_image_shapes"] is True
    assert cfg["initial_global_config_dict"] == {
        "num_gpu_nodes": 1,
        "config_paths": ["gym.yaml"],
        "uv_cache_dir": "/opt/nemo-gym/.uv-cache",
        "uv_venv_dir": "/opt/nemo-gym/venvs",
    }
    # The caller's master_config.env must survive untouched.
    assert env_configs == env_configs_before


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, ("/opt/nemo-gym/.uv-cache", "/opt/nemo-gym/venvs")),
        (
            {"uv_cache_dir": "/custom/cache", "uv_venv_dir": "/custom/venvs"},
            ("/custom/cache", "/custom/venvs"),
        ),
    ],
    ids=["detected", "explicit-wins"],
)
def test_build_nemo_gym_config_uv_dirs(detected_uv_dirs, configured, expected):
    cfg = build_nemo_gym_config(
        _env_configs(**configured),
        base_urls=[],
        model_name="test-model",
        enable_router_replay=False,
        use_fastokens=False,
    )
    global_config = cfg["initial_global_config_dict"]
    assert (global_config["uv_cache_dir"], global_config["uv_venv_dir"]) == expected


def test_build_nemo_gym_config_router_replay_off_uses_default_dtype(detected_uv_dirs):
    cfg = build_nemo_gym_config(
        _env_configs(),
        base_urls=[],
        model_name="test-model",
        enable_router_replay=False,
        use_fastokens=False,
    )
    assert cfg["require_routed_experts"] is False
    assert cfg["routed_experts_dtype"] == "int16"


def test_build_nemo_gym_config_router_replay_resolves_dtype(detected_uv_dirs):
    with patch.object(
        nemo_gym_mod,
        "resolve_routed_experts_dtype_name_for_model",
        return_value="int8",
    ) as mock_resolve:
        cfg = build_nemo_gym_config(
            _env_configs(),
            base_urls=[],
            model_name="test-model",
            enable_router_replay=True,
            use_fastokens=False,
        )

    mock_resolve.assert_called_once_with("test-model")
    assert cfg["require_routed_experts"] is True
    assert cfg["routed_experts_dtype"] == "int8"


@pytest.mark.parametrize("num_gpu_nodes", [0, 1], ids=["no-gpus", "colocated-gpus"])
def test_spinup_nemo_gym_actor(detected_uv_dirs, num_gpu_nodes):
    """The actor gets the registry runtime_env, and node affinity only when it
    has colocated GPUs to land next to."""
    actor = MagicMock()
    actor._spinup.remote.return_value = "spinup-ref"
    actor.set_tokenizer.remote.return_value = "tokenizer-ref"
    tokenizer = MagicMock()
    runtime_env = {"py_executable": "/venv/bin/python"}

    with (
        patch.object(
            nemo_gym_mod, "make_actor_runtime_env", return_value=runtime_env
        ) as mock_runtime_env,
        patch.object(nemo_gym_mod, "NemoGym") as mock_cls,
        patch.object(nemo_gym_mod, "ray") as mock_ray,
    ):
        mock_cls.options.return_value.remote.return_value = actor
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "a" * 56

        result = spinup_nemo_gym_actor(
            _env_configs(num_gpu_nodes=num_gpu_nodes),
            base_urls=["http://vllm-0"],
            model_name="test-model",
            tokenizer=tokenizer,
            enable_router_replay=False,
            use_fastokens=True,
        )

    assert result is actor
    mock_runtime_env.assert_called_once_with(NEMO_GYM_ACTOR_FQN)

    options_kwargs = mock_cls.options.call_args.kwargs
    assert options_kwargs["runtime_env"] is runtime_env
    if num_gpu_nodes:
        assert isinstance(
            options_kwargs["scheduling_strategy"],
            nemo_gym_mod.NodeAffinitySchedulingStrategy,
        )
        assert options_kwargs["scheduling_strategy"].node_id == "a" * 56
    else:
        assert "scheduling_strategy" not in options_kwargs

    cfg = mock_cls.options.return_value.remote.call_args.args[0]
    assert cfg["use_fastokens"] is True

    # Spinup is deferred from __init__, so the factory must await it.
    actor._spinup.remote.assert_called_once_with()
    actor.set_tokenizer.remote.assert_called_once_with(tokenizer)
    assert mock_ray.get.call_args_list == [call("spinup-ref"), call("tokenizer-ref")]
