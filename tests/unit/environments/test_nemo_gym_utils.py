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
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
    sanitize_nemo_gym_example_image_placeholders,
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


def test_sanitize_image_placeholders_uses_structured_images_as_source_of_truth():
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Count <img><image></img>."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA==",
                        },
                    ],
                }
            ]
        }
    }

    sanitized = sanitize_nemo_gym_example_image_placeholders(example)

    assert sanitized["responses_create_params"]["input"][0]["content"][0]["text"] == (
        "Count ."
    )
    assert (
        "<image>"
        in example["responses_create_params"]["input"][0]["content"][0]["text"]
    )


def test_sanitize_image_placeholders_preserves_text_only_meaning():
    example = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Describe the <image>."}]
        }
    }

    sanitized = sanitize_nemo_gym_example_image_placeholders(example)

    assert sanitized["responses_create_params"]["input"][0]["content"] == (
        "Describe the image."
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


def test_spinup_nemo_gym_actor_uses_venv_directory_in_runtime_env(monkeypatch):
    venv_dir = "/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym"
    runtime_env = {
        "py_executable": f"{venv_dir}/bin/python",
        "env_vars": {
            "VIRTUAL_ENV": venv_dir,
            "UV_PROJECT_ENVIRONMENT": venv_dir,
        },
    }
    make_runtime_env = MagicMock(return_value=runtime_env)
    actor = MagicMock()
    spinup_ref = object()
    actor._spinup.remote.return_value = spinup_ref
    nemo_gym_cls = MagicMock()
    nemo_gym_cls.options.return_value.remote.return_value = actor
    ray_get = MagicMock()

    monkeypatch.setattr(nemo_gym_mod, "make_actor_runtime_env", make_runtime_env)
    monkeypatch.setattr(nemo_gym_mod, "NemoGym", nemo_gym_cls)
    monkeypatch.setattr(nemo_gym_mod.ray, "get", ray_get)
    monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: None)
    monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: None)

    result = spinup_nemo_gym_actor(
        env_configs={"nemo_gym": {"config_paths": ["circle_count.yaml"]}},
        base_urls=["http://policy/v1"],
        model_name="policy",
        enable_router_replay=False,
        routed_experts_dtype="int8",
        use_fastokens=False,
    )

    assert result is actor
    make_runtime_env.assert_called_once_with("nemo_rl.environments.nemo_gym.NemoGym")
    assert nemo_gym_cls.options.call_args.kwargs["runtime_env"] == runtime_env
    assert runtime_env["env_vars"]["VIRTUAL_ENV"] == venv_dir
    assert runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"] == venv_dir
    ray_get.assert_called_once_with(spinup_ref)
