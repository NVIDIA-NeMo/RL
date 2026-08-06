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

from pathlib import Path

import pytest

from nemo_rl.environments.vlm_environment import (
    _BUILTIN_REWARD_FUNCTIONS,
    _get_reward_function_registry,
    _resolve_configured_reward_functions,
)


_CUSTOM_REWARD_SOURCE = """
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardOptions:
    scale: float = 1.0


def length_reward(ground_truth, response, scale=1.0):
    return len(response) * scale, None

not_a_reward = 123
"""


def _write_reward_module(path: Path) -> None:
    path.write_text(_CUSTOM_REWARD_SOURCE)


def test_builtin_reward_configuration_is_backward_compatible():
    resolved = _resolve_configured_reward_functions(
        {
            "num_workers": 1,
            "reward_functions": [{"name": "format", "weight": 0.25}],
        }
    )

    assert resolved == [(_BUILTIN_REWARD_FUNCTIONS["format"], 0.25)]


def test_loads_custom_reward_from_importable_module(tmp_path, monkeypatch):
    _write_reward_module(tmp_path / "research_rewards.py")
    monkeypatch.syspath_prepend(tmp_path)

    resolved = _resolve_configured_reward_functions(
        {
            "num_workers": 1,
            "custom_reward_functions": [
                {
                    "name": "length",
                    "module": "research_rewards",
                    "function": "length_reward",
                }
            ],
            "reward_functions": [
                {"name": "length", "weight": 1.0, "kwargs": {"scale": 0.5}}
            ],
        }
    )

    reward_function, weight = resolved[0]
    assert reward_function("unused", "abcd") == (2.0, None)
    assert weight == 1.0


def test_loads_custom_reward_from_python_file(tmp_path):
    reward_file = tmp_path / "research_rewards.py"
    _write_reward_module(reward_file)

    resolved = _resolve_configured_reward_functions(
        {
            "num_workers": 1,
            "custom_reward_functions": [
                {
                    "name": "length",
                    "file": str(reward_file),
                    "function": "length_reward",
                }
            ],
            "reward_functions": [{"name": "length", "weight": 1.0}],
        }
    )

    reward_function, _ = resolved[0]
    assert reward_function("unused", "abc") == (3.0, None)


@pytest.mark.parametrize(
    "custom_config",
    [
        {"name": "length", "function": "length_reward"},
        {
            "name": "length",
            "module": "research_rewards",
            "file": "/tmp/research_rewards.py",
            "function": "length_reward",
        },
    ],
)
def test_custom_reward_requires_exactly_one_source(custom_config):
    with pytest.raises(ValueError, match="exactly one of 'module' or 'file'"):
        _get_reward_function_registry(
            {
                "num_workers": 1,
                "custom_reward_functions": [custom_config],
                "reward_functions": [{"name": "length", "weight": 1.0}],
            }
        )


def test_custom_reward_cannot_replace_builtin(tmp_path):
    reward_file = tmp_path / "research_rewards.py"
    _write_reward_module(reward_file)

    with pytest.raises(ValueError, match="'format' is already registered"):
        _get_reward_function_registry(
            {
                "num_workers": 1,
                "custom_reward_functions": [
                    {
                        "name": "format",
                        "file": str(reward_file),
                        "function": "length_reward",
                    }
                ],
                "reward_functions": [{"name": "format", "weight": 1.0}],
            }
        )


def test_custom_reward_attribute_must_be_callable(tmp_path):
    reward_file = tmp_path / "research_rewards.py"
    _write_reward_module(reward_file)

    with pytest.raises(ValueError, match="resolves to non-callable attribute"):
        _get_reward_function_registry(
            {
                "num_workers": 1,
                "custom_reward_functions": [
                    {
                        "name": "broken",
                        "file": str(reward_file),
                        "function": "not_a_reward",
                    }
                ],
                "reward_functions": [{"name": "broken", "weight": 1.0}],
            }
        )


def test_unknown_reward_error_lists_custom_and_builtin_names(tmp_path):
    reward_file = tmp_path / "research_rewards.py"
    _write_reward_module(reward_file)

    with pytest.raises(ValueError, match="Available reward functions:.*length"):
        _resolve_configured_reward_functions(
            {
                "num_workers": 1,
                "custom_reward_functions": [
                    {
                        "name": "length",
                        "file": str(reward_file),
                        "function": "length_reward",
                    }
                ],
                "reward_functions": [{"name": "typo", "weight": 1.0}],
            }
        )
