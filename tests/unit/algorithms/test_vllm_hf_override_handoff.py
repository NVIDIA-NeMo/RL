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

import copy

import pytest

from nemo_rl.algorithms.grpo import _merge_vllm_hf_overrides


@pytest.mark.parametrize(
    "policy_overrides", [{}, {"rope_theta": 10000, "shared": "policy"}]
)
def test_generation_only_overrides_survive_and_policy_wins(policy_overrides):
    generation_only = {"output_dir": "/job/refit"}
    vllm_kwargs = {
        "skip_mm_profiling": True,
        "hf_overrides": {"generation_only": generation_only, "shared": "generation"},
    }
    policy = {"hf_config_overrides": copy.deepcopy(policy_overrides)}

    _merge_vllm_hf_overrides(vllm_kwargs, policy)

    overrides = vllm_kwargs["hf_overrides"]
    assert overrides["generation_only"] == generation_only
    assert overrides["shared"] == policy_overrides.get("shared", "generation")
    if policy_overrides:
        assert overrides["rope_theta"] == 10000
    assert vllm_kwargs["skip_mm_profiling"] is True
    assert policy["hf_config_overrides"] == policy_overrides


@pytest.mark.parametrize(
    "policy_overrides,generation_overrides,expected",
    [
        (None, {"a": 1}, {"a": 1}),
        ({"rope_theta": 10000}, None, {"rope_theta": 10000}),
        (None, None, {}),
    ],
)
def test_missing_overrides_are_empty(policy_overrides, generation_overrides, expected):
    vllm_kwargs = {"hf_overrides": generation_overrides}
    _merge_vllm_hf_overrides(vllm_kwargs, {"hf_config_overrides": policy_overrides})
    assert vllm_kwargs["hf_overrides"] == expected


def test_policy_without_override_key():
    vllm_kwargs = {}
    _merge_vllm_hf_overrides(vllm_kwargs, {})
    assert vllm_kwargs == {"hf_overrides": {}}
