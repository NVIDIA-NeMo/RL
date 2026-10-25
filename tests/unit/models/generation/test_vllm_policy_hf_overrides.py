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

"""Regression tests for syncing ``policy.hf_config_overrides`` into vLLM kwargs.

The algorithm entry points used to assign ``vllm_kwargs["hf_overrides"]``
wholesale, which silently dropped any generation-only overrides the user set
under ``policy.generation.vllm_kwargs.hf_overrides`` (#3728).
"""

import warnings

import pytest

from nemo_rl.models.generation.vllm.config import merge_policy_hf_overrides


def test_generation_only_overrides_survive():
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 8192}}

    merge_policy_hf_overrides(vllm_kwargs, None)

    assert vllm_kwargs["hf_overrides"] == {"max_position_embeddings": 8192}


def test_policy_overrides_are_added_without_conflict():
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 8192}}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merge_policy_hf_overrides(vllm_kwargs, {"rope_theta": 1_000_000.0})

    assert vllm_kwargs["hf_overrides"] == {
        "max_position_embeddings": 8192,
        "rope_theta": 1_000_000.0,
    }


def test_policy_wins_on_conflict_and_warns():
    vllm_kwargs = {"hf_overrides": {"rope_theta": 1.0, "other": "keep"}}

    with pytest.warns(UserWarning, match=r"\['rope_theta'\]"):
        merge_policy_hf_overrides(vllm_kwargs, {"rope_theta": 2.0})

    assert vllm_kwargs["hf_overrides"] == {"rope_theta": 2.0, "other": "keep"}


def test_equal_values_do_not_warn():
    vllm_kwargs = {"hf_overrides": {"rope_theta": 2.0}}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merge_policy_hf_overrides(vllm_kwargs, {"rope_theta": 2.0})

    assert vllm_kwargs["hf_overrides"] == {"rope_theta": 2.0}


@pytest.mark.parametrize("existing", [{}, {"hf_overrides": None}, {"hf_overrides": {}}])
def test_no_overrides_anywhere_yields_empty_mapping(existing):
    vllm_kwargs = dict(existing)

    merge_policy_hf_overrides(vllm_kwargs, {})

    assert vllm_kwargs["hf_overrides"] == {}


def test_policy_overrides_are_copied_not_aliased():
    policy_overrides = {"rope_theta": 2.0}
    vllm_kwargs = {}

    merge_policy_hf_overrides(vllm_kwargs, policy_overrides)
    vllm_kwargs["hf_overrides"]["extra"] = 1

    assert policy_overrides == {"rope_theta": 2.0}


def test_non_mapping_generation_overrides_are_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        merge_policy_hf_overrides({"hf_overrides": "oops"}, {})
