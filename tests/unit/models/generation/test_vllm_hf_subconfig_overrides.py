# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_rl.models.generation.vllm.vllm_worker import (
    _merge_hf_subconfig_overrides,
)


def test_hf_subconfig_overrides_are_deep_merged_and_consumed():
    vllm_kwargs = {
        "hf_overrides": {
            "video_temporal_patch_size": 2,
            "vision_config": {"existing": "preserved"},
        }
    }
    subconfig_overrides = {
        "vision_config": {
            "video_target_num_patches": 1024,
            "video_maintain_aspect_ratio": True,
        }
    }

    _merge_hf_subconfig_overrides(vllm_kwargs, subconfig_overrides)

    assert vllm_kwargs["hf_overrides"] == {
        "video_temporal_patch_size": 2,
        "vision_config": {
            "existing": "preserved",
            "video_target_num_patches": 1024,
            "video_maintain_aspect_ratio": True,
        },
    }


def test_hf_subconfig_overrides_do_not_mutate_source_mapping():
    subconfig_overrides = {
        "vision_config": {"video_target_num_patches": 1024}
    }
    original = copy.deepcopy(subconfig_overrides)
    vllm_kwargs = {}

    _merge_hf_subconfig_overrides(vllm_kwargs, subconfig_overrides)

    assert subconfig_overrides == original


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "subconfig_overrides",
            "not-a-mapping",
            "hf_subconfig_overrides must be a mapping",
        ),
        (
            "hf_overrides",
            "not-a-mapping",
            "hf_overrides must be a mapping",
        ),
    ],
)
def test_hf_subconfig_overrides_reject_invalid_mappings(key, value, message):
    vllm_kwargs = {"hf_overrides": {}}
    subconfig_overrides = {"vision_config": {}}
    if key == "hf_overrides":
        vllm_kwargs[key] = value
    else:
        subconfig_overrides = value

    with pytest.raises(TypeError, match=message):
        _merge_hf_subconfig_overrides(vllm_kwargs, subconfig_overrides)
