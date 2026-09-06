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

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.megatron.utils import (
    build_image_preprocessing_config,
    build_video_preprocessing_config,
)

pytestmark = pytest.mark.mcore


def _image_processor(**overrides):
    fields = {
        "patch_size": 14,
        "min_num_patches": 1,
        "max_num_patches": 32,
        "norm_mean": [0.1, 0.2, 0.3],
        "norm_std": [0.4, 0.5, 0.6],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_build_image_config_handles_dict_patch_and_downsample_ratio():
    config = build_image_preprocessing_config(
        _image_processor(
            patch_size={"height": 16, "width": 16},
            downsample_ratio=0.5,
        ),
        dynamic_resolution=True,
        vision_model_type="qwen-vl",
    )

    assert config.patch_dim == 16
    assert config.dynamic_resolution is True
    assert config.vision_model_type == "qwen-vl"
    assert config.pixel_shuffle is True
    assert config.spatial_merge_size == 2
    assert config.dynamic_resolution_min_patches == 1
    assert config.dynamic_resolution_max_patches == 32
    assert config.pixel_mean == [0.1, 0.2, 0.3]
    assert config.pixel_std == [0.4, 0.5, 0.6]


@pytest.mark.parametrize(
    ("merge_fields", "expected_merge_size"),
    [
        ({"merge_size": 4}, 4),
        ({"spatial_merge_size": 3}, 3),
        ({}, 1),
    ],
)
def test_build_image_config_merge_size_fallbacks(merge_fields, expected_merge_size):
    config = build_image_preprocessing_config(_image_processor(**merge_fields))

    assert config.spatial_merge_size == expected_merge_size
    assert config.pixel_shuffle is (expected_merge_size > 1)


def test_build_image_config_accepts_alternate_field_names():
    config = build_image_preprocessing_config(
        SimpleNamespace(
            patch_dim=12,
            min_num_patches=2,
            max_num_patches=24,
            image_mean=(0.1, 0.2, 0.3),
            image_std=(0.7, 0.8, 0.9),
        )
    )

    assert config.patch_dim == 12
    assert config.dynamic_resolution_min_patches == 2
    assert config.dynamic_resolution_max_patches == 24
    assert config.pixel_mean == [0.1, 0.2, 0.3]
    assert config.pixel_std == [0.7, 0.8, 0.9]


def test_build_image_config_error_names_all_missing_fields():
    with pytest.raises(ValueError) as exc_info:
        build_image_preprocessing_config(SimpleNamespace())

    for field in (
        "patch_size",
        "min_num_patches",
        "max_num_patches",
        "norm_mean",
        "norm_std",
    ):
        assert field in str(exc_info.value)


def test_build_video_config_returns_none_when_disabled():
    image_config = build_image_preprocessing_config(_image_processor())

    assert (
        build_video_preprocessing_config(
            None,
            {"video_temporal_patch_size": 2, "video_num_frames": 8},
            frame_manifest_magic=b"manifest",
        )
        is None
    )
    assert (
        build_video_preprocessing_config(
            image_config,
            {},
            frame_manifest_magic=b"manifest",
        )
        is None
    )


def test_build_video_config_is_not_enabled_by_temporal_patch_size_alone():
    image_config = build_image_preprocessing_config(_image_processor())

    assert (
        build_video_preprocessing_config(
            image_config,
            {"video_temporal_patch_size": 2},
            frame_manifest_magic=b"manifest",
        )
        is None
    )


def test_build_video_config_uses_default_temporal_patch_size():
    image_config = build_image_preprocessing_config(_image_processor())

    video_config = build_video_preprocessing_config(
        image_config,
        {"video_num_frames": 8},
        frame_manifest_magic=b"manifest",
    )

    assert video_config is not None
    assert video_config.num_frames == 8
    assert video_config.temporal_patch_size == 1


def test_build_video_config_overrides_patch_budget_without_mutating_image_config():
    image_config = build_image_preprocessing_config(_image_processor())

    video_config = build_video_preprocessing_config(
        image_config,
        {
            "video_num_frames": 8,
            "video_temporal_patch_size": 2,
            "video_target_num_patches": 64,
            "video_maintain_aspect_ratio": False,
        },
        frame_manifest_magic=b"manifest",
    )

    assert video_config is not None
    assert video_config.image_config is not image_config
    assert image_config.dynamic_resolution_max_patches == 32
    assert video_config.image_config.dynamic_resolution_max_patches == 64
    assert video_config.num_frames == 8
    assert video_config.temporal_patch_size == 2
    assert video_config.frame_manifest_magic == b"manifest"
    assert video_config.video_maintain_aspect_ratio is False
