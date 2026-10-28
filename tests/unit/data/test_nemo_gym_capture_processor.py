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

import json
from copy import deepcopy

import pytest
import torch
from PIL import Image

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data.processors import (
    nemo_gym_capture_data_processor,
    nemo_gym_data_processor,
)


@pytest.fixture
def cached_video_datum(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_RL_VIDEO_MEDIA_ROOT", str(tmp_path))
    parts = []
    for index in range(4):
        frame_path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(frame_path)
        parts.append(
            {
                "type": "input_image",
                "image_url": str(frame_path),
                "_is_video_frame": True,
                "_video_source": "/fixture/video.mp4",
            }
        )
    parts.append({"type": "input_text", "text": "Describe this video."})
    return {
        "task_name": "nemo_gym",
        "extra_env_info": json.dumps(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": parts}],
                    "metadata": {
                        "extra_body": json.dumps(
                            {"chat_template_kwargs": {"enable_thinking": True}}
                        )
                    },
                }
            }
        ),
    }


@pytest.fixture
def video_spec():
    return TaskDataSpec(
        num_frames=4,
        video_sampling_style="nemotron_vl",
        video_temporal_patch_size=2,
        video_target_num_patches=64,
        video_maintain_aspect_ratio=True,
        min_generation_tokens=16,
    )


class _VideoTokenizer:
    model_input_names = ["input_ids"]

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}


class _VideoProcessor:
    model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]
    tokenizer = _VideoTokenizer()

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        if not tokenize:
            return "<image>\nDescribe this video."
        assert kwargs["video_flags"] == [True] * 4
        assert kwargs["video_temporal_patch_size"] == 2
        assert kwargs["video_target_num_patches"] == 64
        assert kwargs["video_maintain_aspect_ratio"] is True
        assert kwargs["enable_thinking"] is True
        return {
            "input_ids": torch.tensor([[7, 18, 18, 9]]),
            "pixel_values": torch.ones(4, 3, 8, 8),
            "imgs_sizes": torch.tensor([[8, 8]] * 4),
        }


@pytest.mark.parametrize("max_seq_length", [128, 4])
def test_capture_discards_policy_media_but_preserves_preparation(
    cached_video_datum, video_spec, max_seq_length
):
    source = deepcopy(cached_video_datum)
    legacy = nemo_gym_data_processor(
        cached_video_datum, video_spec, _VideoProcessor(), max_seq_length, 3
    )
    captured = nemo_gym_capture_data_processor(
        cached_video_datum, video_spec, _VideoProcessor(), max_seq_length, 3
    )
    assert cached_video_datum == source
    for key in ("extra_env_info", "length", "loss_multiplier", "idx", "task_name"):
        assert captured[key] == legacy[key]
    assert captured["loss_multiplier"] == (1.0 if max_seq_length == 128 else 0.0)
    legacy_message = legacy["message_log"][0]
    captured_message = captured["message_log"][0]
    packed_keys = {
        key for key, value in legacy_message.items() if isinstance(value, PackedTensor)
    }
    assert {"pixel_values", "imgs_sizes", "num_frames"} <= packed_keys
    assert set(captured_message) == set(legacy_message) - packed_keys
    for key, value in captured_message.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, legacy_message[key])
        else:
            assert value == legacy_message[key]


@pytest.mark.parametrize("max_seq_length", [None, 128])
def test_capture_preserves_native_video_manifest(
    cached_video_datum, video_spec, max_seq_length, monkeypatch
):
    # Exercise actual manifest construction from lossless cached frame files.
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym_multimodal.process_nemotron_video_frames",
        lambda *args, **kwargs: {
            "input_ids": torch.tensor([[7, 18, 18, 9]]),
            "pixel_values": torch.ones(4, 3, 16, 16),
            "imgs_sizes": torch.tensor([[16, 16]] * 4),
        },
    )

    class NemotronH_Omni_Reasoning_V3Processor(_VideoProcessor):
        pass

    processor = NemotronH_Omni_Reasoning_V3Processor()
    legacy = nemo_gym_data_processor(
        cached_video_datum, video_spec, processor, max_seq_length, 3
    )
    captured = nemo_gym_capture_data_processor(
        cached_video_datum, video_spec, processor, max_seq_length, 3
    )
    assert captured["extra_env_info"] == legacy["extra_env_info"]
    request = captured["extra_env_info"]["responses_create_params"]
    video = request["input"][0]["content"][0]
    assert video["type"] == "input_video"
    assert video["video_url"]["url"].startswith(
        "data:video/x-nemo-rl-cached-frames;base64,"
    )
    assert not any(
        isinstance(value, PackedTensor) for value in captured["message_log"][0].values()
    )
    assert isinstance(legacy["message_log"][0]["pixel_values"], PackedTensor)


@pytest.mark.parametrize("use_video_spec", [False, True])
def test_capture_leaves_generic_gym_placeholders_unchanged(video_spec, use_video_spec):
    datum = {
        "task_name": "nemo_gym",
        "extra_env_info": json.dumps(
            {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Play."}],
                        }
                    ]
                }
            }
        ),
    }
    spec = video_spec if use_video_spec else None
    legacy = nemo_gym_data_processor(datum, spec, _VideoProcessor(), 128, 0)
    captured = nemo_gym_capture_data_processor(datum, spec, _VideoProcessor(), 128, 0)
    assert captured["extra_env_info"] == legacy["extra_env_info"]
    assert captured["length"] == legacy["length"] == 0
    assert captured["loss_multiplier"] == legacy["loss_multiplier"] == 1.0
    assert captured["message_log"][0].keys() == legacy["message_log"][0].keys()
    assert captured["message_log"][0]["token_ids"].dtype == torch.long
