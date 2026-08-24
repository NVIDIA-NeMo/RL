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
import asyncio
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import requests
import torch
from PIL import Image
from yaml import safe_load

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.multimodal_utils import (
    MULTIMODAL_CONTENT_TYPES,
    PackedTensor,
    image_to_data_url,
)
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    _actor_peak_rss_gib,
    _attach_multimodal_data_to_user_message,
    _compact_json_size,
    _index_per_turn_images,
    _resolve_images_by_media_id,
    _stamp_nemo_gym_rollout_ids,
    build_reward_component_columns,
    extract_reward_components,
    setup_nemo_gym_config,
    validate_reward_components_match_scalar,
)
from nemo_rl.environments.nemo_gym_video import (
    _extract_static_video_messages,
    _inject_vllm_mm_processor_kwargs,
    _metadata_extra_body,
    nemo_gym_example_to_video_datum_spec,
    normalize_video_urls_in_examples,
)
from nemo_rl.environments.nemotron_utils import (
    _expand_nemotron_video_placeholders,
    _flatten_nemotron_video_frame_messages,
)
from nemo_rl.experience.rollouts import (
    _reattach_original_multimodal_payloads,
    attach_static_multimodal_payload,
)
from nemo_rl.models.generation.vllm import VllmGeneration

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import (
    basic_vllm_test_config,
    cluster,  # noqa: F401
)
from tests.unit.models.generation.test_vllm_generation import (
    tokenizer as nemo_gym_tokenizer,  # noqa: F401
)


def _caller_identity_row() -> dict[str, str]:
    return {
        "_nemo_rl_rollout_id": "rollout-1",
        "_nemo_rl_group_id": "group-1",
    }


def test_multimodal_content_types_cover_responses_media_aliases():
    assert {
        "input_image",
        "image",
        "image_url",
        "input_video",
        "video",
        "video_url",
        "input_audio",
        "audio",
        "audio_url",
    } <= MULTIMODAL_CONTENT_TYPES


def test_extract_static_video_message_resolves_local_file(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"test")
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe the clip."},
                        {
                            "type": "input_video",
                            "video_url": {"url": video_path.as_uri()},
                        },
                    ],
                }
            ]
        }
    }

    messages, resolved_path = _extract_static_video_messages(example)

    assert resolved_path == str(video_path.resolve())
    assert messages[0]["content"][0] == {
        "type": "text",
        "text": "Describe the clip.",
    }
    assert messages[0]["content"][1]["type"] == "video"


def test_extract_static_video_message_accepts_cached_frames(tmp_path):
    frame_paths = []
    for index in range(2):
        frame_path = tmp_path / f"frame_{index:04d}.png"
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(frame_path)
        frame_paths.append(frame_path)
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "input_image",
                                "image_url": str(frame_path),
                                "_is_video_frame": True,
                                "_video_source": "/videos/clip.mp4",
                            }
                            for frame_path in frame_paths
                        ],
                        {"type": "input_text", "text": "Describe the clip."},
                    ],
                }
            ]
        }
    }

    messages, resolved_path = _extract_static_video_messages(example)

    assert resolved_path is None
    assert [part["type"] for part in messages[0]["content"]] == [
        "image",
        "image",
        "text",
    ]
    assert all(
        isinstance(part["image"], Image.Image) for part in messages[0]["content"][:2]
    )
    assert all(part["_is_video_frame"] for part in messages[0]["content"][:2])
    assert [part["_video_frame_index"] for part in messages[0]["content"][:2]] == [0, 1]
    assert [part["_video_fps"] for part in messages[0]["content"][:2]] == [1.0, 1.0]

    _, frames, frame_indices, fps = _flatten_nemotron_video_frame_messages(messages)

    assert len(frames) == 2
    assert frame_indices == [0, 1]
    assert fps == 1.0


def test_extract_static_video_message_ignores_still_image_only_row():
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "/images/still.png"},
                        {"type": "input_text", "text": "Describe the image."},
                    ],
                }
            ]
        }
    }

    assert _extract_static_video_messages(example) is None


def test_gym_local_video_path_is_normalized_to_file_url(tmp_path):
    video_path = tmp_path / "clip with spaces.mp4"
    video_path.write_bytes(b"test")
    examples = [
        {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_video",
                                "video_url": {"url": str(video_path)},
                            }
                        ],
                    }
                ]
            }
        }
    ]

    normalize_video_urls_in_examples(examples)

    assert (
        examples[0]["responses_create_params"]["input"][0]["content"][0]["video_url"][
            "url"
        ]
        == video_path.resolve().as_uri()
    )


def test_extract_static_video_message_rejects_multiple_videos(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"test")
    second.write_bytes(b"test")
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": str(first)},
                        {"type": "video_url", "video_url": str(second)},
                    ],
                }
            ]
        }
    }

    with pytest.raises(ValueError, match="exactly one video"):
        _extract_static_video_messages(example)


def test_extract_static_video_message_requires_cached_frame_source(tmp_path):
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(frame_path)
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": str(frame_path),
                            "_is_video_frame": True,
                        }
                    ],
                }
            ]
        }
    }

    with pytest.raises(ValueError, match="exactly one video"):
        _extract_static_video_messages(example)


def test_extract_static_video_message_rejects_mixed_image_and_video(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    image_path = tmp_path / "still.png"
    Image.new("RGB", (2, 2)).save(image_path)
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": str(video_path)},
                        {"type": "input_image", "image_url": str(image_path)},
                    ],
                }
            ]
        }
    }

    with pytest.raises(ValueError, match="mixing still images"):
        _extract_static_video_messages(example)


def test_extract_static_video_message_rejects_audio_and_unknown_parts(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    base_content = [{"type": "input_video", "video_url": str(video_path)}]

    for part, error in (
        (
            {"type": "input_audio", "audio_url": "/audio.wav"},
            "does not support audio",
        ),
        ({"type": "output_text", "text": "answer"}, "Unsupported Gym"),
    ):
        example = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [*base_content, part],
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match=error):
            _extract_static_video_messages(example)


@pytest.mark.parametrize("extra_body", ["{", "[]", 3])
def test_video_metadata_rejects_invalid_extra_body(extra_body):
    example = {"responses_create_params": {"metadata": {"extra_body": extra_body}}}

    with pytest.raises((TypeError, ValueError), match="extra_body"):
        _metadata_extra_body(example)


def test_video_metadata_canonicalizes_mapping_extra_body_to_json_string():
    example = {
        "responses_create_params": {
            "metadata": {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
            }
        }
    }

    _inject_vllm_mm_processor_kwargs(example, {"video_as_images": True})

    extra_body = example["responses_create_params"]["metadata"]["extra_body"]
    assert isinstance(extra_body, str)
    assert json.loads(extra_body) == {
        "chat_template_kwargs": {"enable_thinking": True},
        "mm_processor_kwargs": {"video_as_images": True},
    }


def test_reattach_static_multimodal_payload_to_rollout_user_message():
    payload = PackedTensor([torch.ones(2, 3)], dim_to_pack=0)
    source = [{"role": "user", "content": "", "pixel_values": payload}]
    target = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    attach_static_multimodal_payload(target, source)

    assert target[1]["pixel_values"] is payload


def test_video_datum_uses_temporal_processor_contract(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_video",
                            "video_url": str(video_path),
                            "_request_metadata": "keep",
                        },
                        {"type": "input_text", "text": "Describe the clip."},
                    ],
                }
            ],
            "metadata": {
                "extra_body": json.dumps(
                    {"chat_template_kwargs": {"enable_thinking": True}}
                )
            },
        }
    }

    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym_video.load_video_frames_with_metadata",
        lambda *args, **kwargs: (
            frames,
            {"frames_indices": [0, 3, 6, 9], "fps": 3.0},
        ),
    )

    class _Tokenizer:
        model_input_names = ["input_ids"]

        def __call__(self, text, **kwargs):
            del text, kwargs
            return {"input_ids": [1, 2, 3]}

    class _Processor:
        model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]
        tokenizer = _Tokenizer()

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            if not tokenize:
                return "<image>\nDescribe the clip."
            assert kwargs["video_flags"] == [True, True, True, True]
            assert kwargs["video_temporal_patch_size"] == 2
            assert kwargs["video_target_num_patches"] == 64
            assert kwargs["video_maintain_aspect_ratio"] is True
            assert kwargs["enable_thinking"] is True
            assert all(part.get("type") != "video" for part in messages[0]["content"])
            return {
                "input_ids": torch.tensor([[7, 18, 18, 9]]),
                "pixel_values": torch.ones(4, 3, 8, 8),
                "imgs_sizes": torch.tensor([[8, 8]] * 4),
            }

    data_config = SimpleNamespace(
        num_frames=4,
        video_sampling_style="nemotron_vl",
        video_temporal_patch_size=2,
        video_target_num_patches=64,
        video_maintain_aspect_ratio=True,
        min_generation_tokens=16,
    )
    datum = nemo_gym_example_to_video_datum_spec(
        example,
        processor=_Processor(),
        max_seq_length=128,
        idx=3,
        task_name="nemo_gym",
        data_config=data_config,
    )

    assert datum is not None
    user_message = datum["message_log"][0]
    assert user_message["num_frames"].as_tensor().tolist() == [4]
    assert user_message["imgs_sizes"].as_tensor().dtype == torch.int32
    extra_env_info = datum["extra_env_info"]
    outbound_content = extra_env_info["responses_create_params"]["input"][0]["content"]
    assert outbound_content[0]["_request_metadata"] == "keep"
    assert outbound_content[1]["text"] == "Describe the clip."
    extra_body = json.loads(
        extra_env_info["responses_create_params"]["metadata"]["extra_body"]
    )
    assert extra_body["mm_processor_kwargs"] == {
        "video_as_images": True,
        "max_num_tiles": 1,
    }


def test_video_datum_requires_explicit_video_data_config(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": str(video_path)},
                    ],
                }
            ]
        }
    }

    with pytest.raises(ValueError, match=r"data\.num_frames"):
        nemo_gym_example_to_video_datum_spec(
            example,
            processor=object(),
            max_seq_length=128,
            idx=0,
            task_name="nemo_gym",
            data_config=TaskDataSpec(task_name="nemo_gym"),
        )


def test_recipe_video_defaults_reach_nemo_gym_data_processor(monkeypatch, tmp_path):
    data_path = tmp_path / "video-gym.jsonl"
    data_path.write_text(json.dumps({"row": 1}) + "\n", encoding="utf-8")
    expected_media_config = {
        "num_frames": 32,
        "video_sampling_style": "nemotron_vl",
        "video_target_num_patches": 64,
        "video_temporal_patch_size": 2,
        "video_maintain_aspect_ratio": True,
        "min_generation_tokens": 16,
    }
    data_config = {
        "max_input_seq_length": 128,
        "shuffle": False,
        "train": {
            "dataset_name": "NemoGymDataset",
            "data_path": str(data_path),
            "processor": "nemo_gym_data_processor",
        },
        "validation": None,
        "default": dict(expected_media_config),
    }
    captured = {}

    def fake_video_processor(
        example, *, processor, max_seq_length, idx, task_name, data_config
    ):
        del example, processor, max_seq_length
        captured["task_spec"] = data_config
        return {
            "message_log": [],
            "length": 0,
            "extra_env_info": {},
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": task_name,
        }

    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym_video.nemo_gym_example_to_video_datum_spec",
        fake_video_processor,
    )
    processor = SimpleNamespace(
        apply_chat_template=lambda *args: None,
        tokenizer=object(),
    )

    train_dataset, _ = setup_response_data(
        processor,
        data_config,
        env_configs=None,
        is_vlm=True,
    )

    train_dataset[0]
    task_spec = captured["task_spec"]
    assert {
        name: getattr(task_spec, name) for name in expected_media_config
    } == expected_media_config


def test_video_datum_uses_cached_frames_without_decoding_video(monkeypatch, tmp_path):
    frame_paths = []
    for index in range(4):
        frame_path = tmp_path / f"frame_{index:04d}.png"
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(frame_path)
        frame_paths.append(frame_path)
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "input_image",
                                "image_url": str(frame_path),
                                "_is_video_frame": True,
                                "_video_source": "/videos/clip.mp4",
                            }
                            for frame_path in frame_paths
                        ],
                        {"type": "input_text", "text": "Describe the clip."},
                    ],
                }
            ],
            "metadata": {
                "extra_body": json.dumps(
                    {
                        "chat_template_kwargs": {"enable_thinking": True},
                        "mm_processor_kwargs": {
                            "max_num_tiles": 1,
                            "video_as_images": True,
                        },
                    }
                )
            },
        }
    }
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym_video._video_to_image_content",
        lambda *args, **kwargs: pytest.fail("cached frames must not decode the video"),
    )

    class _Tokenizer:
        model_input_names = ["input_ids"]

    class _Processor:
        model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]
        tokenizer = _Tokenizer()

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            assert tokenize is True
            assert kwargs["video_flags"] == [True, True, True, True]
            assert kwargs["video_temporal_patch_size"] == 2
            assert kwargs["enable_thinking"] is True
            assert all(
                isinstance(part["image"], Image.Image)
                for part in messages[0]["content"][:4]
            )
            return {
                "input_ids": torch.tensor([[7, 18, 18, 9]]),
                "pixel_values": torch.ones(4, 3, 8, 8),
                "imgs_sizes": torch.tensor([[8, 8]] * 4),
            }

    datum = nemo_gym_example_to_video_datum_spec(
        example,
        processor=_Processor(),
        max_seq_length=None,
        idx=3,
        task_name="nemo_gym",
        data_config=SimpleNamespace(
            num_frames=4,
            video_sampling_style="nemotron_vl",
            video_temporal_patch_size=2,
            video_target_num_patches=64,
            video_maintain_aspect_ratio=True,
            min_generation_tokens=16,
        ),
    )

    assert datum is not None
    assert datum["message_log"][0]["num_frames"].as_tensor().tolist() == [4]
    outbound_content = datum["extra_env_info"]["responses_create_params"]["input"][0][
        "content"
    ]
    owned_metadata_keys = {
        "_is_video_frame",
        "_video_source",
        "_video_frame_index",
        "_video_fps",
    }
    assert all(
        owned_metadata_keys.isdisjoint(part)
        for part in outbound_content
        if isinstance(part, dict)
    )
    assert outbound_content[-1]["text"] == "Describe the clip."


def test_nemotron_video_datum_uses_dynamic_tubelet_inputs(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_video",
                            "video_url": str(video_path),
                        },
                        {"type": "input_text", "text": "Describe the clip."},
                    ],
                }
            ],
            "metadata": {
                "extra_body": json.dumps(
                    {"chat_template_kwargs": {"enable_thinking": False}}
                )
            },
        }
    }
    frames = np.zeros((4, 8, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym_video.load_video_frames_with_metadata",
        lambda *args, **kwargs: (
            frames,
            {"frames_indices": [0, 3, 6, 9], "fps": 3.0},
        ),
    )
    monkeypatch.setattr(
        "nemo_rl.environments.nemotron_utils.load_nemotron_video_model_config",
        lambda _model_name: SimpleNamespace(
            patch_size=16,
            downsample_ratio=0.5,
            norm_mean=[0.0, 0.0, 0.0],
            norm_std=[1.0, 1.0, 1.0],
        ),
    )

    class _Tokenizer:
        name_or_path = "nemotron-test"
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self):
            self.rendered_messages = None
            self.expanded_text = None

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            assert kwargs["enable_thinking"] is False
            self.rendered_messages = messages
            return messages[0]["content"] + "\nassistant"

        def __call__(self, text, **kwargs):
            assert kwargs == {
                "add_special_tokens": False,
                "return_tensors": "pt",
            }
            self.expanded_text = text
            token_count = text.count("<image>") + 4
            return {
                "input_ids": torch.arange(token_count).unsqueeze(0),
                "attention_mask": torch.ones(1, token_count, dtype=torch.long),
            }

    class NemotronNanoVLV2Processor:
        model_input_names = [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "imgs_sizes",
        ]

        def __init__(self):
            self.tokenizer = _Tokenizer()

    processor = NemotronNanoVLV2Processor()
    datum = nemo_gym_example_to_video_datum_spec(
        example,
        processor=processor,
        max_seq_length=256,
        idx=3,
        task_name="nemo_gym",
        data_config=SimpleNamespace(
            num_frames=4,
            video_sampling_style="nemotron_vl",
            video_temporal_patch_size=2,
            video_target_num_patches=64,
            video_maintain_aspect_ratio=True,
            min_generation_tokens=16,
        ),
    )

    assert datum is not None
    assert processor.tokenizer.rendered_messages[0]["content"] == (
        "<image>\n<image>\n<image>\n<image>\nDescribe the clip."
    )
    assert processor.tokenizer.expanded_text.count("<img>") == 2
    assert processor.tokenizer.expanded_text.count("</img>") == 2
    assert processor.tokenizer.expanded_text.count("<image>") == 30
    assert processor.tokenizer.expanded_text.startswith(
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.00 seconds: "
    )
    assert (
        "\nFrame 3 sampled at 2.00 seconds and frame 4 sampled at 3.00 seconds: "
        in processor.tokenizer.expanded_text
    )

    user_message = datum["message_log"][0]
    assert user_message["pixel_values"].as_tensor().shape == (4, 3, 96, 160)
    assert user_message["imgs_sizes"].as_tensor().tolist() == [[96, 160]] * 4
    assert user_message["num_frames"].as_tensor().tolist() == [4]
    extra_body = json.loads(
        datum["extra_env_info"]["responses_create_params"]["metadata"]["extra_body"]
    )
    assert "mm_processor_kwargs" not in extra_body


def test_nemotron_video_timestamps_match_vllm_integer_milliseconds():
    expanded = _expand_nemotron_video_placeholders(
        "<image>\n<image>\nquestion",
        embeddings_per_tubelet=[2],
        frame_indices=[0, 30],
        fps=29.97,
        temporal_patch_size=2,
    )

    assert expanded == (
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.99 seconds: "
        "<img><image><image></img>\nquestion"
    )


def test_nemotron_cached_video_uses_native_lossless_manifest(monkeypatch, tmp_path):
    frame_paths = []
    for index in range(4):
        frame_path = tmp_path / f"frame_{index:04d}.png"
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(frame_path)
        frame_paths.append(frame_path)
    example = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "input_image",
                                "image_url": str(frame_path),
                                "_is_video_frame": True,
                                "_video_source": "/videos/clip.mp4",
                            }
                            for frame_path in frame_paths
                        ],
                        {"type": "input_text", "text": "Describe the clip."},
                    ],
                }
            ],
            "metadata": {
                "extra_body": json.dumps(
                    {
                        "chat_template_kwargs": {"enable_thinking": True},
                        "mm_processor_kwargs": {
                            "max_num_tiles": 1,
                            "video_as_images": True,
                        },
                    }
                )
            },
        }
    }
    manifest_calls = []

    def fake_manifest_builder(paths):
        manifest_calls.append(paths)
        return "data:video/x-nemo-rl-cached-frames;base64,dGVzdA=="

    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym_video.build_cached_video_frame_data_url",
        fake_manifest_builder,
    )
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym_video.process_nemotron_video_frames",
        lambda *args, **kwargs: {
            "input_ids": torch.tensor([[7, 18, 18, 9]]),
            "pixel_values": torch.ones(4, 3, 8, 8),
            "imgs_sizes": torch.tensor([[8, 8]] * 4),
        },
    )

    class _Tokenizer:
        name_or_path = "nemotron-test"
        model_input_names = ["input_ids"]

    class NemotronNanoVLV2Processor:
        model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]
        tokenizer = _Tokenizer()

    datum = nemo_gym_example_to_video_datum_spec(
        example,
        processor=NemotronNanoVLV2Processor(),
        max_seq_length=None,
        idx=3,
        task_name="nemo_gym",
        data_config=SimpleNamespace(
            num_frames=4,
            video_temporal_patch_size=2,
            video_target_num_patches=64,
            video_maintain_aspect_ratio=True,
            min_generation_tokens=16,
        ),
    )

    assert datum is not None
    assert manifest_calls == [[str(path) for path in frame_paths]]
    outbound = datum["extra_env_info"]["responses_create_params"]
    assert outbound["input"][0]["content"] == [
        {
            "type": "input_video",
            "video_url": {"url": "data:video/x-nemo-rl-cached-frames;base64,dGVzdA=="},
        },
        {"type": "input_text", "text": "Describe the clip."},
    ]
    extra_body = json.loads(outbound["metadata"]["extra_body"])
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_extract_reward_components():
    assert extract_reward_components({"reward": 1.0}) is None
    assert extract_reward_components({"reward": 1.0, "reward_components": {}}) is None
    assert extract_reward_components(
        {
            "reward": 2.0,
            "reward_components": {"correctness": 1, "format": 0.5},
        }
    ) == {"correctness": 1.0, "format": 0.5}


def test_build_reward_component_columns():
    from nemo_rl.algorithms.utils import get_gdpo_reward_component_keys

    columns = build_reward_component_columns(
        [
            {"correctness": 1.0, "format": 0.0},
            {"correctness": 0.0, "format": 1.0},
        ]
    )
    assert set(columns) == {"reward/correctness", "reward/format"}
    assert torch.equal(columns["reward/correctness"], torch.tensor([1.0, 0.0]))
    assert torch.equal(columns["reward/format"], torch.tensor([0.0, 1.0]))

    columns = build_reward_component_columns([{"b": 2.0}, {"a": 1.0, "b": 3.0}, None])
    assert list(columns) == ["reward/a", "reward/b"]
    assert torch.equal(columns["reward/a"], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(columns["reward/b"], torch.tensor([2.0, 3.0, 0.0]))
    assert get_gdpo_reward_component_keys(columns) == ["reward/a", "reward/b"]
    assert build_reward_component_columns([None, None]) == {}


def test_validate_reward_components_match_scalar():
    validate_reward_components_match_scalar(
        [{"reward": 1.5, "reward_components": {"correctness": 1.0, "format": 0.5}}]
    )
    validate_reward_components_match_scalar(
        [
            {
                "reward": 1.5000001,
                "reward_components": {"correctness": 1.0, "format": 0.5},
            }
        ]
    )
    validate_reward_components_match_scalar([{"reward": 2.0}])
    with pytest.raises(ValueError, match="result 1"):
        validate_reward_components_match_scalar(
            [
                {
                    "reward": 1.5,
                    "reward_components": {"correctness": 1.0, "format": 0.5},
                },
                {
                    "reward": 2.0,
                    "reward_components": {"correctness": 1.0, "format": 0.5},
                },
            ]
        )


_TEST_FINAL_POLICY_DECISION = {
    "policy_name": "recency",
    "policy_version": "1",
    "config_digest": "policy-config",
    "retained_part_count": 1,
    "omitted_part_count": 0,
    "lineage": {
        "transformation_id": "transform-final",
        "transformation_type": "visual_recency",
        "transformation_version": "1",
        "configuration_digest": "policy-config",
        "deterministic": True,
        "lossy": False,
        "generator_contract_id": None,
        "unit_records": [
            {
                "source_unit_id": "part-final",
                "source_digest": "digest-final",
                "disposition": "kept",
                "output_unit_ids": ["part-final"],
                "output_digests": ["digest-final"],
            }
        ],
        "validator_result": "passed",
    },
}


def test_actor_peak_rss_gib_converts_linux_kib(monkeypatch):
    peak_rss_gib = 3.25
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.resource.getrusage",
        lambda _: SimpleNamespace(ru_maxrss=peak_rss_gib * 1024**2),
    )

    assert _actor_peak_rss_gib() == peak_rss_gib


def _test_lineage_deltas(num_turns: int) -> list[dict]:
    final_record = _TEST_FINAL_POLICY_DECISION["lineage"]["unit_records"][0]
    payload = json.dumps(
        [final_record],
        sort_keys=True,
        separators=(",", ":"),
    )
    state_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    parent = None
    deltas = []
    for turn_id in range(1, num_turns + 1):
        transformation_id = f"transform-{turn_id}"
        deltas.append(
            {
                "transformation_id": transformation_id,
                "parent_transformation_id": parent,
                "transformation_type": "visual_recency",
                "transformation_version": "1",
                "configuration_digest": "policy-config",
                "deterministic": True,
                "lossy": False,
                "generator_contract_id": None,
                "unit_upserts": ([final_record] if turn_id == 1 else []),
                "source_unit_count": 1,
                "state_digest": state_digest,
                "validator_result": "passed",
            }
        )
        parent = transformation_id
    return deltas


def _exact_evidence_contract_fields(
    *,
    turn_id: int,
    segment_index: int,
    media_ids: list[str],
    expected_append_compatible: bool,
    compaction_event_id: str | None = None,
) -> dict:
    action_id = f"action-{turn_id}"
    model_call_id = f"model-call-{turn_id}"
    return {
        "prepared_request_id": f"prepared-{turn_id}",
        "request_id": f"request-{turn_id}",
        "context_epoch": segment_index,
        "segment_index": segment_index,
        "segment_id": f"segment-{segment_index}",
        "expected_append_compatible": expected_append_compatible,
        "compaction_event_id": compaction_event_id,
        "policy_decision": {
            "policy_name": "recency",
            "policy_version": "1",
            "config_digest": "policy-config",
            "decision_turn": turn_id,
            "selection_digest": f"selection-{turn_id}",
            "transformation_id": f"transform-{turn_id}",
        },
        "policy_output_spans": [
            {
                "policy_output_span_id": f"span-{turn_id}",
                "model_call_id": model_call_id,
                "action_ids": [action_id],
                "start": 0,
                "end": 1,
                "eligible": turn_id == 1,
                "old_logprobs_alignment": "sampled_tokens",
            }
        ],
        "media_occurrences": [
            {
                "media_id": media_id,
                "occurrence_ordinal": ordinal,
                "model_call_id": model_call_id,
                "placeholder_span_or_position": None,
                "processed_dimensions": None,
                "model_specific_sidecars": {},
            }
            for ordinal, media_id in enumerate(media_ids)
        ],
    }


@pytest.mark.nemo_gym
def test_nemo_gym_stub_module():
    from nemo_gym import config_types

    print(
        f"NeMo-Gym test successfully run! NeMo-Gym config_types module: {config_types}"
    )


@pytest.fixture(scope="function")
def nemo_gym_vllm_generation(cluster, nemo_gym_tokenizer):  # noqa: F811
    generation_config = deepcopy(basic_vllm_test_config)
    master_config = MasterConfig.model_construct(
        policy={"generation": generation_config}
    )
    setup_nemo_gym_config(master_config, nemo_gym_tokenizer)

    generation_config["vllm_cfg"]["max_model_len"] = 16_384
    # This is the tool parser for Qwen/Qwen3-0.6B. This needs to be changed for other models.
    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "enable_auto_tools": True,
        "tool_parser": "hermes",
    }

    vllm_generation = VllmGeneration(cluster, generation_config)

    yield vllm_generation

    vllm_generation.shutdown()


@pytest.fixture(scope="function")
def nemo_gym(nemo_gym_vllm_generation, nemo_gym_tokenizer):  # noqa: F811
    """Create a NeMo-Gym actor for testing."""

    yaml_str = r"""example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py
      domain: instruction_following
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: openai_model
openai_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: true
      uses_reasoning_parser: true
rollout_max_attempts_to_avoid_lp_nan: 1
"""

    config = NemoGymConfig(
        model_name=nemo_gym_vllm_generation.cfg["model_name"],
        base_urls=nemo_gym_vllm_generation.dp_openai_server_base_urls,
        initial_global_config_dict=safe_load(yaml_str),
    )
    env = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.nemo_gym.NemoGym"
            ),
        }
    ).remote(config)

    # Blocking wait for NeMo-Gym to spin up
    ray.get(env._spinup.remote())
    # Install the tokenizer here, as spinup_nemo_gym_actor does, so the fixture
    # yields an actor that can actually run rollouts. Tests reaching the actor
    # through RolloutManager never see set_tokenizer themselves.
    ray.get(env.set_tokenizer.remote(nemo_gym_tokenizer))

    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture(scope="function")
def nemo_gym_sanity_test_data():
    fpath = Path(__file__).parent / "nemo_gym_test_data/test_nemo_gym_sanity.json"
    with open(fpath) as f:
        data = json.load(f)
    return data


def _write_actual_test_data(original_input: list, actual_result: list):
    """Write actual rollout results to actual_test_nemo_gym_sanity.json.

    This makes it easy to update the expected output after a Gym commit bump:
        cp nemo_gym_test_data/actual_test_nemo_gym_sanity.json nemo_gym_test_data/test_nemo_gym_sanity.json
    """

    def _convert(obj):
        """Recursively convert torch tensors to Python lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    cleaned = deepcopy(actual_result)
    for r in cleaned:
        r.pop("full_result", None)
        for msg in r.get("message_log", [])[1:]:
            if "token_ids" in msg:
                msg["token_ids"] = []
            if "generation_logprobs" in msg:
                msg["generation_logprobs"] = []

    output_path = (
        Path(__file__).parent / "nemo_gym_test_data/actual_test_nemo_gym_sanity.json"
    )
    data = _convert({"input": original_input, "expected_output": cleaned})
    with open(output_path, "w") as f:
        json.dump(data, f)
        f.write("\n")
    print(f"Wrote updated test data to {output_path}")


def test_run_rollouts_requires_an_installed_tokenizer():
    """run_rollouts reads the tokenizer off the actor, so reaching it without one fails.

    Every call site installs it via spinup, so this is unreachable in practice -- but
    silently postprocessing with no tokenizer is worse than a named error, and a future
    spinup path that forgets the call should say so here rather than deeper in.
    """
    gym_cls = NemoGym.__ray_metadata__.modified_class
    # Constructed through __init__ rather than object.__new__ so the None comes from
    # the declaration itself: an attribute set only in _spinup would leave a second
    # spinup free to wipe an installed tokenizer.
    gym = gym_cls({})
    assert gym._tokenizer is None
    gym.rh = object()  # satisfies _require_spinup

    stream = gym.run_rollouts([{"_rowidx": 0}], "")
    with pytest.raises(RuntimeError, match="set_tokenizer must be called"):
        asyncio.run(stream.__anext__())


def test_nemo_gym_postprocess_uses_batch_decode():
    class _Tokenizer:
        def __init__(self):
            self.batch_decode_calls = []

        def batch_decode(self, batch):
            self.batch_decode_calls.append([list(token_ids) for token_ids in batch])
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    tokenizer = _Tokenizer()
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            {
                "_nemo_rl_rollout_id": "rollout-1",
                "_nemo_rl_group_id": "group-1",
            },
            nemo_gym_result,
            tokenizer,
        )
    )

    assert tokenizer.batch_decode_calls == [
        [[1, 2], [1, 2, 3, 4, 5]],
        [[3], [6, 7]],
    ]
    assert result["message_log"][0]["token_ids"].tolist() == [1, 2]
    assert result["message_log"][1]["token_ids"].tolist() == [3]
    assert result["message_log"][2]["token_ids"].tolist() == [4, 5]
    assert result["message_log"][3]["token_ids"].tolist() == [6, 7]
    assert len(result["physical_message_logs"]) == 1
    assert result["rollout_id"] == "rollout-1"
    assert result["group_id"] == "group-1"
    assert result["generation_policy_version"] is None
    assert result["physical_trace_ids"] == ["rollout-1:trace-000000"]
    assert result["message_log"][0]["token_loss_mask"].tolist() == [0, 0]
    assert result["message_log"][1]["token_loss_mask"].tolist() == [1]
    assert "rollout_trace_bundle" not in result
    assert nemo_gym_result["response"]["output"][0]["prompt_str"] == "1 2"
    assert nemo_gym_result["response"]["output"][0]["generation_str"] == "3"
    assert nemo_gym_result["response"]["output"][1]["prompt_str"] == "1 2 3 4 5"
    assert nemo_gym_result["response"]["output"][1]["generation_str"] == "6 7"


@pytest.mark.parametrize("include_initial_multimodal_data", [False, True])
def test_nemo_gym_dedup_redacts_initial_images_from_actor_return(
    include_initial_multimodal_data,
):
    data_url = image_to_data_url(Image.new("RGB", (2, 2), color="red"))
    initial_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "count"},
                {"type": "input_image", "image_url": data_url},
            ],
        }
    ]
    nemo_gym_result = {
        "response": {
            "agent_input": deepcopy(initial_input),
            "seed_obs": deepcopy(initial_input),
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                }
            ],
        },
        "responses_create_params": {"input": deepcopy(initial_input)},
        "reward": 1.0,
    }

    class _Tokenizer:
        def batch_decode(self, batch):
            return ["decoded"] * len(batch)

    class _MockSelf:
        cfg = {}
        _processor = None

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            _caller_identity_row(),
            nemo_gym_result,
            _Tokenizer(),
            include_initial_multimodal_data=include_initial_multimodal_data,
        )
    )

    if include_initial_multimodal_data:
        assert "_initial_multimodal_data_omitted" not in result
        assert data_url in json.dumps(result["full_result"])
    else:
        assert result["_initial_multimodal_data_omitted"] is True
        assert data_url not in json.dumps(result["full_result"])
        assert result["full_result"]["responses_create_params"]["input"][0][
            "content"
        ] == [{"type": "input_text", "text": "count"}]


def test_nemo_gym_dedup_omits_actor_initial_tensor_and_preserves_later_media():
    initial_url = image_to_data_url(Image.new("RGB", (1, 1), color=(1, 0, 0)))
    tool_url = image_to_data_url(Image.new("RGB", (1, 1), color=(2, 0, 0)))
    initial_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "inspect"},
                {"type": "input_image", "image_url": initial_url},
            ],
        }
    ]
    template = {
        "response": {
            "agent_input": deepcopy(initial_input),
            "seed_obs": deepcopy(initial_input),
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": tool_url},
                    ],
                },
                {
                    "prompt_token_ids": [1, 2, 3],
                    "generation_token_ids": [4],
                    "generation_log_probs": [-0.2],
                },
            ],
        },
        "responses_create_params": {"input": deepcopy(initial_input)},
        "reward": 1.0,
    }

    class _Tokenizer:
        def batch_decode(self, batch):
            return ["decoded"] * len(batch)

    class _ImageProcessor:
        model_input_names = ["pixel_values"]

    class _TextTokenizer:
        model_input_names = ["input_ids"]

    class _Processor:
        image_token = "<image>"
        image_processor = _ImageProcessor()
        tokenizer = _TextTokenizer()
        model_input_names = ["input_ids", "pixel_values"]

        def __call__(self, *, text, images, return_tensors):
            assert text == "<image>" * len(images)
            assert return_tensors == "pt"
            red_values = [image.getpixel((0, 0))[0] for image in images]
            return {
                "input_ids": torch.tensor([[1]]),
                "pixel_values": torch.tensor(red_values, dtype=torch.float32).view(
                    -1, 1
                ),
            }

    class _MockSelf:
        cfg = {}
        _processor = _Processor()

    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    flag_off = postprocess(
        _MockSelf(),
        _caller_identity_row(),
        deepcopy(template),
        _Tokenizer(),
        include_initial_multimodal_data=True,
    )
    flag_on = postprocess(
        _MockSelf(),
        _caller_identity_row(),
        deepcopy(template),
        _Tokenizer(),
        include_initial_multimodal_data=False,
    )

    off_users = [
        message for message in flag_off["message_log"] if message["role"] == "user"
    ]
    on_users = [
        message for message in flag_on["message_log"] if message["role"] == "user"
    ]
    assert off_users[0]["pixel_values"].as_tensor().item() == 1
    assert off_users[1]["pixel_values"].as_tensor().item() == 2
    assert "pixel_values" not in on_users[0]
    assert on_users[1]["pixel_values"].as_tensor().item() == 2
    assert initial_url not in json.dumps(flag_on["full_result"])
    assert tool_url in json.dumps(flag_on["full_result"])

    original_media = PackedTensor(torch.tensor([[99.0]]), dim_to_pack=0)
    _reattach_original_multimodal_payloads(
        [flag_on],
        [[{"role": "user", "content": "", "pixel_values": original_media}]],
    )
    on_users = [
        message for message in flag_on["message_log"] if message["role"] == "user"
    ]
    assert on_users[0]["pixel_values"] is original_media
    assert on_users[1]["pixel_values"].as_tensor().item() == 2


@pytest.mark.parametrize(
    ("seed_mode", "expected_pixel_values"),
    [
        ("text_only", None),
        ("initial_plus_additional", [1.0, 2.0]),
    ],
)
def test_nemo_gym_dedup_keeps_authoritative_changed_seed_media(
    seed_mode, expected_pixel_values
):
    initial_url = image_to_data_url(Image.new("RGB", (1, 1), color=(1, 0, 0)))
    additional_url = image_to_data_url(Image.new("RGB", (1, 1), color=(2, 0, 0)))
    initial_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "inspect"},
                {"type": "input_image", "image_url": initial_url},
            ],
        }
    ]
    if seed_mode == "text_only":
        seed_obs = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "text only"}],
            }
        ]
    else:
        seed_obs = deepcopy(initial_input)
        seed_obs[0]["content"].append(
            {"type": "input_image", "image_url": additional_url}
        )

    nemo_gym_result = {
        "response": {
            "agent_input": deepcopy(initial_input),
            "seed_obs": seed_obs,
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                }
            ],
        },
        "responses_create_params": {"input": deepcopy(initial_input)},
        "reward": 1.0,
    }

    class _Tokenizer:
        def batch_decode(self, batch):
            return ["decoded"] * len(batch)

    class _ImageProcessor:
        model_input_names = ["pixel_values"]

    class _TextTokenizer:
        model_input_names = ["input_ids"]

    class _Processor:
        image_token = "<image>"
        image_processor = _ImageProcessor()
        tokenizer = _TextTokenizer()
        model_input_names = ["input_ids", "pixel_values"]

        def __call__(self, *, text, images, return_tensors):
            assert text == "<image>" * len(images)
            assert return_tensors == "pt"
            red_values = [image.getpixel((0, 0))[0] for image in images]
            return {
                "input_ids": torch.tensor([[1]]),
                "pixel_values": torch.tensor(red_values, dtype=torch.float32).view(
                    -1, 1
                ),
            }

    class _MockSelf:
        cfg = {}
        _processor = _Processor()

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            _caller_identity_row(),
            nemo_gym_result,
            _Tokenizer(),
            include_initial_multimodal_data=False,
        )
    )

    assert result["_initial_multimodal_data_omitted"] is False
    user_message = next(
        message for message in result["message_log"] if message["role"] == "user"
    )
    if expected_pixel_values is None:
        assert "pixel_values" not in user_message
    else:
        assert user_message["pixel_values"].as_tensor().flatten().tolist() == (
            expected_pixel_values
        )

    original_media = PackedTensor(torch.tensor([[99.0]]), dim_to_pack=0)
    _reattach_original_multimodal_payloads(
        [result],
        [[{"role": "user", "content": "", "pixel_values": original_media}]],
    )
    if expected_pixel_values is None:
        assert "pixel_values" not in user_message
    else:
        assert user_message["pixel_values"].as_tensor().flatten().tolist() == (
            expected_pixel_values
        )


def test_nemo_gym_run_rollouts_normalizes_mixed_media_before_dispatch(tmp_path):
    video_path = tmp_path / "clip with spaces.mp4"
    video_path.write_bytes(b"video")
    image_path = tmp_path / "still.png"
    Image.new("RGB", (2, 2)).save(image_path)

    async def _run():
        nemo_gym_row = {
            "_rowidx": 7,
            "agent_ref": {"name": "test_agent"},
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_video",
                                "video_url": str(video_path),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": str(image_path)},
                            },
                        ],
                    }
                ]
            },
        }
        nemo_gym_result = {"response": {"output": []}}
        tokenizer = object()
        postprocess_calls = []

        class _RolloutCollectionHelper:
            def run_examples(self, examples, head_server_config):
                del head_server_config
                content = examples[0]["responses_create_params"]["input"][0]["content"]
                assert content[0]["video_url"] == video_path.resolve().as_uri()
                assert content[1]["image_url"].startswith("data:image/png;base64,")

                async def _completed_result():
                    return nemo_gym_row, nemo_gym_result

                return [_completed_result()]

        class _MockSelf:
            cfg = {}
            rch = _RolloutCollectionHelper()
            head_server_config = object()
            _rollout_batch_index = 0

            def _require_spinup(self):
                pass

            def _postprocess_nemo_gym_to_nemo_rl_result(
                self,
                row,
                result,
                result_tokenizer,
                *,
                include_initial_multimodal_data,
                generation_only,
            ):
                del self
                postprocess_calls.append(
                    (
                        row,
                        result,
                        result_tokenizer,
                        include_initial_multimodal_data,
                        generation_only,
                    )
                )
                return {"message_log": []}

        mock_self = _MockSelf()
        mock_self._tokenizer = tokenizer
        streamed_results = []
        async for result in NemoGym.__ray_metadata__.modified_class.run_rollouts(
            mock_self, [nemo_gym_row], "test"
        ):
            streamed_results.append(result)

        assert postprocess_calls == [
            (nemo_gym_row, nemo_gym_result, tokenizer, True, False)
        ]
        assert streamed_results[0][0] == 7
        assert streamed_results[0][1] == {"message_log": []}

    asyncio.run(_run())


def test_nemo_gym_postprocess_no_generation_data_raises():
    class _Tokenizer:
        def apply_chat_template(self, input_messages, tokenize=True):
            return list(range(1234))

    nemo_gym_result = {
        "response": {
            "output": [
                {"type": "reasoning"},
                {"type": "function_call"},
            ]
        },
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
    }

    class _MockSelf:
        cfg = {}

    with pytest.raises(ValueError) as excinfo:
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            {
                "_nemo_rl_rollout_id": "rollout-1",
                "_nemo_rl_group_id": "group-1",
            },
            nemo_gym_result,
            _Tokenizer(),
        )

    message = str(excinfo.value)
    assert "no generation data" in message
    assert "1234 tokens" in message
    assert "['reasoning', 'function_call']" in message


def test_nemo_gym_postprocess_no_generation_data_chat_template_failure():
    class _Tokenizer:
        def apply_chat_template(self, input_messages, tokenize=True):
            raise RuntimeError("boom")

    nemo_gym_result = {
        "response": {"output": [{"type": "reasoning"}]},
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
    }

    class _MockSelf:
        cfg = {}

    with pytest.raises(ValueError) as excinfo:
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            {
                "_nemo_rl_rollout_id": "rollout-1",
                "_nemo_rl_group_id": "group-1",
            },
            nemo_gym_result,
            _Tokenizer(),
        )

    message = str(excinfo.value)
    assert "no generation data" in message
    assert "apply_chat_template failed" in message
    assert "RuntimeError" in message
    assert "['reasoning']" in message


def test_nemo_gym_postprocess_rejects_undeclared_rewrite():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    def make_result():
        return {
            "response": {
                "output": [
                    {
                        "prompt_token_ids": [1, 2],
                        "generation_token_ids": [3],
                        "generation_log_probs": [-0.1],
                    },
                    {
                        "prompt_token_ids": [1, 4],
                        "generation_token_ids": [5],
                        "generation_log_probs": [-0.2],
                    },
                ]
            },
            "responses_create_params": {"input": []},
        }

    class _MockSelf:
        cfg = {}

    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    row = {
        "_rowidx": 0,
        "_nemo_rl_group_id": "group-1",
        "_nemo_rl_rollout_id": "rollout-1",
        "_nemo_rl_generation_policy_version": "sync-policy-step-00000000",
    }
    with pytest.raises(ValueError, match="undeclared prompt/media discontinuity"):
        postprocess(_MockSelf(), row, make_result(), _Tokenizer())


def test_nemo_gym_postprocess_accepts_completion_evidence_without_contract():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    nemo_gym_result = {
        "response": {
            "completion_evidence": [
                {
                    "turn_id": 1,
                    "completion_id": "completion-1",
                    "prompt_token_ids": [1, 2],
                    "sampled_token_ids": [3],
                    "sampled_logprobs": [-0.1],
                    "media_ids": [],
                    "finish_reason": "max_output_tokens",
                    "eligible": True,
                }
            ],
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                }
            ],
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            {
                "_nemo_rl_rollout_id": "rollout-1",
                "_nemo_rl_group_id": "group-1",
            },
            nemo_gym_result,
            _Tokenizer(),
        )
    )

    assert [message["token_ids"].tolist() for message in result["message_log"]] == [
        [1, 2],
        [3],
    ]
    assert result["truncated"] is True


def test_nemo_gym_postprocess_builds_exact_compacted_physical_logs():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    boundary = {
        "event_id": "boundary-2",
        "applies_to_step": 2,
        "reason": "history_policy_rewrite",
        "policy_name": "recency",
        "policy_version": "1",
        "config_digest": "policy-config",
    }
    rollout_id = "group-cc:batch-000000:row-000003"
    result_payload = {
        "response": {
            "context_compaction_contract": {
                "schema_version": 2,
                "mode": "exact_trace_authority",
                "rollout_id": rollout_id,
                "group_id": "group-cc",
                "task_id": "task-cc",
                "rollout_index": 3,
                "attempt_index": 0,
            },
            "media_assets": {
                "screen-a": {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,A",
                },
                "screen-b": {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,B",
                },
            },
            "final_policy_decision": _TEST_FINAL_POLICY_DECISION,
            "lineage_deltas": _test_lineage_deltas(2),
            "boundary_events": [boundary],
            "completion_evidence": [
                {
                    "rollout_id": rollout_id,
                    "turn_id": 1,
                    "completion_id": "completion-1",
                    "action_id": "action-1",
                    "prompt_token_ids": [1],
                    "sampled_token_ids": [2],
                    "sampled_logprobs": [-0.1],
                    "media_ids": ["screen-a"],
                    "policy_decision": {
                        "policy_name": "recency",
                        "policy_version": "1",
                        "config_digest": "policy-config",
                    },
                    "finish_reason": "stop",
                    "eligible": True,
                    "evidence_source": "generation_response",
                    **_exact_evidence_contract_fields(
                        turn_id=1,
                        segment_index=0,
                        media_ids=["screen-a"],
                        expected_append_compatible=False,
                    ),
                },
                {
                    "rollout_id": rollout_id,
                    "turn_id": 2,
                    "completion_id": "completion-2",
                    "action_id": "action-2",
                    "prompt_token_ids": [8],
                    "sampled_token_ids": [9],
                    "sampled_logprobs": [-0.2],
                    "media_ids": ["screen-a", "screen-b"],
                    "policy_decision": {
                        "policy_name": "recency",
                        "policy_version": "1",
                        "config_digest": "policy-config",
                    },
                    "finish_reason": "stop",
                    "eligible": False,
                    "evidence_source": "generation_response",
                    **_exact_evidence_contract_fields(
                        turn_id=2,
                        segment_index=1,
                        media_ids=["screen-a", "screen-b"],
                        expected_append_compatible=False,
                        compaction_event_id="boundary-2",
                    ),
                },
            ],
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [8],
                    "generation_token_ids": [9],
                    "generation_log_probs": [-0.2],
                },
            ],
        },
        "responses_create_params": {"input": []},
        "reward": 0.75,
    }
    row = {
        "_rowidx": 3,
        "_nemo_rl_rollout_id": rollout_id,
        "_nemo_rl_group_id": "group-cc",
        "context_compaction_rollout_id": rollout_id,
        "context_compaction_group_id": "group-cc",
        "context_compaction_task_id": "task-cc",
        "context_compaction_rollout_index": 3,
        "context_compaction_attempt_index": 0,
        "_nemo_rl_generation_policy_version": "sync-policy-step-00000000",
    }

    class _MockSelf:
        cfg = {}

    training_result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            row,
            deepcopy(result_payload),
            _Tokenizer(),
            generation_only=False,
        )
    )
    assert [
        [message["token_ids"].tolist() for message in physical_trace]
        for physical_trace in training_result["physical_message_logs"]
    ] == [[[1], [2]], [[8], [9]]]
    assert (
        training_result["message_log"][0]
        is not training_result["physical_message_logs"][0][0]
    )
    assert (
        training_result["message_log"][1]
        is not training_result["physical_message_logs"][0][1]
    )
    assert training_result["rollout_id"] == rollout_id
    assert training_result["group_id"] == "group-cc"
    assert training_result["generation_policy_version"] == "sync-policy-step-00000000"
    assert training_result["physical_trace_ids"] == [
        f"{rollout_id}:trace-000000",
        f"{rollout_id}:trace-000001",
    ]
    assert training_result["physical_message_logs"][0][1][
        "token_loss_mask"
    ].tolist() == [1]
    assert training_result["physical_message_logs"][1][1][
        "token_loss_mask"
    ].tolist() == [0]
    assert "rollout_trace_bundle" not in training_result

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            row,
            result_payload,
            _Tokenizer(),
            generation_only=True,
        )
    )

    assert "nemo_rl_trace_bundle" not in result_payload
    assert "rollout_trace_bundle" not in result
    assert "nemo_rl_trace_bundle" not in result["full_result"]
    assert (
        result["full_result"]["context_compaction_gym_http_bytes"]
        > (result["full_result"]["context_compaction_ray_env_extras_bytes"])
    )
    assert result["full_result"][
        "context_compaction_ray_env_extras_bytes"
    ] == _compact_json_size(result["full_result"])
    assert (
        0.0
        < result["full_result"]["context_compaction_transport_reduction_ratio"]
        < 1.0
    )
    projected_response = result["full_result"]["response"]
    assert set(projected_response).isdisjoint(
        {
            "agent_input",
            "seed_obs",
            "media_assets",
            "completion_evidence",
            "final_policy_decision",
            "lineage_deltas",
        }
    )


def test_nemo_gym_postprocess_exact_authority_rejects_missing_evidence():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    rollout_id = "group-cc:batch-000000:row-000000"
    payload = {
        "response": {
            "context_compaction_contract": {
                "schema_version": 2,
                "mode": "exact_trace_authority",
                "rollout_id": rollout_id,
                "group_id": "group-cc",
                "task_id": "task-cc",
                "rollout_index": 0,
                "attempt_index": 0,
            },
            "media_assets": {},
            "completion_evidence": [],
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                }
            ],
        },
        "responses_create_params": {"input": []},
    }
    row = {
        "_rowidx": 0,
        "_nemo_rl_rollout_id": rollout_id,
        "_nemo_rl_group_id": "group-cc",
        "context_compaction_rollout_id": rollout_id,
        "context_compaction_group_id": "group-cc",
        "context_compaction_task_id": "task-cc",
        "context_compaction_rollout_index": 0,
        "context_compaction_attempt_index": 0,
    }

    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    with pytest.raises(ValueError, match="missing completion_evidence"):
        postprocess(
            type("_MockSelf", (), {"cfg": {}})(),
            row,
            payload,
            _Tokenizer(),
            generation_only=True,
        )


def test_nemo_gym_postprocess_rejects_mismatched_generation_evidence():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    rollout_id = "group-cc:batch-000000:row-000000"
    payload = {
        "response": {
            "context_compaction_contract": {
                "schema_version": 2,
                "mode": "exact_trace_authority",
                "rollout_id": rollout_id,
                "group_id": "group-cc",
                "task_id": "task-cc",
                "rollout_index": 0,
                "attempt_index": 0,
            },
            "media_assets": {},
            "completion_evidence": [
                {
                    "rollout_id": rollout_id,
                    "turn_id": 1,
                    "completion_id": "completion-1",
                    "action_id": "action-1",
                    "prompt_token_ids": [99],
                    "sampled_token_ids": [2],
                    "sampled_logprobs": [-0.1],
                    "media_ids": [],
                    **_exact_evidence_contract_fields(
                        turn_id=1,
                        segment_index=0,
                        media_ids=[],
                        expected_append_compatible=False,
                    ),
                }
            ],
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                }
            ],
        },
        "responses_create_params": {"input": []},
    }
    row = {
        "_rowidx": 0,
        "_nemo_rl_rollout_id": rollout_id,
        "_nemo_rl_group_id": "group-cc",
        "context_compaction_rollout_id": rollout_id,
        "context_compaction_group_id": "group-cc",
        "context_compaction_task_id": "task-cc",
        "context_compaction_rollout_index": 0,
        "context_compaction_attempt_index": 0,
    }

    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        postprocess(
            type("_MockSelf", (), {"cfg": {}})(),
            row,
            payload,
            _Tokenizer(),
            generation_only=True,
        )


def test_context_compaction_rollout_ids_are_unique_within_and_across_batches():
    rows = [
        {
            "_rowidx": row_index,
            "context_compaction_contract_version": 1,
            "context_compaction_group_id": "group-cc",
        }
        for row_index in range(2)
    ]
    _stamp_nemo_gym_rollout_ids(
        rows, rollout_batch_index=4, num_generations_per_prompt=2
    )

    assert [row["context_compaction_rollout_id"] for row in rows] == [
        "group-cc:batch-000004:row-000000",
        "group-cc:batch-000004:row-000001",
    ]

    next_batch = [dict(rows[0])]
    _stamp_nemo_gym_rollout_ids(
        next_batch, rollout_batch_index=5, num_generations_per_prompt=1
    )
    assert next_batch[0]["context_compaction_rollout_id"] == (
        "group-cc:batch-000005:row-000000"
    )


def test_policy_version_supports_rows_without_context_compaction_contract():
    rows = [{"_rowidx": 0}]

    _stamp_nemo_gym_rollout_ids(
        rows,
        rollout_batch_index=0,
        num_generations_per_prompt=1,
        generation_policy_version="sync-policy-step-00000000",
    )

    assert rows[0]["_nemo_rl_group_id"] == "nemo-gym-batch-000000:group-000000"
    assert rows[0]["_nemo_rl_rollout_id"].endswith("rollout-000000")
    assert rows[0]["_nemo_rl_generation_policy_version"] == "sync-policy-step-00000000"


def test_v2_context_compaction_rollout_ids_are_retry_and_order_stable():
    rows = [
        {
            "_rowidx": 19,
            "context_compaction_contract_version": 2,
            "context_compaction_group_id": "group-cc",
            "context_compaction_task_id": task_id,
            "context_compaction_rollout_index": rollout_index,
            "context_compaction_attempt_index": 0,
        }
        for task_id, rollout_index in (("task-a", 0), ("task-b", 1))
    ]
    reordered = [dict(rows[1]), dict(rows[0])]

    _stamp_nemo_gym_rollout_ids(
        rows, rollout_batch_index=4, num_generations_per_prompt=1
    )
    _stamp_nemo_gym_rollout_ids(
        reordered, rollout_batch_index=99, num_generations_per_prompt=1
    )

    by_task = {
        row["context_compaction_task_id"]: row["context_compaction_rollout_id"]
        for row in rows
    }
    reordered_by_task = {
        row["context_compaction_task_id"]: row["context_compaction_rollout_id"]
        for row in reordered
    }
    assert by_task == reordered_by_task
    assert len(set(by_task.values())) == 2

    new_attempt = [dict(rows[0], context_compaction_attempt_index=1)]
    _stamp_nemo_gym_rollout_ids(
        new_attempt, rollout_batch_index=4, num_generations_per_prompt=1
    )
    assert new_attempt[0]["context_compaction_rollout_id"] != by_task["task-a"]


def test_index_per_turn_images_preserves_initial_and_observation_order():
    image_a = Image.new("RGB", (2, 2), "red")
    image_b = Image.new("RGB", (2, 2), "green")
    image_c = Image.new("RGB", (2, 2), "blue")
    initial_input = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image": image_a}],
        }
    ]
    seed_obs = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image": image_b}],
        }
    ]
    output = [
        {"role": "assistant", "generation_token_ids": [1]},
        {"role": "user", "content": [{"type": "input_text", "text": "none"}]},
        {"role": "assistant", "generation_token_ids": [2]},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image": image_c},
                {"type": "input_image", "image": image_a},
            ],
        },
        {"role": "assistant", "generation_token_ids": [3]},
    ]

    per_turn = _index_per_turn_images(
        [*seed_obs, *output],
        input_messages=initial_input,
    )

    assert per_turn == [[image_a, image_b], [], [image_c, image_a]]


def test_media_arena_order_is_preserved_in_processor_packed_tensors():
    image_a = Image.new("RGB", (2, 2), (10, 20, 30))
    image_b = Image.new("RGB", (2, 2), (40, 50, 60))
    media_assets = {
        "image-a": {"type": "input_image", "image": image_a},
        "image-b": {
            "media_id": "image-b",
            "content_digest": "digest-b",
            "source_part": {"type": "input_image", "image": image_b},
            "original_dimensions": (2, 2),
            "color_mode": "RGB",
            "source_format": "png",
        },
    }
    images = _resolve_images_by_media_id(
        media_assets,
        ["image-b", "image-a", "image-b"],
    )

    class _Tokenizer:
        model_input_names = ["input_ids"]

    class _Processor:
        image_token = "<image>"
        model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]
        tokenizer = _Tokenizer()

        def __init__(self):
            self.observed_colors = []

        def __call__(self, *, text, images, return_tensors):
            assert text == "<image><image><image>"
            assert return_tensors == "pt"
            self.observed_colors = [image.getpixel((0, 0)) for image in images]
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.tensor(self.observed_colors),
                "imgs_sizes": torch.tensor([[2, 2]] * len(images)),
            }

    processor = _Processor()
    user_message = {"role": "user", "content": "", "token_ids": torch.tensor([1])}
    _attach_multimodal_data_to_user_message(
        user_message,
        images=images,
        processor=processor,
    )

    assert processor.observed_colors == [
        (40, 50, 60),
        (10, 20, 30),
        (40, 50, 60),
    ]
    assert user_message["pixel_values"].tensors[0].tolist() == [
        [40, 50, 60],
        [10, 20, 30],
        [40, 50, 60],
    ]
    assert user_message["imgs_sizes"].tensors[0].dtype == torch.int32
    assert user_message["num_frames"].tensors[0].tolist() == [1, 1, 1]


@pytest.mark.nemo_gym
def test_nemo_gym_sanity(
    nemo_gym,
    nemo_gym_sanity_test_data,
    nemo_gym_vllm_generation,
):
    """Test basic functionality of MathEnvironment step with simple messages."""

    # Save original input before mutation for writing the actual test data file
    original_input = deepcopy(nemo_gym_sanity_test_data["input"])

    # We need to match NeMo RL generation config params before sending to NeMo-Gym
    generation_config = nemo_gym_vllm_generation.cfg
    examples = nemo_gym_sanity_test_data["input"]
    for idx, example in enumerate(examples):
        example["responses_create_params"]["temperature"] = generation_config[
            "temperature"
        ]
        example["responses_create_params"]["top_p"] = generation_config["top_p"]
        example["_rowidx"] = idx

    actual_result = [None] * len(nemo_gym_sanity_test_data["input"])
    for result_ref in nemo_gym.run_rollouts.options(num_returns="streaming").remote(
        nemo_gym_sanity_test_data["input"], ""
    ):
        rowidx, result, _ = ray.get(result_ref)
        actual_result[rowidx] = result
    expected_result = nemo_gym_sanity_test_data["expected_output"]

    # These are tensors originally and we swap them back to a list for comparison below
    for d in actual_result:
        for message in d["input_message_log"]:
            message["token_ids"] = message["token_ids"].tolist()
        # Right now, we don't need to swap the token ids in the message log since they pointto the same underlying dictionary as above.
        # for message in d["message_log"][:1]:
        #     message["token_ids"] = message["token_ids"].tolist()

    # Write the actual result to a file so it can be used to update the expected output.
    # To update: cp actual_test_nemo_gym_sanity.json test_nemo_gym_sanity.json
    _write_actual_test_data(original_input, actual_result)

    def _standardize_single_result(d: dict):
        d = deepcopy(d)
        d.pop("full_result", None)
        d.pop("physical_message_logs", None)

        # We remove these fields and message from comparison since we cannot guarantee exact generation reproducibility
        d["message_log"] = d["message_log"][:2]
        for message in d["message_log"][1:]:
            if "token_ids" in message:
                message["token_ids"] = []
            if "generation_logprobs" in message:
                message["generation_logprobs"] = []
            if "prompt_str" in message:
                message["prompt_str"] = "dummy prompt_str"
            if "generation_str" in message:
                message["generation_str"] = "dummy generation_str"
            message.setdefault("is_invalid_tool_call", False)
            message.setdefault("has_malformed_thinking", False)

        return d

    def _standardize(l: list[dict]):
        return list(map(_standardize_single_result, l))

    assert _standardize(expected_result) == _standardize(actual_result)


# Sentinel for omitting the top_logprobs field entirely, which is distinct from sending null.
_OMIT_TOP_LOGPROBS = object()


@pytest.mark.nemo_gym
def test_vllm_http_logprobs_contract(nemo_gym_vllm_generation):
    """Pin the vLLM OpenAI HTTP logprobs contract that NeMo-Gym capture depends on.

    NeMo-Gym's vllm_model sets logprobs=True and return_tokens_as_token_ids=True to extract
    per-token ids and logprobs for training (Gym omits top_logprobs on the capture path, so
    vLLM applies its default; Gym PR #1612 additionally pins top_logprobs=0, which is
    equivalent). vLLM computes `logprobs = top_logprobs if logprobs else None`, so omitting
    top_logprobs (default 0) or sending 0 returns logprobs, while an explicit null returns
    none and silently empties the captured token ids. This exercises the real HTTP path where
    that translation lives (the offline LLM API does not), so a vLLM bump that changes the
    contract fails here instead of silently freezing training.

    All three cases share the (expensive) vLLM fixture, so they run in a single test rather
    than as separate parametrized cases.
    """
    base_url = nemo_gym_vllm_generation.dp_openai_server_base_urls[0]
    gen_cfg = nemo_gym_vllm_generation.cfg

    def _chat(top_logprobs_field):
        body = {
            "model": gen_cfg["model_name"],
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
            # The RL HTTP wrapper asserts these match the generation config exactly.
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
            # The fields NeMo-Gym sets to capture token ids.
            "logprobs": True,
            "return_tokens_as_token_ids": True,
        }
        if top_logprobs_field is not _OMIT_TOP_LOGPROBS:
            body["top_logprobs"] = top_logprobs_field

        # The base URL is known once the fixture is ready, but retry briefly to avoid racing
        # the very first connection to the server.
        last_exc = None
        for _ in range(30):
            try:
                return requests.post(
                    f"{base_url}/chat/completions", json=body, timeout=60
                )
            except requests.exceptions.ConnectionError as e:
                last_exc = e
                time.sleep(1)
        raise AssertionError(f"vLLM HTTP server never became reachable: {last_exc}")

    def _assert_has_token_ids(resp, label):
        resp.raise_for_status()
        content = resp.json()["choices"][0]["logprobs"]["content"]
        assert content, f"expected per-token logprobs for {label}"
        # return_tokens_as_token_ids makes each token a "token_id:<int>" string; capture
        # parses these into ints, so they must all parse.
        token_ids = [int(c["token"].removeprefix("token_id:")) for c in content]
        assert len(token_ids) == len(content)

    # Omitting top_logprobs (what Gym does on the capture path; vLLM default 0) and sending 0
    # (the equivalent explicit pin) must both yield per-token logprobs whose tokens decode to ints.
    _assert_has_token_ids(_chat(_OMIT_TOP_LOGPROBS), "omitted top_logprobs")
    _assert_has_token_ids(_chat(0), "top_logprobs=0")

    # Explicit null is the divergence that motivates the Gym fix: vLLM returns no logprobs
    # (200 with logprobs=None) or rejects the request outright. Both mean capture gets
    # nothing. If a future vLLM makes null behave like 0, this fails and signals the Gym
    # workaround can be relaxed.
    null_resp = _chat(None)
    if null_resp.status_code == 200:
        assert null_resp.json()["choices"][0].get("logprobs") is None
    else:
        # A rejection must be a client-side validation error, not an unrelated server failure
        # that would let this branch pass vacuously.
        assert 400 <= null_resp.status_code < 500, (
            f"expected null top_logprobs accepted-with-None or rejected as 4xx, "
            f"got {null_resp.status_code}: {null_resp.text}"
        )
