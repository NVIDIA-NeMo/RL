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

from copy import deepcopy

import pytest
from PIL import Image

from nemo_rl.data.multimodal_utils import (
    extract_input_media_sources_from_responses_messages,
    image_to_data_url,
)
from nemo_rl.environments.nemo_gym_multimodal import (
    _extract_input_images_from_message,
    _extract_static_video_messages,
    _index_per_turn_images,
    _inject_vllm_mm_processor_kwargs,
    _make_overlength_filtered_video_example,
    _remove_vllm_mm_processor_kwargs,
    _strip_local_media_metadata,
    _without_initial_media_sources,
    normalize_media_in_examples,
)
from nemo_rl.environments.nemo_gym_request import _metadata_extra_body
from nemo_rl.environments.nemo_gym_task import get_nemo_gym_task_input


@pytest.mark.parametrize("native", [False, True])
def test_media_normalization_preserves_task_envelope(tmp_path, native):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (2, 3)).save(image_path)
    task_input = {
        "responses_create_params": {"input": [_user(str(image_path))]},
        "task_data": {"media_ref": str(image_path)},
    }
    row = (
        {
            "task_id": {"taskset": "vision:train", "task_id": "image-1"},
            "task_input": task_input,
        }
        if native
        else task_input
    )
    original = deepcopy(row)

    normalize_media_in_examples([row])

    converted = get_nemo_gym_task_input(row)
    source = converted["responses_create_params"]["input"][0]["content"][0]["image_url"]
    assert source.startswith("data:image/")
    assert converted["task_data"] == {"media_ref": str(image_path)}
    if native:
        assert row["task_id"] == original["task_id"]
        assert "responses_create_params" not in row


def test_native_video_preprocessing_updates_nested_request_only(tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video placeholder")
    row = {
        "task_id": {"taskset": "video:train", "task_id": "video-1"},
        "task_input": {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_video",
                                "video_url": str(video_path),
                                "_video_source": str(video_path),
                            }
                        ],
                    }
                ],
                "metadata": {"extra_body": '{"seed": 11}'},
            },
            "task_data": {"verifier": "opaque"},
        },
        "_rowidx": 3,
    }
    messages, resolved_path = _extract_static_video_messages(row)
    assert resolved_path == str(video_path)
    assert messages[0]["content"][0]["type"] == "video"

    _inject_vllm_mm_processor_kwargs(row, {"video_as_images": True, "max_num_tiles": 1})
    _remove_vllm_mm_processor_kwargs(row, {"max_num_tiles"})
    _strip_local_media_metadata(row)
    assert _metadata_extra_body(row) == {
        "seed": 11,
        "mm_processor_kwargs": {"video_as_images": True},
    }
    part = row["task_input"]["responses_create_params"]["input"][0]["content"][0]
    assert "_video_source" not in part

    filtered = _make_overlength_filtered_video_example(row)
    assert filtered["task_id"] == row["task_id"]
    assert filtered["_rowidx"] == 3
    assert filtered["task_input"]["task_data"] == {"verifier": "opaque"}
    assert "responses_create_params" not in filtered
    assert (
        filtered["task_input"]["responses_create_params"]["input"][0]["content"][0][
            "type"
        ]
        == "input_text"
    )
    assert part["type"] == "input_video"


def _image(size: tuple[int, int]) -> str:
    """Return a data URL for a solid RGB image of the given size."""
    return image_to_data_url(Image.new("RGB", size))


def _user(*data_urls: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "input_image", "image_url": url} for url in data_urls],
    }


def _assistant(token_ids: list[int]) -> dict:
    return {"role": "assistant", "generation_token_ids": token_ids}


def test_extract_input_images_handles_flat_and_dict_image_url():
    item = {
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": _image((2, 2))},
            {"type": "input_image", "image_url": {"url": _image((3, 3))}},
            {"type": "input_text", "text": "ignore me"},
        ],
    }
    images = _extract_input_images_from_message(item)
    assert [img.size for img in images] == [(2, 2), (3, 3)]


def test_extract_input_images_returns_empty_for_string_content():
    assert _extract_input_images_from_message({"role": "user", "content": "hi"}) == []
    assert _extract_input_images_from_message({"role": "user"}) == []


def test_extract_input_images_ignores_text_function_call_output():
    item = {
        "type": "function_call_output",
        "call_id": "c1",
        "output": '{"ok": true}',
    }
    assert _extract_input_images_from_message(item) == []

    item["output"] = "Tool failed to create result.png"
    assert _extract_input_images_from_message(item) == []


def test_index_per_turn_images_bins_images():
    output = [
        _user(_image((2, 2))),
        _assistant([1, 2]),
        _user(_image((3, 3)), _image((4, 4))),
        _assistant([3, 4]),
    ]
    per_turn = _index_per_turn_images(output)

    assert len(per_turn) == 2
    assert [img.size for img in per_turn[0]] == [(2, 2)]
    assert [img.size for img in per_turn[1]] == [(3, 3), (4, 4)]


def test_index_per_turn_images_seeds_first_turn_from_input_messages():
    input_messages = [_user(_image((2, 2)))]
    output = [_assistant([1, 2])]

    per_turn = _index_per_turn_images(output, input_messages=input_messages)

    assert len(per_turn) == 1
    assert [img.size for img in per_turn[0]] == [(2, 2)]


def test_index_per_turn_images_text_only_rollout_yields_empty_buckets():
    output = [
        {"role": "user", "content": "solve this"},
        _assistant([1, 2]),
        {"role": "user", "content": "and this"},
        _assistant([3, 4]),
    ]
    assert _index_per_turn_images(output) == [[], []]


def test_index_per_turn_images_assigns_tool_result_image_to_next_turn():
    """A tool-result image contributes to the following assistant turn."""
    output = [
        _user(_image((2, 2))),
        _assistant([1, 2]),
        {"type": "function_call_output", "output": _image((5, 5))},
        _assistant([3, 4]),
    ]
    per_turn = _index_per_turn_images(output)

    assert len(per_turn) == 2
    assert [img.size for img in per_turn[0]] == [(2, 2)]
    assert [img.size for img in per_turn[1]] == [(5, 5)]


def test_index_per_turn_images_aligns_with_postprocess_skip_of_empty_generations():
    """Turns skipped by the postprocess loop must not consume an image bucket.

    ``_postprocess_nemo_gym_to_nemo_rl_result`` skips output items whose
    ``generation_token_ids`` is present but empty, so the bucket list must skip
    them too or every later turn is attached to the wrong images.
    """
    output = [
        _user(_image((2, 2))),
        _assistant([]),  # all-EOS generation, skipped by the postprocess loop
        _user(_image((6, 6))),
        _assistant([7, 8]),
    ]
    per_turn = _index_per_turn_images(output)

    assert len(per_turn) == 1
    assert [img.size for img in per_turn[0]] == [(2, 2), (6, 6)]


def test_index_per_turn_images_flushes_on_non_assistant_trainable_item():
    """Trainable items whose role is not ``assistant`` (reasoning-only responses,
    function_call items) still carry ``generation_token_ids`` and are treated as
    turns by the postprocess loop. The per-turn image bucket must flush for them
    too, or the batched flatten path will see a ``PackedTensor`` for turns
    where the model produced a normal assistant message and a missing key for
    turns where it produced only reasoning — crashing
    ``PackedTensor.flattened_concat`` on the None entry.
    """
    reasoning_only = {"type": "reasoning", "generation_token_ids": [9, 10]}
    output = [
        _user(_image((2, 2))),
        reasoning_only,
    ]
    per_turn = _index_per_turn_images(output)

    assert len(per_turn) == 1
    assert [img.size for img in per_turn[0]] == [(2, 2)]


def test_index_per_turn_images_flushes_on_function_call_trainable_item():
    """Same as the reasoning-only case, but for tool-calling turns where the
    model call's last output item is a ``function_call`` (no ``role`` field)."""
    function_call = {
        "type": "function_call",
        "name": "tool",
        "arguments": "{}",
        "call_id": "c1",
        "generation_token_ids": [11, 12],
    }
    output = [
        _user(_image((2, 2))),
        function_call,
        {"type": "function_call_output", "output": _image((5, 5)), "call_id": "c1"},
        _assistant([13, 14]),
    ]
    per_turn = _index_per_turn_images(output)

    assert len(per_turn) == 2
    assert [img.size for img in per_turn[0]] == [(2, 2)]
    assert [img.size for img in per_turn[1]] == [(5, 5)]


def test_without_initial_media_sources_strips_videos_and_images_in_order():
    """Video parts must be de-duplicated alongside images, in encounter order."""
    image_url = _image((2, 2))
    video_url = "data:video/mp4;base64,dG95"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_video", "video_url": video_url},
                {"type": "input_image", "image_url": image_url},
                {"type": "input_text", "text": "What is shown?"},
            ],
        }
    ]
    initial_sources = extract_input_media_sources_from_responses_messages(messages)
    assert initial_sources == [("video", video_url), ("image", image_url)]

    filtered, fully_consumed = _without_initial_media_sources(messages, initial_sources)

    assert fully_consumed is True
    assert filtered[0]["content"] == [{"type": "input_text", "text": "What is shown?"}]
    # The caller's messages must not be mutated in place.
    assert len(messages[0]["content"]) == 3


def test_without_initial_media_sources_keeps_media_the_agent_added():
    """Only the ordered prefix of initial sources is removed; extras survive."""
    initial_image = _image((2, 2))
    agent_image = _image((4, 4))
    messages = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": initial_image}],
        },
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": agent_image}],
        },
    ]

    filtered, fully_consumed = _without_initial_media_sources(
        messages, [("image", initial_image)]
    )

    assert fully_consumed is True
    assert filtered[0]["content"] == []
    assert filtered[1]["content"] == [{"type": "input_image", "image_url": agent_image}]


def test_without_initial_media_sources_reports_unconsumed_sources():
    """A source that never appears leaves the consumed flag False."""
    filtered, fully_consumed = _without_initial_media_sources(
        [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        [("image", "data:image/png;base64,AA")],
    )

    assert fully_consumed is False
    assert filtered[0]["content"] == [{"type": "input_text", "text": "hi"}]


def test_without_initial_media_sources_passes_through_non_list_messages():
    assert _without_initial_media_sources("not-a-list", []) == ("not-a-list", False)
