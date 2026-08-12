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

from PIL import Image

from nemo_rl.data.multimodal_utils import image_to_data_url
from nemo_rl.environments.nemo_gym import (
    _extract_input_images_from_message,
    _index_per_turn_images,
)


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
