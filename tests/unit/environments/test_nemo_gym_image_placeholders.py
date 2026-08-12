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
import torch
from PIL import Image

from nemo_rl.environments.nemo_gym import (
    _attach_multimodal_data_to_user_message,
    _count_image_payloads,
    _normalize_image_placeholders,
    sanitize_nemo_gym_example_image_placeholders,
)


def _example(*input_items: dict) -> dict:
    return {"responses_create_params": {"input": list(input_items)}}


def _text_part(text: str) -> dict:
    return {"type": "input_text", "text": text}


def _image_part() -> dict:
    return {
        "type": "input_image",
        "image_url": "data:image/png;base64,AAAA",
        "detail": "auto",
    }


# --------------------------------------------------------------------------
# _count_image_payloads
# --------------------------------------------------------------------------


def test_count_image_payloads_counts_structured_items():
    example = _example(
        {"role": "user", "content": [_image_part(), _text_part("hi"), _image_part()]}
    )
    assert _count_image_payloads(example) == 2


def test_count_image_payloads_ignores_text_only_and_malformed():
    assert _count_image_payloads(_example({"role": "user", "content": "plain"})) == 0
    assert _count_image_payloads(_example({"role": "user"})) == 0
    assert _count_image_payloads({"responses_create_params": {"input": "nope"}}) == 0
    assert _count_image_payloads({}) == 0


# --------------------------------------------------------------------------
# _normalize_image_placeholders
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "replacement,expected",
    # Dropping the token leaves the spaces that surrounded it, so the
    # multimodal branch collapses to a double space rather than a single one.
    [("", "a  b"), ("image", "a image b")],
)
def test_normalize_replaces_image_token_with_replacement(replacement, expected):
    assert _normalize_image_placeholders("a <image> b", replacement) == expected


def test_normalize_strips_img_wrappers_in_both_branches():
    # <img>/</img> are structural wrappers, dropped regardless of replacement.
    assert _normalize_image_placeholders("<img>x</img>", "") == "x"
    assert _normalize_image_placeholders("<img>x</img>", "image") == "x"


# --------------------------------------------------------------------------
# sanitize_nemo_gym_example_image_placeholders
# --------------------------------------------------------------------------


def test_sanitize_drops_literal_token_when_images_are_attached():
    """Structured items are the source of truth, so the literal token goes."""
    example = _example(
        {"role": "user", "content": [_image_part(), _text_part("look <image> here")]}
    )
    out = sanitize_nemo_gym_example_image_placeholders(example)
    assert out["responses_create_params"]["input"][0]["content"][1]["text"] == (
        "look  here"
    )


def test_sanitize_rewrites_token_as_a_word_when_text_only():
    """With no attached image the token must not survive as a control token."""
    example = _example({"role": "user", "content": [_text_part("an <image> of a cat")]})
    out = sanitize_nemo_gym_example_image_placeholders(example)
    assert out["responses_create_params"]["input"][0]["content"][0]["text"] == (
        "an image of a cat"
    )


def test_sanitize_does_not_mutate_its_input():
    """Rollout batches share example objects, so the input must be untouched."""
    example = _example({"role": "user", "content": [_text_part("an <image>")]})
    before = copy.deepcopy(example)
    sanitize_nemo_gym_example_image_placeholders(example)
    assert example == before


def test_sanitize_does_not_fabricate_content_on_items_without_it():
    """function_call / reasoning items carry no content key and must not gain one."""
    example = _example(
        {"type": "function_call", "name": "f", "arguments": "{}"},
        {"type": "reasoning", "summary": [{"text": "thinking <image>"}]},
    )
    out = sanitize_nemo_gym_example_image_placeholders(example)
    for item in out["responses_create_params"]["input"]:
        assert "content" not in item


def test_sanitize_handles_string_content_and_string_parts():
    example = _example(
        {"role": "user", "content": "bare <image> string"},
        {"role": "user", "content": ["listed <image> string"]},
    )
    out = sanitize_nemo_gym_example_image_placeholders(example)
    items = out["responses_create_params"]["input"]
    assert items[0]["content"] == "bare image string"
    assert items[1]["content"][0] == "listed image string"


def test_sanitize_returns_example_unchanged_when_input_is_not_a_list():
    example = {"responses_create_params": {"input": "not-a-list"}}
    assert sanitize_nemo_gym_example_image_placeholders(example) == example


# --------------------------------------------------------------------------
# ragged pixel_values path in _attach_multimodal_data_to_user_message
# --------------------------------------------------------------------------


class _Tokenizer:
    model_input_names = ["input_ids", "attention_mask"]


class NemotronNanoVLV2Processor:
    """Placeholder-style stub: the real class name is what uses_image_placeholder keys on."""

    image_token = "<image>"
    model_input_names = ["input_ids", "attention_mask", "pixel_values"]

    def __init__(self, pixel_values, imgs_sizes=None):
        self._pixel_values = pixel_values
        # Real dynamic-resolution processors emit imgs_sizes alongside a ragged
        # pixel_values list. Supplying it here is not incidental: the
        # imgs_sizes backfill reads pixel_values' shape, and a ragged list has
        # no single shape to read, so the ragged path must derive sizes per
        # image before padding.
        self._imgs_sizes = imgs_sizes
        self.tokenizer = _Tokenizer()
        self.calls: list[dict] = []

    def __call__(self, *, text, images, return_tensors):
        self.calls.append({"text": text, "return_tensors": return_tensors})
        # extract_multimodal_model_inputs requires input_ids and validates its
        # rank, so a stub without it fails before reaching the image handling.
        processed = {
            "pixel_values": self._pixel_values,
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
        }
        if self._imgs_sizes is not None:
            processed["imgs_sizes"] = self._imgs_sizes
        return processed


def _ragged(*shapes: tuple[int, ...]) -> NemotronNanoVLV2Processor:
    return NemotronNanoVLV2Processor(
        [torch.ones(*shape) for shape in shapes],
        imgs_sizes=torch.tensor([[4, 4]] * len(shapes), dtype=torch.long),
    )


def _images(n: int) -> list[Image.Image]:
    return [Image.new("RGB", (4, 4)) for _ in range(n)]


def test_ragged_output_requested_only_for_multi_image_turns():
    """The ragged switch needs both the flag and more than one image."""
    for count, flag, expected in [(2, True, None), (1, True, "pt"), (2, False, "pt")]:
        processor = NemotronNanoVLV2Processor(torch.zeros(count, 3, 4, 4))
        _attach_multimodal_data_to_user_message(
            {},
            images=_images(count),
            processor=processor,
            pad_dynamic_image_shapes=flag,
        )
        assert processor.calls[0]["return_tensors"] == expected, (
            f"images={count} flag={flag}"
        )


def test_ragged_pixel_values_are_padded_to_one_tensor():
    """Heterogeneous CHW tensors become a single padded tensor for the message."""
    processor = _ragged((3, 2, 4), (3, 6, 4))
    user_message: dict = {}
    _attach_multimodal_data_to_user_message(
        user_message,
        images=_images(2),
        processor=processor,
        pad_dynamic_image_shapes=True,
    )
    packed = user_message["pixel_values"].as_tensor()
    # Two images, padded up to the tallest, channels preserved.
    assert packed.shape[0] == 2
    assert packed.shape[-3] == 3
    assert packed.shape[-2] == 6


def test_ragged_pixel_values_reject_non_chw_entries():
    processor = _ragged((3, 2, 4), (2, 4))
    with pytest.raises(ValueError, match="one CHW tensor per image"):
        _attach_multimodal_data_to_user_message(
            {},
            images=_images(2),
            processor=processor,
            pad_dynamic_image_shapes=True,
        )


def test_ragged_pixel_values_reject_mixed_channel_counts():
    processor = _ragged((3, 2, 4), (1, 2, 4))
    with pytest.raises(ValueError, match="same channel count"):
        _attach_multimodal_data_to_user_message(
            {},
            images=_images(2),
            processor=processor,
            pad_dynamic_image_shapes=True,
        )


def test_attach_is_a_noop_without_images_or_processor():
    user_message: dict = {}
    _attach_multimodal_data_to_user_message(
        user_message, images=[], processor=NemotronNanoVLV2Processor(None)
    )
    _attach_multimodal_data_to_user_message(
        user_message, images=_images(1), processor=None
    )
    assert user_message == {}
