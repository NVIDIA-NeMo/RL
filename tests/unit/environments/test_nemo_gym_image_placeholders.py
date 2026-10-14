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

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
)

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    attach_image_model_inputs_to_message,
    uses_image_placeholder,
)

# --------------------------------------------------------------------------
# ragged pixel_values path in attach_image_model_inputs_to_message
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
        imgs_sizes=torch.tensor(
            [[shape[-2], shape[-1]] for shape in shapes], dtype=torch.long
        ),
    )


def _images(n: int) -> list[Image.Image]:
    return [Image.new("RGB", (4, 4)) for _ in range(n)]


def test_ragged_output_requested_only_for_multi_image_turns():
    """The ragged switch needs both the flag and more than one image."""
    for count, flag, expected in [(2, True, None), (1, True, "pt"), (2, False, "pt")]:
        processor = NemotronNanoVLV2Processor(torch.zeros(count, 3, 4, 4))
        attach_image_model_inputs_to_message(
            {},
            images=_images(count),
            processor=processor,
            pad_dynamic_image_shapes=flag,
        )
        assert processor.calls[0]["return_tensors"] == expected, (
            f"images={count} flag={flag}"
        )


def test_ragged_pixel_values_are_patchified_to_one_tensor():
    """Heterogeneous CHW tensors become one packed patch sequence."""
    processor = _ragged((3, 16, 32), (3, 32, 16))
    user_message: dict = {}
    attach_image_model_inputs_to_message(
        user_message,
        images=_images(2),
        processor=processor,
        pad_dynamic_image_shapes=True,
    )
    packed = user_message["pixel_values"].as_tensor()
    assert packed.shape == (1, 4, 768)


def test_ragged_pixel_values_reject_non_chw_entries():
    processor = _ragged((3, 2, 4), (2, 4))
    with pytest.raises(ValueError, match="one CHW tensor per image"):
        attach_image_model_inputs_to_message(
            {},
            images=_images(2),
            processor=processor,
            pad_dynamic_image_shapes=True,
        )


def test_ragged_pixel_values_reject_mixed_channel_counts():
    processor = _ragged((3, 2, 4), (1, 2, 4))
    with pytest.raises(ValueError, match="same channel count"):
        attach_image_model_inputs_to_message(
            {},
            images=_images(2),
            processor=processor,
            pad_dynamic_image_shapes=True,
        )


def test_attach_is_a_noop_without_images_or_processor():
    user_message: dict = {}
    attach_image_model_inputs_to_message(
        user_message, images=[], processor=NemotronNanoVLV2Processor(None)
    )
    attach_image_model_inputs_to_message(
        user_message, images=_images(1), processor=None
    )
    assert user_message == {}


@pytest.fixture
def qwen_processor() -> Qwen2_5_VLProcessor:
    """Use the real image processor with an offline, minimal tokenizer."""
    special_tokens = [
        "[UNK]",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {token: index for index, token in enumerate(special_tokens)},
                unk_token="[UNK]",
            )
        ),
        unk_token="[UNK]",
        additional_special_tokens=special_tokens[1:],
    )
    return Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=3136, max_pixels=1003520),
        video_processor=Qwen2VLVideoProcessor(),
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("pad_dynamic_image_shapes", [False, True])
@pytest.mark.parametrize("image_count", [1, 2])
def test_qwen_attach_preserves_patches_grids_and_rollout_tokens(
    qwen_processor, image_count, pad_dynamic_image_shapes
):
    # Different resolutions and colors expose both unwanted CHW padding and
    # image reordering. Qwen merges each 2x2 patch group into one image token.
    images = [
        Image.new("RGB", (56, 56), color=(255, 0, 0)),
        Image.new("RGB", (112, 56), color=(0, 255, 0)),
    ][:image_count]
    prompt = (
        "Describe "
        + "<|vision_start|><|image_pad|><|vision_end|>" * image_count
        + " in Python."
    )
    full_prompt = qwen_processor(text=prompt, images=images, return_tensors="pt")
    rollout_tokens = full_prompt["input_ids"][0].clone()
    message = {"role": "user", "token_ids": rollout_tokens}

    attach_image_model_inputs_to_message(
        message,
        images=images,
        processor=qwen_processor,
        pad_dynamic_image_shapes=pad_dynamic_image_shapes,
    )

    assert not uses_image_placeholder(qwen_processor)
    assert message["token_ids"] is rollout_tokens
    torch.testing.assert_close(rollout_tokens, full_prompt["input_ids"][0])
    for key in ("pixel_values", "image_grid_thw"):
        assert isinstance(message[key], PackedTensor)
        assert message[key].preprocess_mode is None
        torch.testing.assert_close(message[key].as_tensor(), full_prompt[key])
    grid = message["image_grid_thw"].as_tensor()
    torch.testing.assert_close(grid, torch.tensor([[1, 4, 4], [1, 4, 8]][:image_count]))
    patch_count = int(grid.prod(dim=1).sum())
    assert message["pixel_values"].as_tensor().shape == (patch_count, 1176)
    assert int((rollout_tokens == qwen_processor.image_token_id).sum()) == (
        patch_count // qwen_processor.image_processor.merge_size**2
    )
    # Synthetic image-only token/type IDs must not replace full-prompt data.
    assert set(message) == {"role", "token_ids", "pixel_values", "image_grid_thw"}
