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
"""Pixtral (vLLM Mistral3) image capture: multi-row placeholders, learner patch layout, dtype, geometry."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_rl.data.captured_media import MediaCaptureRejected, capture_processed_media
from nemo_rl.data.multimodal_utils import _patchify_segments

pytestmark = pytest.mark.nemo_gym

IMG, BREAK, END = 18, 12, 13
PATCH, MERGE = 14, 2


@dataclass(frozen=True)
class Span:
    offset: int
    length: int
    is_embed: torch.Tensor | None = None


def pixtral_tokens(height, width):
    """vLLM Mistral3: ([IMG] * cols + [IMG_BREAK]) * rows, last token replaced by [IMG_END]."""
    cols, rows = width // (PATCH * MERGE), height // (PATCH * MERGE)
    tokens = ([IMG] * cols + [BREAK]) * rows
    tokens[-1] = END
    return tokens


def prompt_with_image(pixels, prefix=(1, 2, 3), suffix=(4,), with_mask=True):
    image = pixtral_tokens(*pixels.shape[-2:])
    mask = torch.tensor([t == IMG for t in image]) if with_mask else None
    item = SimpleNamespace(get_data=lambda: {"pixel_values": pixels})
    return {
        "prompt_token_ids": list(prefix) + image + list(suffix),
        "mm_placeholders": {"image": [Span(len(prefix), len(image), mask)]},
        "mm_kwargs": {"image": [item]},
    }


def capture(prompt, **kwargs):
    kwargs.setdefault("image_token_id", IMG)
    kwargs.setdefault("patch_size", PATCH)
    kwargs.setdefault("media_family", "pixtral")
    kwargs.setdefault("spatial_merge_size", MERGE)
    return capture_processed_media(prompt, prev_len=0, **kwargs)


@pytest.mark.parametrize("with_mask", [True, False])
def test_rows_of_image_tokens_are_one_image(with_mask):
    pixels = torch.rand(3, 56, 84)
    media = capture(prompt_with_image(pixels, with_mask=with_mask))
    (item,) = media.items
    assert item.token_id == IMG
    assert item.embedding_spans == ((3, 3), (7, 3))
    assert item.imgs_sizes == ((56, 84),)
    assert tuple(media.tensors["imgs"].shape) == (
        1,
        (56 // PATCH) * (84 // PATCH),
        3 * PATCH * PATCH,
    )
    assert media.tensors["imgs_sizes"].tolist() == [[56, 84]]
    assert "num_frames" not in media.tensors


def test_packed_patches_match_the_learner_patchify():
    pixels = torch.rand(3, 56, 84)
    media = capture(prompt_with_image(pixels))
    expected = _patchify_segments([pixels.unsqueeze(0)], patch_dim=PATCH)
    torch.testing.assert_close(media.tensors["imgs"], expected, rtol=0, atol=0)


def test_pixels_are_cast_to_the_staging_dtype():
    pixels = torch.rand(3, 28, 56)
    media = capture(prompt_with_image(pixels), cast_pixel_dtype=torch.bfloat16)
    assert media.tensors["imgs"].dtype == torch.bfloat16
    expected = _patchify_segments(
        [pixels.to(torch.bfloat16).unsqueeze(0)], patch_dim=PATCH
    )
    torch.testing.assert_close(media.tensors["imgs"], expected, rtol=0, atol=0)


def test_embedding_count_must_match_the_geometry():
    pixels = torch.rand(3, 56, 84)
    prompt = prompt_with_image(pixels)
    span = prompt["mm_placeholders"]["image"][0]
    mask = span.is_embed.clone()
    mask[0] = False
    prompt["mm_placeholders"]["image"][0] = Span(span.offset, span.length, mask)
    with pytest.raises(MediaCaptureRejected, match="geometry"):
        capture(prompt)


def test_embedding_positions_must_be_image_tokens():
    # The mask's count matches the geometry, but it marks another token.
    other = 99
    pixels = torch.rand(3, 56, 84)
    prompt = prompt_with_image(pixels)
    image = [other if t == IMG else t for t in pixtral_tokens(56, 84)]
    prompt["prompt_token_ids"] = [1, 2, 3, *image, 4]
    prompt["mm_placeholders"]["image"][0] = Span(
        3, len(image), torch.tensor([t == other for t in image])
    )
    with pytest.raises(MediaCaptureRejected, match="Invalid media embedding tokens"):
        capture(prompt)


def test_image_size_must_be_whole_merge_cells():
    # 42 pixels is 3 patches but 1.5 merge cells; the row and column counts
    # (1 x 3) still match the 3 image tokens.
    pixels = torch.rand(3, 42, 84)
    with pytest.raises(MediaCaptureRejected, match="geometry"):
        capture(prompt_with_image(pixels))


@pytest.mark.parametrize(
    ("dtype", "cast_dtype"),
    [(torch.float64, None), (torch.uint8, torch.bfloat16)],
)
def test_pixel_dtype_outside_the_allow_list_is_rejected(dtype, cast_dtype):
    # Checked before the cast, so a cast does not launder an integer image.
    pixels = (torch.rand(3, 28, 56) * 255).to(dtype)
    with pytest.raises(MediaCaptureRejected, match="pixel dtype"):
        capture(prompt_with_image(pixels), cast_pixel_dtype=cast_dtype)


@pytest.mark.parametrize("channels", [1, 4])
def test_pixels_must_have_three_channels(channels):
    pixels = torch.rand(channels, 28, 56)
    with pytest.raises(MediaCaptureRejected, match=r"\[3, H, W\]"):
        capture(prompt_with_image(pixels))


def test_two_images_in_one_prompt():
    first, second = torch.rand(3, 56, 84), torch.rand(3, 28, 56)
    image_a, image_b = pixtral_tokens(56, 84), pixtral_tokens(28, 56)
    b_offset = 3 + len(image_a) + 2
    prompt = {
        "prompt_token_ids": [1, 2, 3, *image_a, 5, 6, *image_b, 4],
        "mm_placeholders": {
            "image": [
                Span(3, len(image_a), torch.tensor([t == IMG for t in image_a])),
                Span(b_offset, len(image_b), torch.tensor([t == IMG for t in image_b])),
            ]
        },
        "mm_kwargs": {
            "image": [
                SimpleNamespace(get_data=lambda: {"pixel_values": first}),
                SimpleNamespace(get_data=lambda: {"pixel_values": second}),
            ]
        },
    }
    media = capture(prompt)
    item_a, item_b = media.items
    assert item_a.placeholder_offset == 3
    assert item_a.embedding_spans == ((3, 3), (7, 3))
    assert item_a.imgs_sizes == ((56, 84),)
    assert item_b.placeholder_offset == b_offset
    assert item_b.embedding_spans == ((b_offset, 2),)
    assert item_b.imgs_sizes == ((28, 56),)
    # Both images packed in prompt order, as the learner packs them.
    assert media.tensors["imgs_sizes"].tolist() == [[56, 84], [28, 56]]
    expected = _patchify_segments(
        [first.unsqueeze(0), second.unsqueeze(0)], patch_dim=PATCH
    )
    torch.testing.assert_close(media.tensors["imgs"], expected, rtol=0, atol=0)


def test_wrong_pixel_layout_is_rejected():
    pixels = torch.rand(1, 3, 28, 28)
    prompt = prompt_with_image(pixels[0])
    prompt["mm_kwargs"]["image"][0] = SimpleNamespace(
        get_data=lambda: {"pixel_values": pixels}
    )
    with pytest.raises(MediaCaptureRejected, match=r"\[3, H, W\]"):
        capture(prompt)


def test_pixtral_rejects_video():
    item = SimpleNamespace(get_data=lambda: {"pixel_values": torch.rand(3, 28, 28)})
    prompt = {
        "prompt_token_ids": [IMG],
        "mm_placeholders": {"video": [Span(0, 1)]},
        "mm_kwargs": {"video": [item]},
    }
    with pytest.raises(MediaCaptureRejected, match="images only"):
        capture(prompt)


def test_omni_still_requires_one_contiguous_run():
    pixels = torch.rand(3, 56, 84)
    prompt = prompt_with_image(pixels)
    prompt["mm_kwargs"]["image"][0] = SimpleNamespace(
        get_data=lambda: {"pixel_values_flat": pixels, "imgs_sizes": [56, 84]}
    )
    with pytest.raises(MediaCaptureRejected, match="contiguous"):
        capture(prompt, media_family="omni", spatial_merge_size=None)


def test_unknown_family_is_rejected():
    with pytest.raises(MediaCaptureRejected, match="family"):
        capture(prompt_with_image(torch.rand(3, 28, 28)), media_family="other")
