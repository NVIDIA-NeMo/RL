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

"""Unit tests for the Megatron multimodal helper module.

These tests exercise the helpers in isolation against synthetic fake
models constructed from ``SimpleNamespace`` so the suite does not need
to load a real Megatron-LM model into GPU memory. The Megatron-LM
``PackedSeqParams`` import is required at module load time, so the
tests are skipped if Megatron-LM is not available in the environment.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core.packed_seq_params")

from nemo_rl.models.megatron.common import _vlm_sp_repad_collapsed  # noqa: E402
from nemo_rl.models.megatron.multimodal import (  # noqa: E402
    _get_image_token_ids,
    _patchify_for_dynamic_resolution,
    collapse_multimodal_tokens,
    is_llava_model,
    prepare_multimodal_data,
)


IMG_START_ID = 19
IMG_END_ID = 20
IMG_CONTEXT_ID = 18
PATCH_DIM = 16


def _make_dynamic_model(*, video_temporal_patch_size: int = 1) -> SimpleNamespace:
    """Build a fake current-mainline Megatron LLaVA model."""
    return SimpleNamespace(
        img_start_token_id=IMG_START_ID,
        img_end_token_id=IMG_END_ID,
        dynamic_resolution=True,
        temporal_patch_dim=video_temporal_patch_size,
        vision_model=SimpleNamespace(
            patch_dim=PATCH_DIM,
            dynamic_resolution=True,
            temporal_patch_dim=video_temporal_patch_size,
        ),
        sound_model=None,
    )


def _make_static_model(static_img_seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        img_start_token_id=IMG_START_ID,
        img_end_token_id=IMG_END_ID,
        _dynamic_resolution=False,
        img_seq_len=static_img_seq_len,
        vision_model=SimpleNamespace(patch_dim=PATCH_DIM),
        sound_model=None,
    )


def _build_multimodal_input_ids(
    image_token_counts: list[int],
    *,
    text_prefix: int = 4,
    text_suffix: int = 3,
    pad: int = 0,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, int]:
    """Build a single-row input_ids tensor with ``<img><image>×N</img>`` spans.

    Returns the padded tensor and the valid (unpadded) length.
    """
    seq: list[int] = [101] * text_prefix
    for n in image_token_counts:
        seq.append(IMG_START_ID)
        seq.extend([IMG_CONTEXT_ID] * n)
        seq.append(IMG_END_ID)
    seq.extend([102] * text_suffix)
    valid_len = len(seq)
    seq.extend([pad_token_id] * pad)
    return torch.tensor([seq], dtype=torch.int64), valid_len


class TestIsLlavaModel:
    def test_detects_via_image_token_attrs(self):
        model = SimpleNamespace(
            img_start_token_id=IMG_START_ID, img_end_token_id=IMG_END_ID
        )
        assert is_llava_model(model)

    def test_detects_wrapped_llava_model(self):
        wrapped = SimpleNamespace(module=SimpleNamespace(llava_model=object()))
        assert is_llava_model(wrapped)

    def test_detects_via_config(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                img_start_token_id=IMG_START_ID, img_end_token_id=IMG_END_ID
            )
        )
        assert is_llava_model(model)

    def test_text_only_model_is_not_llava(self):
        model = SimpleNamespace(some_attr="value")
        assert not is_llava_model(model)


class TestGetImageTokenIds:
    def test_reads_from_inner_attrs(self):
        model = SimpleNamespace(
            img_start_token_id=IMG_START_ID, img_end_token_id=IMG_END_ID
        )
        assert _get_image_token_ids(model) == (IMG_START_ID, IMG_END_ID)

    def test_unwraps_module_and_llava_model(self):
        inner = SimpleNamespace(
            img_start_token_id=IMG_START_ID, img_end_token_id=IMG_END_ID
        )
        outer = SimpleNamespace(module=SimpleNamespace(llava_model=inner))
        assert _get_image_token_ids(outer) == (IMG_START_ID, IMG_END_ID)

    def test_returns_none_when_missing(self):
        assert _get_image_token_ids(SimpleNamespace()) is None


class TestCollapseMultimodalTokens:
    def test_no_pixel_values_returns_data_unchanged(self):
        model = _make_dynamic_model()
        input_ids, _ = _build_multimodal_input_ids([5])
        data = {"input_ids": input_ids}
        out = collapse_multimodal_tokens(data, model)
        assert out is data

    def test_pixel_values_without_image_tokens_are_dropped(self):
        model = _make_dynamic_model()
        input_ids = torch.tensor([[101, 101, 102, 102]], dtype=torch.int64)
        data = {
            "input_ids": input_ids,
            "pixel_values": torch.zeros(1, 3, 16, 16),
            "imgs_sizes": torch.tensor([[16, 16]], dtype=torch.int32),
        }
        out = collapse_multimodal_tokens(data, model)
        assert "pixel_values" not in out
        assert "imgs_sizes" not in out

    def test_single_image_collapses_correctly(self):
        model = _make_dynamic_model()
        n_image_tokens = 5
        input_ids, valid_len = _build_multimodal_input_ids([n_image_tokens], pad=2)
        data = {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([valid_len], dtype=torch.int64),
            "pixel_values": torch.zeros(1, 3, 16, 16),
        }
        out = collapse_multimodal_tokens(data, model)
        # N context tokens minus the 1 we keep = N-1 removed.
        assert out["tokens_removed_per_sample"].tolist() == [n_image_tokens - 1]
        assert out["input_lengths"].tolist() == [valid_len - (n_image_tokens - 1)]
        # The collapsed sequence contains exactly one IMG_CONTEXT between
        # the img_start/img_end pair.
        collapsed = out["input_ids"][0].tolist()
        start_idx = collapsed.index(IMG_START_ID)
        assert collapsed[start_idx + 1] == IMG_CONTEXT_ID
        assert collapsed[start_idx + 2] == IMG_END_ID

    def test_two_images_each_get_collapsed(self):
        model = _make_dynamic_model()
        input_ids, valid_len = _build_multimodal_input_ids([3, 7])
        data = {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([valid_len], dtype=torch.int64),
            "pixel_values": torch.zeros(2, 3, 16, 16),
        }
        out = collapse_multimodal_tokens(data, model)
        assert out["tokens_removed_per_sample"].tolist() == [(3 - 1) + (7 - 1)]
        # After collapse there are 2 img_start, 2 img_context, 2 img_end markers.
        collapsed = out["input_ids"][0].tolist()
        assert collapsed.count(IMG_START_ID) == 2
        assert collapsed.count(IMG_END_ID) == 2

    def test_static_resolution_records_expansion_delta(self):
        static_img_seq_len = 257
        model = _make_static_model(static_img_seq_len)
        input_ids, valid_len = _build_multimodal_input_ids([5])
        data = {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([valid_len], dtype=torch.int64),
            "pixel_values": torch.zeros(1, 3, 16, 16),
        }
        out = collapse_multimodal_tokens(data, model)
        # Keep the physical collapse delta separate from the model's expansion
        # delta. Downstream packing/SP code prefers vision_expansion_per_sample.
        assert out["tokens_removed_per_sample"].tolist() == [5 - 1]
        assert out["vision_expansion_per_sample"].tolist() == [
            1 * (static_img_seq_len - 1)
        ]

    def test_unmatched_img_start_raises(self):
        model = _make_dynamic_model()
        # img_start without a matching img_end
        input_ids = torch.tensor(
            [[101, IMG_START_ID, IMG_CONTEXT_ID, IMG_CONTEXT_ID, 102]],
            dtype=torch.int64,
        )
        data = {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([5], dtype=torch.int64),
            "pixel_values": torch.zeros(1, 3, 16, 16),
        }
        with pytest.raises(ValueError, match="Malformed multimodal token sequence"):
            collapse_multimodal_tokens(data, model)

    def test_unmatched_generated_img_start_is_kept_as_text(self):
        model = _make_dynamic_model()
        input_ids = torch.tensor(
            [
                [
                    101,
                    IMG_START_ID,
                    IMG_CONTEXT_ID,
                    IMG_CONTEXT_ID,
                    IMG_END_ID,
                    102,
                    IMG_START_ID,
                    103,
                ]
            ],
            dtype=torch.int64,
        )
        data = {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([8], dtype=torch.int64),
            "token_mask": torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1]], dtype=torch.float32),
            "pixel_values": torch.zeros(1, 3, 16, 16),
        }

        out = collapse_multimodal_tokens(data, model)

        assert out["input_ids"][0].tolist() == [
            101,
            IMG_START_ID,
            IMG_CONTEXT_ID,
            IMG_END_ID,
            102,
            IMG_START_ID,
            103,
        ]
        assert out["input_lengths"].tolist() == [7]
        assert out["tokens_removed_per_sample"].tolist() == [1]

    def test_balanced_generated_img_tokens_are_kept_as_text(self):
        model = _make_dynamic_model()
        input_ids = torch.tensor(
            [
                [
                    101,
                    IMG_START_ID,
                    IMG_CONTEXT_ID,
                    IMG_CONTEXT_ID,
                    IMG_END_ID,
                    102,
                    IMG_START_ID,
                    999,
                    IMG_END_ID,
                    103,
                ]
            ],
            dtype=torch.int64,
        )
        data = {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([10], dtype=torch.int64),
            "token_mask": torch.tensor(
                [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32
            ),
            "pixel_values": torch.zeros(1, 3, 16, 16),
        }

        out = collapse_multimodal_tokens(data, model)

        assert out["input_ids"][0].tolist() == [
            101,
            IMG_START_ID,
            IMG_CONTEXT_ID,
            IMG_END_ID,
            102,
            IMG_START_ID,
            999,
            IMG_END_ID,
            103,
        ]
        assert out["input_lengths"].tolist() == [9]
        assert out["tokens_removed_per_sample"].tolist() == [1]


class TestPatchifyForDynamicResolution:
    def test_single_image_shapes(self):
        h = w = 32  # 32 / 16 = 2 patches per side -> 4 patches total
        images = torch.zeros(1, 3, h, w)
        imgs_sizes = torch.tensor([[h, w]], dtype=torch.int32)
        patches, num_tiles, vision_params = _patchify_for_dynamic_resolution(
            images, imgs_sizes, PATCH_DIM
        )
        # patches: (1, total_patches, channels * patch * patch)
        assert patches.shape == (1, 4, 3 * PATCH_DIM * PATCH_DIM)
        assert num_tiles.tolist() == [1]
        assert vision_params.qkv_format == "thd"
        assert vision_params.cu_seqlens_q.tolist() == [0, 4]
        assert int(vision_params.max_seqlen_q.item()) == 4

    def test_two_images_packed(self):
        # Image A: 32x32 -> 4 patches; Image B: 16x32 -> 2 patches.
        images = torch.zeros(2, 3, 32, 32)
        imgs_sizes = torch.tensor([[32, 32], [16, 32]], dtype=torch.int32)
        patches, num_tiles, vision_params = _patchify_for_dynamic_resolution(
            images, imgs_sizes, PATCH_DIM
        )
        assert patches.shape == (1, 4 + 2, 3 * PATCH_DIM * PATCH_DIM)
        assert num_tiles.tolist() == [1, 1]
        assert vision_params.cu_seqlens_q.tolist() == [0, 4, 6]
        assert int(vision_params.max_seqlen_q.item()) == 4


class TestPrepareMultimodalData:
    def test_no_pixel_values_emits_empty_tensors(self):
        model = _make_dynamic_model()
        device = torch.device("cpu")
        mm: dict = {}
        prepare_multimodal_data(mm, model, device)
        assert mm["images"].numel() == 0
        assert mm["imgs_sizes"].shape == (0, 2)
        assert mm["num_image_tiles"].numel() == 0

    def test_dynamic_resolution_image_only(self):
        model = _make_dynamic_model()
        device = torch.device("cpu")
        h = w = 32
        mm = {
            "pixel_values": torch.zeros(1, 3, h, w),
            "imgs_sizes": torch.tensor([[h, w]], dtype=torch.int32),
        }
        prepare_multimodal_data(mm, model, device)
        # pixel_values is consumed, replaced with packed images.
        assert "pixel_values" not in mm
        assert mm["images"].shape == (1, 4, 3 * PATCH_DIM * PATCH_DIM)
        assert mm["num_image_tiles"].tolist() == [1]
        assert mm["vision_packed_seq_params"].qkv_format == "thd"
        # No num_frames added when temporal_patch_size == 1.
        assert "num_frames" not in mm

    def test_dynamic_resolution_with_temporal_compression_adds_num_frames(self):
        model = _make_dynamic_model(video_temporal_patch_size=2)
        device = torch.device("cpu")
        h = w = 32
        mm = {
            "pixel_values": torch.zeros(1, 3, h, w),
            "imgs_sizes": torch.tensor([[h, w]], dtype=torch.int32),
        }
        prepare_multimodal_data(mm, model, device)
        assert mm["num_frames"].tolist() == [1]

    def test_dynamic_resolution_without_imgs_sizes_raises(self):
        model = _make_dynamic_model()
        device = torch.device("cpu")
        mm = {"pixel_values": torch.zeros(1, 3, 32, 32)}
        with pytest.raises(AssertionError, match="imgs_sizes not provided"):
            prepare_multimodal_data(mm, model, device)

    def test_legacy_private_dynamic_resolution_attributes_remain_supported(self):
        model = SimpleNamespace(
            img_start_token_id=IMG_START_ID,
            img_end_token_id=IMG_END_ID,
            _dynamic_resolution=True,
            _video_temporal_patch_size=2,
            vision_model=SimpleNamespace(patch_dim=PATCH_DIM),
            sound_model=None,
        )
        mm = {
            "pixel_values": torch.zeros(2, 3, 32, 32),
            "imgs_sizes": torch.tensor([[32, 32], [32, 32]], dtype=torch.int32),
            "num_frames": torch.tensor([2], dtype=torch.int32),
        }

        prepare_multimodal_data(mm, model, torch.device("cpu"))

        assert mm["images"].shape == (1, 8, 3 * PATCH_DIM * PATCH_DIM)
        assert mm["num_frames"].tolist() == [2]


class TestVlmSpRepadCollapsed:
    def test_no_op_when_tokens_removed_is_none(self):
        ids = torch.zeros(2, 7, dtype=torch.int64)
        out = _vlm_sp_repad_collapsed(ids, None, divisor=4)
        assert out is ids

    def test_no_op_when_divisor_is_one(self):
        ids = torch.zeros(2, 7, dtype=torch.int64)
        removed = torch.tensor([3, 5], dtype=torch.int64)
        out = _vlm_sp_repad_collapsed(ids, removed, divisor=1)
        assert out is ids

    def test_already_aligned_no_pad_needed(self):
        # collapsed_width=7, max_removed=5 -> expanded=12.
        # round_up(12, 4) = 12, required_width = 12 - 5 = 7. No pad needed.
        ids = torch.zeros(1, 7, dtype=torch.int64)
        removed = torch.tensor([5], dtype=torch.int64)
        out = _vlm_sp_repad_collapsed(ids, removed, divisor=4)
        assert out.shape[1] == 7

    def test_pads_when_required_for_tp_divisor(self):
        # SP+TP=4 case: collapsed_width=6, max_removed=5 -> expanded=11.
        # round_up(11, 4) = 12, required_width = 12 - 5 = 7. Pads from 6 -> 7.
        ids = torch.arange(6, dtype=torch.int64).unsqueeze(0)
        removed = torch.tensor([5], dtype=torch.int64)
        out = _vlm_sp_repad_collapsed(ids, removed, divisor=4)
        assert out.shape == (1, 7)
        assert out[0, -1].item() == 0
        assert out[0, :6].tolist() == list(range(6))

    def test_pads_for_cp_divisor(self):
        # CP=2 case: divisor = cp_size * 2 = 4. Same math as TP=4 case but
        # exercises the helper with the CP-derived divisor.
        ids = torch.arange(10, dtype=torch.int64).unsqueeze(0)
        removed = torch.tensor([255], dtype=torch.int64)
        # collapsed=10, max_removed=255 -> expanded=265.
        # round_up(265, 4) = 268, required_width = 268 - 255 = 13. Pad 10 -> 13.
        out = _vlm_sp_repad_collapsed(ids, removed, divisor=4)
        assert out.shape == (1, 13)
        assert out[0, :10].tolist() == list(range(10))
        assert out[0, 10:].tolist() == [0, 0, 0]
