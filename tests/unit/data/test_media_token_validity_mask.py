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

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    attach_media_token_validity_mask,
    build_media_token_validity_mask,
    chunks_accept_media_token_validity_mask,
    image_counts_by_row,
    media_placeholder_token_id_from_chunks,
)

IMG = 7  # stand-in media token id
TXT = 1


def test_returns_base_mask_when_every_row_has_media():
    # Nothing to correct: each row's media token really does mark a feature.
    input_ids = torch.tensor([[TXT, IMG, TXT], [IMG, TXT, TXT]])
    assert build_media_token_validity_mask(input_ids, IMG, [1, 1]) is None


def test_returns_base_mask_when_text_rows_do_not_spell_the_token():
    # Text-only rows exist, but none contain the token, so the model's own
    # derivation is already right and we must not hand it a mask.
    input_ids = torch.tensor([[TXT, IMG, TXT], [TXT, TXT, TXT]])
    assert build_media_token_validity_mask(input_ids, IMG, [1, 0]) is None


def test_masks_the_token_only_in_rows_without_media():
    # Row 0 has an image so its token is a real placeholder; row 1 has none, so
    # its identical token is just prose the author wrote.
    input_ids = torch.tensor([[TXT, IMG, TXT], [IMG, TXT, IMG]])
    mask = build_media_token_validity_mask(input_ids, IMG, [1, 0])
    assert mask is not None
    torch.testing.assert_close(
        mask,
        torch.tensor([[True, True, True], [False, True, False]]),
    )


def test_refines_rather_than_replaces_a_base_mask():
    # A caller combining modalities must keep the positions the base mask
    # already invalidated.
    input_ids = torch.tensor([[IMG, TXT], [IMG, TXT]])
    base = torch.tensor([[True, False], [True, True]])
    mask = build_media_token_validity_mask(input_ids, IMG, [1, 0], base_mask=base)
    assert mask is not None
    torch.testing.assert_close(
        mask,
        torch.tensor([[True, False], [False, True]]),
    )
    # The caller's tensor must not be mutated in place.
    torch.testing.assert_close(base, torch.tensor([[True, False], [True, True]]))


def test_rejects_non_2d_input_ids():
    with pytest.raises(ValueError, match=r"input_ids must be \[B, S\]"):
        build_media_token_validity_mask(torch.tensor([TXT, IMG]), IMG, [0])


def test_rejects_count_length_mismatch():
    with pytest.raises(ValueError, match="one entry per row"):
        build_media_token_validity_mask(torch.tensor([[TXT, IMG]]), IMG, [0, 1])


# --------------------------------------------------------------------------
# capability probes and batch-level attach
# --------------------------------------------------------------------------


class _ChunkWithMask:
    image_token_index = IMG

    def forward(self, input_ids, *, media_token_validity_mask=None):
        raise AssertionError("not called")


class _ChunkWithoutMask:
    image_token_index = IMG

    def forward(self, input_ids):
        raise AssertionError("not called")


def test_placeholder_token_id_is_read_from_the_first_chunk_that_has_one():
    assert media_placeholder_token_id_from_chunks([_ChunkWithMask()]) == IMG
    # A chunk without the attribute must not stop the search.
    assert media_placeholder_token_id_from_chunks([object(), _ChunkWithMask()]) == IMG


def test_placeholder_token_id_is_none_when_no_chunk_declares_one():
    assert media_placeholder_token_id_from_chunks([object(), object()]) is None
    assert media_placeholder_token_id_from_chunks([]) is None


def test_capability_is_read_from_the_forward_signature():
    """A flag could lie; the signature is what decides whether the kwarg lands."""
    assert chunks_accept_media_token_validity_mask([_ChunkWithMask()]) is True
    assert chunks_accept_media_token_validity_mask([_ChunkWithoutMask()]) is False
    assert chunks_accept_media_token_validity_mask([]) is False


def test_capability_skips_chunks_whose_signature_cannot_be_read():
    """An unintrospectable forward must not abort the search."""

    class _Opaque:
        forward = print  # builtins raise ValueError from inspect.signature

    assert (
        chunks_accept_media_token_validity_mask([_Opaque(), _ChunkWithMask()]) is True
    )


def test_image_counts_treats_a_batch_without_pixel_values_as_text_only():
    """No images anywhere is the case the mask exists for, not missing data."""
    assert image_counts_by_row({}, 3) == [0, 0, 0]


def test_image_counts_returns_none_for_unreadable_media():
    """Rather than guess a count and mask against it."""
    assert image_counts_by_row({"pixel_values": torch.ones(2, 3)}, 2) is None


def test_image_counts_reads_logical_segments_per_row():
    packed = PackedTensor([torch.ones(1, 3, 2, 2)], dim_to_pack=0)
    assert image_counts_by_row({"pixel_values": packed}, 1) == [1]


def test_image_counts_returns_none_on_row_count_mismatch():
    packed = PackedTensor([torch.ones(1, 3, 2, 2)], dim_to_pack=0)
    assert image_counts_by_row({"pixel_values": packed}, 2) is None


def test_attach_sets_the_mask_for_a_text_row_that_spells_the_token():
    batch = {"input_ids": torch.tensor([[TXT, IMG, TXT]])}
    attach_media_token_validity_mask(batch, IMG)
    torch.testing.assert_close(
        batch["media_token_validity_mask"],
        torch.tensor([[True, False, True]]),
    )


def test_attach_is_a_noop_without_a_media_token_id():
    """Models that never declare the kwarg must not get a mask."""
    batch = {"input_ids": torch.tensor([[TXT, IMG]])}
    attach_media_token_validity_mask(batch, None)
    assert "media_token_validity_mask" not in batch


def test_attach_is_a_noop_when_nothing_needs_masking():
    """No key at all, so the model keeps deriving its own."""
    packed = PackedTensor([torch.ones(1, 3, 2, 2)], dim_to_pack=0)
    batch = {"input_ids": torch.tensor([[TXT, IMG]]), "pixel_values": packed}
    attach_media_token_validity_mask(batch, IMG)
    assert "media_token_validity_mask" not in batch


def test_attach_ignores_a_batch_whose_input_ids_are_not_2d():
    batch = {"input_ids": torch.tensor([TXT, IMG])}
    attach_media_token_validity_mask(batch, IMG)
    assert "media_token_validity_mask" not in batch
