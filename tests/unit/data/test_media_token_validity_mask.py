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

from nemo_rl.data.multimodal_utils import build_media_token_validity_mask

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
