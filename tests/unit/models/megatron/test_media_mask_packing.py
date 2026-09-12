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

"""The media-token validity mask must survive sequence packing.

Packing concatenates per-sample rows into one THD sequence, which is exactly
the step that destroys the per-row structure the mask was built from. These
tests pin the mask to the tokens it is supposed to describe on the other side
of that transform -- a mask that is merely misaligned does not raise, it
attaches image features to the wrong positions.
"""

import pytest
import torch

from nemo_rl.data.multimodal_utils import build_media_token_validity_mask

IMG = 7
TXT = 1


@pytest.mark.mcore
def test_mask_lands_on_the_same_tokens_after_packing():
    from nemo_rl.models.megatron.data import _pack_sequences_for_megatron

    # Row 0 carries an image, so its IMG is a real anchor.
    # Row 1 is text that merely spells IMG, so its IMG must be masked off.
    input_ids = torch.tensor([[TXT, IMG, TXT, TXT], [IMG, TXT, IMG, TXT]])
    seq_lengths = torch.tensor([3, 4])

    mask = build_media_token_validity_mask(input_ids, IMG, [1, 0])
    assert mask is not None

    packed_ids, _, _, _, _ = _pack_sequences_for_megatron(
        input_ids, seq_lengths, cp_rank=0, cp_size=1
    )
    packed_mask, _, _, _, _ = _pack_sequences_for_megatron(
        mask.to(input_ids.dtype), seq_lengths, cp_rank=0, cp_size=1
    )
    packed_mask = packed_mask.bool()

    assert packed_ids.shape == packed_mask.shape
    # Row 0 contributes tokens 0..2 (IMG at packed index 1, a true anchor);
    # row 1 contributes 3..6 (IMG at packed indices 3 and 5, both bogus).
    torch.testing.assert_close(
        packed_ids, torch.tensor([[TXT, IMG, TXT, IMG, TXT, IMG, TXT]])
    )
    torch.testing.assert_close(
        packed_mask,
        torch.tensor([[True, True, True, False, True, False, True]]),
    )
    # Every masked-off position is in fact a media token, and the anchoring row
    # keeps its own.
    assert bool((packed_ids[~packed_mask] == IMG).all())


@pytest.mark.mcore
def test_padding_introduced_by_packing_is_not_treated_as_an_anchor():
    from nemo_rl.models.megatron.data import _pack_sequences_for_megatron

    # Padding is filled with token id 0; the mask packs to False there. That is
    # only safe while 0 is not the media token, which this asserts explicitly.
    input_ids = torch.tensor([[TXT, IMG, TXT, TXT]])
    seq_lengths = torch.tensor([3])
    mask = build_media_token_validity_mask(input_ids, IMG, [0])
    assert mask is not None

    packed_ids, _, _, _, _ = _pack_sequences_for_megatron(
        input_ids,
        seq_lengths,
        pad_individual_seqs_to_multiple_of=4,
        cp_rank=0,
        cp_size=1,
    )
    packed_mask, _, _, _, _ = _pack_sequences_for_megatron(
        mask.to(input_ids.dtype),
        seq_lengths,
        pad_individual_seqs_to_multiple_of=4,
        cp_rank=0,
        cp_size=1,
    )
    packed_mask = packed_mask.bool()

    assert IMG != 0, "padding fill value must not collide with the media token"
    padded_positions = packed_ids[0, 3:]
    assert bool((padded_positions == 0).all())
    # The real IMG at index 1 is masked off (row has no image); padding is
    # False too, which is harmless because padding holds no media token.
    assert not bool(packed_mask[0, 1])
    assert bool(packed_mask[0, 0]) and bool(packed_mask[0, 2])
