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

pytestmark = pytest.mark.mcore

from nemo_rl.models.megatron.draft.dflash import sample_block_anchors  # noqa: E402


def _make_batch(batch_size=3, seq_len=32, gen_start=(10, 20, 31)):
    token_mask = torch.zeros(batch_size, seq_len)
    for row, start in enumerate(gen_start):
        token_mask[row, start:] = 1.0
    sample_mask = torch.ones(batch_size)
    input_ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    return token_mask, sample_mask, input_ids


def test_sample_block_anchors_candidates_and_determinism():
    token_mask, sample_mask, input_ids = _make_batch()
    anchors_a, valid_a = sample_block_anchors(
        token_mask=token_mask,
        sample_mask=sample_mask,
        input_ids=input_ids,
        num_anchors=8,
        generation_only=True,
    )
    anchors_b, valid_b = sample_block_anchors(
        token_mask=token_mask,
        sample_mask=sample_mask,
        input_ids=input_ids,
        num_anchors=8,
        generation_only=True,
    )
    # Deterministic given identical batch content (TP-rank agreement).
    assert torch.equal(anchors_a, anchors_b)
    assert torch.equal(valid_a, valid_b)

    # Every anchor p must have token_mask[p + 1] == 1 (its first label is a
    # trained generation token).
    for row in range(token_mask.shape[0]):
        for anchor in anchors_a[row][valid_a[row]].tolist():
            assert token_mask[row, anchor + 1] == 1.0

    # Row 2 has exactly one candidate (p = 30, label 31): resampling pads to
    # the requested static shape with duplicates, all valid.
    assert valid_a[2].all()
    assert set(anchors_a[2].tolist()) == {30}


def test_sample_block_anchors_rows_without_candidates_are_invalid():
    token_mask, sample_mask, input_ids = _make_batch()
    token_mask[1] = 0.0  # no generation tokens
    sample_mask[2] = 0.0  # invalid sample
    anchors, valid = sample_block_anchors(
        token_mask=token_mask,
        sample_mask=sample_mask,
        input_ids=input_ids,
        num_anchors=4,
        generation_only=True,
    )
    assert valid[0].all()
    assert not valid[1].any()
    assert not valid[2].any()
    # Dummy anchors are in-range so downstream static-shape code stays safe.
    assert int(anchors.max()) < token_mask.shape[1]


def test_sample_block_anchors_changes_with_batch_content():
    token_mask, sample_mask, input_ids = _make_batch()
    anchors_a, _ = sample_block_anchors(
        token_mask=token_mask,
        sample_mask=sample_mask,
        input_ids=input_ids,
        num_anchors=8,
        generation_only=True,
    )
    anchors_b, _ = sample_block_anchors(
        token_mask=token_mask,
        sample_mask=sample_mask,
        input_ids=input_ids + 1,
        num_anchors=8,
        generation_only=True,
    )
    assert not torch.equal(anchors_a, anchors_b)


def test_anchor_count_map_round_trip_preserves_duplicates():
    from nemo_rl.models.megatron.draft.dflash import (
        anchors_to_count_map,
        count_map_to_anchors,
    )

    seq_len = 12
    anchors = torch.tensor([[3, 3, 7, 0], [5, 6, 7, 8]], dtype=torch.int64)
    anchor_valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    count_map = anchors_to_count_map(anchors, anchor_valid, seq_len)
    assert count_map.shape == (2, seq_len)
    # Duplicates accumulate; invalid anchors contribute nothing.
    assert count_map[0, 3] == 2 and count_map[0, 7] == 1 and count_map[0, 0] == 0
    assert count_map.sum() == 3 + 4

    rebuilt, rebuilt_valid = count_map_to_anchors(count_map)
    # Multiset equality per row (order within a row is irrelevant).
    for row in range(2):
        original = sorted(anchors[row][anchor_valid[row]].tolist())
        recovered = sorted(rebuilt[row][rebuilt_valid[row]].tolist())
        assert original == recovered, (row, original, recovered)
    # Row 0 has 3 blocks, row 1 has 4 -> padded to 4 with invalid dummies.
    assert rebuilt.shape == (2, 4)
    assert rebuilt_valid[0].sum() == 3 and rebuilt_valid[1].sum() == 4

    # Truncation along the sequence dim (what dynamic batching does per
    # microbatch) never drops blocks: anchors only sit on valid positions.
    truncated = count_map[:, :9]
    rebuilt_t, rebuilt_valid_t = count_map_to_anchors(truncated)
    assert int(rebuilt_valid_t.sum()) == 7


def test_count_map_to_anchors_handles_empty_microbatch():
    from nemo_rl.models.megatron.draft.dflash import count_map_to_anchors

    count_map = torch.zeros((2, 8), dtype=torch.int32)
    anchors, anchor_valid = count_map_to_anchors(count_map)
    # Non-degenerate static shape with everything masked.
    assert anchors.shape == (2, 1)
    assert not anchor_valid.any()
