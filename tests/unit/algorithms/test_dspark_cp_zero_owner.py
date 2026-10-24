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

"""Zero-owner CP ranks retain the DSpark graph and fixed collective bins."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.loss.dspark import dspark_tiled_objective

pytestmark = pytest.mark.mcore


def test_dspark_zero_owner_keeps_fixed_bins_and_zero_gradients() -> None:
    block_size, hidden_size, vocab_size, markov_rank = 3, 4, 8, 2
    draft_hidden = torch.empty(
        (0, block_size, hidden_size),
        requires_grad=True,
    )
    markov_w1 = torch.randn(vocab_size, markov_rank, requires_grad=True)
    markov_w2 = torch.randn(vocab_size, markov_rank, requires_grad=True)
    confidence_logits = torch.empty((0, block_size), requires_grad=True)

    stats = dspark_tiled_objective(
        target_logits=torch.empty((0, block_size, vocab_size)),
        draft_hidden=draft_hidden,
        target_output_weight=torch.randn(vocab_size, hidden_size),
        markov_w1=markov_w1,
        markov_w2=markov_w2,
        previous_token_ids=torch.empty((0, block_size), dtype=torch.int64),
        hard_labels=torch.empty((0, block_size), dtype=torch.int64),
        confidence_logits=confidence_logits,
        valid_mask=torch.empty((0, block_size), dtype=torch.bool),
        slot_bins=torch.empty((0, block_size), dtype=torch.int64),
        slot_weights=torch.ones(block_size),
        loss_weights=(0.1, 0.9, 1.0),
        token_chunk_size=2,
        draft_vocab_start_index=0,
        tp_group=None,
    )

    assert stats.combined.numerators.shape == (block_size,)
    assert torch.equal(stats.combined.counts, torch.zeros(block_size))
    stats.combined.normalized(
        normalization_counts=stats.combined.counts,
    ).backward()
    for tensor in (draft_hidden, markov_w1, markov_w2, confidence_logits):
        assert tensor.grad is not None
        assert torch.equal(tensor.grad, torch.zeros_like(tensor))
