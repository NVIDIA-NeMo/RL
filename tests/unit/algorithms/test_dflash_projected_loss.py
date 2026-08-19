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

from nemo_rl.algorithms.loss.draft import (
    dflash_projected_vocab_parallel_soft_ce,
)
from nemo_rl.algorithms.loss.loss_functions import DFlashProjectedLossFn


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(24601)
    draft_hidden = torch.randn(2, 3, 4, generator=generator).requires_grad_(True)
    output_weight = torch.randn(7, 4, generator=generator).requires_grad_(True)
    teacher_logits = torch.randn(2, 4, 7, generator=generator).requires_grad_(True)
    sample_rows = torch.tensor([1, 0])
    label_positions = torch.tensor([[-1, 2, 3], [-1, 1, 2]])
    loss_mask = torch.tensor(
        [[False, True, True], [False, True, False]],
    )
    return (
        draft_hidden,
        output_weight,
        teacher_logits,
        sample_rows,
        label_positions,
        loss_mask,
    )


def test_dflash_adapter_maps_blocks_to_teacher_rows_and_position_bins() -> None:
    """Block metadata selects exact teacher rows and excludes the anchor slot."""
    (
        draft_hidden,
        output_weight,
        teacher_logits,
        sample_rows,
        label_positions,
        loss_mask,
    ) = _inputs()

    stats = dflash_projected_vocab_parallel_soft_ce(
        draft_hidden=draft_hidden,
        output_weight=output_weight,
        teacher_logits=teacher_logits,
        sample_rows=sample_rows,
        label_positions=label_positions,
        loss_mask=loss_mask,
        position_decay=0.5,
        token_chunk_size=2,
        tp_group=None,
    )

    student_logits = draft_hidden.reshape(-1, draft_hidden.shape[-1]) @ output_weight.T
    teacher_rows = torch.tensor([[5, 6, 7], [0, 1, 2]])
    selected_teacher = teacher_logits.reshape(
        -1, teacher_logits.shape[-1]
    ).index_select(
        0,
        teacher_rows.reshape(-1),
    )
    per_slot = (
        -(
            torch.softmax(selected_teacher.detach(), dim=-1)
            * torch.log_softmax(student_logits, dim=-1)
        )
        .sum(dim=-1)
        .reshape_as(loss_mask)
    )
    expected_numerators = torch.stack(
        (
            (per_slot[:, 1] * loss_mask[:, 1]).sum(),
            (per_slot[:, 2] * loss_mask[:, 2]).sum(),
        )
    )

    torch.testing.assert_close(stats.numerators, expected_numerators)
    torch.testing.assert_close(stats.counts, torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(stats.weights, torch.tensor([1.0, 0.5]))
    stats.normalized(normalization_counts=stats.counts).backward()
    assert draft_hidden.grad is not None
    assert teacher_logits.grad is None
    assert output_weight.grad is None


def test_dflash_loss_fn_normalizes_with_external_per_slot_counts() -> None:
    """The production loss seam honors globally reduced per-slot counts."""
    inputs = _inputs()
    loss_fn = DFlashProjectedLossFn(
        vocab_parallel_group=None,
        token_chunk_size=2,
        position_decay=0.5,
    )
    global_counts = torch.tensor([4.0, 8.0])

    loss = loss_fn(
        draft_hidden=inputs[0],
        output_weight=inputs[1],
        teacher_logits=inputs[2],
        sample_rows=inputs[3],
        label_positions=inputs[4],
        loss_mask=inputs[5],
        global_normalization_counts=global_counts,
    )
    raw_stats = dflash_projected_vocab_parallel_soft_ce(
        draft_hidden=inputs[0],
        output_weight=inputs[1],
        teacher_logits=inputs[2],
        sample_rows=inputs[3],
        label_positions=inputs[4],
        loss_mask=inputs[5],
        position_decay=0.5,
        token_chunk_size=2,
        tp_group=None,
    )

    torch.testing.assert_close(
        loss,
        raw_stats.normalized(normalization_counts=global_counts),
    )


def test_dflash_adapter_rejects_a_trained_anchor_slot() -> None:
    """The anchor is context only and can never contribute to a loss bin."""
    inputs = list(_inputs())
    inputs[5] = inputs[5].clone()
    inputs[5][0, 0] = True

    with pytest.raises(ValueError, match="anchor slot"):
        dflash_projected_vocab_parallel_soft_ce(
            draft_hidden=inputs[0],
            output_weight=inputs[1],
            teacher_logits=inputs[2],
            sample_rows=inputs[3],
            label_positions=inputs[4],
            loss_mask=inputs[5],
            position_decay=0.5,
            token_chunk_size=2,
            tp_group=None,
        )
