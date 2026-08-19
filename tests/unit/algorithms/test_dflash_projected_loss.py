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

import nemo_rl.algorithms.loss as loss_api
from nemo_rl.algorithms.loss import loss_functions
from nemo_rl.algorithms.loss.draft import (
    dflash_projected_vocab_parallel_soft_ce,
)


def test_projected_dflash_loss_is_not_exposed_as_generic_loss_function() -> None:
    assert not hasattr(loss_api, "DFlashProjectedLossFn")
    assert not hasattr(loss_functions, "DFlashProjectedLossFn")


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
    # Official DFlash KD alignment: the teacher logit predicting token p is row p - 1.
    teacher_rows = torch.tensor([[0, 5, 6], [0, 0, 1]])
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


def test_dflash_adapter_excludes_anchor_even_when_input_mask_includes_it() -> None:
    """The anchor remains context only without a device-to-host validation sync."""
    inputs = list(_inputs())
    expected = dflash_projected_vocab_parallel_soft_ce(
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
    inputs[5] = inputs[5].clone()
    inputs[5][0, 0] = True

    actual = dflash_projected_vocab_parallel_soft_ce(
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

    torch.testing.assert_close(actual.numerators, expected.numerators)
    torch.testing.assert_close(actual.counts, expected.counts)


def test_dflash_adapter_ignores_invalid_metadata_for_inactive_slots() -> None:
    """Padding sentinels cannot index teacher rows or contribute gradients."""
    generator = torch.Generator().manual_seed(31415)
    draft_hidden = torch.randn(2, 3, 4, generator=generator).requires_grad_(True)
    output_weight = torch.randn(7, 4, generator=generator)
    teacher_logits = torch.randn(2, 4, 7, generator=generator)
    sample_rows = torch.tensor([0, 2])
    label_positions = torch.tensor(
        [
            [-999, 1, 5],
            [-999, -3, 3],
        ]
    )
    loss_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ]
    )

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
    reference = dflash_projected_vocab_parallel_soft_ce(
        draft_hidden=draft_hidden.detach().clone(),
        output_weight=output_weight,
        teacher_logits=teacher_logits,
        sample_rows=torch.tensor([0, 0]),
        label_positions=torch.tensor([[1, 1, 1], [1, 1, 1]]),
        loss_mask=loss_mask,
        position_decay=0.5,
        token_chunk_size=2,
        tp_group=None,
    )

    torch.testing.assert_close(stats.numerators, reference.numerators)
    torch.testing.assert_close(stats.counts, torch.tensor([1.0, 0.0]))
    stats.normalized(normalization_counts=stats.counts).backward()
    assert draft_hidden.grad is not None
    assert torch.count_nonzero(draft_hidden.grad[0, 1]) > 0
    assert torch.count_nonzero(draft_hidden.grad[:, 0]) == 0
    assert torch.count_nonzero(draft_hidden.grad[0, 2]) == 0
    assert torch.count_nonzero(draft_hidden.grad[1]) == 0


def test_dflash_adapter_keeps_invalid_active_rows_failing() -> None:
    """Safety mapping must not hide malformed metadata for a trained slot."""
    inputs = list(_inputs())
    inputs[3] = torch.tensor([1, 9])
    inputs[5] = inputs[5].clone()
    inputs[5][1, 1] = True

    with pytest.raises((IndexError, RuntimeError), match="(?i)index|range"):
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


@pytest.mark.parametrize(
    ("sample_row", "label_position"),
    [
        pytest.param(2, -3, id="invalid_sample_aliases_valid_flat_row"),
        pytest.param(0, 5, id="invalid_position_aliases_next_sample"),
    ],
)
def test_dflash_adapter_rejects_active_coordinate_aliases(
    sample_row: int,
    label_position: int,
) -> None:
    """Each active coordinate component must be valid before flattening."""
    inputs = list(_inputs())
    inputs[3] = inputs[3].clone()
    inputs[3][0] = sample_row
    inputs[4] = inputs[4].clone()
    inputs[4][0, 1] = label_position
    inputs[5] = inputs[5].clone()
    inputs[5][0, 2] = False

    with pytest.raises((IndexError, RuntimeError), match="(?i)index|range"):
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
