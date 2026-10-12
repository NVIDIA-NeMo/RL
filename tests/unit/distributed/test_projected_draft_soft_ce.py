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

from functools import partial
from unittest.mock import patch

import pytest
import torch

from nemo_rl.algorithms.loss.draft import (
    dflash_projected_vocab_parallel_soft_ce,
    projected_streaming_vocab_parallel_soft_ce,
)


def _run_tp2_projected_soft_ce(
    rank: int,
    world_size: int,
    token_chunk_size: int,
) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    num_tokens, teacher_rows = 5, 4
    hidden_size, vocab_size = 6, 16
    local_vocab_size = vocab_size // world_size
    vocab_start = rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size

    generator = torch.Generator(device="cuda").manual_seed(8642)
    student_hidden = torch.randn(
        num_tokens,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_output_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_teacher_logits = torch.randn(
        teacher_rows,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_teacher_logits[:, local_vocab_size - 1] = 8.0
    full_teacher_logits[:, local_vocab_size] = 9.0
    teacher_row_indices = torch.tensor([3, 0, 3, 1, 2], device="cuda")
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0], device="cuda")
    bin_ids = torch.tensor([0, 0, 0, 1, 1], device="cuda")
    weights = torch.tensor([0.5, 1.25, 0.75], device="cuda")

    reference_hidden = student_hidden.detach().clone().requires_grad_(True)
    selected_teacher = full_teacher_logits.index_select(0, teacher_row_indices)
    teacher_probs = torch.softmax(selected_teacher.float(), dim=-1)
    student_log_probs = torch.log_softmax(
        reference_hidden @ full_output_weight.T,
        dim=-1,
        dtype=torch.float32,
    )
    per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
    expected_numerators = torch.zeros(3, device="cuda")
    expected_counts = torch.zeros(3, device="cuda")
    expected_numerators.scatter_add_(0, bin_ids, per_token * mask)
    expected_counts.scatter_add_(0, bin_ids, mask)
    expected_loss = (expected_numerators * weights).sum() / (
        (expected_counts * weights).sum() + 1e-8
    )
    row_scale = (
        weights.index_select(0, bin_ids)
        * mask
        / ((expected_counts * weights).sum() + 1e-8)
    )
    logits_gradient = (student_log_probs.exp() - teacher_probs).mul(
        row_scale.unsqueeze(-1)
    )
    if token_chunk_size >= num_tokens:
        expected_hidden_gradient = (
            logits_gradient[:, vocab_start:vocab_end].to(student_hidden.dtype)
            @ full_output_weight[vocab_start:vocab_end]
        )
        torch.distributed.all_reduce(
            expected_hidden_gradient,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
    else:
        expected_hidden_gradient = (logits_gradient @ full_output_weight.float()).to(
            student_hidden.dtype
        )
    expected_loss.backward()

    local_output_weight = (
        full_output_weight[vocab_start:vocab_end].clone().requires_grad_(True)
    )
    local_teacher_logits = (
        full_teacher_logits[:, vocab_start:vocab_end].clone().requires_grad_(True)
    )
    local_hidden = student_hidden.detach().clone().requires_grad_(True)
    stats = projected_streaming_vocab_parallel_soft_ce(
        student_hidden=local_hidden,
        output_weight=local_output_weight,
        selected_teacher_logits=local_teacher_logits.index_select(
            0, teacher_row_indices
        ),
        mask=mask,
        bin_ids=bin_ids,
        weights=weights,
        token_chunk_size=token_chunk_size,
        tp_group=tp_group,
    )
    loss = stats.normalized(normalization_counts=stats.counts)
    loss.backward()

    torch.testing.assert_close(stats.numerators, expected_numerators)
    torch.testing.assert_close(stats.counts, expected_counts)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(local_hidden.grad, expected_hidden_gradient)
    assert stats.counts[2] == 0
    assert local_teacher_logits.grad is None
    assert local_output_weight.grad is None


@pytest.mark.parametrize(
    "token_chunk_size",
    [pytest.param(5, id="one_tile"), pytest.param(2, id="multiple_tiles")],
)
def test_tp2_projected_soft_ce_sums_hidden_gradient(
    distributed_test_runner,
    token_chunk_size: int,
) -> None:
    distributed_test_runner(
        partial(
            _run_tp2_projected_soft_ce,
            token_chunk_size=token_chunk_size,
        ),
        world_size=2,
    )


def _run_tp2_dflash_projected_soft_ce(rank: int, world_size: int) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    hidden_size, vocab_size = 6, 16
    local_vocab_size = vocab_size // world_size
    vocab_start = rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    generator = torch.Generator(device="cuda").manual_seed(11235)
    draft_hidden = torch.randn(
        2,
        3,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_output_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_teacher_logits = torch.randn(
        2,
        4,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sample_rows = torch.tensor([1, 0], device="cuda")
    label_positions = torch.tensor(
        [[-1, 2, 3], [-1, 1, 2]],
        device="cuda",
    )
    loss_mask = torch.tensor(
        [[False, True, True], [False, True, False]],
        device="cuda",
    )

    reference_hidden = draft_hidden.detach().clone().requires_grad_(True)
    student_logits = reference_hidden.reshape(-1, hidden_size) @ full_output_weight.T
    teacher_rows = torch.tensor([[0, 5, 6], [0, 0, 1]], device="cuda")
    selected_teacher = full_teacher_logits.reshape(-1, vocab_size).index_select(
        0, teacher_rows.reshape(-1)
    )
    per_slot = (
        -(
            torch.softmax(selected_teacher.float(), dim=-1)
            * torch.log_softmax(student_logits.float(), dim=-1)
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
    expected_counts = torch.tensor([2.0, 1.0], device="cuda")
    expected_weights = torch.tensor([1.0, 0.5], device="cuda")
    expected_loss = (expected_numerators * expected_weights).sum() / (
        (expected_counts * expected_weights).sum() + 1e-8
    )
    expected_loss.backward()

    local_hidden = draft_hidden.detach().clone().requires_grad_(True)
    local_output_weight = (
        full_output_weight[vocab_start:vocab_end].clone().requires_grad_(True)
    )
    local_teacher_logits = (
        full_teacher_logits[..., vocab_start:vocab_end].clone().requires_grad_(True)
    )
    stats = dflash_projected_vocab_parallel_soft_ce(
        draft_hidden=local_hidden,
        output_weight=local_output_weight,
        teacher_logits=local_teacher_logits,
        sample_rows=sample_rows,
        label_positions=label_positions,
        loss_mask=loss_mask,
        position_decay=0.5,
        token_chunk_size=2,
        tp_group=tp_group,
    )
    loss = stats.normalized(normalization_counts=stats.counts)
    loss.backward()

    torch.testing.assert_close(stats.numerators, expected_numerators)
    torch.testing.assert_close(stats.counts, expected_counts)
    torch.testing.assert_close(stats.weights, expected_weights)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(
        local_hidden.grad,
        reference_hidden.grad,
        rtol=2e-2,
        atol=2e-2,
    )
    assert local_output_weight.grad is None
    assert local_teacher_logits.grad is None


def test_tp2_dflash_wrapper_matches_dense_hidden_gradient(
    distributed_test_runner,
) -> None:
    distributed_test_runner(_run_tp2_dflash_projected_soft_ce, world_size=2)


def _run_tp2_dflash_invalid_coordinate_aliases(rank: int, world_size: int) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    draft_hidden = torch.ones(1, 2, 3, requires_grad=True)
    output_weight = torch.ones(4, 3)
    teacher_logits = torch.ones(2, 4, 4)
    loss_mask = torch.tensor([[False, True]])

    for sample_row, label_position in ((2, -3), (0, 5)):
        with pytest.raises((IndexError, RuntimeError), match="(?i)index|range"):
            dflash_projected_vocab_parallel_soft_ce(
                draft_hidden=draft_hidden,
                output_weight=output_weight,
                teacher_logits=teacher_logits,
                sample_rows=torch.tensor([sample_row]),
                label_positions=torch.tensor([[-1, label_position]]),
                loss_mask=loss_mask,
                position_decay=0.5,
                token_chunk_size=1,
                tp_group=tp_group,
            )


def test_tp2_dflash_wrapper_rejects_coordinate_aliases_together(
    distributed_test_runner,
) -> None:
    distributed_test_runner(
        _run_tp2_dflash_invalid_coordinate_aliases,
        world_size=2,
        backend="gloo",
    )


def _run_tp2_dflash_position_decay_mismatch(rank: int, world_size: int) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda")

    with pytest.raises(ValueError, match="TP ranks disagree"):
        dflash_projected_vocab_parallel_soft_ce(
            draft_hidden=torch.ones(1, 2, 3, device=device),
            output_weight=torch.ones(4, 3, device=device),
            teacher_logits=torch.ones(1, 2, 4, device=device),
            sample_rows=torch.tensor([0], device=device),
            label_positions=torch.tensor([[-1, 1]], device=device),
            loss_mask=torch.tensor([[False, True]], device=device),
            position_decay=0.5 if rank == 0 else 0.0,
            token_chunk_size=1,
            tp_group=tp_group,
        )


def test_tp2_dflash_wrapper_rejects_position_decay_mismatch_together(
    distributed_test_runner,
) -> None:
    distributed_test_runner(
        _run_tp2_dflash_position_decay_mismatch,
        world_size=2,
    )


def _run_tp2_projected_soft_ce_metadata_mismatch(
    rank: int,
    world_size: int,
    field: str,
) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    student_hidden = torch.ones(5, 3, device="cuda", dtype=torch.bfloat16)
    output_weight = torch.ones(8, 3, device="cuda", dtype=torch.bfloat16)
    teacher_logits = torch.ones(5, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(5, device="cuda")
    bin_ids = torch.tensor([0, 0, 1, 1, 1], device="cuda")
    weights = torch.tensor([1.0, 0.5], device="cuda")
    token_chunk_size = 2
    if rank == 1:
        if field == "mask":
            mask[-1] = 0
        elif field == "bin_ids":
            bin_ids[-1] = 0
        elif field == "weights":
            weights[-1] = 0.25
        elif field == "token_chunk_size":
            token_chunk_size = 3
        elif field == "shape":
            teacher_logits = teacher_logits[:-1]

    with pytest.raises(ValueError, match="TP ranks disagree"):
        projected_streaming_vocab_parallel_soft_ce(
            student_hidden=student_hidden,
            output_weight=output_weight,
            selected_teacher_logits=teacher_logits,
            mask=mask,
            bin_ids=bin_ids,
            weights=weights,
            token_chunk_size=token_chunk_size,
            tp_group=tp_group,
        )


@pytest.mark.parametrize(
    "field", ["mask", "bin_ids", "weights", "token_chunk_size", "shape"]
)
def test_tp2_projected_soft_ce_rejects_rank_local_metadata(
    distributed_test_runner,
    field: str,
) -> None:
    distributed_test_runner(
        partial(_run_tp2_projected_soft_ce_metadata_mismatch, field=field),
        world_size=2,
    )


def _run_tp2_projected_soft_ce_high_rank_mismatch(
    rank: int,
    world_size: int,
) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    fifth_dimension = rank + 1
    student_hidden = torch.ones(1, 1, 1, 1, fifth_dimension, 3)
    output_weight = torch.ones(8, 3)
    teacher_logits = torch.ones(1, 1, 1, 1, fifth_dimension, 8)
    mask = torch.ones(1, 1, 1, 1, fifth_dimension)
    bin_ids = torch.zeros_like(mask, dtype=torch.long)
    weights = torch.ones(1)

    with patch.object(
        torch.distributed,
        "broadcast",
        side_effect=AssertionError("unsafe variable-sized broadcast entered"),
    ):
        with pytest.raises(ValueError, match="at most four dimensions"):
            projected_streaming_vocab_parallel_soft_ce(
                student_hidden=student_hidden,
                output_weight=output_weight,
                selected_teacher_logits=teacher_logits,
                mask=mask,
                bin_ids=bin_ids,
                weights=weights,
                token_chunk_size=1,
                tp_group=tp_group,
            )


def test_tp2_projected_soft_ce_rejects_high_rank_metadata_before_broadcast(
    distributed_test_runner,
) -> None:
    distributed_test_runner(
        _run_tp2_projected_soft_ce_high_rank_mismatch,
        world_size=2,
        backend="gloo",
    )
