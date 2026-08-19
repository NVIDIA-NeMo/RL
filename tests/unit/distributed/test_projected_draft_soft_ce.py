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

import pytest
import torch

from nemo_rl.algorithms.loss.draft import (
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
        teacher_logits=local_teacher_logits,
        teacher_row_indices=teacher_row_indices,
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
