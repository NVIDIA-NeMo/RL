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

from nemo_rl.algorithms.loss.draft import streaming_vocab_parallel_soft_ce


def _run_tp2_soft_ce(
    rank: int,
    world_size: int,
    token_chunk_size: int,
) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    batch_size, sequence_length, vocab_size = 2, 5, 16
    local_vocab_size = vocab_size // world_size
    vocab_start = rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size

    generator = torch.Generator(device="cuda").manual_seed(2468)
    full_student = torch.randn(
        batch_size,
        sequence_length,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    full_teacher = torch.randn(
        batch_size,
        sequence_length,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_teacher[..., local_vocab_size - 1] = 8.0
    full_teacher[..., local_vocab_size] = 9.0
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 0.0]],
        device="cuda",
    )
    bin_ids = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]],
        device="cuda",
    )
    weights = torch.tensor([0.5, 1.25, 0.75], device="cuda")

    teacher_probs = torch.softmax(full_teacher.float(), dim=-1)
    student_log_probs = torch.log_softmax(full_student.float(), dim=-1)
    per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
    expected_numerators = torch.zeros(3, device="cuda")
    expected_counts = torch.zeros(3, device="cuda")
    expected_numerators.scatter_add_(0, bin_ids.flatten(), (per_token * mask).flatten())
    expected_counts.scatter_add_(0, bin_ids.flatten(), mask.flatten())
    expected_loss = (expected_numerators * weights).sum() / (
        (expected_counts * weights).sum() + 1e-8
    )
    expected_loss.backward()
    expected_local_gradient = full_student.grad[..., vocab_start:vocab_end].clone()

    local_student = (
        full_student.detach()[..., vocab_start:vocab_end].clone().requires_grad_(True)
    )
    local_teacher = (
        full_teacher[..., vocab_start:vocab_end].clone().requires_grad_(True)
    )
    stats = streaming_vocab_parallel_soft_ce(
        student_logits=local_student,
        teacher_logits=local_teacher,
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
    torch.testing.assert_close(local_student.grad, expected_local_gradient)
    assert stats.counts[2] == 0
    assert local_teacher.grad is None


@pytest.mark.parametrize(
    "token_chunk_size",
    [pytest.param(10, id="one_tile"), pytest.param(3, id="multiple_tiles")],
)
def test_tp2_streaming_soft_ce_forward_and_backward(
    distributed_test_runner,
    token_chunk_size: int,
) -> None:
    distributed_test_runner(
        partial(_run_tp2_soft_ce, token_chunk_size=token_chunk_size),
        world_size=2,
    )
