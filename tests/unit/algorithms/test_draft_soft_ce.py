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

from nemo_rl.algorithms.loss.draft import streaming_vocab_parallel_soft_ce


def _dense_stats(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    bin_ids: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
    numerators = torch.zeros(num_bins, dtype=torch.float32)
    counts = torch.zeros_like(numerators)
    numerators.scatter_add_(0, bin_ids.flatten(), (per_token * mask).flatten())
    counts.scatter_add_(0, bin_ids.flatten(), mask.flatten().float())
    return numerators, counts


@pytest.mark.parametrize(
    "token_chunk_size",
    [pytest.param(10, id="one_tile"), pytest.param(3, id="multiple_tiles")],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streaming_soft_ce_matches_dense_stats_and_gradient(
    token_chunk_size: int,
    dtype: torch.dtype,
) -> None:
    """Both routes match a dense oracle with masks, weights, and an empty bin."""
    generator = torch.Generator().manual_seed(1234)
    student = torch.randn(2, 5, 7, generator=generator, dtype=dtype).requires_grad_(
        True
    )
    teacher = torch.randn(2, 5, 7, generator=generator, dtype=dtype).requires_grad_(
        True
    )
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0]])
    bin_ids = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
    weights = torch.tensor([0.25, 1.5, 0.75])

    stats = streaming_vocab_parallel_soft_ce(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
        bin_ids=bin_ids,
        weights=weights,
        token_chunk_size=token_chunk_size,
        tp_group=None,
    )
    loss = stats.normalized(normalization_counts=stats.counts)
    loss.backward()
    streaming_grad = student.grad.detach().clone()

    reference_student = student.detach().clone().requires_grad_(True)
    expected_numerators, expected_counts = _dense_stats(
        reference_student,
        teacher.detach(),
        mask,
        bin_ids,
        num_bins=3,
    )
    expected_loss = (expected_numerators * weights).sum() / (
        (expected_counts * weights).sum() + 1e-8
    )
    expected_loss.backward()

    torch.testing.assert_close(stats.numerators, expected_numerators)
    torch.testing.assert_close(stats.counts, expected_counts)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(streaming_grad, reference_student.grad)
    assert stats.counts[2] == 0
    assert teacher.grad is None


def test_streaming_soft_ce_bounds_fp32_tiles(monkeypatch) -> None:
    """The uneven final tile never expands to the full token dimension."""
    from nemo_rl.algorithms.loss import draft

    generator = torch.Generator().manual_seed(5678)
    token_chunk_size = 4
    student = torch.randn(1, 2 * token_chunk_size + 1, 11, generator=generator)
    teacher = torch.randn_like(student)
    mask = torch.ones(student.shape[:-1])
    observed_tile_sizes: list[int] = []
    original = draft._tile_log_normalizers

    def record_tile_sizes(*args, **kwargs):
        observed_tile_sizes.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(draft, "_tile_log_normalizers", record_tile_sizes)

    stats = streaming_vocab_parallel_soft_ce(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
        token_chunk_size=token_chunk_size,
        tp_group=None,
    )

    assert observed_tile_sizes == [4, 4, 1]
    assert stats.numerators.shape == stats.counts.shape == stats.weights.shape == (1,)


def test_stats_accept_external_global_counts() -> None:
    """Callers may normalize raw bins with independently reduced counts."""
    student = torch.zeros(1, 3, 5)
    teacher = torch.zeros_like(student)
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    stats = streaming_vocab_parallel_soft_ce(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
        token_chunk_size=2,
        tp_group=None,
    )
    global_counts = torch.tensor([4.0])

    torch.testing.assert_close(
        stats.normalized(normalization_counts=global_counts),
        stats.numerators.sum() / global_counts.sum(),
    )


@pytest.mark.parametrize(
    ("num_tokens", "expected_vocab_distributions"),
    [
        pytest.param(4, 2, id="one_tile_caches_distributions"),
        pytest.param(5, 0, id="multiple_tiles_recompute_distributions"),
    ],
)
def test_soft_ce_routes_at_the_token_chunk_boundary(
    num_tokens: int,
    expected_vocab_distributions: int,
) -> None:
    """The exact tile boundary caches probabilities; larger inputs recompute them."""
    generator = torch.Generator().manual_seed(7890)
    student = torch.randn(
        num_tokens, 7, generator=generator, dtype=torch.bfloat16
    ).requires_grad_(True)
    teacher = torch.randn_like(student).requires_grad_(True)
    mask = torch.ones(num_tokens)
    saved_tensors: list[torch.Tensor] = []

    def record(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        stats = streaming_vocab_parallel_soft_ce(
            student_logits=student,
            teacher_logits=teacher,
            mask=mask,
            token_chunk_size=4,
            tp_group=None,
        )
        stats.normalized(normalization_counts=stats.counts).backward()

    vocab_distributions = [
        tensor
        for tensor in saved_tensors
        if tensor.dtype == torch.float32 and tensor.shape == student.shape
    ]
    assert len(vocab_distributions) == expected_vocab_distributions
    assert any(
        tensor.dtype == torch.float32 and tensor.shape == (num_tokens, 2)
        for tensor in saved_tensors
    ) is (expected_vocab_distributions == 0)
    assert student.grad is not None
    assert student.grad.dtype == torch.bfloat16
    assert teacher.grad is None


def test_multi_tile_soft_ce_saves_only_bounded_fp32_state() -> None:
    """Backward keeps source logits and row log-normalizers, not FP32 vocab tensors."""
    generator = torch.Generator().manual_seed(9012)
    student = torch.randn(2, 3, 9, generator=generator, dtype=torch.bfloat16)
    teacher = torch.randn(2, 3, 9, generator=generator, dtype=torch.bfloat16)
    student.requires_grad_(True)
    teacher.requires_grad_(True)
    mask = torch.ones(2, 3)
    saved_tensors: list[torch.Tensor] = []

    def record(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        stats = streaming_vocab_parallel_soft_ce(
            student_logits=student,
            teacher_logits=teacher,
            mask=mask,
            token_chunk_size=2,
            tp_group=None,
        )
        stats.normalized(normalization_counts=stats.counts).backward()

    assert student.grad is not None
    assert student.grad.dtype == torch.bfloat16
    assert teacher.grad is None
    assert not any(
        tensor.dtype == torch.float32 and tensor.shape == student.shape
        for tensor in saved_tensors
    )
    assert any(
        tensor.dtype == torch.float32 and tensor.shape == (student.numel() // 9, 2)
        for tensor in saved_tensors
    )


def test_raw_stats_are_additive_across_microbatches_with_an_empty_bin() -> None:
    """Raw bins can be accumulated before PR3 supplies global normalization."""
    generator = torch.Generator().manual_seed(3456)
    student = torch.randn(2, 4, 6, generator=generator)
    teacher = torch.randn_like(student)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    bin_ids = torch.zeros_like(mask, dtype=torch.long)
    weights = torch.tensor([0.75, 2.0])

    combined = streaming_vocab_parallel_soft_ce(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
        bin_ids=bin_ids,
        weights=weights,
        token_chunk_size=3,
        tp_group=None,
    )
    microbatches = [
        streaming_vocab_parallel_soft_ce(
            student_logits=student[index : index + 1],
            teacher_logits=teacher[index : index + 1],
            mask=mask[index : index + 1],
            bin_ids=bin_ids[index : index + 1],
            weights=weights,
            token_chunk_size=3,
            tp_group=None,
        )
        for index in range(student.shape[0])
    ]

    torch.testing.assert_close(
        sum((stats.numerators for stats in microbatches), torch.zeros(2)),
        combined.numerators,
    )
    torch.testing.assert_close(
        sum((stats.counts for stats in microbatches), torch.zeros(2)),
        combined.counts,
    )
    assert combined.counts[1] == 0
