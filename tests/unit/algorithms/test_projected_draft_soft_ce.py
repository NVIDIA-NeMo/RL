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
    projected_streaming_vocab_parallel_soft_ce,
)


def _dense_projected_stats(
    student_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    selected_teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    bin_ids: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_logits = student_hidden.reshape(-1, student_hidden.shape[-1]) @ (
        output_weight.T
    )
    teacher_probs = torch.softmax(selected_teacher_logits.float(), dim=-1)
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
    numerators = torch.zeros(num_bins, dtype=torch.float32)
    counts = torch.zeros_like(numerators)
    numerators.scatter_add_(0, bin_ids.reshape(-1), per_token * mask.reshape(-1))
    counts.scatter_add_(0, bin_ids.reshape(-1), mask.reshape(-1).float())
    return numerators, counts


@pytest.mark.parametrize(
    ("num_tokens", "expected_grad_fn"),
    [
        pytest.param(4, "_CachedVocabParallelSoftCEBackward", id="one_tile_native"),
        pytest.param(
            5,
            "_StreamingProjectedVocabParallelSoftCEBackward",
            id="multiple_tiles_projected",
        ),
    ],
)
def test_projected_soft_ce_routes_at_tile_boundary(
    num_tokens: int,
    expected_grad_fn: str,
) -> None:
    """Only one-tile inputs bypass the projected custom-autograd implementation."""
    generator = torch.Generator().manual_seed(7531)
    student_hidden = torch.randn(
        num_tokens,
        3,
        generator=generator,
    ).requires_grad_(True)
    output_weight = torch.randn(7, 3, generator=generator).requires_grad_(True)
    selected_teacher_logits = torch.randn(
        num_tokens, 7, generator=generator
    ).requires_grad_(True)

    stats = projected_streaming_vocab_parallel_soft_ce(
        student_hidden=student_hidden,
        output_weight=output_weight,
        selected_teacher_logits=selected_teacher_logits,
        mask=torch.ones(num_tokens),
        token_chunk_size=4,
        tp_group=None,
    )
    assert type(stats.numerators.grad_fn).__name__ == expected_grad_fn
    stats.normalized(normalization_counts=stats.counts).backward()

    assert student_hidden.grad is not None
    assert output_weight.grad is None
    assert selected_teacher_logits.grad is None


@pytest.mark.parametrize(
    "num_tokens",
    [pytest.param(4, id="one_tile"), pytest.param(5, id="multiple_tiles")],
)
@pytest.mark.parametrize(
    "invalid_bin_id",
    [pytest.param(-1, id="negative"), pytest.param(2, id="too_large")],
)
def test_projected_soft_ce_rejects_out_of_range_bin_ids(
    num_tokens: int,
    invalid_bin_id: int,
) -> None:
    """Both routes reject invalid bins before projection or scatter operations."""
    generator = torch.Generator().manual_seed(8642)
    hidden_size, vocab_size = 3, 7
    student_hidden = torch.randn(num_tokens, hidden_size, generator=generator)
    output_weight = torch.randn(vocab_size, hidden_size, generator=generator)
    selected_teacher_logits = torch.randn(num_tokens, vocab_size, generator=generator)
    bin_ids = torch.zeros(num_tokens, dtype=torch.long)
    bin_ids[-1] = invalid_bin_id

    with pytest.raises(ValueError, match=r"bin_ids must lie in \[0, 2\)"):
        projected_streaming_vocab_parallel_soft_ce(
            student_hidden=student_hidden,
            output_weight=output_weight,
            selected_teacher_logits=selected_teacher_logits,
            mask=torch.ones(num_tokens),
            bin_ids=bin_ids,
            weights=torch.ones(2),
            token_chunk_size=4,
            tp_group=None,
        )


@pytest.mark.parametrize(
    "num_tokens",
    [pytest.param(4, id="one_tile"), pytest.param(5, id="multiple_tiles")],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projected_soft_ce_matches_dense_hidden_gradient(
    num_tokens: int,
    dtype: torch.dtype,
) -> None:
    """Both routes match dense soft CE with indexed and duplicate teacher rows."""
    generator = torch.Generator().manual_seed(1357)
    hidden_size, vocab_size = 5, 7
    student_hidden = torch.randn(
        num_tokens,
        hidden_size,
        generator=generator,
        dtype=dtype,
    ).requires_grad_(True)
    output_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        dtype=dtype,
    ).requires_grad_(True)
    selected_teacher_logits = torch.randn(
        num_tokens,
        vocab_size,
        generator=generator,
        dtype=dtype,
    ).requires_grad_(True)
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])[:num_tokens]
    bin_ids = torch.tensor([0, 0, 1, 1, 0])[:num_tokens]
    weights = torch.tensor([0.25, 1.5, 0.75])

    stats = projected_streaming_vocab_parallel_soft_ce(
        student_hidden=student_hidden,
        output_weight=output_weight,
        selected_teacher_logits=selected_teacher_logits,
        mask=mask,
        bin_ids=bin_ids,
        weights=weights,
        token_chunk_size=4,
        tp_group=None,
    )
    loss = stats.normalized(normalization_counts=stats.counts)
    loss.backward()

    reference_hidden = student_hidden.detach().clone().requires_grad_(True)
    expected_numerators, expected_counts = _dense_projected_stats(
        reference_hidden,
        output_weight.detach(),
        selected_teacher_logits.detach(),
        mask,
        bin_ids,
        num_bins=3,
    )
    expected_loss = (expected_numerators * weights).sum() / (
        (expected_counts * weights).sum() + 1e-8
    )
    expected_loss.backward()

    tolerance = {"rtol": 2e-2, "atol": 2e-2} if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(stats.numerators, expected_numerators, **tolerance)
    torch.testing.assert_close(stats.counts, expected_counts)
    torch.testing.assert_close(loss, expected_loss, **tolerance)
    torch.testing.assert_close(
        student_hidden.grad,
        reference_hidden.grad,
        **tolerance,
    )
    assert stats.counts[2] == 0
    assert selected_teacher_logits.grad is None
    assert output_weight.grad is None


@pytest.mark.parametrize(
    ("num_tokens", "expected_vocab_distributions"),
    [
        pytest.param(4, 2, id="one_tile_caches_distributions"),
        pytest.param(5, 0, id="multiple_tiles_recompute_distributions"),
    ],
)
def test_projected_soft_ce_saved_state_contract(
    num_tokens: int,
    expected_vocab_distributions: int,
) -> None:
    """Only the one-tile route saves FP32 projected-vocabulary distributions."""
    generator = torch.Generator().manual_seed(9753)
    hidden_size, vocab_size = 3, 7
    student_hidden = torch.randn(
        num_tokens,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    output_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    selected_teacher_logits = torch.randn(
        num_tokens,
        vocab_size,
        generator=generator,
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    mask = torch.ones(num_tokens)
    saved_tensors: list[torch.Tensor] = []

    def record(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        stats = projected_streaming_vocab_parallel_soft_ce(
            student_hidden=student_hidden,
            output_weight=output_weight,
            selected_teacher_logits=selected_teacher_logits,
            mask=mask,
            token_chunk_size=4,
            tp_group=None,
        )
        stats.normalized(normalization_counts=stats.counts).backward()

    vocab_distributions = [
        tensor
        for tensor in saved_tensors
        if tensor.dtype == torch.float32 and tensor.shape == (num_tokens, vocab_size)
    ]
    assert len(vocab_distributions) == expected_vocab_distributions
    assert any(
        tensor.dtype == torch.float32 and tensor.shape == (num_tokens, 2)
        for tensor in saved_tensors
    ) is (expected_vocab_distributions == 0)
    assert student_hidden.grad is not None
    assert student_hidden.grad.dtype == torch.bfloat16
    assert selected_teacher_logits.grad is None
    assert output_weight.grad is None


def test_projected_soft_ce_casts_full_head_once_per_backward() -> None:
    """Token tiling must not repeat the full output-head FP32 conversion."""
    generator = torch.Generator().manual_seed(2468)
    num_tokens, hidden_size, vocab_size = 5, 7, 11
    student_hidden = torch.randn(
        num_tokens,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    output_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    selected_teacher_logits = torch.randn(
        num_tokens,
        vocab_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    stats = projected_streaming_vocab_parallel_soft_ce(
        student_hidden=student_hidden,
        output_weight=output_weight,
        selected_teacher_logits=selected_teacher_logits,
        mask=torch.ones(num_tokens),
        token_chunk_size=2,
        tp_group=None,
    )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as profile:
        stats.normalized(normalization_counts=stats.counts).backward()

    full_head_cast_count = sum(
        event.count
        for event in profile.key_averages(group_by_input_shape=True)
        if event.key == "aten::_to_copy"
        and event.input_shapes
        and event.input_shapes[0] == [vocab_size, hidden_size]
    )
    assert full_head_cast_count == 1


@pytest.mark.parametrize("context_length", [32_768, 262_144])
def test_projected_soft_ce_does_not_retain_full_context_teacher_logits(
    context_length: int,
) -> None:
    """The projected seam accepts and saves only requested teacher rows."""
    generator = torch.Generator().manual_seed(86420)
    num_tokens, hidden_size, vocab_size = 5, 3, 17
    full_teacher_logits = torch.randn(
        context_length,
        vocab_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    selected_teacher_logits = full_teacher_logits.index_select(
        0, torch.tensor([1, 7, 42, context_length - 1, 7])
    )
    full_teacher_storage_bytes = full_teacher_logits.untyped_storage().nbytes()
    student_hidden = torch.randn(
        num_tokens,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    output_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    saved_storage_bytes: list[int] = []

    def record(tensor: torch.Tensor) -> torch.Tensor:
        saved_storage_bytes.append(tensor.untyped_storage().nbytes())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        stats = projected_streaming_vocab_parallel_soft_ce(
            student_hidden=student_hidden,
            output_weight=output_weight,
            selected_teacher_logits=selected_teacher_logits,
            mask=torch.ones(num_tokens),
            token_chunk_size=2,
            tp_group=None,
        )

    assert max(saved_storage_bytes) < full_teacher_storage_bytes
    stats.normalized(normalization_counts=stats.counts).backward()
