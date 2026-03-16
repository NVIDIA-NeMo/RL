# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for DistributedLogprob and ChunkedDistributedLogprob using mp.spawn.

These tests use the distributed_test_runner fixture (torch.multiprocessing.spawn)
so that code coverage is captured by pytest-cov, unlike the Ray actor-based tests
in test_model_utils.py where execution happens in separate Ray worker processes.
"""

import functools

import pytest
import torch

from nemo_rl.distributed.model_utils import (
    ChunkedDistributedEntropy,
    ChunkedDistributedGatherLogprob,
    ChunkedDistributedLogprob,
    DistributedLogprob,
    _compute_distributed_log_softmax,
)


def _torch_baseline_logprob(full_logits, target):
    """Single-GPU PyTorch baseline for log probability computation."""
    log_softmax = torch.nn.functional.log_softmax(full_logits, dim=-1)
    log_probs = torch.gather(log_softmax, -1, target.unsqueeze(-1)).squeeze(-1)
    target_mask = target >= 0
    log_probs = log_probs * target_mask.float()
    return log_probs


def _run_logprob_forward_and_backward(rank, world_size, tp_size, chunk_size):
    """Test DistributedLogprob / ChunkedDistributedLogprob forward and backward passes."""
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))

    batch_size = 4
    seq_len = 8
    full_vocab_size = 1024
    vocab_part_size = full_vocab_size // tp_size

    vocab_start_index = rank * vocab_part_size
    vocab_end_index = (rank + 1) * vocab_part_size

    torch.manual_seed(42)
    full_logits = torch.randn(
        batch_size, seq_len, full_vocab_size, device="cuda", requires_grad=True
    )

    vocab_parallel_logits = (
        full_logits[:, :, vocab_start_index:vocab_end_index]
        .clone()
        .detach()
        .requires_grad_(True)
    )

    torch.manual_seed(43)
    target = torch.randint(0, full_vocab_size, (batch_size, seq_len), device="cuda")

    # === FORWARD PASS ===
    baseline_log_probs_forward = _torch_baseline_logprob(
        full_logits.clone().detach(), target
    )

    if chunk_size is not None:
        distributed_log_probs_inference = ChunkedDistributedLogprob.apply(
            vocab_parallel_logits.clone().detach(),
            target,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            tp_group,
            True,
        )
    else:
        distributed_log_probs_inference = DistributedLogprob.apply(
            vocab_parallel_logits.clone().detach(),
            target,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            True,
        )

    torch.testing.assert_close(
        distributed_log_probs_inference,
        baseline_log_probs_forward,
        rtol=1e-4,
        atol=1e-4,
    )

    # === BACKWARD PASS ===
    baseline_log_probs = _torch_baseline_logprob(full_logits, target)
    baseline_loss = torch.sum(baseline_log_probs)
    baseline_loss.backward()
    baseline_grad = full_logits.grad[:, :, vocab_start_index:vocab_end_index].clone()

    full_logits.grad = None

    if chunk_size is not None:
        distributed_log_probs = ChunkedDistributedLogprob.apply(
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            tp_group,
            False,
        )
    else:
        distributed_log_probs = DistributedLogprob.apply(
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            False,
        )

    distributed_loss = torch.sum(distributed_log_probs)
    distributed_loss.backward()
    distributed_grad = vocab_parallel_logits.grad

    torch.testing.assert_close(distributed_grad, baseline_grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        distributed_log_probs, baseline_log_probs, rtol=1e-4, atol=1e-4
    )


def _run_log_softmax(rank, world_size, tp_size):
    """Test _compute_distributed_log_softmax against PyTorch baseline."""
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))

    batch_size = 3
    seq_len = 5
    full_vocab_size = 256
    vocab_part_size = full_vocab_size // tp_size

    vocab_start_index = rank * vocab_part_size
    vocab_end_index = (rank + 1) * vocab_part_size

    torch.manual_seed(42)
    full_logits = torch.randn(batch_size, seq_len, full_vocab_size, device="cuda")
    vocab_parallel_logits = full_logits[:, :, vocab_start_index:vocab_end_index].clone()

    baseline_log_softmax = torch.nn.functional.log_softmax(full_logits, dim=-1)
    expected = baseline_log_softmax[:, :, vocab_start_index:vocab_end_index]

    distributed = _compute_distributed_log_softmax(vocab_parallel_logits, tp_group)

    torch.testing.assert_close(distributed, expected, rtol=1e-5, atol=1e-5)


def _run_edge_cases(rank, world_size, tp_size):
    """Test numerical stability and boundary conditions for DistributedLogprob."""
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))

    batch_size = 2
    seq_len = 3
    full_vocab_size = 64
    vocab_part_size = full_vocab_size // tp_size

    vocab_start_index = rank * vocab_part_size
    vocab_end_index = (rank + 1) * vocab_part_size

    # Large logits — should not produce NaN or Inf
    torch.manual_seed(42)
    large_logits = (
        torch.randn(batch_size, seq_len, full_vocab_size, device="cuda") * 100
    )
    vocab_parallel_logits = large_logits[
        :, :, vocab_start_index:vocab_end_index
    ].clone()

    torch.manual_seed(43)
    target = torch.randint(0, full_vocab_size, (batch_size, seq_len), device="cuda")

    log_probs = DistributedLogprob.apply(
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        tp_group,
        True,
    )

    assert not torch.isnan(log_probs).any(), "Log probs contain NaN"
    assert not torch.isinf(log_probs).any(), "Log probs contain Inf"

    # All targets pointing to vocab index 0
    zero_target = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")

    log_probs_zero = DistributedLogprob.apply(
        vocab_parallel_logits,
        zero_target,
        vocab_start_index,
        vocab_end_index,
        tp_group,
        True,
    )

    torch.manual_seed(42)
    baseline_large_logits = (
        torch.randn(batch_size, seq_len, full_vocab_size, device="cuda") * 100
    )
    baseline_log_probs = _torch_baseline_logprob(baseline_large_logits, zero_target)

    torch.testing.assert_close(log_probs_zero, baseline_log_probs, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Pytest test functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp_size, chunk_size",
    [
        (1, None),
        (2, None),
        (1, 4),
        (2, 4),
    ],
)
def test_distributed_logprob_forward_and_backward(
    distributed_test_runner, tp_size, chunk_size
):
    test_fn = functools.partial(
        _run_logprob_forward_and_backward, tp_size=tp_size, chunk_size=chunk_size
    )
    distributed_test_runner(test_fn, world_size=tp_size)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_distributed_log_softmax(distributed_test_runner, tp_size):
    test_fn = functools.partial(_run_log_softmax, tp_size=tp_size)
    distributed_test_runner(test_fn, world_size=tp_size)


def test_distributed_logprob_edge_cases(distributed_test_runner):
    test_fn = functools.partial(_run_edge_cases, tp_size=2)
    distributed_test_runner(test_fn, world_size=2)


# ---------------------------------------------------------------------------
# ChunkedDistributedGatherLogprob
# ---------------------------------------------------------------------------


def _run_chunked_gather_logprob(rank, world_size, tp_size, chunk_size, inference_only):
    """Test ChunkedDistributedGatherLogprob forward (and optionally backward)."""
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))

    batch_size = 2
    seq_len = 16
    vocab_size = 256
    gather_k = 3

    torch.manual_seed(1337)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device="cuda")
    global_indices = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len, gather_k), device="cuda"
    )

    vocab_part_size = vocab_size // tp_size
    vocab_start_index = rank * vocab_part_size
    vocab_end_index = (rank + 1) * vocab_part_size

    # Baseline: single-GPU log_softmax + gather
    baseline_logits = full_logits.clone().detach().requires_grad_(not inference_only)
    baseline_log_probs = torch.nn.functional.log_softmax(baseline_logits, dim=-1)
    baseline_selected = torch.gather(baseline_log_probs, dim=-1, index=global_indices)

    if not inference_only:
        torch.gather(baseline_log_probs, dim=-1, index=global_indices).sum().backward()
        baseline_grad = baseline_logits.grad[:, :, vocab_start_index:vocab_end_index]

    # Distributed path
    local_logits = full_logits[:, :, vocab_start_index:vocab_end_index]
    local_logits = local_logits.clone().detach().requires_grad_(not inference_only)

    gathered = ChunkedDistributedGatherLogprob.apply(
        local_logits,
        global_indices,
        vocab_start_index,
        vocab_end_index,
        chunk_size,
        tp_group,
        inference_only,
    )

    torch.testing.assert_close(gathered, baseline_selected, rtol=1e-4, atol=1e-4)

    if not inference_only:
        gathered.sum().backward()
        torch.testing.assert_close(
            local_logits.grad, baseline_grad, rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize(
    "tp_size, chunk_size, inference_only",
    [
        (1, 5, False),
        (2, 4, False),
        (1, 3, True),
    ],
)
def test_chunked_distributed_gather_logprob(
    distributed_test_runner, tp_size, chunk_size, inference_only
):
    test_fn = functools.partial(
        _run_chunked_gather_logprob,
        tp_size=tp_size,
        chunk_size=chunk_size,
        inference_only=inference_only,
    )
    distributed_test_runner(test_fn, world_size=tp_size)


# ---------------------------------------------------------------------------
# ChunkedDistributedEntropy
# ---------------------------------------------------------------------------


def _run_chunked_distributed_entropy(
    rank, world_size, tp_size, chunk_size, inference_only
):
    """Test ChunkedDistributedEntropy forward (and optionally backward)."""
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))

    batch_size = 2
    seq_len = 16
    vocab_size = 256
    vocab_part_size = vocab_size // tp_size
    vocab_start_index = rank * vocab_part_size
    vocab_end_index = (rank + 1) * vocab_part_size

    torch.manual_seed(1337)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device="cuda")

    # Baseline: single-GPU entropy  H = sum_v p_v * log(p_v)
    baseline_logits = full_logits.clone().detach().requires_grad_(not inference_only)
    baseline_log_probs = torch.nn.functional.log_softmax(baseline_logits, dim=-1)
    baseline_probs = baseline_log_probs.exp()
    baseline_entropy = (baseline_probs * baseline_log_probs).sum(dim=-1)

    if not inference_only:
        baseline_entropy.sum().backward()
        baseline_grad = baseline_logits.grad[
            :, :, vocab_start_index:vocab_end_index
        ].clone()

    # Distributed path
    local_logits = full_logits[:, :, vocab_start_index:vocab_end_index]
    local_logits = local_logits.clone().detach().requires_grad_(not inference_only)

    distributed_entropy = ChunkedDistributedEntropy.apply(
        local_logits,
        chunk_size,
        tp_group,
        inference_only,
    )

    torch.testing.assert_close(
        distributed_entropy, baseline_entropy, rtol=1e-4, atol=1e-4
    )

    if not inference_only:
        distributed_entropy.sum().backward()
        torch.testing.assert_close(
            local_logits.grad, baseline_grad, rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize(
    "tp_size, chunk_size, inference_only",
    [
        (1, 5, False),
        (2, 4, False),
        (1, 3, True),
    ],
)
def test_chunked_distributed_entropy(
    distributed_test_runner, tp_size, chunk_size, inference_only
):
    test_fn = functools.partial(
        _run_chunked_distributed_entropy,
        tp_size=tp_size,
        chunk_size=chunk_size,
        inference_only=inference_only,
    )
    distributed_test_runner(test_fn, world_size=tp_size)
