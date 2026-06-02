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

"""Unit tests for MegatronValueWorker helpers.

Targets the sequence-packing unpack logic in `get_values`. The TP/SP gather
fix is exercised end-to-end by the PPO recipes (which now enable
`sequence_parallel=True` and `context_parallel_size=2`); the helper here
covers the pure-tensor unpack path that's easy to verify without a
multi-process distributed setup.
"""

import pytest
import torch

# The worker module pulls in heavy Megatron deps; skip these tests cleanly
# in environments where megatron-bridge / megatron-core are not available.
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.core")

from nemo_rl.models.value.workers.megatron_value_worker import _unpack_values


def _make_packed_values(
    samples: list[list[float]], padded_lengths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a packed values tensor + cu_seqlens_padded from sample lists.

    Each sample is padded out to its `padded_lengths[i]` slot in the packed tensor.
    Returns (values_packed[1, T_packed], cu_seqlens_padded[B+1]).
    """
    assert len(samples) == len(padded_lengths)
    parts = []
    offsets = [0]
    for sample, pad_len in zip(samples, padded_lengths):
        assert len(sample) <= pad_len
        padded = sample + [0.0] * (pad_len - len(sample))
        parts.extend(padded)
        offsets.append(offsets[-1] + pad_len)
    return (
        torch.tensor([parts], dtype=torch.float32),
        torch.tensor(offsets, dtype=torch.int64),
    )


def test_unpack_values_basic():
    """Two samples with different actual lengths, equal padding."""
    # Sample 0: [1, 2, 3], padded to length 4 -> last slot is 0
    # Sample 1: [10, 20], padded to length 4 -> last two slots are 0
    values_packed, cu = _make_packed_values(
        samples=[[1.0, 2.0, 3.0], [10.0, 20.0]],
        padded_lengths=[4, 4],
    )

    out = _unpack_values(
        values_packed=values_packed,
        cu_seqlens_padded=cu,
        unpacked_batch_size=2,
        unpacked_seq_length=4,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0],
            [10.0, 20.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out, expected)


def test_unpack_values_truncates_when_padded_len_exceeds_unpacked_seqlen():
    """Padded slot longer than unpacked S: must truncate, not overflow."""
    values_packed, cu = _make_packed_values(
        samples=[[1.0, 2.0, 3.0, 4.0, 5.0]],
        padded_lengths=[8],  # padded to 8, but unpacked_seq_length=4
    )

    out = _unpack_values(
        values_packed=values_packed,
        cu_seqlens_padded=cu,
        unpacked_batch_size=1,
        unpacked_seq_length=4,
    )

    # Only first 4 elements come through; trailing pad is discarded.
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_unpack_values_accepts_1d_input():
    """Worker may pass either [T_packed] or [1, T_packed]; both should work."""
    values_packed_2d, cu = _make_packed_values(
        samples=[[1.0, 2.0], [3.0, 4.0]],
        padded_lengths=[2, 2],
    )
    values_packed_1d = values_packed_2d.squeeze(0)

    out_2d = _unpack_values(
        values_packed=values_packed_2d,
        cu_seqlens_padded=cu,
        unpacked_batch_size=2,
        unpacked_seq_length=2,
    )
    out_1d = _unpack_values(
        values_packed=values_packed_1d,
        cu_seqlens_padded=cu,
        unpacked_batch_size=2,
        unpacked_seq_length=2,
    )
    torch.testing.assert_close(out_2d, out_1d)


def test_unpack_values_preserves_dtype_and_device():
    """Output should match input dtype / device (CPU here)."""
    values_packed = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.bfloat16)
    cu = torch.tensor([0, 3], dtype=torch.int64)

    out = _unpack_values(
        values_packed=values_packed,
        cu_seqlens_padded=cu,
        unpacked_batch_size=1,
        unpacked_seq_length=3,
    )
    assert out.dtype == torch.bfloat16
    assert out.device == values_packed.device


def test_unpack_values_handles_empty_sample():
    """A sample whose padded slot has zero length should yield all zeros."""
    # Sample 0: [1, 2] padded to 2; sample 1: empty slot (start==end)
    values_packed = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    cu = torch.tensor([0, 2, 2], dtype=torch.int64)

    out = _unpack_values(
        values_packed=values_packed,
        cu_seqlens_padded=cu,
        unpacked_batch_size=2,
        unpacked_seq_length=3,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out, expected)
