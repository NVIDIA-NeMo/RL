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

"""Logical, packed, context-parallel, and sequence-parallel draft layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_LAYOUT_DESCRIPTOR_VERSION = 1


@dataclass(frozen=True, slots=True)
class DraftSequenceLayout:
    """Immutable coordinate maps for one packed draft-training microbatch."""

    cp_rank: int
    cp_size: int
    tp_rank: int
    tp_size: int
    logical_sample_ids: Tensor
    logical_lengths: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_q_padded: Tensor
    packed_to_logical_sample: Tensor
    packed_logical_positions: Tensor
    packed_valid_mask: Tensor
    cp_global_positions: Tensor
    cp_global_to_local: Tensor
    owner_cp_rank: Tensor
    sp_local_positions: Tensor
    descriptor: Tensor


def _validate_rank(*, name: str, rank: int, size: int) -> None:
    if size <= 0:
        raise ValueError(f"{name}_size must be positive")
    if rank < 0 or rank >= size:
        raise ValueError(f"{name}_rank must be in [0, {name}_size)")


def _validate_metadata_tensor(
    tensor: Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.dtype != torch.int64:
        raise TypeError(f"{name} must use torch.int64")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}")


def _metadata_fingerprint(tensor: Tensor) -> Tensor:
    weights = torch.arange(
        1,
        tensor.numel() + 1,
        dtype=torch.int64,
        device=tensor.device,
    )
    return (tensor * weights).sum(dtype=torch.int64)


def _cp_positions_for_segment(
    *,
    start: int,
    padded_length: int,
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> Tensor:
    if cp_size == 1:
        return torch.arange(
            start,
            start + padded_length,
            dtype=torch.int64,
            device=device,
        )

    chunk_length = padded_length // (2 * cp_size)
    first_chunk = cp_rank
    second_chunk = 2 * cp_size - cp_rank - 1
    offsets = torch.arange(
        chunk_length,
        dtype=torch.int64,
        device=device,
    )
    return torch.cat(
        (
            start + first_chunk * chunk_length + offsets,
            start + second_chunk * chunk_length + offsets,
        )
    )


def build_draft_sequence_layout(
    *,
    logical_sample_ids: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_q_padded: Tensor,
    cp_rank: int,
    cp_size: int,
    tp_rank: int,
    tp_size: int,
    device: torch.device,
) -> DraftSequenceLayout:
    """Build canonical packed, CP-local, and target-SP coordinate maps."""
    device = torch.device(device)
    _validate_rank(name="cp", rank=cp_rank, size=cp_size)
    _validate_rank(name="tp", rank=tp_rank, size=tp_size)
    _validate_metadata_tensor(
        logical_sample_ids,
        name="logical_sample_ids",
        device=device,
    )
    _validate_metadata_tensor(cu_seqlens_q, name="cu_seqlens_q", device=device)
    _validate_metadata_tensor(
        cu_seqlens_q_padded,
        name="cu_seqlens_q_padded",
        device=device,
    )

    expected_boundaries = logical_sample_ids.numel() + 1
    if cu_seqlens_q.numel() != expected_boundaries:
        raise ValueError("cu_seqlens_q must have one boundary per logical sample")
    if cu_seqlens_q_padded.numel() != expected_boundaries:
        raise ValueError(
            "cu_seqlens_q_padded must have one boundary per logical sample"
        )
    if cu_seqlens_q[0] != 0 or cu_seqlens_q_padded[0] != 0:
        raise ValueError("packed cumulative sequence lengths must start at zero")

    logical_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    padded_lengths = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
    if torch.any(logical_lengths < 0) or torch.any(padded_lengths < 0):
        raise ValueError("packed cumulative sequence lengths must be non-decreasing")
    if torch.any(padded_lengths < logical_lengths):
        raise ValueError("each padded sequence must be at least the unpadded length")
    if cp_size > 1 and torch.any(padded_lengths % (2 * cp_size) != 0):
        raise ValueError("each padded sequence length must be divisible by 2 * cp_size")

    total_padded = int(cu_seqlens_q_padded[-1])
    packed_to_logical_sample = torch.full(
        (total_padded,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    packed_logical_positions = torch.full_like(packed_to_logical_sample, -1)
    packed_valid_mask = torch.zeros(total_padded, dtype=torch.bool, device=device)

    unpadded_lengths_list = logical_lengths.tolist()
    padded_boundaries_list = cu_seqlens_q_padded.tolist()
    sample_ids_list = logical_sample_ids.tolist()
    for sample_id, logical_length, start, end in zip(
        sample_ids_list,
        unpadded_lengths_list,
        padded_boundaries_list[:-1],
        padded_boundaries_list[1:],
        strict=True,
    ):
        valid_end = start + logical_length
        packed_to_logical_sample[start:valid_end] = sample_id
        packed_logical_positions[start:valid_end] = torch.arange(
            logical_length,
            dtype=torch.int64,
            device=device,
        )
        packed_valid_mask[start:valid_end] = True
        if valid_end > end:
            raise AssertionError("validated logical sequence exceeds padded boundary")

    cp_positions_by_rank: list[list[Tensor]] = [[] for _ in range(cp_size)]
    for start, end in zip(
        padded_boundaries_list[:-1],
        padded_boundaries_list[1:],
        strict=True,
    ):
        padded_length = end - start
        for owner_rank in range(cp_size):
            cp_positions_by_rank[owner_rank].append(
                _cp_positions_for_segment(
                    start=start,
                    padded_length=padded_length,
                    cp_rank=owner_rank,
                    cp_size=cp_size,
                    device=device,
                )
            )

    cp_global_positions_by_rank = [
        torch.cat(rank_positions)
        if rank_positions
        else torch.empty(0, dtype=torch.int64, device=device)
        for rank_positions in cp_positions_by_rank
    ]
    cp_global_positions = cp_global_positions_by_rank[cp_rank]
    cp_global_to_local = torch.full(
        (total_padded,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    cp_global_to_local[cp_global_positions] = torch.arange(
        cp_global_positions.numel(),
        dtype=torch.int64,
        device=device,
    )
    owner_cp_rank = torch.full_like(cp_global_to_local, -1)
    for owner_rank, owner_positions in enumerate(cp_global_positions_by_rank):
        owner_cp_rank[owner_positions] = owner_rank

    cp_local_length = cp_global_positions.numel()
    if cp_local_length % tp_size != 0:
        raise ValueError(
            "the CP-local packed sequence length must be divisible by tp_size"
        )
    sp_shard_length = cp_local_length // tp_size
    sp_start = tp_rank * sp_shard_length
    sp_local_positions = torch.arange(
        sp_start,
        sp_start + sp_shard_length,
        dtype=torch.int64,
        device=device,
    )

    descriptor = torch.stack(
        (
            torch.tensor(
                _LAYOUT_DESCRIPTOR_VERSION,
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                logical_sample_ids.numel(),
                dtype=torch.int64,
                device=device,
            ),
            cu_seqlens_q[-1],
            cu_seqlens_q_padded[-1],
            torch.tensor(cp_size, dtype=torch.int64, device=device),
            torch.tensor(tp_size, dtype=torch.int64, device=device),
            _metadata_fingerprint(logical_sample_ids),
            _metadata_fingerprint(cu_seqlens_q),
            _metadata_fingerprint(cu_seqlens_q_padded),
        )
    )

    return DraftSequenceLayout(
        cp_rank=cp_rank,
        cp_size=cp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        logical_sample_ids=logical_sample_ids,
        logical_lengths=logical_lengths,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        packed_to_logical_sample=packed_to_logical_sample,
        packed_logical_positions=packed_logical_positions,
        packed_valid_mask=packed_valid_mask,
        cp_global_positions=cp_global_positions,
        cp_global_to_local=cp_global_to_local,
        owner_cp_rank=owner_cp_rank,
        sp_local_positions=sp_local_positions,
        descriptor=descriptor,
    )
