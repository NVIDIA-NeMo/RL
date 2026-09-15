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

"""Autograd-safe exchange of projected draft K/V across context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._functional_collectives import all_gather_tensor_autograd

if TYPE_CHECKING:
    from nemo_rl.models.megatron.draft.sequence_layout import DraftSequenceLayout


def _rank_major_to_packed_indices(
    sequence_layout: DraftSequenceLayout,
    *,
    local_length: int,
) -> Tensor:
    owner_cp_rank = sequence_layout.owner_cp_rank
    rank_major_indices = torch.empty_like(owner_cp_rank)
    for owner_rank in range(sequence_layout.cp_size):
        packed_positions = torch.where(owner_cp_rank == owner_rank)[0]
        if packed_positions.numel() != local_length:
            raise ValueError("CP ranks must own equal packed sequence lengths")
        rank_major_indices[packed_positions] = owner_rank * local_length + torch.arange(
            local_length,
            dtype=torch.int64,
            device=owner_cp_rank.device,
        )
    return rank_major_indices


def gather_projected_kv(
    projected: Tensor,
    *,
    sequence_layout: DraftSequenceLayout,
    cp_group: dist.ProcessGroup | None,
    sequence_dim: int,
) -> Tensor:
    """Restore projected CP-local K/V to the global packed sequence order."""
    if projected.ndim == 0:
        raise ValueError("projected K/V must have a sequence dimension")
    normalized_sequence_dim = sequence_dim % projected.ndim
    local_length = sequence_layout.cp_global_positions.numel()
    if projected.shape[normalized_sequence_dim] != local_length:
        raise ValueError("projected K/V length does not match the CP-local layout")
    if sequence_layout.owner_cp_rank.device != projected.device:
        raise ValueError("projected K/V and draft layout must share a device")

    if sequence_layout.cp_size == 1:
        if sequence_layout.cp_rank != 0:
            raise ValueError("CP1 sequence layout must use cp_rank zero")
        if local_length != sequence_layout.owner_cp_rank.numel():
            raise ValueError("CP1 layout does not cover the packed sequence")
        return projected

    if cp_group is None or not dist.is_initialized():
        raise RuntimeError("projected K/V exchange requires an initialized CP group")
    if dist.get_world_size(cp_group) != sequence_layout.cp_size:
        raise ValueError("CP group size does not match the draft sequence layout")
    if dist.get_rank(cp_group) != sequence_layout.cp_rank:
        raise ValueError("CP group rank does not match the draft sequence layout")

    sequence_first = projected.movedim(normalized_sequence_dim, 0).contiguous()
    rank_major = all_gather_tensor_autograd(
        sequence_first,
        gather_dim=0,
        group=cp_group,
    )
    expected_global_length = sequence_layout.owner_cp_rank.numel()
    if rank_major.shape[0] != expected_global_length:
        raise RuntimeError("CP gather did not cover the packed sequence")
    packed = rank_major.index_select(
        0,
        _rank_major_to_packed_indices(
            sequence_layout,
            local_length=local_length,
        ),
    )
    return packed.movedim(0, normalized_sequence_dim)
