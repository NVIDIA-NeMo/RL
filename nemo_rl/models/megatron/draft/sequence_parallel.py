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

"""Autograd-safe target sequence reconstruction within one TP/CP lane."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.distributed as dist
from torch import Tensor
from torch.distributed._functional_collectives import all_gather_tensor_autograd

if TYPE_CHECKING:
    from nemo_rl.models.megatron.draft.sequence_layout import DraftSequenceLayout


def reconstruct_tp_sequence(
    local_sequence: Tensor,
    *,
    sequence_layout: DraftSequenceLayout,
    tp_group: dist.ProcessGroup | None,
    sequence_dim: int,
) -> Tensor:
    """Reconstruct one CP-local sequence from its ordered target-SP shards."""
    if local_sequence.ndim == 0:
        raise ValueError("local_sequence must have a sequence dimension")
    normalized_sequence_dim = sequence_dim % local_sequence.ndim
    expected_local_length = sequence_layout.sp_local_positions.numel()
    if local_sequence.shape[normalized_sequence_dim] != expected_local_length:
        raise ValueError(
            "target-SP shard length does not match the draft sequence layout"
        )

    if sequence_layout.tp_size == 1:
        if sequence_layout.tp_rank != 0:
            raise ValueError("TP1 sequence layout must use tp_rank zero")
        if expected_local_length != sequence_layout.cp_global_positions.numel():
            raise ValueError("TP1 layout does not cover the complete CP-local sequence")
        return local_sequence

    if tp_group is None or not dist.is_initialized():
        raise RuntimeError("target-SP reconstruction requires an initialized TP group")
    if dist.get_world_size(tp_group) != sequence_layout.tp_size:
        raise ValueError("TP group size does not match the draft sequence layout")
    if dist.get_rank(tp_group) != sequence_layout.tp_rank:
        raise ValueError("TP group rank does not match the draft sequence layout")

    sequence_first = local_sequence.movedim(normalized_sequence_dim, 0).contiguous()
    reconstructed = all_gather_tensor_autograd(
        sequence_first,
        gather_dim=0,
        group=tp_group,
    )
    if reconstructed.shape[0] != sequence_layout.cp_global_positions.numel():
        raise RuntimeError("target-SP gather did not reconstruct the CP-local sequence")
    return reconstructed.movedim(0, normalized_sequence_dim)
