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

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_HASH_MODULUS = 2_147_483_647
_HASH_MULTIPLIER = 73_244_475


def _mix_anchor_ids(anchor_ids: Tensor) -> Tensor:
    mixed = torch.remainder(anchor_ids, _HASH_MODULUS)
    mixed = torch.remainder(
        torch.bitwise_xor(mixed, mixed >> 16) * _HASH_MULTIPLIER,
        _HASH_MODULUS,
    )
    mixed = torch.remainder(
        torch.bitwise_xor(mixed, mixed >> 16) * _HASH_MULTIPLIER,
        _HASH_MODULUS,
    )
    return torch.bitwise_xor(mixed, mixed >> 16)


@dataclass(frozen=True, slots=True)
class DFlashBatchPlan:
    """Immutable tensor plan for anchored DFlash blocks."""

    batch_size: int
    sequence_length: int
    anchors_per_sample: int
    gamma: int
    block_size: int
    token_valid_mask: Tensor
    sample_rows: Tensor
    anchor_ids: Tensor
    anchor_positions: Tensor
    trunk_lengths: Tensor
    query_positions: Tensor
    label_positions: Tensor
    block_valid: Tensor
    slot_valid: Tensor
    loss_mask: Tensor


def build_dflash_batch_plan(
    token_valid_mask: Tensor,
    sample_ids: Tensor,
    *,
    anchors_per_sample: int,
    gamma: int,
    optimizer_step: int,
    seed: int,
) -> DFlashBatchPlan:
    """Build a deterministic, device-resident DFlash anchor schedule."""
    if token_valid_mask.ndim != 2:
        raise ValueError("token_valid_mask must have shape [batch, sequence]")
    if token_valid_mask.dtype != torch.bool:
        raise TypeError("token_valid_mask must be a boolean tensor")
    if sample_ids.ndim != 1 or sample_ids.shape[0] != token_valid_mask.shape[0]:
        raise ValueError("sample_ids must have shape [batch]")
    if sample_ids.dtype != torch.int64:
        raise TypeError("sample_ids must use torch.int64")
    if sample_ids.device != token_valid_mask.device:
        raise ValueError("sample_ids and token_valid_mask must share a device")
    if anchors_per_sample <= 0:
        raise ValueError("anchors_per_sample must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    batch_size, sequence_length = token_valid_mask.shape
    block_size = gamma + 1
    device = token_valid_mask.device

    anchor_slots = torch.arange(
        anchors_per_sample,
        dtype=torch.int64,
        device=device,
    )
    anchor_ids_2d = (
        sample_ids[:, None] * 17
        + optimizer_step * 31
        + seed * 43
        + anchor_slots[None, :] * 59
    )

    num_candidate_positions = max(sequence_length - block_size + 1, 0)
    if num_candidate_positions == 0:
        row_block_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        anchor_positions_2d = torch.zeros_like(anchor_ids_2d)
    else:
        valid_windows = token_valid_mask.unfold(1, block_size, 1).all(dim=-1)
        valid_anchor_counts = valid_windows.sum(dim=1, dtype=torch.int64)
        row_block_valid = valid_anchor_counts > 0
        anchor_ordinals_2d = torch.remainder(
            _mix_anchor_ids(anchor_ids_2d),
            valid_anchor_counts.clamp_min(1)[:, None],
        )
        valid_window_ranks = valid_windows.cumsum(dim=1, dtype=torch.int64)
        anchor_positions_2d = torch.searchsorted(
            valid_window_ranks,
            anchor_ordinals_2d + 1,
        )
        anchor_positions_2d = torch.where(
            row_block_valid[:, None],
            anchor_positions_2d,
            torch.zeros_like(anchor_positions_2d),
        )

    valid_prefix_counts = torch.cat(
        (
            torch.zeros((batch_size, 1), dtype=torch.int64, device=device),
            token_valid_mask.cumsum(dim=1, dtype=torch.int64),
        ),
        dim=1,
    )
    trunk_lengths_2d = torch.gather(
        valid_prefix_counts,
        dim=1,
        index=anchor_positions_2d,
    )

    sample_rows = torch.arange(batch_size, device=device).repeat_interleave(
        anchors_per_sample
    )
    anchor_ids = anchor_ids_2d.reshape(-1)
    anchor_positions = anchor_positions_2d.reshape(-1)
    block_valid = row_block_valid.repeat_interleave(anchors_per_sample)

    slot_offsets = torch.arange(block_size, dtype=torch.int64, device=device)
    query_positions = anchor_positions[:, None] + slot_offsets[None, :]
    query_positions = torch.where(
        block_valid[:, None],
        query_positions,
        torch.zeros_like(query_positions),
    )

    if sequence_length == 0:
        slot_valid = torch.zeros_like(query_positions, dtype=torch.bool)
    else:
        slot_valid = token_valid_mask[sample_rows[:, None], query_positions]
        slot_valid = slot_valid & block_valid[:, None]

    loss_mask = slot_valid & (slot_offsets[None, :] > 0)
    return DFlashBatchPlan(
        batch_size=batch_size,
        sequence_length=sequence_length,
        anchors_per_sample=anchors_per_sample,
        gamma=gamma,
        block_size=block_size,
        token_valid_mask=token_valid_mask,
        sample_rows=sample_rows,
        anchor_ids=anchor_ids,
        anchor_positions=anchor_positions,
        trunk_lengths=trunk_lengths_2d.reshape(-1),
        query_positions=query_positions,
        label_positions=query_positions,
        block_valid=block_valid,
        slot_valid=slot_valid,
        loss_mask=loss_mask,
    )
