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
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nemo_rl.models.megatron.draft.sequence_layout import DraftSequenceLayout

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
    global_anchor_positions: Tensor
    global_query_positions: Tensor
    local_anchor_positions: Tensor
    local_query_positions: Tensor
    local_label_positions: Tensor
    owner_cp_ranks: Tensor
    logical_sample_ids: Tensor
    logical_anchor_positions: Tensor
    packed_segment_starts: Tensor
    packed_segment_ends: Tensor
    packed_rope_positions: Tensor
    boundary_valid_mask: Tensor
    excluded_window_count: Tensor
    eligible_window_count: Tensor


def _build_partitioned_dflash_batch_plan(
    token_valid_mask: Tensor,
    sample_ids: Tensor,
    *,
    anchors_per_sample: int,
    gamma: int,
    optimizer_step: int,
    seed: int,
    sequence_layout: DraftSequenceLayout,
) -> DFlashBatchPlan:
    batch_size, sequence_length = token_valid_mask.shape
    block_size = gamma + 1
    device = token_valid_mask.device
    if sequence_layout.logical_sample_ids.device != device:
        raise ValueError("sequence layout and plan inputs must share a device")
    if sequence_layout.logical_sample_ids.shape != sample_ids.shape:
        raise ValueError("sequence layout must describe every logical sample row")
    if not torch.equal(sequence_layout.logical_sample_ids, sample_ids):
        raise ValueError("sequence layout sample IDs must match plan sample IDs")
    if torch.any(sequence_layout.logical_lengths > sequence_length):
        raise ValueError("sequence layout length exceeds the logical plan width")

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
    candidate_positions = torch.arange(
        num_candidate_positions,
        dtype=torch.int64,
        device=device,
    )
    slot_offsets = torch.arange(block_size, dtype=torch.int64, device=device)

    if num_candidate_positions == 0:
        valid_windows = torch.zeros(
            (batch_size, 0),
            dtype=torch.bool,
            device=device,
        )
    else:
        valid_windows = token_valid_mask.unfold(1, block_size, 1).all(dim=-1)
    segment_starts = sequence_layout.cu_seqlens_q_padded[:-1]
    global_candidate_queries = (
        segment_starts[:, None, None]
        + candidate_positions[None, :, None]
        + slot_offsets[None, None, :]
    )
    if num_candidate_positions == 0:
        owner_consistent = valid_windows
    elif sequence_layout.owner_cp_rank.numel() == 0:
        owner_consistent = torch.zeros_like(valid_windows)
    else:
        segment_ends = sequence_layout.cu_seqlens_q_padded[1:]
        within_segment = global_candidate_queries < segment_ends[:, None, None]
        safe_global_candidate_queries = torch.minimum(
            global_candidate_queries,
            (segment_ends - 1).clamp_min(0)[:, None, None],
        )
        candidate_owners = sequence_layout.owner_cp_rank[safe_global_candidate_queries]
        same_owner = (candidate_owners == candidate_owners[..., :1]).all(dim=-1)
        physical_valid = (
            sequence_layout.packed_valid_mask[safe_global_candidate_queries]
            & within_segment
        )
        physical_valid = physical_valid.all(dim=-1)
        owner_consistent = same_owner & physical_valid & (candidate_owners[..., 0] >= 0)
    eligible_windows = valid_windows & owner_consistent
    eligible_counts = eligible_windows.sum(dim=1, dtype=torch.int64)
    row_has_eligible_window = eligible_counts > 0
    anchor_ordinals_2d = torch.remainder(
        _mix_anchor_ids(anchor_ids_2d),
        eligible_counts.clamp_min(1)[:, None],
    )
    eligible_ranks = eligible_windows.cumsum(dim=1, dtype=torch.int64)
    logical_anchor_positions_2d = torch.searchsorted(
        eligible_ranks,
        anchor_ordinals_2d + 1,
    )
    logical_anchor_positions_2d = torch.where(
        row_has_eligible_window[:, None],
        logical_anchor_positions_2d,
        torch.zeros_like(logical_anchor_positions_2d),
    )
    global_anchor_positions_2d = segment_starts[:, None] + logical_anchor_positions_2d
    if sequence_layout.owner_cp_rank.numel() == 0:
        selected_owner_ranks_2d = torch.full_like(
            global_anchor_positions_2d,
            -1,
        )
    else:
        selected_owner_ranks_2d = sequence_layout.owner_cp_rank[
            global_anchor_positions_2d
        ]
    local_selected = row_has_eligible_window[:, None] & (
        selected_owner_ranks_2d == sequence_layout.cp_rank
    )

    all_sample_rows = torch.arange(
        batch_size,
        dtype=torch.int64,
        device=device,
    ).repeat_interleave(anchors_per_sample)
    selected_mask = local_selected.reshape(-1)
    sample_rows = all_sample_rows[selected_mask]
    anchor_ids = anchor_ids_2d.reshape(-1)[selected_mask]
    logical_anchor_positions = logical_anchor_positions_2d.reshape(-1)[selected_mask]
    owner_cp_ranks = selected_owner_ranks_2d.reshape(-1)[selected_mask]
    global_anchor_positions = global_anchor_positions_2d.reshape(-1)[selected_mask]
    global_query_positions = global_anchor_positions[:, None] + slot_offsets[None, :]
    local_query_positions = sequence_layout.cp_global_to_local[global_query_positions]
    local_anchor_positions = local_query_positions[:, 0]
    boundary_valid_mask = sequence_layout.packed_valid_mask[global_query_positions]
    block_valid = torch.ones(
        anchor_ids.shape,
        dtype=torch.bool,
        device=device,
    )
    slot_valid = boundary_valid_mask & block_valid[:, None]
    loss_mask = slot_valid & (slot_offsets[None, :] > 0)
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
        index=logical_anchor_positions_2d,
    )
    trunk_lengths = trunk_lengths_2d.reshape(-1)[selected_mask]
    packed_segment_starts = segment_starts[sample_rows]
    packed_segment_ends = (segment_starts + sequence_layout.logical_lengths)[
        sample_rows
    ]
    packed_rope_positions = logical_anchor_positions[:, None] + slot_offsets[None, :]

    return DFlashBatchPlan(
        batch_size=batch_size,
        sequence_length=sequence_length,
        anchors_per_sample=anchors_per_sample,
        gamma=gamma,
        block_size=block_size,
        token_valid_mask=token_valid_mask,
        sample_rows=sample_rows,
        anchor_ids=anchor_ids,
        anchor_positions=local_anchor_positions,
        trunk_lengths=trunk_lengths,
        query_positions=local_query_positions,
        label_positions=local_query_positions,
        block_valid=block_valid,
        slot_valid=slot_valid,
        loss_mask=loss_mask,
        global_anchor_positions=global_anchor_positions,
        global_query_positions=global_query_positions,
        local_anchor_positions=local_anchor_positions,
        local_query_positions=local_query_positions,
        local_label_positions=local_query_positions,
        owner_cp_ranks=owner_cp_ranks,
        logical_sample_ids=sample_ids[sample_rows],
        logical_anchor_positions=logical_anchor_positions,
        packed_segment_starts=packed_segment_starts,
        packed_segment_ends=packed_segment_ends,
        packed_rope_positions=packed_rope_positions,
        boundary_valid_mask=boundary_valid_mask,
        excluded_window_count=(valid_windows & ~owner_consistent).sum(
            dtype=torch.int64
        ),
        eligible_window_count=eligible_windows.sum(dtype=torch.int64),
    )


@dataclass(frozen=True, slots=True)
class DSparkBatchPlan(DFlashBatchPlan):
    """Anchor-sampled DSpark blocks whose anchor slot predicts the next token."""

    global_label_positions: Tensor
    packed_label_rope_positions: Tensor


def build_dflash_batch_plan(
    token_valid_mask: Tensor,
    sample_ids: Tensor,
    *,
    anchors_per_sample: int,
    gamma: int,
    optimizer_step: int,
    seed: int,
    sequence_layout: DraftSequenceLayout | None = None,
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

    if sequence_layout is not None:
        return _build_partitioned_dflash_batch_plan(
            token_valid_mask,
            sample_ids,
            anchors_per_sample=anchors_per_sample,
            gamma=gamma,
            optimizer_step=optimizer_step,
            seed=seed,
            sequence_layout=sequence_layout,
        )

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
        valid_windows = torch.zeros(
            (batch_size, 0),
            dtype=torch.bool,
            device=device,
        )
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
    eligible_window_count = valid_windows.sum(dtype=torch.int64)
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
        global_anchor_positions=anchor_positions,
        global_query_positions=query_positions,
        local_anchor_positions=anchor_positions,
        local_query_positions=query_positions,
        local_label_positions=query_positions,
        owner_cp_ranks=torch.zeros_like(anchor_positions),
        logical_sample_ids=sample_ids[sample_rows],
        logical_anchor_positions=anchor_positions,
        packed_segment_starts=torch.zeros_like(anchor_positions),
        packed_segment_ends=torch.full_like(anchor_positions, sequence_length),
        packed_rope_positions=query_positions,
        boundary_valid_mask=slot_valid,
        excluded_window_count=torch.zeros((), dtype=torch.int64, device=device),
        eligible_window_count=eligible_window_count,
    )


def build_dspark_batch_plan(
    token_valid_mask: Tensor,
    sample_ids: Tensor,
    *,
    anchors_per_sample: int,
    block_size: int,
    optimizer_step: int,
    seed: int,
    sequence_layout: DraftSequenceLayout | None = None,
) -> DSparkBatchPlan:
    """Build K DSpark slots predicting the K tokens after each anchor."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    extended = build_dflash_batch_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=anchors_per_sample,
        gamma=block_size,
        optimizer_step=optimizer_step,
        seed=seed,
        sequence_layout=sequence_layout,
    )
    query_positions = extended.local_query_positions[:, :block_size]
    label_positions = extended.local_query_positions[:, 1 : block_size + 1]
    global_query_positions = extended.global_query_positions[:, :block_size]
    global_label_positions = extended.global_query_positions[:, 1 : block_size + 1]
    slot_valid = extended.slot_valid[:, 1 : block_size + 1]
    packed_rope_positions = extended.packed_rope_positions[:, :block_size]
    packed_label_rope_positions = extended.packed_rope_positions[
        :, 1 : block_size + 1
    ]
    return DSparkBatchPlan(
        batch_size=extended.batch_size,
        sequence_length=extended.sequence_length,
        anchors_per_sample=extended.anchors_per_sample,
        gamma=block_size - 1,
        block_size=block_size,
        token_valid_mask=extended.token_valid_mask,
        sample_rows=extended.sample_rows,
        anchor_ids=extended.anchor_ids,
        anchor_positions=extended.anchor_positions,
        trunk_lengths=extended.trunk_lengths,
        query_positions=query_positions,
        label_positions=label_positions,
        block_valid=extended.block_valid,
        slot_valid=slot_valid,
        loss_mask=slot_valid,
        global_anchor_positions=extended.global_anchor_positions,
        global_query_positions=global_query_positions,
        global_label_positions=global_label_positions,
        packed_label_rope_positions=packed_label_rope_positions,
        local_anchor_positions=extended.local_anchor_positions,
        local_query_positions=query_positions,
        local_label_positions=label_positions,
        owner_cp_ranks=extended.owner_cp_ranks,
        logical_sample_ids=extended.logical_sample_ids,
        logical_anchor_positions=extended.logical_anchor_positions,
        packed_segment_starts=extended.packed_segment_starts,
        packed_segment_ends=extended.packed_segment_ends,
        packed_rope_positions=packed_rope_positions,
        boundary_valid_mask=extended.boundary_valid_mask[:, 1 : block_size + 1],
        excluded_window_count=extended.excluded_window_count,
        eligible_window_count=extended.eligible_window_count,
    )
