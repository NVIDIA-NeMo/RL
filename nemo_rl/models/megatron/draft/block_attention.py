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

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)

from nemo_rl.models.megatron.draft.block_plan import DFlashBatchPlan
from nemo_rl.models.megatron.draft.context_parallel import gather_projected_kv

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from nemo_rl.models.megatron.draft.sequence_layout import DraftSequenceLayout

_COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)
_FLEX_QUERY_BLOCK_SIZE = 128
_FLEX_KV_BLOCK_SIZE = 128
_FLEX_BLOCK_SIZE = (_FLEX_QUERY_BLOCK_SIZE, _FLEX_KV_BLOCK_SIZE)


def _validate_block_only_attention_inputs(
    *,
    plan: DFlashBatchPlan,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    sequence_layout: DraftSequenceLayout | None,
) -> None:
    tensors = (trunk_k, trunk_v, block_q, block_k, block_v)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("DFlash attention tensors must have rank four")

    num_blocks = plan.sample_rows.numel()
    block_size = plan.block_size
    expected_trunk_shape = (
        (plan.batch_size, plan.sequence_length)
        if sequence_layout is None
        else (1, sequence_layout.owner_cp_rank.numel())
    )
    if trunk_k.shape[:2] != expected_trunk_shape:
        raise ValueError("trunk_k shape does not match the DFlash plan")
    if trunk_v.shape[:2] != expected_trunk_shape:
        raise ValueError("trunk_v shape does not match the DFlash plan")
    if block_q.shape[:2] != (num_blocks, block_size):
        raise ValueError("block_q shape does not match the DFlash plan")
    if block_k.shape[:2] != (num_blocks, block_size):
        raise ValueError("block_k shape does not match the DFlash plan")
    if block_v.shape[:2] != (num_blocks, block_size):
        raise ValueError("block_v shape does not match the DFlash plan")

    device = trunk_k.device
    dtype = trunk_k.dtype
    plan_tensors = (
        plan.token_valid_mask,
        plan.sample_rows,
        plan.anchor_positions,
        plan.slot_valid,
    )
    if sequence_layout is not None:
        plan_tensors = (*plan_tensors, sequence_layout.owner_cp_rank)
    if any(tensor.device != device for tensor in (*tensors, *plan_tensors)):
        raise ValueError("DFlash attention inputs and plan must share a device")
    if not dtype.is_floating_point:
        raise TypeError("DFlash attention inputs must use a floating dtype")
    if any(tensor.dtype != dtype for tensor in tensors):
        raise TypeError("DFlash attention inputs must share a dtype")

    num_query_heads = block_q.shape[2]
    num_kv_heads = trunk_k.shape[2]
    head_dim = block_q.shape[3]
    if num_query_heads == 0 or num_kv_heads == 0:
        raise ValueError("DFlash attention requires nonzero head counts")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    if trunk_k.shape[3] != head_dim or trunk_v.shape[3] != head_dim:
        raise ValueError("DFlash K/V and block query head dimensions must match")
    if trunk_v.shape[2] != num_kv_heads:
        raise ValueError("trunk K/V head counts must match")
    if block_k.shape[2:] != (num_kv_heads, head_dim):
        raise ValueError("trunk and block key shapes must match")
    if block_v.shape[2:] != (num_kv_heads, head_dim):
        raise ValueError("trunk and block value shapes must match")


def _grouped_masked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    visibility: Tensor,
    *,
    scale: float,
) -> Tensor:
    batch_size, query_length, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    heads_per_group = num_query_heads // num_kv_heads

    grouped_query = query.permute(0, 2, 1, 3).reshape(
        batch_size,
        num_kv_heads,
        heads_per_group,
        query_length,
        head_dim,
    )
    key_by_head = key.permute(0, 2, 1, 3)
    value_by_head = value.permute(0, 2, 1, 3)
    scores = torch.einsum(
        "bhgqd,bhkd->bhgqk",
        grouped_query,
        key_by_head,
    )
    scores = scores * scale

    expanded_visibility = visibility[:, None, None, :, :]
    row_valid = visibility.any(dim=-1)
    scores = scores.masked_fill(~expanded_visibility, -torch.inf)
    scores = torch.where(
        row_valid[:, None, None, :, None],
        scores,
        torch.zeros_like(scores),
    )
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    probabilities = torch.where(
        expanded_visibility,
        probabilities,
        torch.zeros_like(probabilities),
    )
    output = torch.einsum(
        "bhgqk,bhkd->bhgqd",
        probabilities,
        value_by_head,
    )
    return (
        output.reshape(batch_size, num_query_heads, query_length, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _block_visibility(
    plan: DFlashBatchPlan,
    sequence_layout: DraftSequenceLayout | None,
) -> Tensor:
    num_blocks = plan.sample_rows.numel()
    device = plan.token_valid_mask.device
    if sequence_layout is not None:
        trunk_key_count = sequence_layout.owner_cp_rank.numel()
        trunk_positions = torch.arange(
            trunk_key_count,
            dtype=torch.int64,
            device=device,
        )
        visible_trunk = (
            (trunk_positions[None, :] >= plan.packed_segment_starts[:, None])
            & (trunk_positions[None, :] < plan.global_anchor_positions[:, None])
            & sequence_layout.packed_valid_mask[None, :]
        )
    elif plan.sequence_length == 0:
        trunk_key_count = 0
        visible_trunk = torch.zeros(
            (num_blocks, 0),
            dtype=torch.bool,
            device=device,
        )
    else:
        trunk_key_count = plan.batch_size * plan.sequence_length
        trunk_key_indices = torch.arange(
            trunk_key_count,
            dtype=torch.int64,
            device=device,
        )
        trunk_key_rows = torch.div(
            trunk_key_indices,
            plan.sequence_length,
            rounding_mode="floor",
        )
        trunk_key_positions = torch.remainder(
            trunk_key_indices,
            plan.sequence_length,
        )
        visible_trunk = (
            (trunk_key_rows[None, :] == plan.sample_rows[:, None])
            & (trunk_key_positions[None, :] < plan.anchor_positions[:, None])
            & plan.token_valid_mask.reshape(-1)[None, :]
        )

    block_key_count = num_blocks * plan.block_size
    block_key_indices = torch.arange(
        block_key_count,
        dtype=torch.int64,
        device=device,
    )
    block_key_rows = torch.div(
        block_key_indices,
        plan.block_size,
        rounding_mode="floor",
    )
    visible_block = (
        block_key_rows[None, :] == torch.arange(num_blocks, device=device)[:, None]
    ) & plan.slot_valid.reshape(-1)[None, :]

    visible_keys = torch.cat((visible_trunk, visible_block), dim=-1)
    return plan.slot_valid[:, :, None] & visible_keys[:, None, :]


def _create_global_block_mask(
    plan: DFlashBatchPlan,
    sequence_layout: DraftSequenceLayout | None,
) -> BlockMask:
    num_blocks = plan.sample_rows.numel()
    trunk_key_count = (
        plan.batch_size * plan.sequence_length
        if sequence_layout is None
        else sequence_layout.owner_cp_rank.numel()
    )
    block_key_count = num_blocks * plan.block_size
    global_key_count = trunk_key_count + block_key_count
    num_query_blocks = (
        plan.block_size + _FLEX_QUERY_BLOCK_SIZE - 1
    ) // _FLEX_QUERY_BLOCK_SIZE
    num_kv_blocks = (global_key_count + _FLEX_KV_BLOCK_SIZE - 1) // _FLEX_KV_BLOCK_SIZE

    kv_block_indices = torch.arange(
        num_kv_blocks,
        dtype=torch.int64,
        device=plan.token_valid_mask.device,
    )
    kv_block_starts = kv_block_indices * _FLEX_KV_BLOCK_SIZE
    kv_block_ends = torch.clamp(
        kv_block_starts + _FLEX_KV_BLOCK_SIZE,
        max=global_key_count,
    )
    if sequence_layout is None:
        sample_starts = plan.sample_rows[:, None] * plan.sequence_length
        sample_prefix_ends = sample_starts + plan.anchor_positions[:, None]
        token_valid_mask = plan.token_valid_mask.reshape(-1)
    else:
        sample_starts = plan.packed_segment_starts[:, None]
        sample_prefix_ends = plan.global_anchor_positions[:, None]
        token_valid_mask = sequence_layout.packed_valid_mask
    own_block_starts = (
        trunk_key_count
        + torch.arange(
            num_blocks,
            dtype=torch.int64,
            device=plan.token_valid_mask.device,
        )[:, None]
        * plan.block_size
    )
    own_block_ends = own_block_starts + plan.block_size
    candidate_blocks = (
        (kv_block_starts[None, :] < sample_prefix_ends)
        & (kv_block_ends[None, :] > sample_starts)
    ) | (
        (kv_block_starts[None, :] < own_block_ends)
        & (kv_block_ends[None, :] > own_block_starts)
    )
    base_kv_num_blocks = candidate_blocks.sum(dim=-1, dtype=torch.int32)
    max_trunk_length = (
        plan.sequence_length if sequence_layout is None else trunk_key_count
    )
    max_trunk_blocks = (
        max_trunk_length + 2 * _FLEX_KV_BLOCK_SIZE - 2
    ) // _FLEX_KV_BLOCK_SIZE
    max_own_blocks = (
        plan.block_size + 2 * _FLEX_KV_BLOCK_SIZE - 2
    ) // _FLEX_KV_BLOCK_SIZE
    max_candidate_blocks = min(
        num_kv_blocks,
        max_trunk_blocks + max_own_blocks,
    )
    base_kv_indices = torch.argsort(
        candidate_blocks.to(torch.int8),
        dim=-1,
        descending=True,
        stable=True,
    )[:, :max_candidate_blocks].to(torch.int32)
    kv_num_blocks = base_kv_num_blocks[:, None, None].expand(
        num_blocks,
        1,
        num_query_blocks,
    )
    kv_indices = base_kv_indices[:, None, None, :].expand(
        num_blocks,
        1,
        num_query_blocks,
        max_candidate_blocks,
    )
    q_num_blocks = candidate_blocks[:, None, :].to(torch.int32) * num_query_blocks
    q_indices = torch.arange(
        num_query_blocks,
        dtype=torch.int32,
        device=plan.token_valid_mask.device,
    )[None, None, None, :].expand(
        num_blocks,
        1,
        num_kv_blocks,
        num_query_blocks,
    )

    token_valid_mask = torch.cat(
        (
            token_valid_mask,
            torch.zeros(
                1,
                dtype=torch.bool,
                device=plan.token_valid_mask.device,
            ),
        )
    )
    slot_valid = plan.slot_valid.reshape(-1)

    def global_mask(
        block_index: Tensor,
        _head_index: Tensor,
        query_index: Tensor,
        key_index: Tensor,
    ) -> Tensor:
        safe_query_index = torch.clamp(query_index, max=plan.block_size - 1)
        safe_trunk_index = torch.clamp(
            key_index,
            max=max(trunk_key_count - 1, 0),
        )
        if sequence_layout is None:
            safe_sequence_length = max(plan.sequence_length, 1)
            trunk_row = torch.div(
                safe_trunk_index,
                safe_sequence_length,
                rounding_mode="floor",
            )
            trunk_position = torch.remainder(
                safe_trunk_index,
                safe_sequence_length,
            )
            visible_trunk = (
                (key_index < trunk_key_count)
                & (trunk_row == plan.sample_rows[block_index])
                & (trunk_position < plan.anchor_positions[block_index])
                & token_valid_mask[safe_trunk_index]
            )
        else:
            visible_trunk = (
                (key_index < trunk_key_count)
                & (safe_trunk_index >= plan.packed_segment_starts[block_index])
                & (safe_trunk_index < plan.global_anchor_positions[block_index])
                & token_valid_mask[safe_trunk_index]
            )

        safe_block_index = torch.clamp(
            key_index - trunk_key_count,
            min=0,
            max=block_key_count - 1,
        )
        key_block_index = torch.div(
            safe_block_index,
            plan.block_size,
            rounding_mode="floor",
        )
        visible_block = (
            (key_index >= trunk_key_count)
            & (key_index < global_key_count)
            & (key_block_index == block_index)
            & slot_valid[safe_block_index]
        )
        return (
            (query_index < plan.block_size)
            & plan.slot_valid[block_index, safe_query_index]
            & (visible_trunk | visible_block)
        )

    return BlockMask(
        seq_lengths=(plan.block_size, global_key_count),
        kv_num_blocks=kv_num_blocks.contiguous(),
        kv_indices=kv_indices.contiguous(),
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_num_blocks.contiguous(),
        q_indices=q_indices.contiguous(),
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
        mask_mod=global_mask,
    )


def _flex_block_only_attention_cuda(
    *,
    plan: DFlashBatchPlan,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    scale: float,
    sequence_layout: DraftSequenceLayout | None,
) -> Tensor:
    num_kv_heads = trunk_k.shape[2]
    head_dim = trunk_k.shape[3]
    global_key = torch.cat(
        (
            trunk_k.reshape(1, -1, num_kv_heads, head_dim),
            block_k.reshape(1, -1, num_kv_heads, head_dim),
        ),
        dim=1,
    ).permute(0, 2, 1, 3)
    global_value = torch.cat(
        (
            trunk_v.reshape(1, -1, num_kv_heads, head_dim),
            block_v.reshape(1, -1, num_kv_heads, head_dim),
        ),
        dim=1,
    ).permute(0, 2, 1, 3)
    return (
        cast(
            Tensor,
            _COMPILED_FLEX_ATTENTION(
                block_q.permute(0, 2, 1, 3),
                global_key,
                global_value,
                block_mask=_create_global_block_mask(plan, sequence_layout),
                scale=scale,
                enable_gqa=block_q.shape[2] != num_kv_heads,
            ),
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def dflash_block_only_attention(
    *,
    plan: DFlashBatchPlan,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    sequence_layout: DraftSequenceLayout | None = None,
    context_parallel_group: ProcessGroup | None = None,
    scale: float | None = None,
) -> Tensor:
    """Apply bidirectional anchored-block attention without trunk queries."""
    if sequence_layout is not None:
        trunk_k = gather_projected_kv(
            trunk_k,
            sequence_layout=sequence_layout,
            cp_group=context_parallel_group,
            sequence_dim=1,
        )
        trunk_v = gather_projected_kv(
            trunk_v,
            sequence_layout=sequence_layout,
            cp_group=context_parallel_group,
            sequence_dim=1,
        )
    elif context_parallel_group is not None:
        raise ValueError("context_parallel_group requires a draft sequence layout")
    _validate_block_only_attention_inputs(
        plan=plan,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
        sequence_layout=sequence_layout,
    )
    effective_scale = block_q.shape[-1] ** -0.5 if scale is None else scale
    if not math.isfinite(effective_scale):
        raise ValueError("attention scale must be finite")

    if block_q.device.type == "cuda":
        block_output = _flex_block_only_attention_cuda(
            plan=plan,
            trunk_k=trunk_k,
            trunk_v=trunk_v,
            block_q=block_q,
            block_k=block_k,
            block_v=block_v,
            scale=effective_scale,
            sequence_layout=sequence_layout,
        )
    else:
        num_kv_heads = trunk_k.shape[2]
        head_dim = trunk_k.shape[3]
        global_key = torch.cat(
            (
                trunk_k.reshape(1, -1, num_kv_heads, head_dim),
                block_k.reshape(1, -1, num_kv_heads, head_dim),
            ),
            dim=1,
        )
        global_value = torch.cat(
            (
                trunk_v.reshape(1, -1, num_kv_heads, head_dim),
                block_v.reshape(1, -1, num_kv_heads, head_dim),
            ),
            dim=1,
        )
        block_output = _grouped_masked_attention(
            block_q,
            global_key,
            global_value,
            _block_visibility(plan, sequence_layout),
            scale=effective_scale,
        )
    return torch.where(
        plan.slot_valid[:, :, None, None],
        block_output,
        torch.zeros_like(block_output),
    )
