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

from dataclasses import replace
from typing import Mapping

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_expert_tensor_and_model_parallel_group,
)

from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.models.megatron.common import _round_up_to_multiple


def uses_hybridep_flex_dispatcher(megatron_cfg: Mapping[str, object]) -> bool:
    return (
        megatron_cfg.get("moe_token_dispatcher_type") == "flex"
        and megatron_cfg.get("moe_flex_dispatcher_backend") == "hybridep"
    )


def _get_hybridep_aligned_seq_len(
    local_seq_len: int,
    multiple: int,
    device: torch.device,
) -> int:
    target = torch.tensor([local_seq_len], dtype=torch.int64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = get_expert_tensor_and_model_parallel_group(check_initialized=False)
        if group is None:
            raise RuntimeError("HybridEP alignment group is not initialized.")
        torch.distributed.all_reduce(
            target,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
        )

    target_seq_len = int(target.item())
    if multiple > 1:
        target_seq_len = _round_up_to_multiple(target_seq_len, multiple)
    return target_seq_len


def _get_hybridep_local_pad_multiple(
    pad_packed_seq_to_multiple_of: int,
    cp_size: int,
) -> int:
    if cp_size == 1:
        return pad_packed_seq_to_multiple_of
    if pad_packed_seq_to_multiple_of <= 1:
        return 1
    assert pad_packed_seq_to_multiple_of % cp_size == 0, (
        "HybridEP packed sequence multiple must be divisible by context "
        f"parallel size; got multiple={pad_packed_seq_to_multiple_of}, "
        f"cp_size={cp_size}."
    )
    return max(1, pad_packed_seq_to_multiple_of // cp_size)


def _get_packed_seq_boundaries(
    cu_seqlens_padded: torch.Tensor,
) -> list[tuple[int, int]]:
    boundaries = cu_seqlens_padded.detach().cpu().tolist()
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(len(boundaries) - 1)
    ]


def _shard_packed_seq_on_this_cp_rank(
    packed_tensor: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    if cp_size == 1:
        return packed_tensor

    cp_chunks = []
    for start, end in _get_packed_seq_boundaries(cu_seqlens_padded):
        slices = [slice(None)] * packed_tensor.dim()
        slices[seq_dim] = slice(start, end)
        cp_chunks.append(
            _get_tokens_on_this_cp_rank(
                packed_tensor[tuple(slices)],
                cp_rank,
                cp_size,
                seq_dim=seq_dim,
            )
        )
    return torch.cat(cp_chunks, dim=seq_dim).contiguous()


def pad_packed_seq_for_hybridep(
    input_ids: torch.Tensor,
    input_ids_cp_sharded: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    cu_seqlens_padded: torch.Tensor,
    pad_packed_seq_to_multiple_of: int,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, PackedSeqParams, torch.Tensor]:
    """Align packed inputs once, before model collectives can overlap."""
    local_seq_len = input_ids_cp_sharded.shape[1]
    local_pad_multiple = _get_hybridep_local_pad_multiple(
        pad_packed_seq_to_multiple_of,
        cp_size,
    )
    target_seq_len = _get_hybridep_aligned_seq_len(
        local_seq_len,
        local_pad_multiple,
        input_ids_cp_sharded.device,
    )

    if target_seq_len == local_seq_len:
        return input_ids, input_ids_cp_sharded, packed_seq_params, cu_seqlens_padded

    local_pad_len = target_seq_len - local_seq_len
    full_pad_len = local_pad_len * cp_size
    input_ids = torch.nn.functional.pad(input_ids, (0, full_pad_len), value=0)

    cu_seqlens_padded = cu_seqlens_padded.clone()
    cu_seqlens_padded[-1] += full_pad_len
    input_ids_cp_sharded = _shard_packed_seq_on_this_cp_rank(
        input_ids,
        cu_seqlens_padded,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
    assert input_ids_cp_sharded.shape[1] == target_seq_len, (
        "HybridEP CP-local input length must match the aligned target length; "
        f"got {input_ids_cp_sharded.shape[1]} vs {target_seq_len}."
    )

    max_last_sequence_len = int(cu_seqlens_padded[-1] - cu_seqlens_padded[-2])
    max_seqlen = max(int(packed_seq_params.max_seqlen_q), max_last_sequence_len)
    packed_seq_params = replace(
        packed_seq_params,
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        total_tokens=target_seq_len,
    )
    return input_ids, input_ids_cp_sharded, packed_seq_params, cu_seqlens_padded


def get_packed_seq_padding_mask(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """Return a mask whose true entries are physical THD padding tokens."""
    token_positions = torch.arange(
        total_tokens,
        dtype=cu_seqlens_padded.dtype,
        device=cu_seqlens_padded.device,
    )
    sequence_indices = torch.searchsorted(
        cu_seqlens_padded[1:].contiguous(), token_positions, right=True
    )
    padded_starts = cu_seqlens_padded[:-1].index_select(0, sequence_indices)
    valid_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).index_select(0, sequence_indices)
    return ((token_positions - padded_starts) >= valid_lengths).unsqueeze(0)
