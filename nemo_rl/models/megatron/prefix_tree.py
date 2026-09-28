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

from dataclasses import dataclass, fields
from typing import Any, Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


@dataclass
class PrefixTreePackedSeqParams(PackedSeqParams):
    """Row packing plus the token tree the decoder runs on.

    The inherited fields describe the ordinary packed rows, so embeddings, MTP
    and the output layer are unchanged. Inside the decoder, tokens are unique
    and in DFS order: ``node_of`` maps each packed row position to its unique
    token and ``owner`` maps each unique token to one packed row position.
    """

    node_of: Optional[torch.Tensor] = None
    owner: Optional[torch.Tensor] = None
    block_mask: Any = None


def share_packed_prefixes(
    input_ids: torch.Tensor,
    rows: PackedSeqParams,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    routed_experts: Optional[torch.Tensor],
    multiple_of: int,
) -> tuple[PrefixTreePackedSeqParams, Optional[torch.Tensor]]:
    """Merge equal token prefixes of packed rows into one token tree.

    Rows are visited in lexicographic order and each row shares its longest
    common prefix with the previous row, which appends new tokens in DFS order.
    Each token's subtree is then the contiguous range ``[token, subtree_end)``.
    Rows with router replay only share prefixes whose routes also match.
    """
    # Built on CPU: the loop is per row, and GPU tensors would sync every step.
    device = input_ids.device
    ids = input_ids[0].cpu()
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    starts = cu_seqlens_padded[:-1].tolist()
    routes = None if routed_experts is None else routed_experts[0].flatten(1).cpu()
    node_of = torch.full_like(ids, -1)
    subtree_end = torch.empty_like(ids)
    count = 0
    previous = None
    for row in sorted(
        range(len(lengths)),
        key=lambda r: ids[starts[r] : starts[r] + lengths[r]].tolist(),
    ):
        start, length = starts[row], lengths[row]
        shared = 0
        if previous is not None:
            span = min(length, previous[1])
            same = ids[start : start + span] == ids[previous[0] : previous[0] + span]
            if routes is not None:
                same &= (
                    routes[start : start + span]
                    == routes[previous[0] : previous[0] + span]
                ).all(1)
            shared = int(same.cumprod(0).sum())
            node_of[start : start + shared] = node_of[
                previous[0] : previous[0] + shared
            ]
        node_of[start + shared : start + length] = torch.arange(
            count, count + length - shared
        )
        count += length - shared
        subtree_end[node_of[start : start + length]] = count
        previous = (start, length)

    # Padding positions become isolated tokens that attend only to themselves.
    padding = (node_of < 0).nonzero().flatten()
    node_of[padding] = torch.arange(count, count + padding.numel())
    unique_count = -(-(count + padding.numel()) // multiple_of) * multiple_of
    subtree_end = torch.cat(
        [subtree_end[:count], torch.arange(count + 1, unique_count + 1)]
    ).to(device)
    owner = torch.zeros(unique_count, dtype=torch.long)
    owner[node_of] = torch.arange(ids.numel())
    node_of, owner = node_of.to(device), owner.to(device)
    block_mask = create_block_mask(
        lambda b, h, q, kv: (kv <= q) & (q < subtree_end[kv]),
        None,
        None,
        unique_count,
        unique_count,
        device=device,
    )
    params = PrefixTreePackedSeqParams(
        **{f.name: getattr(rows, f.name) for f in fields(PackedSeqParams) if f.init},
        node_of=node_of,
        owner=owner,
        block_mask=block_mask,
    )
    if routed_experts is not None:
        routed_experts = routed_experts.index_select(1, owner)
    return params, routed_experts


def install_prefix_tree(model: torch.nn.Module) -> None:
    """Run the decoder on the prefix tree whenever one is packed."""
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.transformer.attention import Attention

    # Only the decoder runs on the tree, MTP layers keep the row layout.
    decoder = next(m.decoder for m in model.modules() if hasattr(m, "decoder"))
    decoder.forward = _tree_decoder(decoder.forward, decoder.config.sequence_parallel)
    for module in decoder.modules():
        if isinstance(module, Attention):
            module.core_attention.forward = _tree_attention(
                module.core_attention.forward
            )
        elif isinstance(module, MambaMixer):
            module._ssm_training = _tree_ssm(module._ssm_training)


def _select_tokens(hidden_states, index, sequence_parallel):
    if not sequence_parallel:
        return hidden_states.index_select(0, index)
    from megatron.core.tensor_parallel import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    # Every rank holds the same full sequence, so backward only splits.
    hidden_states = gather_from_sequence_parallel_region(
        hidden_states, tensor_parallel_output_grad=False
    )
    return scatter_to_sequence_parallel_region(hidden_states.index_select(0, index))


def _tree_decoder(forward, sequence_parallel):
    def wrapped(hidden_states, *args, packed_seq_params=None, **kwargs):
        if not isinstance(packed_seq_params, PrefixTreePackedSeqParams):
            return forward(
                hidden_states, *args, packed_seq_params=packed_seq_params, **kwargs
            )
        if hasattr(hidden_states, "unwrap"):
            hidden_states = hidden_states.unwrap()
        hidden_states = _select_tokens(
            hidden_states, packed_seq_params.owner, sequence_parallel
        )
        output = forward(
            hidden_states, *args, packed_seq_params=packed_seq_params, **kwargs
        )
        rows = output[0] if isinstance(output, tuple) else output
        rows = _select_tokens(rows, packed_seq_params.node_of, sequence_parallel)
        return (rows, *output[1:]) if isinstance(output, tuple) else rows

    return wrapped


def _tree_attention(forward):
    def wrapped(query, key, value, *args, packed_seq_params=None, **kwargs):
        if not isinstance(packed_seq_params, PrefixTreePackedSeqParams):
            return forward(
                query, key, value, *args, packed_seq_params=packed_seq_params, **kwargs
            )
        q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (query, key, value))
        out = _compiled_flex_attention(
            q, k, v, block_mask=packed_seq_params.block_mask, enable_gqa=True
        )
        return out[0].transpose(0, 1).reshape(query.shape[0], -1)

    return wrapped


def _tree_ssm(ssm_training):
    # Scan the original rows so every row keeps its exact recurrent state.
    def wrapped(zxBCdt, packed_seq_params=None):
        if not isinstance(packed_seq_params, PrefixTreePackedSeqParams):
            return ssm_training(zxBCdt, packed_seq_params)
        y = ssm_training(
            zxBCdt.index_select(0, packed_seq_params.node_of), packed_seq_params
        )
        return y.index_select(0, packed_seq_params.owner)

    return wrapped
