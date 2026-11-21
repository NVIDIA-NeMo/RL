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
"""Microbatches of shared-prefix packs.

A tree is one prompt followed by the completions of the rows that start with that
prompt. A pack is one model forward that holds several trees. The packed sequence of
a pack stores each tree's prompt once, then the completion of each row of the tree.
A dense bin is a model forward of ordinary packed sequences, which share nothing.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_tensor_model_parallel_world_size,
)

from nemo_rl.data.packing.shared_prefix.row_metadata import (
    SHARED_PREFIX_PROMPT_LENGTHS,
    SHARED_PREFIX_TREE_IDS,
    UNSHARED_ROW,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from megatron.core.datasets.shared_prefix.inputs import (
        SharedPrefixContextParallelShard,
        SharedPrefixForwardInputs,
    )

    from nemo_rl.models.megatron.data import ProcessedMicrobatch


@dataclass(frozen=True)
class SharedPrefixParams:
    """Tensors that describe one pack, as ``PackedSeqParams`` describes one dense bin.

    Attributes:
        inputs: The pack before context-parallel sharding. ``inputs.layout`` holds the
            lengths of every tree. ``inputs.packed_input_ids`` holds the packed
            tokens. ``inputs.indices`` maps packed positions to rows and columns of
            the microbatch.
        cp_shard: This context-parallel rank's part of the padded pack.
            ``cp_shard.global_token_indices`` holds the packed position of each local
            token.
        sequence_length: Width of the rows of the microbatch.
        exclude_sequence_padding_from_expert_bias: Whether the padding at the end of
            each row stays out of the MoE expert-bias statistics. It has the value
            that the dense bins of the same step use.
    """

    inputs: "SharedPrefixForwardInputs"
    cp_shard: "SharedPrefixContextParallelShard"
    sequence_length: int
    exclude_sequence_padding_from_expert_bias: bool

    def model_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments that make the model forward run this pack."""
        return {
            "shared_prefix_layout": self.inputs.layout,
            "shared_prefix_exclude_sequence_padding_from_expert_bias": (
                self.exclude_sequence_padding_from_expert_bias
            ),
        }


def shared_prefix_tree_sizes(
    data_dict: BatchedDataDict[Any],
) -> Optional[tuple[int, ...]]:
    """Return the number of rows of each tree when a microbatch is a pack.

    The driver plans every microbatch as a pack or as a dense bin. Every row of a pack
    has a tree id of 0 or more, and the rows of a tree are consecutive. Every row of a
    dense bin has the tree id ``UNSHARED_ROW``. A batch that the driver planned
    without shared prefixes has no tree-id column.

    Args:
        data_dict: The rows of one microbatch.

    Returns:
        The tree sizes in row order, or None when the microbatch is a dense bin.

    Raises:
        ValueError: If the microbatch mixes rows of trees with unshared rows, or if
            the ``token_mask`` of a row is nonzero inside its shared prompt.
    """
    if SHARED_PREFIX_TREE_IDS not in data_dict:
        return None
    tree_ids = data_dict[SHARED_PREFIX_TREE_IDS]
    if (tree_ids == UNSHARED_ROW).all():
        return None
    if (tree_ids < 0).any():
        raise ValueError(
            "A microbatch must be a pack or a dense bin, but its tree ids mix rows "
            f"of trees with unshared rows: {tree_ids.tolist()}."
        )
    token_mask = data_dict["token_mask"]
    # The worker builds the prompt lengths on the host. A fetch can return
    # token_mask on the GPU.
    prompt_lengths = data_dict[SHARED_PREFIX_PROMPT_LENGTHS].to(token_mask.device)
    columns = torch.arange(token_mask.shape[1], device=token_mask.device)
    loss_in_prompt = (token_mask != 0) & (columns < prompt_lengths.unsqueeze(1))
    if loss_in_prompt.any():
        row, column = loss_in_prompt.nonzero()[0].tolist()
        raise ValueError(
            f"Row {row} of a pack has a nonzero token_mask at column {column}, inside "
            f"its shared prompt of {int(prompt_lengths[row])} tokens. Rows may share "
            "only tokens that carry no loss."
        )
    return tuple(torch.unique_consecutive(tree_ids, return_counts=True)[1].tolist())


def build_shared_prefix_microbatch(
    data_dict: BatchedDataDict[Any],
    *,
    tree_sizes: Sequence[int],
    padding_multiple: int,
    exclude_sequence_padding_from_expert_bias: bool,
) -> "ProcessedMicrobatch":
    """Build the model inputs of one pack.

    Every row is padded until its prompt plus its completion is a multiple of
    ``padding_multiple``. The pack is then padded at its end to the same multiple.
    The model input is this context-parallel rank's part of that sequence.

    Args:
        data_dict: The rows of the pack on the model's device, tree by tree. It holds
            ``input_ids``, ``input_lengths`` and the ``SHARED_PREFIX_PROMPT_LENGTHS``
            column. ``mtp_loss_mask`` is packed like the tokens when it is present.
        tree_sizes: Number of rows of each tree, from ``shared_prefix_tree_sizes``.
        padding_multiple: ``policy.make_sequence_length_divisible_by``, the multiple
            that the driver planned the pack with.
        exclude_sequence_padding_from_expert_bias: Whether the dense bins of this step
            keep the padding at the end of each row out of the MoE expert-bias
            statistics.

    Returns:
        The microbatch of the pack. ``data_dict`` keeps one row per sequence for the
        loss. ``input_ids_cp_sharded``, ``position_ids`` and ``mtp_loss_mask`` follow
        the packed sequence.
    """
    # Optional dependency: Megatron-LM builds without shared-prefix support lack it.
    from megatron.core.datasets.shared_prefix.inputs import (
        build_shared_prefix_inputs,
        gather_shared_prefix_token_aligned,
        shard_shared_prefix_inputs_for_context_parallel,
        shard_shared_prefix_token_aligned_for_context_parallel,
    )

    # Circular: nemo_rl.models.megatron.data imports this module.
    from nemo_rl.models.megatron.data import ProcessedMicrobatch

    input_ids = data_dict["input_ids"]
    inputs = build_shared_prefix_inputs(
        input_ids,
        input_lengths=data_dict["input_lengths"],
        prompt_lengths=data_dict[SHARED_PREFIX_PROMPT_LENGTHS],
        tree_sizes=tree_sizes,
        padding_multiple=padding_multiple,
        # A dense bin of the same rows is one MTP loss group, so the pack is one too.
        mtp_loss_group_tree_counts=(len(tree_sizes),),
    )
    cp_shard = shard_shared_prefix_inputs_for_context_parallel(
        inputs,
        cp_rank=get_context_parallel_rank(),
        cp_size=get_context_parallel_world_size(),
        tp_size=get_tensor_model_parallel_world_size(),
    )
    mtp_loss_mask = None
    if "mtp_loss_mask" in data_dict:
        mtp_loss_mask = shard_shared_prefix_token_aligned_for_context_parallel(
            gather_shared_prefix_token_aligned(
                data_dict["mtp_loss_mask"], inputs=inputs
            ),
            shard=cp_shard,
        ).unsqueeze(0)
    return ProcessedMicrobatch(
        data_dict=data_dict,
        input_ids=input_ids,
        input_ids_cp_sharded=cp_shard.packed_input_ids.unsqueeze(0),
        attention_mask=None,
        position_ids=cp_shard.position_ids.unsqueeze(0),
        packed_seq_params=None,
        cu_seqlens_padded=None,
        mtp_loss_mask=mtp_loss_mask,
        original_seq_length=input_ids.shape[1],
        shared_prefix_params=SharedPrefixParams(
            inputs=inputs,
            cp_shard=cp_shard,
            sequence_length=input_ids.shape[1],
            exclude_sequence_padding_from_expert_bias=(
                exclude_sequence_padding_from_expert_bias
            ),
        ),
    )
