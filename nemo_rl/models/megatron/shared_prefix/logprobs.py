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
"""Next-token logprobs of shared-prefix packs.

A tree is one prompt followed by the completions of the rows that start with that
prompt. A pack is one model forward that holds several trees and runs each prompt
once. The functions here turn the logits of a pack into one row of next-token
logprobs per sequence.
"""

from typing import Any, Optional

import torch
import torch.distributed.nn.functional

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import DistributedLogprob
from nemo_rl.models.megatron.shared_prefix.microbatches import SharedPrefixParams


def shared_prefix_next_token_logprobs(
    packed_logits: torch.Tensor,
    shared_prefix_params: SharedPrefixParams,
    *,
    vocab_parallel_rank: int,
    vocab_parallel_group: torch.distributed.ProcessGroup,
    context_parallel_group: torch.distributed.ProcessGroup,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """Turn the logits of one pack into the next-token logprobs of every row.

    The logits of a prompt exist once per tree, so each prompt logprob is copied to
    every row of the tree. The last prompt token of a tree predicts the first
    completion token of each of its rows. The gathers and copies are differentiable,
    so the gradients of all rows of a tree add up in the one prompt.

    Only scalar logprobs cross ranks. Each context-parallel rank reads the logits it
    holds, at most ``chunk_size`` positions at a time. The tensor-parallel ranks
    reduce each selected logprob over their vocabulary shards. The context-parallel
    ranks then sum their logprobs.

    Args:
        packed_logits: Logits of this rank's part of the pack, with shape
            ``[1, padded length / CP size, vocabulary size / TP size]``.
        shared_prefix_params: The pack that produced ``packed_logits``.
        vocab_parallel_rank: Rank of this process in ``vocab_parallel_group``.
        vocab_parallel_group: Tensor-parallel group that splits the vocabulary.
        context_parallel_group: Context-parallel group that splits the packed
            sequence.
        chunk_size: Most positions whose logits become logprobs at once
            (``policy.logprob_chunk_size``). None handles all positions at once.

    Returns:
        Logprobs of shape ``[rows, sequence_length - 1]``. Row ``r`` is row ``r`` of
        the microbatch. Column ``c`` holds the logprob of token ``c + 1``. Columns
        past the last token of a row hold 0.

    Raises:
        ValueError: If ``packed_logits`` does not have the shape of this rank's part
            of the pack.
    """
    inputs = shared_prefix_params.inputs
    cp_shard = shared_prefix_params.cp_shard
    tp_size = torch.distributed.get_world_size(vocab_parallel_group)
    cp_size = torch.distributed.get_world_size(context_parallel_group)
    padded_length = cp_shard.padded_total_length
    local_length = padded_length // cp_size
    if packed_logits.ndim != 3 or packed_logits.shape[:2] != (1, local_length):
        raise ValueError(
            f"The logits of a pack must have shape [1, {local_length}, vocabulary "
            f"shard] for a padded length of {padded_length} at context-parallel size "
            f"{cp_size}, got {tuple(packed_logits.shape)}."
        )

    device = packed_logits.device
    trees = list(inputs.layout.iter_roots())
    prompt_predictors = torch.cat(
        [
            torch.arange(offset, offset + tree.prefix_len - 1, device=device)
            for offset, tree in trees
        ]
    )
    predictor_positions = torch.cat(
        (prompt_predictors, inputs.indices.predecessor_positions)
    )
    target_positions = torch.cat(
        (prompt_predictors + 1, inputs.indices.completion_positions)
    )
    target_tokens = inputs.packed_input_ids.index_select(0, target_positions)
    local_vocab_size = packed_logits.shape[-1]
    # At TP > 1, DistributedLogprob returns logprob 0 for a target outside every
    # vocabulary shard instead of failing. This device-side check needs no host sync.
    torch._assert_async(
        ((target_tokens >= 0) & (target_tokens < local_vocab_size * tp_size)).all(),
        "A target token of a pack is outside the model's vocabulary.",
    )

    num_predictions = predictor_positions.numel()
    if cp_size == 1:
        owned_predictions = torch.arange(num_predictions, device=device)
        owned_predictors = predictor_positions
        owned_targets = target_tokens
    else:
        packed_to_local = torch.full((padded_length,), -1, device=device)
        packed_to_local[cp_shard.global_token_indices] = torch.arange(
            local_length, device=device
        )
        local_predictors = packed_to_local.index_select(0, predictor_positions)
        # The number of predictors this rank holds depends on the data, so nonzero
        # syncs with the host.
        owned_predictions = torch.nonzero(local_predictors >= 0).flatten()
        owned_predictors = local_predictors.index_select(0, owned_predictions)
        owned_targets = target_tokens.index_select(0, owned_predictions)

    step = num_predictions if chunk_size is None else chunk_size
    local_logits = packed_logits[0]
    selected_logprobs: list[torch.Tensor] = []
    for start in range(0, owned_predictions.numel(), step):
        logits_chunk = local_logits.index_select(
            0, owned_predictors[start : start + step]
        ).to(torch.float32)
        targets_chunk = owned_targets[start : start + step]
        if tp_size == 1:
            selected_logprobs.append(
                torch.log_softmax(logits_chunk, dim=-1)
                .gather(-1, targets_chunk.unsqueeze(-1))
                .squeeze(-1)
            )
        else:
            # The [1, positions, vocabulary] view selects the scatter backward of
            # DistributedLogprob instead of its one-hot backward.
            selected_logprobs.append(
                DistributedLogprob.apply(
                    logits_chunk.unsqueeze(0),
                    targets_chunk.unsqueeze(0),
                    vocab_parallel_rank * local_vocab_size,
                    (vocab_parallel_rank + 1) * local_vocab_size,
                    vocab_parallel_group,
                    not torch.is_grad_enabled(),
                ).squeeze(0)
            )

    # A rank that holds no predictor still needs a graph from its logits for backward.
    graph_zero = local_logits.reshape(-1)[0].to(torch.float32) * 0.0
    local_logprobs = graph_zero + packed_logits.new_zeros(
        num_predictions, dtype=torch.float32
    )
    if selected_logprobs:
        local_logprobs = local_logprobs.index_copy(
            0, owned_predictions, torch.cat(selected_logprobs)
        )
    packed_logprobs = (
        torch.distributed.nn.functional.all_reduce(
            local_logprobs,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group,
        )
        if cp_size > 1
        else local_logprobs
    )

    num_rows = sum(len(tree.completion_lens) for _, tree in trees)
    logprobs = packed_logprobs.new_zeros(
        (num_rows, shared_prefix_params.sequence_length - 1)
    )
    first_row = first_prediction = 0
    for _, tree in trees:
        last_row = first_row + len(tree.completion_lens)
        last_prediction = first_prediction + tree.prefix_len - 1
        logprobs[first_row:last_row, : tree.prefix_len - 1] = packed_logprobs[
            first_prediction:last_prediction
        ]
        first_row, first_prediction = last_row, last_prediction
    logprobs[
        inputs.indices.completion_scatter_rows,
        inputs.indices.completion_scatter_columns,
    ] = packed_logprobs[first_prediction:]
    return logprobs


def prepare_shared_prefix_loss_input(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
    *,
    vocab_parallel_rank: int,
    vocab_parallel_group: torch.distributed.ProcessGroup,
    context_parallel_group: torch.distributed.ProcessGroup,
    shared_prefix_params: SharedPrefixParams,
    chunk_size: Optional[int],
) -> tuple[dict[str, torch.Tensor], BatchedDataDict[Any]]:
    """Prepare the input of a logprob loss from the logits of one pack.

    The loss gets one row of next-token logprobs per sequence, so it masks and
    weights every row on its own.

    Args:
        logits: Logits of this rank's part of the pack.
        data: The rows of the pack.
        loss_fn: Loss function that takes ``next_token_logprobs``.
        vocab_parallel_rank: Rank of this process in ``vocab_parallel_group``.
        vocab_parallel_group: Tensor-parallel group that splits the vocabulary.
        context_parallel_group: Context-parallel group that splits the packed
            sequence.
        shared_prefix_params: The pack that produced ``logits``.
        chunk_size: Most positions whose logits become logprobs at once.

    Returns:
        The loss input ``{"next_token_logprobs": logprobs}`` and ``data``.
    """
    logprobs = shared_prefix_next_token_logprobs(
        logits,
        shared_prefix_params,
        vocab_parallel_rank=vocab_parallel_rank,
        vocab_parallel_group=vocab_parallel_group,
        context_parallel_group=context_parallel_group,
        chunk_size=chunk_size,
    )
    return {"next_token_logprobs": logprobs}, data
