# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Optional

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from torch.distributed.tensor import DTensor, distribute_tensor
from megatron.core.utils import deprecate_inference_params, get_pg_size

@torch.no_grad()
def _compute_distributed_log_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Compute a stable distributed log softmax across tensor parallel workers.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L265

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_length, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduce operations.

    Returns:
        torch.Tensor: Log softmax output with the same shape as input, but values represent
            log probabilities normalized across the full vocabulary dimension.
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


class DistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    Taken from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        vocab_parallel_logits = vocab_parallel_logits.to(dtype=torch.float32)

        log_probs = _compute_distributed_log_softmax(vocab_parallel_logits, group=group)
        softmax_output = log_probs.exp()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        softmax, target_mask, masked_target = ctx.saved_tensors

        if softmax.ndim == 3:
            B, S, V = softmax.shape

            # skip `torch.nn.functional.one_hot`
            row = (
                torch.arange(B, device=softmax.device)
                .view(-1, 1)
                .expand(-1, S)
                .reshape(-1)
            )
            col = torch.arange(S, device=softmax.device).expand(B, -1).reshape(-1)
            flat_idx = (row * S + col) * V

            flat_chosen = flat_idx.masked_select(
                ~target_mask.reshape(-1)
            ) + masked_target.masked_select(~target_mask)

            # `neg` is zero-copy
            grad_input = softmax.neg()
            grad_input = grad_input.mul_(grad_output.unsqueeze(-1))

            grad_output_selected = grad_output.masked_select(~target_mask)
            grad_input.view(-1).scatter_add_(0, flat_chosen, grad_output_selected)
        else:
            V = softmax.size(-1)
            is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(
                masked_target, num_classes=V
            )
            grad_input = is_chosen.float().sub_(softmax)
            grad_input.mul_(grad_output.unsqueeze(-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


class ChunkedDistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    The log probabilities computation is chunked in the sequence dimension
    to mitigate GPU OOM (especially during backward pass).
    In addition, logits casting from float16 or bfloat16 -> float32 is performed
    inside the chunk loop to avoid materializing a whole float32 logits tensor.

    Adapted from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        chunk_size: int,
        tp_group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size
        all_log_probs = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            log_probs = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )

            log_probs = torch.gather(
                log_probs, -1, masked_target[:, chunk_start:chunk_end].unsqueeze(-1)
            ).squeeze(-1)
            log_probs[target_mask[:, chunk_start:chunk_end]] = 0.0

            torch.distributed.all_reduce(
                log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )

            all_log_probs.append(log_probs)

        log_probs = torch.cat(all_log_probs, dim=1)

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(vocab_parallel_logits, target_mask, masked_target)
            ctx.chunk_size = chunk_size
            ctx.tp_group = tp_group

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        vocab_parallel_logits, target_mask, masked_target = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        tp_group = ctx.tp_group

        partition_vocab_size = int(vocab_parallel_logits.shape[-1])
        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size

        all_grad_input = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            softmax_output = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )
            softmax_output = softmax_output.exp()

            # 1 if it's the chosen log prob, 0 otherwise
            is_chosen = (~(target_mask[:, chunk_start:chunk_end])).unsqueeze(
                -1
            ) * torch.nn.functional.one_hot(
                masked_target[:, chunk_start:chunk_end],
                num_classes=partition_vocab_size,
            )

            grad_input = is_chosen.float().sub_(softmax_output)

            grad_input.mul_(grad_output[:, chunk_start:chunk_end].unsqueeze(dim=-1))

            all_grad_input.append(grad_input)

        grad_input = torch.cat(all_grad_input, dim=1)

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


class ChunkedDistributedGatherLogprob(torch.autograd.Function):
    """Compute distributed log-softmax once and gather logprobs at given global indices.

    Forward computes per-chunk distributed log-softmax across TP, gathers selected
    log probabilities at the provided global indices (shape [B, S, K]), and returns
    a tensor of shape [B, S, K].

    Backward recomputes per-chunk softmax from logits and applies the gradient rule:
      dL/dz = -softmax * sum_k(dL/dy_k) + scatter_add(dL/dy_k) over selected indices.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,  # [B, S, V_local]
        global_indices: torch.Tensor,  # [B, S, K]
        vocab_start_index: int,
        vocab_end_index: int,
        chunk_size: int,
        tp_group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        B, S, V_local = vocab_parallel_logits.shape
        num_chunks = (int(S) + chunk_size - 1) // chunk_size
        out_chunks: list[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            s0 = chunk_idx * chunk_size
            s1 = min(int(S), (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, s0:s1, :].to(dtype=torch.float32)
            # distributed log softmax along full vocab
            log_probs = _compute_distributed_log_softmax(logits, group=tp_group)

            gi = global_indices[:, s0:s1, :]
            in_range = (gi >= int(vocab_start_index)) & (gi < int(vocab_end_index))
            li = (gi - int(vocab_start_index)).clamp(min=0, max=V_local - 1)

            local_vals = torch.gather(log_probs, dim=-1, index=li)
            local_vals = local_vals * in_range.to(dtype=local_vals.dtype)

            torch.distributed.all_reduce(
                local_vals, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )

            out_chunks.append(local_vals)

        out = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]

        if not inference_only:
            ctx.save_for_backward(vocab_parallel_logits, global_indices)
            ctx.chunk_size = int(chunk_size)
            ctx.tp_group = tp_group
            ctx.vocab_start_index = int(vocab_start_index)
            ctx.vocab_end_index = int(vocab_end_index)

        return out.contiguous()

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]  # [B, S, K]
        vocab_parallel_logits, global_indices = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        tp_group = ctx.tp_group
        vocab_start_index = ctx.vocab_start_index
        vocab_end_index = ctx.vocab_end_index

        B, S, V_local = vocab_parallel_logits.shape
        num_chunks = (int(S) + chunk_size - 1) // chunk_size
        all_grad_input: list[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            s0 = chunk_idx * chunk_size
            s1 = min(int(S), (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, s0:s1, :].to(dtype=torch.float32)
            log_probs = _compute_distributed_log_softmax(logits, group=tp_group)
            softmax_output = log_probs.exp()

            gi = global_indices[:, s0:s1, :]
            in_range = (gi >= int(vocab_start_index)) & (gi < int(vocab_end_index))
            li = (gi - int(vocab_start_index)).clamp(min=0, max=V_local - 1)

            # Sum over K for the softmax term
            go_chunk = grad_output[:, s0:s1, :]  # [B, Sc, K]
            go_sum = go_chunk.sum(dim=-1, keepdim=True)  # [B, Sc, 1]

            grad_input = softmax_output.neg()
            grad_input = grad_input.mul_(go_sum)

            # Positive scatter term: add gradients to selected indices
            # Mask grad_output for indices not on this shard
            go_masked = go_chunk * in_range.to(dtype=go_chunk.dtype)
            # Flatten for scatter_add
            flat_grad = grad_input.view(-1)
            # compute flattened indices positions
            Bc, Sc = go_masked.shape[0], go_masked.shape[1]
            # row offset per [B, Sc]
            row = (
                torch.arange(Bc, device=grad_input.device)
                .view(-1, 1)
                .expand(-1, Sc)
                .reshape(-1)
            )
            col = torch.arange(Sc, device=grad_input.device).expand(Bc, -1).reshape(-1)
            flat_idx_base = (row * Sc + col) * V_local  # [Bc*Sc]
            # selected flat indices
            flat_li = li.reshape(-1, li.shape[-1])  # [Bc*Sc, K]
            flat_base_expanded = flat_idx_base.unsqueeze(-1).expand_as(flat_li)
            flat_chosen = (flat_base_expanded + flat_li).reshape(-1)
            flat_go = go_masked.reshape(-1)
            flat_grad.scatter_add_(0, flat_chosen, flat_go)

            all_grad_input.append(grad_input)

        grad_input_total = (
            torch.cat(all_grad_input, dim=1)
            if len(all_grad_input) > 1
            else all_grad_input[0]
        )

        return grad_input_total, None, None, None, None, None, None


def dtensor_from_parallel_logits_to_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: DTensor | torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    seq_index: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP+CP sharded vocab logits.

    Args:
        vocab_parallel_logits (orch.Tensor): Logits distributed across tensor parallel workers,
            with shape [batch_size, seq_len, vocab_size/tp_size].
        target (DTensor): Target token indices with shape [batch_size, seq_len].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        tp_group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        seq_index (Optional[torch.Tensor]): Sequence index tensor with shape [seq_len].
            It is only provided for cp sharded logits. It represents how tensor is sharded across the sequence dimension.
        chunk_size (Optional[int]): Sequence dimension chunk size for computing the log probabilities.

    Returns:
        torch.Tensor: Log probabilities tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.
    """
    cp_size = 1

    if (
        isinstance(target, DTensor)
        and target.device_mesh.mesh_dim_names is not None
        and "cp" in target.device_mesh.mesh_dim_names
    ):
        cp_dim_index = target.device_mesh.mesh_dim_names.index("cp")
        cp_size = target.device_mesh.shape[cp_dim_index]

    if cp_size > 1:
        assert seq_index is not None, "seq_index must be provided for cp sharded logits"
        target_shape = torch.Size(target.shape)
        cp_mesh = target.device_mesh
        cp_placements = target.placements
        _, sorted_indices = torch.sort(seq_index)
        # Recover the original order of the target
        target = target.full_tensor()[:, sorted_indices]
        target = target.roll(shifts=-1, dims=-1)[:, seq_index]

        # Reshard
        target = distribute_tensor(target, cp_mesh, cp_placements)
        target = target.to_local()
    else:
        target = target.roll(shifts=-1, dims=-1)

    if chunk_size is not None:
        logprobs: torch.Tensor = ChunkedDistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            tp_group,
            inference_only,
        ).contiguous()
    else:
        logprobs: torch.Tensor = DistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            inference_only,
        ).contiguous()

    if cp_size > 1:
        # logprobs is sharded on the sequence dimension.
        # Get full sequence tensor, vocab dim has been reduced already.
        logprobs_dtensor = DTensor.from_local(logprobs, cp_mesh, cp_placements)
        logprobs = logprobs_dtensor.full_tensor()[:, sorted_indices]
        assert logprobs.shape == target_shape

    return logprobs[:, :-1]


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP+CP sharded vocab logits.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len // CP, vocab_size // TP]
            where TP is the tensor parallel size.
        target (torch.Tensor): Target token indices with shape [batch_size, seq_len].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        tp_group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.

    Returns:
        torch.Tensor: Log probabilities tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L354
    """
    target = target.roll(shifts=-1, dims=-1)
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    pad_len = 0
    # if cp_size > 1:
    # Pad the targets to local size * cp_size
    pad_len = vocab_parallel_logits.shape[1] * cp_size - target.shape[1]
    if pad_len > 0:
        target = torch.nn.functional.pad(target, (0, pad_len), value=0)

    # Shard the targets by context parallelism
    cp_rank = torch.distributed.get_rank(cp_group)
    target = _get_tokens_on_this_cp_rank(target, cp_rank, cp_size, seq_dim=1)

    if chunk_size is not None:
        logprobs: torch.Tensor = ChunkedDistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            tp_group,
            inference_only,
        ).contiguous()
    else:
        logprobs: torch.Tensor = DistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            inference_only,
        ).contiguous()

    if cp_size > 1:
        # we need to gather the logits by context parallelism
        logprobs = allgather_cp_sharded_tensor(
            logprobs, cp_group, seq_dim=1
        )  # , unpadded_seqlen=target.shape[1])

    if pad_len > 0:
        logprobs = logprobs[:, :-pad_len]

    return logprobs[:, :-1]


def from_parallel_logits_to_logprobs_packed_sequences(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpacked_seqlen: int,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP sharded vocab logits for packed sequences.

    Args:
        vocab_parallel_logits (torch.Tensor): Packed logits tensor with shape [1, T // CP, vocab_size//TP]
            where T is the total number of tokens across all packed sequences.
        target (torch.Tensor): Packed target token indices with shape [1, T].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths tensor with shape [batch_size + 1].
            cu_seqlens[i] indicates the start position of sequence i in the packed format.
        unpacked_seqlen (int): The length of the unpacked sequence tensor.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.

    Returns:
        torch.Tensor: Unpacked log probabilities tensor with shape [batch_size, unpacked_seqlen-1].
            The total length is reduced by batch_size due to target shifting (one token per sequence).
    """
    # Remove batch dimension to work with [T, vocab_size] and [T]
    vocab_parallel_logits = vocab_parallel_logits.squeeze(0)
    target = target.squeeze(0)

    batch_size = cu_seqlens_padded.shape[0] - 1
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else torch.distributed.get_rank(cp_group)

    # Roll each sequence individually
    rolled_targets = torch.zeros(
        target.shape[0] // cp_size, dtype=target.dtype, device=target.device
    )
    for i in range(batch_size):
        start_idx = cu_seqlens_padded[i].item()
        end_idx = cu_seqlens_padded[i + 1].item()

        # Get the sequence targets and roll by -1
        seq_targets = target[start_idx:end_idx]
        rolled_seq_targets = seq_targets.roll(shifts=-1, dims=0)
        rolled_targets[start_idx // cp_size : end_idx // cp_size] = (
            _get_tokens_on_this_cp_rank(rolled_seq_targets, cp_rank, cp_size, seq_dim=0)
        )

    # Add batch dimension back for DistributedLogprob
    rolled_targets = rolled_targets.unsqueeze(0)
    vocab_parallel_logits = vocab_parallel_logits.unsqueeze(0)

    # Apply distributed log probability computation
    if chunk_size is not None:
        probs: torch.Tensor = ChunkedDistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            rolled_targets,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            group,
            inference_only,
        ).contiguous()
    else:
        probs: torch.Tensor = DistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            rolled_targets,
            vocab_start_index,
            vocab_end_index,
            group,
            inference_only,
        ).contiguous()

    # Remove batch dimension for filtering
    probs = probs.squeeze(0)

    # Ensure probs is 1D after squeezing
    if probs.dim() != 1:
        raise ValueError(
            f"Expected probs to be 1D after squeezing, but got shape {probs.shape}. "
            f"Original shape before squeeze: {probs.unsqueeze(0).shape}"
        )

    if cp_size > 1:
        # per-sequence cp_allgather
        final_probs = torch.zeros(probs.shape[0] * cp_size, device=probs.device)
        for i in range(batch_size):
            start_idx = cu_seqlens_padded[i].item()
            end_idx = cu_seqlens_padded[i + 1].item()
            final_probs[start_idx:end_idx] = allgather_cp_sharded_tensor(
                probs[start_idx // cp_size : end_idx // cp_size], cp_group, seq_dim=0
            )
        probs = final_probs

    out_logprobs = torch.zeros(
        (batch_size, unpacked_seqlen - 1), dtype=probs.dtype, device=probs.device
    )
    # Filter out the last token of each sequence
    for i in range(batch_size):
        start_idx = cu_seqlens_padded[i].item()
        end_idx = cu_seqlens_padded[i + 1].item()

        # Exclude the last position (which has the rolled target from position 0)
        if end_idx - start_idx > 0:
            seq_probs = probs[start_idx : end_idx - 1]
            # Ensure seq_probs is 1D
            if seq_probs.dim() > 1:
                seq_probs = seq_probs.squeeze()

            # Ensure we don't exceed the unpacked sequence length
            seq_len = min(seq_probs.shape[0], unpacked_seqlen - 1)
            if seq_len > 0:
                out_logprobs[i, :seq_len] = seq_probs[:seq_len]

    return out_logprobs


def _get_tokens_on_this_cp_rank(
    input_ids: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Get tokens on this context parallelism rank.

    Assumes that input_ids are already padded to a multiple of cp_size * 2 or cp_size == 1.

    Args:
        input_ids: Input token IDs [seq_length, ]
        cp_rank: Context parallelism rank
        cp_size: Context parallelism size

    Returns:
        Tokens on this context parallelism rank [1, seq_length // cp_size]
    """
    if cp_size == 1:
        return input_ids

    # load balance for causal attention
    shard_size = input_ids.shape[seq_dim] // (cp_size * 2)
    shard_inds = (cp_rank, (cp_size * 2) - cp_rank - 1)

    # Create slices for each dimension
    slices = [slice(None)] * input_ids.dim()
    ids_chunks = []

    for ind in shard_inds:
        slices[seq_dim] = slice(ind * shard_size, (ind + 1) * shard_size)
        ids_chunks.append(input_ids[slices])

    ids = torch.cat(ids_chunks, dim=seq_dim)
    return ids


def allgather_cp_sharded_tensor(
    tensor, cp_group, seq_dim=1
):  # , unpadded_seqlen=None):
    return AllGatherCPTensor.apply(tensor, cp_group, seq_dim)  # , unpadded_seqlen)


class AllGatherCPTensor(torch.autograd.Function):
    def forward(
        ctx, tensor, cp_group: torch.distributed.ProcessGroup, seq_dim=1
    ):  # , unpadded_seqlen: Optional[int] = None):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank_chunks = []
        for _ in range(cp_size):
            cp_rank_chunks.append(torch.empty_like(tensor))

        torch.distributed.all_gather(
            tensor_list=cp_rank_chunks, tensor=tensor, group=cp_group
        )

        # undo the CP load balancing chunking
        tensor_chunks = []
        for logit_chunk in cp_rank_chunks:
            tensor_chunks.extend(torch.chunk(logit_chunk, chunks=2, dim=seq_dim))

        chunk_indices = []
        for cp_rank in range(cp_size):
            chunk_indices.append(cp_rank)
            chunk_indices.append(2 * cp_size - cp_rank - 1)

        chunks_and_indices = list(zip(tensor_chunks, chunk_indices))
        chunks_and_indices = sorted(chunks_and_indices, key=lambda tup: tup[1])
        ret_tensor = [chunk for chunk, _ in chunks_and_indices]
        ret_tensor = torch.cat(ret_tensor, dim=seq_dim)

        ctx.seq_dim = seq_dim
        ctx.cp_group = cp_group
        # ctx.unpadded_seqlen = unpadded_seqlen

        return ret_tensor

    def backward(ctx, grad_output):
        cp_size = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        torch.distributed.all_reduce(grad_output, group=ctx.cp_group)

        # chunk the seqdim in 2*cp chunks, and select with a CP load balanced indexing
        seq_dim = ctx.seq_dim
        # if ctx.unpadded_seqlen is not None:
        # # Zero out grad_output along the seq_dim after unpadded_seqlen
        # slicer = [slice(None)] * grad_output.dim()
        # slicer[seq_dim] = slice(ctx.unpadded_seqlen, None)
        #     grad_output[tuple(slicer)] = 0

        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1) :],
        )

        index = torch.tensor(
            [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
        ).cuda(non_blocking=True)

        grad_input = grad_output.index_select(seq_dim, index)
        grad_input = grad_input.view(
            *grad_input.shape[0:seq_dim], -1, *grad_input.shape[(seq_dim + 2) :]
        )

        return grad_input, None, None  # , None


def get_logprobs_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor,
    input_ids: torch.Tensor | DTensor,
    seq_index: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
):
    """Computes log probabilities from vocabulary-parallel logits.

    This function takes logits that are sharded across the vocabulary dimension (tensor parallel)
    and computes the log probabilities for the given input IDs.

    Args:
        vocab_parallel_logits (DTensor): Logits distributed across tensor parallel workers,
            with shape [batch_size, seq_len, vocab_size/tp_size].
        input_ids (torch.Tensor | DTensor): Input token IDs for which to compute log probabilities,
            with shape [batch_size, seq_len].
        seq_index (Optional[torch.Tensor]): Sequence index for the input IDs,
            with shape [sequence_length].
        chunk_size (Optional[int]): Sequence dimension chunk size for computing log probabilities.

    Returns:
        torch.Tensor: Log probabilities for the given input IDs.
    """
    device_mesh = vocab_parallel_logits.device_mesh
    if seq_index is not None:
        assert (
            device_mesh.mesh_dim_names is not None
            and "cp" in device_mesh.mesh_dim_names
        ), "seq_index must be provided for cp sharded logits"

    tp_size = 1

    tp_group = device_mesh.get_group("tp")
    tp_rank = tp_group.rank()
    tp_size = tp_group.size()

    vocab_interval_per_rank = vocab_parallel_logits.shape[-1] // tp_size

    return dtensor_from_parallel_logits_to_logprobs(
        vocab_parallel_logits.to_local(),
        input_ids,
        vocab_interval_per_rank * tp_rank,
        (tp_rank + 1) * vocab_interval_per_rank,
        tp_group,
        inference_only=not torch.is_grad_enabled(),
        seq_index=seq_index,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def distributed_vocab_topk(
    vocab_parallel_logits: torch.Tensor,
    k: int,
    tp_group: torch.distributed.ProcessGroup,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global top-k over TP-sharded vocabulary logits.

    Args:
        vocab_parallel_logits: [B, S, V_local]
        k: number of top tokens to select globally
        tp_group: tensor-parallel process group
        vocab_start_index: global vocab start for this rank (inclusive)
        vocab_end_index: global vocab end for this rank (exclusive)
        chunk_size: optional chunk along sequence dim to bound memory

    Returns:
        topk_vals: [B, S, k]
        topk_global_indices: [B, S, k] (global token ids)
    """
    assert vocab_end_index > vocab_start_index
    world_size = torch.distributed.get_world_size(tp_group)

    logits = vocab_parallel_logits.to(dtype=torch.float32)
    B, S, V_local = logits.shape
    V_total = V_local * world_size
    K_eff = int(min(k, max(1, V_total)))

    if chunk_size is None:
        chunk_size = S

    vals_chunks: list[torch.Tensor] = []
    idx_chunks: list[torch.Tensor] = []

    for s0 in range(0, S, chunk_size):
        s1 = min(S, s0 + chunk_size)
        # local top-k on this TP rank
        local_vals, local_idx_local = torch.topk(
            logits[:, s0:s1, :], min(k, V_local), dim=-1
        )
        local_idx_global = local_idx_local + int(vocab_start_index)

        # gather candidates from all TP ranks
        gathered_vals = [torch.empty_like(local_vals) for _ in range(world_size)]
        gathered_idx = [torch.empty_like(local_idx_global) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_vals, local_vals, group=tp_group)
        torch.distributed.all_gather(gathered_idx, local_idx_global, group=tp_group)

        all_vals = torch.cat(gathered_vals, dim=-1)
        all_idx = torch.cat(gathered_idx, dim=-1)

        sel_vals, sel_pos = torch.topk(all_vals, K_eff, dim=-1)
        sel_idx = torch.gather(all_idx, dim=-1, index=sel_pos)

        vals_chunks.append(sel_vals)
        idx_chunks.append(sel_idx)

    topk_vals = (
        torch.cat(vals_chunks, dim=1) if len(vals_chunks) > 1 else vals_chunks[0]
    )
    topk_global_indices = (
        torch.cat(idx_chunks, dim=1) if len(idx_chunks) > 1 else idx_chunks[0]
    )

    return topk_vals, topk_global_indices


def gather_logits_at_global_indices(
    vocab_parallel_logits: torch.Tensor,
    global_indices: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Gather student logits at given global token indices under TP+CP sharding.

    Differentiable w.r.t. vocab_parallel_logits.

    Args:
        vocab_parallel_logits: [B, S_cp, V_local] where S_cp is CP sharded sequence length
        global_indices: [B, S_full, k] where S_full is full sequence length
        tp_group: Optional tensor-parallel process group. If None, treats logits as full-vocab (no TP) and skips TP all-reduce.
        vocab_start_index: global vocab start for this rank (inclusive)
        vocab_end_index: global vocab end for this rank (exclusive)
        chunk_size: optional chunk along sequence dim to bound memory
        cp_group: Optional context-parallel process group

    Returns:
        gathered_logits: [B, S_full, k]
    """
    # CP support: get CP group and size
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)

    # Handle CP sharding of global_indices (similar to from_parallel_logits_to_logprobs)
    pad_len = 0
    if cp_size > 1:
        # Pad the global_indices to local size * cp_size if needed
        pad_len = vocab_parallel_logits.shape[1] * cp_size - global_indices.shape[1]
        if pad_len > 0:
            global_indices = torch.nn.functional.pad(
                global_indices, (0, 0, 0, pad_len), value=0
            )

        # Shard the global_indices by context parallelism
        cp_rank = torch.distributed.get_rank(cp_group)
        global_indices = _get_tokens_on_this_cp_rank(
            global_indices, cp_rank, cp_size, seq_dim=1
        )

    logits = vocab_parallel_logits.to(dtype=torch.float32)
    B, S, V_local = logits.shape
    if chunk_size is None:
        chunk_size = S

    out_chunks: list[torch.Tensor] = []
    for s0 in range(0, S, chunk_size):
        s1 = min(S, s0 + chunk_size)
        gi = global_indices[:, s0:s1, :]

        in_range = (gi >= int(vocab_start_index)) & (gi < int(vocab_end_index))
        # Map global ids to local shard ids and clamp to valid range to avoid OOB gather
        V_local = logits.shape[-1]
        li = (gi - int(vocab_start_index)).clamp(min=0, max=V_local - 1)

        local_vals = torch.gather(logits[:, s0:s1, :], dim=-1, index=li)
        local_vals = local_vals * in_range.to(dtype=local_vals.dtype)

        if tp_group is not None:
            torch.distributed.all_reduce(
                local_vals, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
        out_chunks.append(local_vals)

    gathered_logits = (
        torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]
    )

    # CP gather: gather the logits by context parallelism
    if cp_size > 1:
        gathered_logits = allgather_cp_sharded_tensor(
            gathered_logits, cp_group, seq_dim=1
        )

        # Remove padding if we added it earlier
        if pad_len > 0:
            gathered_logits = gathered_logits[:, :-pad_len, :]

    return gathered_logits


class ChunkedDistributedEntropy(torch.autograd.Function):
    """Compute H_all = sum_v p_v log p_v across TP with chunking over sequence.

    Forward returns [B, S] tensor of global entropy; backward propagates through logits.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,  # [B, S, V_local]
        chunk_size: int,
        tp_group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        B, S, _ = vocab_parallel_logits.shape
        num_chunks = (int(S) + chunk_size - 1) // chunk_size
        out_chunks: list[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            s0 = chunk_idx * chunk_size
            s1 = min(int(S), (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, s0:s1, :].to(dtype=torch.float32)
            log_probs = _compute_distributed_log_softmax(logits, group=tp_group)
            softmax_output = log_probs.exp()
            H_local = (softmax_output * log_probs).sum(dim=-1)  # [B, Sc]
            torch.distributed.all_reduce(
                H_local, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
            out_chunks.append(H_local)

        H_all = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]

        if not inference_only:
            ctx.save_for_backward(vocab_parallel_logits)
            ctx.chunk_size = int(chunk_size)
            ctx.tp_group = tp_group

        return H_all.contiguous()

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        grad_output = grad_outputs[0]  # [B, S]
        (vocab_parallel_logits,) = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        tp_group = ctx.tp_group

        B, S, V_local = vocab_parallel_logits.shape
        num_chunks = (int(S) + chunk_size - 1) // chunk_size
        grads: list[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            s0 = chunk_idx * chunk_size
            s1 = min(int(S), (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, s0:s1, :].to(dtype=torch.float32)
            log_probs = _compute_distributed_log_softmax(logits, group=tp_group)
            softmax_output = log_probs.exp()
            H_local = (softmax_output * log_probs).sum(dim=-1)
            torch.distributed.all_reduce(
                H_local, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )

            # dH/dz = softmax * (log_probs - H_all)
            grad_chunk = softmax_output * (log_probs - H_local.unsqueeze(-1))
            grad_chunk.mul_(grad_output[:, s0:s1].unsqueeze(-1))
            grads.append(grad_chunk)

        grad_input = torch.cat(grads, dim=1) if len(grads) > 1 else grads[0]
        return grad_input, None, None, None


def from_parallel_hidden_states_to_logprobs(
    tensor_parallel_hidden_states: torch.Tensor, # shape: [batch_size, seq_len // CP, hidden_size], last dimension is hidden_size and not sharded
    output_weight_layer: torch.Tensor,
    output_weight: torch.Tensor,
    runtime_gather_output: bool,
    target: torch.Tensor,
    vocab_start_index: int, # vocab start and end index are needed since the model.lm_head is shared across the vocabulary
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP sharded hidden states.
    """
    # TODO: implement this
    target = target.roll(shifts=-1, dims=-1)

    # cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    # print(f"cp size is {cp_size}, cp group is {cp_group}")
    # pad_len = 0
    # # if cp_size > 1:
    # # Pad the targets to local size * cp_size
    # # the sequence length is the first dimension so we need to pad the first dimension
    # pad_len = tensor_parallel_hidden_states.shape[0] * cp_size - target.shape[1]
    # if pad_len > 0:
    #     target = torch.nn.functional.pad(target, (0, pad_len), value=0)

    # Shard the targets by context parallelism
    # cp_rank = torch.distributed.get_rank(cp_group)
    # target = _get_tokens_on_this_cp_rank(target, cp_rank, cp_size, seq_dim=1)
    # print(f"target shape after sharding: {target.shape} and pad len is {pad_len}")
    # assert chunk_size is not None, "chunk_size is required for hidden states"
    # TODO: implement this
    # print(f"right before calling ChunkedDistributedHiddenStatesToLogprobs, target: {target[:2, :20]} of tp rank {get_tensor_model_parallel_rank()}")
    logprobs: torch.Tensor = ChunkedDistributedHiddenStatesToLogprobs.apply(  # type: ignore
        tensor_parallel_hidden_states,
        target,
        output_weight_layer,
        vocab_start_index,
        vocab_end_index,
        chunk_size,
        tp_group,
        inference_only,
    ).contiguous()

    # if cp_size > 1:
    #     # we need to gather the logits by context parallelism
    #     logprobs = allgather_cp_sharded_tensor(
    #         logprobs, cp_group, seq_dim=1
    #     )  # , unpadded_seqlen=target.shape[1])

    # if pad_len > 0:
    #     logprobs = logprobs[:, :-pad_len]
    # print(f"inspect logprobs inside from_parallel_hidden_states_to_logprobs: {logprobs[:2, :20]} and target: {target[:2, :20]} and pad len is {pad_len} of tp rank {get_tensor_model_parallel_rank()}")
    return logprobs[:, :-1]


class ChunkedDistributedHiddenStatesToLogprobs(torch.autograd.Function):
    """Compute distributed log-softmax once and gather logprobs at given global indices.
    """

    @staticmethod
    def forward(ctx: Any, tensor_parallel_hidden_states: torch.Tensor, target: torch.Tensor, output_weight_layer: torch.Tensor, vocab_start_index: int, vocab_end_index: int, chunk_size: int, tp_group: torch.distributed.ProcessGroup, inference_only: bool = False) -> torch.Tensor:
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0
        # print(f"masked_target shape: {masked_target.shape}, target_mask shape: {target_mask.shape}, tensor_parallel_hidden_states shape: {tensor_parallel_hidden_states.shape}")
        tp_group_size = torch.distributed.get_world_size(tp_group)
        if tp_group_size > 1:
            original_tensor_parallel_hidden_states = tensor_parallel_hidden_states.clone()
            all_hidden_states = [torch.zeros_like(tensor_parallel_hidden_states) for _ in range(tp_group_size)]
            torch.distributed.all_gather(all_hidden_states, tensor_parallel_hidden_states, group=tp_group)
            tensor_parallel_hidden_states = torch.cat(all_hidden_states, dim=0)
        else:
            original_tensor_parallel_hidden_states = tensor_parallel_hidden_states
        # print(f"in exp: global rank {torch.distributed.get_rank()}, tp rank {get_tensor_model_parallel_rank()}, tensor_parallel_hidden_states shape: {tensor_parallel_hidden_states.shape}, value: {tensor_parallel_hidden_states[:2,:, :20]} ")
        # print(f"tp rank {get_tensor_model_parallel_rank()} all_hidden_states shape: {tensor_parallel_hidden_states.shape}, value: {tensor_parallel_hidden_states[:2, :20]} ")
        seq_size = int(tensor_parallel_hidden_states.shape[0])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size
        all_log_probs = []
        tp_rank = get_tensor_model_parallel_rank()
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)
            # this will produce a tensor of shape [batch_size, chunk_size * TP_size, vocab_size // TP_size]
            #logits, _ = output_layer(tensor_parallel_hidden_states[chunk_start:chunk_end, :, :], weight=output_weight, runtime_gather_output=runtime_gather_output)
            logits = torch.matmul(tensor_parallel_hidden_states[chunk_start:chunk_end, :, :], output_weight_layer.T)
            logits = logits.to(dtype=torch.float32).transpose(0, 1).contiguous() # [tp_rank * real_chunk_size: (tp_rank + 1) * real_chunk_size, :, :]
            # print(f"logits shape: {logits.shape} at chunk {chunk_idx} after matmul using the weight layer directly")
            if chunk_idx == 0:
                print(f"rank {tp_rank}, logits shape: {logits.shape}, logits: {logits[:2, :20, :10]} ")
            log_probs = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )
            # print(f"log_probs shape: {log_probs.shape} at chunk {chunk_idx} after compute distributed log softmax")
            # gather the log probabilities for the chosen tokens so the vocab size dimension is gone
            # used_masked_target = [torch.zeros_like(masked_target[:, chunk_start:chunk_end])] * tp_rank + [masked_target[:, chunk_start:chunk_end]] + [torch.zeros_like(masked_target[:, chunk_start:chunk_end])] * (tp_group_size - tp_rank - 1)
            # used_target_mask = [torch.ones_like(target_mask[:, chunk_start:chunk_end])] * tp_rank + [target_mask[:, chunk_start:chunk_end]] + [torch.ones_like(target_mask[:, chunk_start:chunk_end])] * (tp_group_size - tp_rank - 1)
            # used_masked_target = torch.cat(used_masked_target, dim=1)
            # used_target_mask = torch.cat(used_target_mask, dim=1)
            log_probs = torch.gather(
                log_probs, -1, masked_target[:, chunk_start:chunk_end].unsqueeze(-1)
            ).squeeze(-1).detach()
            # print(f"log_probs shape: {log_probs.shape} at chunk {chunk_idx} after gather")
            log_probs[target_mask[:, chunk_start:chunk_end]] = 0.0
            # print(f"log_probs shape: {log_probs.shape} at chunk {chunk_idx} before all reduce, masked_target size: {masked_target[:, chunk_start:chunk_end].shape}")
            # TODO: the following lines are buggy since the log probs are sharded by TP rank on the seq dimension and they should not be all reduced.
            # Need to think about how to handle this.
            all_log_probs.append(log_probs)

        log_probs = torch.cat(all_log_probs, dim=1)
        torch.distributed.all_reduce(
            log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
        # print(f"log_probs shape: {log_probs.shape} after all reduce")
        # print(f"log_probs shape: {log_probs.shape} after cat at rank {tp_rank}")
        # final_log_probs = [torch.zeros_like(log_probs)] * tp_group_size
        # torch.distributed.all_gather(final_log_probs, log_probs, group=tp_group)
        # final_log_probs = torch.cat(final_log_probs, dim=1)
        # print(f"final_log_probs shape: {final_log_probs.shape} after all gather at rank {tp_rank}")
        if not inference_only:
            # only save for backward when we have inference only=False
            # save tensor_parallel_hidden_states and the output_layer to the context
            ctx.save_for_backward(original_tensor_parallel_hidden_states.detach(), target_mask.detach(), masked_target.detach(), output_weight_layer.detach())
            # ctx.output_weight_layer = output_weight_layer
            ctx.chunk_size = chunk_size
            ctx.tp_group = tp_group
            # ctx.output_weight = output_weight
            #ctx.runtime_gather_output = runtime_gather_output
        print(f"log_probs shape: {log_probs.shape} after forward, seq_size: {seq_size}, chunk_size: {chunk_size}, num_chunks: {num_chunks}")
        return log_probs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor,  None, None, None, None, None]:
        grad_output = grad_outputs[0]
        # the tensor_parallel_hidden_states is already all gathered in the forward pass
        tensor_parallel_hidden_states, target_mask, masked_target, output_weight_layer = ctx.saved_tensors
        tp_group = ctx.tp_group
        tp_group_size = torch.distributed.get_world_size(tp_group)
        if tp_group_size > 1:
            all_hidden_states = [torch.zeros_like(tensor_parallel_hidden_states) for _ in range(tp_group_size)]
            torch.distributed.all_gather(all_hidden_states, tensor_parallel_hidden_states, group=tp_group)
            tensor_parallel_hidden_states = torch.cat(all_hidden_states, dim=0)
        # output_weight_layer = ctx.output_weight_layer
        chunk_size = ctx.chunk_size
        tp_group = ctx.tp_group
        #output_weight = ctx.output_weight
        #runtime_gather_output = ctx.runtime_gather_output
        # this is the vocab size for this partition when the output_layer is a ColumnParallelLinear
        partition_vocab_size = output_weight_layer.size(0) # int(output_layer.output_size) // tp_group_size
        seq_size = int(tensor_parallel_hidden_states.shape[0])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size
        tp_rank = get_tensor_model_parallel_rank()
        print(f"num_chunks: {num_chunks}, chunk_size: {chunk_size}, seq_size: {seq_size}, target_mask shape: {target_mask.shape}, masked_target shape: {masked_target.shape} at rank {tp_rank}, grad_output shape: {grad_output.shape}")
        all_grad_input_hidden_states = []
        all_grad_input_output_layer = []
        grad_input_output_layer = torch.zeros_like(output_weight_layer)
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)
            # recalculate the logits using the output_layer
            # logits, _ = output_layer(tensor_parallel_hidden_states[chunk_start:chunk_end, :, :], weight=output_weight, runtime_gather_output=runtime_gather_output)
            logits = torch.matmul(tensor_parallel_hidden_states[chunk_start:chunk_end, :, :], output_weight_layer.T)
            logits = logits.to(dtype=torch.float32).transpose(0, 1).contiguous()# [tp_rank * real_chunk_size: (tp_rank + 1) * real_chunk_size, :, :]
            # print(f"logits shape: {logits.shape} at chunk {chunk_idx} after matmul using the weight layer directly before compute distributed log softmax)
            softmax_output = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )
            softmax_output = softmax_output.exp().detach()
            # print(f"softmax_output shape: {softmax_output.shape} at chunk {chunk_idx} after compute distributed log softmax")
            # 1 if it's the chosen log prob, 0 otherwise
            # used_masked_target = [torch.zeros_like(masked_target[:, chunk_start:chunk_end])] * tp_rank + [masked_target[:, chunk_start:chunk_end]] + [torch.zeros_like(masked_target[:, chunk_start:chunk_end])] * (tp_group_size - tp_rank - 1)
            # used_target_mask = [torch.ones_like(target_mask[:, chunk_start:chunk_end])] * tp_rank + [target_mask[:, chunk_start:chunk_end]] + [torch.ones_like(target_mask[:, chunk_start:chunk_end])] * (tp_group_size - tp_rank - 1)
            # used_masked_target = torch.cat(used_masked_target, dim=1)
            # used_target_mask = torch.cat(used_target_mask, dim=1)
            # print(f"log_probs shape: {log_probs.shape} at chunk {chunk_idx} after gather")
            # is_chosen = (~(target_mask[:, chunk_start:chunk_end])).unsqueeze(
            #     -1
            # ) * torch.nn.functional.one_hot(
            #     masked_target[:, chunk_start:chunk_end],
            #     num_classes=partition_vocab_size,
            # )
            is_chosen = (~(target_mask[:, chunk_start:chunk_end])).unsqueeze(
                -1
            ) * torch.nn.functional.one_hot(
                masked_target[:, chunk_start:chunk_end],
                num_classes=partition_vocab_size,
            )
            # print(f"is_chosen shape: {is_chosen.shape} at chunk {chunk_idx} after one hot")
            grad_input = is_chosen.float().sub_(softmax_output)
            # print(f"grad_input shape: {grad_input.shape} at chunk {chunk_idx} at rank {tp_rank}")
            # used_grad_outputs = [grad_output[:, tp_rank_local * seq_size + chunk_start:tp_rank_local * seq_size + chunk_end] for tp_rank_local in range(tp_group_size)]
            # used_grad_outputs = torch.cat(used_grad_outputs, dim=1)
            used_grad_output = grad_output[:, chunk_start: chunk_end]
            # print(f"grad_output shape: {grad_output.shape}, used_grad_output shape: {used_grad_output.shape} at chunk {chunk_idx} at rank {tp_rank}")
            grad_input.mul_(used_grad_output.unsqueeze(dim=-1))
            # be careful with shape of the matrices
            # print(f"grad_input shape: {grad_input.shape} at chunk {chunk_idx}")
            grad_input_hidden_states = torch.matmul(grad_input, output_weight_layer.to(dtype=torch.float32))# [chunk_start:chunk_end, :, :]
            # grad_input_hidden_states = grad_input_hidden_states_matmul # [:, real_chunk_size * tp_rank: real_chunk_size * (tp_rank + 1):]
            # print(f"grad_input_hidden_states shape: {grad_input_hidden_states.shape}, grad_input_hidden_states_matmul shape: {grad_input_hidden_states_matmul.shape}, grad_input shape: {grad_input.shape}, output_layer.weight shape: {output_layer.weight.shape}")
            # grad_input_output_layer = torch.matmul(tensor_parallel_hidden_states[:, chunk_start:chunk_end, :], grad_input)
            grad_input_output_layer_local = torch.einsum('bsd, bsv -> dv', tensor_parallel_hidden_states[chunk_start:chunk_end, :, :].transpose(0, 1).contiguous().to(dtype=torch.float32), grad_input.to(dtype=torch.float32))
            all_grad_input_hidden_states.append(grad_input_hidden_states)
            grad_input_output_layer.add_(grad_input_output_layer_local.transpose(0, 1).contiguous())
            #all_grad_input_output_layer.append(grad_input_output_layer[None, ...])
            # print(f"grad_input_output_layer shape: {grad_input_output_layer.shape} at chunk {chunk_idx}")

        grad_input_hidden_states = torch.cat(all_grad_input_hidden_states, dim=1).transpose(0, 1).contiguous()
        # print(f"grad_input_hidden_states shape: {grad_input_hidden_states.shape} after cat at rank {tp_rank}")
        # need to reduce the gradient of the hidden states across the tensor parallel group
        #grad_input_hidden_states = torch.distributed.all_reduce(grad_input_hidden_states, group=tp_group)
        # the gradient of the output layer is the mean of the gradients from all the chunks

        # grad_input_output_layer = torch.cat(all_grad_input_output_layer).mean(dim=0)
        # print(f"grad_input_output_layer shape: {grad_input_output_layer.shape} after cat and mean")
        # output_weight_layer.grad = grad_input_output_layer.transpose(0, 1).contiguous()
        weight_grad = grad_input_output_layer
        local_seq_size = seq_size // tp_group_size
        # grad_input_hidden_states = grad_input_hidden_states[local_seq_size * tp_rank:local_seq_size * (tp_rank + 1), :,  :]
        sharded_grad_hidden_states = torch.empty_like(grad_input_hidden_states[:local_seq_size])
        grad_input_hidden_states_list = list(torch.chunk(grad_input_hidden_states, chunks=tp_group_size, dim=0))
        torch.distributed.reduce_scatter(
            sharded_grad_hidden_states,
            grad_input_hidden_states_list,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group
        )
        # grad_input_hidden_states = grad_input_hidden_states[:, local_seq_size * tp_rank:local_seq_size * (tp_rank + 1),  :]
        # TODO: return the gradient of the model parameters
        print(f"grad_input_hidden_states shape: {grad_input_hidden_states.shape} after all reduce at rank {tp_rank}, weight_grad shape: {weight_grad.shape}")
        return sharded_grad_hidden_states, None, weight_grad, None, None, None, None, None


def patch_gpt_model_forward_for_linear_ce_fusion(*, chunk_size: int = 256) -> None:
    if getattr(GPTModel, "_linear_ce_fusion_forward_patched", False):
        GPTModel._linear_ce_fusion_chunk_size = chunk_size
        return
    GPTModel._original_forward_for_linear_ce_fusion = GPTModel.forward
    GPTModel._linear_ce_fusion_chunk_size = chunk_size
    GPTModel.forward = _gpt_forward_with_linear_ce_fusion
    GPTModel._linear_ce_fusion_forward_patched = True



def _gpt_forward_with_linear_ce_fusion(
    self: GPTModel,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    inference_context: Any = None,
    packed_seq_params: Any = None,
    extra_block_kwargs: Optional[dict] = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[Any] = None,
    loss_mask: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    return_logprobs_for_linear_ce_fusion: bool = False,
) -> torch.Tensor:
    if not return_logprobs_for_linear_ce_fusion:
        return self._original_forward_for_linear_ce_fusion(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
        )
    """
    original forward function signature:
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
    """
    if labels is None:
        raise ValueError("labels must be provided when linear CE fusion is enabled")

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preproc_output = self._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
    )
    (
        decoder_input,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        sequence_len_offset,
        padding_mask,
    ) = preproc_output[:6]
    rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None

    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        rotary_pos_cos_sin=rotary_pos_cos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        padding_mask=padding_mask,
        **(extra_block_kwargs or {}),
    )

    # Non post-process pipeline stages do not own the output layer.
    if not self.post_process or not hasattr(self, "output_layer"):
        return hidden_states

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_pg_size(get_tensor_model_parallel_group())
    print(
        f"hidden_states shape: {hidden_states.shape}, tp_rank: {tp_rank}, tp_size: {tp_size}, cp size {get_pg_size(self.cp_group)}"
    )
    # calculate the logprobs for the last token and then return the logprobs
    vocab_start_index = tp_rank * (self.vocab_size // tp_size)
    vocab_end_index = min((tp_rank + 1) * (self.vocab_size // tp_size), self.vocab_size)
    output_weight_layer = self.output_layer.weight
    logprobs = from_parallel_hidden_states_to_logprobs(
        hidden_states, #.transpose(0, 1).contiguous(),
        output_weight_layer,
        self.shared_embedding_or_output_weight()
        if self.share_embeddings_and_output_weights
        else self.output_layer.weight,
        runtime_gather_output,
        labels,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        inference_only=inference_context is not None and not self.training,
        tp_group=get_tensor_model_parallel_group(),
        cp_group=self.cp_group,
        chunk_size=getattr(self, "_linear_ce_fusion_chunk_size", 256),
    )
    print(f"logprobs shape: {logprobs.shape}")
    return logprobs