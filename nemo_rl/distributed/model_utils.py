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

from typing import Any, Tuple

import torch


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
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ):
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0
        vocab_parallel_logits = vocab_parallel_logits.to(torch.float32)

        log_softmax_output = _compute_distributed_log_softmax(
            vocab_parallel_logits, group=group
        )
        log_probs = log_softmax_output.clone()
        softmax_output = log_softmax_output.exp_()

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
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None, None, None, None]:
        softmax, target_mask, masked_target = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        # 1 if it's the chosen log prob, 0 otherwise
        is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(
            masked_target, num_classes=partition_vocab_size
        )

        grad_input = is_chosen.float().sub_(softmax)

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


class DistributedTokenLevelEntropy(torch.autograd.Function):
    """Custom autograd function for computing entropy in a distributed setting."""

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ):
        # TODO: keep the vocab start and end index
        log_probs = _compute_distributed_log_softmax(vocab_parallel_logits, group=group)
        tp_entropy = (log_probs.exp() * log_probs).sum(-1)

        torch.distributed.all_reduce(
            tp_entropy,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        return -tp_entropy

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None, None, None, None]:
        raise NotImplementedError("Backward pass for entropy is not implemented.")


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
) -> torch.Tensor:
    """Get log probabilities from TP sharded vocab logits.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len, vocab_size//TP]
            where TP is the tensor parallel size.
        target (torch.Tensor): Target token indices with shape [batch_size, seq_len].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.

    Returns:
        torch.Tensor: Log probabilities tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L354
    """
    target = target.roll(shifts=-1, dims=-1)
    probs = DistributedLogprob.apply(
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        group,
        inference_only,
    ).contiguous()
    return probs[:, :-1]


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

            log_softmax_output = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )
            log_probs = log_softmax_output.clone()

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

            log_softmax_output = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )
            log_probs = log_softmax_output.clone()
            softmax_output = log_softmax_output.exp_()

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
