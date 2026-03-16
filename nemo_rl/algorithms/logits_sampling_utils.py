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

from dataclasses import dataclass
from typing import Optional

import torch

# Default chunk size for top-k/top-p filtering.
# The sort operation in top-p filtering is memory intensive because it creates
# intermediate tensors of shape [bsz, seq_len, vocab_size] for both sorted values
# and indices. For large vocab sizes (e.g., 152K) and long sequences (e.g., 32K),
# this can cause OOM. Chunking along the sequence dimension reduces peak memory.
# Different chunk sizes have minor performance differences.
TOP_K_TOP_P_CHUNK_SIZE: int = 256


@dataclass
class TrainingSamplingParams:
    """Training-specific sampling parameters to match generation parameters.

    Used to ensure consistency between training and inference by applying the same sampling strategy during
    logprob computation. Not directly using vLLM's SamplingParams class to avoid dependency on vLLM in this env.

    Attributes:
        top_k: Top-k filtering parameter (None or -1 to disable)
        top_p: Top-p filtering parameter (1.0 to disable)
        temperature: Temperature for scaling logits (default: 1.0)
    """

    top_k: int | None = None
    top_p: float = 1.0
    temperature: float = 1.0


def _need_top_k_filtering(top_k: int | None) -> bool:
    """Check if top-k filtering is needed."""
    return top_k is not None and top_k > 0


def _need_top_p_filtering(top_p: float | None) -> bool:
    """Check if top-p filtering is needed."""
    return top_p is not None and top_p != 1.0


def need_top_k_or_top_p_filtering(
    sampling_params: Optional[TrainingSamplingParams],
) -> bool:
    """Check if top-k or top-p filtering is needed."""
    if sampling_params is None:
        return False

    top_k = sampling_params.top_k
    top_p = sampling_params.top_p
    return _need_top_k_filtering(top_k) or _need_top_p_filtering(top_p)


@torch.no_grad()
def _apply_top_k_only_fn(
    logits: torch.Tensor,
    top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply top-k mask to the logits.

    Simplified version of VLLM's implementation for scalar parameters.
    This implementation doesn't involve sorting the entire vocab.

    Based on VLLM's implementation:
    https://github.com/vllm-project/vllm/blob/34a20c49b3f81f64133428b3a0d62309db1256f9/vllm/v1/sample/ops/topk_topp_sampler.py
    SPDX-License-Identifier: Apache-2.0
    Copyright contributors to the vLLM project

    Args:
        logits: Input logits tensor of shape [*, vocab_size].
        top_k: Top-k sampling parameter.

    Returns:
        filtered_logits: Filtered logits tensor with the same shape as input logits.
        keep_mask: Mask tensor with the same shape as input logits, where 1 (True) indicates tokens to be
            kept, 0 (False) indicates tokens to be masked. None if top-k filtering is not needed.
    """
    if not _need_top_k_filtering(top_k):
        return logits, None

    # Get top-k values and create mask
    assert top_k is not None  # Type narrowing
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = top_k_values[..., -1:].expand_as(logits)
    keep_mask = logits >= threshold

    # Apply mask: keep top-k values, set others to -inf
    logits = torch.where(
        keep_mask,
        logits,
        torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype),
    )
    return logits, keep_mask


@torch.no_grad()
def _apply_top_k_top_p_fn(
    logits: torch.Tensor,
    top_k: int | None,
    top_p: float,
    chunk_size: int | None = TOP_K_TOP_P_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply top-k and top-p masks to the logits with chunking for memory efficiency.

    The sort operation in top-p filtering is memory intensive because it creates
    intermediate tensors of shape [num_tokens, vocab_size] for both sorted values
    and indices. For large vocab sizes (e.g., 152K) and many tokens, this can cause OOM.
    This function flattens the input to 2D and processes in chunks along the token
    dimension (controlled by chunk_size) to reduce peak memory.

    Based on VLLM's implementation:
    https://github.com/vllm-project/vllm/blob/34a20c49b3f81f64133428b3a0d62309db1256f9/vllm/v1/sample/ops/topk_topp_sampler.py
    SPDX-License-Identifier: Apache-2.0
    Copyright contributors to the vLLM project

    Args:
        logits: Input logits tensor of shape [*, vocab_size] (e.g., [batch_size, seq_len, vocab_size]
            or [batch_size, vocab_size]). Internally flattened to [num_tokens, vocab_size] for processing.
        top_k: Top-k sampling parameter. Set to -1 or None to consider all tokens.
        top_p: Top-p (nucleus) sampling parameter. Must be in (0, 1]. Set to 1 to consider all tokens
        chunk_size: Number of tokens to process per chunk for memory efficiency. Defaults to TOP_K_TOP_P_CHUNK_SIZE.

    Returns:
        filtered_logits: Filtered logits tensor with the same shape as input logits.
        keep_mask: Mask tensor with the same shape as input logits, where 1 (True) indicates
            tokens to be kept, 0 (False) indicates tokens to be masked.
    """
    if not _need_top_p_filtering(top_p):
        if not _need_top_k_filtering(top_k):
            return logits, None
        # Avoid sorting vocab for top-k only case
        filtered_logits, top_k_keep_mask = _apply_top_k_only_fn(logits, top_k)
        return filtered_logits, top_k_keep_mask

    # Save original shape and flatten to 2D for consistent chunking
    original_shape = logits.shape
    vocab_size = logits.shape[-1]
    logits = logits.view(-1, vocab_size)  # [*, vocab_size] -> [num_tokens, vocab_size]
    num_tokens = logits.shape[0]

    chunk_size = chunk_size if chunk_size is not None else num_tokens

    # Pre-allocate output tensors
    filtered_logits = torch.empty_like(logits)
    keep_mask = torch.empty(
        num_tokens, vocab_size, dtype=torch.bool, device=logits.device
    )

    for start_idx in range(0, num_tokens, chunk_size):
        end_idx = min(start_idx + chunk_size, num_tokens)
        chunk_logits = logits[start_idx:end_idx, :]

        # Sort this chunk
        logits_sort, logits_idx = chunk_logits.sort(dim=-1, descending=False)
        top_k_keep_mask_chunk = None

        if _need_top_k_filtering(top_k):
            assert top_k is not None  # Type narrowing
            # Apply top-k first
            top_k_index = logits_sort.size(-1) - top_k
            index_tensor = torch.full(
                logits_sort.shape[:-1],
                top_k_index,
                device=logits_sort.device,
                dtype=torch.long,
            )
            top_k_threshold = logits_sort.gather(-1, index_tensor.unsqueeze(-1))
            top_k_keep_mask_chunk = logits_sort >= top_k_threshold
            logits_sort.masked_fill_(~top_k_keep_mask_chunk, -float("inf"))

        # Apply top-p
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_keep_mask_chunk = probs_sum > 1 - top_p
        # at least one
        top_p_keep_mask_chunk[..., -1] = True
        logits_sort.masked_fill_(~top_p_keep_mask_chunk, -float("inf"))

        # Scatter back to original order
        chunk_filtered = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
        if top_k_keep_mask_chunk is not None:
            chunk_mask = torch.logical_and(top_k_keep_mask_chunk, top_p_keep_mask_chunk)
        else:
            chunk_mask = top_p_keep_mask_chunk
        chunk_mask = chunk_mask.scatter(dim=-1, index=logits_idx, src=chunk_mask)

        # Store results
        filtered_logits[start_idx:end_idx, :] = chunk_filtered
        keep_mask[start_idx:end_idx, :] = chunk_mask

    # Restore original shape
    filtered_logits = filtered_logits.view(original_shape)
    keep_mask = keep_mask.view(original_shape)

    return filtered_logits, keep_mask


class _ApplyTopKTopP(torch.autograd.Function):
    """Autograd function for top-k and top-p filtering with proper gradient handling."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx,
        logits: torch.Tensor,
        top_k: Optional[int],
        top_p: float,
        chunk_size: int | None = TOP_K_TOP_P_CHUNK_SIZE,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply top-k/top-p filtering and save masks for backward.

        Args:
            logits: Input logits tensor of shape [*, vocab_size].
            top_k: Top-k sampling parameter. Set to -1 or None to consider all tokens.
            top_p: Top-p sampling parameter. Must be in (0, 1]. Set to 1 to consider all tokens.
            chunk_size: Number of tokens to process per chunk. Defaults to TOP_K_TOP_P_CHUNK_SIZE.
        """
        filtered_logits, keep_mask = _apply_top_k_top_p_fn(
            logits, top_k, top_p, chunk_size
        )

        # Save masks for backward pass
        ctx.save_for_backward(keep_mask)

        return filtered_logits, keep_mask

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        """Backward pass: mask out gradients for filtered tokens."""
        grad_filtered_logits = grad_outputs[0]
        (keep_mask,) = ctx.saved_tensors

        # Apply masks to gradients - masked out tokens should not receive gradients
        if keep_mask is not None:
            grad_filtered_logits = grad_filtered_logits.masked_fill(~keep_mask, 0.0)

        return grad_filtered_logits, None, None, None


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: int | None,
    top_p: float,
    chunk_size: int | None = TOP_K_TOP_P_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply top-k and top-p masks to the logits with proper gradient handling.

    Simplified version of VLLM's implementation for scalar parameters.

    When top_p < 1.0, sorting is required which is memory intensive for large vocab sizes.
    Processing is done in chunks (controlled by chunk_size) to reduce peak memory.

    Based on VLLM's implementation:
    https://github.com/vllm-project/vllm/blob/34a20c49b3f81f64133428b3a0d62309db1256f9/vllm/v1/sample/ops/topk_topp_sampler.py
    SPDX-License-Identifier: Apache-2.0
    Copyright contributors to the vLLM project

    Args:
        logits: Input logits tensor of shape [*, vocab_size].
        top_k: Top-k sampling parameter. Set to -1 to consider all tokens.
        top_p: Top-p (nucleus) sampling parameter. Must be in (0, 1]. Set to 1 to consider all tokens.
        chunk_size: Number of tokens to process per chunk. Defaults to TOP_K_TOP_P_CHUNK_SIZE.

    Returns:
        filtered_logits: Filtered logits tensor with the same shape as input logits.
        keep_mask: Mask tensor with the same shape as input logits, where 1 (True) indicates tokens to be
            kept, 0 (False) indicates tokens to be masked.
    """
    if not _need_top_k_filtering(top_k) and not _need_top_p_filtering(top_p):
        return logits, None
    return _ApplyTopKTopP.apply(logits, top_k, top_p, chunk_size)
