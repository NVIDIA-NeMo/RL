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
from typing import Any, Mapping, Optional

import torch

# Default chunk size for top-k/top-p filtering.
# The sort operation in top-p filtering is memory intensive because it creates
# intermediate tensors of shape [bsz, seq_len, vocab_size] for both sorted values
# and indices. For large vocab sizes (e.g., 152K) and long sequences (e.g., 32K),
# this can cause OOM. Chunking along the sequence dimension reduces peak memory.
# Different chunk sizes have minor performance differences.
TOP_K_TOP_P_CHUNK_SIZE: int = 256

SAMPLING_MASK_TOKEN_IDS = "sampling_mask_token_ids"
SAMPLING_MASK_SIZES = "sampling_mask_sizes"


@dataclass
class SamplingMask:
    """Exact per-token vocabulary support captured during rollout.

    The metadata is aligned to input tokens: ``token_ids[b, t]`` contains the
    vocabulary IDs that were eligible when ``input_ids[b, t]`` was sampled.
    ``sizes[b, t]`` is the number of valid entries in that row. The final
    dimension is padded to a common finite width.

    A size of zero is allowed for prompt and padding positions. Logprob kernels
    treat such a row as the singleton support containing its target token, so
    every softmax remains well-defined. Callers that own the actor loss mask
    must separately reject zero-sized rows at active loss positions.
    """

    token_ids: torch.Tensor
    sizes: torch.Tensor


def sampling_mask_from_data(data: Mapping[str, Any]) -> Optional[SamplingMask]:
    """Build a :class:`SamplingMask` from a batch, checking paired presence."""
    has_token_ids = SAMPLING_MASK_TOKEN_IDS in data
    has_sizes = SAMPLING_MASK_SIZES in data
    if not has_token_ids and not has_sizes:
        return None
    if has_token_ids != has_sizes:
        missing = SAMPLING_MASK_SIZES if has_token_ids else SAMPLING_MASK_TOKEN_IDS
        raise ValueError(
            "Sampling-mask replay metadata is incomplete: "
            f"missing required field {missing!r}."
        )

    token_ids = data[SAMPLING_MASK_TOKEN_IDS]
    sizes = data[SAMPLING_MASK_SIZES]
    if not isinstance(token_ids, torch.Tensor) or not isinstance(sizes, torch.Tensor):
        raise TypeError(
            "Sampling-mask replay fields must be tensors, but got "
            f"{SAMPLING_MASK_TOKEN_IDS}={type(token_ids).__name__} and "
            f"{SAMPLING_MASK_SIZES}={type(sizes).__name__}."
        )
    return SamplingMask(token_ids=token_ids, sizes=sizes)


def validate_sampling_mask_shape(
    sampling_mask: SamplingMask,
    target: torch.Tensor,
) -> None:
    """Validate the structural contract against token-aligned ``target``."""
    token_ids = sampling_mask.token_ids
    sizes = sampling_mask.sizes
    if token_ids.ndim != target.ndim + 1:
        raise ValueError(
            "sampling_mask_token_ids must have one trailing support dimension "
            f"beyond target; got token_ids={tuple(token_ids.shape)}, "
            f"target={tuple(target.shape)}."
        )
    if sizes.shape != target.shape or token_ids.shape[:-1] != target.shape:
        raise ValueError(
            "Sampling-mask metadata must align with target tokens; got "
            f"token_ids={tuple(token_ids.shape)}, sizes={tuple(sizes.shape)}, "
            f"target={tuple(target.shape)}."
        )
    if token_ids.shape[-1] <= 0:
        raise ValueError("Sampling-mask support width K must be positive.")
    if (
        token_ids.dtype == torch.bool
        or token_ids.dtype.is_floating_point
        or token_ids.dtype.is_complex
    ):
        raise TypeError(
            f"sampling_mask_token_ids must use an integer dtype, got {token_ids.dtype}."
        )
    if (
        sizes.dtype == torch.bool
        or sizes.dtype.is_floating_point
        or sizes.dtype.is_complex
    ):
        raise TypeError(
            f"sampling_mask_sizes must use an integer dtype, got {sizes.dtype}."
        )


def _sampling_mask_slot_valid(sampling_mask: SamplingMask) -> torch.Tensor:
    support_width = sampling_mask.token_ids.shape[-1]
    slots = torch.arange(
        support_width,
        device=sampling_mask.sizes.device,
        dtype=torch.long,
    )
    return slots < sampling_mask.sizes.unsqueeze(-1)


def validate_sampling_mask_for_active_tokens(
    sampling_mask: SamplingMask,
    target: torch.Tensor,
    active_token_mask: torch.Tensor,
) -> None:
    """Fail if an actor-loss token lacks a nonempty support containing it."""
    validate_sampling_mask_shape(sampling_mask, target)
    if active_token_mask.shape != target.shape:
        raise ValueError(
            "Actor token mask must align with sampling-mask targets; got "
            f"active_token_mask={tuple(active_token_mask.shape)}, "
            f"target={tuple(target.shape)}."
        )

    active = active_token_mask.bool()
    if torch.any(active & (sampling_mask.sizes <= 0)).item():
        raise ValueError(
            "Sampling-mask replay requires a nonempty support at every active "
            "actor-loss token."
        )

    slot_valid = _sampling_mask_slot_valid(sampling_mask)
    target_present = torch.any(
        slot_valid & (sampling_mask.token_ids == target.unsqueeze(-1)), dim=-1
    )
    if torch.any(active & ~target_present).item():
        raise ValueError(
            "Sampling-mask replay support does not contain the sampled target "
            "at one or more active actor-loss tokens."
        )


def validate_sampling_mask_contents(
    sampling_mask: SamplingMask,
    target: torch.Tensor,
    vocab_size: int,
) -> None:
    """Validate support sizes, IDs, and target membership without logits."""
    validate_sampling_mask_shape(sampling_mask, target)
    support_width = sampling_mask.token_ids.shape[-1]
    sizes = sampling_mask.sizes
    if torch.any((sizes < 0) | (sizes > support_width)).item():
        raise ValueError(
            "sampling_mask_sizes entries must be between 0 and the padded "
            f"support width K={support_width}."
        )

    if vocab_size <= 0:
        raise ValueError(
            f"Sampling-mask vocabulary size must be positive, got {vocab_size}."
        )
    if torch.any((target < 0) | (target >= vocab_size)).item():
        raise ValueError(f"Sampling-mask target IDs must be in [0, {vocab_size}).")

    slot_valid = _sampling_mask_slot_valid(sampling_mask)
    token_ids = sampling_mask.token_ids
    invalid_ids = slot_valid & ((token_ids < 0) | (token_ids >= vocab_size))
    if torch.any(invalid_ids).item():
        raise ValueError(
            f"Sampling-mask token IDs must be in [0, {vocab_size}) within "
            "the valid prefix of each support row."
        )

    target_present = torch.any(slot_valid & (token_ids == target.unsqueeze(-1)), dim=-1)
    nonempty = sizes > 0
    if torch.any(nonempty & ~target_present).item():
        raise ValueError(
            "Sampling-mask support does not contain its sampled target in one "
            "or more nonempty rows."
        )


def apply_sampling_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    sampling_mask: SamplingMask,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict logits to an exact rollout support.

    ``logits`` must contain the full vocabulary for each target row. A
    zero-sized metadata row is converted to singleton ``{target}``, which keeps
    prompt/padding rows numerically valid while their loss mask remains zero.
    """
    validate_sampling_mask_shape(sampling_mask, target)
    if logits.shape[:-1] != target.shape:
        raise ValueError(
            "Sampling-mask logits and targets must align; got "
            f"logits={tuple(logits.shape)}, target={tuple(target.shape)}."
        )

    vocab_size = logits.shape[-1]
    validate_sampling_mask_contents(sampling_mask, target, vocab_size)
    slot_valid = _sampling_mask_slot_valid(sampling_mask)
    token_ids = sampling_mask.token_ids
    sizes = sampling_mask.sizes
    nonempty = sizes > 0

    # Accumulate instead of assigning booleans: padded slots commonly contain
    # token ID 0, and a later invalid slot must not overwrite an earlier valid
    # occurrence of that ID. Accumulation also makes duplicate IDs harmless.
    keep_counts = torch.zeros_like(logits, dtype=torch.int32)
    safe_token_ids = token_ids.to(dtype=torch.long).clamp(min=0, max=vocab_size - 1)
    keep_counts.scatter_add_(
        dim=-1,
        index=safe_token_ids,
        src=slot_valid.to(dtype=keep_counts.dtype),
    )

    empty = ~nonempty
    keep_counts.scatter_add_(
        dim=-1,
        index=target.to(dtype=torch.long).unsqueeze(-1),
        src=empty.unsqueeze(-1).to(dtype=keep_counts.dtype),
    )
    keep_mask = keep_counts > 0
    filtered_logits = logits.masked_fill(~keep_mask, -float("inf"))
    return filtered_logits, keep_mask


@dataclass
class TrainingSamplingParams:
    """Training-specific sampling parameters to match generation parameters.

    Used to ensure consistency between training and inference by applying the same sampling strategy during
    logprob computation. Not directly using vLLM's SamplingParams class to avoid dependency on vLLM in this env.

    Attributes:
        top_k: Top-k filtering parameter (None or -1 to disable)
        top_p: Top-p filtering parameter (1.0 to disable)
        temperature: Temperature for scaling logits (default: 1.0)
        replay_sampling_mask: Require and consume exact rollout supports instead
            of independently reconstructing top-k/top-p supports.
    """

    top_k: int | None = None
    top_p: float = 1.0
    temperature: float = 1.0
    replay_sampling_mask: bool = False


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
    logits = logits.reshape(
        -1, vocab_size
    )  # [*, vocab_size] -> [num_tokens, vocab_size]
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
