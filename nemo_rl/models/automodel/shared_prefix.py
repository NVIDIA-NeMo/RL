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

"""Train-time shared-prefix execution for Hugging Face Qwen3.

The logical GRPO microbatch remains a normal ``[batch, sequence]`` tensor.  This
module builds a physical ``[1, compact_tokens]`` representation containing one
copy of each prompt followed by all responses that share it.  Qwen3 norm, MLP,
and projections run directly on that compact representation.  The registered
attention backend preserves the logical causal semantics with two varlen FA2
calls: prompt self-attention and response-to-(prompt + own response) attention.
"""

from dataclasses import dataclass
from typing import Any

import torch


# Deliberately contains no ``flash`` substring. Transformers treats any backend
# name containing it as a built-in/hub FlashAttention request before dispatch.
SHARED_PREFIX_ATTENTION = "nemo_zorro_shared_prefix"
SHARED_PREFIX_GROUP_IDS = "shared_prefix_group_ids"
SHARED_PREFIX_LENGTHS = "shared_prefix_lengths"


@dataclass(frozen=True)
class SharedPrefixLayout:
    """Tensor metadata describing one compact shared-prefix microbatch."""

    compact_input_ids: torch.Tensor
    compact_position_ids: torch.Tensor

    prompt_token_indices: torch.Tensor
    response_token_indices: torch.Tensor
    response_kv_indices: torch.Tensor

    prompt_cu_seqlens: torch.Tensor
    response_cu_seqlens: torch.Tensor
    response_kv_cu_seqlens: torch.Tensor

    predictor_indices: torch.Tensor
    response_target_ids: torch.Tensor
    loss_logprob_scatter_indices: torch.Tensor

    max_prompt_length: int
    max_response_length: int
    max_response_kv_length: int
    original_batch_size: int
    original_sequence_length: int

    @property
    def compact_tokens(self) -> int:
        return int(self.compact_input_ids.shape[1])

    @property
    def response_tokens(self) -> int:
        return int(self.response_target_ids.numel())


def infer_shared_prefix_response_bounds(
    token_mask: torch.Tensor,
    input_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer the prompt start and effective end of one trainable response.

    Each real sequence must have exactly
    ``prompt(0*) + response(1+) + ignored_tail(0*)`` in its token loss mask.
    The ignored tail covers terminal environment observations appended by native
    NeMo-RL rollouts; it is safe to crop because no later trainable token attends
    to it. Multi-turn ``0*1+0+1+`` masks are rejected rather than silently
    receiving incorrect shared-prefix semantics.
    """
    if token_mask.ndim != 2:
        raise ValueError(
            f"token_mask must be rank 2, got shape={tuple(token_mask.shape)}"
        )
    if input_lengths.ndim != 1 or input_lengths.shape[0] != token_mask.shape[0]:
        raise ValueError(
            "input_lengths must be rank 1 with the same batch size as token_mask"
        )

    batch_size, sequence_length = token_mask.shape
    positions = torch.arange(sequence_length, device=token_mask.device).unsqueeze(0)
    lengths = input_lengths.to(device=token_mask.device, dtype=torch.long)
    if torch.any(lengths > sequence_length) or torch.any(lengths <= 0):
        raise ValueError("input_lengths must be in [1, token_mask.shape[1]]")

    mask = token_mask != 0
    real_tokens = positions < lengths.unsqueeze(1)
    response_tokens = mask & real_tokens
    if not torch.all(response_tokens.any(dim=1)):
        raise ValueError(
            "every shared-prefix rollout must contain at least one response token"
        )

    first_response = torch.where(
        response_tokens,
        positions.expand(batch_size, -1),
        sequence_length,
    ).amin(dim=1)
    last_response = torch.where(
        response_tokens,
        positions.expand(batch_size, -1),
        -1,
    ).amax(dim=1)
    expected_mask = (
        (positions >= first_response.unsqueeze(1))
        & (positions <= last_response.unsqueeze(1))
        & real_tokens
    )
    if not torch.equal(mask, expected_mask):
        raise ValueError(
            "shared-prefix training currently requires a single contiguous response span"
        )
    if torch.any(first_response <= 0):
        raise ValueError(
            "every shared-prefix rollout must contain at least one prompt token"
        )
    return first_response, last_response + 1


def build_shared_prefix_layout(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: torch.Tensor,
) -> SharedPrefixLayout:
    """Build the compact physical layout for one already-formed microbatch."""
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be rank 2, got shape={tuple(input_ids.shape)}"
        )
    batch_size, sequence_length = input_ids.shape
    for name, tensor in (
        ("input_lengths", input_lengths),
        ("prompt_lengths", prompt_lengths),
        ("group_ids", group_ids),
    ):
        if tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must be rank 1 with batch size {batch_size}")
    if batch_size == 0:
        raise ValueError("shared-prefix microbatches cannot be empty")

    device = input_ids.device
    # Fetch the small row metadata together. When callers construct a layout
    # directly from CUDA tensors this avoids three separate device syncs.
    row_metadata = torch.stack(
        (
            input_lengths.to(dtype=torch.long),
            prompt_lengths.to(dtype=torch.long),
            group_ids.to(dtype=torch.long),
        ),
        dim=1,
    ).to(device="cpu")
    input_lengths_list = row_metadata[:, 0].tolist()
    prompt_lengths_list = row_metadata[:, 1].tolist()
    group_ids_list = row_metadata[:, 2].tolist()

    group_order: list[int] = []
    rows_by_group: dict[int, list[int]] = {}
    for row, group_id in enumerate(group_ids_list):
        if group_id not in rows_by_group:
            rows_by_group[group_id] = []
            group_order.append(group_id)
        rows_by_group[group_id].append(row)

    compact_ids_parts: list[torch.Tensor] = []
    compact_positions_parts: list[torch.Tensor] = []
    prompt_indices_parts: list[torch.Tensor] = []
    response_indices_parts: list[torch.Tensor] = []
    response_kv_indices_parts: list[torch.Tensor] = []
    predictor_indices_parts: list[torch.Tensor] = []
    response_target_parts: list[torch.Tensor] = []
    loss_scatter_parts: list[torch.Tensor] = []

    prompt_cu = [0]
    response_cu = [0]
    response_kv_cu = [0]
    max_prompt_length = 0
    max_response_length = 0
    max_response_kv_length = 0
    compact_cursor = 0
    prompt_match_checks: list[torch.Tensor] = []

    for group_id in group_order:
        rows = rows_by_group[group_id]
        representative = rows[0]
        prompt_length = prompt_lengths_list[representative]
        if not 0 < prompt_length < input_lengths_list[representative]:
            raise ValueError(
                "each rollout must have at least one prompt token and one response token"
            )

        prompt = input_ids[representative, :prompt_length]
        prompt_start = compact_cursor
        prompt_end = prompt_start + prompt_length
        prompt_indices = torch.arange(prompt_start, prompt_end, device=device)

        compact_ids_parts.append(prompt)
        compact_positions_parts.append(torch.arange(prompt_length, device=device))
        prompt_indices_parts.append(prompt_indices)
        prompt_cu.append(prompt_cu[-1] + prompt_length)
        max_prompt_length = max(max_prompt_length, prompt_length)
        compact_cursor = prompt_end

        for row in rows:
            total_length = input_lengths_list[row]
            row_prompt_length = prompt_lengths_list[row]
            if not 0 < row_prompt_length < total_length <= sequence_length:
                raise ValueError(
                    "each rollout must satisfy 0 < prompt_length < input_length <= sequence_length"
                )
            if row_prompt_length != prompt_length:
                raise ValueError(
                    f"all rows in shared-prefix group {group_id} must have identical prompts"
                )
            if row != representative:
                # Defer conversion to a Python bool until every row has been
                # checked, avoiding one CUDA synchronization per rollout.
                prompt_match_checks.append(
                    torch.all(input_ids[row, :prompt_length] == prompt)
                )

            response = input_ids[row, prompt_length:total_length]
            response_length = total_length - prompt_length
            response_start = compact_cursor
            response_end = response_start + response_length
            response_indices = torch.arange(response_start, response_end, device=device)

            compact_ids_parts.append(response)
            compact_positions_parts.append(
                torch.arange(prompt_length, total_length, device=device)
            )
            response_indices_parts.append(response_indices)
            response_kv_indices_parts.append(
                torch.cat((prompt_indices, response_indices))
            )

            predictor_indices_parts.append(
                torch.cat(
                    (
                        torch.tensor([prompt_end - 1], device=device),
                        response_indices[:-1],
                    )
                )
            )
            response_target_parts.append(response)

            response_positions = torch.arange(
                prompt_length,
                total_length,
                device=device,
                dtype=torch.long,
            )
            loss_scatter_parts.append(
                row * (sequence_length - 1) + response_positions - 1
            )

            response_cu.append(response_cu[-1] + response_length)
            response_kv_length = prompt_length + response_length
            response_kv_cu.append(response_kv_cu[-1] + response_kv_length)
            max_response_length = max(max_response_length, response_length)
            max_response_kv_length = max(max_response_kv_length, response_kv_length)
            compact_cursor = response_end

    if prompt_match_checks and not torch.stack(prompt_match_checks).all().item():
        raise ValueError(
            "all rows in a shared-prefix group must have identical prompts"
        )

    def _cat(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts).to(dtype=torch.long)

    return SharedPrefixLayout(
        compact_input_ids=torch.cat(compact_ids_parts).unsqueeze(0),
        compact_position_ids=torch.cat(compact_positions_parts).unsqueeze(0),
        prompt_token_indices=_cat(prompt_indices_parts),
        response_token_indices=_cat(response_indices_parts),
        response_kv_indices=_cat(response_kv_indices_parts),
        prompt_cu_seqlens=torch.tensor(prompt_cu, device=device, dtype=torch.int32),
        response_cu_seqlens=torch.tensor(response_cu, device=device, dtype=torch.int32),
        response_kv_cu_seqlens=torch.tensor(
            response_kv_cu, device=device, dtype=torch.int32
        ),
        predictor_indices=_cat(predictor_indices_parts),
        response_target_ids=_cat(response_target_parts),
        loss_logprob_scatter_indices=_cat(loss_scatter_parts),
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_response_kv_length=max_response_kv_length,
        original_batch_size=batch_size,
        original_sequence_length=sequence_length,
    )


def _flash_attention_functions():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
    except ImportError as error:  # pragma: no cover - exercised in the GPU environment
        raise ImportError(
            "shared-prefix Qwen3 training requires flash-attn with FA2 varlen support"
        ) from error
    return flash_attn_func, flash_attn_varlen_func


def _standard_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout: float,
    scaling: float | None,
    is_causal: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Keep ordinary logprob/inference forwards working under the custom backend."""
    flash_attn_func, flash_attn_varlen_func = _flash_attention_functions()
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    packed_kwargs = kwargs.get("flash_attn_kwargs")

    def _first_not_none(*values: Any) -> Any:
        return next((value for value in values if value is not None), None)

    def _packed_value(name: str) -> Any:
        if packed_kwargs is None:
            return None
        if isinstance(packed_kwargs, dict):
            return packed_kwargs.get(name)
        return getattr(packed_kwargs, name, None)

    cu_q = _first_not_none(
        _packed_value("cu_seqlens_q"),
        kwargs.get("cu_seq_lens_q"),
        kwargs.get("cu_seqlens_q"),
    )
    cu_k = _first_not_none(
        _packed_value("cu_seqlens_k"),
        kwargs.get("cu_seq_lens_k"),
        kwargs.get("cu_seqlens_k"),
    )
    max_q = _first_not_none(
        _packed_value("max_seqlen_q"),
        kwargs.get("max_length_q"),
        kwargs.get("max_seqlen_q"),
    )
    max_k = _first_not_none(
        _packed_value("max_seqlen_k"),
        kwargs.get("max_length_k"),
        kwargs.get("max_seqlen_k"),
    )

    if cu_q is not None:
        if any(value is None for value in (cu_k, max_q, max_k)):
            raise ValueError("packed FA2 requires Q/K cu_seqlens and max lengths")
        # NeMo-RL's packed metadata may inherit the int64 dtype of
        # ``input_lengths``. The FA2 varlen ABI requires CUDA int32 cumulative
        # lengths, matching Transformers' native FlashAttention wrapper.
        cu_q = cu_q.to(device=query.device, dtype=torch.int32)
        cu_k = cu_k.to(device=query.device, dtype=torch.int32)
        return flash_attn_varlen_func(
            query.reshape(-1, query.shape[-2], query.shape[-1]),
            key.reshape(-1, key.shape[-2], key.shape[-1]),
            value.reshape(-1, value.shape[-2], value.shape[-1]),
            cu_q,
            cu_k,
            max_q,
            max_k,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=is_causal,
        ).unsqueeze(0)

    return flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=is_causal,
    )


def shared_prefix_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    shared_prefix_layout: SharedPrefixLayout | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Transformers ``AttentionInterface`` implementation for Qwen3 + FA2."""
    if query.dtype == torch.float32:
        # Match Transformers' native FA2 wrapper. Qwen3's fp32 norm weights can
        # promote Q/K even while the surrounding forward is autocast to BF16.
        from transformers.integrations.flash_attention import get_target_dtype

        target_dtype = get_target_dtype(query, module)
        if target_dtype is None:
            raise ValueError(
                "shared-prefix FA2 received float32 QKV outside CUDA autocast"
            )
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    if shared_prefix_layout is None:
        if sliding_window is not None:
            raise ValueError("the Qwen3 shared-prefix backend requires full attention")
        output = _standard_flash_attention(
            query,
            key,
            value,
            dropout=dropout,
            scaling=scaling,
            is_causal=(
                kwargs["is_causal"]
                if kwargs.get("is_causal") is not None
                else module.is_causal
            ),
            kwargs=kwargs,
        )
        return output, None

    if attention_mask is not None:
        raise ValueError("shared-prefix attention expects attention_mask=None")
    if sliding_window is not None:
        raise ValueError(
            "shared-prefix attention does not support sliding-window Qwen3"
        )
    if dropout != 0.0:
        raise ValueError("shared-prefix attention requires attention_dropout=0")
    if query.shape[0] != 1:
        raise ValueError("shared-prefix attention expects a physical batch size of 1")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"shared-prefix FA2 requires float16 or bfloat16 QKV, got {query.dtype}"
        )

    _, flash_attn_varlen_func = _flash_attention_functions()
    query_tokens = query.transpose(1, 2).squeeze(0)
    key_tokens = key.transpose(1, 2).squeeze(0)
    value_tokens = value.transpose(1, 2).squeeze(0)
    layout = shared_prefix_layout

    prompt_output = flash_attn_varlen_func(
        query_tokens[layout.prompt_token_indices],
        key_tokens[layout.prompt_token_indices],
        value_tokens[layout.prompt_token_indices],
        layout.prompt_cu_seqlens,
        layout.prompt_cu_seqlens,
        layout.max_prompt_length,
        layout.max_prompt_length,
        dropout_p=0.0,
        softmax_scale=scaling,
        causal=True,
    )
    response_output = flash_attn_varlen_func(
        query_tokens[layout.response_token_indices],
        key_tokens[layout.response_kv_indices],
        value_tokens[layout.response_kv_indices],
        layout.response_cu_seqlens,
        layout.response_kv_cu_seqlens,
        layout.max_response_length,
        layout.max_response_kv_length,
        dropout_p=0.0,
        softmax_scale=scaling,
        causal=True,
    )

    compact_output = torch.empty_like(query_tokens)
    compact_output = compact_output.index_copy(
        0, layout.prompt_token_indices, prompt_output
    )
    compact_output = compact_output.index_copy(
        0, layout.response_token_indices, response_output
    )
    return compact_output.unsqueeze(0), None


def register_shared_prefix_attention() -> None:
    """Register the Qwen3 backend before config/model construction."""
    from transformers import AttentionInterface
    from transformers.masking_utils import (
        AttentionMaskInterface,
        flash_attention_mask,
    )

    AttentionInterface.register(
        SHARED_PREFIX_ATTENTION,
        shared_prefix_flash_attention_forward,
    )
    # Since Transformers 5.5, causal-mask construction dispatches through a
    # registry separate from AttentionInterface.  Reuse FA2's mask policy: an
    # ordinary dense/packed forward receives its 2D padding metadata, while the
    # compact train forward (attention_mask=None) stays mask-free and implements
    # causality inside the two FA2 calls below.
    AttentionMaskInterface.register(
        SHARED_PREFIX_ATTENTION,
        flash_attention_mask,
    )


class _ChunkedTargetLogprobs(torch.autograd.Function):
    """Target-only log-softmax with bounded fp32 activation memory.

    The model already owns the input logits. Forward saves only those logits and
    the target ids, while each fp32 log-softmax chunk is released immediately.
    Backward rematerializes one fp32 softmax chunk at a time instead of retaining
    ``response_tokens * vocab_size`` fp32 activations across the model backward.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        logits: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(logits, targets)

        output = torch.empty(logits.shape[0], device=logits.device, dtype=torch.float32)
        for start in range(0, logits.shape[0], chunk_size):
            end = min(start + chunk_size, logits.shape[0])
            log_probs = torch.nn.functional.log_softmax(
                logits[start:end].float(), dim=-1
            )
            output[start:end] = log_probs.gather(
                -1, targets[start:end].unsqueeze(-1)
            ).squeeze(-1)
            del log_probs
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        logits, targets = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        grad_logits = torch.empty_like(logits)

        for start in range(0, logits.shape[0], chunk_size):
            end = min(start + chunk_size, logits.shape[0])
            probabilities = torch.softmax(logits[start:end].float(), dim=-1)
            probabilities.neg_()
            target_indices = targets[start:end].unsqueeze(-1)
            probabilities.scatter_add_(
                -1,
                target_indices,
                torch.ones_like(target_indices, dtype=probabilities.dtype),
            )
            probabilities.mul_(grad_output[start:end].float().unsqueeze(-1))
            grad_logits[start:end].copy_(probabilities)
            del probabilities, target_indices

        return grad_logits, None, None


def response_logprobs_from_logits(
    logits: torch.Tensor,
    layout: SharedPrefixLayout,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Select response-target logprobs from predictor-only Qwen3 logits."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(
            "shared-prefix Qwen3 logits must have shape [1, response_tokens, vocab]"
        )
    if logits.shape[1] != layout.response_tokens:
        raise ValueError(
            f"expected {layout.response_tokens} predictor logits, got {logits.shape[1]}"
        )
    logits = logits.squeeze(0)
    if chunk_size is None:
        chunk_size = layout.response_tokens
    if chunk_size <= 0:
        raise ValueError("shared-prefix logprob_chunk_size must be positive")
    return _ChunkedTargetLogprobs.apply(
        logits,
        layout.response_target_ids,
        chunk_size,
    )


def scatter_response_logprobs(
    response_logprobs: torch.Tensor,
    layout: SharedPrefixLayout,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore response logprobs to NeMo-RL's logical dense alignment."""
    if (
        response_logprobs.ndim != 1
        or response_logprobs.numel() != layout.response_tokens
    ):
        raise ValueError(
            f"expected {layout.response_tokens} response logprobs, got shape={tuple(response_logprobs.shape)}"
        )
    width = layout.original_sequence_length - 1
    if base is None:
        flat = response_logprobs.new_zeros(layout.original_batch_size * width)
    else:
        if base.shape != (layout.original_batch_size, width):
            raise ValueError(
                "shared-prefix logprob base must match the logical loss shape"
            )
        flat = base.detach().clone().reshape(-1)
    flat = flat.scatter(0, layout.loss_logprob_scatter_indices, response_logprobs)
    return flat.view(layout.original_batch_size, width)
