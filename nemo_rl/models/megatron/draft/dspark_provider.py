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

"""Private provider-shaped adapter for independently tested DSpark pieces."""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from torch import nn

from nemo_rl.algorithms.loss.dspark import (
    DSparkObjectiveStats,
    dspark_tiled_objective,
)
from nemo_rl.models.megatron.draft.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)


def _dtype_code(dtype: torch.dtype) -> int:
    dtype_names = (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    )
    return next(
        (
            index
            for index, name in enumerate(dtype_names)
            if dtype == getattr(torch, name, None)
        ),
        -1,
    )


def _provider_tensor_descriptor(
    tensor: torch.Tensor | None,
    *,
    collective_device: torch.device,
) -> list[int]:
    if tensor is None:
        return [-1, -1, -1, -1, -1, -1, -1]
    shape = [*tensor.shape[:3], *([-1] * max(0, 3 - tensor.ndim))]
    return [
        tensor.ndim,
        *shape,
        _dtype_code(tensor.dtype),
        int(tensor.device == collective_device),
        int(tensor.is_floating_point()),
    ]


def _preflight_provider_inputs(
    *,
    draft_hidden: torch.Tensor,
    target_output_weight: torch.Tensor,
    target_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    slot_bins: torch.Tensor,
    slot_weights: torch.Tensor | None,
    configured_tp_group: torch.distributed.ProcessGroup | None,
    requested_tp_group: torch.distributed.ProcessGroup | None,
) -> None:
    if requested_tp_group is None:
        if configured_tp_group is not None:
            raise ValueError("objective TP group must match the DSpark head TP group")
        return
    if not torch.distributed.is_initialized():
        return
    agreement_group = requested_tp_group
    descriptor = [int(configured_tp_group is requested_tp_group)]
    for tensor in (
        draft_hidden,
        target_output_weight,
        target_logits,
        previous_token_ids,
        valid_mask,
        slot_bins,
        slot_weights,
    ):
        descriptor.extend(
            _provider_tensor_descriptor(
                tensor,
                collective_device=draft_hidden.device,
            )
        )
    local = torch.tensor(descriptor, dtype=torch.int64, device=draft_hidden.device)
    minimum = local.clone()
    maximum = local.clone()
    torch.distributed.all_reduce(
        minimum,
        op=torch.distributed.ReduceOp.MIN,
        group=agreement_group,
    )
    torch.distributed.all_reduce(
        maximum,
        op=torch.distributed.ReduceOp.MAX,
        group=agreement_group,
    )
    if not torch.equal(minimum, maximum):
        raise ValueError("tensor-parallel ranks must agree on DSpark provider inputs")


class _CheckpointableBody(Protocol):
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class _DSparkProviderAdapter(nn.Module):
    """Own the DSpark body and heads while borrowing the live target head."""

    def __init__(
        self,
        *,
        body: nn.Module,
        markov_head: DSparkMarkovHead,
        confidence_head: DSparkConfidenceHead | None,
    ) -> None:
        super().__init__()
        self.body = body
        self.markov_head = markov_head
        self.confidence_head = confidence_head

    def objective_stats(
        self,
        *,
        draft_hidden: torch.Tensor,
        target_output_weight: torch.Tensor,
        target_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        slot_bins: torch.Tensor,
        slot_weights: torch.Tensor | None = None,
        loss_weights: tuple[float, float, float],
        token_chunk_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> DSparkObjectiveStats:
        """Project with the detached live head and evaluate raw DSpark loss bins."""
        _preflight_provider_inputs(
            draft_hidden=draft_hidden,
            target_output_weight=target_output_weight,
            target_logits=target_logits,
            previous_token_ids=previous_token_ids,
            valid_mask=valid_mask,
            slot_bins=slot_bins,
            slot_weights=slot_weights,
            configured_tp_group=self.markov_head.tensor_parallel_group,
            requested_tp_group=tp_group,
        )
        if self.markov_head.tensor_parallel_group is not tp_group:
            raise ValueError("objective TP group must match the DSpark head TP group")
        if draft_hidden.ndim != 3:
            raise ValueError("draft_hidden must have shape [blocks, slots, hidden]")
        if target_output_weight.ndim != 2:
            raise ValueError("target_output_weight must be a matrix")
        if target_output_weight.shape != (
            self.markov_head.local_draft_vocab_size,
            draft_hidden.shape[-1],
        ):
            raise ValueError(
                "target_output_weight must match the local vocabulary and hidden size"
            )
        if target_output_weight.dtype != draft_hidden.dtype:
            raise ValueError("target output weight and draft hidden must share a dtype")
        if target_output_weight.device != draft_hidden.device:
            raise ValueError(
                "target output weight and draft hidden must share a device"
            )

        confidence_logits = None
        if self.confidence_head is not None:
            markov_embeddings = None
            if self.confidence_head.with_markov:
                safe_previous_token_ids = torch.where(
                    valid_mask,
                    previous_token_ids,
                    torch.zeros_like(previous_token_ids),
                ).clamp(0, self.markov_head.target_vocab_size - 1)
                markov_embeddings = self.markov_head.markov_w1(safe_previous_token_ids)
            confidence_logits = self.confidence_head(
                draft_hidden,
                slot_valid=valid_mask,
                markov_embeddings=markov_embeddings,
            )

        return dspark_tiled_objective(
            target_logits=target_logits,
            draft_hidden=draft_hidden,
            target_output_weight=target_output_weight,
            markov_w1=self.markov_head.markov_w1.weight,
            markov_w2=self.markov_head.markov_w2.weight,
            previous_token_ids=previous_token_ids,
            confidence_logits=confidence_logits,
            valid_mask=valid_mask,
            slot_bins=slot_bins,
            slot_weights=slot_weights,
            loss_weights=loss_weights,
            token_chunk_size=token_chunk_size,
            draft_vocab_start_index=self.markov_head.draft_vocab_start_index,
            tp_group=tp_group,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compose body and head checkpoint metadata without target tensors."""
        checkpointable_body = cast(_CheckpointableBody, self.body)
        body_state = checkpointable_body.sharded_state_dict(
            prefix=f"{prefix}body.",
            sharded_offsets=sharded_offsets,
            metadata=metadata,
        )
        state = dict(body_state)
        state.update(
            self.markov_head.sharded_state_dict(
                prefix=f"{prefix}markov_head.",
                sharded_offsets=sharded_offsets,
                metadata=metadata,
            )
        )
        if self.confidence_head is not None:
            from megatron.core.transformer.utils import (
                make_sharded_tensors_for_checkpoint,
            )

            dp_cp_group = None if metadata is None else metadata.get("dp_cp_group")
            state.update(
                make_sharded_tensors_for_checkpoint(
                    self.confidence_head.state_dict(prefix="", keep_vars=True),
                    f"{prefix}confidence_head.",
                    {},
                    sharded_offsets,
                    tp_group=self.markov_head.tensor_parallel_group,
                    dp_cp_group=dp_cp_group,
                )
            )
        return state


def build_dspark_provider(
    *,
    body: nn.Module,
    target_vocab_size: int,
    draft_vocab_size: int,
    hidden_size: int,
    markov_rank: int,
    confidence_enabled: bool,
    confidence_with_markov: bool,
    draft_vocab_start_index: int = 0,
    draft_vocab_end_index: int | None = None,
    tensor_parallel_group: torch.distributed.ProcessGroup | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> _DSparkProviderAdapter:
    """Construct the private DSpark adapter without registering a generic loss."""
    if confidence_with_markov and not confidence_enabled:
        raise ValueError("confidence_with_markov requires confidence_enabled")
    if not callable(getattr(body, "sharded_state_dict", None)):
        raise TypeError("DSpark body must implement sharded_state_dict")
    markov_head = DSparkMarkovHead(
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        markov_rank=markov_rank,
        draft_vocab_start_index=draft_vocab_start_index,
        draft_vocab_end_index=draft_vocab_end_index,
        tensor_parallel_group=tensor_parallel_group,
        device=device,
        dtype=dtype,
    )
    confidence_head = None
    if confidence_enabled:
        confidence_head = DSparkConfidenceHead(
            hidden_size=hidden_size,
            markov_rank=markov_rank,
            with_markov=confidence_with_markov,
            device=device,
            dtype=dtype,
        )
    return _DSparkProviderAdapter(
        body=body,
        markov_head=markov_head,
        confidence_head=confidence_head,
    )


__all__ = ["build_dspark_provider"]
