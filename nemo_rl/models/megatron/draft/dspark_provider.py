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
        hard_labels: torch.Tensor,
        valid_mask: torch.Tensor,
        slot_bins: torch.Tensor,
        loss_weights: tuple[float, float, float],
        token_chunk_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> DSparkObjectiveStats:
        """Project with the detached live head and evaluate raw DSpark loss bins."""
        if self.markov_head.tensor_parallel_group is not tp_group:
            raise ValueError("objective TP group must match the DSpark head TP group")
        if draft_hidden.ndim != 3:
            raise ValueError("draft_hidden must have shape [blocks, slots, hidden]")
        if target_output_weight.ndim != 2:
            raise ValueError("target_output_weight must be a matrix")
        if target_output_weight.shape != (
            self.markov_head.local_vocab_size,
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
                ).clamp(0, self.markov_head.vocab_size - 1)
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
            hard_labels=hard_labels,
            valid_mask=valid_mask,
            slot_bins=slot_bins,
            loss_weights=loss_weights,
            token_chunk_size=token_chunk_size,
            vocab_start_index=self.markov_head.vocab_start_index,
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
    vocab_size: int,
    hidden_size: int,
    markov_rank: int,
    confidence_enabled: bool,
    confidence_with_markov: bool,
    vocab_start_index: int = 0,
    vocab_end_index: int | None = None,
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
        vocab_size=vocab_size,
        markov_rank=markov_rank,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
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
