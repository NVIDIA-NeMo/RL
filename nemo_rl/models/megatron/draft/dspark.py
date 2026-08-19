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

"""Small checkpoint-compatible heads used by DSpark block drafting."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DSparkMarkovHead(nn.Module):
    """Add a previous-token low-rank bias to caller-owned base logits.

    ``markov_w1`` remains a replicated global-vocabulary embedding. The output
    rows of ``markov_w2`` may cover either the full vocabulary or one explicit
    tensor-parallel vocabulary shard. This module owns neither embeddings used
    by the draft backbone nor an LM head.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        vocab_start_index: int = 0,
        vocab_end_index: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if markov_rank <= 0:
            raise ValueError("markov_rank must be positive")

        resolved_vocab_end = vocab_size if vocab_end_index is None else vocab_end_index
        if not 0 <= vocab_start_index < resolved_vocab_end <= vocab_size:
            raise ValueError(
                "vocab shard must satisfy "
                "0 <= vocab_start_index < vocab_end_index <= vocab_size"
            )

        self.vocab_size = vocab_size
        self.markov_rank = markov_rank
        self.vocab_start_index = vocab_start_index
        self.vocab_end_index = resolved_vocab_end
        self.local_vocab_size = resolved_vocab_end - vocab_start_index
        self.markov_w1 = nn.Embedding(
            vocab_size,
            markov_rank,
            device=device,
            dtype=dtype,
        )
        self.markov_w2 = nn.Linear(
            markov_rank,
            self.local_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        base_logits: Tensor,
        *,
        previous_token_ids: Tensor,
        slot_valid: Tensor,
    ) -> Tensor:
        """Return local-vocabulary logits with the Markov bias applied."""
        if base_logits.ndim == 0:
            raise ValueError("base_logits must include a vocabulary dimension")
        leading_shape = base_logits.shape[:-1]
        if previous_token_ids.shape != leading_shape:
            raise ValueError(
                "previous_token_ids must match the base_logits leading shape"
            )
        if slot_valid.shape != leading_shape:
            raise ValueError("slot_valid must match the base_logits leading shape")
        if base_logits.shape[-1] != self.local_vocab_size:
            raise ValueError(
                "base_logits local vocab size does not match the configured vocab shard"
            )
        if not base_logits.dtype.is_floating_point:
            raise TypeError("base_logits must use a floating dtype")
        if previous_token_ids.dtype != torch.int64:
            raise TypeError("previous_token_ids must use torch.int64")
        if slot_valid.dtype != torch.bool:
            raise TypeError("slot_valid must be a boolean tensor")
        if (
            previous_token_ids.device != base_logits.device
            or slot_valid.device != base_logits.device
            or self.markov_w1.weight.device != base_logits.device
            or self.markov_w2.weight.device != base_logits.device
        ):
            raise ValueError("DSpark Markov inputs and weights must share a device")

        safe_previous_token_ids = torch.where(
            slot_valid,
            previous_token_ids,
            torch.zeros_like(previous_token_ids),
        )
        previous_embeddings = self.markov_w1(safe_previous_token_ids)
        corrected_logits = base_logits + self.markov_w2(previous_embeddings)
        return torch.where(
            slot_valid.unsqueeze(-1),
            corrected_logits,
            torch.zeros_like(corrected_logits),
        )


class DSparkConfidenceHead(nn.Module):
    """Predict per-slot acceptance logits from draft and optional Markov state."""

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if markov_rank < 0:
            raise ValueError("markov_rank must be nonnegative")
        if with_markov and markov_rank == 0:
            raise ValueError("with_markov requires a positive markov_rank")

        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.with_markov = with_markov
        input_size = hidden_size + (markov_rank if with_markov else 0)
        self.proj = nn.Linear(
            input_size,
            1,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        slot_valid: Tensor,
        markov_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Return float32 confidence logits, with invalid slots zeroed."""
        if hidden_states.ndim == 0 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have trailing size {self.hidden_size}"
            )
        leading_shape = hidden_states.shape[:-1]
        if slot_valid.shape != leading_shape:
            raise ValueError("slot_valid must match the hidden_states leading shape")
        if not hidden_states.dtype.is_floating_point:
            raise TypeError("hidden_states must use a floating dtype")
        if slot_valid.dtype != torch.bool:
            raise TypeError("slot_valid must be a boolean tensor")
        if slot_valid.device != hidden_states.device:
            raise ValueError("DSpark confidence inputs must share a device")

        if self.with_markov:
            expected_shape = (*leading_shape, self.markov_rank)
            if markov_embeddings is None:
                raise ValueError("markov_embeddings is required when with_markov=True")
            if markov_embeddings.shape != expected_shape:
                raise ValueError(
                    "markov_embeddings must match the hidden_states leading shape "
                    f"and have trailing size {self.markov_rank}"
                )
            if markov_embeddings.device != hidden_states.device:
                raise ValueError("DSpark confidence inputs must share a device")
            if not markov_embeddings.dtype.is_floating_point:
                raise TypeError("markov_embeddings must use a floating dtype")
            features = torch.cat(
                (hidden_states, markov_embeddings.to(dtype=hidden_states.dtype)),
                dim=-1,
            )
        else:
            if markov_embeddings is not None:
                raise ValueError(
                    "markov_embeddings must be omitted when with_markov=False"
                )
            features = hidden_states

        if self.proj.weight.device != hidden_states.device:
            raise ValueError("DSpark confidence inputs and weights must share a device")
        confidence_logits = self.proj(
            features.to(dtype=self.proj.weight.dtype)
        ).squeeze(-1)
        confidence_logits = confidence_logits.float()
        return torch.where(
            slot_valid,
            confidence_logits,
            torch.zeros_like(confidence_logits),
        )


__all__ = ["DSparkConfidenceHead", "DSparkMarkovHead"]
