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

"""Small checkpoint-compatible heads used by DSpark block drafting.

The parameter names ``markov_w1`` / ``markov_w2`` / ``proj``, and the
``nn.Linear(markov_rank, draft_vocab)`` orientation that makes ``markov_w2.weight``
come out ``[draft_vocab, markov_rank]``, are an interop contract with the official
``deepseek-ai/dspark_qwen3_8b_block7`` checkpoint (revision ``03326e50``). Callers
must additionally attach these modules under the ``markov_head`` / ``confidence_head``
attribute names, since those become the checkpoint key prefixes. Renaming any of the
above silently breaks loading of the official artifact; the pinned schema lives in
``tests/unit/models/megatron/fixtures/dspark_qwen3_8b_block7_03326e50.json``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Keep the forward value local and SUM its gradient across TP ranks."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        tensor: Tensor,
        group: torch.distributed.ProcessGroup,
    ) -> Tensor:
        ctx.group = group  # pyrefly: ignore[implicitly-defined-attribute]
        return tensor

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None]:
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input,
            op=torch.distributed.ReduceOp.SUM,
            group=ctx.group,
        )
        return grad_input, None


class DSparkMarkovHead(nn.Module):
    """Add a previous-token low-rank bias to caller-owned base logits.

    ``markov_w1`` remains a replicated target-vocabulary embedding. The output
    rows of ``markov_w2`` cover either the full draft vocabulary or one explicit
    tensor-parallel draft-vocabulary shard. This module owns neither embeddings
    used by the draft backbone nor an LM head.
    """

    def __init__(
        self,
        *,
        target_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        draft_vocab_start_index: int = 0,
        draft_vocab_end_index: int | None = None,
        tensor_parallel_group: torch.distributed.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be positive")
        if draft_vocab_size <= 0:
            raise ValueError("draft_vocab_size must be positive")
        if markov_rank <= 0:
            raise ValueError("markov_rank must be positive")

        resolved_draft_vocab_end = (
            draft_vocab_size if draft_vocab_end_index is None else draft_vocab_end_index
        )
        if not (
            0 <= draft_vocab_start_index < resolved_draft_vocab_end <= draft_vocab_size
        ):
            raise ValueError(
                "draft vocab shard must satisfy 0 <= draft_vocab_start_index "
                "< draft_vocab_end_index <= draft_vocab_size"
            )

        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.markov_rank = markov_rank
        self.draft_vocab_start_index = draft_vocab_start_index
        self.draft_vocab_end_index = resolved_draft_vocab_end
        self.local_draft_vocab_size = resolved_draft_vocab_end - draft_vocab_start_index
        if (
            self.local_draft_vocab_size != draft_vocab_size
            and tensor_parallel_group is None
        ):
            raise ValueError(
                "tensor_parallel_group is required for a draft vocabulary shard"
            )
        if self.local_draft_vocab_size != draft_vocab_size:
            assert tensor_parallel_group is not None
            tp_size = torch.distributed.get_world_size(tensor_parallel_group)
            tp_rank = torch.distributed.get_rank(tensor_parallel_group)
            if (
                self.local_draft_vocab_size * tp_size != draft_vocab_size
                or draft_vocab_start_index != tp_rank * self.local_draft_vocab_size
            ):
                raise ValueError(
                    "draft vocab shard must be an even rank-local partition of "
                    "draft_vocab_size"
                )
        self.tensor_parallel_group = tensor_parallel_group
        self.markov_w1 = nn.Embedding(
            target_vocab_size,
            markov_rank,
            device=device,
            dtype=dtype,
        )
        self.markov_w2 = nn.Linear(
            markov_rank,
            self.local_draft_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if self.local_draft_vocab_size != draft_vocab_size:
            # The optimizer identifies TP-sharded params by these attributes, not by
            # the checkpoint axis map. Without them param_is_not_tensor_parallel_duplicate
            # treats every tp_rank>0 shard as a duplicate and drops it from the global
            # grad norm, so clipping silently under-clips this head.
            from megatron.core.tensor_parallel.layers import (
                set_tensor_model_parallel_attributes,
            )

            set_tensor_model_parallel_attributes(self.markov_w2.weight, True, 0, 1)

    def embed_previous_tokens(
        self,
        *,
        previous_token_ids: Tensor,
        slot_valid: Tensor,
        reduce_across_tensor_parallel: bool,
    ) -> Tensor:
        """Embed previous-token IDs without invalid lookups or replicated TP drift.

        Invalid slots use token ID zero for the lookup; consumers must still apply
        ``slot_valid`` to their outputs.

        ``reduce_across_tensor_parallel`` must reflect the CONSUMER, not this
        head: pass True when the embeddings feed a vocabulary-sharded weight
        (each rank then holds a partial gradient that must be summed into the
        replicated ``markov_w1``, as ``forward`` does for ``markov_w2``); pass
        False when the consumer is TP-replicated (e.g. the confidence head's
        ``proj``) -- every rank already computes the identical, complete
        gradient there, and summing would over-count it by the TP world size.
        """
        if previous_token_ids.shape != slot_valid.shape:
            raise ValueError("slot_valid must match previous_token_ids")
        if previous_token_ids.dtype != torch.int64:
            raise TypeError("previous_token_ids must use torch.int64")
        if slot_valid.dtype != torch.bool:
            raise TypeError("slot_valid must be a boolean tensor")
        if (
            previous_token_ids.device != slot_valid.device
            or self.markov_w1.weight.device != previous_token_ids.device
        ):
            raise ValueError("DSpark Markov inputs and weights must share a device")

        safe_previous_token_ids = torch.where(
            slot_valid,
            previous_token_ids,
            torch.zeros_like(previous_token_ids),
        )
        previous_embeddings = self.markov_w1(safe_previous_token_ids)
        if (
            reduce_across_tensor_parallel
            and self.local_draft_vocab_size != self.draft_vocab_size
        ):
            assert self.tensor_parallel_group is not None
            previous_embeddings = _CopyToTensorParallelRegion.apply(
                previous_embeddings,
                self.tensor_parallel_group,
            )
        return previous_embeddings

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
        if base_logits.shape[-1] != self.local_draft_vocab_size:
            raise ValueError(
                "base_logits local draft vocab size does not match the configured "
                "draft vocab shard"
            )
        if not base_logits.dtype.is_floating_point:
            raise TypeError("base_logits must use a floating dtype")
        if (
            self.markov_w2.weight.device != base_logits.device
            or self.markov_w1.weight.device != base_logits.device
        ):
            raise ValueError("DSpark Markov inputs and weights must share a device")

        previous_embeddings = self.embed_previous_tokens(
            previous_token_ids=previous_token_ids,
            slot_valid=slot_valid,
            reduce_across_tensor_parallel=True,
        )
        corrected_logits = base_logits + self.markov_w2(previous_embeddings)
        return torch.where(
            slot_valid.unsqueeze(-1),
            corrected_logits,
            torch.zeros_like(corrected_logits),
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return Megatron checkpoint metadata for the TP-local output rows."""
        from megatron.core.transformer.utils import (
            make_sharded_tensors_for_checkpoint,
        )
        from megatron.core.utils import get_tensor_model_parallel_group_if_none

        tensor_parallel_layers_axis_map = (
            {"markov_w2.weight": 0}
            if self.local_draft_vocab_size != self.draft_vocab_size
            else {}
        )
        # make_sharded_tensors_for_checkpoint only falls back to parallel_state when
        # *both* groups are None, and get_pg_rank(None) returns 0 rather than the real
        # rank. Forwarding exactly one of them collapses that axis of replica_id, so
        # several ranks claim is_main_replica and the save raises CheckpointingException.
        # See the same failure documented in eagle.py.
        tp_group = get_tensor_model_parallel_group_if_none(self.tensor_parallel_group)
        dp_cp_group = None if metadata is None else metadata.get("dp_cp_group")
        if tp_group is not None and dp_cp_group is None:
            from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

            dp_cp_group = ensure_metadata_has_dp_cp_group(metadata)["dp_cp_group"]
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            tensor_parallel_layers_axis_map,
            sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
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
        features = torch.where(
            slot_valid.unsqueeze(-1),
            features,
            torch.zeros_like(features),
        )
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
