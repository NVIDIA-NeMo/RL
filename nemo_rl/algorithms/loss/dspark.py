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

"""Tiled hard-CE, total-variation, and confidence objectives for DSpark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from nemo_rl.algorithms.loss.draft import (
    _tile_distributions,
    _tile_log_normalizers,
)


@dataclass(frozen=True, slots=True)
class DSparkLossBins:
    """Raw additive numerator and count bins for one DSpark objective."""

    numerators: torch.Tensor
    counts: torch.Tensor

    def __post_init__(self) -> None:
        if self.numerators.ndim != 1 or self.counts.ndim != 1:
            raise ValueError("DSpark loss numerators and counts must be vectors")
        if self.numerators.shape != self.counts.shape:
            raise ValueError(
                "DSpark loss numerators and counts must have the same shape"
            )

    def normalized(
        self,
        *,
        normalization_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize a local differentiable numerator by externally reduced counts."""
        counts = self.counts if normalization_counts is None else normalization_counts
        if counts.shape != self.counts.shape:
            raise ValueError("normalization_counts must match the local count bins")
        denominator = (
            counts.detach()
            .to(
                device=self.numerators.device,
                dtype=torch.float32,
            )
            .sum()
        )
        return self.numerators.sum() / (denominator + 1e-8)

    def __add__(self, other: object) -> DSparkLossBins:
        if not isinstance(other, DSparkLossBins):
            return NotImplemented
        if self.numerators.shape != other.numerators.shape:
            raise ValueError("DSpark loss bins must have matching shapes")
        return DSparkLossBins(
            numerators=self.numerators + other.numerators,
            counts=self.counts + other.counts,
        )


@dataclass(frozen=True, slots=True)
class DSparkObjectiveStats:
    """Raw CE, TV, confidence, and weighted-combined DSpark statistics."""

    ce: DSparkLossBins
    tv: DSparkLossBins
    confidence: DSparkLossBins
    combined: DSparkLossBins

    def __add__(self, other: object) -> DSparkObjectiveStats:
        if not isinstance(other, DSparkObjectiveStats):
            return NotImplemented
        return DSparkObjectiveStats(
            ce=self.ce + other.ce,
            tv=self.tv + other.tv,
            confidence=self.confidence + other.confidence,
            combined=self.combined + other.combined,
        )


def _tp_vocab_contract(
    local_vocab_size: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> tuple[int, int]:
    if tp_group is None:
        return 0, local_vocab_size
    rank = torch.distributed.get_rank(tp_group)
    world_size = torch.distributed.get_world_size(tp_group)
    return rank * local_vocab_size, world_size * local_vocab_size


def _local_label_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_start: int,
    global_vocab_size: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    local_vocab_size = logits.shape[-1]
    owns_label = labels.ge(vocab_start) & labels.lt(vocab_start + local_vocab_size)
    local_labels = labels.sub(vocab_start).clamp(0, local_vocab_size - 1)
    selected = logits.gather(-1, local_labels.unsqueeze(-1)).squeeze(-1)
    selected = torch.where(owns_label & labels.lt(global_vocab_size), selected, 0.0)
    if tp_group is not None:
        torch.distributed.all_reduce(
            selected,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
    return selected


def _global_argmax(
    logits: torch.Tensor,
    *,
    vocab_start: int,
    global_vocab_size: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    local_max, local_index = logits.max(dim=-1)
    if tp_group is None:
        return local_index
    global_max = local_max.clone()
    torch.distributed.all_reduce(
        global_max,
        op=torch.distributed.ReduceOp.MAX,
        group=tp_group,
    )
    global_index = local_index.add(vocab_start)
    candidates = torch.where(
        local_max.eq(global_max),
        global_index,
        torch.full_like(global_index, global_vocab_size),
    )
    torch.distributed.all_reduce(
        candidates,
        op=torch.distributed.ReduceOp.MIN,
        group=tp_group,
    )
    return candidates


class _TiledHardCEAndTV(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        valid_mask: torch.Tensor,
        slot_bins: torch.Tensor,
        num_bins: int,
        token_chunk_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_vocab_size = draft_logits.shape[-1]
        flat_draft = draft_logits.reshape(-1, local_vocab_size)
        flat_target = target_logits.reshape(-1, local_vocab_size)
        flat_labels = hard_labels.reshape(-1)
        flat_mask = valid_mask.reshape(-1).float()
        flat_bins = slot_bins.reshape(-1)
        ce_numerators = torch.zeros(
            num_bins, dtype=torch.float32, device=draft_logits.device
        )
        tv_numerators = torch.zeros_like(ce_numerators)
        verifier_correct = torch.zeros_like(valid_mask, dtype=torch.float32)
        flat_correct = verifier_correct.reshape(-1)
        log_normalizers = torch.empty(
            (flat_draft.shape[0], 2),
            dtype=torch.float32,
            device=draft_logits.device,
        )
        vocab_start, global_vocab_size = _tp_vocab_contract(
            local_vocab_size,
            tp_group,
        )

        for start in range(0, flat_draft.shape[0], token_chunk_size):
            end = min(start + token_chunk_size, flat_draft.shape[0])
            tile_normalizers = _tile_log_normalizers(
                flat_draft[start:end],
                flat_target[start:end],
                tp_group,
            )
            target_probs, draft_probs = _tile_distributions(
                flat_draft[start:end],
                flat_target[start:end],
                tile_normalizers,
            )
            label_logits = _local_label_logits(
                flat_draft[start:end].float(),
                flat_labels[start:end],
                vocab_start=vocab_start,
                global_vocab_size=global_vocab_size,
                tp_group=tp_group,
            )
            ce_rows = tile_normalizers[:, 1] - label_logits
            tv_rows = 0.5 * target_probs.sub(draft_probs).abs().sum(dim=-1)
            if tp_group is not None:
                torch.distributed.all_reduce(
                    tv_rows,
                    op=torch.distributed.ReduceOp.SUM,
                    group=tp_group,
                )
            predicted_tokens = _global_argmax(
                flat_draft[start:end].float(),
                vocab_start=vocab_start,
                global_vocab_size=global_vocab_size,
                tp_group=tp_group,
            )
            flat_correct[start:end].copy_(
                predicted_tokens.eq(flat_labels[start:end]).float()
            )
            ce_numerators.scatter_add_(
                0,
                flat_bins[start:end],
                ce_rows.mul(flat_mask[start:end]),
            )
            tv_numerators.scatter_add_(
                0,
                flat_bins[start:end],
                tv_rows.mul(flat_mask[start:end]),
            )
            log_normalizers[start:end].copy_(tile_normalizers)

        ctx.save_for_backward(
            draft_logits,
            target_logits,
            hard_labels,
            valid_mask,
            slot_bins,
            log_normalizers,
        )
        # pyrefly: ignore[implicitly-defined-attribute]
        ctx.token_chunk_size = token_chunk_size
        ctx.vocab_start = vocab_start  # pyrefly: ignore[implicitly-defined-attribute]
        ctx.mark_non_differentiable(verifier_correct)
        return ce_numerators, tv_numerators, verifier_correct

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        grad_ce: torch.Tensor,
        grad_tv: torch.Tensor,
        _grad_correct: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
        (
            draft_logits,
            target_logits,
            hard_labels,
            valid_mask,
            slot_bins,
            log_normalizers,
        ) = ctx.saved_tensors
        local_vocab_size = draft_logits.shape[-1]
        flat_draft = draft_logits.reshape(-1, local_vocab_size)
        flat_target = target_logits.reshape(-1, local_vocab_size)
        flat_labels = hard_labels.reshape(-1)
        flat_mask = valid_mask.reshape(-1).float()
        flat_bins = slot_bins.reshape(-1)
        flat_gradient = torch.empty_like(flat_draft)

        for start in range(0, flat_draft.shape[0], ctx.token_chunk_size):
            end = min(start + ctx.token_chunk_size, flat_draft.shape[0])
            target_probs, draft_probs = _tile_distributions(
                flat_draft[start:end],
                flat_target[start:end],
                log_normalizers[start:end],
            )
            ce_gradient = draft_probs.clone()
            tile_labels = flat_labels[start:end]
            owns_label = tile_labels.ge(ctx.vocab_start) & tile_labels.lt(
                ctx.vocab_start + local_vocab_size
            )
            local_labels = tile_labels.sub(ctx.vocab_start).clamp(
                0, local_vocab_size - 1
            )
            ce_gradient.scatter_add_(
                -1,
                local_labels.unsqueeze(-1),
                -owns_label.to(dtype=ce_gradient.dtype).unsqueeze(-1),
            )

            probability_gradient = 0.5 * torch.sign(draft_probs - target_probs)
            tv_gradient = draft_probs * (
                probability_gradient
                - (probability_gradient * draft_probs).sum(dim=-1, keepdim=True)
            )
            ce_scale = grad_ce.index_select(0, flat_bins[start:end])
            tv_scale = grad_tv.index_select(0, flat_bins[start:end])
            tile_scale = flat_mask[start:end].unsqueeze(-1)
            tile_gradient = (
                ce_gradient * ce_scale.unsqueeze(-1)
                + tv_gradient * tv_scale.unsqueeze(-1)
            ) * tile_scale
            flat_gradient[start:end].copy_(tile_gradient.to(flat_gradient.dtype))

        return (
            flat_gradient.reshape_as(draft_logits),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _validate_inputs(
    *,
    target_logits: torch.Tensor,
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    confidence_logits: torch.Tensor | None,
    hard_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    slot_bins: torch.Tensor,
    loss_weights: tuple[float, float, float],
    token_chunk_size: int,
) -> None:
    if target_logits.ndim != 3 or target_logits.shape[-1] == 0:
        raise ValueError("DSpark logits must have shape [blocks, slots, vocab]")
    if (
        target_logits.shape != base_logits.shape
        or base_logits.shape != markov_bias.shape
    ):
        raise ValueError("target, base, and Markov logits must have matching shapes")
    slot_shape = target_logits.shape[:-1]
    if hard_labels.shape != slot_shape or valid_mask.shape != slot_shape:
        raise ValueError("hard labels and valid mask must match DSpark slots")
    if slot_bins.shape != slot_shape:
        raise ValueError("slot bins must match DSpark slots")
    if confidence_logits is not None and confidence_logits.shape != slot_shape:
        raise ValueError("confidence logits must match DSpark slots")
    if hard_labels.dtype != torch.long:
        raise TypeError("hard_labels must use torch.long")
    if slot_bins.dtype != torch.long:
        raise TypeError("slot_bins must use torch.long")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if not (
        target_logits.is_floating_point()
        and base_logits.is_floating_point()
        and markov_bias.is_floating_point()
        and (confidence_logits is None or confidence_logits.is_floating_point())
    ):
        raise TypeError("DSpark logits must use floating dtypes")
    if base_logits.dtype != markov_bias.dtype:
        raise ValueError("base logits and Markov bias must have the same dtype")
    devices = {
        target_logits.device,
        base_logits.device,
        markov_bias.device,
        hard_labels.device,
        valid_mask.device,
        slot_bins.device,
    }
    if confidence_logits is not None:
        devices.add(confidence_logits.device)
    if len(devices) != 1:
        raise ValueError("all DSpark objective inputs must share a device")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    if len(loss_weights) != 3 or any(
        not math.isfinite(weight) or weight < 0 for weight in loss_weights
    ):
        raise ValueError("loss_weights must contain three finite nonnegative values")
    if confidence_logits is None and loss_weights[2] != 0:
        raise ValueError("confidence weight must be zero when confidence is disabled")


def dspark_tiled_objective(
    *,
    target_logits: torch.Tensor,
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    confidence_logits: torch.Tensor | None,
    hard_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    slot_bins: torch.Tensor,
    loss_weights: tuple[float, float, float],
    token_chunk_size: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> DSparkObjectiveStats:
    """Compute raw per-slot DSpark CE, TV, confidence, and combined bins."""
    _validate_inputs(
        target_logits=target_logits,
        base_logits=base_logits,
        markov_bias=markov_bias,
        confidence_logits=confidence_logits,
        hard_labels=hard_labels,
        valid_mask=valid_mask,
        slot_bins=slot_bins,
        loss_weights=loss_weights,
        token_chunk_size=token_chunk_size,
    )
    num_bins = target_logits.shape[1]
    safe_labels = torch.where(valid_mask, hard_labels, 0)
    safe_target = torch.where(
        valid_mask.unsqueeze(-1),
        target_logits.detach(),
        torch.zeros_like(target_logits),
    )
    safe_base = torch.where(
        valid_mask.unsqueeze(-1),
        base_logits,
        torch.zeros_like(base_logits),
    )
    safe_markov = torch.where(
        valid_mask.unsqueeze(-1),
        markov_bias,
        torch.zeros_like(markov_bias),
    )
    corrected_logits = safe_base + safe_markov
    ce_numerators, tv_numerators, verifier_correct = _TiledHardCEAndTV.apply(
        corrected_logits,
        safe_target,
        safe_labels,
        valid_mask,
        slot_bins,
        num_bins,
        token_chunk_size,
        tp_group,
    )
    counts = torch.zeros(
        num_bins,
        dtype=torch.float32,
        device=valid_mask.device,
    )
    counts.scatter_add_(
        0,
        slot_bins.reshape(-1),
        valid_mask.reshape(-1).float(),
    )
    counts = counts.detach()

    confidence_counts = counts
    if confidence_logits is None:
        confidence_numerators = torch.zeros_like(ce_numerators)
        confidence_counts = torch.zeros_like(counts)
    else:
        safe_confidence = torch.where(
            valid_mask,
            confidence_logits.float(),
            torch.zeros_like(confidence_logits, dtype=torch.float32),
        )
        confidence_rows = F.binary_cross_entropy_with_logits(
            safe_confidence,
            verifier_correct,
            reduction="none",
        )
        confidence_numerators = torch.zeros_like(ce_numerators)
        confidence_numerators.scatter_add_(
            0,
            slot_bins.reshape(-1),
            confidence_rows.reshape(-1) * valid_mask.reshape(-1).float(),
        )

    ce = DSparkLossBins(numerators=ce_numerators, counts=counts)
    tv = DSparkLossBins(numerators=tv_numerators, counts=counts)
    confidence = DSparkLossBins(
        numerators=confidence_numerators,
        counts=confidence_counts,
    )
    combined_numerators = (
        loss_weights[0] * ce_numerators
        + loss_weights[1] * tv_numerators
        + loss_weights[2] * confidence_numerators
    )
    combined = DSparkLossBins(numerators=combined_numerators, counts=counts)
    return DSparkObjectiveStats(
        ce=ce,
        tv=tv,
        confidence=confidence,
        combined=combined,
    )


__all__ = ["DSparkLossBins", "DSparkObjectiveStats", "dspark_tiled_objective"]
