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

"""Token-tiled hard-CE, total-variation, and confidence objectives for DSpark."""

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


_VOCAB_GRADIENT_CHUNK_SIZE = 4096


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
        "complex32",
        "complex64",
        "complex128",
    )
    return next(
        (
            index
            for index, name in enumerate(dtype_names)
            if dtype == getattr(torch, name, None)
        ),
        -1,
    )


def _tensor_contract(tensor: torch.Tensor | None) -> list[int]:
    if tensor is None:
        return [-1, -1, -1, -1, -1, -1]
    padded_shape = [*tensor.shape[:4], *([-1] * max(0, 4 - tensor.ndim))]
    return [tensor.ndim, *padded_shape, _dtype_code(tensor.dtype)]


def _validate_tp_structural_agreement(
    *,
    tensors: tuple[torch.Tensor | None, ...],
    loss_weights: tuple[float, float, float],
    token_chunk_size: int,
    device: torch.device,
    tp_group: torch.distributed.ProcessGroup | None,
) -> None:
    if tp_group is None:
        return
    contract_values = [len(loss_weights), token_chunk_size]
    for tensor in tensors:
        contract_values.extend(_tensor_contract(tensor))
    contract = torch.tensor(contract_values, dtype=torch.int64, device=device)
    contract_minimum = contract.clone()
    contract_maximum = contract.clone()
    torch.distributed.all_reduce(
        contract_minimum,
        op=torch.distributed.ReduceOp.MIN,
        group=tp_group,
    )
    torch.distributed.all_reduce(
        contract_maximum,
        op=torch.distributed.ReduceOp.MAX,
        group=tp_group,
    )
    if not torch.equal(contract_minimum, contract_maximum):
        raise ValueError(
            "tensor-parallel ranks must agree on DSpark shapes, dtypes, and chunk size"
        )
    if len(loss_weights) == 3:
        weights = torch.tensor(loss_weights, dtype=torch.float64, device=device)
        weights_minimum = weights.clone()
        weights_maximum = weights.clone()
        torch.distributed.all_reduce(
            weights_minimum,
            op=torch.distributed.ReduceOp.MIN,
            group=tp_group,
        )
        torch.distributed.all_reduce(
            weights_maximum,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
        if not torch.equal(weights_minimum, weights_maximum):
            raise ValueError("tensor-parallel ranks must agree on DSpark loss weights")


def _raise_if_tp_any(
    local_invalid: bool,
    *,
    message: str,
    device: torch.device,
    tp_group: torch.distributed.ProcessGroup | None,
) -> None:
    invalid = torch.tensor(local_invalid, dtype=torch.int32, device=device)
    if tp_group is not None:
        torch.distributed.all_reduce(
            invalid,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
    if bool(invalid):
        raise ValueError(message)


def _validate_tp_metadata_agreement(
    *,
    previous_token_ids: torch.Tensor,
    hard_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    slot_bins: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None,
) -> None:
    if tp_group is None or valid_mask.numel() == 0:
        return
    canonical_metadata = torch.stack(
        (
            valid_mask.to(dtype=torch.int64),
            torch.where(valid_mask, hard_labels, 0),
            torch.where(valid_mask, previous_token_ids, 0),
            slot_bins,
        )
    )
    metadata_minimum = canonical_metadata.clone()
    metadata_maximum = canonical_metadata.clone()
    torch.distributed.all_reduce(
        metadata_minimum,
        op=torch.distributed.ReduceOp.MIN,
        group=tp_group,
    )
    torch.distributed.all_reduce(
        metadata_maximum,
        op=torch.distributed.ReduceOp.MAX,
        group=tp_group,
    )
    if not torch.equal(metadata_minimum, metadata_maximum):
        raise ValueError("tensor-parallel ranks must agree on DSpark token metadata")


def _compact_retained_view(tensor: torch.Tensor) -> torch.Tensor:
    compact_nbytes = tensor.numel() * tensor.element_size()
    if tensor.untyped_storage().nbytes() > compact_nbytes:
        return tensor.clone()
    return tensor


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
        normalization_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize a local differentiable numerator by externally reduced counts."""
        if normalization_counts.shape != self.counts.shape:
            raise ValueError("normalization_counts must match the local count bins")
        denominator = (
            normalization_counts.detach()
            .to(device=self.numerators.device, dtype=torch.float32)
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


def _local_label_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_start: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    local_vocab_size = logits.shape[-1]
    owns_label = labels.ge(vocab_start) & labels.lt(vocab_start + local_vocab_size)
    local_labels = labels.sub(vocab_start).clamp(0, local_vocab_size - 1)
    selected = logits.gather(-1, local_labels.unsqueeze(-1)).squeeze(-1)
    selected = torch.where(owns_label, selected, 0.0)
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


class _TiledProjectedHardCEAndTV(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        draft_hidden: torch.Tensor,
        output_weight: torch.Tensor,
        target_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        hard_labels: torch.Tensor,
        valid_mask: torch.Tensor,
        slot_bins: torch.Tensor,
        num_bins: int,
        token_chunk_size: int,
        vocab_start: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_size = draft_hidden.shape[-1]
        local_vocab_size = output_weight.shape[0]
        flat_hidden = draft_hidden.reshape(-1, hidden_size)
        flat_target = target_logits.reshape(-1, local_vocab_size)
        flat_previous_token_ids = previous_token_ids.reshape(-1)
        flat_labels = hard_labels.reshape(-1)
        flat_valid = valid_mask.reshape(-1)
        flat_mask = flat_valid.float()
        flat_bins = slot_bins.reshape(-1)
        ce_numerators = torch.zeros(
            num_bins, dtype=torch.float32, device=draft_hidden.device
        )
        tv_numerators = torch.zeros_like(ce_numerators)
        verifier_correct = torch.zeros_like(valid_mask, dtype=torch.float32)
        flat_correct = verifier_correct.reshape(-1)
        log_normalizers = torch.empty(
            (flat_hidden.shape[0], 2),
            dtype=torch.float32,
            device=draft_hidden.device,
        )
        global_vocab_size = markov_w1.shape[0]

        for start in range(0, flat_hidden.shape[0], token_chunk_size):
            end = min(start + token_chunk_size, flat_hidden.shape[0])
            tile_valid = flat_valid[start:end]
            tile_hidden = torch.where(
                tile_valid.unsqueeze(-1),
                flat_hidden[start:end],
                torch.zeros_like(flat_hidden[start:end]),
            )
            safe_previous_token_ids = torch.where(
                tile_valid,
                flat_previous_token_ids[start:end],
                torch.zeros_like(flat_previous_token_ids[start:end]),
            )
            previous_embeddings = F.embedding(safe_previous_token_ids, markov_w1)
            draft_logits = (
                tile_hidden @ output_weight.detach().T
                + previous_embeddings @ markov_w2.T
            )
            selected_target = torch.where(
                tile_valid.unsqueeze(-1),
                flat_target[start:end].detach(),
                torch.zeros_like(flat_target[start:end]),
            )
            tile_normalizers = _tile_log_normalizers(
                draft_logits,
                selected_target,
                tp_group,
            )
            target_probs, draft_probs = _tile_distributions(
                draft_logits,
                selected_target,
                tile_normalizers,
            )
            label_logits = _local_label_logits(
                draft_logits.float(),
                flat_labels[start:end],
                vocab_start=vocab_start,
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
                draft_logits.float(),
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
            draft_hidden,
            output_weight,
            target_logits,
            previous_token_ids,
            markov_w1,
            markov_w2,
            hard_labels,
            valid_mask,
            slot_bins,
            log_normalizers,
        )
        # pyrefly: ignore[implicitly-defined-attribute]
        ctx.token_chunk_size = token_chunk_size
        ctx.vocab_start = vocab_start  # pyrefly: ignore[implicitly-defined-attribute]
        ctx.tp_group = tp_group  # pyrefly: ignore[implicitly-defined-attribute]
        ctx.mark_non_differentiable(verifier_correct)
        return ce_numerators, tv_numerators, verifier_correct

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        grad_ce: torch.Tensor,
        grad_tv: torch.Tensor,
        _grad_correct: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        None,
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            draft_hidden,
            output_weight,
            target_logits,
            previous_token_ids,
            markov_w1,
            markov_w2,
            hard_labels,
            valid_mask,
            slot_bins,
            log_normalizers,
        ) = ctx.saved_tensors
        hidden_size = draft_hidden.shape[-1]
        local_vocab_size = output_weight.shape[0]
        markov_rank = markov_w1.shape[-1]
        flat_hidden = draft_hidden.reshape(-1, hidden_size)
        flat_target = target_logits.reshape(-1, local_vocab_size)
        flat_previous_token_ids = previous_token_ids.reshape(-1)
        flat_labels = hard_labels.reshape(-1)
        flat_valid = valid_mask.reshape(-1)
        flat_mask = flat_valid.float()
        flat_bins = slot_bins.reshape(-1)
        flat_hidden_gradient = torch.zeros_like(flat_hidden, dtype=torch.float32)
        flat_embedding_gradient = torch.zeros(
            (flat_hidden.shape[0], markov_rank),
            dtype=torch.float32,
            device=draft_hidden.device,
        )
        markov_w2_gradient = torch.zeros_like(markov_w2, dtype=torch.float32)

        for start in range(0, flat_hidden.shape[0], ctx.token_chunk_size):
            end = min(start + ctx.token_chunk_size, flat_hidden.shape[0])
            tile_valid = flat_valid[start:end]
            tile_hidden = torch.where(
                tile_valid.unsqueeze(-1),
                flat_hidden[start:end],
                torch.zeros_like(flat_hidden[start:end]),
            )
            safe_previous_token_ids = torch.where(
                tile_valid,
                flat_previous_token_ids[start:end],
                torch.zeros_like(flat_previous_token_ids[start:end]),
            )
            previous_embeddings = F.embedding(safe_previous_token_ids, markov_w1)
            draft_logits = (
                tile_hidden @ output_weight.detach().T
                + previous_embeddings @ markov_w2.T
            )
            selected_target = torch.where(
                tile_valid.unsqueeze(-1),
                flat_target[start:end].detach(),
                torch.zeros_like(flat_target[start:end]),
            )
            target_probs, draft_probs = _tile_distributions(
                draft_logits,
                selected_target,
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
            probability_expectation = (probability_gradient * draft_probs).sum(
                dim=-1, keepdim=True
            )
            if ctx.tp_group is not None:
                torch.distributed.all_reduce(
                    probability_expectation,
                    op=torch.distributed.ReduceOp.SUM,
                    group=ctx.tp_group,
                )
            tv_gradient = draft_probs * (probability_gradient - probability_expectation)
            ce_scale = grad_ce.index_select(0, flat_bins[start:end])
            tv_scale = grad_tv.index_select(0, flat_bins[start:end])
            tile_scale = flat_mask[start:end].unsqueeze(-1)
            logits_gradient = (
                ce_gradient * ce_scale.unsqueeze(-1)
                + tv_gradient * tv_scale.unsqueeze(-1)
            ) * tile_scale
            logits_gradient_fp32 = logits_gradient.float()
            tile_hidden_gradient = torch.zeros(
                (end - start, hidden_size),
                dtype=torch.float32,
                device=draft_hidden.device,
            )
            tile_embedding_gradient = torch.zeros(
                (end - start, markov_rank),
                dtype=torch.float32,
                device=draft_hidden.device,
            )
            previous_embeddings_fp32 = previous_embeddings.detach().float()
            for vocab_start in range(
                0,
                local_vocab_size,
                _VOCAB_GRADIENT_CHUNK_SIZE,
            ):
                vocab_end = min(
                    vocab_start + _VOCAB_GRADIENT_CHUNK_SIZE,
                    local_vocab_size,
                )
                gradient_chunk = logits_gradient_fp32[:, vocab_start:vocab_end]
                tile_hidden_gradient.addmm_(
                    gradient_chunk,
                    output_weight[vocab_start:vocab_end].detach().float(),
                )
                markov_w2_gradient[vocab_start:vocab_end].addmm_(
                    gradient_chunk.T,
                    previous_embeddings_fp32,
                )
                tile_embedding_gradient.addmm_(
                    gradient_chunk,
                    markov_w2[vocab_start:vocab_end].detach().float(),
                )
            flat_hidden_gradient[start:end].copy_(tile_hidden_gradient)
            flat_embedding_gradient[start:end].copy_(tile_embedding_gradient)

        if ctx.tp_group is not None and flat_hidden.shape[0] > 0:
            torch.distributed.all_reduce(
                flat_hidden_gradient,
                op=torch.distributed.ReduceOp.SUM,
                group=ctx.tp_group,
            )
            torch.distributed.all_reduce(
                flat_embedding_gradient,
                op=torch.distributed.ReduceOp.SUM,
                group=ctx.tp_group,
            )
        markov_w1_gradient = torch.zeros_like(markov_w1, dtype=torch.float32)
        if flat_hidden.shape[0] > 0:
            safe_previous_token_ids = torch.where(
                flat_valid,
                flat_previous_token_ids,
                torch.zeros_like(flat_previous_token_ids),
            )
            markov_w1_gradient.index_add_(
                0,
                safe_previous_token_ids,
                flat_embedding_gradient,
            )

        return (
            flat_hidden_gradient.reshape_as(draft_hidden).to(draft_hidden.dtype),
            None,
            None,
            None,
            markov_w1_gradient.to(markov_w1.dtype),
            markov_w2_gradient.to(markov_w2.dtype),
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
    draft_hidden: torch.Tensor,
    target_output_weight: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    previous_token_ids: torch.Tensor,
    confidence_logits: torch.Tensor | None,
    hard_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    slot_bins: torch.Tensor,
    loss_weights: tuple[float, float, float],
    token_chunk_size: int,
    vocab_start_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> None:
    _validate_tp_structural_agreement(
        tensors=(
            target_logits,
            draft_hidden,
            target_output_weight,
            markov_w1,
            markov_w2,
            previous_token_ids,
            confidence_logits,
            hard_labels,
            valid_mask,
            slot_bins,
        ),
        loss_weights=loss_weights,
        token_chunk_size=token_chunk_size,
        device=target_logits.device,
        tp_group=tp_group,
    )
    if target_logits.ndim != 3 or target_logits.shape[-1] == 0:
        raise ValueError("target_logits must have shape [blocks, slots, local_vocab]")
    if draft_hidden.ndim != 3 or draft_hidden.shape[-1] == 0:
        raise ValueError("draft_hidden must have shape [blocks, slots, hidden]")
    slot_shape = target_logits.shape[:-1]
    if draft_hidden.shape[:-1] != slot_shape:
        raise ValueError("target logits and draft hidden must describe the same slots")
    local_vocab_size = target_logits.shape[-1]
    hidden_size = draft_hidden.shape[-1]
    if target_output_weight.shape != (local_vocab_size, hidden_size):
        raise ValueError(
            "target_output_weight must match local vocabulary and hidden size"
        )
    if markov_w1.ndim != 2 or markov_w2.shape != (
        local_vocab_size,
        markov_w1.shape[-1],
    ):
        raise ValueError("Markov weights must be global-vocab W1 and local-vocab W2")
    global_vocab_size = markov_w1.shape[0]
    if global_vocab_size == 0 or markov_w1.shape[1] == 0:
        raise ValueError("Markov weights must be nonempty")
    if tp_group is None:
        expected_start = 0
        expected_global_vocab_size = local_vocab_size
    else:
        tp_rank = torch.distributed.get_rank(tp_group)
        tp_size = torch.distributed.get_world_size(tp_group)
        expected_start = tp_rank * local_vocab_size
        expected_global_vocab_size = tp_size * local_vocab_size
    _raise_if_tp_any(
        vocab_start_index != expected_start
        or global_vocab_size != expected_global_vocab_size,
        message="vocabulary shard must be an even rank-local TP partition",
        device=target_logits.device,
        tp_group=tp_group,
    )
    for name, tensor in (
        ("previous_token_ids", previous_token_ids),
        ("hard_labels", hard_labels),
        ("valid_mask", valid_mask),
        ("slot_bins", slot_bins),
    ):
        if tensor.shape != slot_shape:
            raise ValueError(f"{name} must match DSpark slots")
    if confidence_logits is not None and confidence_logits.shape != slot_shape:
        raise ValueError("confidence logits must match DSpark slots")
    if previous_token_ids.dtype != torch.long:
        raise TypeError("previous_token_ids must use torch.long")
    if hard_labels.dtype != torch.long:
        raise TypeError("hard_labels must use torch.long")
    if slot_bins.dtype != torch.long:
        raise TypeError("slot_bins must use torch.long")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    floating_inputs = (
        target_logits,
        draft_hidden,
        target_output_weight,
        markov_w1,
        markov_w2,
    )
    if not all(tensor.is_floating_point() for tensor in floating_inputs) or (
        confidence_logits is not None and not confidence_logits.is_floating_point()
    ):
        raise TypeError("DSpark model and objective tensors must use floating dtypes")
    if not (
        draft_hidden.dtype
        == target_output_weight.dtype
        == markov_w1.dtype
        == markov_w2.dtype
    ):
        raise ValueError(
            "draft hidden, live head, and Markov weights must share a dtype"
        )
    devices = {tensor.device for tensor in floating_inputs}
    devices.update(
        {
            previous_token_ids.device,
            hard_labels.device,
            valid_mask.device,
            slot_bins.device,
        }
    )
    if confidence_logits is not None:
        devices.add(confidence_logits.device)
    _raise_if_tp_any(
        len(devices) != 1,
        message="all DSpark objective inputs must share a device",
        device=target_logits.device,
        tp_group=tp_group,
    )
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    if len(loss_weights) != 3 or any(
        not math.isfinite(weight) or weight < 0 for weight in loss_weights
    ):
        raise ValueError("loss_weights must contain three finite nonnegative values")
    if confidence_logits is None and loss_weights[2] != 0:
        raise ValueError("confidence weight must be zero when confidence is disabled")
    _validate_tp_metadata_agreement(
        previous_token_ids=previous_token_ids,
        hard_labels=hard_labels,
        valid_mask=valid_mask,
        slot_bins=slot_bins,
        tp_group=tp_group,
    )

    invalid_metadata = torch.stack(
        (
            (
                valid_mask & (hard_labels.lt(0) | hard_labels.ge(global_vocab_size))
            ).any(),
            (
                valid_mask
                & (previous_token_ids.lt(0) | previous_token_ids.ge(global_vocab_size))
            ).any(),
            (slot_bins.lt(0) | slot_bins.ge(slot_shape[1])).any(),
        )
    ).to(dtype=torch.int32)
    if tp_group is not None:
        torch.distributed.all_reduce(
            invalid_metadata,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
    if bool(invalid_metadata[0]):
        raise ValueError("valid hard_labels must be inside the global vocabulary")
    if bool(invalid_metadata[1]):
        raise ValueError(
            "valid previous_token_ids must be inside the global vocabulary"
        )
    if bool(invalid_metadata[2]):
        raise ValueError("slot_bins must be inside the configured slot bins")


def dspark_tiled_objective(
    *,
    target_logits: torch.Tensor,
    draft_hidden: torch.Tensor,
    target_output_weight: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    previous_token_ids: torch.Tensor,
    confidence_logits: torch.Tensor | None,
    hard_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    slot_bins: torch.Tensor,
    loss_weights: tuple[float, float, float],
    token_chunk_size: int,
    vocab_start_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> DSparkObjectiveStats:
    """Project selected slots tile-by-tile and return raw DSpark objective bins."""
    _validate_inputs(
        target_logits=target_logits,
        draft_hidden=draft_hidden,
        target_output_weight=target_output_weight,
        markov_w1=markov_w1,
        markov_w2=markov_w2,
        previous_token_ids=previous_token_ids,
        confidence_logits=confidence_logits,
        hard_labels=hard_labels,
        valid_mask=valid_mask,
        slot_bins=slot_bins,
        loss_weights=loss_weights,
        token_chunk_size=token_chunk_size,
        vocab_start_index=vocab_start_index,
        tp_group=tp_group,
    )
    num_bins = target_logits.shape[1]
    compact_hidden = _compact_retained_view(draft_hidden)
    compact_target = _compact_retained_view(target_logits.detach())
    compact_previous_token_ids = _compact_retained_view(previous_token_ids)
    compact_hard_labels = _compact_retained_view(hard_labels)
    compact_valid_mask = _compact_retained_view(valid_mask)
    compact_slot_bins = _compact_retained_view(slot_bins)
    ce_numerators, tv_numerators, verifier_correct = _TiledProjectedHardCEAndTV.apply(
        compact_hidden,
        target_output_weight,
        compact_target,
        compact_previous_token_ids,
        markov_w1,
        markov_w2,
        compact_hard_labels,
        compact_valid_mask,
        compact_slot_bins,
        num_bins,
        token_chunk_size,
        vocab_start_index,
        tp_group,
    )
    counts = torch.zeros(num_bins, dtype=torch.float32, device=valid_mask.device)
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
