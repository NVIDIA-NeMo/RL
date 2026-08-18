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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class DraftLossStats:
    """Raw draft-loss reductions with one entry per pass or block slot."""

    numerators: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.numerators.ndim != 1:
            raise ValueError(
                f"draft loss statistics must be one-dimensional, got {self.numerators.shape}."
            )
        if not (self.numerators.shape == self.counts.shape == self.weights.shape):
            raise ValueError(
                "numerators, counts, and weights must have the same shape, "
                f"got {self.numerators.shape}, {self.counts.shape}, and {self.weights.shape}."
            )

    def normalized(
        self,
        *,
        normalization_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize local differentiable sums with externally reduced counts."""
        if normalization_counts.shape != self.counts.shape:
            raise ValueError(
                "normalization_counts must match counts, "
                f"got {normalization_counts.shape} and {self.counts.shape}."
            )
        counts = normalization_counts.detach().to(
            device=self.numerators.device,
            dtype=torch.float32,
        )
        denominator = (counts * self.weights).sum()
        return (self.numerators * self.weights).sum() / (denominator + 1e-8)


def _tile_log_normalizers(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    student_fp32 = student_logits.float()
    teacher_fp32 = teacher_logits.float()
    maxima = torch.stack(
        (teacher_fp32.amax(dim=-1), student_fp32.amax(dim=-1)),
        dim=-1,
    )
    if tp_group is not None:
        torch.distributed.all_reduce(
            maxima,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
    exp_sums = torch.stack(
        (
            (teacher_fp32 - maxima[:, :1]).exp().sum(dim=-1),
            (student_fp32 - maxima[:, 1:]).exp().sum(dim=-1),
        ),
        dim=-1,
    )
    if tp_group is not None:
        torch.distributed.all_reduce(
            exp_sums,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
    return maxima + exp_sums.log()


def _tile_distributions(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    log_normalizers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = (teacher_logits.float() - log_normalizers[:, :1]).exp()
    student_probs = (student_logits.float() - log_normalizers[:, 1:]).exp()
    return teacher_probs, student_probs


class _StreamingVocabParallelSoftCE(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        bin_ids: torch.Tensor,
        num_bins: int,
        token_chunk_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        vocab_size = student_logits.shape[-1]
        flat_student = student_logits.reshape(-1, vocab_size)
        flat_teacher = teacher_logits.reshape(-1, vocab_size)
        flat_mask = mask.reshape(-1).float()
        flat_bin_ids = bin_ids.reshape(-1)
        numerators = torch.zeros(
            num_bins,
            dtype=torch.float32,
            device=student_logits.device,
        )
        saved_log_normalizers = torch.empty(
            (flat_student.shape[0], 2),
            dtype=torch.float32,
            device=student_logits.device,
        )

        for start in range(0, flat_student.shape[0], token_chunk_size):
            end = min(start + token_chunk_size, flat_student.shape[0])
            log_normalizers = _tile_log_normalizers(
                flat_student[start:end],
                flat_teacher[start:end],
                tp_group,
            )
            teacher_probs = (
                flat_teacher[start:end].float() - log_normalizers[:, :1]
            ).exp()
            local_teacher_expectation = torch.einsum(
                "tv,tv->t",
                teacher_probs,
                flat_student[start:end].float(),
            )
            if tp_group is not None:
                torch.distributed.all_reduce(
                    local_teacher_expectation,
                    op=torch.distributed.ReduceOp.SUM,
                    group=tp_group,
                )
            cross_entropy = log_normalizers[:, 1] - local_teacher_expectation
            numerators.scatter_add_(
                0,
                flat_bin_ids[start:end],
                cross_entropy.mul_(flat_mask[start:end]),
            )
            saved_log_normalizers[start:end].copy_(log_normalizers)

        ctx.save_for_backward(
            student_logits,
            teacher_logits,
            mask,
            bin_ids,
            saved_log_normalizers,
        )
        # pyrefly: ignore[implicitly-defined-attribute]
        ctx.token_chunk_size = token_chunk_size
        ctx.tp_group = tp_group  # pyrefly: ignore[implicitly-defined-attribute]
        return numerators

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        (grad_numerators,) = grad_outputs
        student_logits, teacher_logits, mask, bin_ids, log_normalizers = (
            ctx.saved_tensors
        )
        vocab_size = student_logits.shape[-1]
        flat_student = student_logits.reshape(-1, vocab_size)
        flat_teacher = teacher_logits.reshape(-1, vocab_size)
        flat_mask = mask.reshape(-1).float()
        flat_bin_ids = bin_ids.reshape(-1)
        flat_gradient = torch.empty_like(flat_student)

        for start in range(0, flat_student.shape[0], ctx.token_chunk_size):
            end = min(start + ctx.token_chunk_size, flat_student.shape[0])
            teacher_probs, student_probs = _tile_distributions(
                flat_student[start:end],
                flat_teacher[start:end],
                log_normalizers[start:end],
            )
            row_scale = (
                grad_numerators.index_select(0, flat_bin_ids[start:end])
                * flat_mask[start:end]
            )
            tile_gradient = student_probs.sub_(teacher_probs)
            tile_gradient.mul_(row_scale.unsqueeze(-1))
            flat_gradient[start:end].copy_(tile_gradient)

        return (
            flat_gradient.reshape_as(student_logits),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def streaming_vocab_parallel_soft_ce(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    token_chunk_size: int,
    tp_group: torch.distributed.ProcessGroup | None,
    bin_ids: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> DraftLossStats:
    """Compute raw soft-CE bins while bounding FP32 vocab work to one token tile."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape, "
            f"got {student_logits.shape} and {teacher_logits.shape}."
        )
    if student_logits.ndim < 2 or student_logits.numel() == 0:
        raise ValueError(
            "logits must contain at least one token and one vocabulary element, "
            f"got {student_logits.shape}."
        )
    if mask.shape != student_logits.shape[:-1]:
        raise ValueError(
            "mask must match the non-vocabulary logits dimensions, "
            f"got {mask.shape} and {student_logits.shape[:-1]}."
        )
    if not (student_logits.device == teacher_logits.device == mask.device):
        raise ValueError(
            "student_logits, teacher_logits, and mask must share a device, "
            f"got {student_logits.device}, {teacher_logits.device}, and {mask.device}."
        )
    if token_chunk_size < 1:
        raise ValueError(f"token_chunk_size must be positive, got {token_chunk_size}.")
    if weights is None:
        weights = torch.ones(1, dtype=torch.float32, device=mask.device)
    elif weights.ndim != 1 or weights.shape[0] < 1:
        raise ValueError(f"weights must be a nonempty vector, got {weights.shape}.")
    num_bins = weights.shape[0]

    if bin_ids is None:
        if num_bins != 1:
            raise ValueError("bin_ids is required when weights contains multiple bins.")
        bin_ids = torch.zeros_like(mask, dtype=torch.long)
    elif bin_ids.shape != mask.shape:
        raise ValueError(
            f"bin_ids must match mask, got {bin_ids.shape} and {mask.shape}."
        )
    elif bin_ids.dtype != torch.long:
        raise TypeError(f"bin_ids must use torch.long, got {bin_ids.dtype}.")
    elif bin_ids.device != mask.device:
        raise ValueError(
            f"bin_ids and mask must share a device, got {bin_ids.device} and {mask.device}."
        )

    numerators = _StreamingVocabParallelSoftCE.apply(
        student_logits,
        teacher_logits,
        mask,
        bin_ids,
        num_bins,
        token_chunk_size,
        tp_group,
    )
    counts = torch.zeros(num_bins, dtype=torch.float32, device=mask.device)
    counts.scatter_add_(0, bin_ids.reshape(-1), mask.reshape(-1).float())
    return DraftLossStats(
        numerators=numerators,
        counts=counts.detach(),
        weights=weights.to(device=mask.device, dtype=torch.float32).detach(),
    )
