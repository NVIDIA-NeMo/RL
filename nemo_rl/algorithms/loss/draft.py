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

import struct
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


def _float64_bits(value: float) -> int:
    """Encode a Python float as its signed IEEE-754 bit pattern."""
    return struct.unpack("!q", struct.pack("!d", value))[0]


def _tp_assert_projected_metadata_agreement(
    *,
    tp_group: torch.distributed.ProcessGroup | None,
    reference: torch.Tensor,
    tensors: tuple[tuple[str, torch.Tensor | None], ...],
    scalars: tuple[tuple[str, int], ...],
    exact_tensors: tuple[tuple[str, torch.Tensor | None], ...],
) -> None:
    """Fail all TP ranks together before rank-local validation or CE collectives."""
    if tp_group is None:
        return

    dtype_codes = {
        dtype: index
        for index, dtype in enumerate(
            (
                torch.bool,
                torch.int32,
                torch.int64,
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            )
        )
    }
    header_values: list[int] = []
    for _, tensor in tensors:
        if tensor is None:
            header_values.extend((0, 0, 0, 0, 0, 0, -1, 0))
            continue
        shape = (*tensor.shape[:4], *(0 for _ in range(4 - min(tensor.ndim, 4))))
        header_values.extend(
            (
                1,
                tensor.ndim,
                *shape,
                dtype_codes.get(tensor.dtype, -1),
                int(tensor.device == reference.device),
            )
        )
    header_values.extend(value for _, value in scalars)
    header = torch.tensor(header_values, dtype=torch.int64, device=reference.device)
    world_size = torch.distributed.get_world_size(tp_group)
    gathered_headers = [torch.empty_like(header) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_headers, header, group=tp_group)
    if any(
        not torch.equal(gathered_headers[0], other) for other in gathered_headers[1:]
    ):
        raise ValueError("TP ranks disagree on projected soft-CE structure or scalars.")
    if any(
        tensor is not None and tensor.device != reference.device
        for _, tensor in tensors
    ):
        raise ValueError(
            "projected soft-CE tensors must share a device on every TP rank."
        )

    source_rank = torch.distributed.get_global_rank(tp_group, 0)
    for name, tensor in exact_tensors:
        if tensor is None:
            continue
        source_value = tensor.detach().contiguous().clone()
        torch.distributed.broadcast(source_value, src=source_rank, group=tp_group)
        mismatch = torch.logical_not(torch.eq(tensor, source_value).all()).to(
            torch.int64
        )
        torch.distributed.all_reduce(
            mismatch,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
        if mismatch.item():
            raise ValueError(f"TP ranks disagree on projected soft-CE {name}.")


def _cached_tile_distributions_and_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_shifted = student_logits.to(dtype=torch.float32, copy=True)
    teacher_probs = teacher_logits.to(dtype=torch.float32, copy=True)
    maxima = torch.stack(
        (teacher_probs.amax(dim=-1), student_shifted.amax(dim=-1)),
        dim=-1,
    )
    if tp_group is not None:
        torch.distributed.all_reduce(
            maxima,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
    teacher_probs.sub_(maxima[:, :1]).exp_()
    student_shifted.sub_(maxima[:, 1:])
    local_teacher_expectation = torch.einsum(
        "tv,tv->t",
        teacher_probs,
        student_shifted,
    )
    student_probs = student_shifted.exp_()
    reductions = torch.stack(
        (
            teacher_probs.sum(dim=-1),
            student_probs.sum(dim=-1),
            local_teacher_expectation,
        ),
        dim=-1,
    )
    if tp_group is not None:
        torch.distributed.all_reduce(
            reductions,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
    teacher_probs.div_(reductions[:, :1])
    student_probs.div_(reductions[:, 1:2])
    cross_entropy = reductions[:, 1].log() - reductions[:, 2].div_(reductions[:, 0])
    return teacher_probs, student_probs, cross_entropy


class _CachedVocabParallelSoftCE(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        bin_ids: torch.Tensor,
        num_bins: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        vocab_size = student_logits.shape[-1]
        flat_student = student_logits.reshape(-1, vocab_size)
        flat_teacher = teacher_logits.reshape(-1, vocab_size)
        teacher_probs, student_probs, cross_entropy = (
            _cached_tile_distributions_and_cross_entropy(
                flat_student,
                flat_teacher,
                tp_group,
            )
        )
        numerators = torch.zeros(
            num_bins,
            dtype=torch.float32,
            device=student_logits.device,
        )
        numerators.scatter_add_(
            0,
            bin_ids.reshape(-1),
            cross_entropy.mul_(mask.reshape(-1).float()),
        )
        ctx.save_for_backward(
            teacher_probs.reshape_as(teacher_logits),
            student_probs.reshape_as(student_logits),
            mask,
            bin_ids,
        )
        # pyrefly: ignore[implicitly-defined-attribute]
        ctx.student_dtype = student_logits.dtype
        return numerators

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        (grad_numerators,) = grad_outputs
        teacher_probs, student_probs, mask, bin_ids = ctx.saved_tensors
        row_scale = (
            grad_numerators.index_select(0, bin_ids.reshape(-1))
            * mask.reshape(-1).float()
        )
        gradient = student_probs.sub(teacher_probs)
        gradient.mul_(row_scale.reshape(*mask.shape, 1))
        return (
            gradient.to(dtype=ctx.student_dtype),
            None,
            None,
            None,
            None,
            None,
        )


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


class _SumTensorParallelHiddenGradient(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        student_hidden: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        ctx.tp_group = tp_group  # pyrefly: ignore[implicitly-defined-attribute]
        return student_hidden

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        hidden_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        if ctx.tp_group is not None:
            torch.distributed.all_reduce(
                hidden_gradient,
                op=torch.distributed.ReduceOp.SUM,
                group=ctx.tp_group,
            )
        return hidden_gradient, None


class _StreamingProjectedVocabParallelSoftCE(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        student_hidden: torch.Tensor,
        output_weight: torch.Tensor,
        selected_teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        bin_ids: torch.Tensor,
        num_bins: int,
        token_chunk_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        hidden_size = student_hidden.shape[-1]
        flat_hidden = student_hidden.reshape(-1, hidden_size)
        flat_teacher = selected_teacher_logits.reshape(
            -1, selected_teacher_logits.shape[-1]
        )
        flat_mask = mask.reshape(-1).float()
        flat_bin_ids = bin_ids.reshape(-1)
        numerators = torch.zeros(
            num_bins,
            dtype=torch.float32,
            device=student_hidden.device,
        )
        saved_log_normalizers = torch.empty(
            (flat_hidden.shape[0], 2),
            dtype=torch.float32,
            device=student_hidden.device,
        )

        for start in range(0, flat_hidden.shape[0], token_chunk_size):
            end = min(start + token_chunk_size, flat_hidden.shape[0])
            student_logits = flat_hidden[start:end] @ output_weight.T
            selected_teacher = flat_teacher[start:end]
            log_normalizers = _tile_log_normalizers(
                student_logits,
                selected_teacher,
                tp_group,
            )
            teacher_probs = (selected_teacher.float() - log_normalizers[:, :1]).exp()
            local_teacher_expectation = torch.einsum(
                "tv,tv->t",
                teacher_probs,
                student_logits.float(),
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
            student_hidden,
            output_weight,
            selected_teacher_logits,
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
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
        (grad_numerators,) = grad_outputs
        (
            student_hidden,
            output_weight,
            selected_teacher_logits,
            mask,
            bin_ids,
            log_normalizers,
        ) = ctx.saved_tensors
        hidden_size = student_hidden.shape[-1]
        flat_hidden = student_hidden.reshape(-1, hidden_size)
        flat_teacher = selected_teacher_logits.reshape(
            -1, selected_teacher_logits.shape[-1]
        )
        flat_mask = mask.reshape(-1).float()
        flat_bin_ids = bin_ids.reshape(-1)
        flat_hidden_gradient = torch.empty_like(flat_hidden, dtype=torch.float32)
        output_weight_fp32 = output_weight.detach().float()

        for start in range(0, flat_hidden.shape[0], ctx.token_chunk_size):
            end = min(start + ctx.token_chunk_size, flat_hidden.shape[0])
            student_logits = flat_hidden[start:end] @ output_weight.T
            selected_teacher = flat_teacher[start:end]
            teacher_probs, student_probs = _tile_distributions(
                student_logits,
                selected_teacher,
                log_normalizers[start:end],
            )
            row_scale = (
                grad_numerators.index_select(0, flat_bin_ids[start:end])
                * flat_mask[start:end]
            )
            logits_gradient = student_probs.sub_(teacher_probs)
            logits_gradient.mul_(row_scale.unsqueeze(-1))
            flat_hidden_gradient[start:end].copy_(logits_gradient @ output_weight_fp32)

        if ctx.tp_group is not None:
            torch.distributed.all_reduce(
                flat_hidden_gradient,
                op=torch.distributed.ReduceOp.SUM,
                group=ctx.tp_group,
            )
        return (
            flat_hidden_gradient.reshape_as(student_hidden).to(
                dtype=student_hidden.dtype
            ),
            None,
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

    num_tokens = student_logits.numel() // student_logits.shape[-1]
    if num_tokens <= token_chunk_size:
        numerators = _CachedVocabParallelSoftCE.apply(
            student_logits,
            teacher_logits,
            mask,
            bin_ids,
            num_bins,
            tp_group,
        )
    else:
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


def projected_streaming_vocab_parallel_soft_ce(
    *,
    student_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    selected_teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    token_chunk_size: int,
    tp_group: torch.distributed.ProcessGroup | None,
    bin_ids: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> DraftLossStats:
    """Project hidden states against preselected teacher rows with TP-safe metadata.

    TP ranks agree on a fixed structural header and exact mask/bin/weight metadata
    before validation or loss collectives, so malformed rank-local inputs fail together.
    """
    _tp_assert_projected_metadata_agreement(
        tp_group=tp_group,
        reference=student_hidden,
        tensors=(
            ("student_hidden", student_hidden),
            ("output_weight", output_weight),
            ("selected_teacher_logits", selected_teacher_logits),
            ("mask", mask),
            ("bin_ids", bin_ids),
            ("weights", weights),
        ),
        scalars=(("token_chunk_size", token_chunk_size),),
        exact_tensors=(("mask", mask), ("bin_ids", bin_ids), ("weights", weights)),
    )
    if student_hidden.ndim < 2 or student_hidden.numel() == 0:
        raise ValueError(
            "student_hidden must contain at least one token and one hidden element, "
            f"got {student_hidden.shape}."
        )
    if output_weight.ndim != 2 or output_weight.numel() == 0:
        raise ValueError(
            "output_weight must be a nonempty vocabulary-by-hidden matrix, "
            f"got {output_weight.shape}."
        )
    if output_weight.shape[1] != student_hidden.shape[-1]:
        raise ValueError(
            "output_weight hidden size must match student_hidden, "
            f"got {output_weight.shape[1]} and {student_hidden.shape[-1]}."
        )
    if selected_teacher_logits.ndim < 2 or selected_teacher_logits.numel() == 0:
        raise ValueError(
            "selected_teacher_logits must contain at least one row and vocabulary "
            f"element, got {selected_teacher_logits.shape}."
        )
    expected_teacher_shape = (*student_hidden.shape[:-1], output_weight.shape[0])
    if selected_teacher_logits.shape != expected_teacher_shape:
        raise ValueError(
            "selected_teacher_logits must match the student token dimensions and "
            f"local vocabulary, got {selected_teacher_logits.shape} and "
            f"{expected_teacher_shape}."
        )
    if mask.shape != student_hidden.shape[:-1]:
        raise ValueError(
            "mask must match the non-hidden student dimensions, "
            f"got {mask.shape} and {student_hidden.shape[:-1]}."
        )
    if not (
        student_hidden.device
        == output_weight.device
        == selected_teacher_logits.device
        == mask.device
    ):
        raise ValueError(
            "student_hidden, output_weight, selected_teacher_logits, and mask must "
            "share a device."
        )
    if not (
        student_hidden.is_floating_point()
        and output_weight.is_floating_point()
        and selected_teacher_logits.is_floating_point()
    ):
        raise TypeError(
            "student_hidden, output_weight, and selected_teacher_logits must be "
            "floating point."
        )
    if student_hidden.dtype != output_weight.dtype:
        raise ValueError(
            "student_hidden and output_weight must have the same dtype, "
            f"got {student_hidden.dtype} and {output_weight.dtype}."
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

    num_tokens = student_hidden.numel() // student_hidden.shape[-1]
    if num_tokens <= token_chunk_size:
        vocab_parallel_hidden = _SumTensorParallelHiddenGradient.apply(
            student_hidden,
            tp_group,
        )
        student_logits = (
            vocab_parallel_hidden.reshape(-1, student_hidden.shape[-1])
            @ output_weight.detach().T
        )
        return streaming_vocab_parallel_soft_ce(
            student_logits=student_logits.reshape(
                *student_hidden.shape[:-1],
                output_weight.shape[0],
            ),
            teacher_logits=selected_teacher_logits.detach(),
            mask=mask,
            bin_ids=bin_ids,
            weights=weights,
            token_chunk_size=token_chunk_size,
            tp_group=tp_group,
        )
    numerators = _StreamingProjectedVocabParallelSoftCE.apply(
        student_hidden,
        output_weight,
        selected_teacher_logits.detach(),
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


def dflash_projected_vocab_parallel_soft_ce(
    *,
    draft_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    teacher_logits: torch.Tensor,
    sample_rows: torch.Tensor,
    label_positions: torch.Tensor,
    loss_mask: torch.Tensor,
    position_decay: float,
    token_chunk_size: int,
    tp_group: torch.distributed.ProcessGroup | None,
) -> DraftLossStats:
    """Map DFlash block slots to the live target head and indexed teacher rows."""
    _tp_assert_projected_metadata_agreement(
        tp_group=tp_group,
        reference=draft_hidden,
        tensors=(
            ("draft_hidden", draft_hidden),
            ("output_weight", output_weight),
            ("teacher_logits", teacher_logits),
            ("sample_rows", sample_rows),
            ("label_positions", label_positions),
            ("loss_mask", loss_mask),
        ),
        scalars=(
            ("token_chunk_size", token_chunk_size),
            ("position_decay", _float64_bits(position_decay)),
        ),
        exact_tensors=(
            ("sample_rows", sample_rows),
            ("label_positions", label_positions),
            ("loss_mask", loss_mask),
        ),
    )
    if draft_hidden.ndim != 3 or draft_hidden.shape[1] < 2:
        raise ValueError(
            "draft_hidden must have shape [blocks, gamma + 1, hidden] with "
            f"gamma positive, got {draft_hidden.shape}."
        )
    if teacher_logits.ndim != 3:
        raise ValueError(
            "teacher_logits must have shape [batch, sequence, local_vocab], "
            f"got {teacher_logits.shape}."
        )
    block_shape = draft_hidden.shape[:-1]
    num_blocks, block_size = block_shape
    if sample_rows.shape != (num_blocks,):
        raise ValueError(
            f"sample_rows must have shape {(num_blocks,)}, got {sample_rows.shape}."
        )
    if label_positions.shape != block_shape:
        raise ValueError(
            "label_positions must match DFlash block slots, "
            f"got {label_positions.shape} and {block_shape}."
        )
    if loss_mask.shape != block_shape:
        raise ValueError(
            "loss_mask must match DFlash block slots, "
            f"got {loss_mask.shape} and {block_shape}."
        )
    if sample_rows.dtype != torch.long or label_positions.dtype != torch.long:
        raise TypeError("sample_rows and label_positions must use torch.long.")
    if loss_mask.dtype != torch.bool:
        raise TypeError(f"loss_mask must be boolean, got {loss_mask.dtype}.")
    if not (
        draft_hidden.device
        == teacher_logits.device
        == sample_rows.device
        == label_positions.device
        == loss_mask.device
    ):
        raise ValueError(
            "draft_hidden, teacher_logits, sample_rows, label_positions, and "
            "loss_mask must share a device."
        )
    if not 0.0 < position_decay <= 1.0:
        raise ValueError(f"position_decay must be in (0, 1], got {position_decay}.")

    effective_loss_mask = loss_mask.clone()
    effective_loss_mask[:, 0] = False
    batch_size = teacher_logits.shape[0]
    sequence_length = teacher_logits.shape[1]
    teacher_positions = label_positions - 1
    teacher_row_indices = sample_rows[:, None] * sequence_length + teacher_positions
    valid_coordinates = (
        (sample_rows[:, None].ge(0) & sample_rows[:, None].lt(batch_size))
        & teacher_positions.ge(0)
        & teacher_positions.lt(sequence_length)
    )
    teacher_row_indices = torch.where(
        valid_coordinates,
        teacher_row_indices,
        teacher_row_indices.new_full((), batch_size * sequence_length),
    )
    teacher_row_indices = torch.where(
        effective_loss_mask,
        teacher_row_indices,
        teacher_row_indices.new_zeros(()),
    )
    gamma = block_size - 1
    slot_offsets = torch.arange(block_size, device=draft_hidden.device)
    bin_ids = slot_offsets.sub(1).clamp_min_(0).expand(block_shape)
    weights = torch.pow(
        torch.tensor(position_decay, dtype=torch.float32, device=draft_hidden.device),
        torch.arange(gamma, dtype=torch.float32, device=draft_hidden.device),
    )
    if num_blocks == 0:
        if output_weight.ndim != 2 or output_weight.shape[1] != draft_hidden.shape[-1]:
            raise ValueError(
                "output_weight must be a vocabulary-by-hidden matrix matching "
                f"draft_hidden, got {output_weight.shape} and {draft_hidden.shape}."
            )
        if output_weight.device != draft_hidden.device:
            raise ValueError("output_weight and draft_hidden must share a device.")
        if output_weight.dtype != draft_hidden.dtype:
            raise ValueError("output_weight and draft_hidden must share a dtype.")
        zero = draft_hidden.sum(dtype=torch.float32)
        return DraftLossStats(
            numerators=zero.expand(gamma),
            counts=torch.zeros(gamma, dtype=torch.float32, device=draft_hidden.device),
            weights=weights,
        )
    selected_teacher_logits = (
        teacher_logits.detach()
        .reshape(-1, teacher_logits.shape[-1])
        .index_select(0, teacher_row_indices.reshape(-1))
        .reshape(*block_shape, teacher_logits.shape[-1])
    )
    return projected_streaming_vocab_parallel_soft_ce(
        student_hidden=draft_hidden,
        output_weight=output_weight,
        selected_teacher_logits=selected_teacher_logits,
        mask=effective_loss_mask,
        bin_ids=bin_ids,
        weights=weights,
        token_chunk_size=token_chunk_size,
        tp_group=tp_group,
    )
