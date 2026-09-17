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

"""Sequence-packing adapter for cross-tokenizer distillation.

The generic sequence-packing loss wrapper assumes every tensor with a second
dimension uses the student's token axis. xToken batches deliberately contain
three independent axes instead: student tokens, teacher tokens, and alignment
pair slots. This adapter restores each packed student row to its own effective
padded width and crops only fields on the student-token axis.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import torch
import torch.distributed

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    _get_tokens_on_this_cp_rank,
    allgather_cp_sharded_tensor,
)

Tensor = TypeVar("Tensor", bound=torch.Tensor)


XTOKEN_LOGICAL_METRICS_KEY = "__xtoken_logical_metrics__"


_STUDENT_SEQUENCE_KEYS = frozenset(
    {
        "input_ids",
        "token_mask",
        "kd_token_mask",
    }
)
_STUDENT_ALIGNMENT_SEQUENCE_SUFFIXES = (
    "_student_chunk_id",
    "_student_exact_partition_mask",
)


@dataclass(frozen=True)
class LockstepBinGeometry:
    """The controller-owned geometry for one selected physical bin."""

    bin_index: int
    batch_item_ids: tuple[int, ...]
    raw_cu_seqlens: tuple[int, ...]
    padded_cu_seqlens: tuple[int, ...]
    physical_tokens: int


def resolve_lockstep_bin_geometry(
    data: BatchedDataDict[Any],
    *,
    input_lengths: torch.Tensor,
) -> LockstepBinGeometry | None:
    """Resolve and validate the immutable plan geometry attached to ``data``.

    A physical-bin slice retains the complete rank-local bin-index tuple.  The
    exact ordered occurrence IDs identify which entry is being materialized.
    Generic sequence packing has no lockstep metadata and returns ``None``.
    Partially attached or drifting metadata fails closed.
    """
    # Legacy callers and a few backend utility tests still pass a plain
    # mapping. Absence of every attribute means generic (non-lockstep)
    # packing; partially attached metadata continues to fail closed below.
    plan = getattr(data, "lockstep_packing_plan", None)
    side_id = getattr(data, "lockstep_side_id", None)
    rank_bin_indices = getattr(data, "lockstep_bin_indices", None)
    metadata = (plan, side_id, rank_bin_indices)
    if all(value is None for value in metadata):
        return None
    if any(value is None for value in metadata):
        raise ValueError(
            "Lockstep packing metadata is incomplete: plan, side_id, and "
            "rank-local bin indices must be attached together."
        )
    assert plan is not None
    assert side_id is not None
    assert rank_bin_indices is not None

    if side_id not in plan.sides:
        raise ValueError(
            f"Lockstep plan batch_uid={plan.batch_uid} has no selected side "
            f"{side_id!r}."
        )
    side_plan = plan.sides[side_id]
    normalized_rank_bins = tuple(int(value) for value in rank_bin_indices)
    if normalized_rank_bins not in side_plan.rank_bin_indices:
        raise ValueError(
            f"Lockstep side {side_id!r} does not assign rank-local bins "
            f"{normalized_rank_bins}."
        )

    if "batch_item_id" not in data:
        raise ValueError(
            "Plan-driven packing requires batch_item_id on every physical bin."
        )
    item_id_values = data["batch_item_id"]
    if isinstance(item_id_values, torch.Tensor):
        if item_id_values.ndim != 1:
            raise ValueError(
                "batch_item_id must be one-dimensional, got shape "
                f"{tuple(item_id_values.shape)}."
            )
        batch_item_ids = tuple(int(value) for value in item_id_values.tolist())
    elif isinstance(item_id_values, list):
        batch_item_ids = tuple(int(value) for value in item_id_values)
    else:
        raise TypeError(
            "batch_item_id must be a one-dimensional tensor or list, got "
            f"{type(item_id_values).__name__}."
        )
    if len(set(batch_item_ids)) != len(batch_item_ids):
        raise ValueError(
            "batch_item_id values must be unique within a physical bin; got "
            f"{batch_item_ids}."
        )

    matching_bins = [
        bin_index
        for bin_index in normalized_rank_bins
        if tuple(plan.bins[bin_index]) == batch_item_ids
    ]
    if len(matching_bins) != 1:
        raise ValueError(
            "Physical-bin batch_item_id order does not identify exactly one "
            f"assigned lockstep bin: ids={batch_item_ids}, assigned_bins="
            f"{normalized_rank_bins}."
        )
    bin_index = matching_bins[0]

    if input_lengths.ndim != 1 or int(input_lengths.numel()) != len(batch_item_ids):
        raise ValueError(
            "input_lengths must contain one value per planned physical-bin item; "
            f"got shape {tuple(input_lengths.shape)} for {len(batch_item_ids)} IDs."
        )
    actual_lengths = tuple(int(value) for value in input_lengths.tolist())
    raw_cu_seqlens = tuple(side_plan.raw_cu_seqlens_by_bin[bin_index])
    planned_lengths = tuple(
        right - left for left, right in zip(raw_cu_seqlens, raw_cu_seqlens[1:])
    )
    if actual_lengths != planned_lengths:
        raise ValueError(
            f"Lockstep side {side_id!r} bin {bin_index} input lengths drifted: "
            f"actual={actual_lengths}, planned={planned_lengths}."
        )

    padded_cu_seqlens = tuple(side_plan.padded_cu_seqlens_by_bin[bin_index])
    physical_tokens = int(side_plan.physical_tokens_by_bin[bin_index])
    if padded_cu_seqlens[-1] != physical_tokens:
        raise ValueError(
            f"Lockstep side {side_id!r} bin {bin_index} padded boundary ends at "
            f"{padded_cu_seqlens[-1]}, not physical size {physical_tokens}."
        )
    return LockstepBinGeometry(
        bin_index=bin_index,
        batch_item_ids=batch_item_ids,
        raw_cu_seqlens=raw_cu_seqlens,
        padded_cu_seqlens=padded_cu_seqlens,
        physical_tokens=physical_tokens,
    )


def _context_parallel_size(
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
) -> int:
    if context_parallel_group is None:
        return 1
    return torch.distributed.get_world_size(context_parallel_group)


def _packed_row_geometry(
    *,
    sequence_index: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_padded: torch.Tensor,
) -> tuple[int, int, int]:
    """Return ``(raw_length, padded_start, padded_length)`` for one row."""
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q_padded.ndim != 1:
        raise ValueError(
            "Packed xToken cu_seqlens must be one-dimensional, got "
            f"{tuple(cu_seqlens_q.shape)} and "
            f"{tuple(cu_seqlens_q_padded.shape)}."
        )
    num_sequences = int(cu_seqlens_q.numel()) - 1
    if int(cu_seqlens_q_padded.numel()) != num_sequences + 1:
        raise ValueError(
            "Raw and padded cu_seqlens must describe the same number of "
            f"sequences, got {cu_seqlens_q.numel()} and "
            f"{cu_seqlens_q_padded.numel()} entries."
        )
    if sequence_index < 0 or sequence_index >= num_sequences:
        raise IndexError(
            f"sequence_index={sequence_index} is outside [0, {num_sequences})."
        )

    raw_start = int(cu_seqlens_q[sequence_index].item())
    raw_end = int(cu_seqlens_q[sequence_index + 1].item())
    padded_start = int(cu_seqlens_q_padded[sequence_index].item())
    padded_end = int(cu_seqlens_q_padded[sequence_index + 1].item())
    raw_length = raw_end - raw_start
    padded_length = padded_end - padded_start
    if (
        raw_start < 0
        or padded_start < 0
        or raw_length < 0
        or padded_length < raw_length
    ):
        raise ValueError(
            "Invalid packed xToken geometry for sequence "
            f"{sequence_index}: raw=({raw_start}, {raw_end}), "
            f"padded=({padded_start}, {padded_end})."
        )
    return raw_length, padded_start, padded_length


def _is_student_sequence_key(key: str) -> bool:
    """Return whether ``key`` has the student's token axis in dimension one."""
    return key in _STUDENT_SEQUENCE_KEYS or (
        key.startswith("alignment_")
        and key.endswith(_STUDENT_ALIGNMENT_SEQUENCE_SUFFIXES)
    )


def _crop_student_sequence_tensors(
    data: BatchedDataDict[Any],
    *,
    sequence_length: int,
) -> BatchedDataDict[Any]:
    """Crop only known student-token tensors to one row's effective width."""
    for key, value in tuple(data.items()):
        if not _is_student_sequence_key(key):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Packed xToken student sequence field {key!r} must be a "
                f"tensor, got {type(value).__name__}."
            )
        if value.ndim < 2:
            raise ValueError(
                f"Packed xToken student sequence field {key!r} must have a "
                f"sequence axis in dimension one, got shape {tuple(value.shape)}."
            )
        if int(value.shape[1]) < sequence_length:
            raise ValueError(
                f"Packed xToken student sequence field {key!r} has width "
                f"{value.shape[1]}, smaller than row effective width "
                f"{sequence_length}."
            )
        if int(value.shape[1]) > sequence_length:
            data[key] = value.narrow(1, 0, sequence_length)
    return data


def restore_packed_student_logits(
    packed_logits: Tensor,
    *,
    sequence_index: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_padded: torch.Tensor,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """Restore one logical student's logits without touching other data axes.

    ``packed_logits`` follows the backend's packed THD layout. With Megatron
    context parallelism, every physical sequence is independently sharded in
    head/tail order. The returned tensor follows the normal, unpacked backend
    contract: ``[1, S_row, V]`` for CP1, or this CP rank's head/tail shard
    ``[1, S_row / CP, V]`` for CP>1. ``S_row`` is the row's own padded-boundary
    delta, not the batch-wide student rectangle. Padding positions are zero and
    remain masked by the cropped logical row's token mask.
    """
    if packed_logits.ndim != 3 or packed_logits.shape[0] != 1:
        raise ValueError(
            "Packed xToken logits must have shape [1, T, V], got "
            f"{tuple(packed_logits.shape)}."
        )
    raw_length, padded_start, padded_length = _packed_row_geometry(
        sequence_index=sequence_index,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
    )

    cp_size = _context_parallel_size(context_parallel_group)
    if padded_start % cp_size != 0 or padded_length % cp_size != 0:
        raise ValueError(
            "Packed padded boundaries must be divisible by context-parallel "
            f"size {cp_size}; got start={padded_start}, length={padded_length}."
        )
    local_start = padded_start // cp_size
    local_length = padded_length // cp_size
    if local_start + local_length > packed_logits.shape[1]:
        raise ValueError(
            "Packed logit slice exceeds the local sequence dimension: "
            f"start={local_start}, length={local_length}, "
            f"available={packed_logits.shape[1]}."
        )

    physical_local = packed_logits.narrow(1, local_start, local_length)
    if cp_size > 1:
        assert context_parallel_group is not None
        physical_full = allgather_cp_sharded_tensor(
            physical_local, context_parallel_group, seq_dim=1
        )
    else:
        physical_full = physical_local
    valid_logits = physical_full.narrow(1, 0, raw_length)
    if raw_length < padded_length:
        valid_logits = torch.nn.functional.pad(
            valid_logits, (0, 0, 0, padded_length - raw_length)
        )

    if cp_size == 1:
        return valid_logits
    if padded_length % (2 * cp_size) != 0:
        raise ValueError(
            "The effective student sequence width must be divisible by 2 * "
            f"context-parallel size; got width={padded_length}, "
            f"cp_size={cp_size}."
        )
    assert context_parallel_group is not None
    cp_rank = torch.distributed.get_rank(context_parallel_group)
    return _get_tokens_on_this_cp_rank(
        valid_logits,
        cp_rank=cp_rank,
        cp_size=cp_size,
        seq_dim=1,
    ).contiguous()


class XTokenSequencePackingLossWrapper:
    """Run cross-tokenizer distillation at logical microbatch size one.

    Unlike :class:`SequencePackingLossWrapper`, this wrapper distinguishes the
    independent student-token, teacher-token, and pair-slot axes. A sliced
    logical row restores and crops only its student-token rectangle to that
    row's padded boundary delta. Teacher, pair, region, and IPC payloads retain
    their original per-row shapes and contents.
    """

    def __init__(
        self,
        loss_fn: LossFunction,
        prepare_fn: Callable[..., Any],
        cu_seqlens_q: Tensor,
        cu_seqlens_q_padded: Optional[Tensor] = None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.loss_fn = loss_fn
        self.prepare_fn = prepare_fn
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_q_padded = (
            cu_seqlens_q if cu_seqlens_q_padded is None else cu_seqlens_q_padded
        )
        self.vocab_parallel_rank = vocab_parallel_rank
        self.vocab_parallel_group = vocab_parallel_group
        self.context_parallel_group = context_parallel_group

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor | None,
        global_valid_kd_toks: Tensor | None = None,
        global_valid_chunks_by_idx: dict[int, torch.Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        num_sequences = int(self.cu_seqlens_q.numel()) - 1
        if data.size != num_sequences:
            raise ValueError(
                "Packed xToken metadata must contain one boundary per logical "
                f"row: data.size={data.size}, sequences={num_sequences}."
            )
        loss_accum: Any = 0
        metrics_accum: dict[str, Any] = {}
        logical_metrics: list[dict[str, Any]] = []
        for sequence_index in range(num_sequences):
            _, _, effective_sequence_length = _packed_row_geometry(
                sequence_index=sequence_index,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_q_padded=self.cu_seqlens_q_padded,
            )
            logical_logits = restore_packed_student_logits(
                next_token_logits,
                sequence_index=sequence_index,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_q_padded=self.cu_seqlens_q_padded,
                context_parallel_group=self.context_parallel_group,
            )
            sequence_data = data.slice(sequence_index, sequence_index + 1)
            sequence_data = _crop_student_sequence_tensors(
                sequence_data,
                sequence_length=effective_sequence_length,
            )
            loss_input, sequence_data = self.prepare_fn(
                logits=logical_logits,
                data=sequence_data,
                loss_fn=self.loss_fn,
                vocab_parallel_rank=self.vocab_parallel_rank,
                vocab_parallel_group=self.vocab_parallel_group,
                context_parallel_group=self.context_parallel_group,
            )
            extra_loss_kwargs: dict[str, Any] = {}
            if global_valid_chunks_by_idx:
                extra_loss_kwargs["global_valid_chunks_by_idx"] = (
                    global_valid_chunks_by_idx
                )
            loss, metrics = self.loss_fn(
                data=sequence_data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                global_valid_kd_toks=global_valid_kd_toks,
                **extra_loss_kwargs,
                **loss_input,
            )
            loss_accum += loss
            self._accumulate_metrics(metrics_accum, metrics)

            # The worker's established aggregation treats every metric dict as
            # one logical MBS1 cohort.  Preserve that cardinality instead of
            # turning an uneven physical bin into one intensive-metric record.
            # Masked rows still execute the loss (and any collectives), but
            # match DTensor's unpacked behavior by contributing no metric row.
            num_valid_samples = metrics.get("num_valid_samples", 1)
            if isinstance(num_valid_samples, torch.Tensor):
                num_valid_samples = num_valid_samples.item()
            if num_valid_samples > 0:
                logical_metrics.append(dict(metrics))

        metrics_accum[XTOKEN_LOGICAL_METRICS_KEY] = logical_metrics
        return loss_accum, metrics_accum

    @staticmethod
    def _accumulate_metrics(
        accumulated: dict[str, Any], metrics: dict[str, Any]
    ) -> None:
        """Match the established packing wrapper's metric reductions."""
        minimum_metrics = {"probs_ratio_min", "probs_ratio_clamped_min"}
        maximum_metrics = {"probs_ratio_max", "probs_ratio_clamped_max"}
        for key, value in metrics.items():
            if key not in accumulated:
                if key in minimum_metrics:
                    accumulated[key] = float("inf")
                elif key in maximum_metrics:
                    accumulated[key] = float("-inf")
                else:
                    accumulated[key] = 0
            scalar_or_value = (
                value.item()
                if isinstance(value, torch.Tensor) and value.ndim == 0
                else value
            )
            if key in minimum_metrics:
                if not math.isinf(scalar_or_value):
                    accumulated[key] = min(accumulated[key], scalar_or_value)
            elif key in maximum_metrics:
                if not math.isinf(scalar_or_value):
                    accumulated[key] = max(accumulated[key], scalar_or_value)
            else:
                accumulated[key] += scalar_or_value
