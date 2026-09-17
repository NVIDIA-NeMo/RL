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
"""Shared utilities for cross-tokenizer distillation.

Used by both :mod:`token_aligner` and
:mod:`nemo_rl.algorithms.loss.loss_functions`:

- :class:`Fp32SparseMM` — FP32 sparse-dense matmul that ignores BF16
  autocast (no BF16 sparse-mm kernel exists).
- Chunk aggregation: :func:`chunk_log_prob_sums` / :func:`chunk_average_finalize`
  / :func:`chunk_average_log_probs` / :func:`valid_chunk_mask` (the
  partial/finalize split lets callers insert a CP all-reduce between), plus
  :func:`nemo_rl.distributed.model_utils.group_all_reduce_sum` for the global
  valid-chunk denominator.
- Teacher-logit IPC: :func:`rebuild_teacher_full_logits_from_ipc`,
  :func:`rebuild_teacher_sparse_logits_from_ipc`,
  :func:`assemble_teacher_logits_from_shards`,
  :func:`collect_overlapping_teacher_shards` reassemble full-vocab teacher
  logits from per-rank shards across heterogeneous TP/CP.
- Projection: :func:`parse_projection_file`, the
  :func:`get_sparse_projection_matrix` / :func:`get_topk_projection`
  process-local caches, :func:`slice_sparse_projection_rows`, and
  :func:`build_exact_token_map` (cached common/uncommon partition).
- :func:`alignment_from_flat_batch` rehydrates the flat ``alignment_*``
  data-dict keys into an :class:`AlignmentBatch`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch.distributed.tensor import DTensor

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch
from nemo_rl.distributed.model_utils import (
    allgather_cp_contiguous_tensor,
    cp_load_balanced_to_contiguous,
    cp_shift_next,
    get_logprobs_from_vocab_parallel_logits,
    group_all_reduce_sum_with_grad,
    vocab_parallel_argmax,
)
from nemo_rl.models.dtensor.parallelize import to_local_if_dtensor

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.context_parallel import (
        ContextParallelSharder,
    )


SparseTeacherLogits = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]


def alignment_from_flat_batch(data: Mapping[str, Any]) -> AlignmentBatch:
    """Rebuild :class:`AlignmentBatch` from the flat ``alignment_*`` keys.

    The field set is driven off :class:`AlignmentBatch` so the helper
    can't drift from the schema.
    """
    return AlignmentBatch(
        **{f.name: data[f"alignment_{f.name}"] for f in fields(AlignmentBatch)}
    )


class Fp32SparseMM(torch.autograd.Function):
    """FP32 ``M.t() @ dense`` (sparse-dense matmul) ignoring surrounding autocast.

    ``addmm_sparse_cuda`` has no BF16 kernel on either forward or backward.
    The worker wraps forward + loss + backward in ``autocast(BF16)``, so a
    plain ``with autocast(enabled=False):`` around the forward call is not
    enough — ``loss.backward()`` runs inside the outer autocast and the
    sparse-mm backward kernel is still dispatched as BF16. The
    ``custom_fwd(cast_inputs=torch.float32)`` / ``custom_bwd`` decorators
    are PyTorch's official escape: they force FP32 inputs on forward and
    run the backward as if autocast were disabled.

    autograd's builtin sparse-mm backward computes
    ``M @ grad_out``. The gradient w.r.t. the sparse argument isn't
    needed (the projection matrix is frozen), so it's returned as ``None``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx: Any, sparse_M: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        ctx.sparse_M = sparse_M
        return torch.sparse.mm(sparse_M.t(), dense)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[None, torch.Tensor]:
        sparse_M = ctx.sparse_M
        # out = sparse_M.t() @ dense, so d/d_dense = sparse_M @ grad_out.
        grad_dense = torch.sparse.mm(sparse_M, grad_out)
        return None, grad_dense


def chunk_log_prob_sums(
    log_probs: torch.Tensor,
    chunk_id: torch.Tensor,
    max_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local bmm + bucket count, no division.

    Output is summable across CP; callers that need cross-rank chunks to
    aggregate correctly should ``group_all_reduce_sum_with_grad`` both tensors
    before :func:`chunk_average_finalize`. ``chunk_id == -1`` contributes to no
    bucket.
    """
    device = log_probs.device
    chunk_arange = torch.arange(max_chunks, device=device).view(1, 1, -1)
    chunk_mask = chunk_id.unsqueeze(-1) == chunk_arange
    chunk_mask_f = chunk_mask.transpose(1, 2).to(log_probs.dtype)
    chunk_sums = torch.bmm(chunk_mask_f, log_probs)  # [B, C, V]
    chunk_sizes = chunk_mask.sum(dim=1).float()  # [B, C]
    return chunk_sums, chunk_sizes


def chunk_average_finalize(
    chunk_sums: torch.Tensor,
    chunk_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide sums by sizes; ``eps`` guards empty buckets."""
    eps = 1e-10
    chunk_log_probs = chunk_sums / (chunk_sizes.unsqueeze(-1) + eps)
    return chunk_log_probs, chunk_sizes


def chunk_average_log_probs(
    log_probs: torch.Tensor,
    chunk_id: torch.Tensor,
    max_chunks: int,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``log_probs`` over chunks defined by ``chunk_id``.

    Builds a one-hot chunk mask from ``chunk_id`` (``-1`` = no chunk), then
    ``bmm``-aggregates and divides by chunk sizes. When ``cp_group`` has world
    > 1, the per-chunk sums are ``group_all_reduce_sum_with_grad``'d across CP
    ranks before the divide (mean is non-linear, so the reduce must precede it).

    Args:
        log_probs: ``[B, T, V]`` log-probabilities.
        chunk_id: ``[B, T]`` long tensor, values in ``[-1, max_chunks)``.
        max_chunks: number of chunk buckets.
        cp_group: context-parallel group for cross-rank chunk aggregation.

    Returns:
        chunk_log_probs: ``[B, max_chunks, V]`` averaged log-probs.
        chunk_sizes: ``[B, max_chunks]`` float tensor of bucket sizes.
    """
    chunk_sums, chunk_sizes = chunk_log_prob_sums(log_probs, chunk_id, max_chunks)
    if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
        chunk_sums = group_all_reduce_sum_with_grad(chunk_sums, cp_group)
        chunk_sizes = group_all_reduce_sum_with_grad(chunk_sizes, cp_group)
    return chunk_average_finalize(chunk_sums, chunk_sizes)


def slice_sparse_projection_rows(
    sparse_matrix: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Row-slice a sparse-COO projection ``[V_s, V_t]`` to ``[row_end-row_start, V_t]``.

    Filters COO indices in-place: keeps entries with row in ``[row_start, row_end)``
    and shifts the row index by ``-row_start``. Used by the TP-aware P-KL path
    where each rank owns a contiguous slab of the student vocab axis.
    """
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()
    mask = (indices[0] >= row_start) & (indices[0] < row_end)
    local_indices = indices[:, mask].clone()
    local_indices[0] -= row_start
    local_values = values[mask]
    return torch.sparse_coo_tensor(
        local_indices,
        local_values,
        (row_end - row_start, sparse_matrix.size(1)),
        device=sparse_matrix.device,
        dtype=sparse_matrix.dtype,
    ).coalesce()


def slice_sparse_projection_cols(
    sparse_matrix: torch.Tensor,
    col_indices: torch.Tensor,
) -> torch.Tensor:
    """Column-slice a sparse-COO projection ``[V_s, V_t]`` to ``[V_s, len(col_indices)]``.

    ``col_indices`` must be sorted and unique -- ``select_teacher_topk_indices``
    returns exactly that. Keeps COO entries whose column appears in
    ``col_indices`` and remaps that column to its position in the list, so the
    result equals what a dense ``matrix[:, col_indices]`` would give.

    The row slicer above shifts by a constant because ranks own contiguous
    slabs; columns here are an arbitrary sorted subset, hence the searchsorted.
    """
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()
    cols = indices[1]
    # Position each entry's column would take in ``col_indices``; entries whose
    # column is absent land on a neighbour and are dropped by the equality test.
    pos = torch.searchsorted(col_indices, cols).clamp(max=col_indices.numel() - 1)
    mask = col_indices[pos] == cols
    kept_indices = indices[:, mask].clone()
    kept_indices[1] = pos[mask]
    return torch.sparse_coo_tensor(
        kept_indices,
        values[mask],
        (sparse_matrix.size(0), col_indices.numel()),
        device=sparse_matrix.device,
        dtype=sparse_matrix.dtype,
    ).coalesce()


# ---------------------------------------------------------------------------
# TP/CP-aware loss primitives
#
# Each of these collapses to the plain single-rank torch op when the relevant
# process group has world size 1, so the cross-tokenizer loss body stays free
# of any ``tp_world > 1`` / rank / offset branching.
# ---------------------------------------------------------------------------
def project_student_to_teacher_vocab(
    student_probs: torch.Tensor,
    sparse_projection: torch.Tensor,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Project student vocab probs ``[B, T, V_s(/TP)]`` to teacher vocab ``[B, T, V_t]``.

    ``sparse_projection`` is the full ``[V_s, V_t]`` sparse-COO matrix. With
    ``tp_group`` world > 1 the student probs cover only this rank's ``V_s/TP``
    rows, so the matrix is row-sliced to that range, the sparse matmul produces a
    partial teacher-vocab sum, and a ``group_all_reduce_sum_with_grad`` over the
    TP group combines the partials into the full ``V_s`` contraction. Otherwise a
    single sparse matmul over the full matrix is used.
    """
    batch_size, seq_len, local_vocab_size = student_probs.shape
    flat = student_probs.reshape(batch_size * seq_len, local_vocab_size)
    tp_world = torch.distributed.get_world_size(tp_group) if tp_group is not None else 1
    if tp_world > 1:
        tp_rank = torch.distributed.get_rank(tp_group)
        full_student_vocab_size = sparse_projection.size(0)
        rows_per_rank = full_student_vocab_size // tp_world
        local_projection = slice_sparse_projection_rows(
            sparse_projection,
            row_start=tp_rank * rows_per_rank,
            row_end=(tp_rank + 1) * rows_per_rank,
        )
        projected_partial = Fp32SparseMM.apply(local_projection, flat.t()).t()
        projected = group_all_reduce_sum_with_grad(
            projected_partial.contiguous(), tp_group
        )
    else:
        # Fp32SparseMM internally computes M.t() @ dense; passing M (not M.t())
        # avoids a sparse ``.t()`` on a saved tensor in backward.
        projected = Fp32SparseMM.apply(sparse_projection, flat.t()).t()
    teacher_vocab_size = projected.shape[-1]
    return projected.reshape(batch_size, seq_len, teacher_vocab_size)


def select_teacher_topk_indices(
    teacher_logits: torch.Tensor,
    k: int,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Sorted global top-``k`` teacher-vocab ids by max importance over the microbatch.

    Importance is the per-vocab max over flattened ``(B*T)`` teacher logits. With
    ``cp_group`` world > 1 the sequence is CP-sharded, so the local max only sees
    this rank's slice; an ``all_reduce(MAX)`` makes every rank pick the same
    subset. When ``valid_mask`` is provided, invalid predictor rows are excluded
    from the vocabulary-column maximum. If no valid predictor exists globally,
    the first ``k`` columns are returned deterministically. No gradient.
    """
    vocab_size = teacher_logits.shape[-1]
    if valid_mask is not None and valid_mask.shape != teacher_logits.shape[:-1]:
        raise ValueError(
            "valid_mask must match teacher_logits without its vocabulary axis: "
            f"expected {tuple(teacher_logits.shape[:-1])}, got "
            f"{tuple(valid_mask.shape)}."
        )
    with torch.no_grad():
        # reshape (not view): a preceding next-token shift can leave the teacher
        # logits non-contiguous.
        teacher_flat = teacher_logits.reshape(-1, vocab_size)
        if valid_mask is not None:
            valid_flat = valid_mask.to(device=teacher_logits.device, dtype=torch.bool)
            teacher_flat = teacher_flat.masked_fill(
                ~valid_flat.reshape(-1, 1), float("-inf")
            )
        importance = teacher_flat.max(dim=0).values
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
            torch.distributed.all_reduce(
                importance, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )
        if torch.isneginf(importance).all():
            return torch.arange(k, device=teacher_logits.device)
        top_indices = torch.topk(importance, k=k, dim=-1).indices
        return top_indices.sort().values


@dataclass
class LocalizedAlignment:
    """CP-localized alignment tensors consumed by the loss reductions.

    For a cross-tokenizer teacher every field is populated (chunk-averaged
    projection KL / gold path). For a same-tokenizer teacher (no projection,
    identity 1:1 token alignment) the chunk/pair fields stay ``None``: its KD
    term reads only the shared student fields (``student_input_ids`` /
    ``student_token_mask`` / ``sample_mask``).
    """

    sample_mask: torch.Tensor
    student_chunk_id: Optional[torch.Tensor] = None
    teacher_chunk_id: Optional[torch.Tensor] = None
    pair_valid: Optional[torch.Tensor] = None
    pair_is_correct: Optional[torch.Tensor] = None
    # Filled post-construction by prepare_xtoken_cross_tokenizer_loss_input: the
    # CP-relaid contiguous student input_ids / token_mask shared by the
    # next-token-accuracy metric and the same-tokenizer KD path.
    student_input_ids: Optional[torch.Tensor] = None
    student_token_mask: Optional[torch.Tensor] = None
    # KD semantic targets may be narrower than the SFT/CE role mask in chat
    # mode (assistant content plus an explicitly identified EOT token). Text
    # mode aliases this to ``student_token_mask``.
    student_kd_token_mask: Optional[torch.Tensor] = None
    # Filled post-construction for the v6 (prefix_bidir_partition_kl_v3) path:
    # this CP rank's contiguous teacher input ids, per-chunk contiguous position
    # spans ``[B, max_pairs, 2]`` derived from the unshifted chunk ids, and the
    # per-sample chunk count ``[B]``. Left ``None`` for a same-tokenizer teacher.
    teacher_input_ids: Optional[torch.Tensor] = None
    student_spans: Optional[torch.Tensor] = None
    teacher_spans: Optional[torch.Tensor] = None
    num_chunks: Optional[torch.Tensor] = None


def localize_alignment(
    data: Mapping[str, Any],
    *,
    teacher_seq_len: int,
    alignment_prefix: str = "alignment_",
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> LocalizedAlignment:
    """Localize the chunk-alignment data-dict fields for the local CP shard.

    Unwraps the ``{alignment_prefix}*`` / ``sample_mask`` entries from DTensor to
    their local tensors. Student-seq fields (``student_chunk_id``) are left in
    whatever layout the backend produced — load-balanced CP shards on the
    DTensor worker (``cp_buffers``), full ``[B, S]`` on Megatron — and the caller
    (:func:`prepare_xtoken_cross_tokenizer_loss_input`) relays them to this
    rank's contiguous window via :func:`_student_seq_to_contiguous_window`.
    The teacher-seq ``teacher_chunk_id`` is full on both backends, so it is
    sliced contiguously to this CP rank's ``teacher_seq_len`` window to match the
    IPC consumer's contiguous teacher-logit slice.

    Args:
        alignment_prefix: Data-dict key prefix for this teacher's alignment
            tensors (``"alignment_"`` single-teacher, ``"alignment_{i}_"`` per
            teacher in the multi-teacher trainer / collator). ``sample_mask`` is
            student-level and stays unprefixed.
    """
    teacher_chunk_id_full = to_local_if_dtensor(
        data[f"{alignment_prefix}teacher_chunk_id"]
    )
    cp_rank = (
        torch.distributed.get_rank(cp_group)
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1
        else 0
    )
    teacher_seq_start = cp_rank * teacher_seq_len
    teacher_chunk_id = teacher_chunk_id_full[
        :, teacher_seq_start : teacher_seq_start + teacher_seq_len
    ]
    return LocalizedAlignment(
        sample_mask=to_local_if_dtensor(data["sample_mask"]),
        student_chunk_id=to_local_if_dtensor(
            data[f"{alignment_prefix}student_chunk_id"]
        ),
        teacher_chunk_id=teacher_chunk_id,
        pair_valid=to_local_if_dtensor(data[f"{alignment_prefix}pair_valid"]),
        pair_is_correct=to_local_if_dtensor(data[f"{alignment_prefix}pair_is_correct"]),
    )


def student_next_token_ce(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    seq_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token next-token cross-entropy ``[B, T-1]`` on the student.

    DTensor (TP/CP) logits route through the vocab-parallel log-prob helper
    (which also handles the CP roll); plain logits use a local shifted
    ``cross_entropy``. The next-token shift (drop the last predictor) matches the
    convention the KL terms use.
    """
    if isinstance(logits, DTensor):
        next_token_logprobs = get_logprobs_from_vocab_parallel_logits(
            logits, input_ids, seq_index=seq_index
        )
        return -next_token_logprobs
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)


def ce_label_mask(
    *,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    ce_seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Next-token label mask ``[B, ce_seq_len]`` = shifted token_mask * sample_mask.

    ``token_mask`` is gathered to the full sequence (CP) before the shift; both
    inputs are DTensor-unwrapped.
    """
    token_mask = (
        token_mask.full_tensor() if isinstance(token_mask, DTensor) else token_mask
    )
    sample_mask = to_local_if_dtensor(sample_mask)
    return (token_mask[:, 1 : ce_seq_len + 1] * sample_mask.unsqueeze(-1)).to(dtype)


def next_token_accuracy(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Masked next-token top-1 accuracy of the student (scalar, no gradient).

    Uses :func:`vocab_parallel_argmax` for the (possibly TP-sharded) argmax. The
    next-token shift on labels/mask is CP-aware (:func:`cp_shift_next`) so the
    boundary token crosses CP ranks, and the correct/total counts are CP-reduced
    so every rank reports the same global accuracy.
    """
    with torch.no_grad():
        argmax = vocab_parallel_argmax(logits, tp_group=tp_group)
        next_labels = cp_shift_next(to_local_if_dtensor(input_ids), cp_group, fill=0)
        next_mask = cp_shift_next(to_local_if_dtensor(token_mask), cp_group, fill=0)
        acc_mask = (
            next_mask.float() * to_local_if_dtensor(sample_mask).unsqueeze(-1).float()
        )
        correct = ((argmax == next_labels).float() * acc_mask).sum()
        denom = acc_mask.sum()
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
            stats = torch.stack([correct, denom])
            torch.distributed.all_reduce(stats, group=cp_group)
            correct, denom = stats[0], stats[1]
        return correct / denom.clamp(min=1.0)


def collect_overlapping_teacher_shards(
    teacher_shards: list[dict[str, Any]],
    student_cp_rank: int,
    student_cp_size: int,
    full_seq_len: int,
) -> list[tuple[dict[str, Any], slice, slice, slice, slice]]:
    """Plan ``(src_seq, src_vocab, dest_seq, dest_vocab)`` slices per teacher shard.

    Dest is ``[T_t/CP_s, V_t]`` (vocab fully reassembled, seq is this
    student CP rank's range). Shards with no seq overlap are skipped.
    """
    student_seq_start = student_cp_rank * full_seq_len // student_cp_size
    student_seq_end = (student_cp_rank + 1) * full_seq_len // student_cp_size

    from nemo_rl.models.policy.utils import validate_compact_teacher_ipc_handle

    matches: list[tuple[dict[str, Any], slice, slice, slice, slice]] = []
    for handle in teacher_shards:
        teacher_vocab_start = int(handle["vocab_start_index"])
        teacher_vocab_end = int(handle["vocab_end_index"])
        teacher_seq_start = int(handle["global_seq_start"])
        compact_geometry = validate_compact_teacher_ipc_handle(handle)
        stored_seq_len = (
            compact_geometry[1]
            if compact_geometry is not None
            else int(handle["actual_shape"][0])
        )
        # A compact shard's omitted suffix is logically present but known zero.
        # Plan copies only for the physically stored prefix; ``dest`` is
        # zero-initialized by the caller and therefore reconstructs the exact
        # old dense rectangle.
        teacher_seq_end = teacher_seq_start + stored_seq_len

        overlap_seq_start = max(student_seq_start, teacher_seq_start)
        overlap_seq_end = min(student_seq_end, teacher_seq_end)
        if overlap_seq_end <= overlap_seq_start:
            continue

        src_seq = slice(
            overlap_seq_start - teacher_seq_start,
            overlap_seq_end - teacher_seq_start,
        )
        src_vocab = slice(0, teacher_vocab_end - teacher_vocab_start)
        dest_seq = slice(
            overlap_seq_start - student_seq_start,
            overlap_seq_end - student_seq_start,
        )
        dest_vocab = slice(teacher_vocab_start, teacher_vocab_end)
        matches.append((handle, src_seq, src_vocab, dest_seq, dest_vocab))
    return matches


def _rebuild_compact_teacher_ipc_storage(
    handle: Mapping[str, Any], device: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Map and validate one compact producer slab without copying it."""
    from nemo_rl.models.policy.utils import (
        rebuild_cuda_tensor_from_ipc,
        validate_compact_teacher_ipc_handle,
    )

    compact_geometry = validate_compact_teacher_ipc_handle(handle)
    if compact_geometry is None:
        raise ValueError("Expected a compact dense teacher IPC handle.")
    _token_offset, stored_seq_len, _used_tokens, _local_vocab_size = compact_geometry
    if stored_seq_len <= 0:
        raise ValueError("A zero-length compact IPC row has no storage to rebuild.")
    src_full = rebuild_cuda_tensor_from_ipc(handle["payload_ipc"], device).detach()
    expected_storage_shape = tuple(int(value) for value in handle["storage_shape"])
    if tuple(src_full.shape) != expected_storage_shape or src_full.ndim != 2:
        raise ValueError(
            "Compact dense teacher IPC rebuilt an unexpected storage shape: "
            f"actual={tuple(src_full.shape)}, expected={expected_storage_shape}."
        )
    if not src_full.is_contiguous():
        raise ValueError("Compact dense teacher IPC storage must be contiguous.")
    if src_full.dtype != handle["dtype"]:
        raise ValueError(
            "Compact dense teacher IPC rebuilt an unexpected dtype: "
            f"actual={src_full.dtype}, expected={handle['dtype']}."
        )
    return src_full, compact_geometry


def _rebuild_teacher_ipc_row(handle: Mapping[str, Any], device: int) -> torch.Tensor:
    """Return the physically stored ``[T_stored, V_local]`` row view."""
    from nemo_rl.models.policy.utils import (
        rebuild_cuda_tensor_from_ipc,
        validate_compact_teacher_ipc_handle,
    )

    compact_geometry = validate_compact_teacher_ipc_handle(handle)
    if compact_geometry is not None:
        token_offset, stored_seq_len, _used_tokens, local_vocab_size = compact_geometry
        if stored_seq_len == 0:
            raise ValueError("A zero-length compact IPC row has no physical view.")
        src_full, _ = _rebuild_compact_teacher_ipc_storage(handle, device)
        return src_full[token_offset : token_offset + stored_seq_len, :local_vocab_size]

    src_full = rebuild_cuda_tensor_from_ipc(handle["payload_ipc"], device).detach()
    actual_shape = handle["actual_shape"]
    if (
        not isinstance(actual_shape, (list, tuple, torch.Size))
        or len(actual_shape) != 2
    ):
        raise ValueError(
            "Dense teacher IPC actual_shape must be two-dimensional, got "
            f"{actual_shape!r}."
        )
    local_seq_len, local_vocab_size = (int(value) for value in actual_shape)
    buf_idx = int(handle["buf_idx"])
    sample_idx = int(handle["sample_index_in_buf"])
    if (
        src_full.ndim != 4
        or buf_idx < 0
        or buf_idx >= src_full.shape[0]
        or sample_idx < 0
        or sample_idx >= src_full.shape[1]
        or local_seq_len <= 0
        or local_seq_len > src_full.shape[2]
        or local_vocab_size <= 0
        or local_vocab_size > src_full.shape[3]
    ):
        raise ValueError(
            "Dense teacher IPC rectangular handle is outside rebuilt storage: "
            f"storage={tuple(src_full.shape)}, buf_idx={buf_idx}, "
            f"sample_idx={sample_idx}, actual_shape={tuple(actual_shape)}."
        )
    return src_full[buf_idx, sample_idx, :local_seq_len, :local_vocab_size]


def assemble_teacher_logits_from_shards(
    teacher_shards: list[dict[str, Any]],
    student_cp_rank: int,
    student_cp_size: int,
    device: int,
) -> torch.Tensor:
    """P2P-IPC-read overlapping teacher shards into a ``[T_t/CP_s, V_t]`` dest.

    ``device`` is a CUDA device index (matches
    :func:`rebuild_cuda_tensor_from_ipc`'s ``device_id`` signature).
    """
    if not teacher_shards:
        raise ValueError("teacher_shards must be non-empty")
    full_seq_len = int(teacher_shards[0]["full_seq_len"])
    full_vocab_size = int(teacher_shards[0]["full_vocab_size"])
    # CP seq-padding guarantees this; assert it so the contiguous-window math
    # below (and the `dest` size) can't silently go out of bounds if a caller
    # ever passes an unpadded length.
    assert full_seq_len % student_cp_size == 0, (
        f"full_seq_len={full_seq_len} not divisible by student_cp_size={student_cp_size}"
    )
    local_seq_len = full_seq_len // student_cp_size

    dest = torch.zeros(
        (local_seq_len, full_vocab_size),
        dtype=torch.float32,
        device=device,
    )
    matches = collect_overlapping_teacher_shards(
        teacher_shards,
        student_cp_rank=student_cp_rank,
        student_cp_size=student_cp_size,
        full_seq_len=full_seq_len,
    )
    for handle, src_seq, src_vocab, dest_seq, dest_vocab in matches:
        src = _rebuild_teacher_ipc_row(handle, device)
        dest[dest_seq, dest_vocab] = src[src_seq, src_vocab].to(torch.float32)
    return dest


def _try_zero_copy_teacher_logits(
    per_sample_entries: list[dict[str, Any]],
    *,
    student_cp_rank: int,
    student_cp_size: int,
    device: int,
) -> Optional[torch.Tensor]:
    """Zero-copy ``[B, T_t/CP_s, V_t]`` view of the teacher logits, or None.

    Returns a view into the producer's IPC storage only when reassembly is
    unnecessary: every sample's seq range is covered by a single full-vocab
    teacher shard (i.e. teacher ``tp_size == 1`` and ``teacher_cp == student_cp``
    or ``teacher_cp == 1``), and the microbatch's samples are a contiguous slab
    (same payload + ``buf_idx``, sample rows ``0..B-1``) in one storage slot.
    Otherwise returns None and the caller falls back to assemble + stack.
    """
    if not per_sample_entries:
        return None
    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    first_shards = per_sample_entries[0]["teacher_shards"]
    if not first_shards:
        return None
    full_seq_len = int(first_shards[0]["full_seq_len"])
    full_vocab_size = int(first_shards[0]["full_vocab_size"])
    student_seq_start = student_cp_rank * full_seq_len // student_cp_size
    student_seq_end = (student_cp_rank + 1) * full_seq_len // student_cp_size

    from nemo_rl.models.policy.utils import validate_compact_teacher_ipc_handle

    # Exactly one physically stored full-vocab shard must cover this student
    # rank's seq range. A compact zero suffix is logical coverage, but cannot be
    # represented by a view and therefore takes the zero-filling fallback.
    chosen: list[dict[str, Any]] = []
    for entry in per_sample_entries:
        covering = []
        for handle in entry["teacher_shards"]:
            compact_geometry = validate_compact_teacher_ipc_handle(handle)
            stored_seq_len = (
                compact_geometry[1]
                if compact_geometry is not None
                else int(handle["actual_shape"][0])
            )
            if (
                int(handle["vocab_start_index"]) == 0
                and int(handle["vocab_end_index"]) == full_vocab_size
                and int(handle["global_seq_start"]) <= student_seq_start
                and int(handle["global_seq_start"]) + stored_seq_len >= student_seq_end
            ):
                covering.append(handle)
        if len(covering) != 1:
            return None
        chosen.append(covering[0])

    compact_geometries = [
        validate_compact_teacher_ipc_handle(handle) for handle in chosen
    ]
    if any(geometry is not None for geometry in compact_geometries):
        if any(geometry is None for geometry in compact_geometries):
            raise ValueError(
                "Dense teacher IPC zero-copy candidates mix compact and "
                "rectangular storage layouts."
            )
        h0 = chosen[0]
        payload = h0["payload_ipc"]
        teacher_seq_start = int(h0["global_seq_start"])
        seq_lo = student_seq_start - teacher_seq_start
        seq_hi = student_seq_end - teacher_seq_start
        first_geometry = compact_geometries[0]
        assert first_geometry is not None
        first_offset, first_stored_len, _, local_vocab_size = first_geometry
        for index, (handle, geometry) in enumerate(
            zip(chosen, compact_geometries, strict=True)
        ):
            assert geometry is not None
            token_offset, stored_seq_len, _, handle_vocab_size = geometry
            if (
                handle["payload_ipc"] != payload
                or int(handle["global_seq_start"]) != teacher_seq_start
                or tuple(handle["storage_shape"]) != tuple(h0["storage_shape"])
                or handle["dtype"] != h0["dtype"]
                or handle_vocab_size != local_vocab_size
                or stored_seq_len != first_stored_len
                or token_offset != first_offset + index * first_stored_len
            ):
                return None
        src_full, _ = _rebuild_compact_teacher_ipc_storage(h0, device)
        if len(chosen) == 1:
            return src_full[
                first_offset + seq_lo : first_offset + seq_hi,
                :local_vocab_size,
            ].unsqueeze(0)
        return src_full.as_strided(
            size=(len(chosen), seq_hi - seq_lo, local_vocab_size),
            stride=(first_stored_len * local_vocab_size, local_vocab_size, 1),
            storage_offset=(
                src_full.storage_offset() + (first_offset + seq_lo) * local_vocab_size
            ),
        )

    # Legacy rectangular samples must form a contiguous slab in one storage
    # slot.
    h0 = chosen[0]
    payload = h0["payload_ipc"]
    buf_idx = int(h0["buf_idx"])
    teacher_seq_start = int(h0["global_seq_start"])
    for i, h in enumerate(chosen):
        if (
            h["payload_ipc"] != payload
            or int(h["buf_idx"]) != buf_idx
            or int(h["sample_index_in_buf"]) != i
            or int(h["global_seq_start"]) != teacher_seq_start
        ):
            return None

    src_full = rebuild_cuda_tensor_from_ipc(payload, device).detach()
    seq_lo = student_seq_start - teacher_seq_start
    seq_hi = student_seq_end - teacher_seq_start
    return src_full[buf_idx, : len(chosen), seq_lo:seq_hi, :full_vocab_size]


def rebuild_teacher_full_logits_from_ipc(
    per_sample_entries: list[dict[str, Any]],
    cp_group: Optional[torch.distributed.ProcessGroup],
    device: int,
) -> tuple[torch.Tensor, int]:
    """Rebuild teacher logits and report actual fallback reconstructions.

    Returns ``([B, T_t/CP_s, V_t], fallback_count)``. ``fallback_count`` is
    zero when the zero-copy view is used and otherwise counts the logical rows
    that went through shard assembly.  Reporting the value from this branch,
    instead of predicting it from controller-side topology, keeps IPC
    observability tied to the path that actually ran.

    Fast path (zero-copy view via :func:`_try_zero_copy_teacher_logits`): when the
    teacher is not vocab-sharded and each sample's seq range is covered by a
    single shard, return a view into the IPC storage. Otherwise reassemble each
    sample from its overlapping shards and stack.
    """
    student_cp_rank = (
        torch.distributed.get_rank(cp_group) if cp_group is not None else 0
    )
    student_cp_size = (
        torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
    )

    # Bypass: when the teacher layout lines up with this student rank (no
    # vocab sharding, seq covered by one shard), skip reassembly and return a
    # zero-copy view of the IPC storage. Returns None when reassembly is needed.
    view = _try_zero_copy_teacher_logits(
        per_sample_entries,
        student_cp_rank=student_cp_rank,
        student_cp_size=student_cp_size,
        device=device,
    )
    if view is not None:
        return view, 0

    rebuilt = [
        assemble_teacher_logits_from_shards(
            entry["teacher_shards"],
            student_cp_rank=student_cp_rank,
            student_cp_size=student_cp_size,
            device=device,
        )
        for entry in per_sample_entries
    ]
    # Packed xToken executes the loss at logical MBS1. Preserve the assembled
    # row's storage in that overwhelmingly common path: torch.stack would
    # allocate and copy another full [T_t/CP_s, V_t] tensor (about 1.24 GiB for
    # Qwen3-14B at TP2/CP2) only to add a unit batch dimension.
    if len(rebuilt) == 1:
        return rebuilt[0].unsqueeze(0), 1
    return torch.stack(rebuilt, dim=0), len(rebuilt)


def rebuild_teacher_sparse_logits_from_ipc(
    per_sample_entries: list[dict[str, Any]],
    *,
    device: int,
) -> SparseTeacherLogits:
    """Rebuild full-sequence sparse teacher payloads from per-CP-shard IPC."""
    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    if not per_sample_entries:
        raise ValueError("Sparse teacher IPC payload is empty.")

    per_sample_logits: list[torch.Tensor] = []
    per_sample_indices: list[torch.Tensor] = []
    per_sample_log_z: list[torch.Tensor] = []
    per_sample_gt_in_topk: list[torch.Tensor] = []
    has_gt_in_topk: Optional[bool] = None

    for sample_entry in per_sample_entries:
        shards = sample_entry.get("teacher_shards", [sample_entry])
        shards = sorted(shards, key=lambda shard: int(shard["global_seq_start"]))
        if not shards:
            raise ValueError("Sparse teacher IPC sample has no sequence shards.")
        shard_has_gt = ["gt_in_topk_ipc" in shard for shard in shards]
        if len(set(shard_has_gt)) != 1:
            raise ValueError(
                "Sparse teacher IPC handles mix gt_in_topk and non-gt_in_topk "
                "payloads within one sample."
            )
        if has_gt_in_topk is None:
            has_gt_in_topk = shard_has_gt[0]
        elif has_gt_in_topk != shard_has_gt[0]:
            raise ValueError(
                "Sparse teacher IPC handles mix gt_in_topk and non-gt_in_topk "
                "payloads across samples."
            )

        expected_start = 0
        for shard in shards:
            shard_start = int(shard["global_seq_start"])
            if shard_start != expected_start:
                raise ValueError(
                    "Sparse teacher CP shards must cover the sequence "
                    f"contiguously; expected start {expected_start}, got "
                    f"{shard_start}."
                )
            expected_start += int(shard["topk_shape"][0])
        full_seq_len = int(shards[0]["full_seq_len"])
        if expected_start != full_seq_len:
            raise ValueError(
                "Sparse teacher CP shards do not cover the full sequence: "
                f"covered={expected_start}, full_seq_len={full_seq_len}."
            )

        per_sample_logits.append(
            torch.cat(
                [
                    rebuild_cuda_tensor_from_ipc(
                        shard["topk_logits_ipc"], device
                    ).detach()
                    for shard in shards
                ],
                dim=0,
            )
        )
        per_sample_indices.append(
            torch.cat(
                [
                    rebuild_cuda_tensor_from_ipc(
                        shard["topk_indices_ipc"], device
                    ).detach()
                    for shard in shards
                ],
                dim=0,
            )
        )
        per_sample_log_z.append(
            torch.cat(
                [
                    rebuild_cuda_tensor_from_ipc(shard["log_z_ipc"], device).detach()
                    for shard in shards
                ],
                dim=0,
            )
        )
        if has_gt_in_topk:
            per_sample_gt_in_topk.append(
                torch.cat(
                    [
                        rebuild_cuda_tensor_from_ipc(
                            shard["gt_in_topk_ipc"], device
                        ).detach()
                        for shard in shards
                    ],
                    dim=0,
                )
            )

    gt_in_topk = (
        torch.stack(per_sample_gt_in_topk, dim=0).to(torch.bool)
        if has_gt_in_topk
        else None
    )
    return (
        torch.stack(per_sample_logits, dim=0).float(),
        torch.stack(per_sample_indices, dim=0).to(torch.int32),
        torch.stack(per_sample_log_z, dim=0).float(),
        gt_in_topk,
    )


def valid_chunk_mask(
    s_sizes: torch.Tensor,
    t_sizes: torch.Tensor,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-chunk validity gate: both sides non-empty and pair is valid."""
    return (s_sizes > 0) & (t_sizes > 0) & pair_valid


def parse_projection_file(
    path: Union[str, os.PathLike],
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Parse a projection-matrix file into COO components.

    Detects either the dense top-k format (``dict["indices"]`` /
    ``dict["likelihoods"]``) or the sparse multi-token format
    (``dict[(student_id, teacher_id)] -> count``) and converts both to
    a uniform COO representation.

    The function does **not** apply any sizing or validity policy: the
    ``-1`` sentinel used by ``_exact_map_remapped`` projection files is
    preserved in the returned ``indices``, and the inferred vocab sizes
    are derived from the file alone (caller may override them upward
    against tokenizer / config knowledge). This keeps a single parser
    while letting :mod:`token_aligner` and the loss fn keep their own
    clipping rules.

    Args:
        path: Path to a ``torch.save``d projection-matrix file.

    Returns:
        indices: ``LongTensor[2, nnz]`` — ``(student_idx, teacher_idx)``.
        values:  ``FloatTensor[nnz]``.
        v_student_inferred: ``int`` — dense format: row count; sparse
            format: ``max(student_idx) + 1``.
        v_teacher_inferred: ``int`` — ``max(positive teacher_idx) + 1``
            (``0`` if no positive entries exist).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the file is not in a recognized format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection matrix file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "indices" in data and "likelihoods" in data:
        # Dense top-k format: indices [V_s, top_k] holds teacher token ids;
        # likelihoods [V_s, top_k] holds the projection weights. Unfold to
        # COO so downstream code uses a uniform sparse-matmul path.
        top_indices: torch.Tensor = data["indices"].long()
        top_likelihoods: torch.Tensor = data["likelihoods"].float()
        if top_indices.shape != top_likelihoods.shape:
            raise ValueError(
                f"indices/likelihoods shape mismatch in {path}: "
                f"{top_indices.shape} vs {top_likelihoods.shape}"
            )
        v_student, top_k = top_indices.shape
        student_idx = torch.arange(v_student).unsqueeze(1).expand(-1, top_k).reshape(-1)
        teacher_idx = top_indices.reshape(-1)
        values = top_likelihoods.reshape(-1)
        indices = torch.stack([student_idx, teacher_idx], dim=0)
        positive = teacher_idx[teacher_idx >= 0]
        v_teacher = int(positive.max().item()) + 1 if positive.numel() > 0 else 0
        return indices, values, int(v_student), v_teacher

    if isinstance(data, dict) and all(
        isinstance(k, tuple) and len(k) == 2 for k in data.keys()
    ):
        # Sparse multi-token format: dict[(student_id, teacher_id)] -> count.
        keys = list(data.keys())
        values_list = list(data.values())
        student_idx = torch.tensor([k[0] for k in keys], dtype=torch.long)
        teacher_idx = torch.tensor([k[1] for k in keys], dtype=torch.long)
        indices = torch.stack([student_idx, teacher_idx], dim=0)
        values = torch.tensor(values_list, dtype=torch.float32)
        v_student = int(student_idx.max().item()) + 1 if student_idx.numel() > 0 else 0
        v_teacher = int(teacher_idx.max().item()) + 1 if teacher_idx.numel() > 0 else 0
        return indices, values, v_student, v_teacher

    raise ValueError(
        f"Unrecognized projection matrix format at {path}; expected dict "
        f"with 'indices'/'likelihoods' tensors or "
        f"dict[(student_id, teacher_id)] -> count."
    )


# Process-local projection-matrix caches. Each Ray worker / dataloader
# process has its own Python interpreter, so these dicts are effectively
# worker-local: a cache miss on one worker doesn't fill caches on other
# workers, and the driver process — which never enters a forward / loss
# path — never populates them.
#
# Keyed by ``(path, device, student_vocab_size, teacher_vocab_size)`` for
# the sparse cache because the sparse-COO shape's ``V_s`` and ``V_t`` are
# both sized from the configured vocab sizes; same path with a different
# size would build a different tensor. The top-k cache key is
# ``(path, device)`` — the raw top-k arrays don't depend on a vocab-size
# knob.
_SPARSE_PROJECTION_CACHE: dict[Tuple[str, torch.device, int, int], torch.Tensor] = {}
_TOPK_PROJECTION_CACHE: dict[
    Tuple[str, torch.device], Tuple[torch.Tensor, torch.Tensor]
] = {}


def get_sparse_projection_matrix(
    path: Union[str, os.PathLike],
    device: torch.device,
    *,
    student_vocab_size: int,
    teacher_vocab_size: int,
) -> torch.Tensor:
    """Return the sparse-COO projection matrix on ``device`` (cached).

    On a cache miss, parses the file via :func:`parse_projection_file`,
    drops ``-1`` teacher sentinels (illegal in sparse-COO), sizes
    ``V_s = max(student_vocab_size, max_observed_student_idx + 1)`` and
    ``V_t = max(teacher_vocab_size, max_observed_teacher_idx + 1)``, and
    builds a coalesced ``torch.sparse_coo_tensor`` on ``device``.
    Subsequent calls with the same
    ``(path, device, student_vocab_size, teacher_vocab_size)`` return the
    cached tensor — no disk I/O, no re-materialization.

    Both vocab sizes are keyword-only to prevent a positional swap (two
    same-magnitude ints, no error if confused).

    Args:
        path: Path to a ``torch.save``d projection-matrix file.
        device: Device the sparse tensor must live on.
        student_vocab_size: Minimum width of the student-side axis.
        teacher_vocab_size: Minimum width of the teacher-side axis.

    Returns:
        ``torch.sparse_coo_tensor`` of shape ``(V_s, V_t)``, coalesced,
        ``dtype=float32``.
    """
    key = (
        str(path),
        device,
        int(student_vocab_size),
        int(teacher_vocab_size),
    )
    cached = _SPARSE_PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    indices, values, _v_student, _ = parse_projection_file(path)
    # `_exact_map_remapped` projection files use -1 as a padding
    # sentinel for student rows that have fewer than top_k teacher
    # mappings. A negative column index is illegal in a sparse tensor
    # and causes CUDA illegal-memory-access in sparse.mm (forward and
    # backward). We drop those entries entirely.
    keep = indices[1] >= 0
    indices = indices[:, keep]
    values = values[keep]
    # Size both axes from the configured tokenizer vocabs, not from the
    # highest ids observed in the projection file. The sparse format
    # only stores entries for (student_id, teacher_id) pairs that
    # appeared during projection prep, so the highest valid vocab ids
    # may be absent. Sizing V_s from `max(observed student_id)+1` would
    # then make V_s < logits.shape[-1] and silently break the sparse
    # matmul; the symmetric concern on V_t lets the P-KL global top-k
    # gather go out of bounds. We clamp up against the projection's
    # observed max as a defensive fallback in case the file happens to
    # cover ids beyond the configured size.
    projection_max_student = (
        int(indices[0].max().item()) + 1 if indices.numel() > 0 else 0
    )
    projection_max_teacher = (
        int(indices[1].max().item()) + 1 if indices.numel() > 0 else 0
    )
    v_student = max(int(student_vocab_size), projection_max_student)
    v_teacher = max(int(teacher_vocab_size), projection_max_teacher)

    sparse = torch.sparse_coo_tensor(
        indices,
        values,
        (v_student, v_teacher),
        device=device,
        dtype=torch.float32,
    ).coalesce()
    _SPARSE_PROJECTION_CACHE[key] = sparse
    return sparse


def get_topk_projection(
    path: Union[str, os.PathLike],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the dense top-k ``(indices, likelihoods)`` projection on ``device`` (cached).

    Used by the gold-loss exact-map builder, which needs the per-row
    top-k weights — the sparse ``dict[(s, t)] -> count`` projection
    format doesn't carry those, so this loader rejects it.

    Args:
        path: Path to a ``torch.save``d projection-matrix file.
        device: Device the returned tensors must live on.

    Returns:
        ``(indices, likelihoods)`` — ``LongTensor[V_s, top_k]`` and
        ``FloatTensor[V_s, top_k]`` on ``device``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the file is not in the dense top-k format.
    """
    key = (str(path), device)
    cached = _TOPK_PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection matrix file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(data, dict) and "indices" in data and "likelihoods" in data):
        raise ValueError(
            f"gold_loss requires the dense projection-matrix format "
            f"(dict with 'indices' and 'likelihoods' tensors). File "
            f"{path} uses an unsupported format."
        )
    indices = data["indices"].long().to(device)
    likelihoods = data["likelihoods"].float().to(device)
    result = (indices, likelihoods)
    _TOPK_PROJECTION_CACHE[key] = result
    return result


# Process-local cache. Keyed by every input that affects the partition:
# the same file with a different ``xtoken_loss`` or ``teacher_vocab_size``
# would yield a different partition. Lives alongside
# ``_TOPK_PROJECTION_CACHE`` so the gold-loss build is amortized to one
# pass per (path, device, knob) on each worker.
_EXACT_TOKEN_MAP_CACHE: dict[
    Tuple[str, torch.device, bool, int], Dict[str, torch.Tensor]
] = {}


def build_exact_token_map(
    path: Union[str, os.PathLike],
    device: torch.device,
    *,
    xtoken_loss: bool,
    teacher_vocab_size: int,
) -> Dict[str, torch.Tensor]:
    """Build the common/uncommon vocab partition for the gold path (cached).

    Reads the dense projection arrays via :func:`get_topk_projection`, sorts each
    student row's projection weights descending, then picks an exact-token
    map per the ``xtoken_loss`` flag:

    - ``xtoken_loss=False`` (strict): ``has_exact_map = sorted_values[:, 0] == 1.0``.
      On collision (multiple students mapping to the same teacher id),
      the earliest (lowest) student index wins.
    - ``xtoken_loss=True`` (relaxed): ``has_exact_map = sorted_values[:, 0] >= 0.6``.
      On collision, the student with the highest first-projection
      weight wins; ties are broken by lowest student index.

    Both branches are vectorized via ``scatter_reduce`` so the build is
    O(V_s) and happens once per ``(path, device, xtoken_loss,
    teacher_vocab_size)`` for the run.

    Args:
        path: Path to a ``torch.save``d projection-matrix file (dense
            top-k format).
        device: Device the returned tensors must live on.
        xtoken_loss: Selects strict vs relaxed exact-map rule (see above).
        teacher_vocab_size: Width of the teacher-side vocab axis. The
            partition is bounded by this — teacher ids outside the range
            are dropped.

    Returns:
        Dict with keys ``common_student``, ``common_teacher`` (paired),
        ``uncommon_student``, ``uncommon_teacher`` (each independently
        sorted). All ``[long]`` tensors on ``device``.
    """
    key = (str(path), device, bool(xtoken_loss), int(teacher_vocab_size))
    cached = _EXACT_TOKEN_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    indices, likelihoods = get_topk_projection(path, device)
    v_student = indices.shape[0]
    v_teacher = int(teacher_vocab_size)

    sorted_values, sorted_in_topk = torch.sort(likelihoods, dim=-1, descending=True)
    if xtoken_loss:
        has_exact_map = sorted_values[:, 0] >= 0.6
    else:
        # Strict: top-k entry with weight 1.0 (no probability mass leaks to
        # other teacher ids — the student token maps 1-to-1). The legacy
        # `_exact_map_remapped` files used `indices[:, 1] == -1` as a sentinel
        # for "no second mapping", but modern projection builders
        # (tools/x_token/minimal_projection_via_multitoken.py with
        # --enable-exact-match) leave indices[:, 1+] filled with valid ids whose
        # weight is 0. Do NOT re-add the sentinel guard: it silently empties the
        # common vocab for every such matrix (e.g. nemotron-nano-3.5 <- Qwen3-30B
        # went 68131 -> 0 common tokens, routing all 496k 1-to-1 chunks down the
        # mismatch path). The top1 == 1.0 check is sufficient, since by
        # normalization weight 1.0 on one entry means 0 elsewhere.
        has_exact_map = sorted_values[:, 0] == 1.0

    # Gather (s_idx, t_idx, prob) for each exact-map candidate.
    s_candidates = torch.where(has_exact_map)[0]
    if s_candidates.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        result = {
            "common_student": empty,
            "common_teacher": empty,
            "uncommon_student": torch.arange(v_student, device=device),
            "uncommon_teacher": torch.arange(v_teacher, device=device),
        }
        _EXACT_TOKEN_MAP_CACHE[key] = result
        return result

    t_candidates = indices[s_candidates, sorted_in_topk[s_candidates, 0]]
    prob_candidates = sorted_values[s_candidates, 0]

    in_bounds = (t_candidates >= 0) & (t_candidates < v_teacher)
    s_vec = s_candidates[in_bounds]
    t_vec = t_candidates[in_bounds]
    prob_vec = prob_candidates[in_bounds]

    # Strict mode: any candidate is eligible (first one wins).
    # Relaxed mode: only candidates whose prob ties the per-teacher max.
    if xtoken_loss:
        max_prob_per_t = torch.full(
            (v_teacher,),
            float("-inf"),
            device=device,
            dtype=prob_vec.dtype,
        )
        max_prob_per_t.scatter_reduce_(
            0, t_vec, prob_vec, reduce="amax", include_self=True
        )
        eligible = prob_vec >= max_prob_per_t[t_vec]
    else:
        eligible = torch.ones_like(t_vec, dtype=torch.bool)

    # For each teacher id, pick the smallest student index among the
    # eligible candidates. Sentinel = v_student so non-eligible rows
    # lose the amin reduction.
    sentinel = torch.tensor(v_student, dtype=s_vec.dtype, device=device)
    eligible_s = torch.where(eligible, s_vec, sentinel.expand_as(s_vec))
    min_s_per_t = torch.full((v_teacher,), v_student, device=device, dtype=s_vec.dtype)
    min_s_per_t.scatter_reduce_(0, t_vec, eligible_s, reduce="amin", include_self=True)
    winner_mask = eligible & (s_vec == min_s_per_t[t_vec])

    common_student = s_vec[winner_mask]
    common_teacher = t_vec[winner_mask]
    # Sort by student index so the paired arrays match.
    sort_perm = torch.argsort(common_student)
    common_student = common_student[sort_perm]
    common_teacher = common_teacher[sort_perm]

    common_s_mask = torch.zeros(v_student, dtype=torch.bool, device=device)
    common_s_mask[common_student] = True
    common_t_mask = torch.zeros(v_teacher, dtype=torch.bool, device=device)
    common_t_mask[common_teacher] = True
    uncommon_student = (~common_s_mask).nonzero(as_tuple=True)[0]
    uncommon_teacher = (~common_t_mask).nonzero(as_tuple=True)[0]

    result = {
        "common_student": common_student,
        "common_teacher": common_teacher,
        "uncommon_student": uncommon_student,
        "uncommon_teacher": uncommon_teacher,
    }
    _EXACT_TOKEN_MAP_CACHE[key] = result
    return result


def _chunk_ids_to_spans(chunk_id: torch.Tensor, max_pairs: int) -> torch.Tensor:
    """Contiguous ``[start, end)`` position span per chunk id.

    Args:
        chunk_id: ``[B, T]`` where each entry is the pair/chunk index the
            position belongs to (sentinel ``< 0`` or ``>= max_pairs`` for
            unaligned positions).
        max_pairs: chunk-id space width (the ``pair_valid`` slot count).

    Returns:
        ``[B, max_pairs, 2]`` long tensor; row ``k`` is ``[first, last + 1)``
        over the positions with ``chunk_id == k``. A chunk id absent from a
        sample gets ``[0, 0)`` (zero-length — the v6 ``M/N > 0`` gate and
        ``pair_valid`` both skip it). Assumes each chunk's positions are
        contiguous (the offset-based cluster alignment guarantees this).
    """
    b, t = chunk_id.shape
    device = chunk_id.device
    pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
    valid = (chunk_id >= 0) & (chunk_id < max_pairs)
    idx = chunk_id.clamp(0, max_pairs - 1)
    # Invalid positions carry T (ignored by amin) / -1 (ignored by amax), so a
    # clamped-in bucket is never corrupted by an unaligned position.
    pos_for_min = torch.where(valid, pos, torch.full_like(pos, t))
    pos_for_max = torch.where(valid, pos, torch.full_like(pos, -1))
    first = torch.full((b, max_pairs), t, dtype=torch.long, device=device)
    last = torch.full((b, max_pairs), -1, dtype=torch.long, device=device)
    first.scatter_reduce_(1, idx, pos_for_min, reduce="amin", include_self=True)
    last.scatter_reduce_(1, idx, pos_for_max, reduce="amax", include_self=True)
    has = first < t
    spans = torch.zeros(b, max_pairs, 2, dtype=torch.long, device=device)
    spans[:, :, 0] = torch.where(has, first, torch.zeros_like(first))
    spans[:, :, 1] = torch.where(has, last + 1, torch.zeros_like(last))
    return spans


def loss_replica_group(
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> Optional[torch.distributed.ProcessGroup]:
    """Group over which to reduce the loss's global normalizers, or ``None``.

    The loss needs a "sum one contribution per distinct sample shard" group. The
    obvious choice, ``torch.distributed.group.WORLD``, **deadlocks under pipeline
    parallelism**: only the last stage runs the loss, so a WORLD collective is
    never joined by the earlier stages and every rank hangs until the NCCL
    watchdog fires.

    Megatron's data-parallel(-with-CP) group is the right group instead: its
    members all share the same pipeline-stage / tensor-parallel coordinates, so
    it is confined to the stage that actually runs the loss, and it spans exactly
    the DP x CP axes the normalizers want. Returns ``None`` when Megatron is not
    the active backend (DTensor, or a single-process / CPU test), leaving callers
    on their existing WORLD path.
    """
    if cp_group is None or not torch.distributed.is_initialized():
        return None
    try:
        # Local import keeps the optional Megatron dependency boundary intact;
        # a non-Megatron caller falls back to None.
        from megatron.core import parallel_state
    except ImportError:
        return None
    if not parallel_state.is_initialized():
        return None
    return parallel_state.get_data_parallel_group(with_context_parallel=True)


def _student_seq_to_contiguous_window(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    data_is_cp_sharded: bool,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Relay a student-sequence data tensor to this CP rank's contiguous window.

    The two training backends hand the loss different layouts for the
    student-sequence entries of the microbatch dict:

    * DTensor worker: the buffers were CP-sharded in place by PyTorch's
      ``context_parallel``, i.e. they are in the *load-balanced* (``2*cp``
      interleaved) layout and must be all-gathered before the contiguous window
      can be sliced out (``data_is_cp_sharded=True``).
    * Megatron worker: only the model input and the returned logits are
      CP-sharded; the microbatch dict the loss sees still holds the FULL
      ``[B, S]`` tensors, so this rank's window is a plain ``narrow``
      (``data_is_cp_sharded=False``).

    No-op without a CP group (or at CP world size 1), so both single-GPU paths
    are byte-identical to before.
    """
    if cp_group is None or torch.distributed.get_world_size(cp_group) <= 1:
        return x
    if data_is_cp_sharded:
        return cp_load_balanced_to_contiguous(x, cp_group=cp_group, seq_dim=seq_dim)
    local = to_local_if_dtensor(x)
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    local_len = local.shape[seq_dim] // cp_size
    return local.narrow(seq_dim, cp_rank * local_len, local_len).contiguous()


def prepare_xtoken_cross_tokenizer_loss_input(
    logits: torch.Tensor,
    data: Mapping[str, Any],
    *,
    projection_matrix_paths: list[Optional[str]],
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_sharder: Optional["ContextParallelSharder"] = None,
) -> tuple[
    torch.Tensor,
    Dict[int, torch.Tensor],
    Dict[int, SparseTeacherLogits],
    Dict[int, LocalizedAlignment],
    Dict[int, int],
    Optional[torch.distributed.ProcessGroup],
    Optional[torch.distributed.ProcessGroup],
    Optional[torch.distributed.ProcessGroup],
]:
    """Build the per-teacher cross-tokenizer distillation loss pieces from student logits + IPC teacher data.

    Rebuilds each teacher's dense full-vocab or sparse top-k + logZ logits from
    its per-rank CUDA IPC handles and does the shared CP-resolution the loss
    needs. The contiguous student logits / input_ids / token_mask are relaid
    once and shared across teachers. Per teacher, a localized alignment is built: a
    cross-tokenizer teacher (``projection_matrix_paths[i]`` set) gets the
    localized, next-token-shifted chunk alignment from its ``alignment_{i}_*``
    keys; a same-tokenizer teacher (``None`` path) gets a thin alignment carrying
    only the shared student fields (identity 1:1 token alignment, no chunks).
    TP/CP groups come from the student ``logits``' device mesh, falling back to
    the passed groups for non-DTensor logits.

    Args:
        projection_matrix_paths: Per-teacher projection paths. Its length is the
            teacher count and drives the ``teacher_{i}_*`` / ``alignment_{i}_*``
            keys read here; a ``None`` entry marks a same-tokenizer teacher.
        cp_sharder: Automodel's model-owned sequence layout. When provided, it
            replaces the legacy load-balanced CP relayout for student tensors.

    Returns:
        ``(student_logits_contig, teacher_full_logits_by_idx,
        teacher_sparse_logits_by_idx, aligns_by_idx,
        dense_reconstruction_fallbacks_by_idx, tp_group, cp_group,
        dp_cp_group)``. The reconstruction counts come from the actual dense
        rebuild branch. ``dp_cp_group`` is the group the loss reduces its global
        normalizers over; see :func:`loss_replica_group` for why it must not be
        ``WORLD`` under pipeline parallelism.
    """
    if isinstance(logits, DTensor):
        mesh = logits.device_mesh
        mesh_names = mesh.mesh_dim_names or ()
        tp_group = mesh.get_group("tp") if "tp" in mesh_names else None
        cp_group = (
            context_parallel_group
            if cp_sharder is not None
            else (mesh.get_group("cp") if "cp" in mesh_names else None)
        )
        # The DTensor worker CP-shards the microbatch dict itself (torch's
        # ``context_parallel(buffers=...)``) on the legacy path. The model-owned
        # sharder path restores those buffers before loss preparation.
        data_is_cp_sharded = cp_sharder is None
    else:
        cp_group = context_parallel_group
        tp_group = vocab_parallel_group
        data_is_cp_sharded = False

    device = torch.cuda.current_device()

    # Student CP relay is computed once and shared by every teacher's KD term
    # and the next-token-accuracy metric.
    # Automodel restores its own model layout before NeMo RL selects the
    # contiguous IPC-consumer window. Legacy callers retain the existing
    # load-balanced-to-contiguous conversion.
    if cp_sharder is not None:
        # Keep the student-logit autograd graph intact: to_local_if_dtensor()
        # runs DTensor.to_local() under torch.no_grad(), which would detach the
        # X-token KD loss from the student model for TP > 1.
        local_logits = logits.to_local() if isinstance(logits, DTensor) else logits
        full_student_logits = cp_sharder.gather_token_tensor(
            local_logits, seq_dim=1, trim=True
        )
        cp_size = (
            torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
        )
        full_student_seq_len = full_student_logits.shape[1]
        if full_student_seq_len % cp_size != 0:
            raise ValueError(
                "X-token student sequence length must be divisible by the student "
                "context parallel size, but got "
                f"sequence_length={full_student_seq_len}, cp_size={cp_size}. "
                "Set policy.make_sequence_length_divisible_by to a multiple of "
                "policy.dtensor_cfg.context_parallel_size."
            )
        cp_rank = torch.distributed.get_rank(cp_group) if cp_group is not None else 0
        student_seq_len = full_student_seq_len // cp_size
        student_seq_start = cp_rank * student_seq_len
        student_logits_contig = full_student_logits.narrow(
            1, student_seq_start, student_seq_len
        ).contiguous()
        student_input_ids = to_local_if_dtensor(data["input_ids"])[
            :, student_seq_start : student_seq_start + student_seq_len
        ].contiguous()
        student_token_mask = to_local_if_dtensor(data["token_mask"])[
            :, student_seq_start : student_seq_start + student_seq_len
        ].contiguous()
        student_kd_token_mask = to_local_if_dtensor(
            data.get("kd_token_mask", data["token_mask"])
        )[:, student_seq_start : student_seq_start + student_seq_len].contiguous()
    else:
        # Logits are load-balanced on both backends. Student-sequence data is
        # load-balanced only for the legacy DTensor path; Megatron leaves it
        # full, so select its contiguous window directly.
        student_logits_contig = cp_load_balanced_to_contiguous(
            logits, cp_group=cp_group
        )
        student_input_ids = _student_seq_to_contiguous_window(
            data["input_ids"], cp_group, data_is_cp_sharded=data_is_cp_sharded
        )
        student_token_mask = _student_seq_to_contiguous_window(
            data["token_mask"], cp_group, data_is_cp_sharded=data_is_cp_sharded
        )
        student_kd_token_mask = _student_seq_to_contiguous_window(
            data.get("kd_token_mask", data["token_mask"]),
            cp_group,
            data_is_cp_sharded=data_is_cp_sharded,
        )
    sample_mask = to_local_if_dtensor(data["sample_mask"])

    teacher_full_logits_by_idx: Dict[int, torch.Tensor] = {}
    teacher_sparse_logits_by_idx: Dict[int, SparseTeacherLogits] = {}
    aligns_by_idx: Dict[int, LocalizedAlignment] = {}
    dense_reconstruction_fallbacks_by_idx: Dict[int, int] = {}
    student_cp_size = (
        torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
    )
    for i, proj_path in enumerate(projection_matrix_paths):
        sparse_key = f"teacher_{i}_sparse_logits_ipc"
        full_key = f"teacher_{i}_full_logits_ipc"
        has_sparse_logits = sparse_key in data
        has_full_logits = full_key in data
        if has_sparse_logits == has_full_logits:
            raise ValueError(
                f"Teacher {i} must provide exactly one dense or sparse logits "
                f"IPC payload; dense={has_full_logits}, sparse={has_sparse_logits}."
            )
        if has_sparse_logits:
            if proj_path is None:
                raise ValueError(
                    f"Same-vocab teacher {i} cannot use sparse xToken IPC."
                )
            teacher_sparse_logits = rebuild_teacher_sparse_logits_from_ipc(
                data[sparse_key],
                device=device,
            )
            teacher_sparse_logits_by_idx[i] = teacher_sparse_logits
            full_teacher_seq_len = int(teacher_sparse_logits[0].shape[1])
            if full_teacher_seq_len % student_cp_size != 0:
                raise ValueError(
                    "Sparse teacher sequence length must be divisible by the "
                    f"student CP size; teacher={i}, seq={full_teacher_seq_len}, "
                    f"student_cp={student_cp_size}."
                )
            teacher_seq_len = full_teacher_seq_len // student_cp_size
        else:
            (
                teacher_full_logits,
                dense_reconstruction_fallbacks,
            ) = rebuild_teacher_full_logits_from_ipc(
                data[full_key],
                cp_group=cp_group,
                device=device,
            )
            teacher_full_logits_by_idx[i] = teacher_full_logits
            dense_reconstruction_fallbacks_by_idx[i] = dense_reconstruction_fallbacks
            teacher_seq_len = int(teacher_full_logits.shape[1])
        if proj_path is None:
            # Same-tokenizer teacher: identity token alignment, no chunk
            # localization. Carry only the shared student fields.
            aligns_by_idx[i] = LocalizedAlignment(
                sample_mask=sample_mask,
                student_input_ids=student_input_ids,
                student_token_mask=student_token_mask,
                student_kd_token_mask=student_kd_token_mask,
            )
            continue
        alignment_prefix = f"alignment_{i}_"
        if cp_sharder is not None:
            student_chunk_id_source_full = to_local_if_dtensor(
                data[f"{alignment_prefix}student_chunk_id"]
            )
            student_chunk_id_contig = student_chunk_id_source_full[
                :, student_seq_start : student_seq_start + student_seq_len
            ].contiguous()

            teacher_chunk_id_source_full = to_local_if_dtensor(
                data[f"{alignment_prefix}teacher_chunk_id"]
            )
            teacher_seq_start = cp_rank * teacher_seq_len
            teacher_chunk_id_contig = teacher_chunk_id_source_full[
                :, teacher_seq_start : teacher_seq_start + teacher_seq_len
            ].contiguous()
            align = LocalizedAlignment(
                sample_mask=sample_mask,
                pair_valid=to_local_if_dtensor(data[f"{alignment_prefix}pair_valid"]),
                pair_is_correct=to_local_if_dtensor(
                    data[f"{alignment_prefix}pair_is_correct"]
                ),
            )
        else:
            align = localize_alignment(
                data,
                teacher_seq_len=teacher_seq_len,
                alignment_prefix=alignment_prefix,
                cp_group=cp_group,
            )
            student_chunk_id_contig = _student_seq_to_contiguous_window(
                align.student_chunk_id,
                cp_group,
                data_is_cp_sharded=data_is_cp_sharded,
            )
            teacher_chunk_id_contig = align.teacher_chunk_id
            cp_rank = (
                torch.distributed.get_rank(cp_group)
                if cp_group is not None
                and torch.distributed.get_world_size(cp_group) > 1
                else 0
            )
            teacher_seq_start = cp_rank * teacher_seq_len

        # Contiguous, UNSHIFTED chunk ids -> per-chunk position spans for the v6
        # path. v6 reads spans and applies its own per-chunk kl_chunk_shift, so
        # it must NOT see the P-KL/gold next-token-shifted chunk ids.
        max_pairs = align.pair_valid.shape[1]
        # GLOBAL spans: the v6 KD term gathers the student/teacher logits to the
        # full sequence, so its spans must index global positions. Gather the
        # contiguous-window chunk ids to the full sequence before deriving spans
        # (no-op at CP=1).
        if cp_sharder is not None:
            # Automodel has already restored these data tensors to the full,
            # canonical sequence before the local contiguous window is sliced.
            student_chunk_id_global = student_chunk_id_source_full
            teacher_chunk_id_global = teacher_chunk_id_source_full
        else:
            student_chunk_id_global = allgather_cp_contiguous_tensor(
                student_chunk_id_contig, cp_group
            )
            teacher_chunk_id_global = allgather_cp_contiguous_tensor(
                teacher_chunk_id_contig, cp_group
            )
        align.student_spans = _chunk_ids_to_spans(student_chunk_id_global, max_pairs)
        align.teacher_spans = _chunk_ids_to_spans(teacher_chunk_id_global, max_pairs)
        # Preserve the aligner's real per-sample chunk count rather than
        # iterating every padded pair slot; pair_valid remains the final gate.
        align.num_chunks = to_local_if_dtensor(data[f"{alignment_prefix}num_chunks"])
        # Teacher input ids for this CP rank's contiguous teacher window (matches
        # the teacher-logit / teacher_chunk_id slice).
        teacher_ids_full = to_local_if_dtensor(data[f"teacher_{i}_input_ids"])
        align.teacher_input_ids = teacher_ids_full[
            :, teacher_seq_start : teacher_seq_start + teacher_seq_len
        ]
        # Preserve the established next-token-shifted localized fields for
        # metrics and compatibility while v6 consumes the unshifted spans.
        if cp_sharder is not None:
            student_chunk_id_shifted = student_chunk_id_source_full.roll(
                shifts=-1, dims=1
            )
            student_chunk_id_shifted[:, -1] = -1
            align.student_chunk_id = student_chunk_id_shifted[
                :, student_seq_start : student_seq_start + student_seq_len
            ].contiguous()
            teacher_chunk_id_shifted = teacher_chunk_id_source_full.roll(
                shifts=-1, dims=1
            )
            teacher_chunk_id_shifted[:, -1] = -1
            align.teacher_chunk_id = teacher_chunk_id_shifted[
                :, teacher_seq_start : teacher_seq_start + teacher_seq_len
            ].contiguous()
        else:
            align.student_chunk_id = cp_shift_next(
                student_chunk_id_contig, cp_group, fill=-1
            )
            align.teacher_chunk_id = cp_shift_next(
                teacher_chunk_id_contig, cp_group, fill=-1
            )
        align.student_input_ids = student_input_ids
        align.student_token_mask = student_token_mask
        align.student_kd_token_mask = student_kd_token_mask
        aligns_by_idx[i] = align
    return (
        student_logits_contig,
        teacher_full_logits_by_idx,
        teacher_sparse_logits_by_idx,
        aligns_by_idx,
        dense_reconstruction_fallbacks_by_idx,
        tp_group,
        cp_group,
        loss_replica_group(cp_group),
    )
