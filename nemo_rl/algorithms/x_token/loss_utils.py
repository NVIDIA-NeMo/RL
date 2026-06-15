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

Hosts pieces that are used by both :mod:`token_aligner` (in this package)
and :mod:`nemo_rl.algorithms.loss.loss_functions`:

    - :class:`Fp32SparseMM` — FP32 sparse-dense matmul autograd Function
      that ignores the surrounding BF16 autocast (PyTorch has no BF16
      sparse-mm kernel).
    - :func:`chunk_average_log_probs`, :func:`valid_chunk_mask` —
      chunk-aggregation helpers for the cross-tokenizer KL paths.
    - :func:`dp_all_reduce_sum` — sum-reduce a scalar count across the
      data-parallel group so the chunk-KL denominator is the global
      valid-chunk count rather than a per-rank mean.
    - :func:`parse_projection_file` — single source of truth for
      reading the on-disk projection matrix file (both the dense top-k
      format and the sparse ``dict[(s, t)] -> count`` format) into COO
      components. Callers retain their own validation / sizing rules.
    - :func:`get_sparse_projection_matrix`, :func:`get_topk_projection`
      — process-local lazy caches for the materialized projection
      matrix on a given device. Driver processes never trigger a fill;
      each Ray worker populates its own cache on the first loss call.
    - :func:`build_exact_token_map` — derived common/uncommon vocab
      partition for the gold-loss path. Cached per
      ``(path, device, xtoken_loss, teacher_vocab_size)`` because the
      partition depends on those four inputs.
    - :func:`alignment_from_flat_batch` — rehydrate the flat
      ``alignment_*`` transport keys on the loss data dict into a single
      :class:`AlignmentBatch` so the loss bodies access alignment via
      attributes instead of repeating flat field names.
"""

from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, Dict, Mapping, Tuple, Union

import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch


def alignment_from_flat_batch(
    data: Mapping[str, Any], prefix: str = "alignment_"
) -> AlignmentBatch:
    """Rebuild :class:`AlignmentBatch` from the flat ``<prefix>*`` keys.

    The field set is driven off :class:`AlignmentBatch` so the helper
    can't drift from the schema.

    Args:
        data: The loss data dict carrying the flat alignment keys.
        prefix: Key prefix to read. Defaults to ``"alignment_"`` (the
            single-teacher layout). Multi-teacher callers pass
            ``f"alignment_{i}_"`` so teacher ``i`` reads its own alignment
            payload rather than teacher 0's.
    """
    return AlignmentBatch(
        **{f.name: data[f"{prefix}{f.name}"] for f in fields(AlignmentBatch)}
    )


def rebuild_teacher_full_logits_from_ipc(
    data: Mapping[str, Any], key: str = "teacher_full_logits_ipc"
) -> torch.Tensor:
    """View-only rebuild of the microbatch's teacher-logits slice from IPC.

    The producer maintains a **persistent** IPC buffer on its GPU sized
    ``[B_r, T_t, V_t]``; the buffer (and the IPC handle it was captured
    with) survives across training steps, with fresh logits ``.copy_()``-ed
    in each step. Because the producer never frees the buffer between steps,
    holding a view into the IPC-imported storage is safe: the producer-side
    allocation isn't fighting the consumer's refcount, it's simply alive for
    the worker's lifetime.

    Every per-sample entry in ``teacher_full_logits_ipc`` carries the same
    stable rank-level handle plus its rank-local ``sample_idx_within_rank``.
    We rebuild that single handle once and slice ``[mb_start:mb_end]`` for
    the current microbatch -- zero allocation on the consumer, dtype
    preserved (the loss fn casts if/where it needs fp32).

    Args:
        data: The loss data dict, carrying the IPC handle list under ``key``
            -- a list of per-sample IPC handle dicts produced by
            ``Policy.get_full_logits_ipc``.
        key: Data-dict key holding the IPC handle list. Defaults to
            ``"teacher_full_logits_ipc"`` (single-teacher layout);
            multi-teacher callers pass ``f"teacher_{i}_full_logits_ipc"``.

    Returns:
        A ``[mb_B, T_t, V_t]`` view into the producer's GPU memory (no copy).
    """
    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    entries = data[key]
    consumer_device = torch.cuda.current_device()

    first = entries[0]
    last = entries[-1]
    rank_view = rebuild_cuda_tensor_from_ipc(
        first["rank_logits_ipc"], consumer_device
    )  # [B_r, T_t, V_t] view into producer's GPU memory

    mb_start = first["sample_idx_within_rank"]
    mb_end = last["sample_idx_within_rank"] + 1
    # Contract: BatchedDataDict.slice and shard_by_batch_size preserve
    # contiguous sample order, so sample_idx_within_rank is monotone in
    # the microbatch. If a future change ever reorders samples, fall back
    # to advanced indexing (which DOES copy -- defeats the no-copy win);
    # assert loudly instead of silently regressing.
    assert mb_end - mb_start == len(entries), (
        "expected contiguous monotonic sample_idx_within_rank within a "
        f"microbatch; got entries with indices "
        f"{[e['sample_idx_within_rank'] for e in entries]}"
    )
    return rank_view[mb_start:mb_end]  # [mb_B, T_t, V_t] view, no copy


def rebuild_teacher_topk_from_ipc(
    data: Mapping[str, Any], key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """View-only rebuild of the microbatch's top-K logits + indices from IPC.

    The two-buffer counterpart of :func:`rebuild_teacher_full_logits_from_ipc`:
    the producer keeps persistent ``[B_r, T, k]`` value and index buffers and
    refills them each step (``DTensorPolicyWorkerV2.get_teacher_logits_ipc`` with
    ``mode="topk"``). Each per-sample entry carries the same two rank-level
    handles plus its rank-local ``sample_idx_within_rank``; we rebuild the
    handles once and slice ``[mb_start:mb_end]`` for the current microbatch --
    zero allocation on the consumer.

    Values are shipped in native compute dtype (bf16); indices in int32 (the
    loss casts to int64 at the gather). Both are lossless relative to the bf16
    teacher forward.

    Args:
        data: The loss data dict carrying the IPC handle list under ``key``.
        key: Data-dict key holding the per-sample top-K IPC handle list (e.g.
            ``f"teacher_{i}_topk_ipc"``).

    Returns:
        ``(topk_logits, topk_indices)`` -- ``[mb_B, T, k]`` views into the
        producer's GPU memory (no copy).
    """
    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    entries = data[key]
    consumer_device = torch.cuda.current_device()

    first = entries[0]
    last = entries[-1]
    vals_view = rebuild_cuda_tensor_from_ipc(
        first["rank_topk_vals_ipc"], consumer_device
    )  # [B_r, T, k] view into producer GPU memory
    idx_view = rebuild_cuda_tensor_from_ipc(
        first["rank_topk_idx_ipc"], consumer_device
    )  # [B_r, T, k] view into producer GPU memory

    mb_start = first["sample_idx_within_rank"]
    mb_end = last["sample_idx_within_rank"] + 1
    # Same contiguity contract as the full-logits rebuild: advanced indexing
    # would copy and defeat the zero-copy win, so assert loudly instead.
    assert mb_end - mb_start == len(entries), (
        "expected contiguous monotonic sample_idx_within_rank within a "
        f"microbatch; got entries with indices "
        f"{[e['sample_idx_within_rank'] for e in entries]}"
    )
    return vals_view[mb_start:mb_end], idx_view[mb_start:mb_end]


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


def chunk_average_log_probs(
    log_probs: torch.Tensor,
    chunk_id: torch.Tensor,
    max_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``log_probs`` over the chunks defined by ``chunk_id``.

    Builds a one-hot chunk mask from ``chunk_id`` (``-1`` means "no
    chunk", contributes to no bucket), then ``bmm``-aggregates and
    divides by chunk sizes.

    Args:
        log_probs: ``[B, T, V]`` log-probabilities.
        chunk_id: ``[B, T]`` long tensor, values in ``[-1, max_chunks)``.
        max_chunks: number of chunk buckets.

    Returns:
        chunk_log_probs: ``[B, max_chunks, V]`` averaged log-probs.
        chunk_sizes:    ``[B, max_chunks]`` float tensor of bucket sizes.
    """
    eps = 1e-10
    device = log_probs.device
    chunk_arange = torch.arange(max_chunks, device=device).view(1, 1, -1)
    # [B, T, max_chunks] — -1 entries compare false everywhere.
    chunk_mask = chunk_id.unsqueeze(-1) == chunk_arange
    chunk_mask_f = chunk_mask.transpose(1, 2).to(log_probs.dtype)
    chunk_sums = torch.bmm(chunk_mask_f, log_probs)  # [B, C, V]
    chunk_sizes = chunk_mask.sum(dim=1).float()  # [B, C]
    chunk_log_probs = chunk_sums / (chunk_sizes.unsqueeze(-1) + eps)
    return chunk_log_probs, chunk_sizes


def valid_chunk_mask(
    s_sizes: torch.Tensor,
    t_sizes: torch.Tensor,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-chunk validity gate: both sides non-empty and pair is valid."""
    return (s_sizes > 0) & (t_sizes > 0) & pair_valid


def dp_all_reduce_sum(local: torch.Tensor) -> torch.Tensor:
    """Sum-reduce a scalar count across the data-parallel group.

    Used to compute ``global_valid_chunks`` from each rank's local
    chunk count, so the chunk-KL denominator matches the
    ``sum(global_valid_chunk_kl) / sum(global_valid_chunks)`` objective
    (the same convention CE follows via ``global_valid_toks``). The
    cross-tokenizer setup asserts ``tensor_parallel_size=1`` and
    ``context_parallel_size=1`` in
    ``xtoken_off_policy_distillation.setup``, so the default process
    group equals the DP group — calling all-reduce on the default group
    therefore sums across DP only.

    Returns a fresh ``float32`` scalar; the input tensor is not
    modified. Falls back to a copy of the local value when distributed
    is not initialized (unit tests).
    """
    out = local.detach().to(torch.float32).clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(out)
    return out


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

    - ``xtoken_loss=False`` (strict): ``has_exact_map = (sorted_values[:, 0] == 1.0) & (projection_indices[:, 1] == -1)``.
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
        # Strict: exactly one top-k entry with weight 1.0, no second
        # mapping. `indices[:, 1] == -1` is the sentinel used by the
        # `_exact_map_remapped` projection files for "no second
        # mapping".
        has_exact_map = (sorted_values[:, 0] == 1.0) & (indices[:, 1] == -1)

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
