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
    - :func:`parse_projection_file` — single source of truth for
      reading the on-disk projection matrix file (both the dense top-k
      format and the sparse ``dict[(s, t)] -> count`` format) into COO
      components. Callers retain their own validation / sizing rules.
    - :func:`get_sparse_projection_matrix`, :func:`get_topk_projection`
      — process-local lazy caches for the materialized projection
      matrix on a given device. Driver processes never trigger a fill;
      each Ray worker populates its own cache on the first loss call.
"""

from __future__ import annotations

import os
from typing import Any, Tuple, Union

import torch


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

    Math matches PT reference ``project_token_likelihoods_ultra_fast``:
    autograd's builtin sparse-mm backward computes the same
    ``M @ grad_out``. The gradient w.r.t. the sparse argument isn't
    needed (the projection matrix is frozen), so it's returned as ``None``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        ctx: Any, sparse_M: torch.Tensor, dense: torch.Tensor
    ) -> torch.Tensor:
        ctx.sparse_M = sparse_M
        return torch.sparse.mm(sparse_M.t(), dense)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
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
    divides by chunk sizes. Both inputs and outputs match PT's
    chunk-averaging math at ``tokenalign.py:3617–3637``.

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
    chunk_sums = torch.bmm(chunk_mask_f, log_probs)        # [B, C, V]
    chunk_sizes = chunk_mask.sum(dim=1).float()            # [B, C]
    chunk_log_probs = chunk_sums / (chunk_sizes.unsqueeze(-1) + eps)
    return chunk_log_probs, chunk_sizes


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
        student_idx = (
            torch.arange(v_student).unsqueeze(1).expand(-1, top_k).reshape(-1)
        )
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
        v_student = (
            int(student_idx.max().item()) + 1 if student_idx.numel() > 0 else 0
        )
        v_teacher = (
            int(teacher_idx.max().item()) + 1 if teacher_idx.numel() > 0 else 0
        )
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
# Keyed by ``(path, device, teacher_vocab_size)`` for the sparse cache
# because the sparse-COO shape's ``V_t`` is sized from
# ``teacher_vocab_size``; same path with a different size would build a
# different tensor. The top-k cache key is ``(path, device)`` — the raw
# top-k arrays don't depend on a vocab-size knob.
_SPARSE_PROJECTION_CACHE: dict[
    Tuple[str, torch.device, int], torch.Tensor
] = {}
_TOPK_PROJECTION_CACHE: dict[
    Tuple[str, torch.device], Tuple[torch.Tensor, torch.Tensor]
] = {}


def get_sparse_projection_matrix(
    path: Union[str, os.PathLike],
    device: torch.device,
    teacher_vocab_size: int,
) -> torch.Tensor:
    """Return the sparse-COO projection matrix on ``device`` (cached).

    On a cache miss, parses the file via :func:`parse_projection_file`,
    drops ``-1`` teacher sentinels (illegal in sparse-COO), sizes
    ``V_t = max(teacher_vocab_size, max_observed_teacher_idx + 1)``,
    and builds a coalesced ``torch.sparse_coo_tensor`` on ``device``.
    Subsequent calls with the same ``(path, device, teacher_vocab_size)``
    return the cached tensor — no disk I/O, no re-materialization.

    Args:
        path: Path to a ``torch.save``d projection-matrix file.
        device: Device the sparse tensor must live on.
        teacher_vocab_size: Minimum width of the teacher-side axis.

    Returns:
        ``torch.sparse_coo_tensor`` of shape ``(V_s, V_t)``, coalesced,
        ``dtype=float32``.
    """
    key = (str(path), device, int(teacher_vocab_size))
    cached = _SPARSE_PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    indices, values, v_student, _ = parse_projection_file(path)
    # `_exact_map_remapped` projection files use -1 as a padding
    # sentinel for student rows that have fewer than top_k teacher
    # mappings. A negative column index is illegal in a sparse tensor
    # and causes CUDA illegal-memory-access in sparse.mm (forward and
    # backward). PT's tokenalign clamps to col 0 and zeros the value;
    # we drop those entries entirely.
    keep = indices[1] >= 0
    indices = indices[:, keep]
    values = values[keep]
    # Use the teacher's full vocab size as V_t — not max(teacher_idx)+1.
    # The P-KL global top-k pick happens over the teacher's full vocab,
    # including ids the projection doesn't cover. Sizing the projected
    # output to the full teacher vocab makes those columns all-zero
    # (correct semantics: unmapped teacher tokens get zero projected
    # probability) and keeps the gather in bounds.
    projection_max_teacher = (
        int(indices[1].max().item()) + 1 if indices.numel() > 0 else 0
    )
    v_teacher = max(int(teacher_vocab_size), projection_max_teacher)

    sparse = torch.sparse_coo_tensor(
        indices, values, (v_student, v_teacher),
        device=device, dtype=torch.float32,
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
    if not (
        isinstance(data, dict)
        and "indices" in data
        and "likelihoods" in data
    ):
        raise ValueError(
            f"gold_loss requires the dense projection-matrix format "
            f"(dict with 'indices' and 'likelihoods' tensors). File "
            f"{path} uses an unsupported format."
        )
    indices = data["indices"].long().to(device)
    likelihoods = data["likelihoods"].float().to(device)
    _TOPK_PROJECTION_CACHE[key] = (indices, likelihoods)
    return indices, likelihoods
