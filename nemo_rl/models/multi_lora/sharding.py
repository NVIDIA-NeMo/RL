"""Data-parallel sharding helpers for Multi-LoRA batches."""

from __future__ import annotations

import torch


def rank_striped_indices(adapter_ids: torch.Tensor, dp_size: int) -> list[int]:
    """Return a rank-major permutation for block-contiguous adapter rows.

    Input rows are expected as equal adapter blocks, e.g. ``A*16, B*16``.
    For ``dp_size=8``, the returned order is ``rank0(A*2,B*2), ...,
    rank7(A*2,B*2)``. Stock contiguous DP sharding can then give every rank
    the same per-adapter row slice as an independent single-adapter run.
    """
    if not isinstance(adapter_ids, torch.Tensor) or adapter_ids.ndim != 1:
        raise TypeError("adapter_ids must be a 1-D torch.Tensor")
    if adapter_ids.dtype != torch.long:
        raise TypeError(f"adapter_ids must have dtype torch.long, got {adapter_ids.dtype}")
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if adapter_ids.numel() == 0:
        return []

    unique_ids, counts = torch.unique_consecutive(adapter_ids, return_counts=True)
    if unique_ids.numel() != torch.unique(adapter_ids).numel():
        raise ValueError("adapter_ids must be block-contiguous")
    if unique_ids.numel() <= 1:
        return list(range(adapter_ids.numel()))
    if not bool(torch.all(counts == counts[0])):
        raise ValueError(
            "rank-striped sharding requires equal rows per adapter; "
            f"got counts={counts.tolist()}"
        )

    rows_per_adapter = int(counts[0].item())
    if rows_per_adapter % dp_size != 0:
        raise ValueError(
            "rows per adapter must be divisible by DP size; "
            f"got rows={rows_per_adapter}, dp={dp_size}"
        )
    rows_per_rank = rows_per_adapter // dp_size

    indices: list[int] = []
    for rank_idx in range(dp_size):
        for adapter_idx in range(unique_ids.numel()):
            start = adapter_idx * rows_per_adapter + rank_idx * rows_per_rank
            indices.extend(range(start, start + rows_per_rank))
    return indices
