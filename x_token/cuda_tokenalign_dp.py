"""
CUDA DP implementation that matches TokenAligner's DP transition rules.

This module mirrors the move set used by:
`TokenAligner.align_tokens_with_combinations_numpy_jit`:
  - diag (match/mismatch)
  - up / left (gap)
  - comb_s1_over_s2_k (10 + k)
  - comb_s2_over_s1_k (20 + k)

It also mirrors the current chunked recursion strategy:
split at midpoint when sequence length exceeds `chunk_size`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load


_INVALID = np.int64(-1)


@lru_cache(maxsize=1)
def _load_cuda_ext():
    src = Path(__file__).with_name("cuda_tokenalign_dp_kernel.cu")
    return load(
        name="cuda_tokenalign_dp_ext",
        sources=[str(src)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def _build_ids_and_joined_tables(
    seq1: List[str],
    seq2: List[str],
    max_comb_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    token_to_id: dict[str, int] = {}
    next_id = 0

    def get_id(s: str) -> int:
        nonlocal next_id
        maybe = token_to_id.get(s)
        if maybe is None:
            maybe = next_id
            token_to_id[s] = maybe
            next_id += 1
        return maybe

    n1 = len(seq1)
    n2 = len(seq2)
    ids1 = np.array([get_id(t) for t in seq1], dtype=np.int64)
    ids2 = np.array([get_id(t) for t in seq2], dtype=np.int64)

    joined1 = np.full((n1 + 1, max_comb_len + 1), _INVALID, dtype=np.int64)
    for i in range(n1 + 1):
        for k in range(2, min(i, max_comb_len) + 1):
            joined1[i, k] = get_id("".join(seq1[i - k : i]))

    joined2 = np.full((n2 + 1, max_comb_len + 1), _INVALID, dtype=np.int64)
    for j in range(n2 + 1):
        for k in range(2, min(j, max_comb_len) + 1):
            joined2[j, k] = get_id("".join(seq2[j - k : j]))

    return ids1, ids2, joined1, joined2


def _backtrack_from_trace(
    seq1: List[str],
    seq2: List[str],
    trace_np: np.ndarray,
) -> List[Tuple[List[str], List[str], int, int, int, int]]:
    n1 = len(seq1)
    n2 = len(seq2)
    aligned: List[Tuple[List[str], List[str], int, int, int, int]] = []

    i, j = n1, n2
    while i > 0 or j > 0:
        move = int(trace_np[i, j])
        if move == 1:
            aligned.append(([seq1[i - 1]], [seq2[j - 1]], i - 1, i, j - 1, j))
            i -= 1
            j -= 1
        elif move == 2:
            aligned.append(([seq1[i - 1]], [], i - 1, i, -1, -1))
            i -= 1
        elif move == 3:
            aligned.append(([], [seq2[j - 1]], -1, -1, j - 1, j))
            j -= 1
        elif 10 <= move < 20:
            k = move - 10
            aligned.append(([seq1[i - 1]], seq2[j - k : j], i - 1, i, j - k, j))
            i -= 1
            j -= k
        elif 20 <= move < 30:
            k = move - 20
            aligned.append((seq1[i - k : i], [seq2[j - 1]], i - k, i, j - 1, j))
            i -= k
            j -= 1
        else:
            break

    aligned.reverse()
    return aligned


def align_tokens_with_combinations_cuda_chunked(
    seq1: List[str],
    seq2: List[str],
    exact_match_score: float = 3.0,
    combination_score_multiplier: float = 1.5,
    gap_penalty: float = -1.5,
    max_combination_len: int = 4,
    chunk_size: int = 128,
) -> tuple[List[Tuple[List[str], List[str], int, int, int, int]], float]:
    """
    CUDA version of TokenAligner's chunked DP.

    Notes:
    - Mirrors midpoint recursion in current python implementation.
    - Uses CUDA only at the base chunk DP solve.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for align_tokens_with_combinations_cuda_chunked")

    n1, n2 = len(seq1), len(seq2)
    if n1 <= chunk_size and n2 <= chunk_size:
        ids1, ids2, joined1, joined2 = _build_ids_and_joined_tables(seq1, seq2, max_combination_len)
        device = torch.device("cuda")
        ids1_t = torch.from_numpy(ids1).to(device=device, dtype=torch.int64, non_blocking=True)
        ids2_t = torch.from_numpy(ids2).to(device=device, dtype=torch.int64, non_blocking=True)
        j1_t = torch.from_numpy(joined1).to(device=device, dtype=torch.int64, non_blocking=True)
        j2_t = torch.from_numpy(joined2).to(device=device, dtype=torch.int64, non_blocking=True)

        ext = _load_cuda_ext()
        trace_t, score_t = ext.dp_chunk_cuda(
            ids1_t.contiguous(),
            ids2_t.contiguous(),
            j1_t.contiguous(),
            j2_t.contiguous(),
            float(exact_match_score),
            float(combination_score_multiplier),
            float(gap_penalty),
            int(max_combination_len),
        )
        trace_np = trace_t.cpu().numpy()
        aligned = _backtrack_from_trace(seq1, seq2, trace_np)
        score = float(score_t.item())
        return aligned, score

    # Mirrors existing midpoint split strategy.
    mid1, mid2 = n1 // 2, n2 // 2
    left_aligned, left_score = align_tokens_with_combinations_cuda_chunked(
        seq1[:mid1],
        seq2[:mid2],
        exact_match_score=exact_match_score,
        combination_score_multiplier=combination_score_multiplier,
        gap_penalty=gap_penalty,
        max_combination_len=max_combination_len,
        chunk_size=chunk_size,
    )
    right_aligned, right_score = align_tokens_with_combinations_cuda_chunked(
        seq1[mid1:],
        seq2[mid2:],
        exact_match_score=exact_match_score,
        combination_score_multiplier=combination_score_multiplier,
        gap_penalty=gap_penalty,
        max_combination_len=max_combination_len,
        chunk_size=chunk_size,
    )

    adjusted_right = []
    for s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end in right_aligned:
        new_s1_start = s1_start + mid1 if s1_start >= 0 else -1
        new_s1_end = s1_end + mid1 if s1_end >= 0 else -1
        new_s2_start = s2_start + mid2 if s2_start >= 0 else -1
        new_s2_end = s2_end + mid2 if s2_end >= 0 else -1
        adjusted_right.append((s1_tokens, s2_tokens, new_s1_start, new_s1_end, new_s2_start, new_s2_end))

    return left_aligned + adjusted_right, (left_score + right_score)


def monkeypatch_tokenaligner_cuda_basecase() -> None:
    """
    Monkeypatch TokenAligner base chunk DP with CUDA kernel-backed version.

    Usage:
        from cuda_tokenalign_dp import monkeypatch_tokenaligner_cuda_basecase
        monkeypatch_tokenaligner_cuda_basecase()
    """
    from nemo_rl.algorithms.x_token.tokenalign import TokenAligner

    def _cuda_chunked(
        seq1: List[str],
        seq2: List[str],
        exact_match_score: float = 3.0,
        combination_score_multiplier: float = 1.5,
        gap_penalty: float = -1.5,
        max_combination_len: int = 4,
        ignore_leading_char_diff: bool = False,
        chunk_size: int = 256,
    ):
        if ignore_leading_char_diff:
            return TokenAligner.align_tokens_combinations_chunked(
                seq1=seq1,
                seq2=seq2,
                exact_match_score=exact_match_score,
                combination_score_multiplier=combination_score_multiplier,
                gap_penalty=gap_penalty,
                max_combination_len=max_combination_len,
                ignore_leading_char_diff=ignore_leading_char_diff,
                chunk_size=chunk_size,
            )
        return align_tokens_with_combinations_cuda_chunked(
            seq1=seq1,
            seq2=seq2,
            exact_match_score=exact_match_score,
            combination_score_multiplier=combination_score_multiplier,
            gap_penalty=gap_penalty,
            max_combination_len=max_combination_len,
            chunk_size=chunk_size,
        )

    TokenAligner.align_tokens_combinations_chunked = staticmethod(_cuda_chunked)

