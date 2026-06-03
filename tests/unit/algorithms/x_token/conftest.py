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
"""Shared fixtures for the cross-tokenizer distillation test harness.

Tests in this directory verify the invariants raised in the PR-2508 review
without launching real Ray actors, policies, or distributed runtimes. All
fixtures here are pure-CPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.x_token.loss_utils import (
    _SPARSE_PROJECTION_CACHE,
    _TOPK_PROJECTION_CACHE,
)

# Synthetic vocab sizes used across CT unit tests. Small enough to keep dense
# matrices cheap; large enough that "max(observed_id) + 1" sizing would still
# undersize if the high-id rows are sparse.
SYNTH_V_STUDENT = 32
SYNTH_V_TEACHER = 24
SYNTH_TOP_K = 2


@pytest.fixture(autouse=True)
def _reset_projection_caches():
    """Clear x_token projection caches between tests so fixture paths don't collide."""
    _SPARSE_PROJECTION_CACHE.clear()
    _TOPK_PROJECTION_CACHE.clear()
    yield
    _SPARSE_PROJECTION_CACHE.clear()
    _TOPK_PROJECTION_CACHE.clear()


@pytest.fixture
def synth_topk_projection_path(tmp_path: Path) -> str:
    """Write a dense top-k projection file and return its path.

    Layout: each student row maps to ``SYNTH_TOP_K`` teacher ids. Row 0 and
    row ``V_s - 1`` are exact-mapped (likelihood 1.0, second slot sentinel
    ``-1``) so the resulting matrix exercises the "high student id has an
    exact map" path. Other rows distribute weights ~ uniform across two
    teacher ids picked deterministically.
    """
    v_s, v_t, k = SYNTH_V_STUDENT, SYNTH_V_TEACHER, SYNTH_TOP_K
    indices = torch.full((v_s, k), -1, dtype=torch.long)
    likelihoods = torch.zeros((v_s, k), dtype=torch.float32)

    # Exact-mapped rows at both ends of the student vocab.
    indices[0, 0] = 0
    likelihoods[0, 0] = 1.0
    indices[v_s - 1, 0] = v_t - 1
    likelihoods[v_s - 1, 0] = 1.0

    # Fuzzy rows in the middle.
    for s in range(1, v_s - 1):
        t1 = s % v_t
        t2 = (s * 3 + 1) % v_t
        if t2 == t1:
            t2 = (t1 + 1) % v_t
        indices[s, 0] = t1
        indices[s, 1] = t2
        likelihoods[s, 0] = 0.7
        likelihoods[s, 1] = 0.3

    path = tmp_path / "synth_projection.pt"
    torch.save({"indices": indices, "likelihoods": likelihoods}, path)
    return str(path)


@pytest.fixture
def synth_sparse_projection_path(tmp_path: Path) -> str:
    """Write a sparse ``dict[(s, t)] -> count`` projection file.

    Intentionally seeds an entry at the **highest** student/teacher id so a
    fix that infers ``V_s = max(id) + 1`` without a vocab-size floor would
    still produce the correct shape — and so a buggy inference that drops
    high ids becomes detectable.
    """
    data: dict[tuple[int, int], float] = {
        (0, 0): 1.0,
        (1, 5): 2.0,
        (SYNTH_V_STUDENT - 1, SYNTH_V_TEACHER - 1): 3.0,
    }
    path = tmp_path / "synth_sparse_projection.pt"
    torch.save(data, path)
    return str(path)


def make_loss_cfg(
    projection_matrix_path: str,
    *,
    gold_loss: bool = False,
    xtoken_loss: bool = False,
    reverse_kl: bool = False,
    exact_token_match_only: bool = False,
    student_vocab_size: int = SYNTH_V_STUDENT,
    teacher_vocab_size: int = SYNTH_V_TEACHER,
    temperature: float = 1.0,
    vocab_topk: int = 8,
    uncommon_topk: int = 4,
    kl_loss_weight: float = 1.0,
    ce_loss_scale: float = 1.0,
    dynamic_loss_scaling: bool = False,
) -> dict[str, Any]:
    """Minimal config dict matching ``CrossTokenizerDistillationLossConfig``."""
    return {
        "projection_matrix_path": projection_matrix_path,
        "gold_loss": gold_loss,
        "xtoken_loss": xtoken_loss,
        "temperature": temperature,
        "vocab_topk": vocab_topk,
        "uncommon_topk": uncommon_topk,
        "reverse_kl": reverse_kl,
        "exact_token_match_only": exact_token_match_only,
        "kl_loss_weight": kl_loss_weight,
        "ce_loss_scale": ce_loss_scale,
        "dynamic_loss_scaling": dynamic_loss_scaling,
        "student_vocab_size": student_vocab_size,
        "teacher_vocab_size": teacher_vocab_size,
    }


def make_ct_data_dict(
    *,
    batch_size: int,
    t_student: int,
    t_teacher: int,
    max_pairs: int,
    num_chunks: list[int],
    student_chunk_id: torch.Tensor,
    teacher_chunk_id: torch.Tensor,
    pair_valid: torch.Tensor,
    sample_mask: torch.Tensor,
    pair_is_correct: torch.Tensor | None = None,
    student_exact_partition_mask: torch.Tensor | None = None,
    teacher_exact_partition_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the per-microbatch CT data dict shape expected by the loss fn.

    ``teacher_full_logits_ipc`` is left as an empty list — tests that need
    it monkeypatch ``_rebuild_teacher_full_logits`` directly so we never
    have to construct real CUDA IPC handles.
    """
    if pair_is_correct is None:
        pair_is_correct = pair_valid.clone()
    if student_exact_partition_mask is None:
        student_exact_partition_mask = torch.zeros(
            (batch_size, t_student), dtype=torch.bool
        )
    if teacher_exact_partition_mask is None:
        teacher_exact_partition_mask = torch.zeros(
            (batch_size, t_teacher), dtype=torch.bool
        )

    return {
        "input_ids": torch.zeros((batch_size, t_student), dtype=torch.long),
        "input_lengths": torch.full((batch_size,), t_student, dtype=torch.long),
        "token_mask": torch.ones((batch_size, t_student), dtype=torch.long),
        "sample_mask": sample_mask,
        "teacher_full_logits_ipc": [],
        "alignment_pair_valid": pair_valid,
        "alignment_pair_is_correct": pair_is_correct,
        "alignment_student_exact_partition_mask": student_exact_partition_mask,
        "alignment_teacher_exact_partition_mask": teacher_exact_partition_mask,
        "alignment_student_chunk_id": student_chunk_id,
        "alignment_teacher_chunk_id": teacher_chunk_id,
        "alignment_num_chunks": torch.tensor(num_chunks, dtype=torch.long),
    }


def has_gloo() -> bool:
    """Whether torch.distributed has a usable gloo backend."""
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


def cpu_only() -> bool:
    """Whether the suite should restrict itself to CPU paths (env override)."""
    return os.environ.get("NRL_XTOKEN_GPU_SMOKE", "0") != "1"
