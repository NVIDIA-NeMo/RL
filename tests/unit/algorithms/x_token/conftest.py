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

from pathlib import Path

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
