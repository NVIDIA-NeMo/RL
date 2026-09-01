# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import warnings

import torch


def test_empty_cache_guard_noops_without_expandable_segments(monkeypatch):
    from nemo_rl.models.policy.utils import (
        make_empty_cache_best_effort_under_expandable_segments,
    )

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    original = torch.cuda.empty_cache
    try:
        make_empty_cache_best_effort_under_expandable_segments()
        assert torch.cuda.empty_cache is original
    finally:
        torch.cuda.empty_cache = original


def test_empty_cache_guard_swallows_variant_runtime_error(monkeypatch):
    from nemo_rl.models.policy.utils import (
        make_empty_cache_best_effort_under_expandable_segments,
    )

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    original = torch.cuda.empty_cache

    def _raises():
        raise RuntimeError("std::get: wrong index for variant")

    try:
        torch.cuda.empty_cache = _raises
        make_empty_cache_best_effort_under_expandable_segments()
        assert torch.cuda.empty_cache is not _raises
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.empty_cache()  # must not raise
        assert any("expandable_segments" in str(w.message) for w in caught)
        # idempotent: re-applying must not double-wrap
        wrapped = torch.cuda.empty_cache
        make_empty_cache_best_effort_under_expandable_segments()
        assert torch.cuda.empty_cache is wrapped
    finally:
        torch.cuda.empty_cache = original


def test_empty_cache_guard_reraises_unrelated_runtime_errors(monkeypatch):
    """Only the known expandable-segments allocator error is swallowed;
    any other RuntimeError must propagate."""
    import pytest

    from nemo_rl.models.policy.utils import (
        make_empty_cache_best_effort_under_expandable_segments,
    )

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    original = torch.cuda.empty_cache

    def _raises_other():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    try:
        torch.cuda.empty_cache = _raises_other
        make_empty_cache_best_effort_under_expandable_segments()
        with pytest.raises(RuntimeError, match="illegal memory access"):
            torch.cuda.empty_cache()
    finally:
        torch.cuda.empty_cache = original
