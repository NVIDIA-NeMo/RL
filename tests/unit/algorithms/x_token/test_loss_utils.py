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
"""Unit tests for ``nemo_rl/algorithms/x_token/loss_utils.py``.

CPU-only. ``Fp32SparseMM`` is GPU-pinned via ``custom_fwd(device_type="cuda")``
and is intentionally not covered here.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from nemo_rl.algorithms.x_token.loss_utils import (
    _EXACT_TOKEN_MAP_CACHE,
    _SPARSE_PROJECTION_CACHE,
    _TOPK_PROJECTION_CACHE,
    alignment_from_flat_batch,
    build_exact_token_map,
    chunk_average_log_probs,
    get_sparse_projection_matrix,
    get_topk_projection,
    parse_projection_file,
    valid_chunk_mask,
)
from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch

# ---------------------------------------------------------------------------
# alignment_from_flat_batch
# ---------------------------------------------------------------------------


class TestAlignmentFromFlatBatch:
    def _flat(self, B: int = 1, T_s: int = 4, T_t: int = 4, P: int = 2) -> dict:
        return {
            "alignment_pair_valid": torch.ones((B, P), dtype=torch.bool),
            "alignment_pair_is_correct": torch.ones((B, P), dtype=torch.bool),
            "alignment_student_exact_partition_mask": torch.zeros(
                (B, T_s), dtype=torch.bool
            ),
            "alignment_teacher_exact_partition_mask": torch.zeros(
                (B, T_t), dtype=torch.bool
            ),
            "alignment_student_chunk_id": torch.zeros((B, T_s), dtype=torch.long),
            "alignment_teacher_chunk_id": torch.zeros((B, T_t), dtype=torch.long),
            "alignment_num_chunks": torch.tensor([P] * B, dtype=torch.long),
        }

    def test_returns_alignment_batch_with_all_fields(self):
        ab = alignment_from_flat_batch(self._flat())
        assert isinstance(ab, AlignmentBatch)
        for f in fields(AlignmentBatch):
            assert getattr(ab, f.name) is not None

    def test_field_set_matches_dataclass_schema(self):
        # Schema-drift detector: if AlignmentBatch grows / loses a field,
        # the helper consumes that change automatically, but the flat
        # keys must follow.
        flat = self._flat()
        expected_keys = {f"alignment_{f.name}" for f in fields(AlignmentBatch)}
        assert expected_keys.issubset(flat.keys())

    def test_values_are_tensor_identity_preserved(self):
        flat = self._flat()
        ab = alignment_from_flat_batch(flat)
        assert ab.pair_valid is flat["alignment_pair_valid"]
        assert ab.num_chunks is flat["alignment_num_chunks"]


# ---------------------------------------------------------------------------
# chunk_average_log_probs
# ---------------------------------------------------------------------------


class TestChunkAverageLogProbs:
    def test_chunk_id_minus_one_contributes_nothing(self):
        # B=1, T=4, V=2, max_chunks=2. Position 0 in chunk 0, position 1
        # in chunk -1 (skipped), positions 2-3 in chunk 1.
        log_probs = torch.tensor([[[1.0, 2.0], [99.0, 99.0], [3.0, 4.0], [5.0, 6.0]]])
        chunk_id = torch.tensor([[0, -1, 1, 1]])
        avg, sizes = chunk_average_log_probs(log_probs, chunk_id, max_chunks=2)

        assert avg.shape == (1, 2, 2)
        assert sizes.shape == (1, 2)
        # chunk 0 has one token (position 0); chunk 1 has two tokens.
        assert sizes[0, 0].item() == 1
        assert sizes[0, 1].item() == 2
        # chunk 0 mean = position 0 = [1, 2].
        assert torch.allclose(avg[0, 0], torch.tensor([1.0, 2.0]))
        # chunk 1 mean = (positions 2 + 3) / 2 = [4, 5].
        assert torch.allclose(avg[0, 1], torch.tensor([4.0, 5.0]))
        # Position 1 (chunk_id=-1) did not contribute to either bucket.

    def test_empty_chunks_yield_zero_with_epsilon(self):
        log_probs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        chunk_id = torch.tensor([[0, 0]])
        avg, sizes = chunk_average_log_probs(log_probs, chunk_id, max_chunks=3)
        # Chunk 0 has 2 tokens; chunks 1, 2 are empty.
        assert sizes[0, 0].item() == 2
        assert sizes[0, 1].item() == 0
        assert sizes[0, 2].item() == 0
        # Empty chunks divide by ~eps → near-zero (numerator is also 0).
        assert torch.allclose(avg[0, 1], torch.zeros(2), atol=1e-5)


# ---------------------------------------------------------------------------
# valid_chunk_mask
# ---------------------------------------------------------------------------


class TestValidChunkMask:
    def test_all_three_conditions_required(self):
        s = torch.tensor([1, 0, 2, 3])
        t = torch.tensor([1, 2, 0, 4])
        pv = torch.tensor([True, True, True, False])
        # only index 0 has all three.
        out = valid_chunk_mask(s, t, pv)
        assert out.tolist() == [True, False, False, False]


# ---------------------------------------------------------------------------
# parse_projection_file — uses the conftest fixtures.
# ---------------------------------------------------------------------------


class TestParseProjectionFile:
    def test_dense_topk_format(self, synth_topk_projection_path):
        indices, values, v_s, v_t = parse_projection_file(synth_topk_projection_path)
        # COO layout: indices [2, nnz] = (student_idx, teacher_idx).
        assert indices.dim() == 2 and indices.shape[0] == 2
        assert values.dim() == 1 and values.shape[0] == indices.shape[1]
        assert v_s == 32  # SYNTH_V_STUDENT
        # v_t = max positive teacher idx + 1; with synth conftest the
        # synth_topk fixture seeds row v_s-1 -> v_t-1 = 23, so v_t == 24.
        assert v_t == 24
        # sentinel -1 entries in `indices` row 1 are preserved (the
        # parser does not filter them).
        assert (indices[1] == -1).any().item()

    def test_sparse_dict_format(self, synth_sparse_projection_path):
        indices, values, v_s, v_t = parse_projection_file(synth_sparse_projection_path)
        assert indices.shape[0] == 2
        # The synth sparse fixture seeds 3 entries.
        assert indices.shape[1] == 3
        # Inferred sizes come from observed max ids.
        assert v_s == 32
        assert v_t == 24

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_projection_file(tmp_path / "does_not_exist.pt")

    def test_unknown_format_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.pt"
        torch.save({"not_indices": torch.zeros(2)}, bad)
        with pytest.raises(ValueError):
            parse_projection_file(bad)


# ---------------------------------------------------------------------------
# get_sparse_projection_matrix — cache semantics.
# ---------------------------------------------------------------------------


class TestGetSparseProjectionMatrix:
    def test_first_call_populates_cache(self, synth_topk_projection_path):
        device = torch.device("cpu")
        assert (
            str(synth_topk_projection_path),
            device,
            32,
            24,
        ) not in _SPARSE_PROJECTION_CACHE
        out = get_sparse_projection_matrix(
            synth_topk_projection_path,
            device,
            student_vocab_size=32,
            teacher_vocab_size=24,
        )
        assert out.is_sparse
        assert out.shape == (32, 24)
        assert (
            str(synth_topk_projection_path),
            device,
            32,
            24,
        ) in _SPARSE_PROJECTION_CACHE

    def test_repeat_call_hits_cache(self, synth_topk_projection_path):
        device = torch.device("cpu")
        out1 = get_sparse_projection_matrix(
            synth_topk_projection_path,
            device,
            student_vocab_size=32,
            teacher_vocab_size=24,
        )
        out2 = get_sparse_projection_matrix(
            synth_topk_projection_path,
            device,
            student_vocab_size=32,
            teacher_vocab_size=24,
        )
        assert out2 is out1  # same tensor object from cache

    def test_different_size_distinct_cache_entries(self, synth_topk_projection_path):
        device = torch.device("cpu")
        a = get_sparse_projection_matrix(
            synth_topk_projection_path,
            device,
            student_vocab_size=32,
            teacher_vocab_size=24,
        )
        b = get_sparse_projection_matrix(
            synth_topk_projection_path,
            device,
            student_vocab_size=64,
            teacher_vocab_size=48,
        )
        assert a is not b
        assert a.shape == (32, 24)
        assert b.shape == (64, 48)

    def test_negative_teacher_sentinels_dropped(self, synth_topk_projection_path):
        # The dense topk format has sentinel -1 entries (per
        # parse_projection_file test); get_sparse_projection_matrix
        # must drop them — sparse_coo_tensor disallows negative indices.
        out = get_sparse_projection_matrix(
            synth_topk_projection_path,
            torch.device("cpu"),
            student_vocab_size=32,
            teacher_vocab_size=24,
        )
        ind = out.indices()
        assert (ind[1] >= 0).all().item()


# ---------------------------------------------------------------------------
# get_topk_projection — cache + format gate.
# ---------------------------------------------------------------------------


class TestGetTopkProjection:
    def test_first_call_populates_cache(self, synth_topk_projection_path):
        device = torch.device("cpu")
        assert (str(synth_topk_projection_path), device) not in _TOPK_PROJECTION_CACHE
        ind, lik = get_topk_projection(synth_topk_projection_path, device)
        assert ind.dtype == torch.long
        assert lik.dtype == torch.float32
        assert ind.shape == lik.shape
        assert (str(synth_topk_projection_path), device) in _TOPK_PROJECTION_CACHE

    def test_repeat_call_hits_cache(self, synth_topk_projection_path):
        device = torch.device("cpu")
        a = get_topk_projection(synth_topk_projection_path, device)
        b = get_topk_projection(synth_topk_projection_path, device)
        assert a is b  # cache returns the same tuple object

    def test_sparse_format_rejected(self, synth_sparse_projection_path):
        with pytest.raises(ValueError, match="dense"):
            get_topk_projection(synth_sparse_projection_path, torch.device("cpu"))


# ---------------------------------------------------------------------------
# build_exact_token_map — cache keying + partition output shape.
# ---------------------------------------------------------------------------


class TestBuildExactTokenMap:
    def test_returns_partition_dict(self, synth_topk_projection_path):
        out = build_exact_token_map(
            synth_topk_projection_path,
            torch.device("cpu"),
            xtoken_loss=False,
            teacher_vocab_size=24,
        )
        assert set(out.keys()) == {
            "common_student",
            "common_teacher",
            "uncommon_student",
            "uncommon_teacher",
        }
        # The synth fixture seeds row 0 -> col 0 and row v_s-1 -> col v_t-1
        # with likelihood 1.0 and sentinel -1 second slot, so strict mode
        # produces at least two exact-mapped students.
        assert out["common_student"].numel() >= 2

    def test_cache_key_includes_xtoken_loss(self, synth_topk_projection_path):
        device = torch.device("cpu")
        a = build_exact_token_map(
            synth_topk_projection_path,
            device,
            xtoken_loss=False,
            teacher_vocab_size=24,
        )
        b = build_exact_token_map(
            synth_topk_projection_path,
            device,
            xtoken_loss=True,
            teacher_vocab_size=24,
        )
        # Different xtoken_loss → distinct cache entries (different dicts).
        assert a is not b
        keys = {
            (str(synth_topk_projection_path), device, False, 24),
            (str(synth_topk_projection_path), device, True, 24),
        }
        assert keys.issubset(set(_EXACT_TOKEN_MAP_CACHE.keys()))

    def test_repeat_call_hits_cache(self, synth_topk_projection_path):
        device = torch.device("cpu")
        a = build_exact_token_map(
            synth_topk_projection_path,
            device,
            xtoken_loss=False,
            teacher_vocab_size=24,
        )
        b = build_exact_token_map(
            synth_topk_projection_path,
            device,
            xtoken_loss=False,
            teacher_vocab_size=24,
        )
        assert a is b  # cache returns the same dict object
