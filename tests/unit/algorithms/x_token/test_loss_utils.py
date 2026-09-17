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

Most tests here are CPU-only. ``Fp32SparseMM`` is GPU-pinned via
``custom_fwd(device_type="cuda")`` and is intentionally not covered. The TP/CP
> 1 code paths require real collectives, so they live in the multi-GPU NCCL
tests at the bottom of this file (skipped when < 2 GPUs are available).
"""

from __future__ import annotations

import os
import traceback
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import ray
import torch

from nemo_rl.algorithms.x_token.loss_utils import (
    _EXACT_TOKEN_MAP_CACHE,
    _SPARSE_PROJECTION_CACHE,
    _TOPK_PROJECTION_CACHE,
    _try_zero_copy_teacher_logits,
    alignment_from_flat_batch,
    assemble_teacher_logits_from_shards,
    build_exact_token_map,
    chunk_average_log_probs,
    collect_overlapping_teacher_shards,
    get_sparse_projection_matrix,
    get_topk_projection,
    localize_alignment,
    parse_projection_file,
    prepare_xtoken_cross_tokenizer_loss_input,
    rebuild_teacher_full_logits_from_ipc,
    rebuild_teacher_sparse_logits_from_ipc,
    select_teacher_topk_indices,
    slice_sparse_projection_cols,
    valid_chunk_mask,
)
from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch
from nemo_rl.distributed.model_utils import group_all_reduce_sum_with_grad
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.policy.utils import DENSE_TEACHER_IPC_FLAT_LAYOUT

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


def test_automodel_cp_layout_localizes_xtoken_windows_after_global_shift():
    """Automodel layout restoration precedes contiguous x-token localization."""
    cp_group = object()
    local_logits = torch.randn(1, 3, 2)
    full_student_logits = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    teacher_logits = torch.randn(1, 2, 4)
    cp_sharder = MagicMock()
    cp_sharder.gather_token_tensor.return_value = full_student_logits
    teacher_ipc = [{"opaque": "handle"}]
    data = {
        "input_ids": torch.arange(6).unsqueeze(0),
        "token_mask": torch.ones(1, 6),
        "sample_mask": torch.ones(1),
        "teacher_0_full_logits_ipc": teacher_ipc,
        "teacher_0_input_ids": torch.arange(20, 24).unsqueeze(0),
        "alignment_0_student_chunk_id": torch.tensor([[10, 11, 12, 13, 14, 15]]),
        "alignment_0_teacher_chunk_id": torch.tensor([[20, 21, 22, 23]]),
        "alignment_0_pair_valid": torch.ones(1, 2, dtype=torch.bool),
        "alignment_0_pair_is_correct": torch.ones(1, 2, dtype=torch.bool),
        "alignment_0_num_chunks": torch.tensor([2]),
    }

    with (
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=1),
        patch(
            "nemo_rl.algorithms.x_token.loss_utils."
            "rebuild_teacher_full_logits_from_ipc",
            return_value=(teacher_logits, 0),
        ) as rebuild_teacher,
    ):
        (
            student_logits,
            teachers,
            sparse_teachers,
            aligns,
            dense_reconstruction_fallbacks,
            tp_group,
            returned_cp_group,
            dp_cp_group,
        ) = prepare_xtoken_cross_tokenizer_loss_input(
            local_logits,
            data,
            projection_matrix_paths=["projection.pt"],
            context_parallel_group=cp_group,
            cp_sharder=cp_sharder,
        )

    torch.testing.assert_close(student_logits, full_student_logits[:, 3:6])
    assert teachers[0] is teacher_logits
    assert sparse_teachers == {}
    assert dense_reconstruction_fallbacks == {0: 0}
    assert tp_group is None
    assert returned_cp_group is cp_group
    assert dp_cp_group is None
    torch.testing.assert_close(aligns[0].student_input_ids, data["input_ids"][:, 3:6])
    torch.testing.assert_close(aligns[0].student_token_mask, data["token_mask"][:, 3:6])
    torch.testing.assert_close(aligns[0].student_chunk_id, torch.tensor([[14, 15, -1]]))
    torch.testing.assert_close(aligns[0].teacher_chunk_id, torch.tensor([[23, -1]]))
    cp_sharder.gather_token_tensor.assert_called_once_with(
        local_logits, seq_dim=1, trim=True
    )
    rebuild_teacher.assert_called_once_with(teacher_ipc, cp_group=cp_group, device=0)


def test_automodel_cp_layout_rejects_non_divisible_student_sequence():
    cp_group = object()
    local_logits = torch.randn(1, 3, 2)
    cp_sharder = MagicMock()
    cp_sharder.gather_token_tensor.return_value = torch.randn(1, 5, 2)

    with (
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.distributed.get_world_size", return_value=2),
        pytest.raises(
            ValueError,
            match=(
                "X-token student sequence length must be divisible by the student "
                "context parallel size"
            ),
        ),
    ):
        prepare_xtoken_cross_tokenizer_loss_input(
            local_logits,
            {},
            projection_matrix_paths=[],
            context_parallel_group=cp_group,
            cp_sharder=cp_sharder,
        )

    cp_sharder.gather_token_tensor.assert_called_once_with(
        local_logits, seq_dim=1, trim=True
    )


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

    @staticmethod
    def _write_topk(tmp_path: Path, indices_rows, likelihood_rows) -> str:
        """Save a dense top-k projection file from per-student rows."""
        path = tmp_path / "collision_projection.pt"
        torch.save(
            {
                "indices": torch.tensor(indices_rows, dtype=torch.long),
                "likelihoods": torch.tensor(likelihood_rows, dtype=torch.float32),
            },
            path,
        )
        return str(path)

    def test_relaxed_mode_collision_resolution(self, tmp_path: Path):
        """Relaxed (xtoken_loss=True) collisions: among students whose top
        projection lands on the same teacher, the highest first-projection
        weight wins, ties break to the lowest student index, and the
        exact-map threshold is ``>= 0.6``.

          s0, s1 collide on teacher 2 (0.8 vs 0.7) -> s0 wins (higher weight)
          s2, s3 collide on teacher 5 (0.9 == 0.9) -> s2 wins (lower index)
          s4 -> teacher 1 @ 0.65 (unique, above threshold)
          s5 -> top weight 0.5 (below threshold, no exact map)
          s6 -> teacher 4 @ 0.6 (boundary, included)
          s7 -> teacher 0 @ 0.95 (unique)
        """
        indices = [
            [2, 3],
            [2, 4],
            [5, 0],
            [5, 1],
            [1, 0],
            [3, 4],
            [4, 2],
            [0, 1],
        ]
        likelihoods = [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.65, 0.35],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.95, 0.05],
        ]
        out = build_exact_token_map(
            self._write_topk(tmp_path, indices, likelihoods),
            torch.device("cpu"),
            xtoken_loss=True,
            teacher_vocab_size=6,
        )
        # Winners, paired and sorted by student index.
        assert out["common_student"].tolist() == [0, 2, 4, 6, 7]
        assert out["common_teacher"].tolist() == [2, 5, 1, 4, 0]
        # Collision losers (s1, s3) and the sub-threshold student (s5) fall to
        # the uncommon partition; uncommon_teacher is the unmapped complement.
        assert out["uncommon_student"].tolist() == [1, 3, 5]
        assert out["uncommon_teacher"].tolist() == [3]

    def test_strict_mode_collision_picks_lowest_student(self, tmp_path: Path):
        """Strict (xtoken_loss=False) exact maps require weight 1.0 and a
        ``-1`` sentinel in the second slot. On a teacher collision the lowest
        student index wins; fuzzy rows (second slot != -1) are excluded.

          s0, s1 both exact-map to teacher 1 -> s0 wins (lowest index)
          s2 fuzzy (second slot 0) -> excluded
          s3 exact-maps to teacher 0 (unique)
        """
        indices = [[1, -1], [1, -1], [2, 0], [0, -1]]
        likelihoods = [[1.0, 0.0], [1.0, 0.0], [0.7, 0.3], [1.0, 0.0]]
        out = build_exact_token_map(
            self._write_topk(tmp_path, indices, likelihoods),
            torch.device("cpu"),
            xtoken_loss=False,
            teacher_vocab_size=3,
        )
        assert out["common_student"].tolist() == [0, 3]
        assert out["common_teacher"].tolist() == [1, 0]
        assert out["uncommon_student"].tolist() == [1, 2]
        assert out["uncommon_teacher"].tolist() == [2]


# NOTE: the TP/CP > 1 paths of vocab_parallel_log_softmax / vocab_parallel_full_log_softmax /
# project_student_to_teacher_vocab / vocab_parallel_argmax /
# select_teacher_topk_indices are covered by the real multi-GPU NCCL tests at
# the bottom of this file (single-rank fallbacks would not exercise any of the
# collectives).
class TestSelectTeacherTopKIndices:
    def test_valid_mask_excludes_masked_samples_and_predictor_padding(self):
        teacher_logits = torch.full((2, 4, 6), -10.0)
        teacher_logits[0, 0, 0:2] = torch.tensor([8.0, 7.0])
        teacher_logits[0, 1, 0:2] = torch.tensor([7.0, 8.0])
        # These invalid rows would dominate the vocabulary-wide max without
        # the selection mask.
        teacher_logits[0, 2:, 2:4] = torch.tensor([100.0, 90.0])
        teacher_logits[1, :, 4:6] = torch.tensor([200.0, 190.0])
        valid_mask = torch.tensor(
            [[True, True, False, False], [False, False, False, False]]
        )

        selected = select_teacher_topk_indices(teacher_logits, 2, valid_mask=valid_mask)

        assert selected.tolist() == [0, 1]

    def test_all_masked_uses_deterministic_columns(self):
        teacher_logits = torch.randn(2, 3, 5)

        selected = select_teacher_topk_indices(
            teacher_logits,
            3,
            valid_mask=torch.zeros(2, 3, dtype=torch.bool),
        )

        assert selected.tolist() == [0, 1, 2]

    def test_valid_mask_shape_must_match_predictor_axes(self):
        with pytest.raises(ValueError, match="valid_mask must match"):
            select_teacher_topk_indices(
                torch.randn(2, 3, 5),
                2,
                valid_mask=torch.ones(2, 2, dtype=torch.bool),
            )


class TestLocalizeAlignment:
    def _data(self, B: int = 2, T_s: int = 5, T_t: int = 6, P: int = 3) -> dict:
        return {
            "alignment_student_chunk_id": torch.zeros((B, T_s), dtype=torch.long),
            "alignment_teacher_chunk_id": torch.arange(B * T_t).reshape(B, T_t),
            "alignment_pair_valid": torch.ones((B, P), dtype=torch.bool),
            "alignment_pair_is_correct": torch.zeros((B, P), dtype=torch.bool),
            "sample_mask": torch.ones(B, dtype=torch.bool),
        }

    def test_no_cp_keeps_full_teacher_chunk_id(self):
        data = self._data()
        align = localize_alignment(data, teacher_seq_len=6, cp_group=None)
        # cp_group=None → start offset 0, full teacher_chunk_id passed through.
        assert torch.equal(align.teacher_chunk_id, data["alignment_teacher_chunk_id"])
        assert torch.equal(align.student_chunk_id, data["alignment_student_chunk_id"])
        assert torch.equal(align.pair_valid, data["alignment_pair_valid"])
        assert torch.equal(align.pair_is_correct, data["alignment_pair_is_correct"])
        assert torch.equal(align.sample_mask, data["sample_mask"])


class TestCollectOverlappingTeacherShards:
    @staticmethod
    def _shard(vstart, vend, gss, seq):
        return {
            "vocab_start_index": vstart,
            "vocab_end_index": vend,
            "global_seq_start": gss,
            "actual_shape": (seq, vend - vstart),
        }

    def test_single_full_shard_no_cp(self):
        m = collect_overlapping_teacher_shards(
            [self._shard(0, 5, 0, 4)],
            student_cp_rank=0,
            student_cp_size=1,
            full_seq_len=4,
        )
        assert len(m) == 1
        _, src_seq, src_vocab, dest_seq, dest_vocab = m[0]
        assert (src_seq, dest_seq) == (slice(0, 4), slice(0, 4))
        assert (src_vocab, dest_vocab) == (slice(0, 5), slice(0, 5))

    def test_cp_selects_matching_rank_only(self):
        s0, s1 = self._shard(0, 5, 0, 4), self._shard(0, 5, 4, 4)
        m = collect_overlapping_teacher_shards(
            [s0, s1], student_cp_rank=1, student_cp_size=2, full_seq_len=8
        )
        assert len(m) == 1 and m[0][0] is s1
        assert m[0][3] == slice(0, 4)

    def test_tp_shards_map_to_vocab_ranges(self):
        m = collect_overlapping_teacher_shards(
            [self._shard(0, 4, 0, 2), self._shard(4, 8, 0, 2)],
            student_cp_rank=0,
            student_cp_size=1,
            full_seq_len=2,
        )
        assert [x[4] for x in m] == [slice(0, 4), slice(4, 8)]


class TestZeroCopyTeacherLogits:
    @staticmethod
    def _entry(sample_idx, vend=5, buf=0, payload=("p",)):
        return {
            "teacher_shards": [
                {
                    "payload_ipc": payload,
                    "buf_idx": buf,
                    "sample_index_in_buf": sample_idx,
                    "vocab_start_index": 0,
                    "vocab_end_index": vend,
                    "global_seq_start": 0,
                    "actual_shape": (4, vend),
                    "full_seq_len": 4,
                    "full_vocab_size": vend,
                }
            ]
        }

    @staticmethod
    def _compact_entry(
        *,
        token_offset: int,
        valid_seq_len: int,
        stored_seq_len: int,
        used_tokens: int,
        capacity_tokens: int,
        full_seq_len: int = 4,
        vocab_size: int = 5,
        payload="compact",
    ):
        return {
            "teacher_shards": [
                {
                    "payload_ipc": payload,
                    "buf_idx": 0,
                    "sample_index_in_buf": 0,
                    "ipc_layout": DENSE_TEACHER_IPC_FLAT_LAYOUT,
                    "storage_shape": (capacity_tokens, vocab_size),
                    "storage_token_offset": token_offset,
                    "storage_used_tokens": used_tokens,
                    "storage_capacity_tokens": capacity_tokens,
                    "stored_seq_len": stored_seq_len,
                    "valid_seq_len": valid_seq_len,
                    "vocab_start_index": 0,
                    "vocab_end_index": vocab_size,
                    "global_seq_start": 0,
                    "actual_shape": (full_seq_len, vocab_size),
                    "full_seq_len": full_seq_len,
                    "full_vocab_size": vocab_size,
                    "dtype": torch.float32,
                }
            ]
        }

    def test_fastpath_returns_zero_copy_view(self, monkeypatch):
        storage = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(1, 2, 4, 5)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda h, d: storage,
        )
        out = _try_zero_copy_teacher_logits(
            [self._entry(0), self._entry(1)],
            student_cp_rank=0,
            student_cp_size=1,
            device=0,
        )
        assert out is not None and out.data_ptr() == storage.data_ptr()
        assert torch.equal(out, storage[0, :2])

    def test_fastpath_applies_cp_seq_offset(self, monkeypatch):
        # student_cp_size=2, rank=1: the view must be sliced to this rank's
        # contiguous seq window [T//2 : T], not the whole sequence. With
        # cp_size=1 (above) seq_lo/seq_hi collapse to [0:T], so this case is
        # what guards the `student_seq_start - teacher_seq_start` offset.
        storage = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(1, 2, 4, 5)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda h, d: storage,
        )
        out = _try_zero_copy_teacher_logits(
            [self._entry(0), self._entry(1)],
            student_cp_rank=1,
            student_cp_size=2,
            device=0,
        )
        # full_seq_len=4 -> rank 1 owns seq [2:4]; teacher_seq_start=0. Still a
        # zero-copy view, but starting at the seq-offset row (so its data_ptr
        # matches the offset slice, not storage's base pointer).
        expected = storage[0, :2, 2:4, :]
        assert out is not None and out.data_ptr() == expected.data_ptr()
        assert torch.equal(out, expected)

    def test_none_when_vocab_sharded(self):
        e = self._entry(0)
        e["teacher_shards"][0]["vocab_end_index"] = 3
        assert (
            _try_zero_copy_teacher_logits(
                [e], student_cp_rank=0, student_cp_size=1, device=0
            )
            is None
        )

    def test_none_when_samples_not_contiguous(self):
        assert (
            _try_zero_copy_teacher_logits(
                [self._entry(0), self._entry(2)],
                student_cp_rank=0,
                student_cp_size=1,
                device=0,
            )
            is None
        )

    def test_rebuild_reports_zero_for_actual_zero_copy(self, monkeypatch):
        storage = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(1, 2, 4, 5)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda _handle, _device: storage,
        )

        out, fallback_count = rebuild_teacher_full_logits_from_ipc(
            [self._entry(0), self._entry(1)], cp_group=None, device=0
        )

        assert fallback_count == 0
        assert out.data_ptr() == storage.data_ptr()
        assert torch.equal(out, storage[0, :2])

    def test_compact_mbs1_full_row_is_a_live_zero_copy_view(self, monkeypatch):
        storage = torch.full((9, 5), -99.0)
        intended_row = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
        storage[3:7].copy_(intended_row)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda _handle, _device: storage,
        )
        entry = self._compact_entry(
            token_offset=3,
            valid_seq_len=4,
            stored_seq_len=4,
            used_tokens=9,
            capacity_tokens=9,
        )

        out, fallback_count = rebuild_teacher_full_logits_from_ipc(
            [entry], cp_group=None, device="cpu"
        )

        assert fallback_count == 0
        assert out.data_ptr() == storage[3].data_ptr()
        assert torch.equal(out[0], intended_row)
        storage[4, 2] = -123.0
        assert out[0, 1, 2].item() == -123.0

    def test_compact_short_row_reconstructs_exact_zero_suffix(self, monkeypatch):
        storage = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda _handle, _device: storage,
        )
        entry = self._compact_entry(
            token_offset=0,
            valid_seq_len=2,
            stored_seq_len=2,
            used_tokens=2,
            capacity_tokens=2,
        )

        out, fallback_count = rebuild_teacher_full_logits_from_ipc(
            [entry], cp_group=None, device="cpu"
        )

        expected = torch.zeros(1, 4, 5)
        expected[0, :2] = storage
        assert fallback_count == 1
        assert torch.equal(out, expected)

    def test_compact_zero_row_never_maps_a_payload(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda _handle, _device: pytest.fail("zero row must not map IPC"),
        )
        entry = self._compact_entry(
            token_offset=0,
            valid_seq_len=0,
            stored_seq_len=0,
            used_tokens=0,
            capacity_tokens=0,
            payload=None,
        )

        out, fallback_count = rebuild_teacher_full_logits_from_ipc(
            [entry], cp_group=None, device="cpu"
        )

        assert fallback_count == 1
        assert torch.equal(out, torch.zeros(1, 4, 5))

    def test_compact_rebuilt_storage_shape_must_match_handle(self, monkeypatch):
        storage = torch.zeros(3, 5)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda _handle, _device: storage,
        )
        entry = self._compact_entry(
            token_offset=0,
            valid_seq_len=4,
            stored_seq_len=4,
            used_tokens=4,
            capacity_tokens=4,
        )

        with pytest.raises(ValueError, match="unexpected storage shape"):
            rebuild_teacher_full_logits_from_ipc([entry], cp_group=None, device="cpu")

    def test_rebuild_counts_rows_that_use_shard_assembly(self, monkeypatch):
        vocab_left = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)
        vocab_right = 100 + torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(
            1, 2, 4, 2
        )
        storages = {"left": vocab_left, "right": vocab_right}
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda handle, _device: storages[handle],
        )

        entries = []
        for sample_idx in range(2):
            entries.append(
                {
                    "teacher_shards": [
                        {
                            "payload_ipc": "left",
                            "buf_idx": 0,
                            "sample_index_in_buf": sample_idx,
                            "vocab_start_index": 0,
                            "vocab_end_index": 3,
                            "global_seq_start": 0,
                            "actual_shape": (4, 3),
                            "full_seq_len": 4,
                            "full_vocab_size": 5,
                        },
                        {
                            "payload_ipc": "right",
                            "buf_idx": 0,
                            "sample_index_in_buf": sample_idx,
                            "vocab_start_index": 3,
                            "vocab_end_index": 5,
                            "global_seq_start": 0,
                            "actual_shape": (4, 2),
                            "full_seq_len": 4,
                            "full_vocab_size": 5,
                        },
                    ]
                }
            )

        out, fallback_count = rebuild_teacher_full_logits_from_ipc(
            entries, cp_group=None, device="cpu"
        )

        assert fallback_count == 2
        assert torch.equal(out, torch.cat([vocab_left[0], vocab_right[0]], dim=-1))

    def test_single_row_fallback_adds_batch_view_without_copy(self, monkeypatch):
        assembled = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
        entry = self._entry(0)
        # TP-sharded metadata prevents the full-vocabulary zero-copy path.
        entry["teacher_shards"][0]["vocab_end_index"] = 3
        monkeypatch.setattr(
            "nemo_rl.algorithms.x_token.loss_utils.assemble_teacher_logits_from_shards",
            lambda *_args, **_kwargs: assembled,
        )

        out, fallback_count = rebuild_teacher_full_logits_from_ipc(
            [entry], cp_group=None, device=0
        )

        assert fallback_count == 1
        assert out.shape == (1, 4, 5)
        assert out.data_ptr() == assembled.data_ptr()
        assert torch.equal(out[0], assembled)


class TestRebuildSparseTeacherLogits:
    @staticmethod
    def _shard(prefix: str, start: int, seq_len: int, full_seq_len: int) -> dict:
        return {
            "topk_logits_ipc": f"{prefix}-logits",
            "topk_indices_ipc": f"{prefix}-indices",
            "log_z_ipc": f"{prefix}-logz",
            "gt_in_topk_ipc": f"{prefix}-gt",
            "topk_shape": (seq_len, 2),
            "global_seq_start": start,
            "full_seq_len": full_seq_len,
        }

    def test_orders_cp_shards_and_preserves_dtypes(self, monkeypatch):
        tensors = {
            "a-logits": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "a-indices": torch.tensor([[1, 3], [2, 4]], dtype=torch.int64),
            "a-logz": torch.tensor([5.0, 6.0]),
            "a-gt": torch.tensor([1, 0], dtype=torch.int32),
            "b-logits": torch.tensor([[7.0, 8.0]]),
            "b-indices": torch.tensor([[5, 6]], dtype=torch.int64),
            "b-logz": torch.tensor([9.0]),
            "b-gt": torch.tensor([0], dtype=torch.int32),
        }
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda handle, _device: tensors[handle],
        )
        entries = [
            {
                "teacher_shards": [
                    self._shard("b", 2, 1, 3),
                    self._shard("a", 0, 2, 3),
                ]
            }
        ]

        logits, indices, log_z, gt_in_topk = rebuild_teacher_sparse_logits_from_ipc(
            entries, device=0
        )

        assert logits.shape == (1, 3, 2)
        assert logits.dtype == torch.float32
        assert torch.equal(logits[0, :, 0], torch.tensor([1.0, 3.0, 7.0]))
        assert indices.dtype == torch.int32
        assert torch.equal(indices[0], torch.tensor([[1, 3], [2, 4], [5, 6]]))
        assert log_z.dtype == torch.float32
        assert torch.equal(log_z, torch.tensor([[5.0, 6.0, 9.0]]))
        assert gt_in_topk is not None and gt_in_topk.dtype == torch.bool
        assert torch.equal(gt_in_topk, torch.tensor([[True, False, False]]))

    def test_rejects_noncontiguous_cp_coverage(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda _handle, _device: torch.zeros(1),
        )
        entries = [
            {"teacher_shards": [self._shard("gap", 1, 1, 2)]},
        ]
        with pytest.raises(ValueError, match="expected start 0, got 1"):
            rebuild_teacher_sparse_logits_from_ipc(entries, device=0)


class TestAssembleTeacherLogitsFromShards:
    """CPU coverage of the heterogeneous teacher-TP/CP -> student-CP reassembly.

    Monkeypatches ``rebuild_cuda_tensor_from_ipc`` so the IPC slice-write logic
    (the #2682 path) is exercised without GPUs/IPC. A known full teacher-logit
    tensor is split into a teacher TP2xCP2 grid (4 shards); each student CP rank
    must reassemble exactly its ``[T_t/CP_s, V_t]`` contiguous-seq, full-vocab
    window.
    """

    @staticmethod
    def _build_tp2cp2(full):
        """Split ``full`` [T, V] into a teacher TP2xCP2 grid -> (shards, storage_map)."""
        full_seq_len, full_vocab_size = full.shape
        seq_mid, vocab_mid = full_seq_len // 2, full_vocab_size // 2
        storage_map = {}
        shards = []
        for cp, (s0, s1) in enumerate([(0, seq_mid), (seq_mid, full_seq_len)]):
            for tp, (v0, v1) in enumerate(
                [(0, vocab_mid), (vocab_mid, full_vocab_size)]
            ):
                payload = (f"cp{cp}tp{tp}",)
                # Producer storage layout: [N_microbatches, B_mb, T_local, V_local].
                storage_map[payload] = (
                    full[s0:s1, v0:v1].clone().reshape(1, 1, s1 - s0, v1 - v0)
                )
                shards.append(
                    {
                        "payload_ipc": payload,
                        "buf_idx": 0,
                        "sample_index_in_buf": 0,
                        "vocab_start_index": v0,
                        "vocab_end_index": v1,
                        "global_seq_start": s0,
                        "actual_shape": (s1 - s0, v1 - v0),
                        "full_seq_len": full_seq_len,
                        "full_vocab_size": full_vocab_size,
                    }
                )
        return shards, storage_map

    @staticmethod
    def _build_compact_tp2cp2(full, valid_seq_len):
        """Split one valid-prefix row into flat teacher TP2xCP2 IPC slabs."""
        full_seq_len, full_vocab_size = full.shape
        seq_mid, vocab_mid = full_seq_len // 2, full_vocab_size // 2
        storage_map = {}
        shards = []
        for cp, (seq_start, seq_end) in enumerate(
            [(0, seq_mid), (seq_mid, full_seq_len)]
        ):
            stored_seq_len = max(0, min(valid_seq_len, seq_end) - seq_start)
            for tp, (vocab_start, vocab_end) in enumerate(
                [(0, vocab_mid), (vocab_mid, full_vocab_size)]
            ):
                payload = f"compact-cp{cp}tp{tp}"
                if stored_seq_len > 0:
                    storage_map[payload] = full[
                        seq_start : seq_start + stored_seq_len,
                        vocab_start:vocab_end,
                    ].clone()
                    payload_ipc = payload
                else:
                    payload_ipc = None
                local_vocab_size = vocab_end - vocab_start
                shards.append(
                    {
                        "payload_ipc": payload_ipc,
                        "buf_idx": 0,
                        "sample_index_in_buf": 0,
                        "ipc_layout": DENSE_TEACHER_IPC_FLAT_LAYOUT,
                        "storage_shape": (stored_seq_len, local_vocab_size),
                        "storage_token_offset": 0,
                        "storage_used_tokens": stored_seq_len,
                        "storage_capacity_tokens": stored_seq_len,
                        "stored_seq_len": stored_seq_len,
                        "valid_seq_len": valid_seq_len,
                        "vocab_start_index": vocab_start,
                        "vocab_end_index": vocab_end,
                        "global_seq_start": seq_start,
                        "actual_shape": (seq_end - seq_start, local_vocab_size),
                        "full_seq_len": full_seq_len,
                        "full_vocab_size": full_vocab_size,
                        "tp_rank": tp,
                        "tp_size": 2,
                        "cp_rank": cp,
                        "cp_size": 2,
                        "dtype": torch.float32,
                    }
                )
        return shards, storage_map

    @pytest.mark.parametrize("student_cp_rank", [0, 1])
    def test_tp2cp2_reassembly(self, monkeypatch, student_cp_rank):
        full = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
        shards, storage_map = self._build_tp2cp2(full)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda payload, device: storage_map[payload],
        )
        out = assemble_teacher_logits_from_shards(
            shards,
            student_cp_rank=student_cp_rank,
            student_cp_size=2,
            device="cpu",
        )
        lo, hi = student_cp_rank * 4, (student_cp_rank + 1) * 4
        assert out.shape == (4, 6)
        assert torch.equal(out, full[lo:hi, :])

    @pytest.mark.parametrize(
        "student_cp_size,student_cp_rank",
        [(1, 0), (2, 0), (2, 1), (4, 2), (4, 3)],
    )
    def test_compact_tp2cp2_reassembly_zero_fills_across_cp_remap(
        self, monkeypatch, student_cp_size, student_cp_rank
    ):
        full = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
        valid_seq_len = 5
        shards, storage_map = self._build_compact_tp2cp2(full, valid_seq_len)
        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
            lambda payload, _device: storage_map[payload],
        )

        out = assemble_teacher_logits_from_shards(
            shards,
            student_cp_rank=student_cp_rank,
            student_cp_size=student_cp_size,
            device="cpu",
        )

        expected = full.clone()
        expected[valid_seq_len:].zero_()
        local_seq_len = full.shape[0] // student_cp_size
        seq_start = student_cp_rank * local_seq_len
        assert out.shape == (local_seq_len, full.shape[1])
        assert torch.equal(out, expected[seq_start : seq_start + local_seq_len])

    def test_compact_zero_teacher_cp_shard_is_not_mapped(self, monkeypatch):
        full = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
        shards, storage_map = self._build_compact_tp2cp2(full, valid_seq_len=4)

        def rebuild(payload, _device):
            assert payload is not None
            return storage_map[payload]

        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc", rebuild
        )
        out = assemble_teacher_logits_from_shards(
            shards, student_cp_rank=0, student_cp_size=1, device="cpu"
        )

        expected = full.clone()
        expected[4:].zero_()
        assert torch.equal(out, expected)

    def test_non_divisible_seq_len_asserts(self):
        # full_seq_len=7 not divisible by student_cp_size=2 -> guard fires before
        # the out-of-bounds dest write the assert protects against.
        shards = [
            {
                "payload_ipc": ("x",),
                "buf_idx": 0,
                "sample_index_in_buf": 0,
                "vocab_start_index": 0,
                "vocab_end_index": 6,
                "global_seq_start": 0,
                "actual_shape": (7, 6),
                "full_seq_len": 7,
                "full_vocab_size": 6,
            }
        ]
        with pytest.raises(AssertionError):
            assemble_teacher_logits_from_shards(
                shards, student_cp_rank=0, student_cp_size=2, device="cpu"
            )


# ---------------------------------------------------------------------------
# Real multi-GPU (NCCL) tests for the TP/CP loss helpers.
#
# tp2cp1 vocab-shards the student logits across 2 ranks; tp1cp2 seq-shards the
# teacher logits across 2 ranks. Each rank compares its sharded helper output
# against the single-rank full-tensor reference. Skipped when < 2 GPUs.
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class XtokenShardTestActor:
    def __init__(self, tp_size, cp_size, sharding):
        self.tp_size = tp_size
        self.cp_size = cp_size

    def run_tp(self):
        from nemo_rl.algorithms.x_token.loss_utils import (
            project_student_to_teacher_vocab,
            slice_sparse_projection_cols,
        )
        from nemo_rl.distributed.model_utils import (
            vocab_parallel_argmax,
            vocab_parallel_full_log_softmax,
            vocab_parallel_log_softmax,
        )

        try:
            torch.distributed.init_process_group(backend="nccl")
            rank = int(os.environ["RANK"])
            ws = int(os.environ["WORLD_SIZE"])
            tp = torch.distributed.new_group(ranks=list(range(ws)))

            torch.manual_seed(0)
            B, T, Vs, Vt, temp = 2, 4, 8, 6, 1.5
            shard = Vs // ws
            sl = slice(rank * shard, (rank + 1) * shard)
            full = torch.randn(B, T, Vs, device="cuda")
            full_log = torch.log_softmax(full.float() / temp, dim=-1)
            local = full[:, :, sl].contiguous()

            torch.testing.assert_close(
                vocab_parallel_log_softmax(local, temp, tp_group=tp),
                full_log[:, :, sl],
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                vocab_parallel_full_log_softmax(local, temp, tp_group=tp),
                full_log,
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                vocab_parallel_argmax(local, tp_group=tp), full.argmax(dim=-1)
            )

            idx = torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 0, 1]], device="cuda"
            )
            m = torch.sparse_coo_tensor(
                idx, torch.ones(8, device="cuda"), (Vs, Vt)
            ).coalesce()
            full_probs = full_log.exp()
            proj = project_student_to_teacher_vocab(
                full_probs[:, :, sl].contiguous(), m, tp_group=tp
            )
            ref = (full_probs.reshape(-1, Vs) @ m.to_dense()).reshape(B, T, Vt)
            torch.testing.assert_close(proj, ref, rtol=1e-4, atol=1e-4)

            # Column-slicing the projection must compose with the per-rank ROW
            # slice that project_student_to_teacher_vocab does internally: the
            # two act on different axes. _compute_p_kl relies on this to
            # produce only the top-k teacher columns instead of all V_t.
            cols = torch.tensor([1, 3, 5], device="cuda")
            proj_sliced = project_student_to_teacher_vocab(
                full_probs[:, :, sl].contiguous(),
                slice_sparse_projection_cols(m, cols),
                tp_group=tp,
            )
            torch.testing.assert_close(
                proj_sliced, ref[..., cols], rtol=1e-4, atol=1e-4
            )
            assert proj_sliced.shape == (B, T, cols.numel())
            return {"success": True, "error": None}
        except Exception:
            return {"success": False, "error": traceback.format_exc()}
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    def run_cp(self):
        from nemo_rl.algorithms.x_token.loss_utils import (
            chunk_average_log_probs,
            select_teacher_topk_indices,
        )

        try:
            torch.distributed.init_process_group(backend="nccl")
            rank = int(os.environ["RANK"])
            ws = int(os.environ["WORLD_SIZE"])
            cp = torch.distributed.new_group(ranks=list(range(ws)))

            torch.manual_seed(0)
            B, T, V, k = 2, 8, 6, 3
            shard = T // ws
            sl = slice(rank * shard, (rank + 1) * shard)
            full = torch.randn(B, T, V, device="cuda")

            torch.testing.assert_close(
                select_teacher_topk_indices(
                    full[:, sl, :].contiguous(), k, cp_group=cp
                ),
                torch.topk(full.reshape(-1, V).max(dim=0).values, k)
                .indices.sort()
                .values,
            )

            full_lp = torch.log_softmax(full, dim=-1)
            cid = torch.tensor(
                [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0]], device="cuda"
            )
            avg, sizes = chunk_average_log_probs(
                full_lp[:, sl, :].contiguous(), cid[:, sl].contiguous(), 2, cp_group=cp
            )
            ref_avg, ref_sizes = chunk_average_log_probs(full_lp, cid, 2, cp_group=None)
            torch.testing.assert_close(avg, ref_avg, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(sizes, ref_sizes)

            x = torch.full((3,), float(rank + 1), device="cuda", requires_grad=True)
            y = group_all_reduce_sum_with_grad(x, cp)
            torch.testing.assert_close(
                y, torch.full((3,), float(ws * (ws + 1) // 2), device="cuda")
            )
            y.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones(3, device="cuda"))
            return {"success": True, "error": None}
        except Exception:
            return {"success": False, "error": traceback.format_exc()}
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    def run_v6(self):
        """TP/CP parallelism-invariance for the v6 cross-tokenizer partition KL.

        Builds a full-tensor single-GPU reference (loss + student-logit grad) and
        a sharded run — student vocab-sharded over TP, student/teacher sequence
        contiguous-block-sharded over CP — of
        ``_compute_prefix_bidir_partition_kl_v3``, then asserts:

          * each rank's loss == ``reference_loss / cp_size``. The in-loss
            ``global_valid_chunks`` normalizer is the DP×CP-reduced chunk count
            (WORLD-reduce ÷ tp_world), so TP is *replicated* (no TP factor,
            matching CE's ``global_valid_toks``) while each CP rank holds a
            ``1/cp_size`` share that sums back to the full loss; and
          * the student-logit gradient, reconstructed from the shards, equals
            ``reference_grad`` exactly. The CP contiguous-gather backward
            all-reduces the per-rank grad to full magnitude (the CP factor
            cancels) and TP carries no loss-scale factor, so nothing is left over.

        The reference passes ``global_valid_chunks`` explicitly (so it does *not*
        WORLD-reduce) with ``tp_group=cp_group=None`` (gathers are no-ops), so it
        is a purely local per-rank computation with no collectives.
        """
        import os
        import tempfile

        from nemo_rl.algorithms.loss.loss_functions import (
            CrossTokenizerDistillationLossFn,
        )
        from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment

        try:
            torch.distributed.init_process_group(backend="nccl")
            rank = int(os.environ["RANK"])
            ws = int(os.environ["WORLD_SIZE"])
            tp_size, cp_size = self.tp_size, self.cp_size
            assert ws == tp_size * cp_size
            device = torch.device("cuda")
            # NamedSharding layout is ``arange(ws).reshape(tp, cp)`` so
            # ``rank == tp_idx * cp_size + cp_idx``.
            tp_idx, cp_idx = rank // cp_size, rank % cp_size

            # TP / CP subgroups. Every rank must invoke every ``new_group``
            # (it is a collective over the default group), keeping only its own.
            tp_group = None
            for c in range(cp_size):
                g = torch.distributed.new_group(
                    ranks=[t * cp_size + c for t in range(tp_size)]
                )
                if c == cp_idx:
                    tp_group = g
            cp_group = None
            for t in range(tp_size):
                g = torch.distributed.new_group(
                    ranks=[t * cp_size + c for c in range(cp_size)]
                )
                if t == tp_idx:
                    cp_group = g
            if tp_size == 1:
                tp_group = None
            if cp_size == 1:
                cp_group = None

            # ---- Fixture (identical on every rank; sharded below) ----------
            # Qualification-shaped v6 fixture with repeated common, 1-to-2, and
            # 2-to-1 chunks. Counts cross both fixed streaming caps (64 common,
            # 8 position-0 rows); shifted teacher predictors live on both CP
            # owners. Position-0 uses JSD and mismatch-last uses KL.
            torch.manual_seed(0)
            B, S, T, v_s, v_t = 2, 6, 6, 16, 16
            temp = 1.0
            student_ids = torch.tensor(
                [[0, 5, 0, 9, 5, 0], [1, 6, 0, 9, 6, 0]], dtype=torch.long
            )
            teacher_ids = torch.tensor(
                [[0, 8, 3, 0, 5, 0], [1, 8, 4, 0, 6, 0]], dtype=torch.long
            )
            chunks = [(0, 1, 0, 1)] * 33 + [(1, 2, 1, 3), (3, 5, 4, 5)] * 5
            max_pairs = len(chunks)
            s_spans = torch.zeros(B, max_pairs, 2, dtype=torch.long)
            t_spans = torch.zeros(B, max_pairs, 2, dtype=torch.long)
            pair_valid = torch.zeros(B, max_pairs, dtype=torch.bool)
            num_chunks = torch.full((B,), max_pairs, dtype=torch.long)
            for b in range(B):
                for k, (ss, se, ts, te) in enumerate(chunks):
                    s_spans[b, k, 0], s_spans[b, k, 1] = ss, se
                    t_spans[b, k, 0], t_spans[b, k, 1] = ts, te
                    pair_valid[b, k] = True
            student_full = torch.randn(B, S, v_s)  # CPU -> deterministic
            # Huge padded columns prove dense v6 slices to the real tokenizer
            # vocabulary before exact logZ and CP-owner lookups.
            teacher_full = torch.randn(B, T, v_t + 2)
            teacher_full[..., v_t:] = 1.0e4
            gvc = float(B * len(chunks))  # total valid chunks (single-rank)

            student_ids = student_ids.to(device)
            teacher_ids = teacher_ids.to(device)
            s_spans, t_spans = s_spans.to(device), t_spans.to(device)
            pair_valid, num_chunks = pair_valid.to(device), num_chunks.to(device)
            student_full, teacher_full = (
                student_full.to(device),
                teacher_full.to(device),
            )

            # Tiny common + bidirectional prefix-support tables.
            tmpdir = tempfile.mkdtemp()
            subtoks = torch.full((v_s, 3), -1, dtype=torch.long)
            lengths = torch.zeros((v_s,), dtype=torch.long)
            for s in range(10):
                subtoks[s, 0], lengths[s] = s, 1
            subtoks[10, 0], subtoks[10, 1], lengths[10] = 8, 3, 2
            subtoks[11, 0], subtoks[11, 1], lengths[11] = 8, 4, 2
            for s in range(12, v_s):
                subtoks[s, 0], subtoks[s, 1], lengths[s] = s % 10, (s + 3) % 10, 2
            fwd_path = os.path.join(tmpdir, "fwd.pt")
            rev_path = os.path.join(tmpdir, "rev.pt")
            torch.save({"subtoks": subtoks, "lengths": lengths}, fwd_path)
            reverse_subtoks = torch.full((v_t, 3), -1, dtype=torch.long)
            reverse_lengths = torch.zeros((v_t,), dtype=torch.long)
            for t in range(10):
                reverse_subtoks[t, 0], reverse_lengths[t] = t, 1
            reverse_subtoks[10, 0], reverse_subtoks[10, 1], reverse_lengths[10] = (
                9,
                3,
                2,
            )
            reverse_subtoks[11, 0], reverse_subtoks[11, 1], reverse_lengths[11] = (
                9,
                4,
                2,
            )
            for t in range(12, v_t):
                reverse_subtoks[t, 0], reverse_subtoks[t, 1], reverse_lengths[t] = (
                    t % 10,
                    (t + 3) % 10,
                    2,
                )
            torch.save(
                {"subtoks": reverse_subtoks, "lengths": reverse_lengths}, rev_path
            )

            cfg = {
                "temperature": temp,
                "vocab_topk": 8,
                "reverse_kl": False,
                "kl_loss_weight": 1.0,
                "ce_loss_scale": 1.0,
                "dynamic_loss_scaling": False,
                "kd_loss_mode": "sum",
                "normalize_teacher_by_vocab": False,
                "alpha": 1.0,
                "sum_weights_metric": None,
                "student_vocab_size": v_s,
                "teacher_vocab_sizes": [v_t],
                "projection_matrix_paths": ["dummy.pt"],
                "teacher_weights": [1.0],
                "common_indices_from_subtoks": True,
                "pseudo_target_paths": [fwd_path],
                "reverse_pseudo_target_paths": [rev_path],
                "kl_chunk_shift": True,
                "prefix_bidir_v3_position_0_kl": True,
                "prefix_bidir_v3_loss_fn": "jsd",
                "prefix_bidir_v3_last_pos_loss_fn": "kl",
                "prefix_bidir_v3_jsd_beta": 0.5,
                "prefix_bidir_v3_noise_filter_topk": 0,
                "teacher_topk_ipc_k": 0,
                "teacher_topk_ipc_support_mode": "row_topk",
                "teacher_topk_ipc_keep_realized": True,
            }
            loss_fn = CrossTokenizerDistillationLossFn(cfg)

            def _align(student_input_ids, teacher_input_ids):
                return LocalizedAlignment(
                    sample_mask=torch.ones(B, dtype=torch.bool, device=device),
                    pair_valid=pair_valid,
                    student_input_ids=student_input_ids,
                    teacher_input_ids=teacher_input_ids,
                    student_spans=s_spans,
                    teacher_spans=t_spans,
                    num_chunks=num_chunks,
                )

            # ---- Single-GPU reference (explicit gvc => no WORLD-reduce; tp/cp
            #      None => gathers are no-ops => purely local, no collectives). --
            ref_logits = student_full.clone().requires_grad_(True)
            ref_loss, ref_metrics = loss_fn._compute_prefix_bidir_partition_kl_v3(
                0,
                ref_logits,
                teacher_full[..., :v_t].clone(),
                _align(student_ids, teacher_ids),
                teacher_vocab_size=v_t,
                tp_group=None,
                cp_group=None,
                global_valid_chunks=torch.tensor(gvc, device=device),
            )
            ref_loss.backward()
            ref_grad = ref_logits.grad.detach().clone()  # [B, S, v_s]

            # ---- Sharded run: CP contiguous seq block x TP vocab slice. -------
            sblk, tblk, vblk = S // cp_size, T // cp_size, v_s // tp_size
            s_lo, t_lo, v_lo = cp_idx * sblk, cp_idx * tblk, tp_idx * vblk
            student_shard = (
                student_full[:, s_lo : s_lo + sblk, v_lo : v_lo + vblk]
                .detach()
                .clone()
                .requires_grad_(True)
            )
            # Teacher is full-vocab (only CP-sharded on the sequence).
            teacher_shard = teacher_full[:, t_lo : t_lo + tblk, :].detach().clone()
            import nemo_rl.algorithms.loss.loss_functions as loss_functions_module

            original_cp_gather = loss_functions_module.allgather_cp_contiguous_tensor

            def reject_dense_teacher_sequence_gather(tensor, group, seq_dim=1):
                if (
                    group is cp_group
                    and tensor.ndim == 3
                    and not tensor.requires_grad
                    and tensor.shape[0] == B
                    and tensor.shape[1] == tblk
                    and tensor.shape[-1] >= v_t
                ):
                    raise AssertionError(
                        "dense teacher [B,T/CP,V] entered a full CP sequence gather"
                    )
                return original_cp_gather(tensor, group, seq_dim)

            loss_functions_module.allgather_cp_contiguous_tensor = (
                reject_dense_teacher_sequence_gather
            )
            try:
                sh_loss, sh_metrics = loss_fn._compute_prefix_bidir_partition_kl_v3(
                    0,
                    student_shard,
                    teacher_shard,
                    _align(
                        student_ids[:, s_lo : s_lo + sblk].contiguous(),
                        teacher_ids[:, t_lo : t_lo + tblk].contiguous(),
                    ),
                    teacher_vocab_size=v_t,
                    tp_group=tp_group,
                    cp_group=cp_group,
                    global_valid_chunks=None,  # => in-loss WORLD-reduce over ws
                )
            finally:
                loss_functions_module.allgather_cp_contiguous_tensor = (
                    original_cp_gather
                )
            sh_loss.backward()
            sh_grad = student_shard.grad.detach().clone()  # [B, S/cp, v_s/tp]

            # Loss invariance: every rank == reference / cp_size (TP replicated;
            # each CP rank holds a 1/cp_size share of the full loss).
            torch.testing.assert_close(
                sh_loss.detach(),
                ref_loss.detach() / cp_size,
                rtol=1e-4,
                atol=1e-4,
            )

            # Grad invariance: reconstruct the full student grad from the shards;
            # it equals the single-GPU reference exactly (CP contributions
            # all-reduce to full magnitude, TP carries no loss-scale factor).
            gathered = [torch.empty_like(sh_grad) for _ in range(ws)]
            torch.distributed.all_gather(gathered, sh_grad.contiguous())
            recon = torch.zeros_like(ref_grad)
            for r in range(ws):
                r_tp, r_cp = r // cp_size, r % cp_size
                recon[
                    :, r_cp * sblk : (r_cp + 1) * sblk, r_tp * vblk : (r_tp + 1) * vblk
                ] = gathered[r]
            torch.testing.assert_close(recon, ref_grad, rtol=1e-4, atol=1e-4)

            # Production supplies the full-batch chunk denominator. The raw
            # replicated CP loss then equals the CP1 reference; MCore's CP loss
            # wrapper contributes the 1/CP factor before backward.
            explicit_student_shard = (
                student_full[:, s_lo : s_lo + sblk, v_lo : v_lo + vblk]
                .detach()
                .clone()
                .requires_grad_(True)
            )
            explicit_loss, explicit_metrics = (
                loss_fn._compute_prefix_bidir_partition_kl_v3(
                    0,
                    explicit_student_shard,
                    teacher_shard,
                    _align(
                        student_ids[:, s_lo : s_lo + sblk].contiguous(),
                        teacher_ids[:, t_lo : t_lo + tblk].contiguous(),
                    ),
                    teacher_vocab_size=v_t,
                    tp_group=tp_group,
                    cp_group=cp_group,
                    global_valid_chunks=torch.tensor(gvc, device=device),
                )
            )
            torch.testing.assert_close(
                explicit_loss.detach(), ref_loss.detach(), rtol=1e-4, atol=1e-4
            )
            (explicit_loss / float(cp_size)).backward()
            explicit_grad = explicit_student_shard.grad.detach().clone()
            gathered_explicit = [torch.empty_like(explicit_grad) for _ in range(ws)]
            torch.distributed.all_gather(
                gathered_explicit,
                explicit_grad.contiguous(),
            )
            explicit_recon = torch.zeros_like(ref_grad)
            for r in range(ws):
                r_tp, r_cp = r // cp_size, r % cp_size
                explicit_recon[
                    :,
                    r_cp * sblk : (r_cp + 1) * sblk,
                    r_tp * vblk : (r_tp + 1) * vblk,
                ] = gathered_explicit[r]
            torch.testing.assert_close(
                explicit_recon,
                ref_grad,
                rtol=1e-4,
                atol=1e-4,
            )
            sh_metrics = explicit_metrics
            assert sh_metrics["num_common_chunks"] > 0
            assert sh_metrics["num_mismatch_chunks"] > 0
            assert torch.isfinite(
                torch.tensor(sh_metrics["kl_partition_first_per_chunk"])
            )
            for metric_name in (
                "kl_common_per_chunk",
                "kl_partition_last_per_chunk",
                "kl_partition_first_per_chunk",
                "kl_mismatch_combined_per_chunk",
                "kl_mismatch_scaled_per_chunk",
                "top1_acc_per_chunk",
            ):
                assert sh_metrics[metric_name] == pytest.approx(
                    ref_metrics[metric_name], rel=1.0e-4, abs=1.0e-4
                )
            for metric_name in (
                "num_common_chunks",
                "num_mismatch_chunks",
                "num_valid_samples",
            ):
                assert sh_metrics[metric_name] == ref_metrics[metric_name]
            return {"success": True, "error": None}
        except Exception:
            return {"success": False, "error": traceback.format_exc()}
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


_ACTOR_FQN = f"{XtokenShardTestActor.__module__}.XtokenShardTestActor"


@pytest.fixture
def register_actor():
    original = ACTOR_ENVIRONMENT_REGISTRY.get(_ACTOR_FQN)
    ACTOR_ENVIRONMENT_REGISTRY[_ACTOR_FQN] = PY_EXECUTABLES.SYSTEM
    yield _ACTOR_FQN
    if original is None:
        ACTOR_ENVIRONMENT_REGISTRY.pop(_ACTOR_FQN, None)
    else:
        ACTOR_ENVIRONMENT_REGISTRY[_ACTOR_FQN] = original


def _run(register_actor, tp_size, cp_size, method):
    world_size = tp_size * cp_size
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, got {torch.cuda.device_count()}")
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[world_size], use_gpus=True)
    try:
        sharding = NamedSharding(
            layout=np.arange(world_size).reshape(tp_size, cp_size), names=["tp", "cp"]
        )
        worker_group = RayWorkerGroup(
            cluster=cluster,
            remote_worker_builder=RayWorkerBuilder(
                register_actor, tp_size, cp_size, sharding
            ),
            workers_per_node=None,
            sharding_annotations=sharding,
        )
        results = ray.get(worker_group.run_all_workers_single_data(method))
        for i, r in enumerate(results):
            assert r["success"], f"rank {i} failed:\n{r['error']}"
        worker_group.shutdown(force=True)
    finally:
        cluster.shutdown()


def test_tp2cp1_sharded_helpers(register_actor):
    _run(register_actor, tp_size=2, cp_size=1, method="run_tp")


def test_tp1cp2_sharded_helpers(register_actor):
    _run(register_actor, tp_size=1, cp_size=2, method="run_cp")


class TestSliceSparseProjectionCols:
    """Column-slicing the projection must equal slicing the dense product.

    ``_compute_p_kl`` relies on this: it folds the top-k column slice into the
    sparse matrix instead of projecting all V_t columns and discarding most of
    them. Every teacher column of ``M.t() @ p`` is an independent contraction
    over the student axis, so the reorder is value-preserving.
    """

    @staticmethod
    def _random_sparse(v_s: int, v_t: int, nnz: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        dense = torch.zeros(v_s, v_t)
        flat = torch.randint(0, v_s * v_t, (nnz,), generator=generator)
        dense.view(-1)[flat] = torch.randn(flat.numel(), generator=generator)
        return dense, dense.to_sparse_coo().coalesce()

    def test_matches_dense_column_slice(self):
        dense, sparse = self._random_sparse(40, 60, nnz=200)
        cols = torch.tensor([3, 7, 11, 40, 59])

        sliced = slice_sparse_projection_cols(sparse, cols).to_dense()

        assert sliced.shape == (40, cols.numel())
        assert torch.equal(sliced, dense[:, cols])

    def test_projection_commutes_with_the_column_slice(self):
        """slice-then-project == project-then-slice.

        Not asserted bitwise: the two matmuls contract over the same values but
        have different widths (V_t vs k), so BLAS is free to block and
        accumulate them differently. The per-column accumulation *set* is what
        is preserved, which is what makes the reorder legal.
        """
        generator = torch.Generator().manual_seed(3)
        dense, sparse = self._random_sparse(40, 60, nnz=200, seed=1)
        cols = torch.tensor([1, 2, 17, 33, 58])
        student = torch.randn(5, 40, generator=generator)

        project_then_slice = (student @ dense)[:, cols]
        slice_then_project = (
            student @ slice_sparse_projection_cols(sparse, cols).to_dense()
        )

        torch.testing.assert_close(
            project_then_slice, slice_then_project, rtol=1e-6, atol=1e-6
        )

    def test_columns_with_no_entries_are_kept_as_zeros(self):
        dense = torch.zeros(6, 10)
        dense[0, 4] = 1.5
        sparse = dense.to_sparse_coo().coalesce()
        # column 9 has no entries at all
        cols = torch.tensor([4, 9])

        sliced = slice_sparse_projection_cols(sparse, cols).to_dense()

        assert sliced.shape == (6, 2)
        assert sliced[0, 0].item() == pytest.approx(1.5)
        assert sliced[:, 1].abs().sum().item() == 0.0

    def test_selecting_every_column_is_the_identity(self):
        dense, sparse = self._random_sparse(12, 9, nnz=30, seed=2)
        cols = torch.arange(9)

        sliced = slice_sparse_projection_cols(sparse, cols).to_dense()

        assert torch.equal(sliced, dense)


def test_v6_tp2cp1(register_actor):
    """v6 partition-KL is invariant under TP=2 vocab sharding."""
    _run(register_actor, tp_size=2, cp_size=1, method="run_v6")


def test_v6_tp1cp2(register_actor):
    """v6 partition-KL is invariant under CP=2 sequence sharding."""
    _run(register_actor, tp_size=1, cp_size=2, method="run_v6")


def test_v6_tp2cp2(register_actor):
    """v6 partition-KL is invariant under composed TP=2 x CP=2 sharding."""
    _run(register_actor, tp_size=2, cp_size=2, method="run_v6")
