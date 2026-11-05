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
tests at the bottom of this file (skipped when < 2 GPUs are available). The
same-tokenizer KD helpers' CP=2 path runs on CPU over gloo.
"""

from __future__ import annotations

import datetime
import functools
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import ray
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.algorithms.x_token.loss_utils import (
    _EXACT_TOKEN_MAP_CACHE,
    _SPARSE_PROJECTION_CACHE,
    _TOPK_PROJECTION_CACHE,
    _try_zero_copy_teacher_logits,
    assemble_teacher_logits_from_shards,
    build_exact_token_map,
    chunk_average_log_probs,
    collect_overlapping_teacher_shards,
    get_sparse_projection_matrix,
    get_topk_projection,
    localize_alignment,
    parse_projection_file,
    prepare_xtoken_cross_tokenizer_loss_input,
    same_tokenizer_full_vocab_kl,
    same_tokenizer_topk_kl,
    select_teacher_topk_indices,
    slice_sparse_projection_cols,
    valid_chunk_mask,
)
from nemo_rl.distributed.model_utils import (
    cp_shift_next,
    group_all_reduce_sum_with_grad,
    to_local_if_dtensor,
    vocab_parallel_full_log_softmax,
    vocab_parallel_gather_columns,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


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
        "alignment_0_student_chunk_id": torch.tensor([[10, 11, 12, 13, 14, 15]]),
        "alignment_0_teacher_chunk_id": torch.tensor([[20, 21, 22, 23]]),
        "alignment_0_pair_valid": torch.ones(1, 2, dtype=torch.bool),
        "alignment_0_pair_is_correct": torch.ones(1, 2, dtype=torch.bool),
    }

    with (
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=1),
        patch(
            "nemo_rl.algorithms.x_token.loss_utils."
            "rebuild_teacher_full_logits_from_ipc",
            return_value=teacher_logits,
        ) as rebuild_teacher,
    ):
        student_logits, teachers, aligns, tp_group, returned_cp_group = (
            prepare_xtoken_cross_tokenizer_loss_input(
                local_logits,
                data,
                projection_matrix_paths=["projection.pt"],
                context_parallel_group=cp_group,
                cp_sharder=cp_sharder,
            )
        )

    torch.testing.assert_close(student_logits, full_student_logits[:, 3:6])
    assert teachers[0] is teacher_logits
    assert tp_group is None
    assert returned_cp_group is cp_group
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
        align = localize_alignment(data, teacher_seq_len=6)
        # teacher_seq_len covers the whole tensor, so it passes through.
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


# ---------------------------------------------------------------------------
# Same-tokenizer KD helpers: same_tokenizer_topk_kl /
# same_tokenizer_full_vocab_kl.
#
# The oracle below is a frozen copy of the same-tokenizer KD math from before
# the shared helpers existed; the helpers must match it bitwise. Keep it frozen
# (do not update it to track the helpers).
# ---------------------------------------------------------------------------

_V = 24  # shared tokenizer vocab size
_PAD = 8  # lm_head padding columns past the shared vocab


def _old_select_teacher_topk_indices(
    teacher_logits: torch.Tensor,
    k: int,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    vocab_size = teacher_logits.shape[-1]
    with torch.no_grad():
        teacher_flat = teacher_logits.reshape(-1, vocab_size)
        importance = teacher_flat.max(dim=0).values
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
            torch.distributed.all_reduce(
                importance, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )
        top_indices = torch.topk(importance, k=k, dim=-1).indices
        return top_indices.sort().values


def _old_same_vocab_masked_kl(
    per_pos: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    global_valid_toks: torch.Tensor,
    *,
    temperature: float,
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    T = temperature
    next_mask = cp_shift_next(to_local_if_dtensor(token_mask), cp_group, fill=0)
    sample_mask = to_local_if_dtensor(sample_mask)
    mask = next_mask.float() * sample_mask.unsqueeze(-1).float()
    return (
        masked_mean(per_pos, mask, global_normalization_factor=global_valid_toks)
        * T
        * T
    )


def _old_direct_topk_kl(
    student_logits: torch.Tensor,
    teacher_full_logits: torch.Tensor,
    *,
    kd_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    num_kd_tokens: torch.Tensor,
    vocab_size: int,
    vocab_topk: int,
    temperature: float,
    reverse_kl: bool,
    tp_group: Optional[torch.distributed.ProcessGroup],
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    T = temperature
    v_s = vocab_size
    teacher = teacher_full_logits
    if teacher.shape[-1] > v_s:
        teacher = teacher[..., :v_s]
    vocab_topk = min(vocab_topk, teacher.shape[-1])
    topk_idx = _old_select_teacher_topk_indices(teacher, vocab_topk, cp_group=cp_group)
    student_topk_logits = vocab_parallel_gather_columns(
        student_logits, topk_idx, tp_group=tp_group
    )
    student_log_probs_k = torch.log_softmax(student_topk_logits / T, dim=-1)
    teacher_log_probs_k = torch.log_softmax(teacher[..., topk_idx].float() / T, dim=-1)
    if reverse_kl:
        per_pos = torch.nn.functional.kl_div(
            teacher_log_probs_k,
            student_log_probs_k,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
    else:
        per_pos = torch.nn.functional.kl_div(
            student_log_probs_k,
            teacher_log_probs_k,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
    return _old_same_vocab_masked_kl(
        per_pos,
        kd_mask,
        sample_mask,
        num_kd_tokens,
        temperature=T,
        cp_group=cp_group,
    )


def _old_direct_full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    kd_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    num_kd_tokens: torch.Tensor,
    temperature: float,
    reverse_kl: bool,
    tp_group: Optional[torch.distributed.ProcessGroup],
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    T = temperature
    student_log_probs = vocab_parallel_full_log_softmax(
        student_logits, T, tp_group=tp_group
    )
    v_s = student_log_probs.shape[-1]
    teacher = teacher_logits.float()
    if teacher.shape[-1] > v_s:
        teacher = teacher[..., :v_s]
    teacher_log_probs = torch.log_softmax(teacher / T, dim=-1)
    if reverse_kl:
        per_pos = torch.nn.functional.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
    else:
        per_pos = torch.nn.functional.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
    return _old_same_vocab_masked_kl(
        per_pos,
        kd_mask,
        sample_mask,
        num_kd_tokens,
        temperature=T,
        cp_group=cp_group,
    )


def _manual_kd_weights(
    kd_mask: torch.Tensor, sample_mask: torch.Tensor
) -> torch.Tensor:
    """Predictor ``t`` receives KD iff target ``t + 1`` does and its sample is kept."""
    weights = torch.zeros(kd_mask.shape, dtype=torch.float32)
    weights[:, :-1] = kd_mask[:, 1:].float()
    return weights * sample_mask.float().unsqueeze(-1)


def _manual_topk_columns(teacher_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sorted top-``k`` columns by per-vocab max over every teacher row."""
    importance = teacher_logits.reshape(-1, teacher_logits.shape[-1]).float()
    return torch.topk(importance.max(dim=0).values, k).indices.sort().values


def _manual_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
    num_kd_tokens: float,
    *,
    temperature: float,
    reverse_kl: bool,
    columns: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """float64 ``sum_t(w_t * KL_t) / num_kd_tokens * T**2`` from explicit sums."""
    student = student_logits.double()
    teacher = teacher_logits.double()
    if columns is not None:
        student = student[..., columns]
        teacher = teacher[..., columns]
    log_q = torch.log_softmax(student / temperature, dim=-1)
    log_p = torch.log_softmax(teacher / temperature, dim=-1)
    if reverse_kl:
        per_pos = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
    else:
        per_pos = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return (per_pos * weights.double()).sum() / num_kd_tokens * temperature**2


def _same_tokenizer_kd(
    path: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    vocab_topk: int = 4,
    **kwargs: Any,
) -> torch.Tensor:
    if path == "topk":
        return same_tokenizer_topk_kl(
            student_logits,
            teacher_logits,
            vocab_size=_V,
            vocab_topk=vocab_topk,
            **kwargs,
        )
    assert path == "full_vocab"
    return same_tokenizer_full_vocab_kl(
        student_logits, teacher_logits, vocab_size=_V, **kwargs
    )


def _loss_and_grad(
    kd_fn: Callable[..., torch.Tensor],
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KD loss and its gradient w.r.t. a fresh leaf copy of the student logits."""
    student = student_logits.detach().clone().requires_grad_(True)
    loss = kd_fn(student, teacher_logits, **kwargs)
    loss.backward()
    assert student.grad is not None
    return loss.detach(), student.grad


def _kd_masks(
    kind: str, batch: int, seq: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(kd_mask [B, T], sample_mask [B])`` for one masking pattern."""
    if kind == "all_kd":
        return torch.ones(batch, seq), torch.ones(batch)
    if kind == "no_kd":
        return torch.zeros(batch, seq), torch.ones(batch)
    assert kind == "partial"
    kd_mask = (torch.rand(batch, seq, generator=generator) < 0.6).float()
    sample_mask = torch.ones(batch)
    sample_mask[1] = 0.0
    return kd_mask, sample_mask


class TestSameTokenizerKdMatchesPreviousMath:
    @pytest.mark.parametrize("mask_kind", ["all_kd", "partial", "no_kd"])
    @pytest.mark.parametrize(
        "teacher_width", [_V, _V + _PAD], ids=["no_pad", "lm_head_pad"]
    )
    @pytest.mark.parametrize("vocab_topk", [4, 8, _V, _V + _PAD])
    @pytest.mark.parametrize(
        "teacher_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
    )
    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    @pytest.mark.parametrize("reverse_kl", [False, True], ids=["fwd", "rev"])
    def test_topk_bitwise_for_any_mask(
        self,
        reverse_kl: bool,
        temperature: float,
        teacher_dtype: torch.dtype,
        vocab_topk: int,
        teacher_width: int,
        mask_kind: str,
    ):
        batch, seq = 3, 7
        generator = torch.Generator().manual_seed(0)
        student = torch.randn(batch, seq, teacher_width, generator=generator)
        teacher = torch.randn(batch, seq, teacher_width, generator=generator) * 3
        # lm_head padding columns dominate every row, so they would take the
        # top-K slots if they were not dropped.
        teacher[..., _V:] = 25.0
        teacher = teacher.to(teacher_dtype)
        kd_mask, sample_mask = _kd_masks(mask_kind, batch, seq, generator)
        kwargs = dict(
            kd_mask=kd_mask,
            sample_mask=sample_mask,
            num_kd_tokens=_manual_kd_weights(kd_mask, sample_mask).sum(),
            vocab_size=_V,
            vocab_topk=vocab_topk,
            temperature=temperature,
            reverse_kl=reverse_kl,
            tp_group=None,
            cp_group=None,
        )

        new_loss, new_grad = _loss_and_grad(
            same_tokenizer_topk_kl, student, teacher, **kwargs
        )
        old_loss, old_grad = _loss_and_grad(
            _old_direct_topk_kl, student, teacher, **kwargs
        )

        assert torch.isfinite(new_loss)
        assert torch.equal(new_loss, old_loss)
        assert torch.equal(new_grad, old_grad)

    @pytest.mark.parametrize("mask_kind", ["all_kd", "partial", "no_kd"])
    @pytest.mark.parametrize(
        "teacher_width", [_V, _V + _PAD], ids=["no_pad", "lm_head_pad"]
    )
    @pytest.mark.parametrize(
        "teacher_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
    )
    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    @pytest.mark.parametrize("reverse_kl", [False, True], ids=["fwd", "rev"])
    def test_full_vocab_bitwise_for_any_mask(
        self,
        reverse_kl: bool,
        temperature: float,
        teacher_dtype: torch.dtype,
        teacher_width: int,
        mask_kind: str,
    ):
        # No top-K choice is involved and the student has no lm_head padding,
        # so the full-vocab helper must match the previous math bitwise for any
        # mask.
        batch, seq = 3, 7
        generator = torch.Generator().manual_seed(2)
        student = torch.randn(batch, seq, _V, generator=generator)
        teacher = torch.randn(batch, seq, teacher_width, generator=generator) * 3
        teacher[..., _V:] = 25.0
        teacher = teacher.to(teacher_dtype)
        kd_mask, sample_mask = _kd_masks(mask_kind, batch, seq, generator)
        kwargs = dict(
            kd_mask=kd_mask,
            sample_mask=sample_mask,
            num_kd_tokens=_manual_kd_weights(kd_mask, sample_mask).sum(),
            temperature=temperature,
            reverse_kl=reverse_kl,
            tp_group=None,
            cp_group=None,
        )

        new_loss, new_grad = _loss_and_grad(
            same_tokenizer_full_vocab_kl, student, teacher, vocab_size=_V, **kwargs
        )
        old_loss, old_grad = _loss_and_grad(
            _old_direct_full_vocab_kl, student, teacher, **kwargs
        )

        assert torch.equal(new_loss, old_loss)
        assert torch.equal(new_grad, old_grad)


class TestSameTokenizerKdReduction:
    @staticmethod
    def _inputs(
        teacher_width: int = _V,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(4)
        student = torch.randn(3, 7, _V, generator=generator)
        teacher = torch.randn(3, 7, teacher_width, generator=generator) * 2
        kd_mask, sample_mask = _kd_masks("partial", 3, 7, generator)
        return student, teacher, kd_mask, sample_mask

    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    @pytest.mark.parametrize("reverse_kl", [False, True], ids=["fwd", "rev"])
    @pytest.mark.parametrize("path", ["topk", "full_vocab"])
    def test_matches_manual_reference(
        self, path: str, reverse_kl: bool, temperature: float
    ):
        student, teacher, kd_mask, sample_mask = self._inputs()
        weights = _manual_kd_weights(kd_mask, sample_mask)
        assert 0 < weights.sum() < weights.numel()
        # A DP/CP-global count larger than this rank's local KD rows.
        num_kd_tokens = weights.sum().item() + 5.0
        vocab_topk = 6
        columns = _manual_topk_columns(teacher, vocab_topk) if path == "topk" else None

        loss = _same_tokenizer_kd(
            path,
            student,
            teacher,
            vocab_topk=vocab_topk,
            kd_mask=kd_mask,
            sample_mask=sample_mask,
            num_kd_tokens=torch.tensor(num_kd_tokens),
            temperature=temperature,
            reverse_kl=reverse_kl,
            tp_group=None,
            cp_group=None,
        )
        expected = _manual_kd_loss(
            student,
            teacher,
            weights,
            num_kd_tokens,
            temperature=temperature,
            reverse_kl=reverse_kl,
            columns=columns,
        )

        torch.testing.assert_close(loss.double(), expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("path", ["topk", "full_vocab"])
    def test_doubling_num_kd_tokens_halves_the_loss(self, path: str):
        student, teacher, kd_mask, sample_mask = self._inputs()
        kwargs = dict(
            kd_mask=kd_mask,
            sample_mask=sample_mask,
            temperature=1.5,
            reverse_kl=False,
            tp_group=None,
            cp_group=None,
        )

        loss = _same_tokenizer_kd(
            path, student, teacher, num_kd_tokens=torch.tensor(9.0), **kwargs
        )
        loss_doubled = _same_tokenizer_kd(
            path, student, teacher, num_kd_tokens=torch.tensor(18.0), **kwargs
        )

        assert loss > 0
        torch.testing.assert_close(loss_doubled * 2, loss, rtol=1e-6, atol=0.0)

    @pytest.mark.parametrize("reverse_kl", [False, True], ids=["fwd", "rev"])
    def test_topk_drops_lm_head_padding(self, reverse_kl: bool):
        student, teacher, kd_mask, sample_mask = self._inputs(_V + _PAD)
        student = torch.cat([student, torch.full((3, 7, _PAD), 50.0)], dim=-1)
        # Padding columns would win every top-K slot if they were kept.
        teacher[..., _V:] = 50.0
        kwargs = dict(
            kd_mask=kd_mask,
            sample_mask=sample_mask,
            num_kd_tokens=_manual_kd_weights(kd_mask, sample_mask).sum(),
            vocab_size=_V,
            vocab_topk=4,
            temperature=1.0,
            reverse_kl=reverse_kl,
            tp_group=None,
            cp_group=None,
        )

        loss, grad = _loss_and_grad(same_tokenizer_topk_kl, student, teacher, **kwargs)
        unpadded_loss, unpadded_grad = _loss_and_grad(
            same_tokenizer_topk_kl,
            student[..., :_V].contiguous(),
            teacher[..., :_V].contiguous(),
            **kwargs,
        )

        assert torch.equal(loss, unpadded_loss)
        assert torch.equal(grad[..., :_V], unpadded_grad)
        assert not grad[..., _V:].any()

    @pytest.mark.parametrize("reverse_kl", [False, True], ids=["fwd", "rev"])
    def test_full_vocab_slices_teacher_to_student_width(self, reverse_kl: bool):
        student, teacher, kd_mask, sample_mask = self._inputs(_V + _PAD)
        teacher[..., _V:] = 50.0
        weights = _manual_kd_weights(kd_mask, sample_mask)
        kwargs = dict(
            kd_mask=kd_mask,
            sample_mask=sample_mask,
            num_kd_tokens=weights.sum(),
            vocab_size=_V,
            temperature=1.0,
            reverse_kl=reverse_kl,
            tp_group=None,
            cp_group=None,
        )

        loss, grad = _loss_and_grad(
            same_tokenizer_full_vocab_kl, student, teacher, **kwargs
        )
        unpadded_loss, unpadded_grad = _loss_and_grad(
            same_tokenizer_full_vocab_kl,
            student,
            teacher[..., :_V].contiguous(),
            **kwargs,
        )

        assert torch.equal(loss, unpadded_loss)
        assert torch.equal(grad, unpadded_grad)
        torch.testing.assert_close(
            loss.double(),
            _manual_kd_loss(
                student,
                teacher[..., :_V],
                weights,
                weights.sum().item(),
                temperature=1.0,
                reverse_kl=reverse_kl,
            ),
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("reverse_kl", [False, True], ids=["fwd", "rev"])
    def test_full_vocab_drops_student_lm_head_padding(self, reverse_kl: bool):
        student, teacher, kd_mask, sample_mask = self._inputs(_V + _PAD)
        # The student's lm_head pads past the shared vocab as well.
        student = torch.cat([student, torch.full((3, 7, _PAD), 5.0)], dim=-1)
        teacher[..., _V:] = 50.0
        kwargs = dict(
            kd_mask=kd_mask,
            sample_mask=sample_mask,
            num_kd_tokens=_manual_kd_weights(kd_mask, sample_mask).sum(),
            vocab_size=_V,
            temperature=1.0,
            reverse_kl=reverse_kl,
            tp_group=None,
            cp_group=None,
        )

        loss, grad = _loss_and_grad(
            same_tokenizer_full_vocab_kl, student, teacher, **kwargs
        )
        unpadded_loss, unpadded_grad = _loss_and_grad(
            same_tokenizer_full_vocab_kl,
            student[..., :_V].contiguous(),
            teacher[..., :_V].contiguous(),
            **kwargs,
        )

        torch.testing.assert_close(loss, unpadded_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(grad[..., :_V], unpadded_grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            grad[..., _V:], torch.zeros_like(grad[..., _V:]), rtol=0.0, atol=1e-6
        )
        # Same vocab support as the top-K helper when K covers the shared vocab.
        topk_loss = same_tokenizer_topk_kl(student, teacher, vocab_topk=_V, **kwargs)
        torch.testing.assert_close(loss, topk_loss, rtol=1e-5, atol=1e-6)
        # The previous math kept the student's padding columns in the softmax.
        old_kwargs = {
            key: value for key, value in kwargs.items() if key != "vocab_size"
        }
        old_loss = _old_direct_full_vocab_kl(student, teacher, **old_kwargs)
        assert not torch.allclose(old_loss, loss)


# ---------------------------------------------------------------------------
# CP=2 over gloo (CPU): the sequence is split contiguously across 2 ranks. Both
# ranks must choose the single-process top-K list, the CP-aware next-token shift
# must let a rank's last predictor read the next rank's first target, and the
# per-rank losses must sum to the single-process loss.
# ---------------------------------------------------------------------------

_CP2_SEQ = 8
_CP2_TOPK = 4
_CP2_RANK1_COLUMN = 20


def _cp2_case_inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global ``(student, teacher, kd_mask, sample_mask)``, identical on every rank."""
    generator = torch.Generator().manual_seed(6)
    student = torch.randn(2, _CP2_SEQ, _V, generator=generator)
    teacher = torch.randn(2, _CP2_SEQ, _V + _PAD, generator=generator) * 3
    teacher[..., _V:] = 50.0
    # Dominant on one row of rank 1's window only, so rank 0 can only choose it
    # through the CP all_reduce(MAX).
    teacher[1, _CP2_SEQ - 2, _CP2_RANK1_COLUMN] = 30.0
    if case == "kd_across_rank_boundary":
        # Sample 0's predictor 3 (last row on rank 0) supervises target 4,
        # which lives on rank 1.
        kd_mask = torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32
        )
    elif case == "one_rank_without_kd":
        # Every KD predictor sits on rank 0 (positions 0-3).
        kd_mask = torch.tensor(
            [[0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.float32
        )
    else:
        assert case == "no_kd_anywhere"
        kd_mask = torch.zeros(2, _CP2_SEQ)
    sample_mask = torch.ones(2)
    return student, teacher, kd_mask, sample_mask


def _check_same_tokenizer_cp2(
    rank: int, world_size: int, cp_group: torch.distributed.ProcessGroup
) -> None:
    local_seq = _CP2_SEQ // world_size
    window = slice(rank * local_seq, (rank + 1) * local_seq)
    for case in ("kd_across_rank_boundary", "one_rank_without_kd", "no_kd_anywhere"):
        student, teacher, kd_mask, sample_mask = _cp2_case_inputs(case)
        weights = _manual_kd_weights(kd_mask, sample_mask)

        # Same KD weights the helper builds on this rank.
        local_weights = cp_shift_next(
            kd_mask[:, window].contiguous(), cp_group, fill=0
        ).float() * sample_mask.unsqueeze(-1)
        assert torch.equal(local_weights, weights[:, window]), case

        chosen = select_teacher_topk_indices(
            teacher[:, window, :_V].contiguous(), _CP2_TOPK, cp_group=cp_group
        )
        single = select_teacher_topk_indices(teacher[..., :_V], _CP2_TOPK)
        assert torch.equal(chosen, single), (case, chosen, single)
        assert torch.equal(
            single, _manual_topk_columns(teacher[..., :_V], _CP2_TOPK)
        ), case
        peers = [torch.empty_like(chosen) for _ in range(world_size)]
        dist.all_gather(peers, chosen, group=cp_group)
        assert all(torch.equal(peer, chosen) for peer in peers), (case, peers)
        assert _CP2_RANK1_COLUMN in chosen.tolist(), (case, chosen)
        if case == "one_rank_without_kd":
            assert not weights[:, local_seq:].any()

        for path in ("topk", "full_vocab"):
            for reverse_kl in (False, True):
                kd_fn = functools.partial(
                    _same_tokenizer_kd, path, vocab_topk=_CP2_TOPK
                )
                common = dict(
                    sample_mask=sample_mask,
                    num_kd_tokens=weights.sum(),
                    temperature=1.5,
                    reverse_kl=reverse_kl,
                    tp_group=None,
                )
                single_loss, single_grad = _loss_and_grad(
                    kd_fn,
                    student,
                    teacher,
                    kd_mask=kd_mask,
                    cp_group=None,
                    **common,
                )
                local_loss, local_grad = _loss_and_grad(
                    kd_fn,
                    student[:, window].contiguous(),
                    teacher[:, window].contiguous(),
                    kd_mask=kd_mask[:, window].contiguous(),
                    cp_group=cp_group,
                    **common,
                )
                total = local_loss.reshape(1).clone()
                dist.all_reduce(total, op=dist.ReduceOp.SUM, group=cp_group)
                label = (case, path, reverse_kl, rank)
                torch.testing.assert_close(
                    total[0], single_loss, rtol=1e-6, atol=1e-7, msg=str(label)
                )
                torch.testing.assert_close(
                    local_grad,
                    single_grad[:, window],
                    rtol=1e-6,
                    atol=1e-7,
                    msg=str(label),
                )
                if case == "no_kd_anywhere" or (
                    case == "one_rank_without_kd" and rank == 1
                ):
                    assert local_loss.item() == 0.0, label
                    assert not local_grad.any(), label
                else:
                    assert local_loss.item() > 0.0, label


def _same_tokenizer_cp2_worker(
    rank: int, world_size: int, init_file: str, result_queue: Any
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        cp_group = dist.new_group(ranks=list(range(world_size)))
        _check_same_tokenizer_cp2(rank, world_size, cp_group)
        result_queue.put((rank, None))
    except BaseException:
        result_queue.put((rank, traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="gloo backend unavailable",
)
def test_same_tokenizer_kd_cp2_gloo_matches_single_process(tmp_path: Path):
    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_same_tokenizer_cp2_worker,
            args=(rank, world_size, str(tmp_path / "init"), result_queue),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    try:
        results = dict(result_queue.get(timeout=300) for _ in procs)
    finally:
        for proc in procs:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()

    failures = {rank: err for rank, err in results.items() if err is not None}
    assert not failures, "\n".join(
        f"rank {rank} failed:\n{err}" for rank, err in sorted(failures.items())
    )
    assert sorted(results) == list(range(world_size))
    assert [proc.exitcode for proc in procs] == [0] * world_size
