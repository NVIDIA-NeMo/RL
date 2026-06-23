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
    build_exact_token_map,
    chunk_average_log_probs,
    collect_overlapping_teacher_shards,
    get_sparse_projection_matrix,
    get_topk_projection,
    localize_alignment,
    parse_projection_file,
    valid_chunk_mask,
)
from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch
from nemo_rl.distributed.model_utils import AllReduceSum
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup

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
            y = AllReduceSum.apply(x, cp)
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
