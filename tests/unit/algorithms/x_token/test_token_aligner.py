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
"""Unit tests for the offset-based cross-tokenizer ``TokenAligner``.

The aligner pairs student and teacher tokens by the character spans they cover
in the shared source text. These tests exercise the single-sample kernel
(:func:`align_by_offsets_cluster`) and the batched :meth:`TokenAligner.align`
entry point with hand-built offsets, plus the invariants downstream loss code
relies on (chunk-id sentinels for padding, the 1-1 exact partition).
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from dataclasses import is_dataclass

import torch

from nemo_rl.algorithms.x_token import token_aligner as ta
from nemo_rl.algorithms.x_token.loss_utils import (
    chunk_average_log_probs,
    valid_chunk_mask,
)
from nemo_rl.algorithms.x_token.token_aligner import (
    AlignmentPair,
    TokenAligner,
    align_by_offsets_cluster,
)

# ---------------------------------------------------------------------------
# Test scaffolding — barebones tokenizer + aligner, no real HF / projection.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal HF stand-in: id -> token string plus special-token role attrs.

    The offset kernel only needs ``convert_ids_to_tokens`` (for the returned
    token strings) and the ``*_token_id`` / ``all_special_ids`` attributes
    used to classify ``(0, 0)``-offset positions by role.
    """

    def __init__(
        self,
        id_to_tok: dict[int, str],
        *,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = 0,
    ):
        self._id_to_tok = id_to_tok
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = None
        self.sep_token_id = None
        self.cls_token_id = None
        self.mask_token_id = None
        self.all_special_ids = [
            i for i in (bos_token_id, eos_token_id, pad_token_id) if i is not None
        ]

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_tok.get(int(i), f"<unk{i}>") for i in ids]


def _make_aligner(
    student_vocab: dict[int, str],
    teacher_vocab: dict[int, str],
    *,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = 0,
) -> TokenAligner:
    """Bypass projection loading and produce a TokenAligner for align()."""
    aligner = TokenAligner.__new__(TokenAligner)
    aligner.student_tokenizer = _FakeTokenizer(
        student_vocab,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    aligner.teacher_tokenizer = _FakeTokenizer(
        teacher_vocab,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return aligner


# ---------------------------------------------------------------------------
# AlignmentPair dataclass shape (call sites unpack these fields by name).
# ---------------------------------------------------------------------------


EXPECTED_ALIGNMENT_PAIR_FIELDS = (
    "s_tokens",
    "t_tokens",
    "s_start",
    "s_end",
    "t_start",
    "t_end",
    "is_correct",
)


def test_alignment_pair_dataclass_has_expected_fields():
    """``AlignmentPair`` pins the 7-field record ``_pairs_to_batch`` reads."""
    assert is_dataclass(AlignmentPair)
    field_names = tuple(f.name for f in dc_fields(AlignmentPair))
    assert set(field_names) == set(EXPECTED_ALIGNMENT_PAIR_FIELDS), field_names


# ---------------------------------------------------------------------------
# Single-sample offset kernel.
# ---------------------------------------------------------------------------


def test_kernel_one_to_one_identical_tokenization():
    """Same tokenization on both sides → one 1-1 content pair per token."""
    tok = _FakeTokenizer({1: "Hello", 2: "Ġworld"})
    ids = [1, 2]
    offs = [(0, 5), (5, 11)]
    pairs = align_by_offsets_cluster(ids, offs, tok, ids, offs, tok)

    assert len(pairs) == 2
    for s_toks, t_toks, s0, s1, t0, t1, ok in pairs:
        assert len(s_toks) == 1 and len(t_toks) == 1
        assert (s1 - s0) == 1 and (t1 - t0) == 1
        assert ok is True


def test_kernel_many_student_tokens_to_one_teacher_token():
    """Student splits a word the teacher keeps whole → one N-to-1 group whose
    char spans reconcile (student ``[0,2)+[2,5)`` covers teacher ``[0,5)``).
    """
    s_tok = _FakeTokenizer({1: "He", 2: "llo"})
    t_tok = _FakeTokenizer({3: "Hello"})
    pairs = align_by_offsets_cluster(
        [1, 2], [(0, 2), (2, 5)], s_tok, [3], [(0, 5)], t_tok
    )

    assert len(pairs) == 1
    s_toks, t_toks, s0, s1, t0, t1, ok = pairs[0]
    assert s_toks == ["He", "llo"] and t_toks == ["Hello"]
    assert (s0, s1, t0, t1) == (0, 2, 0, 1)
    assert ok is True


def test_kernel_coverage_divergence_emits_orphans():
    """When char coverage can't reconcile, each side is emitted as an orphan
    group (empty on the other side, ``is_correct=False``).
    """
    s_tok = _FakeTokenizer({1: "ab"})
    t_tok = _FakeTokenizer({2: "abc"})
    pairs = align_by_offsets_cluster([1], [(0, 2)], s_tok, [2], [(0, 3)], t_tok)

    assert len(pairs) == 2
    sidedness = {(bool(s), bool(t)) for s, t, *_ in pairs}
    assert sidedness == {(True, False), (False, True)}
    assert all(ok is False for *_, ok in pairs)


def test_kernel_leading_specials_paired_by_role():
    """Leading BOS tokens with different surface forms pair 1-1 by role."""
    s_tok = _FakeTokenizer({7: "<bos>", 1: "Hi"}, bos_token_id=7)
    t_tok = _FakeTokenizer({8: "<s>", 1: "Hi"}, bos_token_id=8)
    pairs = align_by_offsets_cluster(
        [7, 1], [(0, 0), (0, 2)], s_tok, [8, 1], [(0, 0), (0, 2)], t_tok
    )

    assert pairs[0][0] == ["<bos>"] and pairs[0][1] == ["<s>"]
    assert pairs[1][0] == ["Hi"] and pairs[1][1] == ["Hi"]


# ---------------------------------------------------------------------------
# Batched align(): shapes, padding sentinels, exact partition.
# ---------------------------------------------------------------------------


def test_align_end_to_end_shapes_are_consistent():
    """``align`` emits a fully-shaped :class:`AlignmentBatch` for trivial input."""
    vocab = {0: "<pad>", 1: "Hello", 2: "Ġworld", 3: "!"}
    aligner = _make_aligner(vocab, vocab, pad_token_id=0)
    student_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    teacher_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    offsets = torch.tensor([[[0, 5], [5, 11], [11, 12], [0, 0]]], dtype=torch.long)

    batch = aligner.align(
        student_ids,
        teacher_ids,
        student_offsets=offsets,
        teacher_offsets=offsets.clone(),
    )
    b, t_s = student_ids.shape
    _, t_t = teacher_ids.shape

    assert batch.student_chunk_id.shape == (b, t_s)
    assert batch.teacher_chunk_id.shape == (b, t_t)
    assert batch.pair_valid.dtype == torch.bool
    assert batch.num_chunks.shape == (b,)
    assert batch.num_chunks[0] <= batch.pair_valid.shape[1]


def test_align_exact_partition_marks_one_to_one_correct_pairs():
    """Identical 1-1 tokenization → every content position sits in the
    gold-loss exact partition on both sides.
    """
    vocab = {1: "Hello", 2: "Ġworld", 3: "!"}
    aligner = _make_aligner(vocab, vocab, pad_token_id=0)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    offsets = torch.tensor([[[0, 5], [5, 11], [11, 12]]], dtype=torch.long)

    batch = aligner.align(
        ids, ids.clone(), student_offsets=offsets, teacher_offsets=offsets.clone()
    )

    assert batch.student_exact_partition_mask[0].tolist() == [True, True, True]
    assert batch.teacher_exact_partition_mask[0].tolist() == [True, True, True]
    assert (batch.student_chunk_id[0] != -1).all()
    assert (batch.teacher_chunk_id[0] != -1).all()


def test_align_padding_positions_receive_sentinel_chunk_id():
    """Tail-padded positions must keep ``chunk_id == -1`` once the attention
    mask is supplied; real content is untouched, and no surviving chunk draws
    only from padding.
    """
    vocab = {0: "<pad>", 1: "Hello", 2: "Ġworld", 3: "!"}
    aligner = _make_aligner(vocab, vocab, pad_token_id=0)

    # Sample 0: 3 content + 3 pad. Sample 1: 6 content (no padding).
    student_ids = torch.tensor(
        [[1, 2, 3, 0, 0, 0], [1, 2, 3, 2, 3, 1]], dtype=torch.long
    )
    teacher_ids = student_ids.clone()
    content_offsets = [[0, 5], [5, 11], [11, 12], [12, 18], [18, 19], [19, 24]]
    pad_offsets = [[0, 5], [5, 11], [11, 12], [0, 0], [0, 0], [0, 0]]
    offsets = torch.tensor([pad_offsets, content_offsets], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
    )

    masked = aligner.align(
        student_ids,
        teacher_ids,
        student_offsets=offsets,
        teacher_offsets=offsets.clone(),
        student_attention_mask=attention_mask,
        teacher_attention_mask=attention_mask,
    )

    # Pad positions of sample 0 are not-in-any-chunk on both sides.
    assert (masked.student_chunk_id[0, -3:] == -1).all(), masked.student_chunk_id[0]
    assert (masked.teacher_chunk_id[0, -3:] == -1).all(), masked.teacher_chunk_id[0]
    assert not masked.student_exact_partition_mask[0, -3:].any()
    # Real content is untouched: sample 0's first 3 positions keep a chunk,
    # and the fully-real sample 1 is all-chunked.
    assert (masked.student_chunk_id[0, :3] != -1).all()
    assert (masked.student_chunk_id[1] != -1).all()

    # Downstream gate: no surviving chunk may draw only from padding.
    max_chunks = int(masked.pair_valid.shape[1])
    dummy = torch.zeros((2, student_ids.shape[1], 1))
    _, s_sizes = chunk_average_log_probs(dummy, masked.student_chunk_id, max_chunks)
    _, t_sizes = chunk_average_log_probs(dummy, masked.teacher_chunk_id, max_chunks)
    chunk_mask = valid_chunk_mask(s_sizes, t_sizes, masked.pair_valid)
    assert chunk_mask.sum() > 0
    assert bool((s_sizes[chunk_mask] > 0).all())
    assert bool((t_sizes[chunk_mask] > 0).all())


def test_module_exposes_alignment_kernel():
    """The offset kernel is reachable from the package namespace (callers use
    it directly instead of poking at name-mangled internals).
    """
    assert hasattr(ta, "align_by_offsets_cluster")
