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
"""PR #2508 review-fix verifications for ``TokenAligner`` and friends."""

from __future__ import annotations

import inspect
from dataclasses import is_dataclass
from typing import List

import pytest
import torch

from nemo_rl.algorithms.x_token import token_aligner as ta
from nemo_rl.algorithms.x_token.loss_utils import (
    chunk_average_log_probs,
    valid_chunk_mask,
)
from nemo_rl.algorithms.x_token.token_aligner import (
    TokenAligner,
    _canonicalize_sequence,
)

# ---------------------------------------------------------------------------
# Test scaffolding — barebones aligner without real tokenizers / projection.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Map id -> token string via a precomputed dict; minimal HF stand-in."""

    def __init__(self, id_to_tok: dict[int, str]):
        self._id_to_tok = id_to_tok

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_tok.get(int(i), f"<unk{i}>") for i in ids]


def _make_aligner(
    student_vocab: dict[int, str], teacher_vocab: dict[int, str]
) -> TokenAligner:
    """Bypass projection loading and produce a TokenAligner suitable for align()."""
    aligner = TokenAligner.__new__(TokenAligner)
    aligner.student_tokenizer = _FakeTokenizer(student_vocab)
    aligner.teacher_tokenizer = _FakeTokenizer(teacher_vocab)
    aligner.max_combination_len = 4
    return aligner


# ---------------------------------------------------------------------------
# T1 — TokenAligner-coupled helpers should live on the class (or be public).
# ---------------------------------------------------------------------------


def test_T1_align_dp_is_addressable_from_token_aligner_namespace():
    """Class B. The DP kernel and canonicalize-sequence helpers should be
    reachable from the public alignment surface — either as
    ``TokenAligner`` methods or as documented module-level helpers — not
    accessed by name-mangling from outside callers.
    """
    # The plan acknowledges these may remain module-level. The test asserts
    # the contract that a caller can refer to them via the package without
    # poking at private leading-underscore module attributes — i.e., they
    # must show up either as TokenAligner attributes OR exported from the
    # package's __init__. This catches a refactor that hides them entirely.
    has_on_class = hasattr(TokenAligner, "_align_dp") or hasattr(
        TokenAligner, "align_dp"
    )
    has_at_module = hasattr(ta, "_align_dp") or hasattr(ta, "align_dp")
    assert has_on_class or has_at_module, (
        "Neither TokenAligner nor the token_aligner module exposes the "
        "DP kernel; the helper-scope refactor has hidden it."
    )


# ---------------------------------------------------------------------------
# T2 — canonicalization should preserve sequence length OR the consumer must
# track the length change. The chunk_id math indexes into the ORIGINAL token
# axis, so a shrinking canonicalization silently misaligns spans.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq",
    [
        ["Hello", "Ġworld", "!"],
        ["abc", "def", "ghi", "jkl"],
        ["The", "Ġquick", "Ġbrown", "Ġfox"],
        ["<bos>", "Hello"],
        ["a", "b", "c"],
    ],
    ids=["short", "ascii", "spaced", "with-bos", "single-chars"],
)
def test_T2_canonicalize_sequence_preserves_length_for_clean_inputs(
    seq: List[str],
):
    """Class A. For inputs that don't contain multi-token mojibake patterns,
    canonicalization MUST preserve length AND emit an identity index map
    so the alignment spans (computed over canonical tokens) trivially
    round-trip back to the original token axis.
    """
    canon, ranges = _canonicalize_sequence(seq)
    assert len(canon) == len(seq), (
        f"_canonicalize_sequence shrunk a clean input: "
        f"{len(seq)} -> {len(canon)} (seq={seq}, canon={canon})"
    )
    assert ranges == [(i, i + 1) for i in range(len(seq))], (
        f"canon_to_orig map is not the identity for a clean input: {ranges}"
    )


def test_T2_canonicalization_artifact_merge_emits_remappable_ranges():
    """Class A. The multi-token-artifact and consecutive-byte merges
    SHRINK the canonical sequence. The fix is to expose a parallel
    ``canon_to_orig`` map so DP-output indices over the canonical tokens
    can be remapped to the original token axis before chunk-id writes.
    This test pins the shrink-and-remap shape.
    """
    # The first entry in _MULTI_TOKEN_ARTIFACT_FIXES collapses two tokens
    # into one: ["ĠâĪ", "ĳ"] -> ["Ġ∑"].
    seq = ["ĠâĪ", "ĳ"]
    canon, ranges = _canonicalize_sequence(seq)
    assert len(canon) == 1, f"expected merge to length 1, got {canon!r}"
    assert ranges == [(0, 2)], (
        f"merged canonical token should cover the original [0, 2) range; got {ranges}"
    )


# ---------------------------------------------------------------------------
# T3 — padding tokens must NOT receive a valid chunk_id.
# ---------------------------------------------------------------------------


def test_T3_padding_tokens_receive_sentinel_chunk_id():
    """Class A (RayenTian review). Tail-padded positions must keep
    ``chunk_id == -1`` once the attention mask is supplied.

    ``align`` runs the DP over the fully padded id tensors, so the pad run
    on each side aligns against itself (``<pad>`` canonicalizes to ``""`` on
    both sides) and inherits a chunk index — which then survives
    :func:`valid_chunk_mask` and leaks into the loss. The fix threads the
    attention masks into ``align`` and resets ``chunk_id`` /
    ``exact_partition_mask`` at padded positions. Pad-token-id filtering is
    *not* sufficient: the collator falls back to ``pad_token = eos_token``,
    so a real end-of-doc ``eos`` (mid-sequence under packing) would be
    dropped too; only the attention mask separates pad-eos from real-eos.
    """
    student_vocab = {1: "Hello", 2: "Ġworld", 3: "!", 0: "<pad>"}
    teacher_vocab = {1: "Hello", 2: "Ġworld", 3: "!", 0: "<pad>"}
    aligner = _make_aligner(student_vocab, teacher_vocab)

    # Sample 0: 3 content + 3 pad. Sample 1: 6 content (no padding).
    student_ids = torch.tensor(
        [[1, 2, 3, 0, 0, 0], [1, 2, 3, 2, 3, 1]], dtype=torch.long
    )
    teacher_ids = student_ids.clone()
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
    )

    # Without masks the leak is present: padding gets a real chunk_id. This
    # asserts the fixture actually exercises the bug being fixed.
    unmasked = aligner.align(student_ids, teacher_ids)
    assert (unmasked.student_chunk_id[0, -3:] != -1).any(), (
        "expected the unmasked path to assign padded positions a chunk_id; "
        "if this no longer holds the fixture no longer exercises the leak"
    )

    masked = aligner.align(
        student_ids,
        teacher_ids,
        student_attention_mask=attention_mask,
        teacher_attention_mask=attention_mask,
    )
    # Pad positions of sample 0 are not-in-any-chunk on both sides.
    assert (masked.student_chunk_id[0, -3:] == -1).all(), masked.student_chunk_id[
        0
    ].tolist()
    assert (masked.teacher_chunk_id[0, -3:] == -1).all(), masked.teacher_chunk_id[
        0
    ].tolist()
    assert not masked.student_exact_partition_mask[0, -3:].any()
    # Real content is untouched: sample 0's first 3 positions keep their
    # alignment, and the fully-real sample 1 is unchanged.
    assert (masked.student_chunk_id[0, :3] != -1).all()
    assert torch.equal(masked.student_chunk_id[1], unmasked.student_chunk_id[1])

    # Downstream gate: no surviving chunk may draw only from padding.
    max_chunks = int(masked.pair_valid.shape[1])
    dummy = torch.zeros((2, student_ids.shape[1], 1))
    _, s_sizes = chunk_average_log_probs(dummy, masked.student_chunk_id, max_chunks)
    _, t_sizes = chunk_average_log_probs(dummy, masked.teacher_chunk_id, max_chunks)
    chunk_mask = valid_chunk_mask(s_sizes, t_sizes, masked.pair_valid)
    assert chunk_mask.sum() > 0
    assert bool((s_sizes[chunk_mask] > 0).all())
    assert bool((t_sizes[chunk_mask] > 0).all())


# ---------------------------------------------------------------------------
# T4 — ``_align_with_anchors`` should accept explicit kwargs.
# ---------------------------------------------------------------------------


def test_T4_align_with_anchors_signature_is_explicit():
    """Class B. After the fix, ``_align_with_anchors`` must not accept
    ``**kwargs`` — its scoring options should be enumerated so the call
    surface is grep-able and silent typos at call sites raise.
    """
    sig = inspect.signature(TokenAligner._align_with_anchors)
    var_keyword_params = [
        p for p in sig.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD
    ]
    assert not var_keyword_params, (
        f"_align_with_anchors still accepts **kwargs "
        f"(params: {[p.name for p in var_keyword_params]}). The reviewer "
        "asked for explicit parameters."
    )


def test_T4_align_dp_signature_is_explicit():
    """Class B. Same invariant on ``_align_dp`` — both functions are paired
    in the kwargs forwarding chain.
    """
    sig = inspect.signature(TokenAligner._align_dp)
    var_keyword_params = [
        p for p in sig.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD
    ]
    assert not var_keyword_params, (
        f"_align_dp still accepts **kwargs "
        f"(params: {[p.name for p in var_keyword_params]})."
    )


# ---------------------------------------------------------------------------
# T5 — ``AlignmentPair`` dataclass for tuple-unpack call sites.
# ---------------------------------------------------------------------------


# Field name set the pair tuple is unpacked into at token_aligner.py:302 — the
# fix should introduce a dataclass with these exact field names.
EXPECTED_ALIGNMENT_PAIR_FIELDS = (
    "s_tokens",
    "t_tokens",
    "s_start",
    "s_end",
    "t_start",
    "t_end",
    "is_correct",
)


def test_T5_alignment_pair_dataclass_exists_with_expected_fields():
    """Class B. The tuple ``(s_tokens, t_tokens, s_start, s_end, t_start,
    t_end, is_correct)`` returned by ``_align_single`` is now an
    :class:`AlignmentPair` dataclass so call sites stop unpacking
    positionally. This test pins the dataclass's field set.
    """
    pair_cls = getattr(ta, "AlignmentPair", None)
    assert pair_cls is not None, (
        "AlignmentPair dataclass missing from nemo_rl.algorithms.x_token."
        "token_aligner; the reviewer requested a named record to replace "
        "the 7-tuple at the _pairs_to_batch unpack site."
    )
    assert is_dataclass(pair_cls)
    from dataclasses import fields as dc_fields

    field_names = tuple(f.name for f in dc_fields(pair_cls))
    assert set(field_names) == set(EXPECTED_ALIGNMENT_PAIR_FIELDS), (
        f"AlignmentPair fields drift from the tuple shape: "
        f"got {field_names}, expected {EXPECTED_ALIGNMENT_PAIR_FIELDS}"
    )


# ---------------------------------------------------------------------------
# T6 — alignment spans must index the ORIGINAL token axis even when
# canonicalization shrinks the sequence (multi-token artifact merge AND
# consecutive-byte merge).
# ---------------------------------------------------------------------------


def test_T6_artifact_merge_remaps_spans_to_original_axis():
    """Class A. Student tokens ``["Hello", "ĠâĪ", "ĳ", "."]`` canonicalize
    to ``["Hello", "Ġ∑", "."]`` via the first ``_MULTI_TOKEN_ARTIFACT_FIXES``
    entry. After ``align``, the two merged original-axis positions (1, 2)
    must share a chunk_id, and that chunk_id must match the teacher-side
    position of ``Ġ∑``. Without the canonical→original remap, DP indices
    would land in canonical coordinates and the chunk_id at original
    position 3 would be ``-1`` (and position 2 would carry the ``.``
    chunk_id from canonical position 2).
    """
    student_vocab = {1: "Hello", 2: "ĠâĪ", 3: "ĳ", 4: "."}
    teacher_vocab = {1: "Hello", 2: "Ġ∑", 3: "."}
    aligner = _make_aligner(student_vocab, teacher_vocab)

    student_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)  # T_s = 4
    teacher_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)  # T_t = 3

    batch = aligner.align(student_ids, teacher_ids)

    assert batch.student_chunk_id.shape == (1, 4)
    assert batch.teacher_chunk_id.shape == (1, 3)

    student_ids_row = batch.student_chunk_id[0].tolist()
    teacher_ids_row = batch.teacher_chunk_id[0].tolist()

    # Every original position belongs to some chunk.
    assert all(cid != -1 for cid in student_ids_row), student_ids_row
    assert all(cid != -1 for cid in teacher_ids_row), teacher_ids_row

    # Original-axis positions 1 and 2 (the merged pair) share a chunk_id,
    # and it matches the teacher-side ``Ġ∑`` chunk.
    assert student_ids_row[1] == student_ids_row[2], student_ids_row
    assert student_ids_row[1] == teacher_ids_row[1], (
        student_ids_row,
        teacher_ids_row,
    )
    # The flanking "Hello" / "." pairs are intact and match across sides.
    assert student_ids_row[0] == teacher_ids_row[0]
    assert student_ids_row[3] == teacher_ids_row[2]

    # Chunk-id values stay inside the per-sample chunk count.
    n = int(batch.num_chunks[0])
    assert all(0 <= cid < n for cid in student_ids_row + teacher_ids_row)

    # The merged pair is correct-by-canonical-text but is NOT a single-
    # original-token-to-single-original-token pair, so the gold-loss
    # exact-partition mask must be False at original positions 1, 2 on
    # the student side (and at teacher position 1). The 1-1 flanks stay
    # in the partition.
    s_part = batch.student_exact_partition_mask[0].tolist()
    t_part = batch.teacher_exact_partition_mask[0].tolist()
    assert s_part == [True, False, False, True], s_part
    assert t_part == [True, False, True], t_part


def test_T6_byte_fallback_merge_remaps_spans_to_original_axis():
    """Class A. Latin-1 byte tokens ``["Ã", "©"]`` decode to ``"é"`` via
    ``_try_merge_byte_buffer`` (ord 233 > 127, 2-byte UTF-8). The same
    shared-chunk_id invariant must hold for this code path.
    """
    student_vocab = {1: "Ã", 2: "©", 3: "中"}
    teacher_vocab = {1: "é", 2: "中"}
    aligner = _make_aligner(student_vocab, teacher_vocab)

    student_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)  # T_s = 3
    teacher_ids = torch.tensor([[1, 2]], dtype=torch.long)  # T_t = 2

    batch = aligner.align(student_ids, teacher_ids)

    assert batch.student_chunk_id.shape == (1, 3)
    assert batch.teacher_chunk_id.shape == (1, 2)

    s_row = batch.student_chunk_id[0].tolist()
    t_row = batch.teacher_chunk_id[0].tolist()

    assert all(cid != -1 for cid in s_row), s_row
    assert all(cid != -1 for cid in t_row), t_row

    # Original positions 0 and 1 (the two byte tokens) share a chunk_id
    # that matches the teacher ``é`` chunk.
    assert s_row[0] == s_row[1], s_row
    assert s_row[0] == t_row[0], (s_row, t_row)
    # The trailing ``中`` aligns 1-1.
    assert s_row[2] == t_row[1]


# ---------------------------------------------------------------------------
# Bonus regression: ``align()`` produces well-shaped tensors end-to-end.
# Guards against shape regressions from any of the fixes above.
# ---------------------------------------------------------------------------


def test_align_end_to_end_shapes_are_consistent():
    """Regression: ``align`` must always emit a fully-shaped
    :class:`AlignmentBatch` even for trivial inputs.
    """
    vocab = {0: "<pad>", 1: "Hello", 2: "Ġworld", 3: "!", 4: "Ġfoo"}
    aligner = _make_aligner(vocab, vocab)
    student_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    teacher_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)

    batch = aligner.align(student_ids, teacher_ids)
    B, T_s = student_ids.shape
    _, T_t = teacher_ids.shape

    assert batch.student_chunk_id.shape == (B, T_s)
    assert batch.teacher_chunk_id.shape == (B, T_t)
    assert batch.pair_valid.dtype == torch.bool
    assert batch.num_chunks.shape == (B,)
    assert batch.num_chunks[0] <= batch.pair_valid.shape[1]
