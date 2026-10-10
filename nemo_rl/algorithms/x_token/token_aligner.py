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
"""Cross-tokenizer token alignment.

The student and teacher tokenize the same source text; this module pairs their
tokens by the character spans each token covers (``return_offsets_mapping=True``
on a fast HF tokenizer). Consecutive tokens that share an exact
``(char_start, char_end)`` collapse into a cluster, and a strict char-end
walker pairs student and teacher clusters covering the same character range.
Special tokens (BOS / EOS / pad, offset ``(0, 0)``) are paired separately by
role. The result feeds the cross-tokenizer distillation loss.

Public surface:
    - :class:`AlignmentPair` — per-pair record produced by the alignment;
      replaces the loose ``(s_tokens, t_tokens, s_start, s_end, t_start,
      t_end, is_correct)`` tuples that the helpers used to pass around.
    - :class:`AlignmentBatch` — dense-padded per-batch alignment payload that
      covers all three loss modes (P-KL, gold_loss, xtoken_loss).
    - :class:`TokenAligner` — owns the two tokenizers and the projection
      matrix, exposes :meth:`align` and :meth:`align_chat` for the collator.
    - :func:`align_by_offsets_cluster` — the single-sample offset alignment
      kernel, also usable directly.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Collection, List, Tuple

import torch

# Visual byte representations used by some BPE tokenizers (especially for
# emojis / non-ASCII bytes). These constants are content-coupled to the
# tokenizers we align across.
VISUAL_BYTE_MAP = {
    "ð": 240,
    "Ɩ": 241,
    "Ɨ": 242,
    "Ƙ": 243,
    "ƙ": 244,
    "ƚ": 245,
    "ƛ": 246,
    "Ɯ": 247,
    "Ɲ": 248,
    "ƞ": 249,
    "Ɵ": 250,
    "Ơ": 251,
    "ơ": 252,
    "Ƣ": 253,
    "ƣ": 254,
    "Ƥ": 255,
    "Ł": 156,
    "ł": 157,
    "Ń": 158,
    "ń": 159,
    "ĺ": 149,
    "Ļ": 150,
    "ļ": 151,
    "Ľ": 152,
    "ľ": 153,
    "Ŀ": 154,
    "ŀ": 155,
    "Ĭ": 135,
    "ĭ": 136,
    "Į": 137,
    "į": 138,
    "İ": 139,
    "ı": 140,
    "Ĳ": 141,
    "ĳ": 142,
    "Ĵ": 143,
    "ĵ": 144,
    "Ķ": 145,
    "ķ": 146,
    "ĸ": 147,
    "Ĺ": 148,
    "ĥ": 128,
    "Ħ": 129,
    "ħ": 130,
    "Ĩ": 131,
    "ĩ": 132,
    "Ī": 133,
    "ī": 134,
    "Ģ": 162,
    "ģ": 163,
    "Ĝ": 28,
    "ĝ": 29,
    "Ğ": 30,
    "ğ": 31,
}

# Multi-token encoding artifacts (mojibake patterns) where the broken byte
# sequence spans tokens. Patterns are checked left-to-right with the first
# match wins. Trimmed to the high-frequency entries.
_MULTI_TOKEN_ARTIFACT_FIXES: list[tuple[list[str], list[str]]] = [
    (["ĠâĪ", "ĳ"], ["Ġ∑"]),
    (["âĪ", "ĳ"], ["∑"]),
    (["ĠâĪ", "ı"], ["Ġ∏"]),
    (["âĪ", "ı"], ["∏"]),
    (["ĠâĪ", "Ĥ"], ["Ġ∂"]),
    (["âĪ", "Ĥ"], ["∂"]),
    (["ĠâĪ", "ĩ"], ["Ġ∇"]),
    (["âĪ", "ĩ"], ["∇"]),
    (["ĠâĪ", "ŀ"], ["Ġ∞"]),
    (["âĪ", "ŀ"], ["∞"]),
    (["ĠâĪ", "ļ"], ["Ġ√"]),
    (["âĪ", "ļ"], ["√"]),
    (["ĠâĪ", "«"], ["Ġ∫"]),
    (["âĪ", "«"], ["∫"]),
    (["Ġâī", "ł"], ["Ġ≠"]),
    (["âī", "ł"], ["≠"]),
    (["Ġä¸", "Ń"], ["Ġ中"]),
    (["ä¸", "Ń"], ["中"]),
    (["æĸ", "ĩ"], ["文"]),
    (["Ġæĸ", "ĩ"], ["Ġ文"]),
]
_MULTI_TOKEN_ARTIFACT_FIXES_BY_FIRST: dict[str, list[tuple[list[str], list[str]]]] = {
    first: [
        (pattern, replacement)
        for pattern, replacement in _MULTI_TOKEN_ARTIFACT_FIXES
        if pattern[0] == first
    ]
    for first in {pattern[0] for pattern, _ in _MULTI_TOKEN_ARTIFACT_FIXES}
}

_UNICODE_FIXES = {
    "Ã±": "ñ",
    "Ã¡": "á",
    "Ã©": "é",
    "Ã­": "í",
    "Ã³": "ó",
    "Ãº": "ú",
    "Ã": "À",
    "Ã¢": "â",
    "Ã§": "ç",
    "Ã¨": "è",
    "Ã«": "ë",
    "Ã®": "î",
    "Ã´": "ô",
    "Ã¹": "ù",
    "Ã»": "û",
    "Ã¿": "ÿ",
    "ä¸Ń": "中",
    "æĸĩ": "文",
    "æĹ¥æľ¬": "日本",
    "èªŀ": "語",
    "ÐłÑĥÑģ": "Рус",
    "ÑģÐºÐ¸Ð¹": "ский",
    "Ø§ÙĦØ¹Ø±Ø¨ÙĬØ©": "العربية",
    "à¤¹": "ह",
    "à¤¿à¤Ĥ": "हिं",
    "à¤¦à¥Ģ": "दी",
    "âĪĳ": "∑",
    "âĪı": "∏",
    "âĪĤ": "∂",
    "âĪĩ": "∇",
    "âĪŀ": "∞",
    "âĪļ": "√",
    "âĪ«": "∫",
    "âīĪ": "≈",
    "âīł": "≠",
    "âī¤": "≤",
    "âī¥": "≥",
}

_SPECIAL_TOKEN_MAP = {
    "<|begin_of_text|>": "<bos>",
    "<bos>": "<bos>",
    "<pad>": "",
}


@dataclass
class AlignmentPair:
    """One aligned span between student and teacher token sequences.

    The alignment builds these as it walks the two token sequences;
    the shared matcher fills in ``is_correct`` from strict decoded-text
    comparison. Insertions/deletions (orphan groups) use ``-1`` for the empty
    side's start/end indices.

    Attributes:
        s_tokens: Student tokens covered by this pair.
        t_tokens: Teacher tokens covered by this pair.
        s_start: Inclusive start index into the student token sequence
            (``-1`` for teacher-only insertions).
        s_end: Exclusive end index into the student token sequence
            (``-1`` for teacher-only insertions).
        t_start: Inclusive start index into the teacher token sequence
            (``-1`` for student-only insertions).
        t_end: Exclusive end index into the teacher token sequence
            (``-1`` for student-only insertions).
        is_correct: ``True`` when the decoded student span text
            matches the decoded teacher span text. Defaults to
            ``False`` so the aligner can build pairs before computing the
            mask.
    """

    s_tokens: List[str]
    t_tokens: List[str]
    s_start: int
    s_end: int
    t_start: int
    t_end: int
    is_correct: bool = False


@dataclass(frozen=True, kw_only=True)
class NativeAlignmentPart:
    """One source-validated answer piece on a model's rendered chat surface.

    Differently spelled Boolean/null values may retain CE without exact-text KD.
    """

    name: str
    span: tuple[int, int]
    text: str
    allow_native_difference: bool = False


@dataclass(frozen=True, kw_only=True)
class NativeAlignmentRegions:
    """Selected semantic character spans in one native assistant turn."""

    reasoning: tuple[int, int] | None = None
    close: tuple[int, int] | None = None
    answer: tuple[int, int] | None = None

    def get(self, name: str) -> tuple[int, int] | None:
        """Return the named semantic span, rejecting unknown region names."""
        if name == "reasoning":
            return self.reasoning
        if name == "close":
            return self.close
        if name == "answer":
            return self.answer
        raise ValueError(f"Unsupported native region: {name!r}")


@dataclass
class AlignmentBatch:
    """Per-batch alignment payload covering all three loss modes.

    The collator hands this dataclass directly to the loss fn alongside the
    tokenized batch. Tensors are dense-padded to the batch maximum so DTensor
    V2 can shard on dim 0 without knowing about cross-tokenizer specifics.

    Attributes:
        pair_valid: ``[B, max_pairs]`` bool. False on padding entries.
        pair_is_correct: ``[B, max_pairs]`` bool. True when decoded student
            and teacher text match, or native source validation succeeds.
        student_chunk_id: ``[B, T_s]`` long. Chunk index (= pair index) the
            student token belongs to; ``-1`` if not in any chunk
            (insertion-only pair on student side).
        teacher_chunk_id: ``[B, T_t]`` long. Counterpart.
    """

    pair_valid: torch.Tensor
    pair_is_correct: torch.Tensor
    student_chunk_id: torch.Tensor
    teacher_chunk_id: torch.Tensor


class TokenAligner:
    """Aligns student and teacher tokenizations of the same source text.

    Alignment is offset-based: each token carries the ``(char_start,
    char_end)`` span it covers in the shared source text
    (``return_offsets_mapping=True``). Consecutive tokens with the same span
    collapse into a cluster, a strict char-end walker pairs student and
    teacher clusters covering the same character range, and special tokens
    (offset ``(0, 0)``) are paired by role.

    Args:
        student_tokenizer: HF tokenizer for the student model. Must be a fast
            tokenizer so the collator can emit ``offset_mapping``.
        teacher_tokenizer: HF tokenizer for the teacher model.
        projection_matrix_path: Path retained on the aligner for downstream
            callers (e.g. the loss fn) that materialize the projection on
            their training device via
            :func:`nemo_rl.algorithms.x_token.loss_utils.get_sparse_projection_matrix`
            or :func:`nemo_rl.algorithms.x_token.loss_utils.get_topk_projection`.
    """

    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        projection_matrix_path: str,
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.projection_matrix_path = projection_matrix_path

    def align(
        self,
        student_ids: torch.Tensor,
        teacher_ids: torch.Tensor,
        *,
        student_offsets: torch.Tensor,
        teacher_offsets: torch.Tensor,
        student_attention_mask: torch.Tensor | None = None,
        teacher_attention_mask: torch.Tensor | None = None,
    ) -> AlignmentBatch:
        """Align a batch of student/teacher token id tensors by char offsets.

        Args:
            student_ids: ``[B, T_s]`` long tensor.
            teacher_ids: ``[B, T_t]`` long tensor.
            student_offsets: ``[B, T_s, 2]`` long tensor of ``(char_start,
                char_end)`` per token, as produced by a fast tokenizer with
                ``return_offsets_mapping=True``. Special/padding tokens carry
                ``(0, 0)``.
            teacher_offsets: ``[B, T_t, 2]`` counterpart.
            student_attention_mask: optional ``[B, T_s]`` mask (1 = real
                token, 0 = padding). When given, padded positions are forced
                to ``chunk_id = -1`` so tokenizer padding never forms a
                valid chunk.
            teacher_attention_mask: optional ``[B, T_t]`` counterpart.

        Returns:
            An :class:`AlignmentBatch` with all fields populated for the
            three loss modes.
        """
        assert student_ids.dim() == 2 and teacher_ids.dim() == 2
        assert student_ids.shape[0] == teacher_ids.shape[0], (
            f"student/teacher batch size mismatch: "
            f"{student_ids.shape[0]} vs {teacher_ids.shape[0]}"
        )
        b, t_s = student_ids.shape
        _, t_t = teacher_ids.shape

        per_sample_pairs: List[List[AlignmentPair]] = []
        for i in range(b):
            pairs = self._align_single(
                student_ids[i].tolist(),
                teacher_ids[i].tolist(),
                student_offsets[i].tolist(),
                teacher_offsets[i].tolist(),
            )
            per_sample_pairs.append(pairs)

        batch = self._pairs_to_batch(per_sample_pairs, b=b, t_s=t_s, t_t=t_t)
        self._drop_padding(
            batch,
            student_attention_mask=student_attention_mask,
            teacher_attention_mask=teacher_attention_mask,
        )
        return batch

    def align_chat(
        self,
        student_ids: torch.Tensor,
        teacher_ids: torch.Tensor,
        *,
        student_offsets: torch.Tensor,
        teacher_offsets: torch.Tensor,
        student_asst_char_spans: list[list[tuple[int, int]]],
        teacher_asst_char_spans: list[list[tuple[int, int]]],
        student_attention_mask: torch.Tensor | None = None,
        teacher_attention_mask: torch.Tensor | None = None,
        student_asst_mask: torch.Tensor | None = None,
        teacher_asst_mask: torch.Tensor | None = None,
        student_eot_indices: list[list[int]] | None = None,
        teacher_eot_indices: list[list[int]] | None = None,
        student_alignment_regions: list[list[NativeAlignmentRegions]] | None = None,
        teacher_alignment_regions: list[list[NativeAlignmentRegions]] | None = None,
        student_answer_parts: list[list[list[NativeAlignmentPart] | None]]
        | None = None,
        teacher_answer_parts: list[list[list[NativeAlignmentPart] | None]]
        | None = None,
        student_rendered_texts: list[str] | None = None,
        teacher_rendered_texts: list[str] | None = None,
        drop_first_content_pair: bool = False,
        included_regions: Collection[str] | None = None,
    ) -> AlignmentBatch:
        """Align assistant turns in a batch of independently rendered chats.

        Args:
            student_ids: Padded ``[B, T_s]`` student token IDs.
            teacher_ids: Padded ``[B, T_t]`` teacher token IDs.
            student_offsets: ``[B, T_s, 2]`` offsets in each student render.
            teacher_offsets: ``[B, T_t, 2]`` offsets in each teacher render.
            student_asst_char_spans: Per-sample assistant content spans in the
                student renders. Turns must correspond to the teacher spans.
            teacher_asst_char_spans: Counterpart in the teacher renders.
            student_attention_mask: Optional ``[B, T_s]`` real-token mask.
                Padding is excluded from the returned chunk IDs.
            teacher_attention_mask: Optional ``[B, T_t]`` counterpart.
            student_asst_mask: Optional ``[B, T_s]`` mask selecting tokens
                within assistant content regions.
            teacher_asst_mask: Optional ``[B, T_t]`` counterpart.
            student_eot_indices: Optional per-sample, per-turn EOT positions.
                Use ``-1`` when truncation removed a turn's terminator. If
                omitted, locate EOT by the end of the content span.
            teacher_eot_indices: Counterpart for the teacher tokens.
            student_alignment_regions: Optional per-sample, per-turn named
                reasoning, close, and answer character spans.
            teacher_alignment_regions: Counterpart for the teacher renders.
            student_answer_parts: Optional per-sample, per-turn structured
                answer pieces with original text and character spans.
            teacher_answer_parts: Counterpart for the teacher renders.
            student_rendered_texts: Original student renders, required for
                source-validated native Unicode repair.
            teacher_rendered_texts: Counterpart for the teacher renders.
            drop_first_content_pair: Omit each turn's first paired content
                chunk from KD while retaining its EOT pair.
            included_regions: Optional subset of reasoning, close, answer,
                and eot to align when semantic regions are supplied.

        Returns:
            A dense-padded :class:`AlignmentBatch`, using the same token-index
            coordinates and padding sentinels as :meth:`align`.
        """
        if student_ids.ndim != 2 or teacher_ids.ndim != 2:
            raise ValueError("student_ids and teacher_ids must have shape [B, T]")
        b, t_s = student_ids.shape
        teacher_b, t_t = teacher_ids.shape
        if teacher_b != b:
            raise ValueError(f"student/teacher batch size mismatch: {b} vs {teacher_b}")
        for name, offsets, ids, attention_mask, asst_mask in (
            (
                "student",
                student_offsets,
                student_ids,
                student_attention_mask,
                student_asst_mask,
            ),
            (
                "teacher",
                teacher_offsets,
                teacher_ids,
                teacher_attention_mask,
                teacher_asst_mask,
            ),
        ):
            if offsets.shape != (*ids.shape, 2):
                raise ValueError(f"{name}_offsets must have shape [B, T, 2]")
            for mask_name, mask in (
                ("attention_mask", attention_mask),
                ("asst_mask", asst_mask),
            ):
                if mask is not None and mask.shape != ids.shape:
                    raise ValueError(f"{name}_{mask_name} must have shape [B, T]")
        for name, values in (
            ("student_asst_char_spans", student_asst_char_spans),
            ("teacher_asst_char_spans", teacher_asst_char_spans),
            ("student_eot_indices", student_eot_indices),
            ("teacher_eot_indices", teacher_eot_indices),
            ("student_alignment_regions", student_alignment_regions),
            ("teacher_alignment_regions", teacher_alignment_regions),
            ("student_answer_parts", student_answer_parts),
            ("teacher_answer_parts", teacher_answer_parts),
            ("student_rendered_texts", student_rendered_texts),
            ("teacher_rendered_texts", teacher_rendered_texts),
        ):
            if values is not None and len(values) != b:
                raise ValueError(f"{name} must have one entry per batch sample ({b})")

        per_sample_pairs: list[list[AlignmentPair]] = []
        for i in range(b):
            try:
                per_sample_pairs.append(
                    self.align_one_offset_per_asst(
                        student_ids[i].tolist(),
                        [tuple(offset) for offset in student_offsets[i].tolist()],
                        student_asst_char_spans[i],
                        teacher_ids[i].tolist(),
                        [tuple(offset) for offset in teacher_offsets[i].tolist()],
                        teacher_asst_char_spans[i],
                        student_asst_mask=(
                            student_asst_mask[i].tolist()
                            if student_asst_mask is not None
                            else None
                        ),
                        teacher_asst_mask=(
                            teacher_asst_mask[i].tolist()
                            if teacher_asst_mask is not None
                            else None
                        ),
                        student_eot_indices=(
                            student_eot_indices[i]
                            if student_eot_indices is not None
                            else None
                        ),
                        teacher_eot_indices=(
                            teacher_eot_indices[i]
                            if teacher_eot_indices is not None
                            else None
                        ),
                        student_alignment_regions=(
                            student_alignment_regions[i]
                            if student_alignment_regions is not None
                            else None
                        ),
                        teacher_alignment_regions=(
                            teacher_alignment_regions[i]
                            if teacher_alignment_regions is not None
                            else None
                        ),
                        student_answer_parts=(
                            student_answer_parts[i]
                            if student_answer_parts is not None
                            else None
                        ),
                        teacher_answer_parts=(
                            teacher_answer_parts[i]
                            if teacher_answer_parts is not None
                            else None
                        ),
                        student_rendered_text=(
                            student_rendered_texts[i]
                            if student_rendered_texts is not None
                            else None
                        ),
                        teacher_rendered_text=(
                            teacher_rendered_texts[i]
                            if teacher_rendered_texts is not None
                            else None
                        ),
                        drop_first_content_pair=drop_first_content_pair,
                        included_regions=included_regions,
                    )
                )
            except ValueError as error:
                raise ValueError(f"Chat alignment sample {i}: {error}") from error
        batch = self._pairs_to_batch(per_sample_pairs, b=b, t_s=t_s, t_t=t_t)
        self._drop_padding(
            batch,
            student_attention_mask=student_attention_mask,
            teacher_attention_mask=teacher_attention_mask,
        )
        return batch

    @staticmethod
    def _drop_padding(
        batch: AlignmentBatch,
        *,
        student_attention_mask: torch.Tensor | None,
        teacher_attention_mask: torch.Tensor | None,
    ) -> None:
        """Strip tokenizer padding out of the chunk-id tensors.

        Mutates ``batch`` in place. For every position the attention mask
        marks as padding, reset ``*_chunk_id`` to ``-1``. Gating per position
        (rather than trimming a contiguous span) keeps this correct under either
        left- or right-padding. A pair whose tokens are entirely padding on
        one side then has size 0 there and is dropped by
        :func:`nemo_rl.algorithms.x_token.loss_utils.valid_chunk_mask`; a pair
        straddling the real/pad boundary shrinks to its real tokens.
        """
        if student_attention_mask is not None:
            s_pad = student_attention_mask == 0
            batch.student_chunk_id[s_pad] = -1
        if teacher_attention_mask is not None:
            t_pad = teacher_attention_mask == 0
            batch.teacher_chunk_id[t_pad] = -1

    @staticmethod
    def _pairs_to_batch(
        per_sample_pairs: List[List[AlignmentPair]],
        *,
        b: int,
        t_s: int,
        t_t: int,
    ) -> AlignmentBatch:
        """Pack per-sample alignment lists into dense-padded tensors."""
        max_pairs = max((len(p) for p in per_sample_pairs), default=0)
        # Guarantee at least one slot so downstream tensor shapes stay sane.
        max_pairs = max(max_pairs, 1)

        pair_valid = torch.zeros((b, max_pairs), dtype=torch.bool)
        pair_is_correct = torch.zeros((b, max_pairs), dtype=torch.bool)
        student_chunk_id = torch.full((b, t_s), -1, dtype=torch.long)
        teacher_chunk_id = torch.full((b, t_t), -1, dtype=torch.long)

        for batch_i, pairs in enumerate(per_sample_pairs):
            for pair_i, pair in enumerate(pairs):
                if pair.s_start != -1 and pair.s_end != -1:
                    if 0 <= pair.s_start < t_s and 0 < pair.s_end <= t_s:
                        student_chunk_id[batch_i, pair.s_start : pair.s_end] = pair_i
                if pair.t_start != -1 and pair.t_end != -1:
                    if 0 <= pair.t_start < t_t and 0 < pair.t_end <= t_t:
                        teacher_chunk_id[batch_i, pair.t_start : pair.t_end] = pair_i
                pair_valid[batch_i, pair_i] = True
                pair_is_correct[batch_i, pair_i] = bool(pair.is_correct)

        return AlignmentBatch(
            pair_valid=pair_valid,
            pair_is_correct=pair_is_correct,
            student_chunk_id=student_chunk_id,
            teacher_chunk_id=teacher_chunk_id,
        )

    # ------------------------------------------------------------------ #
    # Per-sample alignment
    # ------------------------------------------------------------------ #
    def _align_single(
        self,
        student_ids: List[int],
        teacher_ids: List[int],
        student_offsets: List[Tuple[int, int]],
        teacher_offsets: List[Tuple[int, int]],
    ) -> List[AlignmentPair]:
        """Use the shared strict decoded matcher for ordinary text."""
        return self._align_one_offset(
            student_ids, teacher_ids, student_offsets, teacher_offsets
        )

    def _align_one_offset(
        self,
        student_ids: List[int],
        teacher_ids: List[int],
        student_offsets: List[Tuple[int, int]],
        teacher_offsets: List[Tuple[int, int]],
    ) -> List[AlignmentPair]:
        """Run strict offset clustering, then validate pairs by decoded text."""
        student_tokens_str = self.student_tokenizer.convert_ids_to_tokens(student_ids)
        teacher_tokens_str = self.teacher_tokenizer.convert_ids_to_tokens(teacher_ids)
        student_offsets = _normalize_canonical_merge_offsets(
            student_tokens_str,
            student_offsets,
            token_ids=student_ids,
            special_token_ids=getattr(self.student_tokenizer, "all_special_ids", [])
            or [],
        )
        teacher_offsets = _normalize_canonical_merge_offsets(
            teacher_tokens_str,
            teacher_offsets,
            token_ids=teacher_ids,
            special_token_ids=getattr(self.teacher_tokenizer, "all_special_ids", [])
            or [],
        )
        raw_pairs = align_by_offsets_cluster(
            student_ids,
            student_offsets,
            self.student_tokenizer,
            teacher_ids,
            teacher_offsets,
            self.teacher_tokenizer,
            student_tokens_str=student_tokens_str,
            teacher_tokens_str=teacher_tokens_str,
        )
        pairs: List[AlignmentPair] = []
        for s_tokens, t_tokens, s_start, s_end, t_start, t_end, _ in raw_pairs:
            is_correct = False
            if s_start >= 0 and t_start >= 0:
                s_text = self.student_tokenizer.decode(
                    student_ids[s_start:s_end],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                t_text = self.teacher_tokenizer.decode(
                    teacher_ids[t_start:t_end],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                is_correct = s_text == t_text
            pairs.append(
                AlignmentPair(
                    s_tokens=s_tokens,
                    t_tokens=t_tokens,
                    s_start=s_start,
                    s_end=s_end,
                    t_start=t_start,
                    t_end=t_end,
                    is_correct=is_correct,
                )
            )
        return pairs

    def align_one_offset_per_asst(
        self,
        student_ids: List[int],
        student_offsets: List[Tuple[int, int]],
        student_asst_char_spans: List[Tuple[int, int]],
        teacher_ids: List[int],
        teacher_offsets: List[Tuple[int, int]],
        teacher_asst_char_spans: List[Tuple[int, int]],
        *,
        student_asst_mask: List[int] | None = None,
        teacher_asst_mask: List[int] | None = None,
        student_alignment_regions: List[NativeAlignmentRegions] | None = None,
        teacher_alignment_regions: List[NativeAlignmentRegions] | None = None,
        student_eot_indices: List[int] | None = None,
        teacher_eot_indices: List[int] | None = None,
        student_answer_parts: List[List[NativeAlignmentPart] | None] | None = None,
        teacher_answer_parts: List[List[NativeAlignmentPart] | None] | None = None,
        student_rendered_text: str | None = None,
        teacher_rendered_text: str | None = None,
        drop_first_content_pair: bool = False,
        included_regions: Collection[str] | None = None,
    ) -> List[AlignmentPair]:
        """Align native chat-template surfaces one assistant region at a time.

        Student and teacher chat templates generally place the same assistant
        content at different absolute character offsets. Native-thinking
        templates can also use different whitespace around ``</think>``. This
        method aligns paired semantic regions after rebasing each region to
        character offset zero, then translates the resulting token spans back
        to full-sequence indices. Template scaffold and side-specific boundary
        whitespace are therefore excluded from the KD payload.

        Structured calls additionally bind prose and each tool payload to
        separate native pieces. Template-only separators, differently rendered
        scalar values, and tokens crossing those piece boundaries keep CE but
        are not compared as exact-text KD events.
        """
        if len(student_asst_char_spans) != len(teacher_asst_char_spans):
            raise ValueError(
                "assistant-turn count mismatch: "
                f"{len(student_asst_char_spans)} vs "
                f"{len(teacher_asst_char_spans)}"
            )
        if (student_alignment_regions is None) != (teacher_alignment_regions is None):
            raise ValueError("student/teacher alignment regions must be paired")
        if student_alignment_regions is not None and (
            len(student_alignment_regions) != len(student_asst_char_spans)
            or len(teacher_alignment_regions or []) != len(teacher_asst_char_spans)
        ):
            raise ValueError("alignment-region count must match assistant turns")
        if (student_eot_indices is None) != (teacher_eot_indices is None):
            raise ValueError("student/teacher EOT index lists must be paired")
        if student_eot_indices is not None and (
            len(student_eot_indices) != len(student_asst_char_spans)
            or len(teacher_eot_indices or []) != len(teacher_asst_char_spans)
        ):
            raise ValueError("EOT index count must match assistant turns")
        if (student_answer_parts is None) != (teacher_answer_parts is None):
            raise ValueError("student/teacher native answer parts must be paired")
        if student_answer_parts is not None and (
            len(student_answer_parts) != len(student_asst_char_spans)
            or len(teacher_answer_parts or []) != len(teacher_asst_char_spans)
        ):
            raise ValueError("native answer-part count must match assistant turns")
        if (student_rendered_text is None) != (teacher_rendered_text is None):
            raise ValueError("student/teacher rendered native texts must be paired")
        if student_answer_parts is not None and student_alignment_regions is None:
            raise ValueError("native answer parts require semantic alignment regions")
        for side, indices, ids in (
            ("student", student_eot_indices, student_ids),
            ("teacher", teacher_eot_indices, teacher_ids),
        ):
            if indices is not None and any(
                index < -1 or index >= len(ids) for index in indices
            ):
                raise ValueError(
                    f"{side} EOT indices must be -1 or valid token positions"
                )

        included = frozenset(included_regions) if included_regions is not None else None
        valid_regions = {"reasoning", "close", "answer", "eot"}
        if included is not None:
            invalid = included - valid_regions
            if invalid:
                raise ValueError(
                    f"included_regions contains unsupported values: {sorted(invalid)!r}"
                )
            if not included:
                raise ValueError("included_regions must not be empty")
            if student_alignment_regions is None:
                raise ValueError(
                    "included_regions requires native semantic alignment regions"
                )

        combined: List[AlignmentPair] = []
        for turn_i, ((s_start, s_end), (t_start, t_end)) in enumerate(
            zip(student_asst_char_spans, teacher_asst_char_spans)
        ):
            paired_regions: List[
                Tuple[str, Tuple[int, int], Tuple[int, int], str | None]
            ]
            if student_alignment_regions is None:
                paired_regions = [("content", (s_start, s_end), (t_start, t_end), None)]
            else:
                s_regions = student_alignment_regions[turn_i]
                t_regions = (teacher_alignment_regions or [])[turn_i]
                paired_regions = []
                for name in ("reasoning", "close", "answer"):
                    s_region = s_regions.get(name)
                    t_region = t_regions.get(name)
                    if (s_region is None) != (t_region is None):
                        raise ValueError(
                            f"turn {turn_i}: semantic region {name!r} is missing on one side"
                        )
                    if (
                        s_region is not None
                        and t_region is not None
                        and (included is None or name in included)
                    ):
                        s_parts = (
                            student_answer_parts[turn_i]
                            if name == "answer" and student_answer_parts is not None
                            else None
                        )
                        t_parts = (
                            teacher_answer_parts[turn_i]
                            if name == "answer" and teacher_answer_parts is not None
                            else None
                        )
                        if (s_parts is None) != (t_parts is None):
                            raise ValueError(
                                "native answer pieces are missing on one side"
                            )
                        if s_parts is None:
                            native_text: str | None = None
                            if student_rendered_text is not None:
                                assert teacher_rendered_text is not None
                                native_text = student_rendered_text[
                                    s_region[0] : s_region[1]
                                ]
                                if (
                                    native_text
                                    != teacher_rendered_text[t_region[0] : t_region[1]]
                                ):
                                    raise ValueError(
                                        f"Native region {name!r} has different "
                                        "student/teacher text; explicit native "
                                        "answer pieces are required."
                                    )
                            paired_regions.append(
                                (name, s_region, t_region, native_text)
                            )
                            continue
                        assert t_parts is not None
                        if len(s_parts) != len(t_parts):
                            raise ValueError("native answer-piece counts differ")
                        for s_part, t_part in zip(s_parts, t_parts):
                            for side, part, region, rendered in (
                                ("student", s_part, s_region, student_rendered_text),
                                ("teacher", t_part, t_region, teacher_rendered_text),
                            ):
                                start, end = part.span
                                if not region[0] <= start <= end <= region[1]:
                                    raise ValueError(
                                        f"{side} turn {turn_i}: answer piece {part.name!r} is outside its region"
                                    )
                                if (
                                    rendered is not None
                                    and rendered[start:end] != part.text
                                ):
                                    raise ValueError(
                                        f"{side} turn {turn_i}: answer piece {part.name!r} differs from rendered source"
                                    )
                            if (
                                s_part.name != t_part.name
                                or s_part.allow_native_difference
                                != t_part.allow_native_difference
                            ):
                                raise ValueError(
                                    "native answer-piece identities differ"
                                )
                            if s_part.text != t_part.text:
                                permitted_scalar_spellings = (
                                    {"true", "True"},
                                    {"false", "False"},
                                    {"null", "None"},
                                )
                                if s_part.allow_native_difference and any(
                                    {s_part.text, t_part.text} <= spellings
                                    for spellings in permitted_scalar_spellings
                                ):
                                    continue
                                raise ValueError(
                                    f"Native answer piece {s_part.name!r} has different "
                                    "student/teacher text; refusing offset-based KD."
                                )
                            paired_regions.append(
                                (s_part.name, s_part.span, t_part.span, s_part.text)
                            )

            turn_pairs: List[AlignmentPair] = []
            for _name, (s_region_start, s_region_end), (
                t_region_start,
                t_region_end,
            ), native_text in paired_regions:
                s_indices = [
                    i
                    for i, (start, end) in enumerate(student_offsets)
                    if end > start
                    and start < s_region_end
                    and end > s_region_start
                    and (
                        (start >= s_region_start and end <= s_region_end)
                        or (
                            student_alignment_regions is None
                            and student_asst_mask is not None
                        )
                    )
                    and (student_asst_mask is None or student_asst_mask[i] == 1)
                ]
                t_indices = [
                    i
                    for i, (start, end) in enumerate(teacher_offsets)
                    if end > start
                    and start < t_region_end
                    and end > t_region_start
                    and (
                        (start >= t_region_start and end <= t_region_end)
                        or (
                            teacher_alignment_regions is None
                            and teacher_asst_mask is not None
                        )
                    )
                    and (teacher_asst_mask is None or teacher_asst_mask[i] == 1)
                ]
                if not s_indices or not t_indices:
                    continue

                s_slice_ids = [student_ids[i] for i in s_indices]
                t_slice_ids = [teacher_ids[i] for i in t_indices]
                s_slice_offsets = [
                    (
                        max(student_offsets[i][0], s_region_start) - s_region_start,
                        min(student_offsets[i][1], s_region_end) - s_region_start,
                    )
                    for i in s_indices
                ]
                t_slice_offsets = [
                    (
                        max(teacher_offsets[i][0], t_region_start) - t_region_start,
                        min(teacher_offsets[i][1], t_region_end) - t_region_start,
                    )
                    for i in t_indices
                ]
                slice_pairs = self._align_one_offset(
                    student_ids=s_slice_ids,
                    teacher_ids=t_slice_ids,
                    student_offsets=s_slice_offsets,
                    teacher_offsets=t_slice_offsets,
                )
                if native_text is not None:
                    # Part text was checked equal before rebasing. Repair
                    # byte-fallback / NFC offset boundaries against that text,
                    # rather than accepting equal replacement characters.
                    slice_pairs = self._coalesce_native_piece_pairs(
                        slice_pairs,
                        student_ids=s_slice_ids,
                        teacher_ids=t_slice_ids,
                        student_offsets=s_slice_offsets,
                        teacher_offsets=t_slice_offsets,
                        text=native_text,
                    )
                for pair in slice_pairs:
                    turn_pairs.append(
                        AlignmentPair(
                            s_tokens=pair.s_tokens,
                            t_tokens=pair.t_tokens,
                            s_start=(
                                s_indices[pair.s_start] if pair.s_start >= 0 else -1
                            ),
                            s_end=(
                                s_indices[pair.s_end - 1] + 1
                                if pair.s_start >= 0
                                else -1
                            ),
                            t_start=(
                                t_indices[pair.t_start] if pair.t_start >= 0 else -1
                            ),
                            t_end=(
                                t_indices[pair.t_end - 1] + 1
                                if pair.t_start >= 0
                                else -1
                            ),
                            is_correct=pair.is_correct,
                        )
                    )

            if drop_first_content_pair:
                for pair_i, pair in enumerate(turn_pairs):
                    if pair.s_start >= 0 and pair.t_start >= 0:
                        del turn_pairs[pair_i]
                        break
            combined.extend(turn_pairs)

            include_eot = included is None or "eot" in included
            if not include_eot:
                continue
            if student_eot_indices is None:
                s_eot = next(
                    (
                        i
                        for i, (start, end) in enumerate(student_offsets)
                        if start == s_end and end > start
                    ),
                    -1,
                )
                t_eot = next(
                    (
                        i
                        for i, (start, end) in enumerate(teacher_offsets)
                        if start == t_end and end > start
                    ),
                    -1,
                )
            else:
                s_eot = student_eot_indices[turn_i]
                t_eot = (teacher_eot_indices or [])[turn_i]
            if s_eot >= 0 and t_eot >= 0:
                student_eot_id = int(student_ids[s_eot])
                teacher_eot_id = int(teacher_ids[t_eot])
                student_eot_tokens = self.student_tokenizer.convert_ids_to_tokens(
                    [student_eot_id]
                )
                teacher_eot_tokens = self.teacher_tokenizer.convert_ids_to_tokens(
                    [teacher_eot_id]
                )
                surfaces_match = student_eot_tokens == teacher_eot_tokens
                semantic_eos_match = student_eot_id == getattr(
                    self.student_tokenizer, "eos_token_id", None
                ) and teacher_eot_id == getattr(
                    self.teacher_tokenizer, "eos_token_id", None
                )
                combined.append(
                    AlignmentPair(
                        s_tokens=student_eot_tokens,
                        t_tokens=teacher_eot_tokens,
                        s_start=s_eot,
                        s_end=s_eot + 1,
                        t_start=t_eot,
                        t_end=t_eot + 1,
                        is_correct=surfaces_match or semantic_eos_match,
                    )
                )

        return combined

    def _coalesce_native_piece_pairs(
        self,
        pairs: List[AlignmentPair],
        *,
        student_ids: List[int],
        teacher_ids: List[int],
        student_offsets: List[Tuple[int, int]],
        teacher_offsets: List[Tuple[int, int]],
        text: str,
    ) -> List[AlignmentPair]:
        """Bind complete Unicode fragments on an identical native piece.

        A fast tokenizer may place an NFC-composed token over just the base
        character's offset, or split an emoji into several byte tokens with
        the same offset. Adjacent offset clusters must then be considered
        together. Every emitted pair must decode to the same NFC text as its
        original character envelope. Boundary orphans remain CE-only.
        """
        result: List[AlignmentPair] = []
        pending: List[AlignmentPair] = []
        for pair in pairs:
            if not pending and (pair.s_start < 0 or pair.t_start < 0):
                continue
            pending.append(pair)
            s_start = min(p.s_start for p in pending if p.s_start >= 0)
            s_end = max(p.s_end for p in pending if p.s_start >= 0)
            t_start = min(p.t_start for p in pending if p.t_start >= 0)
            t_end = max(p.t_end for p in pending if p.t_start >= 0)
            student_text = self.student_tokenizer.decode(
                student_ids[s_start:s_end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            teacher_text = self.teacher_tokenizer.decode(
                teacher_ids[t_start:t_end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            offsets = student_offsets[s_start:s_end] + teacher_offsets[t_start:t_end]
            start = min(a for a, _ in offsets)
            end = max(b for _, b in offsets)
            if (s_end < len(student_offsets) and student_offsets[s_end][0] < end) or (
                t_end < len(teacher_offsets) and teacher_offsets[t_end][0] < end
            ):
                # Consume every byte token covering this envelope, including
                # when an incomplete decode happens to equal a literal U+FFFD.
                continue
            expected = unicodedata.normalize("NFC", text[start:end])
            if (
                unicodedata.normalize("NFC", student_text) != expected
                or unicodedata.normalize("NFC", teacher_text) != expected
            ):
                continue
            result.append(
                AlignmentPair(
                    s_tokens=self.student_tokenizer.convert_ids_to_tokens(
                        student_ids[s_start:s_end]
                    ),
                    t_tokens=self.teacher_tokenizer.convert_ids_to_tokens(
                        teacher_ids[t_start:t_end]
                    ),
                    s_start=s_start,
                    s_end=s_end,
                    t_start=t_start,
                    t_end=t_end,
                    is_correct=True,
                )
            )
            pending.clear()
        if pending:
            raise ValueError(
                "Native answer piece produced unequal decoded token spans; "
                "refusing incorrect KD."
            )
        return result


# =====================================================================
# Module-level helpers: canonicalization + flexible string comparison.
# These are reused by ``tools/x_token/`` projection-prep CLIs, so they
# stay at module scope rather than living on ``TokenAligner``.
# =====================================================================


def canonical_token(token: str, *, enabled: bool = True) -> str:
    """Return a canonical representation of a tokenizer token.

    Public helper consumed by the alignment pipeline AND by the
    projection-prep CLIs in ``tools/x_token/``. The ``enabled`` flag is
    a passthrough toggle: when ``False`` the input is returned unchanged
    (lets CLI call sites gate canonicalization via a single flag without
    branching at every site).
    """
    if not enabled:
        return token
    if not token:
        return token

    # Normalize space prefixes.
    if token.startswith(" "):
        token = "Ġ" + token[1:]
    elif token.startswith("_"):
        token = "Ġ" + token[1:]
    elif token.startswith("▁"):
        token = "Ġ" + token[1:]

    # Newline and whitespace normalization.
    if token == "Ċ":
        token = "\n"
    elif token == "\\n":
        token = "\n"
    elif token == "ĉ":
        token = "\n"
    elif token == "Ġ\n":
        token = "\n"
    elif "Ċ" in token:
        token = token.replace("Ċ", "\n")
    elif "\\n" in token:
        token = token.replace("\\n", "\n")

    if token == "Ġ,":
        token = ","
    elif token == "Ġ.":
        token = "."
    elif token == "Ġ;":
        token = ";"
    elif token == "Ġ:":
        token = ":"

    # SentencePiece byte fallback like <0x20>.
    if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
        try:
            byte_val = int(token[3:5], 16)
            if 0 <= byte_val <= 255:
                return chr(byte_val)
        except ValueError:
            pass

    for broken, fixed in _UNICODE_FIXES.items():
        if broken in token:
            token = token.replace(broken, fixed)

    if token in _SPECIAL_TOKEN_MAP:
        return _SPECIAL_TOKEN_MAP[token]

    return token


def _canonicalize_sequence(
    seq: List[str],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Canonicalize every token in a sequence, including byte-merging.

    Returns:
        ``(canon, canon_to_orig)``. ``canon_to_orig[k]`` is a half-open
        ``[orig_start, orig_end)`` range giving the original-token positions
        that canonical token ``k`` was built from. Ranges are
        non-overlapping, strictly increasing, and jointly cover
        ``range(len(seq))``, preserving the original token-index coordinates
        when canonical groups are used to normalize byte-fragment offsets.
    """
    merged, ranges = _merge_encoding_artifacts(seq)
    canon = [canonical_token(t) for t in merged]
    return _merge_consecutive_bytes(canon, ranges)


def _normalize_canonical_merge_offsets(
    token_strings: List[str],
    offsets: List[Tuple[int, int]],
    *,
    token_ids: List[int],
    special_token_ids: Collection[int],
) -> List[Tuple[int, int]]:
    """Give every token in a canonical N-to-1 merge one offset envelope.

    Fast byte-level tokenizers can assign nested offsets to the pieces of one
    Unicode character.  Strict offset clustering would consume the outer
    piece alone and emit the nested piece as an orphan.  The canonicalizer
    already identifies which consecutive token ranges form one character;
    assigning those original positions the same connected offset envelope
    lets offset clustering retain the complete N-to-1 span.

    Token ids, strings, order, and sequence length are unchanged.  A range is
    left untouched when it contains a special/zero-width token or its offsets
    are non-monotonic or separated by a character gap.
    """
    if len(token_strings) != len(offsets) or len(token_ids) != len(offsets):
        raise ValueError(
            "token strings, token ids, and offsets must have matching lengths; "
            f"got {len(token_strings)}, {len(token_ids)}, and {len(offsets)}"
        )
    if not offsets:
        return []

    original_offsets = [(int(start), int(end)) for start, end in offsets]
    normalized_offsets = list(original_offsets)
    _, canonical_ranges = _canonicalize_sequence(token_strings)
    special_ids = {int(token_id) for token_id in special_token_ids}

    cursor = 0
    for range_start, range_end in canonical_ranges:
        if not (range_start == cursor and range_start < range_end <= len(offsets)):
            raise RuntimeError(
                "canonical token ranges must form an ordered partition of the "
                f"original sequence; got ({range_start}, {range_end}) after {cursor}"
            )
        cursor = range_end
        if range_end - range_start == 1:
            continue

        range_ids = token_ids[range_start:range_end]
        range_offsets = original_offsets[range_start:range_end]
        if any(int(token_id) in special_ids for token_id in range_ids):
            continue
        if any(end <= start for start, end in range_offsets):
            continue

        envelope_start, envelope_end = range_offsets[0]
        previous_start = envelope_start
        connected = True
        for start, end in range_offsets[1:]:
            if start < previous_start or start > envelope_end:
                connected = False
                break
            previous_start = start
            envelope_end = max(envelope_end, end)
        if not connected:
            continue

        envelope = (envelope_start, envelope_end)
        normalized_offsets[range_start:range_end] = [envelope] * (
            range_end - range_start
        )

    if cursor != len(offsets):
        raise RuntimeError(
            "canonical token ranges do not cover the original sequence; "
            f"covered {cursor} of {len(offsets)} positions"
        )
    return normalized_offsets


def _merge_encoding_artifacts(
    tokens: List[str],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Merge known multi-token mojibake patterns into single tokens.

    Returns:
        ``(merged, ranges)`` with one ``(orig_start, orig_end)`` entry per
        output token. Every entry in :data:`_MULTI_TOKEN_ARTIFACT_FIXES`
        rewrites to a single replacement token, so each merge contributes
        exactly one range covering the matched pattern.
    """
    if not tokens:
        return [], []
    result: List[str] = []
    ranges: List[Tuple[int, int]] = []
    next_start = 0
    for index, token in enumerate(tokens):
        if index < next_start:
            continue
        for pattern, replacement in _MULTI_TOKEN_ARTIFACT_FIXES_BY_FIRST.get(token, ()):
            end = index + len(pattern)
            if end <= len(tokens) and tokens[index:end] == pattern:
                # Every artifact fix contributes one merged token and its
                # unchanged half-open range in the original token sequence.
                assert len(replacement) == 1, (
                    "Multi-token artifact fix replacement must be a single "
                    f"token; got {replacement!r}"
                )
                result.extend(replacement)
                ranges.append((index, end))
                next_start = end
                break
        else:
            result.append(token)
            ranges.append((index, index + 1))
    return result, ranges


def _get_byte_value(token_char: str) -> int | None:
    """Return the byte value (0..255) for a single character, or None."""
    if len(token_char) != 1:
        return None
    char_ord = ord(token_char)
    if char_ord < 256:
        return char_ord
    return VISUAL_BYTE_MAP.get(token_char)


def _merge_consecutive_bytes(
    tokens: List[str],
    in_ranges: List[Tuple[int, int]],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Merge consecutive byte-fallback tokens back into Unicode characters.

    Propagates ``in_ranges`` parallel to ``tokens``: when a byte buffer
    collapses to one character, its parallel range slice is collapsed to a
    single ``(start, end)``; otherwise ranges pass through unchanged.
    """
    if not tokens:
        return [], []
    assert len(tokens) == len(in_ranges), (
        f"tokens/ranges length mismatch: {len(tokens)} vs {len(in_ranges)}"
    )
    result: List[str] = []
    result_ranges: List[Tuple[int, int]] = []
    byte_buffer: List[str] = []
    byte_buffer_ranges: List[Tuple[int, int]] = []
    for token, rng in zip(tokens, in_ranges):
        clean = token.lstrip("Ġ")
        if not clean:
            all_bytes = False
        elif clean.isascii():
            all_bytes = True
        else:
            all_bytes = all(_get_byte_value(c) is not None for c in clean)
        if all_bytes:
            byte_buffer.append(token)
            byte_buffer_ranges.append(rng)
        else:
            if byte_buffer:
                merged, merged_ranges = _try_merge_byte_buffer(
                    byte_buffer, byte_buffer_ranges
                )
                result.extend(merged)
                result_ranges.extend(merged_ranges)
                byte_buffer = []
                byte_buffer_ranges = []
            result.append(token)
            result_ranges.append(rng)
    if byte_buffer:
        merged, merged_ranges = _try_merge_byte_buffer(byte_buffer, byte_buffer_ranges)
        result.extend(merged)
        result_ranges.extend(merged_ranges)
    return result, result_ranges


def _try_merge_byte_buffer(
    byte_tokens: List[str],
    byte_ranges: List[Tuple[int, int]],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Decode 2-4 buffered byte tokens as a single UTF-8 character.

    Returns the merged single-character token plus a single collapsed
    range covering the whole buffer, or the unchanged buffer + ranges
    when no merge is possible.
    """
    if not byte_tokens:
        return [], []
    if len(byte_tokens) == 1:
        token = byte_tokens[0]
        clean = token.lstrip("Ġ")
        if len(clean) <= 1:
            return byte_tokens, byte_ranges

    space_prefix = "Ġ" if byte_tokens[0].startswith("Ġ") else ""
    raw_bytes: List[int] = []
    for token in byte_tokens:
        clean = token.lstrip("Ġ")
        for c in clean:
            v = _get_byte_value(c)
            if v is None:
                return byte_tokens, byte_ranges
            raw_bytes.append(v)
            # This helper only decodes one UTF-8 character. Once a buffer
            # exceeds four bytes, later input cannot make it mergeable.
            if len(raw_bytes) > 4:
                return byte_tokens, byte_ranges

    if len(raw_bytes) < 2:
        return byte_tokens, byte_ranges
    try:
        decoded = bytes(raw_bytes).decode("utf-8")
        if len(decoded) == 1 and ord(decoded) > 127:
            return (
                [space_prefix + decoded],
                [(byte_ranges[0][0], byte_ranges[-1][1])],
            )
        return byte_tokens, byte_ranges
    except UnicodeDecodeError:
        return byte_tokens, byte_ranges


# Pair padding/EOS aliases before attention masks remove padded positions.
_PAD_EQUIVALENT_ROLES = {"pad", "eos"}


def _role_of(tok, token_id: int) -> str:
    """Classify a token id as a special-token role or ``"content"``."""
    if token_id == getattr(tok, "bos_token_id", None):
        return "bos"
    if token_id == getattr(tok, "eos_token_id", None):
        return "eos"
    if token_id == getattr(tok, "pad_token_id", None):
        return "pad"
    if token_id == getattr(tok, "unk_token_id", None):
        return "unk"
    if token_id == getattr(tok, "sep_token_id", None):
        return "sep"
    if token_id == getattr(tok, "cls_token_id", None):
        return "cls"
    if token_id == getattr(tok, "mask_token_id", None):
        return "mask"
    special_ids = getattr(tok, "all_special_ids", []) or []
    if token_id in special_ids:
        try:
            return f"special:{tok.convert_ids_to_tokens(int(token_id))}"
        except Exception:
            return f"special:id={int(token_id)}"
    return "content"


def _partition(
    input_ids: List[int], offsets: List[Tuple[int, int]]
) -> Tuple[List[int], List[Tuple[int, int, int]], List[int]]:
    """Split a sequence into leading-specials / content / trailing-specials.

    Content vs special is decided by ``(0, 0)`` offset; mid-stream specials
    are folded into trailing.
    """
    n = len(input_ids)
    is_content = [offsets[i][1] > offsets[i][0] for i in range(n)]
    first = next((i for i in range(n) if is_content[i]), None)
    last = next((i for i in range(n - 1, -1, -1) if is_content[i]), None)
    if first is None:
        return list(range(n)), [], []
    assert last is not None
    leading = [i for i in range(first) if not is_content[i]]
    trailing = [i for i in range(last + 1, n) if not is_content[i]]
    content = [
        (int(offsets[i][0]), int(offsets[i][1]), i)
        for i in range(first, last + 1)
        if is_content[i]
    ]
    mid_specials = [i for i in range(first, last + 1) if not is_content[i]]
    trailing = sorted(set(trailing + mid_specials))
    return leading, content, trailing


def _pair_specials_by_role(
    s_positions: List[int],
    s_ids: List[int],
    s_tok,
    t_positions: List[int],
    t_ids: List[int],
    t_tok,
) -> List[Tuple[List[int], List[int]]]:
    """Pair leading/trailing special tokens 1<->1 by matching role."""
    groups: List[Tuple[List[int], List[int]]] = []
    si = ti = 0
    while si < len(s_positions) and ti < len(t_positions):
        s_role = _role_of(s_tok, s_ids[s_positions[si]])
        t_role = _role_of(t_tok, t_ids[t_positions[ti]])
        if s_role == t_role or (
            s_role in _PAD_EQUIVALENT_ROLES and t_role in _PAD_EQUIVALENT_ROLES
        ):
            groups.append(([s_positions[si]], [t_positions[ti]]))
            si += 1
            ti += 1
        elif s_role in _PAD_EQUIVALENT_ROLES:
            si += 1
        elif t_role in _PAD_EQUIVALENT_ROLES:
            ti += 1
        else:
            groups.append(([s_positions[si]], []))
            groups.append(([], [t_positions[ti]]))
            si += 1
            ti += 1
    for i in range(si, len(s_positions)):
        if _role_of(s_tok, s_ids[s_positions[i]]) in _PAD_EQUIVALENT_ROLES:
            continue
        groups.append(([s_positions[i]], []))
    for j in range(ti, len(t_positions)):
        if _role_of(t_tok, t_ids[t_positions[j]]) in _PAD_EQUIVALENT_ROLES:
            continue
        groups.append(([], [t_positions[j]]))
    return groups


def _cluster_same_span(
    content: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, List[int]]]:
    """Collapse consecutive tokens sharing the *exact* same ``(cs, ce)`` span.

    Each run of same-span tokens becomes one cluster; different-span tokens
    (even overlapping) stay separate.
    """
    if not content:
        return []
    clusters: List[Tuple[int, int, List[int]]] = []
    i = 0
    while i < len(content):
        cs, ce, pos = content[i]
        positions = [pos]
        j = i + 1
        while j < len(content) and content[j][0] == cs and content[j][1] == ce:
            positions.append(content[j][2])
            j += 1
        clusters.append((cs, ce, positions))
        i = j
    return clusters


def _content_align_offset_cluster(
    s_content: List[Tuple[int, int, int]],
    t_content: List[Tuple[int, int, int]],
) -> List[Tuple[List[int], List[int]]]:
    """Pair student/teacher clusters covering the same character range.

    Pre-merges same-span clusters, then runs a strict char-end walker with
    orphan emission. Every paired group satisfies::

        min(cs over G_s) == min(cs over G_t)
        max(ce over G_s) == max(ce over G_t)
    """
    s_clusters = _cluster_same_span(s_content)
    t_clusters = _cluster_same_span(t_content)
    n_s, n_t = len(s_clusters), len(t_clusters)

    groups: List[Tuple[List[int], List[int]]] = []
    si = ti = 0

    while si < n_s and ti < n_t:
        while si < n_s and ti < n_t and s_clusters[si][0] != t_clusters[ti][0]:
            if s_clusters[si][0] < t_clusters[ti][0]:
                groups.append((list(s_clusters[si][2]), []))
                si += 1
            else:
                groups.append(([], list(t_clusters[ti][2])))
                ti += 1
        if si >= n_s or ti >= n_t:
            break

        s_group_start = si
        t_group_start = ti
        s_end = s_clusters[si][1]
        t_end = t_clusters[ti][1]
        exhausted = False
        while s_end != t_end:
            if s_end < t_end:
                si += 1
                if si >= n_s:
                    exhausted = True
                    break
                s_end = s_clusters[si][1]
            else:
                ti += 1
                if ti >= n_t:
                    exhausted = True
                    break
                t_end = t_clusters[ti][1]

        if not exhausted:
            s_pos = [p for c in s_clusters[s_group_start : si + 1] for p in c[2]]
            t_pos = [p for c in t_clusters[t_group_start : ti + 1] for p in c[2]]
            groups.append((s_pos, t_pos))
            si += 1
            ti += 1
        else:
            for c in s_clusters[s_group_start : min(si + 1, n_s)]:
                groups.append((list(c[2]), []))
            for c in t_clusters[t_group_start : min(ti + 1, n_t)]:
                groups.append(([], list(c[2])))
            si = n_s
            ti = n_t
            break

    while si < n_s:
        groups.append((list(s_clusters[si][2]), []))
        si += 1
    while ti < n_t:
        groups.append(([], list(t_clusters[ti][2])))
        ti += 1
    return groups


def align_by_offsets_cluster(
    student_ids: List[int],
    student_offsets: List[Tuple[int, int]],
    student_tokenizer,
    teacher_ids: List[int],
    teacher_offsets: List[Tuple[int, int]],
    teacher_tokenizer,
    *,
    student_tokens_str: List[str] | None = None,
    teacher_tokens_str: List[str] | None = None,
) -> List[Tuple[List[str], List[str], int, int, int, int, bool]]:
    """Cluster + strict char-end offset alignment for a single sample.

    Args:
        student_ids: list of student token ids (len = ctx_length, incl. pad).
        student_offsets: list of ``(cs, ce)`` tuples per token, from
            ``return_offsets_mapping=True`` on a fast HF tokenizer.
        student_tokenizer: HF tokenizer (used for special-token role lookup).
        teacher_ids/offsets/tokenizer: same for teacher.

    Returns:
        list of 7-tuples ``(s_tok_strs, t_tok_strs, s_start, s_end, t_start,
        t_end, is_correct)`` — paired groups have ``is_correct=True`` and
        contiguous position ranges on both sides; orphan groups have
        ``is_correct=False`` and the empty side carries ``start=end=-1``.
    """
    if len(student_ids) != len(student_offsets) or len(teacher_ids) != len(
        teacher_offsets
    ):
        raise ValueError("token IDs and offsets must have matching lengths")
    s_off_tuples = [(start, end) for start, end in student_offsets]
    t_off_tuples = [(start, end) for start, end in teacher_offsets]

    s_lead, s_cont, s_trail = _partition(student_ids, s_off_tuples)
    t_lead, t_cont, t_trail = _partition(teacher_ids, t_off_tuples)

    groups: List[Tuple[List[int], List[int]]] = []
    groups += _pair_specials_by_role(
        s_lead,
        student_ids,
        student_tokenizer,
        t_lead,
        teacher_ids,
        teacher_tokenizer,
    )
    groups += _content_align_offset_cluster(s_cont, t_cont)
    groups += _pair_specials_by_role(
        s_trail,
        student_ids,
        student_tokenizer,
        t_trail,
        teacher_ids,
        teacher_tokenizer,
    )

    if student_tokens_str is None:
        student_tokens_str = student_tokenizer.convert_ids_to_tokens(student_ids)
    if teacher_tokens_str is None:
        teacher_tokens_str = teacher_tokenizer.convert_ids_to_tokens(teacher_ids)
    if len(student_tokens_str) != len(student_ids):
        raise ValueError("student token strings must match the token ID count")
    if len(teacher_tokens_str) != len(teacher_ids):
        raise ValueError("teacher token strings must match the token ID count")

    aligned_pairs: List[Tuple[Any, ...]] = []
    for s_pos, t_pos in groups:
        s_toks = [student_tokens_str[i] for i in s_pos]
        t_toks = [teacher_tokens_str[i] for i in t_pos]
        s_start = s_pos[0] if s_pos else -1
        s_end = s_pos[-1] + 1 if s_pos else -1
        t_start = t_pos[0] if t_pos else -1
        t_end = t_pos[-1] + 1 if t_pos else -1
        is_correct = bool(s_pos and t_pos)
        aligned_pairs.append(
            (s_toks, t_toks, s_start, s_end, t_start, t_end, is_correct)
        )

    return aligned_pairs
