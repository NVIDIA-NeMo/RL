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

Ports the alignment algorithm from the PyTorch tokenalign reference
(``train_distillation_ddp.py`` companion ``tokenalign.py``). The DP kernel,
canonicalization helpers, anchor optimization, and post-processing are kept
faithful so loss-side parity with that reference is preserved. Anything
unrelated to running cross-tokenizer alignment for off-policy distillation
(rule tracking, learnable projection, MSE loss, multiple compute_loss
variants, accuracy, translation) is dropped.

Public surface:
    - :class:`AlignmentPair` — per-pair record produced by the DP / anchor
      pipeline; replaces the loose ``(s_tokens, t_tokens, s_start, s_end,
      t_start, t_end, is_correct)`` tuples that the helpers used to pass
      around.
    - :class:`AlignmentBatch` — dense-padded per-batch alignment payload that
      covers all three loss modes (P-KL, gold_loss, xtoken_loss).
    - :class:`TokenAligner` — owns the two tokenizers and the projection
      matrix, exposes :meth:`align` for the collator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

# Visual byte representations used by some BPE tokenizers (especially for
# emojis / non-ASCII bytes). Copied from the PT reference verbatim — these
# constants are content-coupled to the tokenizers we align across.
VISUAL_BYTE_MAP = {
    "ð": 240, "Ɩ": 241, "Ɨ": 242, "Ƙ": 243, "ƙ": 244, "ƚ": 245, "ƛ": 246, "Ɯ": 247,
    "Ɲ": 248, "ƞ": 249, "Ɵ": 250, "Ơ": 251, "ơ": 252, "Ƣ": 253, "ƣ": 254, "Ƥ": 255,
    "Ł": 156, "ł": 157, "Ń": 158, "ń": 159, "ĺ": 149, "Ļ": 150, "ļ": 151, "Ľ": 152,
    "ľ": 153, "Ŀ": 154, "ŀ": 155, "Ĭ": 135, "ĭ": 136, "Į": 137, "į": 138, "İ": 139,
    "ı": 140, "Ĳ": 141, "ĳ": 142, "Ĵ": 143, "ĵ": 144, "Ķ": 145, "ķ": 146, "ĸ": 147,
    "Ĺ": 148, "ĥ": 128, "Ħ": 129, "ħ": 130, "Ĩ": 131, "ĩ": 132, "Ī": 133, "ī": 134,
    "Ģ": 162, "ģ": 163, "Ĝ": 28, "ĝ": 29, "Ğ": 30, "ğ": 31,
}

# Multi-token encoding artifacts (mojibake patterns) where the broken byte
# sequence spans tokens. Patterns are checked left-to-right with the first
# match wins. From PT reference; trimmed to the high-frequency entries.
_MULTI_TOKEN_ARTIFACT_FIXES = [
    (["ĠâĪ", "ĳ"], ["Ġ∑"]), (["âĪ", "ĳ"], ["∑"]),
    (["ĠâĪ", "ı"], ["Ġ∏"]), (["âĪ", "ı"], ["∏"]),
    (["ĠâĪ", "Ĥ"], ["Ġ∂"]), (["âĪ", "Ĥ"], ["∂"]),
    (["ĠâĪ", "ĩ"], ["Ġ∇"]), (["âĪ", "ĩ"], ["∇"]),
    (["ĠâĪ", "ŀ"], ["Ġ∞"]), (["âĪ", "ŀ"], ["∞"]),
    (["ĠâĪ", "ļ"], ["Ġ√"]), (["âĪ", "ļ"], ["√"]),
    (["ĠâĪ", "«"], ["Ġ∫"]), (["âĪ", "«"], ["∫"]),
    (["Ġâī", "ł"], ["Ġ≠"]), (["âī", "ł"], ["≠"]),
    (["Ġä¸", "Ń"], ["Ġ中"]), (["ä¸", "Ń"], ["中"]),
    (["æĸ", "ĩ"], ["文"]), (["Ġæĸ", "ĩ"], ["Ġ文"]),
]

# Per-token canonicalizations applied after multi-token artifact fixes.
_UNICODE_FIXES = {
    "Ã±": "ñ", "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
    "Ã": "À", "Ã¢": "â", "Ã§": "ç",
    "Ã¨": "è", "Ã«": "ë", "Ã®": "î", "Ã´": "ô",
    "Ã¹": "ù", "Ã»": "û", "Ã¿": "ÿ",
    "ä¸Ń": "中", "æĸĩ": "文", "æĹ¥æľ¬": "日本", "èªŀ": "語",
    "ÐłÑĥÑģ": "Рус", "ÑģÐºÐ¸Ð¹": "ский",
    "Ø§ÙĦØ¹Ø±Ø¨ÙĬØ©": "العربية",
    "à¤¹": "ह", "à¤¿à¤Ĥ": "हिं", "à¤¦à¥Ģ": "दी",
    "âĪĳ": "∑", "âĪı": "∏", "âĪĤ": "∂", "âĪĩ": "∇",
    "âĪŀ": "∞", "âĪļ": "√", "âĪ«": "∫", "âīĪ": "≈",
    "âīł": "≠", "âī¤": "≤", "âī¥": "≥",
}

_SPECIAL_TOKEN_MAP = {
    "<|begin_of_text|>": "<bos>",
    "<bos>": "<bos>",
    "<pad>": "",
}


@dataclass
class AlignmentPair:
    """One aligned span between student and teacher token sequences.

    The DP / anchor / post-process helpers construct these as they trace
    the alignment; ``_align_single`` then fills in ``is_correct`` from the
    canonicalized-text comparison. Insertions/deletions use ``-1`` for the
    empty side's start/end indices.

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
        is_correct: ``True`` when the canonicalized student span text
            matches the canonicalized teacher span text. Defaults to
            ``False`` so DP / anchor stages can build pairs without
            computing the mask up front.
    """

    s_tokens: List[str]
    t_tokens: List[str]
    s_start: int
    s_end: int
    t_start: int
    t_end: int
    is_correct: bool = False


@dataclass
class AlignmentBatch:
    """Per-batch alignment payload covering all three loss modes.

    The collator hands this dataclass directly to the loss fn alongside the
    tokenized batch. Tensors are dense-padded to the batch maximum so DTensor
    V2 can shard on dim 0 without knowing about cross-tokenizer specifics.

    Attributes:
        pair_valid: ``[B, max_pairs]`` bool. False on padding entries.
        pair_is_correct: ``[B, max_pairs]`` bool. True when canonicalized
            student span text matches canonicalized teacher span text.
        student_exact_partition_mask: ``[B, T_s]`` bool. True at student
            tokens that sit on a 1-1 exact-match pair (gold_loss partition).
        teacher_exact_partition_mask: ``[B, T_t]`` bool. Counterpart.
        student_chunk_id: ``[B, T_s]`` long. Chunk index (= pair index) the
            student token belongs to; ``-1`` if not in any chunk
            (insertion-only pair on student side).
        teacher_chunk_id: ``[B, T_t]`` long. Counterpart.
        num_chunks: ``[B]`` long. Number of valid chunks in each sample.
    """

    pair_valid: torch.Tensor
    pair_is_correct: torch.Tensor
    student_exact_partition_mask: torch.Tensor
    teacher_exact_partition_mask: torch.Tensor
    student_chunk_id: torch.Tensor
    teacher_chunk_id: torch.Tensor
    num_chunks: torch.Tensor


class TokenAligner:
    """Aligns student and teacher tokenizations of the same source text.

    The alignment algorithm is a Needleman-Wunsch DP over canonicalized token
    strings, augmented with multi-token combination scoring (one student
    token can match a span of teacher tokens and vice versa, up to
    ``max_comb_len``) and anchor-based segmentation for long sequences.

    Args:
        student_tokenizer: HF tokenizer for the student model.
        teacher_tokenizer: HF tokenizer for the teacher model.
        projection_matrix_path: Path retained on the aligner for downstream
            callers (e.g. the loss fn) that materialize the projection on
            their training device via
            :func:`nemo_rl.algorithms.x_token.loss_utils.get_sparse_projection_matrix`
            or :func:`nemo_rl.algorithms.x_token.loss_utils.get_topk_projection`.
        max_comb_len: Maximum span length considered when matching one token
            on one side against multiple tokens on the other.
    """

    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        projection_matrix_path: str,
        max_comb_len: int = 4,
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_combination_len = max_comb_len
        self.projection_matrix_path = projection_matrix_path

    def align(
        self,
        student_ids: torch.Tensor,
        teacher_ids: torch.Tensor,
    ) -> AlignmentBatch:
        """Align a batch of student/teacher token id tensors.

        Args:
            student_ids: ``[B, T_s]`` long tensor.
            teacher_ids: ``[B, T_t]`` long tensor.

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

        student_token_lists: List[List[str]] = [
            self.student_tokenizer.convert_ids_to_tokens(student_ids[i].tolist())
            for i in range(b)
        ]
        teacher_token_lists: List[List[str]] = [
            self.teacher_tokenizer.convert_ids_to_tokens(teacher_ids[i].tolist())
            for i in range(b)
        ]

        per_sample_pairs: List[List[AlignmentPair]] = []
        for s_toks, t_toks in zip(student_token_lists, teacher_token_lists):
            pairs = self._align_single(s_toks, t_toks)
            per_sample_pairs.append(pairs)

        return self._pairs_to_batch(per_sample_pairs, b=b, t_s=t_s, t_t=t_t)

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
        student_partition = torch.zeros((b, t_s), dtype=torch.bool)
        teacher_partition = torch.zeros((b, t_t), dtype=torch.bool)
        student_chunk_id = torch.full((b, t_s), -1, dtype=torch.long)
        teacher_chunk_id = torch.full((b, t_t), -1, dtype=torch.long)
        num_chunks = torch.zeros((b,), dtype=torch.long)

        for batch_i, pairs in enumerate(per_sample_pairs):
            num_chunks[batch_i] = len(pairs)
            for pair_i, pair in enumerate(pairs):
                if pair.s_start != -1 and pair.s_end != -1:
                    if 0 <= pair.s_start < t_s and 0 < pair.s_end <= t_s:
                        student_chunk_id[batch_i, pair.s_start : pair.s_end] = pair_i
                if pair.t_start != -1 and pair.t_end != -1:
                    if 0 <= pair.t_start < t_t and 0 < pair.t_end <= t_t:
                        teacher_chunk_id[batch_i, pair.t_start : pair.t_end] = pair_i
                pair_valid[batch_i, pair_i] = True
                pair_is_correct[batch_i, pair_i] = bool(pair.is_correct)
                # gold_loss partition: tokens on a 1-1 exact-match pair.
                if (
                    pair.is_correct
                    and pair.s_start != -1
                    and pair.t_start != -1
                    and (pair.s_end - pair.s_start) == 1
                    and (pair.t_end - pair.t_start) == 1
                ):
                    if 0 <= pair.s_start < t_s:
                        student_partition[batch_i, pair.s_start] = True
                    if 0 <= pair.t_start < t_t:
                        teacher_partition[batch_i, pair.t_start] = True

        return AlignmentBatch(
            pair_valid=pair_valid,
            pair_is_correct=pair_is_correct,
            student_exact_partition_mask=student_partition,
            teacher_exact_partition_mask=teacher_partition,
            student_chunk_id=student_chunk_id,
            teacher_chunk_id=teacher_chunk_id,
            num_chunks=num_chunks,
        )

    # ------------------------------------------------------------------ #
    # Per-sample alignment pipeline
    # ------------------------------------------------------------------ #
    def _align_single(
        self,
        student_tokens: List[str],
        teacher_tokens: List[str],
        exact_match_score: float = 3.0,
        combination_score_multiplier: float = 1.5,
        gap_penalty: float = -1.5,
        anchor_lengths: Tuple[int, ...] = (3,),
    ) -> List[AlignmentPair]:
        """Run canonicalize -> anchor-DP -> post-process for one sample.

        Returns:
            A list of :class:`AlignmentPair`. Insertions/deletions use
            ``-1`` for the empty side's start/end. Pair start/end indices
            address the **original** token sequences (not canonical
            space), so they can be written straight into the chunk-id
            tensors in :meth:`_pairs_to_batch`.
        """
        student_canon, student_canon_to_orig = _canonicalize_sequence(student_tokens)
        teacher_canon, teacher_canon_to_orig = _canonicalize_sequence(teacher_tokens)

        aligned, _ = self._align_with_anchors(
            student_canon,
            teacher_canon,
            anchor_lengths=anchor_lengths,
            exact_match_score=exact_match_score,
            combination_score_multiplier=combination_score_multiplier,
            gap_penalty=gap_penalty,
            max_combination_len=self.max_combination_len,
            ignore_leading_char_diff=False,
        )
        aligned = self._post_process_alignment(
            aligned,
            exact_match_score=exact_match_score,
            combination_score_multiplier=combination_score_multiplier,
            gap_penalty=gap_penalty,
            max_combination_len=self.max_combination_len,
        )
        # Remap canonical-space span indices back onto the original token
        # axis BEFORE the is_correct mask write — the mask reads
        # pair.s_tokens (canonical strings) and is order-independent, but
        # everything downstream of _align_single (the chunk_id writes in
        # _pairs_to_batch) needs original-space indices.
        self._remap_pairs_to_original(
            aligned,
            student_canon_to_orig=student_canon_to_orig,
            teacher_canon_to_orig=teacher_canon_to_orig,
        )
        for pair, m in zip(aligned, self._alignment_mask(aligned)):
            pair.is_correct = m
        return aligned

    @staticmethod
    def _remap_pairs_to_original(
        pairs: List[AlignmentPair],
        *,
        student_canon_to_orig: List[Tuple[int, int]],
        teacher_canon_to_orig: List[Tuple[int, int]],
    ) -> None:
        """Remap pair start/end indices from canonical to original space.

        Mutates ``pairs`` in place. Insertion/deletion pairs (where the
        empty side already has ``-1`` start/end) keep the sentinel.
        ``s_end - 1`` / ``t_end - 1`` indexes the last canonical token in
        the span; its ``orig_end`` is the new exclusive end.
        """
        for pair in pairs:
            if pair.s_start != -1:
                s0 = student_canon_to_orig[pair.s_start][0]
                s1 = student_canon_to_orig[pair.s_end - 1][1]
                pair.s_start, pair.s_end = s0, s1
            if pair.t_start != -1:
                t0 = teacher_canon_to_orig[pair.t_start][0]
                t1 = teacher_canon_to_orig[pair.t_end - 1][1]
                pair.t_start, pair.t_end = t0, t1

    # ------------------------------------------------------------------ #
    # Anchor-based segmentation
    # ------------------------------------------------------------------ #
    def _align_with_anchors(
        self,
        student_tokens: List[str],
        teacher_tokens: List[str],
        anchor_lengths: Tuple[int, ...] = (3,),
        *,
        exact_match_score: float,
        combination_score_multiplier: float,
        gap_penalty: float,
        max_combination_len: int,
        ignore_leading_char_diff: bool,
    ) -> Tuple[List[AlignmentPair], float]:
        """Optimize long alignments by pinning unique n-gram matches as anchors.

        Falls back to plain DP when no anchors exist or when
        ``anchor_lengths`` is empty.
        """
        dp_kwargs = dict(
            exact_match_score=exact_match_score,
            combination_score_multiplier=combination_score_multiplier,
            gap_penalty=gap_penalty,
            max_combination_len=max_combination_len,
            ignore_leading_char_diff=ignore_leading_char_diff,
        )
        if not anchor_lengths:
            return self._align_dp(student_tokens, teacher_tokens, **dp_kwargs)

        # Find unique n-gram matches in both sequences.
        all_potential_anchors: List[Tuple[int, int, int]] = []
        for anchor_len in anchor_lengths:
            if anchor_len == 1:
                student_counts: dict[str, List[int]] = {}
                teacher_counts: dict[str, List[int]] = {}
                for i, t in enumerate(student_tokens):
                    student_counts.setdefault(t, []).append(i)
                for j, t in enumerate(teacher_tokens):
                    teacher_counts.setdefault(t, []).append(j)
                for token in student_counts.keys() & teacher_counts.keys():
                    if len(student_counts[token]) == 1 and len(teacher_counts[token]) == 1:
                        all_potential_anchors.append(
                            (student_counts[token][0], teacher_counts[token][0], 1)
                        )
            else:
                student_ngrams: dict[Tuple[str, ...], List[int]] = {}
                teacher_ngrams: dict[Tuple[str, ...], List[int]] = {}
                for i in range(len(student_tokens) - anchor_len + 1):
                    student_ngrams.setdefault(tuple(student_tokens[i : i + anchor_len]), []).append(i)
                for j in range(len(teacher_tokens) - anchor_len + 1):
                    teacher_ngrams.setdefault(tuple(teacher_tokens[j : j + anchor_len]), []).append(j)
                for ngram in student_ngrams.keys() & teacher_ngrams.keys():
                    if len(student_ngrams[ngram]) == 1 and len(teacher_ngrams[ngram]) == 1:
                        i = student_ngrams[ngram][0]
                        j = teacher_ngrams[ngram][0]
                        if (
                            i + anchor_len <= len(student_tokens)
                            and j + anchor_len <= len(teacher_tokens)
                            and student_tokens[i : i + anchor_len] == teacher_tokens[j : j + anchor_len]
                        ):
                            all_potential_anchors.append((i, j, anchor_len))

        # Greedy non-conflicting selection, preferring longer anchors.
        all_potential_anchors.sort(key=lambda x: (-x[2], x[0], x[1]))
        used_student: set[int] = set()
        used_teacher: set[int] = set()
        selected: List[Tuple[int, int, int]] = []
        for i, j, k in all_potential_anchors:
            r1 = set(range(i, i + k))
            r2 = set(range(j, j + k))
            if not (r1 & used_student) and not (r2 & used_teacher):
                selected.append((i, j, k))
                used_student.update(r1)
                used_teacher.update(r2)
        selected.sort()

        # Validate monotonic ordering.
        validated: List[Tuple[int, int, int]] = []
        last_j = -1
        for i, j, k in selected:
            if j > last_j and student_tokens[i : i + k] == teacher_tokens[j : j + k]:
                validated.append((i, j, k))
                last_j = j + k - 1

        if not validated:
            return self._align_dp(student_tokens, teacher_tokens, **dp_kwargs)

        full_alignment: List[AlignmentPair] = []
        last_i, last_j = 0, 0
        for i, j, k in validated:
            student_seg, teacher_seg = student_tokens[last_i:i], teacher_tokens[last_j:j]
            if student_seg or teacher_seg:
                aligned_segment, _ = self._align_dp(student_seg, teacher_seg, **dp_kwargs)
                full_alignment.extend(
                    self._shift_pairs(aligned_segment, last_i, last_j)
                )
            # Anchor itself splits to 1-1 matches.
            for kk in range(k):
                full_alignment.append(
                    AlignmentPair(
                        s_tokens=[student_tokens[i + kk]],
                        t_tokens=[teacher_tokens[j + kk]],
                        s_start=i + kk,
                        s_end=i + kk + 1,
                        t_start=j + kk,
                        t_end=j + kk + 1,
                    )
                )
            last_i, last_j = i + k, j + k

        student_seg, teacher_seg = student_tokens[last_i:], teacher_tokens[last_j:]
        if student_seg or teacher_seg:
            aligned_segment, _ = self._align_dp(student_seg, teacher_seg, **dp_kwargs)
            full_alignment.extend(self._shift_pairs(aligned_segment, last_i, last_j))

        return full_alignment, 0.0

    # ------------------------------------------------------------------ #
    # DP kernel + post-process (algorithm-internal helpers).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _align_dp(
        student_tokens: List[str],
        teacher_tokens: List[str],
        *,
        exact_match_score: float,
        combination_score_multiplier: float,
        gap_penalty: float,
        max_combination_len: int,
        ignore_leading_char_diff: bool,
    ) -> Tuple[List[AlignmentPair], float]:
        """Needleman-Wunsch DP with up-to-``max_combination_len`` token spans."""
        n1, n2 = len(student_tokens), len(teacher_tokens)
        dp = np.zeros((n1 + 1, n2 + 1), dtype=np.float32)
        trace = np.full((n1 + 1, n2 + 1), "", dtype=object)

        for i in range(1, n1 + 1):
            dp[i, 0] = dp[i - 1, 0] + gap_penalty
            trace[i, 0] = "up"
        for j in range(1, n2 + 1):
            dp[0, j] = dp[0, j - 1] + gap_penalty
            trace[0, j] = "left"

        joined_student = {
            (i - k, i): "".join(student_tokens[i - k : i])
            for i in range(n1 + 1)
            for k in range(1, min(i, max_combination_len) + 1)
        }
        joined_teacher = {
            (j - k, j): "".join(teacher_tokens[j - k : j])
            for j in range(n2 + 1)
            for k in range(1, min(j, max_combination_len) + 1)
        }

        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                student_val, teacher_val = student_tokens[i - 1], teacher_tokens[j - 1]
                match_score = (
                    exact_match_score
                    if _strings_equal_flexible(student_val, teacher_val, ignore_leading_char_diff)
                    else -exact_match_score
                )
                score_diag = dp[i - 1, j - 1] + match_score
                score_up = dp[i - 1, j] + gap_penalty
                score_left = dp[i, j - 1] + gap_penalty

                max_score = score_diag
                best_move = "diag"
                if score_up > max_score:
                    max_score = score_up
                    best_move = "up"
                if score_left > max_score:
                    max_score = score_left
                    best_move = "left"

                for k in range(2, min(j + 1, max_combination_len + 1)):
                    key = (j - k, j)
                    if key in joined_teacher and _strings_equal_flexible(
                        student_val, joined_teacher[key], ignore_leading_char_diff
                    ):
                        cand = dp[i - 1, j - k] + combination_score_multiplier * k
                        if cand > max_score:
                            max_score = cand
                            best_move = f"comb_s1_over_s2_{k}"

                for k in range(2, min(i + 1, max_combination_len + 1)):
                    key = (i - k, i)
                    if key in joined_student and _strings_equal_flexible(
                        teacher_val, joined_student[key], ignore_leading_char_diff
                    ):
                        cand = dp[i - k, j - 1] + combination_score_multiplier * k
                        if cand > max_score:
                            max_score = cand
                            best_move = f"comb_s2_over_s1_{k}"

                dp[i, j] = max_score
                trace[i, j] = best_move

        aligned: List[AlignmentPair] = []
        i, j = n1, n2
        while i > 0 or j > 0:
            move = trace[i, j]
            if move == "diag":
                aligned.append(
                    AlignmentPair(
                        s_tokens=[student_tokens[i - 1]],
                        t_tokens=[teacher_tokens[j - 1]],
                        s_start=i - 1,
                        s_end=i,
                        t_start=j - 1,
                        t_end=j,
                    )
                )
                i -= 1
                j -= 1
            elif move == "up":
                aligned.append(
                    AlignmentPair(
                        s_tokens=[student_tokens[i - 1]],
                        t_tokens=[],
                        s_start=i - 1,
                        s_end=i,
                        t_start=-1,
                        t_end=-1,
                    )
                )
                i -= 1
            elif move == "left":
                aligned.append(
                    AlignmentPair(
                        s_tokens=[],
                        t_tokens=[teacher_tokens[j - 1]],
                        s_start=-1,
                        s_end=-1,
                        t_start=j - 1,
                        t_end=j,
                    )
                )
                j -= 1
            elif move.startswith("comb_s1_over_s2_"):
                k = int(move.rsplit("_", 1)[-1])
                aligned.append(
                    AlignmentPair(
                        s_tokens=[student_tokens[i - 1]],
                        t_tokens=teacher_tokens[j - k : j],
                        s_start=i - 1,
                        s_end=i,
                        t_start=j - k,
                        t_end=j,
                    )
                )
                i -= 1
                j -= k
            elif move.startswith("comb_s2_over_s1_"):
                k = int(move.rsplit("_", 1)[-1])
                aligned.append(
                    AlignmentPair(
                        s_tokens=student_tokens[i - k : i],
                        t_tokens=[teacher_tokens[j - 1]],
                        s_start=i - k,
                        s_end=i,
                        t_start=j - 1,
                        t_end=j,
                    )
                )
                i -= k
                j -= 1
            else:
                break
        aligned.reverse()
        return aligned, float(dp[n1, n2])

    @staticmethod
    def _shift_pairs(
        pairs: List[AlignmentPair], shift_s: int, shift_t: int
    ) -> List[AlignmentPair]:
        """Offset start/end indices of pairs after segment-level alignment."""
        out: List[AlignmentPair] = []
        for pair in pairs:
            ns = pair.s_start + shift_s if pair.s_start != -1 else -1
            ne = pair.s_end + shift_s if pair.s_end != -1 else -1
            nts = pair.t_start + shift_t if pair.t_start != -1 else -1
            nte = pair.t_end + shift_t if pair.t_end != -1 else -1
            # Split coarse same-token spans into 1-1 matches.
            if (
                len(pair.s_tokens) > 1
                and len(pair.s_tokens) == len(pair.t_tokens)
                and pair.s_tokens == pair.t_tokens
                and ns >= 0
                and nts >= 0
            ):
                for k in range(len(pair.s_tokens)):
                    out.append(
                        AlignmentPair(
                            s_tokens=[pair.s_tokens[k]],
                            t_tokens=[pair.t_tokens[k]],
                            s_start=ns + k,
                            s_end=ns + k + 1,
                            t_start=nts + k,
                            t_end=nts + k + 1,
                        )
                    )
            else:
                out.append(
                    AlignmentPair(
                        s_tokens=pair.s_tokens,
                        t_tokens=pair.t_tokens,
                        s_start=ns,
                        s_end=ne,
                        t_start=nts,
                        t_end=nte,
                    )
                )
        return out

    @staticmethod
    def _alignment_mask(aligned_pairs: List[AlignmentPair]) -> List[bool]:
        """Compute is_correct for each pair using canonicalized text comparison."""
        out: List[bool] = []
        for pair in aligned_pairs:
            s_canon = "".join(canonical_token(tk) for tk in pair.s_tokens) if pair.s_tokens else ""
            t_canon = "".join(canonical_token(tk) for tk in pair.t_tokens) if pair.t_tokens else ""
            out.append(_strings_equal_flexible(s_canon, t_canon, ignore_leading_char_diff=False))
        return out

    @staticmethod
    def _post_process_alignment(
        aligned_pairs: List[AlignmentPair],
        *,
        exact_match_score: float,
        combination_score_multiplier: float,
        gap_penalty: float,
        max_combination_len: int,
        end_mismatch_threshold: float = 0.2,
    ) -> List[AlignmentPair]:
        """Post-process: combine misaligned consecutive pairs and re-align bad spans.

        Mirrors ``post_process_alignment_optimized`` from the PT reference.
        """
        if not aligned_pairs:
            return []

        # Step 1: combine consecutive misaligned pairs (away from sequence end).
        pair_strings = TokenAligner._build_pair_strings(aligned_pairs)
        aligned_pairs = TokenAligner._combine_consecutive_misaligned(
            aligned_pairs, pair_strings, end_mismatch_threshold
        )
        pair_strings = TokenAligner._build_pair_strings(aligned_pairs)

        # Step 2: split exact-token coarse alignments and re-align small bad spans.
        processed: List[AlignmentPair] = []
        align_cache: dict[
            Tuple[Tuple[str, ...], Tuple[str, ...]],
            Tuple[List[AlignmentPair], bool],
        ] = {}
        i = 0
        while i < len(aligned_pairs):
            cur = aligned_pairs[i]
            if (
                len(cur.s_tokens) > 1
                and len(cur.s_tokens) == len(cur.t_tokens)
                and cur.s_tokens == cur.t_tokens
            ):
                for k in range(len(cur.s_tokens)):
                    processed.append(
                        AlignmentPair(
                            s_tokens=[cur.s_tokens[k]],
                            t_tokens=[cur.t_tokens[k]],
                            s_start=cur.s_start + k,
                            s_end=cur.s_start + k + 1,
                            t_start=cur.t_start + k,
                            t_end=cur.t_start + k + 1,
                        )
                    )
                i += 1
                continue

            bad_start = -1
            for j in range(i, len(aligned_pairs)):
                if not pair_strings[j][2]:
                    bad_start = j
                    break
            if bad_start == -1:
                processed.extend(aligned_pairs[i:])
                break
            processed.extend(aligned_pairs[i:bad_start])

            found = False
            max_chunk = min(10, len(aligned_pairs) - bad_start)
            for chunk_size in range(2, max_chunk + 1):
                chunk = aligned_pairs[bad_start : bad_start + chunk_size]
                chunk_s1, chunk_s2, s1_idx, s2_idx = TokenAligner._flatten_chunk(chunk)
                chunk_s1_str = "".join(canonical_token(t) for t in chunk_s1)
                chunk_s2_str = "".join(canonical_token(t) for t in chunk_s2)
                if not _strings_equal_flexible(
                    chunk_s1_str, chunk_s2_str, ignore_leading_char_diff=False
                ):
                    continue
                cache_key = (tuple(chunk_s1), tuple(chunk_s2))
                if cache_key in align_cache:
                    sub_pairs, perfect = align_cache[cache_key]
                else:
                    sub_pairs, _ = TokenAligner._align_dp(
                        chunk_s1,
                        chunk_s2,
                        exact_match_score=exact_match_score,
                        combination_score_multiplier=combination_score_multiplier,
                        gap_penalty=gap_penalty,
                        max_combination_len=max_combination_len,
                        ignore_leading_char_diff=False,
                    )
                    perfect = all(
                        _strings_equal_flexible(
                            "".join(canonical_token(t) for t in p.s_tokens),
                            "".join(canonical_token(t) for t in p.t_tokens),
                            ignore_leading_char_diff=False,
                        )
                        for p in sub_pairs
                    )
                    align_cache[cache_key] = (sub_pairs, perfect)

                s1_chunk_start = min(s1_idx[::2]) if s1_idx else -1
                s2_chunk_start = min(s2_idx[::2]) if s2_idx else -1
                if perfect:
                    for sub in sub_pairs:
                        ns = s1_chunk_start + sub.s_start if sub.s_start != -1 else -1
                        ne = s1_chunk_start + sub.s_end if sub.s_end != -1 else -1
                        nts = s2_chunk_start + sub.t_start if sub.t_start != -1 else -1
                        nte = s2_chunk_start + sub.t_end if sub.t_end != -1 else -1
                        processed.append(
                            AlignmentPair(
                                s_tokens=sub.s_tokens,
                                t_tokens=sub.t_tokens,
                                s_start=ns,
                                s_end=ne,
                                t_start=nts,
                                t_end=nte,
                            )
                        )
                else:
                    s1_chunk_end = max(s1_idx[1::2]) if s1_idx else -1
                    s2_chunk_end = max(s2_idx[1::2]) if s2_idx else -1
                    processed.append(
                        AlignmentPair(
                            s_tokens=chunk_s1,
                            t_tokens=chunk_s2,
                            s_start=s1_chunk_start,
                            s_end=s1_chunk_end,
                            t_start=s2_chunk_start,
                            t_end=s2_chunk_end,
                        )
                    )
                i = bad_start + chunk_size
                found = True
                break
            if not found:
                processed.append(aligned_pairs[bad_start])
                i = bad_start + 1
        return processed

    @staticmethod
    def _build_pair_strings(
        aligned_pairs: List[AlignmentPair],
    ) -> List[Tuple[str, str, bool]]:
        """Precompute (s_str, t_str, is_match) for each pair."""
        out: List[Tuple[str, str, bool]] = []
        for pair in aligned_pairs:
            s_canon = "".join(canonical_token(t) for t in pair.s_tokens) if pair.s_tokens else ""
            t_canon = "".join(canonical_token(t) for t in pair.t_tokens) if pair.t_tokens else ""
            is_match = _strings_equal_flexible(s_canon, t_canon, ignore_leading_char_diff=False)
            out.append((s_canon, t_canon, is_match))
        return out

    @staticmethod
    def _combine_consecutive_misaligned(
        aligned_pairs: List[AlignmentPair],
        pair_strings: List[Tuple[str, str, bool]],
        end_mismatch_threshold: float,
    ) -> List[AlignmentPair]:
        """Combine consecutive misaligned pairs into single multi-token chunks."""
        if not aligned_pairs or len(aligned_pairs) < 2:
            return aligned_pairs
        end_boundary = int(len(aligned_pairs) * (1 - end_mismatch_threshold))
        out: List[AlignmentPair] = []
        i = 0
        while i < len(aligned_pairs):
            if (
                i < end_boundary
                and not pair_strings[i][2]
                and i + 1 < len(aligned_pairs)
            ):
                run = [i]
                j = i + 1
                while (
                    j < end_boundary
                    and j < len(aligned_pairs)
                    and not pair_strings[j][2]
                ):
                    run.append(j)
                    j += 1
                if len(run) >= 2:
                    combined_s1: List[str] = []
                    combined_s2: List[str] = []
                    s1_idx: List[int] = []
                    s2_idx: List[int] = []
                    for idx in run:
                        pair = aligned_pairs[idx]
                        combined_s1.extend(pair.s_tokens)
                        combined_s2.extend(pair.t_tokens)
                        if pair.s_tokens and pair.s_start != -1:
                            s1_idx.extend([pair.s_start, pair.s_end])
                        if pair.t_tokens and pair.t_start != -1:
                            s2_idx.extend([pair.t_start, pair.t_end])
                    cs1_start = min(s1_idx[::2]) if s1_idx else -1
                    cs1_end = max(s1_idx[1::2]) if s1_idx else -1
                    cs2_start = min(s2_idx[::2]) if s2_idx else -1
                    cs2_end = max(s2_idx[1::2]) if s2_idx else -1
                    out.append(
                        AlignmentPair(
                            s_tokens=combined_s1,
                            t_tokens=combined_s2,
                            s_start=cs1_start,
                            s_end=cs1_end,
                            t_start=cs2_start,
                            t_end=cs2_end,
                        )
                    )
                    i = j
                    continue
            out.append(aligned_pairs[i])
            i += 1
        return out

    @staticmethod
    def _flatten_chunk(
        chunk: List[AlignmentPair],
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Concatenate tokens and collect span indices across a chunk of pairs."""
        chunk_s1: List[str] = []
        chunk_s2: List[str] = []
        s1_idx: List[int] = []
        s2_idx: List[int] = []
        for pair in chunk:
            chunk_s1.extend(pair.s_tokens)
            chunk_s2.extend(pair.t_tokens)
            if pair.s_tokens:
                s1_idx.extend([pair.s_start, pair.s_end])
            if pair.t_tokens:
                s2_idx.extend([pair.t_start, pair.t_end])
        return chunk_s1, chunk_s2, s1_idx, s2_idx


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
        ``range(len(seq))`` — required so that DP-output indices over
        ``canon`` can be remapped to positions on the original input-id
        axis (see :meth:`TokenAligner._remap_pairs_to_original`).
    """
    merged, ranges = _merge_encoding_artifacts(seq)
    canon = [canonical_token(t) for t in merged]
    return _merge_consecutive_bytes(canon, ranges)


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
    i = 0
    while i < len(tokens):
        matched = False
        for pattern, replacement in _MULTI_TOKEN_ARTIFACT_FIXES:
            pl = len(pattern)
            if i + pl <= len(tokens) and tokens[i : i + pl] == pattern:
                # Every fix in _MULTI_TOKEN_ARTIFACT_FIXES is N→1; the
                # remap relies on that to attach the matched original
                # range to a single output token.
                assert len(replacement) == 1, (
                    "Multi-token artifact fix replacement must be a single "
                    f"token; got {replacement!r}"
                )
                result.extend(replacement)
                ranges.append((i, i + pl))
                i += pl
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            ranges.append((i, i + 1))
            i += 1
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
        merged, merged_ranges = _try_merge_byte_buffer(
            byte_buffer, byte_buffer_ranges
        )
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

    if len(raw_bytes) < 2 or len(raw_bytes) > 4:
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


def _strings_equal_flexible(s1: str, s2: str, ignore_leading_char_diff: bool) -> bool:
    """Compare two strings, optionally after canonicalization."""
    if not ignore_leading_char_diff:
        return s1 == s2
    return canonical_token(s1) == canonical_token(s2)


