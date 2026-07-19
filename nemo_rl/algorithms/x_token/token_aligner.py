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
      matrix, exposes :meth:`align` for the collator.
    - :func:`align_by_offsets_cluster` — the single-sample offset alignment
      kernel, also usable directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import torch

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
    ``_align_single`` then fills in ``is_correct`` from the canonicalized-text
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
        is_correct: ``True`` when the canonicalized student span text
            matches the canonicalized teacher span text. Defaults to
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
                to the ``chunk_id = -1`` / partition-``False`` sentinels so
                tokenizer padding never forms a valid chunk.
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

    @staticmethod
    def _drop_padding(
        batch: AlignmentBatch,
        *,
        student_attention_mask: torch.Tensor | None,
        teacher_attention_mask: torch.Tensor | None,
    ) -> None:
        """Strip tokenizer padding out of the chunk-id / partition tensors.

        Mutates ``batch`` in place. For every position the attention mask
        marks as padding, reset ``*_chunk_id`` to ``-1`` and
        ``*_exact_partition_mask`` to ``False``. Gating per position (rather
        than trimming a contiguous span) keeps this correct under either
        left- or right-padding. A pair whose tokens are entirely padding on
        one side then has size 0 there and is dropped by
        :func:`nemo_rl.algorithms.x_token.loss_utils.valid_chunk_mask`; a pair
        straddling the real/pad boundary shrinks to its real tokens.
        """
        if student_attention_mask is not None:
            s_pad = student_attention_mask == 0
            batch.student_chunk_id[s_pad] = -1
            batch.student_exact_partition_mask[s_pad] = False
        if teacher_attention_mask is not None:
            t_pad = teacher_attention_mask == 0
            batch.teacher_chunk_id[t_pad] = -1
            batch.teacher_exact_partition_mask[t_pad] = False

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
    # Per-sample alignment
    # ------------------------------------------------------------------ #
    def _align_single(
        self,
        student_ids: List[int],
        teacher_ids: List[int],
        student_offsets: List[Tuple[int, int]],
        teacher_offsets: List[Tuple[int, int]],
    ) -> List[AlignmentPair]:
        """Offset-cluster alignment for one sample.

        Runs :func:`align_by_offsets_cluster`, then recomputes ``is_correct``
        via the canonicalized-text mask so ``pair_is_correct`` and the
        gold_loss exact partition reflect true token-text equality rather than
        mere co-location: a paired-but-different-text group is downgraded to
        ``is_correct=False`` and orphan groups stay ``False``.

        Returns:
            A list of :class:`AlignmentPair`. Insertions/deletions use ``-1``
            for the empty side's start/end. Pair start/end indices address the
            token sequences directly, so they can be written straight into the
            chunk-id tensors in :meth:`_pairs_to_batch`.
        """
        raw_pairs = align_by_offsets_cluster(
            student_ids,
            student_offsets,
            self.student_tokenizer,
            teacher_ids,
            teacher_offsets,
            self.teacher_tokenizer,
        )
        pairs = [
            AlignmentPair(
                s_tokens=s_toks,
                t_tokens=t_toks,
                s_start=s_start,
                s_end=s_end,
                t_start=t_start,
                t_end=t_end,
            )
            for (
                s_toks,
                t_toks,
                s_start,
                s_end,
                t_start,
                t_end,
                _paired,
            ) in raw_pairs
        ]
        for pair, m in zip(pairs, self._alignment_mask(pairs)):
            pair.is_correct = m
        return pairs

    @staticmethod
    def _alignment_mask(aligned_pairs: List[AlignmentPair]) -> List[bool]:
        """Compute is_correct for each pair using canonicalized text comparison."""
        out: List[bool] = []
        for pair in aligned_pairs:
            s_canon = (
                "".join(canonical_token(tk) for tk in pair.s_tokens)
                if pair.s_tokens
                else ""
            )
            t_canon = (
                "".join(canonical_token(tk) for tk in pair.t_tokens)
                if pair.t_tokens
                else ""
            )
            out.append(
                _strings_equal_flexible(
                    s_canon, t_canon, ignore_leading_char_diff=False
                )
            )
        return out


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


def _strings_equal_flexible(s1: str, s2: str, ignore_leading_char_diff: bool) -> bool:
    """Compare two strings, optionally after canonicalization."""
    if not ignore_leading_char_diff:
        return s1 == s2
    return canonical_token(s1) == canonical_token(s2)


# =====================================================================
# Offset-based alignment kernel (cluster + strict char-end walker).
# =====================================================================


# When one tokenizer has pad_token_id == eos_token_id (e.g. Llama-3.2 with the
# collator fallback that sets pad_token = eos_token if undefined), trailing pad
# positions get role "eos"; the other tokenizer with a separate pad_token_id
# tags its trailing pads as "pad". Treating these roles as equivalent pairs
# those positions 1<->1, giving the student KL signal at pad positions.
# Without this set, role-based pairing silently drops the mismatched roles and
# the student loses training signal at every trailing pad on the mismatched
# side.
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
    s_off_tuples = [tuple(o) for o in student_offsets]
    t_off_tuples = [tuple(o) for o in teacher_offsets]

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

    student_tokens_str = student_tokenizer.convert_ids_to_tokens(student_ids)
    teacher_tokens_str = teacher_tokenizer.convert_ids_to_tokens(teacher_ids)

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
