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
"""Collator that tokenizes the student once, then tokenizes+aligns each teacher.

The collator runs inside DataLoader worker processes. It does:

1. Tokenizes the source text once with the student tokenizer (no chat
   template, no special handling); this tokenization is shared by all
   teachers.
2. For each *cross-tokenizer* teacher, tokenizes with that teacher's
   tokenizer and calls :class:`TokenAligner.align` to produce a dense-padded
   :class:`AlignmentBatch` (P-KL, gold_loss, xtoken_loss), emitted under
   teacher-indexed keys ``teacher_{i}_*`` / ``alignment_{i}_*``.
3. *Same-tokenizer* teachers (``aligners[i] is None``) emit nothing extra —
   their forward reuses the student tokenization, so projection and alignment
   are skipped.
4. Returns a :class:`BatchedDataDict` with the keys :class:`Policy.train`
   expects (``input_ids``, ``input_lengths``, ``token_mask``,
   ``sample_mask``) plus per-teacher tensors and alignment tensors.

Loss-side projection-matrix work happens inside the loss fn; nothing related
to KL/CE math runs here.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, List, Optional

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import AlignmentPair, TokenAligner
from nemo_rl.data.chat_templates import find_rendered_message_content_span
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_CHAT_ALIGNMENT_METHODS = {"offset_cluster_decode_fix", "same_tokenizer_identity"}


class CrossTokenizerCollator:
    """Tokenize the student once, tokenize+align each teacher, return a flat batch.

    Supports N teachers. The student text is tokenized once and shared; each
    cross-tokenizer teacher is tokenized with its own tokenizer and aligned
    with its own :class:`TokenAligner`, emitting teacher-indexed keys
    (``teacher_{i}_*`` and ``alignment_{i}_*``). A *same-tokenizer* teacher
    (``aligners[i] is None``) emits nothing extra — its forward reuses the
    student tokenization, so projection and alignment are skipped entirely.

    Args:
        student_tokenizer: HF tokenizer matching the student model.
        teacher_tokenizers: Per-teacher HF tokenizers. May be ``None`` for a
            same-tokenizer teacher (its tokenization is the student's).
        aligners: Per-teacher :class:`TokenAligner`. ``None`` marks a
            same-tokenizer teacher (no projection / no alignment).
        ctx_length_student: Hard tokenization length cap on the student
            side (also the padded sequence length of the student tensor).
        ctx_length_teachers: Per-teacher tokenization length caps.
        make_seq_div_by_student: Round student sequence length up to a
            multiple of this value (typically TP * CP * 2 for DTensor V2).
        make_seq_div_by_teachers: Per-teacher sequence-length divisors.
    """

    def __init__(
        self,
        *,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: List[Optional[PreTrainedTokenizerBase]],
        aligners: List[Optional[TokenAligner]],
        ctx_length_student: int,
        ctx_length_teachers: List[int],
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teachers: Optional[List[int]] = None,
        mode: str = "text",
        drop_first_assistant_chunk_kl: bool = False,
        include_thinking_in_loss: bool = False,
        native_thinking_alignment: bool = False,
        kd_alignment_regions: Optional[List[str]] = None,
        num_packed_rows: int = 1,
    ):
        n = len(aligners)
        assert len(teacher_tokenizers) == n and len(ctx_length_teachers) == n, (
            "teacher_tokenizers, aligners, and ctx_length_teachers must all "
            f"have length == num_teachers ({n})."
        )
        if make_seq_div_by_teachers is None:
            make_seq_div_by_teachers = [1] * n
        assert len(make_seq_div_by_teachers) == n
        if mode not in ("text", "chat"):
            raise ValueError(f"mode must be 'text' or 'chat', got {mode!r}")
        if mode == "chat":
            for i, aligner in enumerate(aligners):
                if aligner is None:
                    continue
                if aligner.alignment_method not in _CHAT_ALIGNMENT_METHODS:
                    raise ValueError(
                        f"mode='chat' requires an offset-aware alignment method; "
                        f"teacher {i} has {aligner.alignment_method!r}, expected "
                        f"one of {sorted(_CHAT_ALIGNMENT_METHODS)!r}."
                    )
                if not getattr(student_tokenizer, "is_fast", False) or not getattr(
                    teacher_tokenizers[i], "is_fast", False
                ):
                    raise ValueError(
                        "mode='chat' requires fast student/teacher tokenizers for "
                        "return_offsets_mapping=True."
                    )
        if native_thinking_alignment and (
            mode != "chat" or not include_thinking_in_loss
        ):
            raise ValueError(
                "native_thinking_alignment requires mode='chat' and "
                "include_thinking_in_loss=true."
            )
        if kd_alignment_regions is not None and not native_thinking_alignment:
            raise ValueError(
                "kd_alignment_regions requires native_thinking_alignment=true."
            )
        # Native-thinking semantic-region parsing and multi-doc lockstep packing
        # are not yet wired in this collator; fail loudly rather than silently
        # produce whole-message alignment when they were requested.
        if native_thinking_alignment:
            raise NotImplementedError(
                "native_thinking_alignment is not yet implemented in this "
                "collator (whole-message chat alignment only); this is a "
                "planned follow-up."
            )
        if num_packed_rows != 1:
            raise NotImplementedError(
                "num_packed_rows > 1 (lockstep packing) is not yet implemented "
                "in this collator; use num_packed_rows=1."
            )
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.aligners = aligners
        self.ctx_length_student = ctx_length_student
        self.ctx_length_teachers = ctx_length_teachers
        self.make_seq_div_by_student = make_seq_div_by_student
        self.make_seq_div_by_teachers = make_seq_div_by_teachers
        self.mode = mode
        self.drop_first_assistant_chunk_kl = drop_first_assistant_chunk_kl
        self.include_thinking_in_loss = include_thinking_in_loss
        # Downstream consumers assume real tokens occupy the leading
        # positions: ``input_lengths = attention_mask.sum(-1)`` plus the
        # ``[:length]`` slices in the policy forward and the token-chunk
        # alignment all treat ``input_ids[:, :length]`` as the content.
        # Pin right-padding rather than trust each tokenizer's default
        # (some tokenizer configs default to left-padding, which would
        # silently misalign without changing the lengths).
        self.student_tokenizer.padding_side = "right"
        if self.student_tokenizer.pad_token_id is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        # Same pinning for each cross-tokenizer teacher tokenizer.
        for i, tok in enumerate(self.teacher_tokenizers):
            if self.aligners[i] is None or tok is None:
                continue
            tok.padding_side = "right"
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

    def __call__(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        if self.mode == "chat":
            return self._call_chat(batch)
        return self._call_text(batch)

    def _call_text(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        # kd_data_processor carries the raw text as a single assistant
        # message; the collator tokenizes that content for the student and
        # each cross-tokenizer teacher.
        texts = [datum["message_log"][0]["content"] for datum in batch]
        student_input_ids, student_attention_mask, student_offsets = (
            self._tokenize_batch(
                texts,
                self.student_tokenizer,
                self.ctx_length_student,
                self.make_seq_div_by_student,
            )
        )

        sample_mask = torch.tensor(
            [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
        )
        idx = [datum["idx"] for datum in batch]

        out: dict[str, Any] = {
            # Student-side keys map onto Policy.train's expected names. A
            # single student tokenization is shared across all teachers.
            "input_ids": student_input_ids,
            "input_lengths": student_attention_mask.sum(dim=-1).long(),
            "token_mask": student_attention_mask.long(),
            "sample_mask": sample_mask,
            "idx": idx,
        }

        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                # Same-tokenizer teacher: no re-tokenization, no projection,
                # no alignment. Its forward reuses the student tokenization.
                continue
            teacher_input_ids, teacher_attention_mask, teacher_offsets = (
                self._tokenize_batch(
                    texts,
                    self.teacher_tokenizers[i],
                    self.ctx_length_teachers[i],
                    self.make_seq_div_by_teachers[i],
                )
            )
            alignment = aligner.align(
                student_input_ids,
                teacher_input_ids,
                student_offsets=student_offsets,
                teacher_offsets=teacher_offsets,
                student_attention_mask=student_attention_mask,
                teacher_attention_mask=teacher_attention_mask,
            )
            # Teacher-side keys travel with the batch for the teacher forward.
            out[f"teacher_{i}_input_ids"] = teacher_input_ids
            out[f"teacher_{i}_input_lengths"] = teacher_attention_mask.sum(
                dim=-1
            ).long()
            out[f"teacher_{i}_token_mask"] = teacher_attention_mask.long()
            # Alignment payload, dense-padded so DTensor V2 can shard on dim 0.
            # Keys are driven off AlignmentBatch fields so they can't drift
            # from `alignment_from_flat_batch(data, prefix=f"alignment_{i}_")`.
            for f in dataclass_fields(alignment):
                out[f"alignment_{i}_{f.name}"] = getattr(alignment, f.name)

        return BatchedDataDict(out)

    def _call_chat(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        """Chat/instruct path: align each assistant message independently.

        Renders each side's chat template, then calls
        :meth:`TokenAligner.align_one_offset_per_asst` per assistant message.
        Loss fires only on assistant content (``token_mask = attention_mask *
        assistant_mask``). Same-tokenizer teachers (``aligners[i] is None``)
        reuse the student tokenization, as in the text path.
        """
        b = len(batch)
        s_ids, s_off, s_mask, s_spans = [], [], [], []
        for datum in batch:
            ids, off, mask, spans = self._render_and_tokenize_chat(
                self.student_tokenizer,
                datum["message_log"],
                self.ctx_length_student,
            )
            s_ids.append(ids)
            s_off.append(off)
            s_mask.append(mask)
            s_spans.append(spans)

        student_input_ids, student_attention_mask, _, student_asst_mask = (
            self._pad_chat_batch(
                s_ids,
                s_off,
                s_mask,
                self.student_tokenizer.pad_token_id,
                self.make_seq_div_by_student,
            )
        )
        t_s = student_input_ids.shape[1]

        out: dict[str, Any] = {
            "input_ids": student_input_ids,
            "input_lengths": student_attention_mask.sum(dim=-1).long(),
            # Loss only on assistant-content tokens.
            "token_mask": (student_attention_mask * student_asst_mask).long(),
            "sample_mask": torch.tensor(
                [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
            ),
            "idx": [datum["idx"] for datum in batch],
        }

        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                continue
            t_ids, t_off, t_mask, t_spans = [], [], [], []
            for datum in batch:
                ids, off, mask, spans = self._render_and_tokenize_chat(
                    self.teacher_tokenizers[i],
                    datum["message_log"],
                    self.ctx_length_teachers[i],
                )
                t_ids.append(ids)
                t_off.append(off)
                t_mask.append(mask)
                t_spans.append(spans)

            teacher_input_ids, teacher_attention_mask, _, _ = self._pad_chat_batch(
                t_ids,
                t_off,
                t_mask,
                self.teacher_tokenizers[i].pad_token_id,
                self.make_seq_div_by_teachers[i],
            )
            t_t = teacher_input_ids.shape[1]

            per_sample_pairs: List[List[AlignmentPair]] = []
            for j in range(b):
                raw = aligner.align_one_offset_per_asst(
                    s_ids[j],
                    s_off[j],
                    s_spans[j],
                    t_ids[j],
                    t_off[j],
                    t_spans[j],
                    student_asst_mask=s_mask[j],
                    teacher_asst_mask=t_mask[j],
                    drop_first_content_pair=self.drop_first_assistant_chunk_kl,
                )
                per_sample_pairs.append(
                    [
                        AlignmentPair(p[0], p[1], p[2], p[3], p[4], p[5], p[6])
                        for p in raw
                    ]
                )

            alignment = aligner._pairs_to_batch(per_sample_pairs, b=b, t_s=t_s, t_t=t_t)
            aligner._drop_padding(
                alignment,
                student_attention_mask=student_attention_mask,
                teacher_attention_mask=teacher_attention_mask,
            )
            out[f"teacher_{i}_input_ids"] = teacher_input_ids
            out[f"teacher_{i}_input_lengths"] = teacher_attention_mask.sum(
                dim=-1
            ).long()
            out[f"teacher_{i}_token_mask"] = teacher_attention_mask.long()
            for f in dataclass_fields(alignment):
                out[f"alignment_{i}_{f.name}"] = getattr(alignment, f.name)

        return BatchedDataDict(out)

    @staticmethod
    def _render_and_tokenize_chat(
        tokenizer: PreTrainedTokenizerBase,
        messages: List[dict],
        ctx_length: int,
    ) -> tuple[List[int], List[tuple[int, int]], List[int], List[tuple[int, int]]]:
        """Render + tokenize one conversation and derive its assistant spans.

        Renders with the tokenizer's chat template, tokenizes with char offsets,
        and derives the per-token assistant mask and each assistant message's
        char span. Returns ``(input_ids, offsets, assistant_mask,
        assistant_char_spans)`` for a single (unpadded) sample. The rendered
        string already carries the template's special tokens, so tokenization
        uses ``add_special_tokens=False``.
        """
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        encoded = tokenizer(
            rendered,
            truncation=True,
            max_length=ctx_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offsets = [tuple(o) for o in encoded["offset_mapping"]]

        # Locate each assistant message's content in the rendered text, scanning
        # left-to-right so repeated content in later turns doesn't rematch early.
        asst_char_spans: List[tuple[int, int]] = []
        cursor = 0
        for message in messages:
            span = find_rendered_message_content_span(
                rendered, message.get("content", ""), cursor
            )
            if span is None:
                continue
            cursor = span[1]
            if message.get("role") == "assistant":
                asst_char_spans.append((span[0], span[1]))

        assistant_mask = [
            1
            if any(s <= cs and ce <= e and ce > cs for (s, e) in asst_char_spans)
            else 0
            for (cs, ce) in offsets
        ]
        return input_ids, offsets, assistant_mask, asst_char_spans

    @staticmethod
    def _pad_chat_batch(
        ids_list: List[List[int]],
        off_list: List[List[tuple[int, int]]],
        mask_list: List[List[int]],
        pad_token_id: int,
        make_seq_div_by: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad per-sample chat tokenizations into dense ``[B, T]`` tensors.

        Pads to the batch-max length rounded up to ``make_seq_div_by``. Padding
        positions get ``attention_mask=0``, ``offset=(0, 0)``,
        ``assistant_mask=0``.
        """
        b = len(ids_list)
        max_len = max((len(ids) for ids in ids_list), default=0)
        if make_seq_div_by > 1 and max_len % make_seq_div_by:
            max_len += make_seq_div_by - (max_len % make_seq_div_by)
        max_len = max(max_len, make_seq_div_by, 1)

        input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        offsets = torch.zeros((b, max_len, 2), dtype=torch.long)
        assistant_mask = torch.zeros((b, max_len), dtype=torch.long)
        for j, (ids, off, mask) in enumerate(zip(ids_list, off_list, mask_list)):
            n = len(ids)
            input_ids[j, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, :n] = 1
            if n:
                offsets[j, :n] = torch.tensor(off, dtype=torch.long)
                assistant_mask[j, :n] = torch.tensor(mask, dtype=torch.long)
        return input_ids, attention_mask, offsets, assistant_mask

    @staticmethod
    def _tokenize_batch(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        make_seq_div_by: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a batch and pad to a multiple of ``make_seq_div_by``.

        Also returns per-token character offsets (``offset_mapping``), which
        :meth:`TokenAligner.align` needs to align the student and teacher
        tokenizations of the same source text. This requires a *fast* HF
        tokenizer; special and padding positions carry ``(0, 0)``.
        """
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=ctx_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoded["input_ids"]
        attention_mask: torch.Tensor = encoded["attention_mask"]
        offset_mapping: torch.Tensor = encoded["offset_mapping"]

        b, t = input_ids.shape
        pad = (make_seq_div_by - (t % make_seq_div_by)) % make_seq_div_by
        if pad > 0:
            pad_ids = torch.full(
                (b, pad),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
            )
            pad_mask = torch.zeros((b, pad), dtype=attention_mask.dtype)
            pad_offsets = torch.zeros((b, pad, 2), dtype=offset_mapping.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
            offset_mapping = torch.cat([offset_mapping, pad_offsets], dim=1)

        return input_ids, attention_mask, offset_mapping
