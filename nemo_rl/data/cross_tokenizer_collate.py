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

from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


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
    ):
        n = len(aligners)
        assert len(teacher_tokenizers) == n and len(ctx_length_teachers) == n, (
            "teacher_tokenizers, aligners, and ctx_length_teachers must all "
            f"have length == num_teachers ({n})."
        )
        if make_seq_div_by_teachers is None:
            make_seq_div_by_teachers = [1] * n
        assert len(make_seq_div_by_teachers) == n
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.aligners = aligners
        self.ctx_length_student = ctx_length_student
        self.ctx_length_teachers = ctx_length_teachers
        self.make_seq_div_by_student = make_seq_div_by_student
        self.make_seq_div_by_teachers = make_seq_div_by_teachers
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
        # kd_data_processor carries the raw text as a single assistant
        # message; the collator tokenizes that content for the student and
        # each cross-tokenizer teacher.
        texts = [datum["message_log"][0]["content"] for datum in batch]
        student_input_ids, student_attention_mask = self._tokenize_batch(
            texts,
            self.student_tokenizer,
            self.ctx_length_student,
            self.make_seq_div_by_student,
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
            teacher_input_ids, teacher_attention_mask = self._tokenize_batch(
                texts,
                self.teacher_tokenizers[i],
                self.ctx_length_teachers[i],
                self.make_seq_div_by_teachers[i],
            )
            alignment = aligner.align(
                student_input_ids,
                teacher_input_ids,
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

    @staticmethod
    def _tokenize_batch(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        make_seq_div_by: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch and pad to a multiple of ``make_seq_div_by``."""
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=ctx_length,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoded["input_ids"]
        attention_mask: torch.Tensor = encoded["attention_mask"]

        b, t = input_ids.shape
        pad = (make_seq_div_by - (t % make_seq_div_by)) % make_seq_div_by
        if pad > 0:
            pad_ids = torch.full(
                (b, pad),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
            )
            pad_mask = torch.zeros((b, pad), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        return input_ids, attention_mask
