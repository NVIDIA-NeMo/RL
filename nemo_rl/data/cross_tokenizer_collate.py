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
"""Collator that tokenizes raw text twice (student + teacher) and aligns.

The collator runs inside DataLoader worker processes. It does:

1. Tokenizes the same source text once with the student tokenizer and once
   with the teacher tokenizer (no chat template, no special handling).
2. Calls :class:`TokenAligner.align` to produce a dense-padded
   :class:`AlignmentBatch` covering all three loss modes (P-KL, gold_loss,
   xtoken_loss).
3. Returns a :class:`BatchedDataDict` with the keys
   :class:`Policy.train` expects (``input_ids``, ``input_lengths``,
   ``token_mask``, ``sample_mask``) plus teacher tensors and alignment
   tensors.

Loss-side projection-matrix work happens inside the loss fn; nothing related
to KL/CE math runs here.
"""

from __future__ import annotations

from typing import Any, List

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class CrossTokenizerCollator:
    """Tokenize twice, align once, return a flat tensor batch.

    Args:
        student_tokenizer: HF tokenizer matching the student model.
        teacher_tokenizer: HF tokenizer matching the teacher model.
        aligner: Pre-constructed :class:`TokenAligner`.
        ctx_length_student: Hard tokenization length cap on the student
            side (also the padded sequence length of the student tensor).
        ctx_length_teacher: Same on the teacher side.
        make_seq_div_by_student: Round student sequence length up to a
            multiple of this value (typically TP * CP * 2 for DTensor V2).
        make_seq_div_by_teacher: Same for the teacher side.
    """

    def __init__(
        self,
        *,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
        aligner: TokenAligner,
        ctx_length_student: int,
        ctx_length_teacher: int,
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teacher: int = 1,
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.aligner = aligner
        self.ctx_length_student = ctx_length_student
        self.ctx_length_teacher = ctx_length_teacher
        self.make_seq_div_by_student = make_seq_div_by_student
        self.make_seq_div_by_teacher = make_seq_div_by_teacher
        # Downstream consumers assume real tokens occupy the leading
        # positions: ``input_lengths = attention_mask.sum(-1)`` plus the
        # ``[:length]`` slices in the policy forward and the token-chunk
        # alignment all treat ``input_ids[:, :length]`` as the content.
        # Pin right-padding rather than trust each tokenizer's default
        # (some tokenizer configs default to left-padding, which would
        # silently misalign without changing the lengths).
        self.student_tokenizer.padding_side = "right"
        self.teacher_tokenizer.padding_side = "right"
        # Defensive: HF tokenizers without a pad token can't pad batches.
        if self.student_tokenizer.pad_token_id is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        if self.teacher_tokenizer.pad_token_id is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

    def __call__(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        # kd_data_processor carries the raw text as a single assistant
        # message; the collator tokenizes that content for both sides.
        texts = [datum["message_log"][0]["content"] for datum in batch]
        student_input_ids, student_attention_mask = self._tokenize_batch(
            texts,
            self.student_tokenizer,
            self.ctx_length_student,
            self.make_seq_div_by_student,
        )
        teacher_input_ids, teacher_attention_mask = self._tokenize_batch(
            texts,
            self.teacher_tokenizer,
            self.ctx_length_teacher,
            self.make_seq_div_by_teacher,
        )
        alignment = self.aligner.align(
            student_input_ids,
            teacher_input_ids,
            student_attention_mask=student_attention_mask,
            teacher_attention_mask=teacher_attention_mask,
        )

        sample_mask = torch.tensor(
            [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
        )
        idx = [datum["idx"] for datum in batch]

        return BatchedDataDict(
            # Student-side keys map onto Policy.train's expected names.
            input_ids=student_input_ids,
            input_lengths=student_attention_mask.sum(dim=-1).long(),
            token_mask=student_attention_mask.long(),
            sample_mask=sample_mask,
            # Teacher-side keys travel with the batch for the teacher
            # forward pass in the trainer.
            teacher_input_ids=teacher_input_ids,
            teacher_input_lengths=teacher_attention_mask.sum(dim=-1).long(),
            teacher_token_mask=teacher_attention_mask.long(),
            # Alignment payload, dense-padded so DTensor V2 can shard on dim 0.
            alignment_pair_valid=alignment.pair_valid,
            alignment_pair_is_correct=alignment.pair_is_correct,
            alignment_student_exact_partition_mask=(
                alignment.student_exact_partition_mask
            ),
            alignment_teacher_exact_partition_mask=(
                alignment.teacher_exact_partition_mask
            ),
            alignment_student_chunk_id=alignment.student_chunk_id,
            alignment_teacher_chunk_id=alignment.teacher_chunk_id,
            alignment_num_chunks=alignment.num_chunks,
            idx=idx,
        )

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
