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

1. Tokenizes the student input once; this tokenization is shared by all
   teachers. In ``mode="text"``, tokenizes raw text without a chat template.
   In ``mode="chat"``, renders the student's chat template and identifies
   assistant content and end-of-turn tokens for the loss mask.
2. For each *cross-tokenizer* teacher, tokenizes with that teacher's
   tokenizer and aligns with its :class:`TokenAligner`. Text mode aligns
   the full text; chat mode renders the teacher's own chat template and
   aligns assistant messages independently. Teacher scoring masks also
   select assistant content and end-of-turn tokens in chat mode. Dense-padded alignment and
   teacher inputs are emitted under ``alignment_{i}_*`` / ``teacher_{i}_*``.
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
from typing import Any, List, Literal, Optional

import torch
from pydantic import BaseModel, PositiveInt
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.native_chat import RenderedChatDocument, _render_and_tokenize_chat
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class CrossTokenizerCollatorConfig(BaseModel, extra="allow"):
    """Shared batching options for cross-tokenizer distillation.

    Attributes:
        mode: ``"text"`` tokenizes raw text; ``"chat"`` renders each model's
            chat template and supervises assistant turns.
        include_thinking_in_loss: Include reasoning text and its closing marker
            in assistant supervision, excluding the opening scaffold. Separate
            ``reasoning_content`` fields require native thinking alignment;
            ordinary chat supports inline ``<think>`` content.
        native_thinking_alignment: Align reasoning, close, answer/tool and EOT
            independently on supported Qwen/Nano ChatML templates.
            Requires chat mode and thinking loss.
        kd_alignment_regions: Optional subset of reasoning, close, answer,
            and eot regions. Requires native thinking alignment.
        num_packed_rows: Positive number of examples packed into each row.
            Only one is currently supported.
    """

    mode: Literal["text", "chat"] = "text"
    include_thinking_in_loss: bool = False
    native_thinking_alignment: bool = False
    kd_alignment_regions: (
        list[Literal["reasoning", "close", "answer", "eot"]] | None
    ) = None
    num_packed_rows: PositiveInt = 1


class CrossTokenizerCollator:
    """Tokenize the student once, tokenize+align each teacher, return a flat batch.

    Supports N teachers in raw-text or chat mode. Student inputs are tokenized
    once and shared; each cross-tokenizer teacher uses its own tokenizer and
    :class:`TokenAligner`, emitting teacher-indexed keys
    (``teacher_{i}_*`` and ``alignment_{i}_*``). A *same-tokenizer* teacher
    (``aligners[i] is None``) emits nothing extra — its forward reuses the
    student tokenization, so projection and alignment are skipped entirely.
    Chat mode applies each model's chat template, masks loss and teacher
    scoring to assistant content and end-of-turn tokens, and aligns assistant
    messages separately.

    Args:
        config: Typed batching options, including text/chat mode and the
            thinking-region and packing settings.
        student_tokenizer: HF tokenizer matching the student model.
        teacher_tokenizers: Per-teacher HF tokenizers. May be ``None`` for a
            same-tokenizer teacher (its tokenization is the student's).
        aligners: Per-teacher :class:`TokenAligner`. ``None`` marks a
            same-tokenizer teacher (no projection / no alignment).
        ctx_length_student: Hard tokenization length cap on the student
            side. Text mode pads to this cap; chat mode pads to the batch's
            longest tokenization. Sequence-length divisors may add padding.
        ctx_length_teachers: Per-teacher tokenization length caps.
        drop_first_assistant_chunk_kl_by_teacher: Per-teacher flags controlling
            whether chat alignment drops the first content pair in each
            assistant message. The list retains a slot for same-tokenizer
            teachers so its indices match ``aligners``.
        make_seq_div_by_student: Round student sequence length up to a
            multiple of this value (typically TP * CP * 2 for DTensor V2).
        make_seq_div_by_teachers: Per-teacher sequence-length divisors.
    """

    def __init__(
        self,
        *,
        config: CrossTokenizerCollatorConfig,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: List[Optional[PreTrainedTokenizerBase]],
        aligners: List[Optional[TokenAligner]],
        ctx_length_student: int,
        ctx_length_teachers: List[int],
        drop_first_assistant_chunk_kl_by_teacher: List[bool],
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teachers: Optional[List[int]] = None,
    ) -> None:
        n = len(aligners)
        assert len(teacher_tokenizers) == n and len(ctx_length_teachers) == n, (
            "teacher_tokenizers, aligners, and ctx_length_teachers must all "
            f"have length == num_teachers ({n})."
        )
        if len(drop_first_assistant_chunk_kl_by_teacher) != n:
            raise ValueError(
                "drop_first_assistant_chunk_kl_by_teacher must have length "
                f"== num_teachers ({n}), got "
                f"{len(drop_first_assistant_chunk_kl_by_teacher)}."
            )
        if make_seq_div_by_teachers is None:
            make_seq_div_by_teachers = [1] * n
        assert len(make_seq_div_by_teachers) == n
        if config.mode == "chat":
            if not student_tokenizer.is_fast:
                raise ValueError("mode='chat' requires a fast student tokenizer")
            for i, aligner in enumerate(aligners):
                if aligner is None:
                    continue
                if not getattr(student_tokenizer, "is_fast", False) or not getattr(
                    teacher_tokenizers[i], "is_fast", False
                ):
                    raise ValueError(
                        "mode='chat' requires fast student/teacher tokenizers for "
                        "return_offsets_mapping=True."
                    )
        if config.native_thinking_alignment and (
            config.mode != "chat" or not config.include_thinking_in_loss
        ):
            raise ValueError(
                "native_thinking_alignment requires mode='chat' and "
                "include_thinking_in_loss=true."
            )
        if (
            config.kd_alignment_regions is not None
            and not config.native_thinking_alignment
        ):
            raise ValueError(
                "kd_alignment_regions requires native_thinking_alignment=true."
            )
        if config.kd_alignment_regions == []:
            raise ValueError(
                "kd_alignment_regions must be a non-empty subset of reasoning, close, answer, eot"
            )
        if config.num_packed_rows != 1:
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
        self.mode = config.mode
        self.drop_first_assistant_chunk_kl_by_teacher = list(
            drop_first_assistant_chunk_kl_by_teacher
        )
        self.include_thinking_in_loss = config.include_thinking_in_loss
        self.native_thinking_alignment = config.native_thinking_alignment
        self.kd_alignment_regions = (
            frozenset(config.kd_alignment_regions)
            if config.kd_alignment_regions is not None
            else None
        )
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
        texts: list[str] = []
        for datum in batch:
            content = datum["message_log"][0]["content"]
            if not isinstance(content, str):
                raise TypeError(
                    "CrossTokenizerCollator text mode requires string content"
                )
            texts.append(content)
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
            # Keys follow AlignmentBatch fields to keep the payload consistent.
            for f in dataclass_fields(alignment):
                out[f"alignment_{i}_{f.name}"] = getattr(alignment, f.name)

        return BatchedDataDict(out)

    def _render_document(
        self,
        tokenizer: PreTrainedTokenizerBase,
        datum: DatumSpec,
        ctx_length: int,
        side: str,
    ) -> RenderedChatDocument:
        try:
            return _render_and_tokenize_chat(
                tokenizer,
                datum["message_log"],
                ctx_length,
                tools=datum.get("tools"),
                message_loss_mask=datum.get("message_loss_mask"),
                include_thinking_in_loss=self.include_thinking_in_loss,
                native_thinking_alignment=self.native_thinking_alignment,
                skip_overlength=True,
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"{side}, sample idx={datum['idx']}: {error}") from error

    def _call_chat(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        """Render complete conversations and align selected assistant regions."""
        student_docs = [
            self._render_document(
                self.student_tokenizer, datum, self.ctx_length_student, "student"
            )
            for datum in batch
        ]
        (
            student_input_ids,
            student_attention_mask,
            student_offsets,
            student_asst_mask,
        ) = self._pad_chat_batch(
            [doc.input_ids for doc in student_docs],
            [doc.offsets for doc in student_docs],
            [doc.assistant_mask for doc in student_docs],
            self.student_tokenizer.pad_token_id,
            self.make_seq_div_by_student,
        )
        out: dict[str, Any] = {
            "input_ids": student_input_ids,
            "input_lengths": student_attention_mask.sum(dim=-1).long(),
            "token_mask": (student_attention_mask * student_asst_mask).long(),
            "sample_mask": torch.tensor(
                [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
            ),
            "idx": [datum["idx"] for datum in batch],
        }
        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                for datum, doc in zip(batch, student_docs):
                    if len(doc.input_ids) > self.ctx_length_teachers[i]:
                        raise ValueError(
                            f"teacher {i}, sample idx={datum['idx']}: shared tokenization exceeds context length {self.ctx_length_teachers[i]}; overlength rows are rejected"
                        )
                continue
            tokenizer = self.teacher_tokenizers[i]
            teacher_docs = [
                self._render_document(
                    tokenizer, datum, self.ctx_length_teachers[i], f"teacher {i}"
                )
                for datum in batch
            ]
            for sample, (student_doc, teacher_doc) in enumerate(
                zip(student_docs, teacher_docs)
            ):
                if student_doc.source_turn_indices != teacher_doc.source_turn_indices:
                    raise ValueError(
                        f"sample {sample}, teacher {i}: selected assistant turn identities differ"
                    )
            (
                teacher_input_ids,
                teacher_attention_mask,
                teacher_offsets,
                teacher_asst_mask,
            ) = self._pad_chat_batch(
                [doc.input_ids for doc in teacher_docs],
                [doc.offsets for doc in teacher_docs],
                [doc.assistant_mask for doc in teacher_docs],
                tokenizer.pad_token_id,
                self.make_seq_div_by_teachers[i],
            )
            alignment = aligner.align_chat(
                student_input_ids,
                teacher_input_ids,
                student_offsets=student_offsets,
                teacher_offsets=teacher_offsets,
                student_asst_char_spans=[doc.assistant_spans for doc in student_docs],
                teacher_asst_char_spans=[doc.assistant_spans for doc in teacher_docs],
                student_attention_mask=student_attention_mask,
                teacher_attention_mask=teacher_attention_mask,
                student_asst_mask=student_asst_mask,
                teacher_asst_mask=teacher_asst_mask,
                student_eot_indices=[doc.eot_indices for doc in student_docs],
                teacher_eot_indices=[doc.eot_indices for doc in teacher_docs],
                drop_first_content_pair=self.drop_first_assistant_chunk_kl_by_teacher[
                    i
                ],
                student_alignment_regions=[
                    doc.alignment_regions for doc in student_docs
                ]
                if self.native_thinking_alignment
                else None,
                teacher_alignment_regions=[
                    doc.alignment_regions for doc in teacher_docs
                ]
                if self.native_thinking_alignment
                else None,
                student_answer_parts=[doc.answer_parts for doc in student_docs]
                if self.native_thinking_alignment
                else None,
                teacher_answer_parts=[doc.answer_parts for doc in teacher_docs]
                if self.native_thinking_alignment
                else None,
                student_rendered_texts=[doc.rendered_text for doc in student_docs]
                if self.native_thinking_alignment
                else None,
                teacher_rendered_texts=[doc.rendered_text for doc in teacher_docs]
                if self.native_thinking_alignment
                else None,
                included_regions=self.kd_alignment_regions,
            )
            out[f"teacher_{i}_input_ids"] = teacher_input_ids
            out[f"teacher_{i}_input_lengths"] = teacher_attention_mask.sum(
                dim=-1
            ).long()
            out[f"teacher_{i}_token_mask"] = (
                teacher_attention_mask * teacher_asst_mask
            ).long()
            for field in dataclass_fields(alignment):
                out[f"alignment_{i}_{field.name}"] = getattr(alignment, field.name)
        return BatchedDataDict(out)

    @staticmethod
    def _render_and_tokenize_chat(
        tokenizer: PreTrainedTokenizerBase,
        messages: List[dict],
        ctx_length: int,
    ) -> tuple[
        List[int], List[tuple[int, int]], List[int], List[tuple[int, int]], List[int]
    ]:
        """Compatibility helper for inspecting retained truncation boundaries.

        Production chat collation uses complete documents and rejects overflow.
        """
        doc = _render_and_tokenize_chat(
            tokenizer,
            messages,
            ctx_length,
            include_thinking_in_loss=False,
            native_thinking_alignment=False,
            skip_overlength=False,
        )
        return (
            doc.input_ids,
            doc.offsets,
            doc.assistant_mask,
            doc.assistant_spans,
            doc.eot_indices,
        )

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
            add_special_tokens=False,
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
