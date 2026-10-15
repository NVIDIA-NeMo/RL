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
   In ``mode="chat"``, renders the full conversation with the student's chat
   template and supervises only the final assistant turn (by default its
   content and end-of-turn token).
2. For each *cross-tokenizer* teacher, tokenizes with that teacher's
   tokenizer and aligns with its :class:`TokenAligner`. Text mode aligns
   the full text; chat mode renders the teacher's own chat template and
   aligns the final assistant message. Teacher scoring masks also select only
   the final assistant content and end-of-turn token. Dense-padded alignment and
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
from nemo_rl.data.xtoken.native_chat import (
    RenderedChatDocument,
    _render_and_tokenize_chat,
    _render_chat_text,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class CrossTokenizerCollatorConfig(BaseModel, extra="allow"):
    """Shared batching options for cross-tokenizer distillation.

    Attributes:
        mode: ``"text"`` tokenizes raw text; ``"chat"`` renders each model's
            chat template and supervises assistant turns.
        include_thinking_in_loss: Include reasoning text and its closing marker
            in assistant supervision, excluding the opening scaffold. Separate
            ``reasoning_content`` fields require native thinking alignment;
            ordinary chat supports inline ``<think>`` content only when the
            template renders content verbatim. Only the final assistant turn
            is supervised; earlier turns remain context.
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
    Chat mode applies each model's chat template to the full conversation,
    masks teacher scoring (and, by default, loss) to the final assistant content
    and end-of-turn token, and aligns only that final message.

    Args:
        config: Typed batching options, including text/chat mode and the
            thinking-region and packing settings.
        student_tokenizer: HF tokenizer matching the student model.
        teacher_tokenizers: Per-teacher HF tokenizers. May be ``None`` for a
            same-tokenizer teacher (its tokenization is the student's).
        aligners: Per-teacher :class:`TokenAligner`. ``None`` marks a
            same-tokenizer teacher (no projection / no alignment).
        ctx_length_student: Hard tokenization length cap on the student
            side. Batches pad to their longest tokenization rounded up to
            ``make_seq_div_by_student``, so the padded width can exceed a
            cap that is not a multiple of the divisor.
        ctx_length_teachers: Per-teacher tokenization length caps.
        drop_first_assistant_chunk_kl_by_teacher: Per-teacher flags controlling
            whether chat alignment drops the first content pair in each
            assistant message. The list retains a slot for same-tokenizer
            teachers so its indices match ``aligners``.
        make_seq_div_by_student: Round student sequence length up to a
            multiple of this value (typically TP * CP * 2 for Automodel).
        make_seq_div_by_teachers: Per-teacher sequence-length divisors.
        reject_overlength_text: Text-mode policy for rows longer than a
            context length. ``False`` truncates each side to its own context
            length. ``True`` raises ``ValueError`` for a row whose student or
            teacher tokenization, or that length rounded up to the side's
            sequence-length divisor, exceeds that side's context length, so
            every side covers the same text and no padded batch is wider
            than its context; same-tokenizer teachers check the shared
            student tokenization against their own context length and
            divisor. Chat mode always rejects overlength rows.
        ce_covers_assistant_turn: Chat-mode CE target. ``False`` supervises
            the final assistant content and end-of-turn token. ``True``
            supervises every token of the final assistant turn (SFT-style),
            including chat-template markers and any rendered reasoning, and
            emits the content and end-of-turn mask as ``kd_token_mask``.
            Alignment and teacher token masks are unchanged. The turn starts
            where the rendering of all earlier messages ends, or at the start
            of the rendering when there are none; that rendering must be a
            prefix of the full rendering. Requires ``mode="chat"`` without
            ``native_thinking_alignment``.
        add_special_tokens_in_text: Text-mode special tokens. ``False``
            tokenizes the raw text only. ``True`` tokenizes the student and
            every cross-tokenizer teacher with each tokenizer's own special
            tokens (for example a beginning-of-text marker at the start of
            each row); ``reject_overlength_text`` counts them. Added tokens
            carry ``(0, 0)`` offsets, so alignment pairs them by role and a
            marker without a counterpart forms its own chunk with an empty
            other side. Requires ``mode="text"``; chat templates insert their
            own markers.
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
        reject_overlength_text: bool = False,
        ce_covers_assistant_turn: bool = False,
        add_special_tokens_in_text: bool = False,
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
        if ce_covers_assistant_turn and config.mode != "chat":
            raise ValueError(
                "ce_covers_assistant_turn requires mode='chat'; text mode has "
                "no chat-template assistant turn."
            )
        if ce_covers_assistant_turn and config.native_thinking_alignment:
            raise ValueError(
                "ce_covers_assistant_turn is not supported with "
                "native_thinking_alignment=true, which defines its own "
                "supervised regions."
            )
        if add_special_tokens_in_text and config.mode != "text":
            raise ValueError(
                "add_special_tokens_in_text requires mode='text'; chat templates "
                "insert their own special tokens."
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
        self.reject_overlength_text = reject_overlength_text
        self.ce_covers_assistant_turn = ce_covers_assistant_turn
        self.add_special_tokens_in_text = add_special_tokens_in_text
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
        idx = [datum["idx"] for datum in batch]
        if self.reject_overlength_text:
            self._reject_overlength_texts(
                texts,
                idx,
                self.student_tokenizer,
                self.ctx_length_student,
                "student",
                make_seq_div_by=self.make_seq_div_by_student,
                add_special_tokens=self.add_special_tokens_in_text,
            )
        student_input_ids, student_attention_mask, student_offsets = (
            self._tokenize_batch(
                texts,
                self.student_tokenizer,
                self.ctx_length_student,
                self.make_seq_div_by_student,
                add_special_tokens=self.add_special_tokens_in_text,
            )
        )

        sample_mask = torch.tensor(
            [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
        )

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
                if self.reject_overlength_text:
                    ctx_length = self.ctx_length_teachers[i]
                    make_seq_div_by = self.make_seq_div_by_teachers[i]
                    for row_idx, length in zip(idx, out["input_lengths"].tolist()):
                        if length > ctx_length:
                            raise ValueError(
                                f"teacher {i}, sample idx={row_idx}: shared tokenization exceeds context length {ctx_length}; overlength rows are rejected"
                            )
                        padded = self._padded_width(length, make_seq_div_by)
                        if padded > ctx_length:
                            raise ValueError(
                                f"teacher {i}, sample idx={row_idx}: shared tokenization has {length} tokens, padded to {padded} for a sequence-length multiple of {make_seq_div_by}, exceeding context length {ctx_length}; overlength rows are rejected"
                            )
                continue
            if self.reject_overlength_text:
                self._reject_overlength_texts(
                    texts,
                    idx,
                    self.teacher_tokenizers[i],
                    self.ctx_length_teachers[i],
                    f"teacher {i}",
                    make_seq_div_by=self.make_seq_div_by_teachers[i],
                    add_special_tokens=self.add_special_tokens_in_text,
                )
            teacher_input_ids, teacher_attention_mask, teacher_offsets = (
                self._tokenize_batch(
                    texts,
                    self.teacher_tokenizers[i],
                    self.ctx_length_teachers[i],
                    self.make_seq_div_by_teachers[i],
                    add_special_tokens=self.add_special_tokens_in_text,
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
            # Alignment payload, dense-padded so Automodel can shard on dim 0.
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
                message_loss_mask=datum["message_loss_mask"],
                include_thinking_in_loss=self.include_thinking_in_loss,
                native_thinking_alignment=self.native_thinking_alignment,
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"{side}, sample idx={datum['idx']}: {error}") from error

    def _final_turn_start(self, datum: DatumSpec, doc: RenderedChatDocument) -> int:
        """Return the index of the first student token of the final assistant turn.

        The turn spans from the end of the rendering of the earlier messages
        (empty when there are none) to the end of the full rendering.
        """
        messages = datum["message_log"]
        try:
            # Same template controls as the full rendering; __init__ rejects
            # native thinking alignment with this option.
            history = (
                _render_chat_text(
                    self.student_tokenizer,
                    messages[:-1],
                    preserve_thinking=self.include_thinking_in_loss,
                    tools=datum.get("tools"),
                )
                if len(messages) > 1
                else ""
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"student, sample idx={datum['idx']}: {error}") from error
        if not doc.rendered_text.startswith(history):
            raise ValueError(
                f"student, sample idx={datum['idx']}: rendering the messages before the final assistant turn does not give a prefix of the full rendering; cannot locate the turn boundary"
            )
        turn_char = len(history)
        first = next(
            (i for i, (_, end) in enumerate(doc.offsets) if end > turn_char), None
        )
        if first is None:
            raise ValueError(
                f"student, sample idx={datum['idx']}: final assistant turn has no tokens"
            )
        # A token may absorb whitespace that ends the history, but not text.
        if doc.rendered_text[doc.offsets[first][0] : turn_char].strip():
            raise ValueError(
                f"student, sample idx={datum['idx']}: token {first} crosses the final assistant turn boundary"
            )
        if any(doc.assistant_mask[:first]):
            raise ValueError(
                f"student, sample idx={datum['idx']}: assistant targets precede the final assistant turn"
            )
        return first

    def _call_chat(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        """Render complete conversations and align their final assistant regions."""
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
        if self.ce_covers_assistant_turn:
            turn_starts = torch.tensor(
                [
                    self._final_turn_start(datum, doc)
                    for datum, doc in zip(batch, student_docs)
                ],
                dtype=torch.long,
            )
            positions = torch.arange(student_input_ids.shape[1], dtype=torch.long)
            turn_mask = (positions[None, :] >= turn_starts[:, None]).long()
            out["kd_token_mask"] = out["token_mask"]
            out["token_mask"] = (student_attention_mask * turn_mask).long()
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
    def _padded_width(length: int, make_seq_div_by: int) -> int:
        """``length`` rounded up to a multiple of ``make_seq_div_by``, and at least ``max(make_seq_div_by, 1)``."""
        if make_seq_div_by > 1 and length % make_seq_div_by:
            length += make_seq_div_by - (length % make_seq_div_by)
        return max(length, make_seq_div_by, 1)

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
        max_len = CrossTokenizerCollator._padded_width(
            max((len(ids) for ids in ids_list), default=0), make_seq_div_by
        )

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
    def _reject_overlength_texts(
        texts: List[str],
        idx: List[int],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        side: str,
        *,
        make_seq_div_by: int,
        add_special_tokens: bool,
    ) -> None:
        """Raise ``ValueError`` if any text needs more than ``ctx_length`` positions.

        A row needs its token count rounded up to ``make_seq_div_by``.
        """
        encoded = tokenizer(
            texts, truncation=False, add_special_tokens=add_special_tokens
        )
        for row_idx, ids in zip(idx, encoded["input_ids"]):
            if len(ids) > ctx_length:
                raise ValueError(
                    f"{side}, sample idx={row_idx}: text tokenization has {len(ids)} tokens, exceeding context length {ctx_length}; overlength rows are rejected"
                )
            padded = CrossTokenizerCollator._padded_width(len(ids), make_seq_div_by)
            if padded > ctx_length:
                raise ValueError(
                    f"{side}, sample idx={row_idx}: text tokenization has {len(ids)} tokens, padded to {padded} for a sequence-length multiple of {make_seq_div_by}, exceeding context length {ctx_length}; overlength rows are rejected"
                )

    @staticmethod
    def _tokenize_batch(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        make_seq_div_by: int,
        *,
        add_special_tokens: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a batch, truncating to ``ctx_length``, and pad to its longest row.

        Rows are right-padded to the longest row rounded up to a multiple of
        ``make_seq_div_by``, the same width rule as chat mode. Also returns
        per-token character offsets (``offset_mapping``), which
        :meth:`TokenAligner.align` needs to align the student and teacher
        tokenizations of the same source text. This requires a *fast* HF
        tokenizer; special and padding positions carry ``(0, 0)``.
        """
        encoded = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=ctx_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        input_ids: torch.Tensor = encoded["input_ids"]
        attention_mask: torch.Tensor = encoded["attention_mask"]
        offset_mapping: torch.Tensor = encoded["offset_mapping"]

        b, t = input_ids.shape
        pad = CrossTokenizerCollator._padded_width(t, make_seq_div_by) - t
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
