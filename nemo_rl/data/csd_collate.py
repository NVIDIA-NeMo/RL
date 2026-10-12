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
"""CSD collation with durable sample identities and strict sequence boundaries."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import fields as dataclass_fields
from typing import Any, List, Optional, cast

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import AlignmentPair, TokenAligner
from nemo_rl.data.chat_utils import normalize_message_loss_mask
from nemo_rl.data.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import get_first_index_that_differs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_CHAT_CONTENT_SENTINEL_STEM = "__3141592653589793238462643383279"


def _find_probed_content_span(
    rendered: str, probe_rendered: str, sentinel: str
) -> tuple[int, int] | None:
    """Locate a content insertion using its unchanged template surroundings."""
    if probe_rendered.count(sentinel) != 1:
        return None
    start = probe_rendered.index(sentinel)
    suffix = probe_rendered[start + len(sentinel) :]
    if not rendered.startswith(probe_rendered[:start]) or (
        suffix and not rendered.endswith(suffix)
    ):
        return None
    end = len(rendered) - len(suffix) if suffix else len(rendered)
    return (start, end) if end >= start else None


class CSDCrossTokenizerCollator(CrossTokenizerCollator):
    """Use shared offset alignment with CSD's untruncated logical samples."""

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
        require_routed_experts: bool,
        student_chat_template_kwargs: Optional[dict[str, Any]],
        teacher_chat_template_kwargs: list[dict[str, Any]],
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teachers: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            config=config,
            student_tokenizer=student_tokenizer,
            teacher_tokenizers=teacher_tokenizers,
            aligners=aligners,
            ctx_length_student=ctx_length_student,
            ctx_length_teachers=ctx_length_teachers,
            drop_first_assistant_chunk_kl_by_teacher=drop_first_assistant_chunk_kl_by_teacher,
            make_seq_div_by_student=make_seq_div_by_student,
            make_seq_div_by_teachers=make_seq_div_by_teachers,
        )
        self.require_routed_experts = require_routed_experts
        self.student_chat_template_kwargs = self._validate_template_kwargs(
            student_chat_template_kwargs or {}, side_id="student"
        )
        if len(teacher_chat_template_kwargs) != len(self.aligners):
            raise ValueError(
                "teacher_chat_template_kwargs must have one entry per teacher"
            )
        self.teacher_chat_template_kwargs = [
            self._validate_template_kwargs(value, side_id=f"teacher_{i}")
            for i, value in enumerate(teacher_chat_template_kwargs)
        ]

        if self.native_thinking_alignment and (
            self.student_chat_template_kwargs or any(self.teacher_chat_template_kwargs)
        ):
            raise ValueError(
                "CSD native_thinking_alignment does not support chat_template_kwargs; "
                "use the native templates without overrides or disable native alignment."
            )

    def _call_text(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        # csd_data_processor carries the raw text as a single assistant
        # message; the collator tokenizes that content for the student and
        # each cross-tokenizer teacher.
        texts: list[str] = []
        for datum in batch:
            content = datum["message_log"][0]["content"]
            if not isinstance(content, str):
                raise ValueError("CSD text mode requires string content")
            texts.append(content)
        sample_ids = self._required_sample_ids(batch)
        student_input_ids, student_attention_mask, student_offsets = (
            self._tokenize_csd_batch(
                texts,
                self.student_tokenizer,
                self.ctx_length_student,
                self.make_seq_div_by_student,
                sample_ids=sample_ids,
                side_id="student",
            )
        )

        sample_mask = torch.tensor(
            [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
        )
        idx = [datum["idx"] for datum in batch]
        student_input_lengths = student_attention_mask.sum(dim=-1).long()

        out: dict[str, Any] = {
            # Student-side keys map onto Policy.train's expected names. A
            # single student tokenization is shared across all teachers.
            "input_ids": student_input_ids,
            "input_lengths": student_input_lengths,
            "token_mask": student_attention_mask.long(),
            # Plain-text KD and CE share the same target region. Chat mode
            # overrides this with assistant content plus explicit EOT targets.
            "kd_token_mask": student_attention_mask.long(),
            "sample_mask": sample_mask,
            "idx": idx,
            "sample_id": sample_ids,
        }

        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                # Same-tokenizer teacher: no re-tokenization, no projection,
                # no alignment. Its forward reuses the student tokenization,
                # but its independent context and padding constraints still
                # apply to every reused row.
                self._validate_reused_teacher_context(
                    student_input_lengths.tolist(),
                    sample_ids=sample_ids,
                    ctx_length=self.ctx_length_teachers[i],
                    make_seq_div_by=self.make_seq_div_by_teachers[i],
                    side_id=f"teacher_{i}",
                    length_description="exact token length",
                )
                continue
            teacher_input_ids, teacher_attention_mask, teacher_offsets = (
                self._tokenize_csd_batch(
                    texts,
                    self.teacher_tokenizers[i],
                    self.ctx_length_teachers[i],
                    self.make_seq_div_by_teachers[i],
                    sample_ids=sample_ids,
                    side_id=f"teacher_{i}",
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

        self._add_router_replay_metadata(
            out,
            batch,
            student_input_ids=student_input_ids,
            student_input_lengths=student_input_lengths,
        )
        return BatchedDataDict(out)

    def _call_chat(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        """Chat/instruct path: align each assistant message independently.

        Renders each side's chat template, then calls
        :meth:`TokenAligner.align_one_offset_per_asst` per assistant message.
        CE uses the template-derived assistant message chunks while KD alignment
        uses the separate assistant-content semantic regions. Same-tokenizer
        teachers (``aligners[i] is None``) reuse the student tokenization only
        after setup proves full tokenizer/template equivalence.
        """
        if self.native_thinking_alignment:
            return self._call_native_chat(batch)
        b = len(batch)
        sample_ids = self._required_sample_ids(batch)
        s_ids, s_off, s_ce_mask, s_kd_mask, s_spans, s_eot = [], [], [], [], [], []
        for datum in batch:
            ids, off, ce_mask, kd_mask, spans, eot_indices = self._render_csd_chat(
                self.student_tokenizer,
                datum["message_log"],
                self.ctx_length_student,
                sample_id=sample_ids[len(s_ids)],
                side_id="student",
                make_seq_div_by=self.make_seq_div_by_student,
                tools=datum.get("tools"),
                message_loss_mask=datum.get("message_loss_mask"),
                template_kwargs=self.student_chat_template_kwargs,
            )
            s_ids.append(ids)
            s_off.append(off)
            s_ce_mask.append(ce_mask)
            s_kd_mask.append(kd_mask)
            s_spans.append(spans)
            s_eot.append(eot_indices)

        student_input_ids, student_attention_mask, _, student_asst_mask = (
            self._pad_chat_batch(
                s_ids,
                s_off,
                s_ce_mask,
                self.student_tokenizer.pad_token_id,
                self.make_seq_div_by_student,
            )
        )
        t_s = student_input_ids.shape[1]
        student_kd_mask = self._pad_chat_masks(s_kd_mask, max_len=t_s)

        out: dict[str, Any] = {
            "input_ids": student_input_ids,
            "input_lengths": student_attention_mask.sum(dim=-1).long(),
            # SFT policy: supervise complete assistant-rendered chunks.  KD
            # regions remain semantic content plus one explicit EOT target.
            "token_mask": (student_attention_mask * student_asst_mask).long(),
            "kd_token_mask": (student_attention_mask * student_kd_mask).long(),
            "sample_mask": torch.tensor(
                [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
            ),
            "idx": [datum["idx"] for datum in batch],
            "sample_id": sample_ids,
            "tools": [datum.get("tools") for datum in batch],
            "student_semantic_regions": [
                self._assistant_token_regions(
                    offsets,
                    spans,
                    eot_indices,
                    datum["message_log"],
                    datum.get("message_loss_mask"),
                )
                for offsets, spans, eot_indices, datum in zip(
                    s_off, s_spans, s_eot, batch, strict=True
                )
            ],
        }

        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                # Safe reuse was proven against the complete tokenizer/template
                # signature, so the teacher side has identical semantic
                # coordinates. Keep an explicit side-local metadata record for
                # occurrence-key binding even though no duplicate tensors are
                # emitted.
                self._validate_reused_teacher_context(
                    [len(ids) for ids in s_ids],
                    sample_ids=sample_ids,
                    ctx_length=self.ctx_length_teachers[i],
                    make_seq_div_by=self.make_seq_div_by_teachers[i],
                    side_id=f"teacher_{i}",
                    length_description="exact post-template length",
                )
                out[f"teacher_{i}_semantic_regions"] = [
                    tuple(regions) for regions in out["student_semantic_regions"]
                ]
                continue
            t_ids, t_off, t_ce_mask, t_kd_mask, t_spans, t_eot = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for datum in batch:
                ids, off, ce_mask, kd_mask, spans, eot_indices = self._render_csd_chat(
                    self.teacher_tokenizers[i],
                    datum["message_log"],
                    self.ctx_length_teachers[i],
                    sample_id=sample_ids[len(t_ids)],
                    side_id=f"teacher_{i}",
                    make_seq_div_by=self.make_seq_div_by_teachers[i],
                    tools=datum.get("tools"),
                    message_loss_mask=datum.get("message_loss_mask"),
                    template_kwargs=self.teacher_chat_template_kwargs[i],
                )
                t_ids.append(ids)
                t_off.append(off)
                t_ce_mask.append(ce_mask)
                t_kd_mask.append(kd_mask)
                t_spans.append(spans)
                t_eot.append(eot_indices)

            teacher_input_ids, teacher_attention_mask, _, _ = self._pad_chat_batch(
                t_ids,
                t_off,
                t_ce_mask,
                self.teacher_tokenizers[i].pad_token_id,
                self.make_seq_div_by_teachers[i],
            )
            t_t = teacher_input_ids.shape[1]
            teacher_kd_mask = self._pad_chat_masks(t_kd_mask, max_len=t_t)

            per_sample_pairs: List[List[AlignmentPair]] = []
            for j in range(b):
                raw = aligner.align_one_offset_per_asst(
                    s_ids[j],
                    s_off[j],
                    s_spans[j],
                    t_ids[j],
                    t_off[j],
                    t_spans[j],
                    student_asst_mask=s_kd_mask[j],
                    teacher_asst_mask=t_kd_mask[j],
                    student_eot_indices=s_eot[j],
                    teacher_eot_indices=t_eot[j],
                    drop_first_content_pair=(
                        self.drop_first_assistant_chunk_kl_by_teacher[i]
                    ),
                )
                per_sample_pairs.append(raw)

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
            out[f"teacher_{i}_token_mask"] = (
                teacher_attention_mask * teacher_kd_mask
            ).long()
            out[f"teacher_{i}_semantic_regions"] = [
                self._assistant_token_regions(
                    offsets,
                    spans,
                    eot_indices,
                    datum["message_log"],
                    datum.get("message_loss_mask"),
                )
                for offsets, spans, eot_indices, datum in zip(
                    t_off, t_spans, t_eot, batch, strict=True
                )
            ]
            for f in dataclass_fields(alignment):
                out[f"alignment_{i}_{f.name}"] = getattr(alignment, f.name)

        self._add_router_replay_metadata(
            out,
            batch,
            student_input_ids=student_input_ids,
            student_input_lengths=out["input_lengths"],
        )
        return BatchedDataDict(out)

    def _add_router_replay_metadata(
        self,
        out: MutableMapping[str, Any],
        batch: List[DatumSpec],
        *,
        student_input_ids: torch.Tensor,
        student_input_lengths: torch.Tensor,
    ) -> None:
        """Validate and batch rollout-recorded routes for the student forward.

        Router replay is meaningful only when the recorded routes describe the
        exact token sequence that the student will train on.  CSD normally
        re-tokenizes source text/chat in the collator, so compare the persisted
        per-message ``token_ids`` with that result before carrying the matching
        ``routed_experts`` tensor forward.  This turns stale tokenizer/template
        settings into an actionable input error instead of silently replaying
        routes at the wrong token positions.
        """
        if not self.require_routed_experts:
            return

        padded_seq_len = student_input_ids.shape[1]
        routed_rows: list[torch.Tensor] = []
        route_shape: Optional[tuple[int, int]] = None
        route_dtype: Optional[torch.dtype] = None

        for row_index, datum in enumerate(batch):
            sample_id = datum["sample_id"]
            token_parts: list[torch.Tensor] = []
            route_parts: list[torch.Tensor] = []
            message_log = cast(List[dict[str, Any]], datum["message_log"])
            for turn_index, message in enumerate(message_log):
                token_ids_value = message.get("token_ids")
                routed_experts_value = message.get("routed_experts")
                if token_ids_value is None:
                    raise RuntimeError(
                        "policy.router_replay.enabled=true requires rollout-recorded "
                        "token_ids and routed_experts on every message; "
                        f"sample_id={sample_id!r}, turn_index={turn_index} is "
                        "missing metadata. Use a vLLM router-replay rollout as "
                        "the CSD data source."
                    )

                token_ids = torch.as_tensor(token_ids_value, dtype=torch.long)
                if token_ids.dim() != 1:
                    raise ValueError(
                        "router-replay token_ids must have shape [tokens]; "
                        f"sample_id={sample_id!r}, turn_index={turn_index}, "
                        f"got {tuple(token_ids.shape)}."
                    )
                # Some message serializers erase the trailing dimensions of an
                # empty [0, L, K] tensor. Empty turns contribute no positions,
                # so they can be skipped without weakening route coverage.
                if token_ids.numel() == 0:
                    continue
                if routed_experts_value is None:
                    raise RuntimeError(
                        "policy.router_replay.enabled=true requires rollout-recorded "
                        "token_ids and routed_experts on every non-empty message; "
                        f"sample_id={sample_id!r}, turn_index={turn_index} is "
                        "missing routed_experts. Use a vLLM router-replay "
                        "rollout as the CSD data source."
                    )
                routed_experts = torch.as_tensor(routed_experts_value)
                if routed_experts.dim() != 3:
                    raise ValueError(
                        "router-replay routed_experts must have shape "
                        "[tokens, layers, topk]; "
                        f"sample_id={sample_id!r}, turn_index={turn_index}, "
                        f"got {tuple(routed_experts.shape)}."
                    )
                if routed_experts.shape[0] != token_ids.shape[0]:
                    raise ValueError(
                        "router-replay token_ids and routed_experts token axes "
                        f"differ for sample_id={sample_id!r}, "
                        f"turn_index={turn_index}: {token_ids.shape[0]} != "
                        f"{routed_experts.shape[0]}."
                    )

                current_shape = (
                    int(routed_experts.shape[1]),
                    int(routed_experts.shape[2]),
                )
                if route_shape is None:
                    route_shape = current_shape
                    route_dtype = routed_experts.dtype
                elif (
                    current_shape != route_shape or routed_experts.dtype != route_dtype
                ):
                    raise ValueError(
                        "router-replay routed_experts must use one [layers, topk] "
                        "shape and dtype across the batch; "
                        f"expected {route_shape}/{route_dtype}, got "
                        f"{current_shape}/{routed_experts.dtype} for "
                        f"sample_id={sample_id!r}, turn_index={turn_index}."
                    )
                token_parts.append(token_ids.detach().cpu())
                route_parts.append(routed_experts.detach().cpu())

            if not token_parts or route_shape is None or route_dtype is None:
                raise RuntimeError(
                    "policy.router_replay.enabled=true requires non-empty "
                    f"rollout route metadata; sample_id={sample_id!r} has none."
                )

            recorded_token_ids = torch.cat(token_parts, dim=0)
            recorded_routes = torch.cat(route_parts, dim=0)
            student_length = int(student_input_lengths[row_index].item())
            expected_token_ids = student_input_ids[row_index, :student_length].cpu()
            if not torch.equal(recorded_token_ids, expected_token_ids):
                raise RuntimeError(
                    "Cannot replay routed experts because rollout-recorded "
                    "student token_ids do not match the CSD student "
                    f"tokenization for sample_id={sample_id!r} "
                    f"(recorded={recorded_token_ids.shape[0]} tokens, "
                    f"retokenized={student_length}). Keep the rollout and "
                    "training tokenizer/chat-template settings identical."
                )

            pad_len = padded_seq_len - recorded_routes.shape[0]
            if pad_len < 0:
                raise RuntimeError(
                    "router-replay route metadata is longer than the student "
                    f"batch row for sample_id={sample_id!r}: "
                    f"{recorded_routes.shape[0]} > {padded_seq_len}."
                )
            if pad_len:
                recorded_routes = torch.nn.functional.pad(
                    recorded_routes, (0, 0, 0, 0, 0, pad_len), value=0
                )
            routed_rows.append(recorded_routes)

        out["routed_experts"] = torch.stack(routed_rows, dim=0)

    @staticmethod
    def _required_sample_ids(batch: List[DatumSpec]) -> list[str | int]:
        """Return durable source IDs; post-transform indices are not identity."""
        sample_ids: list[str | int] = []
        for row_index, datum in enumerate(batch):
            sample_id = datum.get("sample_id")
            if sample_id is None or not str(sample_id).strip():
                raise ValueError(
                    "CSDCrossTokenizerCollator requires a durable, non-empty "
                    "sample_id on every row; positional idx cannot survive "
                    f"filtering/splitting (batch row {row_index})."
                )
            sample_ids.append(sample_id)
        return sample_ids

    @staticmethod
    def _validate_reused_teacher_context(
        raw_lengths: List[int],
        *,
        sample_ids: List[str | int],
        ctx_length: int,
        make_seq_div_by: int,
        side_id: str,
        length_description: str,
    ) -> None:
        """Enforce one same-tokenizer teacher's independent context limit."""
        if len(raw_lengths) != len(sample_ids):
            raise ValueError("raw_lengths and sample_ids must have the same length")
        for raw_length, sample_id in zip(raw_lengths, sample_ids, strict=True):
            effective_len = (
                (raw_length + make_seq_div_by - 1) // make_seq_div_by
            ) * make_seq_div_by
            if raw_length > ctx_length or effective_len > ctx_length:
                raise ValueError(
                    f"CSD sample_id={sample_id!r} exceeds {side_id} context: "
                    f"{length_description} {raw_length}, effective length "
                    f"{effective_len}, capacity {ctx_length}; truncation is "
                    "forbidden."
                )

    @staticmethod
    def _render_csd_chat(
        tokenizer: PreTrainedTokenizerBase,
        messages: List[dict],
        ctx_length: int,
        *,
        sample_id: str | int,
        side_id: str,
        make_seq_div_by: int = 1,
        tools: Optional[list[dict[str, Any]]] = None,
        template_kwargs: Optional[dict[str, Any]] = None,
        message_loss_mask: Optional[list[int]] = None,
    ) -> tuple[
        List[int],
        List[tuple[int, int]],
        List[int],
        List[int],
        List[tuple[int, int]],
        List[int],
    ]:
        """Render + tokenize one conversation and derive its assistant spans.

        Renders with the tokenizer's chat template, tokenizes with char offsets,
        and derives the per-token assistant masks and each assistant message's
        char span. Returns ``(input_ids, offsets, assistant_ce_mask,
        assistant_kd_mask, assistant_char_spans, assistant_eot_indices)`` for
        one unpadded sample. The rendered string already carries the template's
        special tokens, so tokenization uses ``add_special_tokens=False``.
        """
        selected_turns = normalize_message_loss_mask(messages, message_loss_mask)
        for turn_index, message in enumerate(messages):
            content = message.get("content")
            if (
                selected_turns[turn_index]
                and message.get("tool_calls")
                and (
                    content is None
                    or (isinstance(content, str) and not content.strip())
                )
            ):
                raise ValueError(
                    "CSD tool-call target regions are not implemented: "
                    f"sample_id={sample_id!r}, turn_index={turn_index} has an "
                    "assistant target only in tool_calls."
                )
            if content is not None and not isinstance(content, str):
                raise TypeError(
                    "CSD chat alignment currently requires string message "
                    f"content; sample_id={sample_id!r}, turn_index={turn_index}, "
                    f"type={type(content).__name__}."
                )

        render_kwargs = {"tokenize": False, **(template_kwargs or {})}
        if tools is not None:
            render_kwargs["tools"] = tools
        rendered = tokenizer.apply_chat_template(messages, **render_kwargs)
        encoded = tokenizer(
            rendered,
            truncation=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offsets = [tuple(o) for o in encoded["offset_mapping"]]
        effective_len = (
            (len(input_ids) + make_seq_div_by - 1) // make_seq_div_by
        ) * make_seq_div_by
        if len(input_ids) > ctx_length or effective_len > ctx_length:
            raise ValueError(
                f"CSD sample_id={sample_id!r} exceeds {side_id} context: "
                f"exact post-template length {len(input_ids)}, effective "
                f"length {effective_len}, capacity {ctx_length}; truncation "
                "is forbidden."
            )

        # Match the ordinary SFT role policy: supervise the complete rendered
        # chunk introduced by each assistant turn (headers/EOT included), while
        # retaining raw assistant-content spans separately for cross-tokenizer
        # semantic alignment.
        assistant_chunk_spans: List[tuple[int, int]] = []
        previous_rendered = ""
        for turn_index, message in enumerate(messages):
            prefix = tokenizer.apply_chat_template(
                messages[: turn_index + 1], **render_kwargs
            )
            chunk_start = get_first_index_that_differs(previous_rendered, prefix)
            if selected_turns[turn_index]:
                assistant_chunk_spans.append((chunk_start, len(prefix)))
            previous_rendered = prefix

        # Structurally locate every assistant content insertion with a complete
        # conversation probe. Raw substring search is unsafe because the same
        # text may occur earlier in a role header, tool schema, or another turn.
        asst_char_spans: List[tuple[int, int]] = []
        for turn_index, message in enumerate(messages):
            if not selected_turns[turn_index]:
                continue
            sentinel_ordinal = 0
            while True:
                sentinel = (
                    f"{_CHAT_CONTENT_SENTINEL_STEM}_{turn_index}_{sentinel_ordinal}__"
                )
                if sentinel not in rendered:
                    break
                sentinel_ordinal += 1
            probe_messages = [dict(item) for item in messages]
            probe_messages[turn_index]["content"] = sentinel
            probe_rendered = tokenizer.apply_chat_template(
                probe_messages, **render_kwargs
            )
            span = _find_probed_content_span(rendered, probe_rendered, sentinel)
            if span is None:
                raise ValueError(
                    "CSD could not structurally locate assistant content in "
                    f"rendered chat: sample_id={sample_id!r}, side={side_id!r}, "
                    f"turn_index={turn_index}; the sentinel probe was missing, "
                    "duplicated, or changed text outside the content insertion."
                )
            asst_char_spans.append(span)

        assistant_content_mask = [
            1
            if any(s <= cs and ce <= e and ce > cs for (s, e) in asst_char_spans)
            else 0
            for (cs, ce) in offsets
        ]
        assistant_ce_mask = [
            1
            if any(s <= cs and ce <= e and ce > cs for (s, e) in assistant_chunk_spans)
            else 0
            for (cs, ce) in offsets
        ]
        if len(asst_char_spans) != len(assistant_chunk_spans):
            raise ValueError(
                "CSD could not derive one content span for every assistant "
                f"turn: sample_id={sample_id!r}, side={side_id!r}, "
                f"content_spans={len(asst_char_spans)}, "
                f"assistant_chunks={len(assistant_chunk_spans)}."
            )
        assistant_turn_indices = [
            turn_index
            for turn_index, message in enumerate(messages)
            if selected_turns[turn_index]
        ]
        for turn_index, (content_start, content_end), (chunk_start, chunk_end) in zip(
            assistant_turn_indices,
            asst_char_spans,
            assistant_chunk_spans,
            strict=True,
        ):
            if not (
                chunk_start <= content_start <= content_end <= chunk_end
                and chunk_end <= len(rendered)
            ):
                raise ValueError(
                    "CSD assistant content is outside its rendered assistant "
                    f"chunk: sample_id={sample_id!r}, side={side_id!r}, "
                    f"turn_index={turn_index}, content_span="
                    f"{(content_start, content_end)}, chunk_span="
                    f"{(chunk_start, chunk_end)}."
                )
        assistant_eot_indices: list[int] = []
        special_token_ids = {
            int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())
        }
        if not special_token_ids:
            raise ValueError(
                "CSD cannot identify explicit chat EOT targets because "
                f"{side_id!r} exposes no all_special_ids; "
                f"sample_id={sample_id!r}."
            )
        for turn_index, ((_, content_end), (_, chunk_end)) in enumerate(
            zip(asst_char_spans, assistant_chunk_spans, strict=True)
        ):
            eot_candidates = [
                token_index
                for token_index, ((start, end), token_id) in enumerate(
                    zip(offsets, input_ids, strict=True)
                )
                if start >= content_end
                and end <= chunk_end
                and end > start
                and bool(rendered[start:end].strip())
                and int(token_id) in special_token_ids
            ]
            if len(eot_candidates) != 1:
                raise ValueError(
                    "CSD requires exactly one registered special "
                    "post-content EOT token for "
                    f"every assistant turn; sample_id={sample_id!r}, "
                    f"side={side_id!r}, assistant_turn={turn_index}, "
                    f"candidate_indices={eot_candidates}."
                )
            eot_index = eot_candidates[0]
            assistant_eot_indices.append(eot_index)
        assistant_kd_mask = list(assistant_content_mask)
        for eot_index in assistant_eot_indices:
            assistant_kd_mask[eot_index] = 1
        return (
            input_ids,
            offsets,
            assistant_ce_mask,
            assistant_kd_mask,
            asst_char_spans,
            assistant_eot_indices,
        )

    @staticmethod
    def _pad_chat_masks(masks: List[List[int]], *, max_len: int) -> torch.Tensor:
        """Right-pad semantic masks to an already materialized sequence width."""
        padded = torch.zeros((len(masks), max_len), dtype=torch.long)
        for row_index, mask in enumerate(masks):
            if len(mask) > max_len:
                raise ValueError(
                    f"semantic mask length {len(mask)} exceeds padded width {max_len}"
                )
            if mask:
                padded[row_index, : len(mask)] = torch.tensor(mask, dtype=torch.long)
        return padded

    @staticmethod
    def _tokenize_csd_batch(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        make_seq_div_by: int,
        *,
        sample_ids: Optional[List[str | int]] = None,
        side_id: str = "model",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a batch and pad to a multiple of ``make_seq_div_by``.

        Also returns per-token character offsets (``offset_mapping``), which
        :meth:`TokenAligner.align` needs to align the student and teacher
        tokenizations of the same source text. This requires a *fast* HF
        tokenizer; special and padding positions carry ``(0, 0)``.
        """
        if sample_ids is None:
            sample_ids = list(range(len(texts)))
        if len(sample_ids) != len(texts):
            raise ValueError("sample_ids and texts must have the same length")

        ids_list: list[list[int]] = []
        offsets_list: list[list[tuple[int, int]]] = []
        for text, sample_id in zip(texts, sample_ids, strict=True):
            encoded = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            ids = list(encoded["input_ids"])
            offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
            effective_len = (
                (len(ids) + make_seq_div_by - 1) // make_seq_div_by
            ) * make_seq_div_by
            if len(ids) > ctx_length or effective_len > ctx_length:
                raise ValueError(
                    f"CSD sample_id={sample_id!r} exceeds {side_id} context: "
                    f"exact token length {len(ids)}, effective length "
                    f"{effective_len}, capacity {ctx_length}; truncation is "
                    "forbidden."
                )
            ids_list.append(ids)
            offsets_list.append(offsets)

        b = len(ids_list)
        max_len = max((len(ids) for ids in ids_list), default=0)
        max_len = max(max_len, 1)
        if make_seq_div_by > 1:
            max_len = (
                (max_len + make_seq_div_by - 1) // make_seq_div_by
            ) * make_seq_div_by
        input_ids = torch.full((b, max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        offset_mapping = torch.zeros((b, max_len, 2), dtype=torch.long)
        for batch_index, (ids, offsets) in enumerate(
            zip(ids_list, offsets_list, strict=True)
        ):
            length = len(ids)
            if length:
                input_ids[batch_index, :length] = torch.tensor(ids, dtype=torch.long)
                attention_mask[batch_index, :length] = 1
                offset_mapping[batch_index, :length] = torch.tensor(
                    offsets, dtype=torch.long
                )
        return input_ids, attention_mask, offset_mapping

    @staticmethod
    def _validate_template_kwargs(
        kwargs: dict[str, Any], *, side_id: str
    ) -> dict[str, Any]:
        reserved = {"tokenize", "tools"}.intersection(kwargs)
        if reserved:
            raise ValueError(
                f"{side_id} chat template kwargs may not override controller-owned "
                f"keys {sorted(reserved)!r}."
            )
        return dict(kwargs)

    @staticmethod
    def _assistant_token_regions(
        offsets: List[tuple[int, int]],
        spans: List[tuple[int, int]],
        eot_indices: List[int],
        messages: List[dict],
        message_loss_mask: Optional[list[int]] = None,
    ) -> tuple[tuple[int, str, str, int, int], ...]:
        """Return durable side-local assistant-content token regions."""
        regions: list[tuple[int, str, str, int, int]] = []
        selected_turns = normalize_message_loss_mask(messages, message_loss_mask)
        assistant_turn_indices = [
            turn_index
            for turn_index, message in enumerate(messages)
            if selected_turns[turn_index] and isinstance(message.get("content"), str)
        ]
        if len(assistant_turn_indices) != len(spans) or len(spans) != len(eot_indices):
            raise ValueError(
                "assistant semantic-region count does not match rendered spans"
            )
        for turn_index, (start_char, end_char), eot_index in zip(
            assistant_turn_indices, spans, eot_indices, strict=True
        ):
            token_indices = [
                token_index
                for token_index, (start, end) in enumerate(offsets)
                if start >= start_char and end <= end_char and end > start
            ]
            if token_indices:
                regions.append(
                    (
                        turn_index,
                        "assistant",
                        "content",
                        token_indices[0],
                        token_indices[-1] + 1,
                    )
                )
            regions.append((turn_index, "assistant", "eot", eot_index, eot_index + 1))
        return tuple(regions)

    def _call_native_chat(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        """Retain shared native reasoning/tool alignment with CSD sample metadata."""
        out = super()._call_chat(batch)
        out["sample_id"] = self._required_sample_ids(batch)
        out["kd_token_mask"] = out["token_mask"].clone()
        out["tools"] = [datum.get("tools") for datum in batch]
        for side, tokenizer, limit, divisor in [
            (
                "student",
                self.student_tokenizer,
                self.ctx_length_student,
                self.make_seq_div_by_student,
            ),
            *[
                (
                    f"teacher_{i}",
                    tokenizer or self.student_tokenizer,
                    self.ctx_length_teachers[i],
                    self.make_seq_div_by_teachers[i],
                )
                for i, tokenizer in enumerate(self.teacher_tokenizers)
            ],
        ]:
            documents = [
                self._render_document(tokenizer, datum, limit, side) for datum in batch
            ]
            self._validate_reused_teacher_context(
                [len(doc.input_ids) for doc in documents],
                sample_ids=out["sample_id"],
                ctx_length=limit,
                make_seq_div_by=divisor,
                side_id=side,
                length_description="exact post-template length",
            )
            regions = []
            for doc in documents:
                row = []
                for turn, (start, end), eot in zip(
                    doc.source_turn_indices,
                    doc.assistant_spans,
                    doc.eot_indices,
                    strict=True,
                ):
                    indices = [
                        i
                        for i, (a, b) in enumerate(doc.offsets)
                        if start <= a < b <= end
                    ]
                    if indices:
                        row.append(
                            (turn, "assistant", "content", indices[0], indices[-1] + 1)
                        )
                    if eot >= 0:
                        row.append((turn, "assistant", "eot", eot, eot + 1))
                regions.append(tuple(row))
            out[f"{side}_semantic_regions"] = regions
        self._add_router_replay_metadata(
            out,
            batch,
            student_input_ids=out["input_ids"],
            student_input_lengths=out["input_lengths"],
        )
        return out
