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
"""CSD collation: shared cross-tokenizer batches plus durable row identities."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, List, Optional, cast

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.cross_tokenizer.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class CSDCrossTokenizerCollator(CrossTokenizerCollator):
    """Shared cross-tokenizer collation plus CSD row identities and MoE routes.

    Batches are built by :class:`CrossTokenizerCollator`. Every batch also
    carries ``sample_id``, the durable source identity of each row that CSD's
    lockstep packing keys on, and, with ``require_routed_experts``,
    ``routed_experts``: the rollout-recorded MoE routes of every student
    token, right-padded to ``[B, T, num_layers, topk]``.

    Args:
        require_routed_experts: Emit ``routed_experts`` from each message's
            rollout-recorded ``token_ids`` and ``routed_experts``, which must
            reproduce the student tokenization exactly.
        clip_overlength_text: Text-mode policy for rows that do not fit. A
            row fits when, on the student and on every teacher, its
            tokenization rounded up to that side's sequence-length divisor
            is within that side's context length; same-tokenizer teachers
            use the student tokenization. ``True`` replaces the text of each
            row that does not fit with a prefix that fits every side, so the
            student and every teacher still cover the same text. Requires
            ``mode="text"``.
        All other arguments are those of :class:`CrossTokenizerCollator`.
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
        require_routed_experts: bool,
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teachers: Optional[List[int]] = None,
        reject_overlength_text: bool = False,
        ce_covers_assistant_turn: bool = False,
        add_special_tokens_in_text: bool = False,
        clip_overlength_text: bool = False,
    ) -> None:
        if clip_overlength_text and config.mode != "text":
            raise ValueError(
                "clip_overlength_text requires mode='text'; chat rows are not clipped."
            )
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
            reject_overlength_text=reject_overlength_text,
            ce_covers_assistant_turn=ce_covers_assistant_turn,
            add_special_tokens_in_text=add_special_tokens_in_text,
        )
        self.require_routed_experts = require_routed_experts
        self.clip_overlength_text = clip_overlength_text

    def __call__(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        sample_ids = self._required_sample_ids(batch)
        if self.clip_overlength_text:
            batch = self._clip_overlength_rows(batch)
        out = super().__call__(batch)
        out["sample_id"] = sample_ids
        self._add_router_replay_metadata(
            out,
            batch,
            student_input_ids=out["input_ids"],
            student_input_lengths=out["input_lengths"],
        )
        return out

    def _clip_overlength_rows(self, batch: List[DatumSpec]) -> List[DatumSpec]:
        """``batch`` with each text row that does not fit every side clipped.

        Rows that fit are returned as is; clipped rows are copies whose text is
        a prefix that fits every side. Non-string content is left for the
        shared collator to reject.
        """
        texts = [datum["message_log"][0]["content"] for datum in batch]
        if not all(isinstance(text, str) for text in texts):
            return batch
        texts = cast(List[str], texts)
        clipped: List[DatumSpec] = []
        for datum, text, fits in zip(batch, texts, self._fits_every_side(texts)):
            if fits:
                clipped.append(datum)
                continue
            message_log = cast(List[dict[str, Any]], datum["message_log"])
            prefix = self._fitting_text_prefix(text, datum["idx"])
            clipped.append(
                cast(
                    DatumSpec,
                    {
                        **datum,
                        "message_log": [
                            {**message_log[0], "content": prefix},
                            *message_log[1:],
                        ],
                    },
                )
            )
        return clipped

    def _fits_every_side(self, texts: List[str]) -> List[bool]:
        """Whether each text fits the student and every teacher."""

        def lengths(tokenizer: PreTrainedTokenizerBase) -> List[int]:
            encoded = tokenizer(
                texts,
                truncation=False,
                add_special_tokens=self.add_special_tokens_in_text,
            )
            return [len(ids) for ids in encoded["input_ids"]]

        student_lengths = lengths(self.student_tokenizer)
        sides = [
            (student_lengths, self.ctx_length_student, self.make_seq_div_by_student)
        ]
        for i, aligner in enumerate(self.aligners):
            teacher_tokenizer = self.teacher_tokenizers[i]
            ctx_length = self.ctx_length_teachers[i]
            if aligner is None or teacher_tokenizer is None:
                # Same-tokenizer teachers reuse the student tokenization, padded
                # to the student's divisor (packed rows pad to the teacher's).
                sides.append(
                    (student_lengths, ctx_length, self.make_seq_div_by_student)
                )
                sides.append(
                    (student_lengths, ctx_length, self.make_seq_div_by_teachers[i])
                )
                continue
            sides.append(
                (
                    lengths(teacher_tokenizer),
                    ctx_length,
                    self.make_seq_div_by_teachers[i],
                )
            )
        return [
            all(
                self._padded_width(side_lengths[row], make_seq_div_by) <= ctx_length
                for side_lengths, ctx_length, make_seq_div_by in sides
            )
            for row in range(len(texts))
        ]

    def _fitting_text_prefix(self, text: str, row_idx: int) -> str:
        """A non-empty prefix of ``text`` that fits every side.

        Binary search over the prefix length. Token counts need not grow
        monotonically with the prefix length, so the search keeps only
        prefixes whose encodings fit: the result always fits but need not be
        the longest prefix that does.
        """
        low, high, best = 1, len(text) - 1, ""
        while low <= high:
            middle = (low + high) // 2
            if self._fits_every_side([text[:middle]])[0]:
                best = text[:middle]
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            raise ValueError(
                f"sample idx={row_idx}: no non-empty prefix of the text fits "
                "every context length"
            )
        return best

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
