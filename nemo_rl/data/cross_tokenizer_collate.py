# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Cross-tokenizer collate function for off-policy distillation.

Moves teacher tokenize + DP alignment off the training critical path and into
``StatefulDataLoader`` worker processes. With ``num_workers=N, prefetch_factor=P``
there are up to ``N*P`` batches of CT work in flight, so the consumer pulls
already-processed batches and CT is hidden behind teacher forward.

Mirrors the train_distillation_ddp / TokenizeAndAlignCollator shape in
tokenalign/src/pytorch_data_loader.py.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing import TypedDict
    from typing_extensions import NotRequired

import torch
from transformers import AutoTokenizer

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class TeacherCTSpec(TypedDict):
    """Per-teacher spec passed to CrossTokenizerCollator.

    All fields are pickle-cheap primitives so the collator itself ships
    cheaply to DataLoader workers. Tokenizers and aligners are built lazily
    in each worker.
    """

    teacher_tokenizer_name: str
    student_tokenizer_name: str
    projection_matrix_path: str
    use_sparse_format: bool
    learnable: bool
    max_comb_len: int
    projection_matrix_multiplier: float
    project_teacher_to_student: bool
    max_teacher_len: int
    dp_chunk_size: int
    exact_token_match_only: bool


def _build_chunk_coo(
    aligned_pairs: list,
    student_seq_len: int,
    teacher_seq_len: int,
    exact_match_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pre-filter alignment pairs and emit per-sample chunk mask indices.

    Applies the exact_match_only, ``-1`` sentinel, and padded-length bounds
    filters that the loss fn used to run per-microbatch, then flattens each
    surviving chunk's student/teacher spans into COO rows ``[pos, chunk_id]``
    that the loss fn can ``index_put_`` into the dense ``proj_mask``/``tgt_mask``.

    Returns ``(student_chunk_coo, teacher_chunk_coo, num_chunks)``. COO tensors
    are empty ``(0, 2)`` when a sample has no surviving chunks.
    """
    student_rows: list[tuple[int, int]] = []
    teacher_rows: list[tuple[int, int]] = []
    chunk_id = 0
    for pair in aligned_pairs:
        s1_start, s1_end, s2_start, s2_end = pair[2], pair[3], pair[4], pair[5]
        if exact_match_only and (
            s1_end - s1_start != 1 or s2_end - s2_start != 1
        ):
            continue
        if s1_start == -1 or s2_start == -1:
            continue
        if s1_end > student_seq_len or s2_end > teacher_seq_len:
            continue
        for pos in range(s1_start, s1_end):
            student_rows.append((pos, chunk_id))
        for pos in range(s2_start, s2_end):
            teacher_rows.append((pos, chunk_id))
        chunk_id += 1

    student_coo = (
        torch.tensor(student_rows, dtype=torch.int64)
        if student_rows
        else torch.empty((0, 2), dtype=torch.int64)
    )
    teacher_coo = (
        torch.tensor(teacher_rows, dtype=torch.int64)
        if teacher_rows
        else torch.empty((0, 2), dtype=torch.int64)
    )
    return student_coo, teacher_coo, chunk_id


class CrossTokenizerCollator:
    """Collator that does base collate + message flatten + per-teacher CT.

    Designed to be pickled into ``torch.utils.data.DataLoader`` worker
    processes. On first call inside a worker, it constructs its own
    ``TokenAligner`` instances and teacher tokenizers from the specs, then
    reuses them for every subsequent batch in that worker.

    DP-only: does not run the char-offset alignment fast path (consumers
    that want char-offset should restore it here).
    """

    def __init__(
        self,
        pad_token_id: int,
        make_sequence_length_divisible_by: int,
        teacher_ct_specs: list[Optional[TeacherCTSpec]],
        fallback_student_tokenizer_name: Optional[str] = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.make_seq_div_by = int(make_sequence_length_divisible_by)
        self.teacher_ct_specs = list(teacher_ct_specs)
        self.fallback_student_tokenizer_name = fallback_student_tokenizer_name

        # Lazy per-worker state — excluded from __getstate__ below.
        self._initialized: bool = False
        self._aligners: list[Optional[Any]] = []
        self._teacher_tokenizers: list[Optional[Any]] = []
        self._student_tokenizer: Optional[Any] = None

    def __getstate__(self) -> dict[str, Any]:
        # Only ship pickle-cheap primitives across the fork/spawn boundary.
        return {
            "pad_token_id": self.pad_token_id,
            "make_seq_div_by": self.make_seq_div_by,
            "teacher_ct_specs": self.teacher_ct_specs,
            "fallback_student_tokenizer_name": self.fallback_student_tokenizer_name,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.pad_token_id = state["pad_token_id"]
        self.make_seq_div_by = state["make_seq_div_by"]
        self.teacher_ct_specs = state["teacher_ct_specs"]
        self.fallback_student_tokenizer_name = state["fallback_student_tokenizer_name"]
        self._initialized = False
        self._aligners = []
        self._teacher_tokenizers = []
        self._student_tokenizer = None

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        # Import TokenAligner lazily so module import stays cheap and so
        # workers that don't need CT never touch x_token.
        from nemo_rl.algorithms.x_token.tokenalign import TokenAligner

        for spec in self.teacher_ct_specs:
            if spec is None:
                self._aligners.append(None)
                self._teacher_tokenizers.append(None)
                continue

            teacher_tokenizer = AutoTokenizer.from_pretrained(
                spec["teacher_tokenizer_name"]
            )
            if teacher_tokenizer.pad_token is None:
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

            aligner = TokenAligner(
                teacher_tokenizer_name=spec["teacher_tokenizer_name"],
                student_tokenizer_name=spec["student_tokenizer_name"],
                max_comb_len=int(spec["max_comb_len"]),
                projection_matrix_multiplier=float(
                    spec["projection_matrix_multiplier"]
                ),
            )
            aligner._load_logits_projection_map(
                file_path=spec["projection_matrix_path"],
                use_sparse_format=bool(spec["use_sparse_format"]),
                learnable=bool(spec["learnable"]),
                device="cpu",
            )
            if bool(spec["project_teacher_to_student"]):
                aligner.create_reverse_projection_matrix(device="cpu")

            self._teacher_tokenizers.append(teacher_tokenizer)
            self._aligners.append(aligner)

        self._initialized = True

    def _get_student_tokenizer(self) -> Any:
        if self._student_tokenizer is not None:
            return self._student_tokenizer
        name = self.fallback_student_tokenizer_name
        if name is None:
            # Best-effort: reuse any CT spec's student name.
            for spec in self.teacher_ct_specs:
                if spec is not None:
                    name = spec["student_tokenizer_name"]
                    break
        if name is None:
            raise RuntimeError(
                "CrossTokenizerCollator needs a student tokenizer for the decode "
                "fallback, but no name was provided and no CT spec supplied one."
            )
        self._student_tokenizer = AutoTokenizer.from_pretrained(name)
        if self._student_tokenizer.pad_token is None:
            self._student_tokenizer.pad_token = self._student_tokenizer.eos_token
        return self._student_tokenizer

    def __call__(self, data_batch: list[DatumSpec]) -> BatchedDataDict[Any]:
        self._lazy_init()

        base = rl_collate_fn(data_batch)

        # --- Message-flatten (ported from _prepare_train_batch_data) ---
        for message_log in base["message_log"]:
            for m in message_log:
                if "token_loss_mask" not in m:
                    m["token_loss_mask"] = (
                        torch.ones_like(m["token_ids"])
                        if m["role"] == "assistant"
                        else torch.zeros_like(m["token_ids"])
                    )
        flat_messages, input_lengths = batched_message_log_to_flat_message(
            base["message_log"],
            pad_value_dict={"token_ids": self.pad_token_id},
            make_sequence_length_divisible_by=self.make_seq_div_by,
        )
        base["input_ids"] = flat_messages["token_ids"]
        base["input_lengths"] = input_lengths
        base["token_mask"] = flat_messages["token_loss_mask"]
        base["sample_mask"] = base["loss_multiplier"]
        base["flat_messages"] = flat_messages
        mm_dict = flat_messages.get_multimodal_dict(as_tensors=False)
        if mm_dict:
            for k, v in mm_dict.items():
                base[k] = v

        # --- Per-teacher CT (DP-only) ---
        student_ids = base["input_ids"]
        extra_env = base.get("extra_env_info")
        batch_size = student_ids.shape[0]

        has_raw_text = (
            extra_env is not None
            and len(extra_env) == batch_size
            and all(
                isinstance(e, dict) and "raw_text" in e for e in extra_env
            )
        )
        texts_cache: Optional[list[str]] = None

        per_teacher_ct_data: list[Optional[dict[str, Any]]] = []
        any_ct = any(spec is not None for spec in self.teacher_ct_specs)
        if any_ct:
            if has_raw_text:
                texts_cache = [e["raw_text"] for e in extra_env]
            else:
                texts_cache = self._get_student_tokenizer().batch_decode(
                    student_ids.tolist(), skip_special_tokens=True
                )

        for t_idx, spec in enumerate(self.teacher_ct_specs):
            if spec is None:
                per_teacher_ct_data.append(None)
                continue

            aligner = self._aligners[t_idx]
            teacher_tokenizer = self._teacher_tokenizers[t_idx]

            enc = teacher_tokenizer(
                texts_cache,
                max_length=int(spec["max_teacher_len"]),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            teacher_input_ids = enc["input_ids"]
            teacher_attention_mask = enc["attention_mask"]
            teacher_input_lengths = teacher_attention_mask.sum(dim=1)
            teacher_token_mask = (teacher_attention_mask > 0).to(torch.float32)

            dp_chunk_size = int(spec["dp_chunk_size"])
            exact_match_only = bool(spec.get("exact_token_match_only", False))
            student_seq_len = int(student_ids.shape[1])
            teacher_seq_len = int(teacher_input_ids.shape[1])

            aligned_pairs: list[Any] = []
            student_chunk_coo: list[torch.Tensor] = []
            teacher_chunk_coo: list[torch.Tensor] = []
            num_chunks_per_sample: list[int] = []
            for b in range(batch_size):
                s_t = student_ids[b : b + 1]
                t_t = teacher_input_ids[b : b + 1]
                result = aligner.align(s_t, t_t, chunk_size=dp_chunk_size)
                pairs = result[0]
                aligned_pairs.append(pairs)

                s_coo, t_coo, n_chunks = _build_chunk_coo(
                    pairs, student_seq_len, teacher_seq_len, exact_match_only,
                )
                student_chunk_coo.append(s_coo)
                teacher_chunk_coo.append(t_coo)
                num_chunks_per_sample.append(n_chunks)

            teacher_data: BatchedDataDict[Any] = BatchedDataDict(
                {
                    "input_ids": teacher_input_ids,
                    "input_lengths": teacher_input_lengths,
                    "token_mask": teacher_token_mask,
                    "sample_mask": base["loss_multiplier"],
                }
            )
            teacher_data.to("cpu")

            per_teacher_ct_data.append(
                {
                    "teacher_input_ids": teacher_input_ids,
                    "aligned_pairs": aligned_pairs,
                    "teacher_data": teacher_data,
                    "student_chunk_coo": student_chunk_coo,
                    "teacher_chunk_coo": teacher_chunk_coo,
                    "num_chunks": num_chunks_per_sample,
                }
            )

        base["per_teacher_ct_data"] = per_teacher_ct_data
        return base
