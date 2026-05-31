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
"""Hermetic CPU tests for ``CrossTokenizerCollator``.

Companion to the env-gated snapshot suite at
``tests/unit/algorithms/x_token/test_alignment_snapshot.py`` — these
tests pin the collator's *contract* (output keys, shapes, padding,
truncation) without needing real HF tokenizers or pre-captured artifacts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, TokenAligner
from nemo_rl.data.cross_tokenizer_collate import CrossTokenizerCollator


# ---------------------------------------------------------------------------
# Fake tokenizer — deterministic, no HF dependency.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal tokenizer that satisfies ``CrossTokenizerCollator``'s contract.

    Tokenization is character-level with a fixed pad token at id 0.
    Texts longer than ``max_length`` are truncated; shorter texts are
    padded to ``max_length``. Returns the dict shape HF tokenizers
    produce (``input_ids``, ``attention_mask``) when called with
    ``return_tensors="pt"``.
    """

    eos_token = "<eos>"

    def __init__(self, vocab_size: int, prefix: str = "tok") -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self._prefix = prefix

    def __call__(
        self, texts, padding, truncation, max_length, return_tensors
    ):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        all_ids = []
        all_mask = []
        for t in texts:
            ids = [
                2 + (ord(c) % (self.vocab_size - 2)) for c in t
            ][:max_length]
            mask = [1] * len(ids)
            if len(ids) < max_length:
                pad_n = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_n
                mask = mask + [0] * pad_n
            all_ids.append(ids)
            all_mask.append(mask)
        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_mask, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        # Prefix the tokenizer name into each token id so two FakeTokenizer
        # instances with different prefixes do NOT trivially share tokens.
        return [f"{self._prefix}_{int(i)}" for i in ids]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _datum(text: str, idx: int = 0, loss_multiplier: float = 1.0) -> dict:
    """Build a DatumSpec-like dict matching what ``kd_data_processor`` emits."""
    return {
        "message_log": [],
        "length": len(text),
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "raw_text": text,
    }


def _fake_aligner(b: int, t_s: int, t_t: int, max_pairs: int = 2) -> MagicMock:
    """Return a MagicMock TokenAligner whose ``.align()`` yields a real
    ``AlignmentBatch`` with realistic shapes for ``(b, t_s, t_t, max_pairs)``.
    Keeps the collator test focused on the collator's contract, not on
    alignment quality (which is covered separately).
    """
    aligner = MagicMock(spec=TokenAligner)
    aligner.align.return_value = AlignmentBatch(
        pair_valid=torch.ones((b, max_pairs), dtype=torch.bool),
        pair_is_correct=torch.ones((b, max_pairs), dtype=torch.bool),
        student_exact_partition_mask=torch.zeros((b, t_s), dtype=torch.bool),
        teacher_exact_partition_mask=torch.zeros((b, t_t), dtype=torch.bool),
        student_chunk_id=torch.zeros((b, t_s), dtype=torch.long),
        teacher_chunk_id=torch.zeros((b, t_t), dtype=torch.long),
        num_chunks=torch.tensor([max_pairs] * b, dtype=torch.long),
    )
    return aligner


# Expected keys consumed by xtoken_off_policy_distillation.py:459-476 — drift
# detector. Adding/removing keys here in lockstep with the trainer's
# train_data packer catches silent breakage.
_EXPECTED_COLLATOR_KEYS = {
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
    "teacher_input_ids",
    "teacher_input_lengths",
    "teacher_token_mask",
    "alignment_pair_valid",
    "alignment_pair_is_correct",
    "alignment_student_exact_partition_mask",
    "alignment_teacher_exact_partition_mask",
    "alignment_student_chunk_id",
    "alignment_teacher_chunk_id",
    "alignment_num_chunks",
    "idx",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollatorOutputKeys:
    def test_emits_all_expected_keys(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=2, t_s=8, t_t=8)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            aligner=aligner,
            ctx_length_student=8,
            ctx_length_teacher=8,
        )
        out = collator([_datum("hello", 0), _datum("world", 1)])
        assert _EXPECTED_COLLATOR_KEYS.issubset(set(out.keys()))


class TestCollatorShapes:
    def test_student_and_teacher_axes_independent(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=2, t_s=8, t_t=16, max_pairs=3)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            aligner=aligner,
            ctx_length_student=8,
            ctx_length_teacher=16,
        )
        out = collator([_datum("ab", 0), _datum("cd", 1)])
        assert out["input_ids"].shape == (2, 8)
        assert out["token_mask"].shape == (2, 8)
        assert out["teacher_input_ids"].shape == (2, 16)
        assert out["teacher_token_mask"].shape == (2, 16)
        # alignment payload shapes come from the aligner mock.
        assert out["alignment_pair_valid"].shape == (2, 3)
        assert out["alignment_student_chunk_id"].shape == (2, 8)
        assert out["alignment_teacher_chunk_id"].shape == (2, 16)
        assert out["alignment_num_chunks"].shape == (2,)

    def test_input_lengths_match_attention_mask_sum(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=8, t_t=8)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            aligner=aligner,
            ctx_length_student=8,
            ctx_length_teacher=8,
        )
        out = collator([_datum("abc", 0)])  # 3 chars → 3 real tokens
        assert int(out["input_lengths"][0]) == 3
        assert int(out["token_mask"][0].sum()) == 3
        assert int(out["teacher_input_lengths"][0]) == 3


class TestCollatorTruncation:
    def test_long_text_truncates_trailing_tokens_not_drops_sample(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        ctx = 4
        aligner = _fake_aligner(b=1, t_s=ctx, t_t=ctx)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            aligner=aligner,
            ctx_length_student=ctx,
            ctx_length_teacher=ctx,
        )
        # 10 chars; ctx=4 → sample is kept, trailing 6 chars dropped.
        out = collator([_datum("abcdefghij", 0)])
        assert out["input_ids"].shape == (1, 4)
        # token_mask should be all 1s (4 real tokens, no padding).
        assert int(out["token_mask"][0].sum()) == 4
        assert int(out["input_lengths"][0]) == 4


class TestCollatorSequenceDivisibility:
    def test_seq_padded_up_to_multiple(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        # ctx=10, make_seq_div_by=8 → output sequence length = 16.
        # Verify both student and teacher pad independently.
        aligner = _fake_aligner(b=1, t_s=16, t_t=12)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            aligner=aligner,
            ctx_length_student=10,
            ctx_length_teacher=10,
            make_seq_div_by_student=8,
            make_seq_div_by_teacher=4,
        )
        out = collator([_datum("hi", 0)])
        assert out["input_ids"].shape == (1, 16)
        assert out["token_mask"].shape == (1, 16)
        assert out["teacher_input_ids"].shape == (1, 12)
        assert out["teacher_token_mask"].shape == (1, 12)
        # Padded slots have token_mask=0.
        assert int(out["token_mask"][0].sum()) == 2  # only "hi" -> 2 real toks


class TestCollatorPadTokenFallback:
    def test_missing_pad_token_set_from_eos(self):
        # Tokenizer without a pad token id — collator must set
        # pad_token = eos_token in __init__.
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        student_tok.pad_token_id = None
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=4, t_t=4)
        _ = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            aligner=aligner,
            ctx_length_student=4,
            ctx_length_teacher=4,
        )
        # Setting `pad_token` to the eos string is enough; HF tokenizers
        # update pad_token_id from that assignment. Our fake doesn't have
        # that machinery — so we just verify the assignment hook fires.
        assert student_tok.pad_token == student_tok.eos_token


class TestCollatorTextKey:
    def test_custom_text_key_is_honored(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=8, t_t=8)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            aligner=aligner,
            ctx_length_student=8,
            ctx_length_teacher=8,
            text_key="my_field",
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "my_field": "alt-text",
                }
            ]
        )
        # No raise = the key remap worked; check the output is non-trivial.
        assert int(out["input_lengths"][0]) == len("alt-text")
