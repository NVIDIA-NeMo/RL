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

These tests pin the collator's *contract* (output keys, shapes, padding,
truncation) without needing real HF tokenizers or pre-captured artifacts.
The collator is multi-teacher: it takes per-teacher lists and emits
teacher-indexed keys (``teacher_{i}_*`` / ``alignment_{i}_*``). These tests
exercise the single-cross-tokenizer-teacher case (index 0).
"""

from __future__ import annotations

import zlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, TokenAligner
from nemo_rl.data.cross_tokenizer.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)

# ---------------------------------------------------------------------------
# Fake tokenizer — deterministic, no HF dependency.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal tokenizer that satisfies ``CrossTokenizerCollator``'s contract.

    Tokenization is character-level with a fixed pad token at id 0.
    Texts longer than ``max_length`` are truncated; shorter texts are
    padded to the batch's longest row. Returns the dict shape HF fast tokenizers
    produce (``input_ids``, ``attention_mask``, and ``offset_mapping`` when
    ``return_offsets_mapping=True``) when called with ``return_tensors="pt"``.
    Character-level tokenization makes offsets trivial: content token ``k``
    covers ``(k, k+1)``; padding covers ``(0, 0)``.
    """

    eos_token = "<eos>"

    def __init__(self, vocab_size: int, prefix: str = "tok") -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self._prefix = prefix

    def __call__(
        self,
        texts,
        padding,
        truncation,
        max_length,
        return_tensors,
        return_offsets_mapping=False,
        add_special_tokens=True,
    ):
        assert not add_special_tokens
        assert padding == "longest"
        assert truncation is True
        assert return_tensors == "pt"
        rows = [
            [2 + (ord(c) % (self.vocab_size - 2)) for c in t][:max_length]
            for t in texts
        ]
        width = max(len(ids) for ids in rows)
        all_ids = []
        all_mask = []
        all_offsets = []
        for ids in rows:
            pad_n = width - len(ids)
            all_ids.append(ids + [self.pad_token_id] * pad_n)
            all_mask.append([1] * len(ids) + [0] * pad_n)
            all_offsets.append([(k, k + 1) for k in range(len(ids))] + [(0, 0)] * pad_n)
        out = {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_mask, dtype=torch.long),
        }
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor(all_offsets, dtype=torch.long)
        return out

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
        "message_log": [{"role": "assistant", "content": text}],
        "length": len(text),
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
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
        student_chunk_id=torch.zeros((b, t_s), dtype=torch.long),
        teacher_chunk_id=torch.zeros((b, t_t), dtype=torch.long),
    )
    return aligner


def _batch_checksums(batch: Any) -> dict[str, int]:
    """CRC32 of every batch value: dtype, shape, and raw bytes for tensors."""
    checksums = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            payload = f"{value.dtype}{tuple(value.shape)}".encode()
            payload += value.contiguous().numpy().tobytes()
        else:
            payload = repr(value).encode()
        checksums[key] = zlib.crc32(payload)
    return checksums


def _assert_batches_equal(actual: Any, expected: Any) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert actual[key].dtype == value.dtype, key
            assert torch.equal(actual[key], value), key
        else:
            assert actual[key] == value, key


_WORDS = ("alpha", "beta", "gamma", "delta", "epsilon")


def _word_tokenizer(*, bos: bool = False) -> PreTrainedTokenizerFast:
    """Real fast tokenizer with one token per whitespace-separated word.

    With ``bos``, ``add_special_tokens=True`` starts each row with ``<bos>``;
    word ids are the same either way.
    """
    tokens = ("[UNK]", "[PAD]", "<eos>", *_WORDS) + (("<bos>",) if bos else ())
    vocab = {token: i for i, token in enumerate(tokens)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    if bos:
        backend.post_processor = TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
        bos_token="<bos>" if bos else None,
    )


def _char_tokenizer() -> PreTrainedTokenizerFast:
    """Real fast tokenizer with one token per non-space character."""
    characters = sorted(set("".join(_WORDS)))
    vocab = {
        token: i for i, token in enumerate(("[UNK]", "[PAD]", "<eos>", *characters))
    }
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
    )


def _text_collator(
    *,
    ctx_length_student: int,
    ctx_length_teachers: list[int],
    student: PreTrainedTokenizerFast | None = None,
    make_seq_div_by_teachers: tuple[int, int] = (3, 1),
    **options: bool,
) -> CrossTokenizerCollator:
    """Word-level student, character-level teacher 0, same-tokenizer teacher 1."""
    if student is None:
        student = _word_tokenizer()
    teacher = _char_tokenizer()
    return CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(),
        student_tokenizer=student,
        teacher_tokenizers=[teacher, student],
        aligners=[
            TokenAligner(student, teacher, projection_matrix_path="unused.pt"),
            None,
        ],
        ctx_length_student=ctx_length_student,
        ctx_length_teachers=ctx_length_teachers,
        drop_first_assistant_chunk_kl_by_teacher=[False, False],
        make_seq_div_by_student=4,
        make_seq_div_by_teachers=list(make_seq_div_by_teachers),
        **options,
    )


def _text_snapshot_batch() -> Any:
    """Default text batch with a row that each side truncates."""
    collator = _text_collator(ctx_length_student=4, ctx_length_teachers=[12, 4])
    return collator(
        [
            _datum("alpha beta gamma delta epsilon", idx=3),
            _datum("beta", idx=4, loss_multiplier=0.5),
        ]
    )


def _full_context_tokenize_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizerFast,
    ctx_length: int,
    make_seq_div_by: int,
    *,
    add_special_tokens: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Text tokenization padded to the full context length, rounded up to the divisor."""
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=ctx_length,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    pad = -ctx_length % make_seq_div_by
    return (
        torch.nn.functional.pad(
            encoded["input_ids"], (0, pad), value=tokenizer.pad_token_id
        ),
        torch.nn.functional.pad(encoded["attention_mask"], (0, pad)),
        torch.nn.functional.pad(encoded["offset_mapping"], (0, 0, 0, pad)),
    )


# Expected keys consumed by xtoken_off_policy_distillation.py's per-teacher
# train_data packer — drift detector. Adding/removing keys here in lockstep
# with the trainer catches silent breakage. Teacher-indexed (single cross-
# tokenizer teacher at index 0).
_EXPECTED_COLLATOR_KEYS = {
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
    "teacher_0_input_ids",
    "teacher_0_input_lengths",
    "teacher_0_token_mask",
    "alignment_0_pair_valid",
    "alignment_0_pair_is_correct",
    "alignment_0_student_chunk_id",
    "alignment_0_teacher_chunk_id",
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
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=8,
            ctx_length_teachers=[8],
            drop_first_assistant_chunk_kl_by_teacher=[False],
        )
        out = collator([_datum("hello", 0), _datum("world", 1)])
        assert _EXPECTED_COLLATOR_KEYS.issubset(set(out.keys()))

    def test_drop_first_flags_must_match_teacher_count(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=8, t_t=8)

        with pytest.raises(
            ValueError, match="drop_first_assistant_chunk_kl_by_teacher"
        ):
            CrossTokenizerCollator(
                config=CrossTokenizerCollatorConfig(),
                student_tokenizer=student_tok,
                teacher_tokenizers=[teacher_tok],
                aligners=[aligner],
                ctx_length_student=8,
                ctx_length_teachers=[8],
                drop_first_assistant_chunk_kl_by_teacher=[],
            )


class TestCollatorShapes:
    def test_student_and_teacher_axes_independent(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=2, t_s=8, t_t=16, max_pairs=3)
        collator = CrossTokenizerCollator(
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=8,
            ctx_length_teachers=[16],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            make_seq_div_by_student=8,
            make_seq_div_by_teachers=[16],
        )
        out = collator([_datum("ab", 0), _datum("cd", 1)])
        assert out["input_ids"].shape == (2, 8)
        assert out["token_mask"].shape == (2, 8)
        assert out["teacher_0_input_ids"].shape == (2, 16)
        assert out["teacher_0_token_mask"].shape == (2, 16)
        # alignment payload shapes come from the aligner mock.
        assert out["alignment_0_pair_valid"].shape == (2, 3)
        assert out["alignment_0_student_chunk_id"].shape == (2, 8)
        assert out["alignment_0_teacher_chunk_id"].shape == (2, 16)

    def test_input_lengths_match_attention_mask_sum(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=8, t_t=8)
        collator = CrossTokenizerCollator(
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=8,
            ctx_length_teachers=[8],
            drop_first_assistant_chunk_kl_by_teacher=[False],
        )
        out = collator([_datum("abc", 0)])  # 3 chars → 3 real tokens
        assert int(out["input_lengths"][0]) == 3
        assert int(out["token_mask"][0].sum()) == 3
        assert int(out["teacher_0_input_lengths"][0]) == 3


class TestCollatorTruncation:
    def test_long_text_truncates_trailing_tokens_not_drops_sample(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        ctx = 4
        aligner = _fake_aligner(b=1, t_s=ctx, t_t=ctx)
        collator = CrossTokenizerCollator(
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=ctx,
            ctx_length_teachers=[ctx],
            drop_first_assistant_chunk_kl_by_teacher=[False],
        )
        # 10 chars; ctx=4 → sample is kept, trailing 6 chars dropped.
        out = collator([_datum("abcdefghij", 0)])
        assert out["input_ids"].shape == (1, 4)
        # token_mask should be all 1s (4 real tokens, no padding).
        assert int(out["token_mask"][0].sum()) == 4
        assert int(out["input_lengths"][0]) == 4


# Captured from the collator before the optional overlength and CE-turn
# behaviors existed; the default batch must stay bitwise identical. Its first
# row fills each side's context, so padding to the longest row keeps the
# full-context width.
_TEXT_SNAPSHOT_CHECKSUMS = {
    "input_ids": 3661199532,
    "input_lengths": 868467641,
    "token_mask": 464442051,
    "sample_mask": 586538992,
    "idx": 1297290009,
    "teacher_0_input_ids": 548984487,
    "teacher_0_input_lengths": 137384594,
    "teacher_0_token_mask": 3787835366,
    "alignment_0_pair_valid": 64817739,
    "alignment_0_pair_is_correct": 13516310,
    "alignment_0_student_chunk_id": 1888554585,
    "alignment_0_teacher_chunk_id": 1481747482,
}


def test_default_text_batch_matches_pinned_output():
    batch = _text_snapshot_batch()
    assert int(batch["input_lengths"][0]) == 4
    assert int(batch["teacher_0_input_lengths"][0]) == 12
    assert _batch_checksums(batch) == _TEXT_SNAPSHOT_CHECKSUMS


class TestRejectOverlengthText:
    def test_student_overlength_row_raises(self):
        batch = [_datum("beta", idx=2), _datum("alpha beta gamma delta epsilon", idx=3)]
        truncated = _text_collator(ctx_length_student=4, ctx_length_teachers=[64, 4])(
            batch
        )
        assert truncated["input_lengths"].tolist() == [1, 4]
        collator = _text_collator(
            ctx_length_student=4,
            ctx_length_teachers=[64, 4],
            reject_overlength_text=True,
        )
        with pytest.raises(
            ValueError,
            match=r"^student, sample idx=3: text tokenization has 5 tokens, "
            r"exceeding context length 4; overlength rows are rejected$",
        ):
            collator(batch)

    def test_cross_tokenizer_teacher_overlength_row_raises(self):
        collator = _text_collator(
            ctx_length_student=8,
            ctx_length_teachers=[10, 8],
            reject_overlength_text=True,
        )
        with pytest.raises(
            ValueError,
            match=r"^teacher 0, sample idx=5: text tokenization has 14 tokens, "
            r"exceeding context length 10; overlength rows are rejected$",
        ):
            collator([_datum("beta", idx=4), _datum("alpha beta gamma", idx=5)])

    def test_same_tokenizer_teacher_overlength_row_raises(self):
        collator = _text_collator(
            ctx_length_student=8,
            ctx_length_teachers=[64, 2],
            reject_overlength_text=True,
        )
        with pytest.raises(
            ValueError,
            match=r"^teacher 1, sample idx=6: shared tokenization exceeds context "
            r"length 2; overlength rows are rejected$",
        ):
            collator([_datum("alpha beta gamma", idx=6)])

    def test_student_padded_width_exceeding_context_raises(self):
        # Five words fit a 5-token context but pad to 8 (a multiple of 4).
        batch = [_datum("alpha beta gamma delta epsilon", idx=8)]
        truncated = _text_collator(ctx_length_student=5, ctx_length_teachers=[27, 5])(
            batch
        )
        assert truncated["input_lengths"].tolist() == [5]
        assert truncated["input_ids"].shape == (1, 8)
        collator = _text_collator(
            ctx_length_student=5,
            ctx_length_teachers=[27, 5],
            reject_overlength_text=True,
        )
        with pytest.raises(
            ValueError,
            match=r"^student, sample idx=8: text tokenization has 5 tokens, padded "
            r"to 8 for a sequence-length multiple of 4, exceeding context length "
            r"5; overlength rows are rejected$",
        ):
            collator(batch)

    def test_cross_tokenizer_teacher_padded_width_exceeding_context_raises(self):
        # 26 characters fit a 26-token context but pad to 27 (a multiple of 3).
        collator = _text_collator(
            ctx_length_student=8,
            ctx_length_teachers=[26, 8],
            reject_overlength_text=True,
        )
        with pytest.raises(
            ValueError,
            match=r"^teacher 0, sample idx=8: text tokenization has 26 tokens, "
            r"padded to 27 for a sequence-length multiple of 3, exceeding context "
            r"length 26; overlength rows are rejected$",
        ):
            collator([_datum("alpha beta gamma delta epsilon", idx=8)])

    def test_same_tokenizer_teacher_padded_width_exceeding_context_raises(self):
        # Three words fit teacher 1's 3-token context but pad to its divisor 4.
        collator = _text_collator(
            ctx_length_student=4,
            ctx_length_teachers=[64, 3],
            make_seq_div_by_teachers=(3, 4),
            reject_overlength_text=True,
        )
        with pytest.raises(
            ValueError,
            match=r"^teacher 1, sample idx=6: shared tokenization has 3 tokens, "
            r"padded to 4 for a sequence-length multiple of 4, exceeding context "
            r"length 3; overlength rows are rejected$",
        ):
            collator([_datum("alpha beta gamma", idx=6)])

    @pytest.mark.parametrize(
        "texts,student_lengths,teacher_lengths,widths",
        [
            # Row 6 exactly fills every context: 4 words, 21 characters.
            (("alpha beta gamma epsilon", "delta"), [4, 1], [21, 5], (4, 21)),
            # Every row is shorter than every context.
            (("beta", "delta gamma"), [1, 2], [4, 10], (4, 12)),
        ],
    )
    def test_rows_that_fit_match_default(
        self, texts, student_lengths, teacher_lengths, widths
    ):
        batch = [
            _datum(texts[0], idx=6),
            _datum(texts[1], idx=7, loss_multiplier=0.0),
        ]
        expected = _text_collator(ctx_length_student=4, ctx_length_teachers=[21, 4])(
            batch
        )
        actual = _text_collator(
            ctx_length_student=4,
            ctx_length_teachers=[21, 4],
            reject_overlength_text=True,
        )(batch)
        assert expected["input_lengths"].tolist() == student_lengths
        assert expected["teacher_0_input_lengths"].tolist() == teacher_lengths
        # Both sides pad to their longest row, rounded up to the divisor.
        assert expected["input_ids"].shape == (2, widths[0])
        assert expected["teacher_0_input_ids"].shape == (2, widths[1])
        _assert_batches_equal(actual, expected)

    def test_ce_covers_assistant_turn_requires_chat_mode(self):
        with pytest.raises(
            ValueError, match="ce_covers_assistant_turn requires mode='chat'"
        ):
            _text_collator(
                ctx_length_student=8,
                ctx_length_teachers=[64, 8],
                ce_covers_assistant_turn=True,
            )


class TestAddSpecialTokensInText:
    _TEXTS = ("alpha beta gamma", "delta")

    def _batches(self) -> tuple[PreTrainedTokenizerFast, Any, Any]:
        """Option-off and option-on batches for a student that adds ``<bos>``."""
        student = _word_tokenizer(bos=True)
        batch = [_datum(text, idx=i) for i, text in enumerate(self._TEXTS)]
        off = _text_collator(
            ctx_length_student=8, ctx_length_teachers=[64, 8], student=student
        )(batch)
        on = _text_collator(
            ctx_length_student=8,
            ctx_length_teachers=[64, 8],
            student=student,
            add_special_tokens_in_text=True,
        )(batch)
        return student, off, on

    def test_student_rows_start_with_marker(self):
        student, off, on = self._batches()
        assert on["input_ids"][:, 0].eq(student.bos_token_id).all()
        assert on["token_mask"][:, 0].eq(1).all()
        assert torch.equal(on["input_lengths"], off["input_lengths"] + 1)
        for row, n in enumerate(off["input_lengths"].tolist()):
            assert torch.equal(
                on["input_ids"][row, 1 : n + 1], off["input_ids"][row, :n]
            )
            assert on["token_mask"][row, : n + 1].eq(1).all()
        _, _, offsets = CrossTokenizerCollator._tokenize_batch(
            list(self._TEXTS), student, 8, 4, add_special_tokens=True
        )
        assert offsets[:, 0].eq(0).all()
        # The character-level teacher adds no special tokens.
        for key in (
            "teacher_0_input_ids",
            "teacher_0_input_lengths",
            "teacher_0_token_mask",
        ):
            assert torch.equal(on[key], off[key]), key

    def test_marker_forms_its_own_chunk(self):
        _, off, on = self._batches()
        for row, n in enumerate(off["input_lengths"].tolist()):
            on_student = on["alignment_0_student_chunk_id"][row]
            off_student = off["alignment_0_student_chunk_id"][row]
            on_teacher = on["alignment_0_teacher_chunk_id"][row]
            off_teacher = off["alignment_0_teacher_chunk_id"][row]
            # Pair 0 is the marker alone; no teacher token joins it, so the
            # KD loss skips it.
            assert on["alignment_0_pair_valid"][row, 0]
            assert on_student[0] == 0 and on_student[1:].ne(0).all()
            assert on_teacher.ne(0).all()
            # Every other pair is the option-off pair, one student position
            # and one pair index later.
            assert off_student[:n].ge(0).all()
            assert torch.equal(on_student[1 : n + 1], off_student[:n] + 1)
            real = off["teacher_0_token_mask"][row].bool()
            assert torch.equal(on_teacher[real], off_teacher[real] + 1)
            pairs = off_student[:n].unique()
            for key in ("alignment_0_pair_valid", "alignment_0_pair_is_correct"):
                assert torch.equal(on[key][row, pairs + 1], off[key][row, pairs]), key

    def test_overlength_rejection_counts_marker(self):
        # Four words fill the student context; the marker makes a fifth token.
        batch = [_datum("alpha beta gamma epsilon", idx=9)]
        student = _word_tokenizer(bos=True)

        def collator(add_special_tokens_in_text: bool) -> CrossTokenizerCollator:
            return _text_collator(
                ctx_length_student=4,
                ctx_length_teachers=[21, 4],
                student=student,
                reject_overlength_text=True,
                add_special_tokens_in_text=add_special_tokens_in_text,
            )

        assert collator(False)(batch)["input_lengths"].tolist() == [4]
        with pytest.raises(
            ValueError,
            match=r"^student, sample idx=9: text tokenization has 5 tokens, "
            r"exceeding context length 4; overlength rows are rejected$",
        ):
            collator(True)(batch)

    def test_requires_text_mode(self):
        tok = FakeChatTokenizer({"assistant": ("[A]", "[E]")})
        with pytest.raises(
            ValueError, match="add_special_tokens_in_text requires mode='text'"
        ):
            CrossTokenizerCollator(
                config=CrossTokenizerCollatorConfig(mode="chat"),
                student_tokenizer=tok,
                teacher_tokenizers=[tok],
                aligners=[None],
                ctx_length_student=32,
                ctx_length_teachers=[32],
                drop_first_assistant_chunk_kl_by_teacher=[False],
                add_special_tokens_in_text=True,
            )


class TestCollatorSequenceDivisibility:
    @pytest.mark.parametrize(
        "texts,make_seq_div_by,widths",
        [
            # The longest row (6 characters) sets the width on each side.
            (("hi", "hello!"), (1, 1), (6, 6)),
            (("hi", "hello!"), (8, 4), (8, 8)),
            # 12 characters truncate to the 10-token context; the student
            # width rounds up past the context to a multiple of 8.
            (("hi", "hello world!"), (8, 4), (16, 12)),
        ],
    )
    def test_width_is_longest_row_rounded_up(self, texts, make_seq_div_by, widths):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=2, t_s=widths[0], t_t=widths[1])
        collator = CrossTokenizerCollator(
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=10,
            ctx_length_teachers=[10],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            make_seq_div_by_student=make_seq_div_by[0],
            make_seq_div_by_teachers=[make_seq_div_by[1]],
        )
        out = collator([_datum(text, i) for i, text in enumerate(texts)])
        lengths = [min(len(text), 10) for text in texts]
        for prefix, tokenizer, width in (
            ("", student_tok, widths[0]),
            ("teacher_0_", teacher_tok, widths[1]),
        ):
            ids = out[f"{prefix}input_ids"]
            mask = out[f"{prefix}token_mask"]
            assert ids.shape == mask.shape == (2, width)
            assert out[f"{prefix}input_lengths"].tolist() == lengths
            for row, n in enumerate(lengths):
                assert mask[row, :n].eq(1).all() and mask[row, n:].eq(0).all()
                assert ids[row, n:].eq(tokenizer.pad_token_id).all()


class TestTextPadsToLongestRow:
    _TEXTS = ("alpha beta gamma delta epsilon", "beta", "gamma delta")

    def test_tokenization_matches_full_context_padding(self):
        for tokenizer, ctx_length, make_seq_div_by in (
            (_word_tokenizer(), 16, 4),
            (_char_tokenizer(), 64, 3),
        ):
            new = CrossTokenizerCollator._tokenize_batch(
                list(self._TEXTS),
                tokenizer,
                ctx_length,
                make_seq_div_by,
                add_special_tokens=False,
            )
            old = _full_context_tokenize_batch(
                list(self._TEXTS),
                tokenizer,
                ctx_length,
                make_seq_div_by,
                add_special_tokens=False,
            )
            width = new[0].shape[1]
            assert width % make_seq_div_by == 0 and width < old[0].shape[1]
            for new_value, old_value in zip(new, old):
                assert torch.equal(new_value, old_value[:, :width])
            ids, mask, offsets = old
            assert ids[:, width:].eq(tokenizer.pad_token_id).all()
            assert mask[:, width:].eq(0).all() and offsets[:, width:].eq(0).all()

    def test_batch_matches_full_context_padding_at_real_positions(self):
        batch = [
            _datum(text, idx=i, loss_multiplier=1.0 - 0.5 * i)
            for i, text in enumerate(self._TEXTS)
        ]
        collator = _text_collator(ctx_length_student=16, ctx_length_teachers=[64, 16])
        new = collator(batch)
        with patch.object(
            CrossTokenizerCollator,
            "_tokenize_batch",
            staticmethod(_full_context_tokenize_batch),
        ):
            old = collator(batch)
        # Longest rows: 5 words pad to 8 (multiple of 4); 26 characters pad to
        # 27 (multiple of 3). Full-context padding gave 16 and 66.
        assert new["input_ids"].shape == (3, 8) and old["input_ids"].shape == (3, 16)
        assert new["teacher_0_input_ids"].shape == (3, 27)
        assert old["teacher_0_input_ids"].shape == (3, 66)
        assert set(new.keys()) == set(old.keys())
        for key in ("input_lengths", "sample_mask", "teacher_0_input_lengths"):
            assert torch.equal(new[key], old[key]), key
        assert new["idx"] == old["idx"]
        for prefix, chunk_key, pad_token_id in (
            (
                "",
                "alignment_0_student_chunk_id",
                collator.student_tokenizer.pad_token_id,
            ),
            (
                "teacher_0_",
                "alignment_0_teacher_chunk_id",
                collator.teacher_tokenizers[0].pad_token_id,
            ),
        ):
            width = new[f"{prefix}input_ids"].shape[1]
            for key, fill in (
                (f"{prefix}input_ids", pad_token_id),
                (f"{prefix}token_mask", 0),
                (chunk_key, -1),
            ):
                assert torch.equal(new[key], old[key][:, :width]), key
                assert old[key][:, width:].eq(fill).all(), key
        # Pairs holding real tokens keep their index and flags. The aligner
        # also pairs trailing padding with padding; those pairs hold no tokens
        # (chunk id -1), so the loss ignores them, and narrower rows have
        # fewer of them.
        for row in range(len(batch)):
            pairs = torch.cat(
                [
                    new["alignment_0_student_chunk_id"][row],
                    new["alignment_0_teacher_chunk_id"][row],
                ]
            ).unique()
            pairs = pairs[pairs >= 0]
            for key in ("alignment_0_pair_valid", "alignment_0_pair_is_correct"):
                assert torch.equal(new[key][row, pairs], old[key][row, pairs]), key

    def test_all_empty_batch_pads_to_the_divisor(self):
        collator = _text_collator(ctx_length_student=8, ctx_length_teachers=[16, 8])
        out = collator([_datum("", idx=0), _datum("", idx=1)])
        # No row has tokens, so each side pads to one sequence-length multiple.
        assert out["input_ids"].shape == (2, 4)
        assert out["teacher_0_input_ids"].shape == (2, 3)
        assert out["input_lengths"].tolist() == [0, 0]
        assert out["teacher_0_input_lengths"].tolist() == [0, 0]
        assert out["token_mask"].eq(0).all()
        assert out["teacher_0_token_mask"].eq(0).all()
        assert out["alignment_0_student_chunk_id"].eq(-1).all()
        assert out["alignment_0_teacher_chunk_id"].eq(-1).all()


class TestCollatorPadTokenFallback:
    def test_missing_pad_token_set_from_eos(self):
        # Tokenizer without a pad token id — collator must set
        # pad_token = eos_token in __init__.
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        student_tok.pad_token_id = None
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=4, t_t=4)
        _ = CrossTokenizerCollator(
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=4,
            ctx_length_teachers=[4],
            drop_first_assistant_chunk_kl_by_teacher=[False],
        )
        # Setting `pad_token` to the eos string is enough; HF tokenizers
        # update pad_token_id from that assignment. Our fake doesn't have
        # that machinery — so we just verify the assignment hook fires.
        assert student_tok.pad_token == student_tok.eos_token


class TestCollatorReadsMessageLog:
    def test_text_read_from_message_log_content(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=8, t_t=8)
        collator = CrossTokenizerCollator(
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=8,
            ctx_length_teachers=[8],
            drop_first_assistant_chunk_kl_by_teacher=[False],
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "message_log": [{"role": "assistant", "content": "alt-text"}],
                }
            ]
        )
        # The collator tokenizes message_log[0]["content"].
        assert int(out["input_lengths"][0]) == len("alt-text")


# ---------------------------------------------------------------------------
# Chat mode — char-level fake tokenizer with a chat template.
# ---------------------------------------------------------------------------


class FakeChatTokenizer:
    """Char-level tokenizer with a chat template (one token per char).

    ``apply_chat_template`` wraps each message in per-role scaffold markers so
    the student and teacher render the *same* content in *different* full-string
    coordinates. Token i == char i, so offsets are ``(i, i+1)`` and positions
    map directly to chars — which makes the assistant-span asserts readable.
    """

    is_fast = True

    def __init__(self, scaffold: dict) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.bos_token_id = None
        self.unk_token_id = None
        self.sep_token_id = None
        self.cls_token_id = None
        self.mask_token_id = None
        self.all_special_ids = [0, 1]
        self._scaffold = scaffold

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        parts = []
        for m in messages:
            pre, suf = self._scaffold.get(m["role"], ("", ""))
            parts.append(pre + m["content"] + suf)
        return "".join(parts)

    def __call__(
        self,
        text,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
        return_offsets_mapping=False,
    ):
        chars = list(text)
        if max_length is not None:
            chars = chars[:max_length]
        out = {"input_ids": [ord(c) for c in chars]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(k, k + 1) for k in range(len(chars))]
        return out

    def convert_ids_to_tokens(self, ids):
        return [chr(int(i)) for i in ids]

    def decode(self, ids, **kwargs):
        return "".join(chr(int(i)) for i in ids)


def _chat_aligner(student_tok, teacher_tok) -> TokenAligner:
    aligner = TokenAligner.__new__(TokenAligner)
    aligner.student_tokenizer = student_tok
    aligner.teacher_tokenizer = teacher_tok
    return aligner


class TestCollatorChatMode:
    def test_teacher_scoring_masks_only_final_assistant_content_and_terminator(self):
        student_tok = FakeChatTokenizer(
            {"system": ("[S]", ""), "user": ("[U]", ""), "assistant": ("[A]", "[E]")}
        )
        teacher_tok = FakeChatTokenizer(
            {
                "system": ("<system>", "</system>"),
                "user": ("<user>", "</user>"),
                "assistant": ("<assistant>", "<end>"),
            }
        )
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=256,
            ctx_length_teachers=[256],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            make_seq_div_by_student=8,
            make_seq_div_by_teachers=[16],
            config=CrossTokenizerCollatorConfig(mode="chat"),
        )
        conversations = [
            [
                {"role": "system", "content": "Follow instructions"},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Next question"},
                {"role": "assistant", "content": "Next answer"},
            ],
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Ready?"},
                {"role": "assistant", "content": "Yes"},
            ],
        ]
        message_masks = [[0, 0, 1, 0, 1], [0, 0, 1]]
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": i,
                    "message_loss_mask": message_masks[i],
                    "message_log": messages,
                }
                for i, messages in enumerate(conversations)
            ]
        )

        # Each model retains the full conversation as context, but scores only
        # final assistant content and EOT at its own positions. Different templates
        # prevent accidentally reusing the student's mask for the teacher.
        assert out["input_ids"].shape != out["teacher_0_input_ids"].shape
        for prefix, tokenizer in (("", student_tok), ("teacher_0_", teacher_tok)):
            for i, messages in enumerate(conversations):
                rendered = tokenizer.apply_chat_template(messages, tokenize=False)
                length = len(rendered)
                ids = out[f"{prefix}input_ids"][i]
                mask = out[f"{prefix}token_mask"][i]
                expected_mask = torch.zeros_like(mask)
                final_content = messages[-1]["content"]
                start = rendered.rindex(final_content)
                expected_mask[start : start + len(final_content) + 1] = 1

                assert out[f"{prefix}input_lengths"][i].item() == length
                assert ids[:length].tolist() == [ord(char) for char in rendered]
                assert torch.equal(mask, expected_mask)
                assert ids[length:].eq(tokenizer.pad_token_id).all()
                side = "teacher" if prefix else "student"
                chunks = out[f"alignment_0_{side}_chunk_id"][i]
                assert chunks[mask == 0].eq(-1).all()
                assert chunks[mask == 1].ge(0).all()

            # The shorter conversation has padding in both tokenizations.
            assert out[f"{prefix}input_lengths"][1] < out[f"{prefix}input_ids"].shape[1]

    def test_chat_aligns_only_assistant_content(self):
        # Same content, different scaffold => different full-string coordinates.
        student_tok = FakeChatTokenizer(
            {"user": ("[U]", ""), "assistant": ("[A]", "[E]")}
        )
        teacher_tok = FakeChatTokenizer(
            {"user": ("<usr>", ""), "assistant": ("<asst>", "<end>")}
        )
        aligner = _chat_aligner(student_tok, teacher_tok)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=64,
            ctx_length_teachers=[64],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            config=CrossTokenizerCollatorConfig(mode="chat"),
        )
        datum = {
            "loss_multiplier": 1.0,
            "idx": 0,
            "message_loss_mask": [0, 1],
            "message_log": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello world"},
            ],
        }
        out = collator([datum])

        # student render "[U]hi[A]Hello world[E]" -> assistant content chars [8,19)
        tok_mask = out["token_mask"][0].tolist()
        assert [p for p, v in enumerate(tok_mask) if v == 1] == list(range(8, 20))

        # Alignment covers assistant content + the EOT token (char 19), scaffold
        # stays unaligned (chunk_id == -1).
        assert out["alignment_0_pair_valid"][0].any()
        s_chunk = out["alignment_0_student_chunk_id"][0].tolist()
        assert all(s_chunk[p] != -1 for p in range(8, 20)), s_chunk
        assert all(s_chunk[p] == -1 for p in list(range(0, 8)) + [20, 21]), s_chunk

    def test_same_tokenizer_teacher_bypasses_alignment(self):
        tok = FakeChatTokenizer({"user": ("[U]", ""), "assistant": ("[A]", "[E]")})
        collator = CrossTokenizerCollator(
            student_tokenizer=tok,
            teacher_tokenizers=[tok],
            aligners=[None],
            ctx_length_student=32,
            ctx_length_teachers=[32],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            config=CrossTokenizerCollatorConfig(mode="chat"),
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "message_loss_mask": [0, 1],
                    "message_log": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok"},
                    ],
                }
            ]
        )

        assert not any(
            key.startswith(("teacher_0_", "alignment_0_")) for key in out.keys()
        )

    def test_drop_first_flag_uses_original_teacher_index(self):
        student_tok = FakeChatTokenizer(
            {"user": ("[U]", ""), "assistant": ("[A]", "[E]")}
        )
        teacher_0_tok = FakeChatTokenizer(
            {"user": ("<u0>", ""), "assistant": ("<a0>", "<e0>")}
        )
        teacher_2_tok = FakeChatTokenizer(
            {"user": ("<u2>", ""), "assistant": ("<a2>", "<e2>")}
        )
        aligner_0 = _chat_aligner(student_tok, teacher_0_tok)
        aligner_2 = _chat_aligner(student_tok, teacher_2_tok)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_0_tok, student_tok, teacher_2_tok],
            aligners=[aligner_0, None, aligner_2],
            ctx_length_student=64,
            ctx_length_teachers=[64, 64, 64],
            drop_first_assistant_chunk_kl_by_teacher=[False, False, True],
            config=CrossTokenizerCollatorConfig(mode="chat"),
        )
        datum = {
            "loss_multiplier": 1.0,
            "idx": 0,
            "message_loss_mask": [0, 1],
            "message_log": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
        }

        with (
            patch.object(
                aligner_0,
                "align_chat",
                wraps=aligner_0.align_chat,
            ) as align_0,
            patch.object(
                aligner_2,
                "align_chat",
                wraps=aligner_2.align_chat,
            ) as align_2,
        ):
            out = collator([datum])

        assert align_0.call_args.kwargs["drop_first_content_pair"] is False
        assert align_2.call_args.kwargs["drop_first_content_pair"] is True
        assert "teacher_0_input_ids" in out and "alignment_0_pair_valid" in out
        assert "teacher_2_input_ids" in out and "alignment_2_pair_valid" in out
        assert not any(
            key.startswith(("teacher_1_", "alignment_1_")) for key in out.keys()
        )

    def test_native_thinking_alignment_rejects_unsupported_layout(self):
        tok = FakeChatTokenizer({"assistant": ("", "")})
        aligner = _chat_aligner(tok, tok)
        with pytest.raises(ValueError, match="unsupported native layout"):
            collator = CrossTokenizerCollator(
                student_tokenizer=tok,
                teacher_tokenizers=[tok],
                aligners=[aligner],
                ctx_length_student=32,
                ctx_length_teachers=[32],
                drop_first_assistant_chunk_kl_by_teacher=[False],
                config=CrossTokenizerCollatorConfig(
                    mode="chat",
                    include_thinking_in_loss=True,
                    native_thinking_alignment=True,
                ),
            )
            collator(
                [
                    {
                        "idx": 0,
                        "loss_multiplier": 1.0,
                        "message_loss_mask": [1],
                        "message_log": [{"role": "assistant", "content": "hello"}],
                    }
                ]
            )
