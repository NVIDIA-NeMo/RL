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
from nemo_rl.data.xtoken.cross_tokenizer_collate import (
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
    padded to ``max_length``. Returns the dict shape HF fast tokenizers
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
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        all_ids = []
        all_mask = []
        all_offsets = []
        for t in texts:
            ids = [2 + (ord(c) % (self.vocab_size - 2)) for c in t][:max_length]
            mask = [1] * len(ids)
            offsets = [(k, k + 1) for k in range(len(ids))]
            if len(ids) < max_length:
                pad_n = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_n
                mask = mask + [0] * pad_n
                offsets = offsets + [(0, 0)] * pad_n
            all_ids.append(ids)
            all_mask.append(mask)
            all_offsets.append(offsets)
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
        make_seq_div_by_teachers=[3, 1],
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
# behaviors existed; the default batch must stay bitwise identical.
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

    @pytest.mark.parametrize(
        "texts,student_lengths,teacher_lengths",
        [
            # Row 6 exactly fills every context: 5 words, 26 characters.
            (("alpha beta gamma delta epsilon", "delta"), [5, 1], [26, 5]),
            # Every row is shorter than every context.
            (("beta", "delta gamma"), [1, 2], [4, 10]),
        ],
    )
    def test_rows_that_fit_match_default(self, texts, student_lengths, teacher_lengths):
        batch = [
            _datum(texts[0], idx=6),
            _datum(texts[1], idx=7, loss_multiplier=0.0),
        ]
        expected = _text_collator(ctx_length_student=5, ctx_length_teachers=[26, 5])(
            batch
        )
        actual = _text_collator(
            ctx_length_student=5,
            ctx_length_teachers=[26, 5],
            reject_overlength_text=True,
        )(batch)
        assert expected["input_lengths"].tolist() == student_lengths
        assert expected["teacher_0_input_lengths"].tolist() == teacher_lengths
        # Both sides pad to their context length, rounded up to the divisor.
        assert expected["input_ids"].shape == (2, 8)
        assert expected["teacher_0_input_ids"].shape == (2, 27)
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
    def test_seq_padded_up_to_multiple(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        # ctx=10, make_seq_div_by=8 → output sequence length = 16.
        # Verify both student and teacher pad independently.
        aligner = _fake_aligner(b=1, t_s=16, t_t=12)
        collator = CrossTokenizerCollator(
            config=CrossTokenizerCollatorConfig(),
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=10,
            ctx_length_teachers=[10],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            make_seq_div_by_student=8,
            make_seq_div_by_teachers=[4],
        )
        out = collator([_datum("hi", 0)])
        assert out["input_ids"].shape == (1, 16)
        assert out["token_mask"].shape == (1, 16)
        assert out["teacher_0_input_ids"].shape == (1, 12)
        assert out["teacher_0_token_mask"].shape == (1, 12)
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
