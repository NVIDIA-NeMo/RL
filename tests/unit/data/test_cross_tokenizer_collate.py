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
teacher-indexed keys (``teacher_{i}_*`` / ``alignment_{i}_*``). Cross-tokenizer
cases use one teacher; reuse-limit tests use multiple teachers to prove that
each teacher's independent context is checked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, TokenAligner
from nemo_rl.data.cross_tokenizer_collate import CrossTokenizerCollator

# ---------------------------------------------------------------------------
# Fake tokenizer — deterministic, no HF dependency.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal tokenizer that satisfies ``CrossTokenizerCollator``'s contract.

    Tokenization is character-level with a fixed pad token at id 0.
    The production collator requests exact, untruncated tokenization one row at
    a time and performs dense padding itself.
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
        text,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=False,
    ):
        assert truncation is False
        ids = [2 + (ord(c) % (self.vocab_size - 2)) for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(k, k + 1) for k in range(len(ids))]
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
        "sample_id": f"test-source#{idx}",
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


# Expected keys consumed by xtoken_off_policy_distillation.py's per-teacher
# train_data packer — drift detector. Adding/removing keys here in lockstep
# with the trainer catches silent breakage. Teacher-indexed (single cross-
# tokenizer teacher at index 0).
_EXPECTED_COLLATOR_KEYS = {
    "input_ids",
    "input_lengths",
    "token_mask",
    "kd_token_mask",
    "sample_mask",
    "teacher_0_input_ids",
    "teacher_0_input_lengths",
    "teacher_0_token_mask",
    "alignment_0_pair_valid",
    "alignment_0_pair_is_correct",
    "alignment_0_student_exact_partition_mask",
    "alignment_0_teacher_exact_partition_mask",
    "alignment_0_student_chunk_id",
    "alignment_0_teacher_chunk_id",
    "alignment_0_num_chunks",
    "idx",
    "sample_id",
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
                student_tokenizer=student_tok,
                teacher_tokenizers=[teacher_tok],
                aligners=[aligner],
                ctx_length_student=8,
                ctx_length_teachers=[8],
                drop_first_assistant_chunk_kl_by_teacher=[],
            )


class TestRouterReplayMetadata:
    def test_preserves_and_pads_rollout_routes(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        text = "abc"
        token_ids = student_tok(text)["input_ids"]
        routed_experts = torch.arange(3 * 2 * 2).reshape(3, 2, 2)
        datum = _datum(text)
        # Use lists to exercise the JSON-serialized rollout representation.
        datum["message_log"][0]["token_ids"] = token_ids
        datum["message_log"][0]["routed_experts"] = routed_experts.tolist()
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[student_tok],
            aligners=[None],
            ctx_length_student=8,
            ctx_length_teachers=[8],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            make_seq_div_by_student=8,
            require_routed_experts=True,
        )

        out = collator([datum])

        assert out["routed_experts"].shape == (1, 8, 2, 2)
        assert torch.equal(out["routed_experts"][0, :3], routed_experts)
        assert torch.count_nonzero(out["routed_experts"][0, 3:]) == 0

    def test_enabled_replay_fails_clearly_when_routes_are_unavailable(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        datum = _datum("abc")
        datum["message_log"][0]["token_ids"] = student_tok("abc")["input_ids"]
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[student_tok],
            aligners=[None],
            ctx_length_student=8,
            ctx_length_teachers=[8],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            require_routed_experts=True,
        )

        with pytest.raises(
            RuntimeError,
            match=r"policy\.router_replay\.enabled=true.*missing routed_experts",
        ):
            collator([datum])

    def test_rejects_routes_from_a_different_student_tokenization(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        datum = _datum("abc")
        datum["message_log"][0]["token_ids"] = [99, 98, 97]
        datum["message_log"][0]["routed_experts"] = torch.zeros(3, 2, 2)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[student_tok],
            aligners=[None],
            ctx_length_student=8,
            ctx_length_teachers=[8],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            require_routed_experts=True,
        )

        with pytest.raises(RuntimeError, match="do not match.*student tokenization"):
            collator([datum])

    def test_chat_mode_preserves_per_message_rollout_routes(self):
        student_tok = FakeChatTokenizer(
            {"user": ("[U]", ""), "assistant": ("[A]", "[E]")}
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        rendered_prefix = student_tok.apply_chat_template(messages[:1])
        rendered = student_tok.apply_chat_template(messages)
        full_token_ids = student_tok(rendered, add_special_tokens=False)["input_ids"]
        prefix_len = len(rendered_prefix)
        full_routes = torch.arange(len(full_token_ids) * 2 * 2).reshape(
            len(full_token_ids), 2, 2
        )
        messages[0]["token_ids"] = full_token_ids[:prefix_len]
        messages[0]["routed_experts"] = full_routes[:prefix_len]
        messages[1]["token_ids"] = full_token_ids[prefix_len:]
        messages[1]["routed_experts"] = full_routes[prefix_len:]
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[student_tok],
            aligners=[None],
            ctx_length_student=32,
            ctx_length_teachers=[32],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
            require_routed_experts=True,
        )

        datum = _datum("")
        datum["sample_id"] = "chat#routes"
        datum["message_log"] = messages
        out = collator([datum])

        assert torch.equal(out["routed_experts"][0, : len(full_token_ids)], full_routes)


class TestCollatorShapes:
    def test_student_and_teacher_axes_independent(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=2, t_s=8, t_t=16, max_pairs=3)
        collator = CrossTokenizerCollator(
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
        assert out["alignment_0_num_chunks"].shape == (2,)

    def test_input_lengths_match_attention_mask_sum(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        aligner = _fake_aligner(b=1, t_s=8, t_t=8)
        collator = CrossTokenizerCollator(
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


class TestCollatorOverflow:
    def test_long_text_fails_with_sample_id_instead_of_truncating(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        ctx = 4
        aligner = _fake_aligner(b=1, t_s=ctx, t_t=ctx)
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[aligner],
            ctx_length_student=ctx,
            ctx_length_teachers=[ctx],
            drop_first_assistant_chunk_kl_by_teacher=[False],
        )
        datum = _datum("abcdefghij", 0)
        datum["sample_id"] = "corpus#17"
        with pytest.raises(ValueError, match="corpus#17.*truncation is forbidden"):
            collator([datum])

    @pytest.mark.parametrize(
        ("teacher_ctx", "teacher_divisor", "effective_length"),
        [(4, 1, 5), (6, 4, 8)],
    )
    def test_same_token_teacher_enforces_independent_text_context(
        self,
        teacher_ctx,
        teacher_divisor,
        effective_length,
    ):
        tokenizer = FakeTokenizer(vocab_size=32)
        collator = CrossTokenizerCollator(
            student_tokenizer=tokenizer,
            teacher_tokenizers=[tokenizer, tokenizer],
            aligners=[None, None],
            ctx_length_student=16,
            ctx_length_teachers=[16, teacher_ctx],
            drop_first_assistant_chunk_kl_by_teacher=[False, False],
            make_seq_div_by_teachers=[1, teacher_divisor],
        )
        datum = _datum("abcde")
        datum["sample_id"] = "same-token#text"

        with pytest.raises(
            ValueError,
            match=(
                "same-token#text.*teacher_1 context: exact token length 5, "
                f"effective length {effective_length}, capacity {teacher_ctx}"
            ),
        ):
            collator([datum])


class TestCollatorSequenceDivisibility:
    def test_seq_padded_up_to_multiple(self):
        student_tok = FakeTokenizer(vocab_size=32, prefix="s")
        teacher_tok = FakeTokenizer(vocab_size=24, prefix="t")
        # ctx=16, make_seq_div_by=8 → output sequence length = 8 for two
        # tokens. Verify both student and teacher pad independently.
        # Verify both student and teacher pad independently.
        aligner = _fake_aligner(b=1, t_s=16, t_t=12)
        collator = CrossTokenizerCollator(
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
        assert out["input_ids"].shape == (1, 8)
        assert out["token_mask"].shape == (1, 8)
        assert out["teacher_0_input_ids"].shape == (1, 4)
        assert out["teacher_0_token_mask"].shape == (1, 4)
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
                    "sample_id": "chat#alt-text",
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

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
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
        assert truncation is False
        input_ids = [ord(c) for c in chars]
        for _, suffix in self._scaffold.values():
            if not suffix:
                continue
            search_from = 0
            while (suffix_start := text.find(suffix, search_from)) >= 0:
                marker_offset = next(
                    (i for i, char in enumerate(suffix) if char.strip()), None
                )
                if marker_offset is not None:
                    input_ids[suffix_start + marker_offset] = self.eos_token_id
                search_from = suffix_start + len(suffix)
        out = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(k, k + 1) for k in range(len(chars))]
        return out

    def convert_ids_to_tokens(self, ids):
        return [chr(int(i)) for i in ids]


def _chat_aligner(student_tok, teacher_tok) -> TokenAligner:
    aligner = TokenAligner.__new__(TokenAligner)
    aligner.student_tokenizer = student_tok
    aligner.teacher_tokenizer = teacher_tok
    return aligner


class TestCollatorChatMode:
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
            mode="chat",
        )
        datum = {
            "loss_multiplier": 1.0,
            "idx": 0,
            "sample_id": "chat#content",
            "message_log": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello world"},
            ],
        }
        out = collator([datum])

        # The SFT CE mask covers the complete rendered assistant chunk, while
        # KD alignment below remains assistant content + EOT only.
        tok_mask = out["token_mask"][0].tolist()
        assert [p for p, v in enumerate(tok_mask) if v == 1] == list(range(5, 22))
        kd_mask = out["kd_token_mask"][0].tolist()
        assert [p for p, v in enumerate(kd_mask) if v == 1] == list(range(8, 20))
        assert out["student_semantic_regions"][0] == (
            (1, "assistant", "content", 8, 19),
            (1, "assistant", "eot", 19, 20),
        )

        # Alignment covers assistant content + the EOT token (char 19), scaffold
        # stays unaligned (chunk_id == -1).
        assert int(out["alignment_0_num_chunks"][0]) > 0
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
            mode="chat",
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "sample_id": "chat#same-tokenizer",
                    "message_log": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok"},
                    ],
                }
            ]
        )

        assert "teacher_0_input_ids" not in out
        assert not any(key.startswith("alignment_0_") for key in out.keys())

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
            mode="chat",
        )
        datum = {
            "loss_multiplier": 1.0,
            "idx": 0,
            "sample_id": "chat#teacher-index",
            "message_log": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
        }

        with (
            patch.object(
                aligner_0,
                "align_one_offset_per_asst",
                wraps=aligner_0.align_one_offset_per_asst,
            ) as align_0,
            patch.object(
                aligner_2,
                "align_one_offset_per_asst",
                wraps=aligner_2.align_one_offset_per_asst,
            ) as align_2,
        ):
            out = collator([datum])

        assert align_0.call_args.kwargs["drop_first_content_pair"] is False
        assert align_2.call_args.kwargs["drop_first_content_pair"] is True
        assert "teacher_0_input_ids" in out and "alignment_0_pair_valid" in out
        assert "teacher_2_input_ids" in out and "alignment_2_pair_valid" in out
        assert "teacher_1_input_ids" not in out
        assert not any(key.startswith("alignment_1_") for key in out.keys())

    def test_repeated_assistant_content_cannot_match_role_or_prior_turns(self):
        student_tok = FakeChatTokenizer(
            {"user": ("[U]", ""), "assistant": ("assistant|", "[E]")}
        )
        teacher_tok = FakeChatTokenizer(
            {"user": ("<U>", ""), "assistant": ("<assistant>", "<STOP>")}
        )
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=128,
            ctx_length_teachers=[128],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "sample_id": "chat#repeated-collision",
                    "message_log": [
                        {"role": "user", "content": "assistant"},
                        {"role": "assistant", "content": "assistant"},
                        {"role": "user", "content": "assistant"},
                        {"role": "assistant", "content": "assistant"},
                    ],
                }
            ]
        )

        assert out["student_semantic_regions"][0] == (
            (1, "assistant", "content", 22, 31),
            (1, "assistant", "eot", 31, 32),
            (3, "assistant", "content", 56, 65),
            (3, "assistant", "eot", 65, 66),
        )
        assert out["teacher_0_semantic_regions"][0] == (
            (1, "assistant", "content", 23, 32),
            (1, "assistant", "eot", 32, 33),
            (3, "assistant", "content", 61, 70),
            (3, "assistant", "eot", 70, 71),
        )
        assert torch.nonzero(out["kd_token_mask"][0]).flatten().tolist() == [
            *range(22, 32),
            *range(56, 66),
        ]
        student_chunk_ids = out["alignment_0_student_chunk_id"][0]
        assert torch.all(student_chunk_ids[:22] == -1)
        assert torch.all(student_chunk_ids[32:56] == -1)
        assert torch.all(student_chunk_ids[66:] == -1)

    def test_explicit_eot_skips_whitespace_and_pairs_different_suffixes(self):
        student_tok = FakeChatTokenizer({"assistant": ("[A]", " \n<END>")})
        teacher_tok = FakeChatTokenizer({"assistant": ("<A>", "\t<STOP>")})
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=64,
            ctx_length_teachers=[64],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "sample_id": "chat#eot",
                    "message_log": [{"role": "assistant", "content": "ok"}],
                }
            ]
        )

        # [A]ok<space><newline><END>: only content and the first non-whitespace
        # EOT token ('<') are semantic KD targets.
        assert torch.nonzero(out["kd_token_mask"][0]).flatten().tolist() == [3, 4, 7]
        assert out["student_semantic_regions"][0][-1] == (
            0,
            "assistant",
            "eot",
            7,
            8,
        )
        assert out["teacher_0_semantic_regions"][0][-1][2] == "eot"
        eot_chunk = int(out["alignment_0_student_chunk_id"][0, 7])
        assert eot_chunk >= 0
        assert bool(out["alignment_0_pair_is_correct"][0, eot_chunk])

    def test_missing_eot_fails_closed(self):
        student_tok = FakeChatTokenizer({"assistant": ("[A]", "")})
        teacher_tok = FakeChatTokenizer({"assistant": ("<A>", "<E>")})
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=64,
            ctx_length_teachers=[64],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        with pytest.raises(ValueError, match="post-content EOT.*student"):
            collator(
                [
                    {
                        "loss_multiplier": 1.0,
                        "idx": 0,
                        "sample_id": "chat#no-eot",
                        "message_log": [{"role": "assistant", "content": "answer"}],
                    }
                ]
            )

    def test_ambiguous_multiple_special_eot_tokens_fail_closed(self):
        class AmbiguousEotTokenizer(FakeChatTokenizer):
            def __call__(self, text, **kwargs):
                encoded = super().__call__(text, **kwargs)
                suffix_start = text.rfind("<E>")
                if suffix_start >= 0:
                    encoded["input_ids"][suffix_start + 1] = self.eos_token_id
                return encoded

        student_tok = AmbiguousEotTokenizer({"assistant": ("[A]", "<E>")})
        teacher_tok = FakeChatTokenizer({"assistant": ("<A>", "<E>")})
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=64,
            ctx_length_teachers=[64],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        with pytest.raises(ValueError, match=r"candidate_indices=\[3, 4\]"):
            collator(
                [
                    {
                        "loss_multiplier": 1.0,
                        "idx": 0,
                        "sample_id": "chat#ambiguous-eot",
                        "message_log": [{"role": "assistant", "content": ""}],
                    }
                ]
            )

    def test_empty_assistant_keeps_termination_only_target(self):
        student_tok = FakeChatTokenizer({"assistant": ("[A]", "[E]")})
        teacher_tok = FakeChatTokenizer({"assistant": ("<A>", "<E>")})
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=64,
            ctx_length_teachers=[64],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "sample_id": "chat#empty",
                    "message_log": [{"role": "assistant", "content": ""}],
                }
            ]
        )
        assert torch.nonzero(out["kd_token_mask"][0]).flatten().tolist() == [3]
        assert out["student_semantic_regions"][0] == ((0, "assistant", "eot", 3, 4),)

    def test_whitespace_assistant_content_cannot_match_header_whitespace(self):
        tokenizer = FakeChatTokenizer({"assistant": ("[A] \n", "[E]")})
        collator = CrossTokenizerCollator(
            student_tokenizer=tokenizer,
            teacher_tokenizers=[tokenizer],
            aligners=[None],
            ctx_length_student=32,
            ctx_length_teachers=[32],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "sample_id": "chat#whitespace",
                    "message_log": [{"role": "assistant", "content": " \n"}],
                }
            ]
        )

        assert torch.nonzero(out["kd_token_mask"][0]).flatten().tolist() == [5, 6, 7]
        assert out["student_semantic_regions"][0] == (
            (0, "assistant", "content", 5, 7),
            (0, "assistant", "eot", 7, 8),
        )

    def test_same_token_chat_rejects_slow_student_during_construction(self):
        student_tok = FakeChatTokenizer({"assistant": ("[A]", "[E]")})
        student_tok.is_fast = False
        with pytest.raises(ValueError, match="fast student tokenizer"):
            CrossTokenizerCollator(
                student_tokenizer=student_tok,
                teacher_tokenizers=[student_tok],
                aligners=[None],
                ctx_length_student=64,
                ctx_length_teachers=[64],
                drop_first_assistant_chunk_kl_by_teacher=[False],
                mode="chat",
            )

    def test_same_token_chat_emits_explicit_side_local_semantic_regions(self):
        tokenizer = FakeChatTokenizer({"assistant": ("[A]", "[E]")})
        collator = CrossTokenizerCollator(
            student_tokenizer=tokenizer,
            teacher_tokenizers=[tokenizer],
            aligners=[None],
            ctx_length_student=64,
            ctx_length_teachers=[64],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        out = collator(
            [
                {
                    "loss_multiplier": 1.0,
                    "idx": 0,
                    "sample_id": "chat#same",
                    "message_log": [{"role": "assistant", "content": "ok"}],
                }
            ]
        )
        assert out["teacher_0_semantic_regions"] == out["student_semantic_regions"]

    @pytest.mark.parametrize(
        ("teacher_ctx", "teacher_divisor", "content", "raw_length", "effective_length"),
        [(7, 1, "ok", 8, 8), (10, 4, "hey", 9, 12)],
    )
    def test_same_token_teacher_enforces_independent_chat_context(
        self,
        teacher_ctx,
        teacher_divisor,
        content,
        raw_length,
        effective_length,
    ):
        tokenizer = FakeChatTokenizer({"assistant": ("[A]", "[E]")})
        collator = CrossTokenizerCollator(
            student_tokenizer=tokenizer,
            teacher_tokenizers=[tokenizer, tokenizer],
            aligners=[None, None],
            ctx_length_student=16,
            ctx_length_teachers=[16, teacher_ctx],
            drop_first_assistant_chunk_kl_by_teacher=[False, False],
            make_seq_div_by_teachers=[1, teacher_divisor],
            mode="chat",
        )

        with pytest.raises(
            ValueError,
            match=(
                "same-token#chat.*teacher_1 context: exact post-template length "
                f"{raw_length}, effective length {effective_length}, "
                f"capacity {teacher_ctx}"
            ),
        ):
            collator(
                [
                    {
                        "loss_multiplier": 1.0,
                        "idx": 0,
                        "sample_id": "same-token#chat",
                        "message_log": [{"role": "assistant", "content": content}],
                    }
                ]
            )

    def test_teacher_post_template_overflow_names_teacher_side(self):
        student_tok = FakeChatTokenizer({"assistant": ("[A]", "[E]")})
        teacher_tok = FakeChatTokenizer({"assistant": ("<VERY-LONG-ASSISTANT>", "<E>")})
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=64,
            ctx_length_teachers=[8],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        with pytest.raises(ValueError, match="teacher_0 context.*truncation"):
            collator(
                [
                    {
                        "loss_multiplier": 1.0,
                        "idx": 0,
                        "sample_id": "chat#teacher-overflow",
                        "message_log": [{"role": "assistant", "content": "x"}],
                    }
                ]
            )

    def test_native_thinking_alignment_not_implemented(self):
        tok = FakeChatTokenizer({"assistant": ("", "")})
        aligner = _chat_aligner(tok, tok)
        with pytest.raises(NotImplementedError):
            CrossTokenizerCollator(
                student_tokenizer=tok,
                teacher_tokenizers=[tok],
                aligners=[aligner],
                ctx_length_student=32,
                ctx_length_teachers=[32],
                drop_first_assistant_chunk_kl_by_teacher=[False],
                mode="chat",
                include_thinking_in_loss=True,
                native_thinking_alignment=True,
            )

    def test_tools_reach_each_template_and_tool_call_only_fails_closed(self):
        class RecordingChatTokenizer(FakeChatTokenizer):
            def __init__(self, scaffold):
                super().__init__(scaffold)
                self.seen_tools = []

            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                self.seen_tools.append(kwargs.get("tools"))
                return super().apply_chat_template(messages, tokenize=tokenize)

        student_tok = RecordingChatTokenizer({"assistant": ("[A]", "[E]")})
        teacher_tok = RecordingChatTokenizer({"assistant": ("<A>", "<E>")})
        collator = CrossTokenizerCollator(
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
            aligners=[_chat_aligner(student_tok, teacher_tok)],
            ctx_length_student=64,
            ctx_length_teachers=[64],
            drop_first_assistant_chunk_kl_by_teacher=[False],
            mode="chat",
        )
        tools = [{"type": "function", "function": {"name": "weather"}}]
        datum = {
            "loss_multiplier": 1.0,
            "idx": 0,
            "sample_id": "chat#0",
            "tools": tools,
            "message_log": [{"role": "assistant", "content": "sunny"}],
        }
        collator([datum])
        assert student_tok.seen_tools and all(
            seen == tools for seen in student_tok.seen_tools
        )
        assert teacher_tok.seen_tools and all(
            seen == tools for seen in teacher_tok.seen_tools
        )

        datum["message_log"] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "weather"}}],
            }
        ]
        with pytest.raises(ValueError, match="chat#0.*tool_calls"):
            collator([datum])

        datum["message_log"][0]["content"] = " \n\t"
        with pytest.raises(ValueError, match="chat#0.*tool_calls"):
            collator([datum])
