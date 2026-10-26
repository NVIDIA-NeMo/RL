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
"""Hermetic CPU tests for ``CSDCrossTokenizerCollator``.

CSD batches are built by the shared ``CrossTokenizerCollator`` (tested in
``test_cross_tokenizer_collate.py``, ``test_cross_tokenizer_chat_eot.py`` and
``test_cross_tokenizer_native_chat.py``). These tests cover what CSD adds,
row ``sample_id`` values and MoE router-replay routes, and the batches the
shared collator builds with the options CSD's trainer turns on: special tokens
and overlength clipping in text mode, and whole-final-turn CE in chat mode.
Tokenizers are small local fast tokenizers; nothing is downloaded.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, Optional
from unittest.mock import patch

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
from nemo_rl.data.cross_tokenizer.csd_collate import CSDCrossTokenizerCollator
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import kd_data_processor
from tests.unit.data.test_cross_tokenizer_collate import _assert_batches_equal
from tests.unit.data.test_cross_tokenizer_native_chat import _messages, _native_pair

_TEXT = CrossTokenizerCollatorConfig(mode="text")
_CHAT = CrossTokenizerCollatorConfig(mode="chat")
_NATIVE_CHAT = CrossTokenizerCollatorConfig(
    mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
)

# ---------------------------------------------------------------------------
# Collator construction, mirroring the trainer's CSD options.
# ---------------------------------------------------------------------------


def _shared_options(config: CrossTokenizerCollatorConfig) -> dict[str, bool]:
    """Shared-collator options CSD's trainer turns on for ``config``."""
    options = {"reject_overlength_text": True}
    if config.mode == "text":
        options["add_special_tokens_in_text"] = True
    return options


def _collator_args(
    config: CrossTokenizerCollatorConfig,
    student: Any,
    teachers: list[Optional[Any]],
    *,
    ctx_length_student: int = 64,
    ctx_length_teachers: Optional[list[int]] = None,
    make_seq_div_by_student: int = 1,
    make_seq_div_by_teachers: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Constructor arguments common to the CSD and the shared collator.

    A ``None`` entry in ``teachers`` is a same-tokenizer teacher; every other
    entry is a cross-tokenizer teacher's tokenizer.
    """
    n = len(teachers)
    return {
        "config": config,
        "student_tokenizer": student,
        "teacher_tokenizers": [student if t is None else t for t in teachers],
        "aligners": [
            None
            if t is None
            else TokenAligner(student, t, projection_matrix_path="unused.pt")
            for t in teachers
        ],
        "ctx_length_student": ctx_length_student,
        "ctx_length_teachers": ctx_length_teachers or [64] * n,
        "drop_first_assistant_chunk_kl_by_teacher": [False] * n,
        "make_seq_div_by_student": make_seq_div_by_student,
        "make_seq_div_by_teachers": make_seq_div_by_teachers or [1] * n,
    }


def _csd_collator(
    config: CrossTokenizerCollatorConfig,
    student: Any,
    teachers: list[Optional[Any]],
    *,
    require_routed_experts: bool = False,
    **sizes: Any,
) -> CSDCrossTokenizerCollator:
    """CSD collator built with the options the trainer passes for ``config``."""
    return CSDCrossTokenizerCollator(
        **_collator_args(config, student, teachers, **sizes),
        require_routed_experts=require_routed_experts,
        clip_overlength_text=config.mode == "text",
        **_shared_options(config),
    )


def _shared_collator(
    config: CrossTokenizerCollatorConfig,
    student: Any,
    teachers: list[Optional[Any]],
    **sizes: Any,
) -> CrossTokenizerCollator:
    """Shared collator with the same arguments and options as ``_csd_collator``."""
    return CrossTokenizerCollator(
        **_collator_args(config, student, teachers, **sizes),
        **_shared_options(config),
    )


# ---------------------------------------------------------------------------
# Local fast tokenizers
# ---------------------------------------------------------------------------

_WORDS = ("alpha", "beta", "gamma", "delta", "epsilon")


def _text_tokenizer(*, bos: str, char_level: bool) -> PreTrainedTokenizerFast:
    """Fast tokenizer whose special tokens put ``bos`` at the start of each row.

    Word-level tokenizers emit one token per whitespace-separated word;
    character-level ones one token per non-space character.
    """
    pieces = sorted(set("".join(_WORDS))) if char_level else list(_WORDS)
    vocab = {
        token: i for i, token in enumerate(("[UNK]", "[PAD]", "<eos>", bos, *pieces))
    }
    backend = Tokenizer(
        BPE(vocab=vocab, merges=[], unk_token="[UNK]")
        if char_level
        else WordLevel(vocab, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single=f"{bos} $A", special_tokens=[(bos, vocab[bos])]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
        bos_token=bos,
    )


def _text_pair() -> tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    """Word-level student marked ``<s>`` and character-level teacher marked ``<t>``."""
    return (
        _text_tokenizer(bos="<s>", char_level=False),
        _text_tokenizer(bos="<t>", char_level=True),
    )


_CHAT_SPECIALS = ("<eot>", "<user>", "<assistant>", "<u>", "<a>")


def _chat_tokenizer(
    *, teacher: bool = False, suffix: str = "<eot>"
) -> PreTrainedTokenizerFast:
    """Word-level fast tokenizer with a chat template; role markers and EOT are special.

    Each message renders as its role marker, its content, then ``suffix``.
    The student marks roles ``<user>``/``<assistant>``, the teacher ``<u>``/``<a>``.
    """
    tokens = (
        "[UNK]",
        "[PAD]",
        "<eos>",
        *_CHAT_SPECIALS,
        "Question",
        "Answer",
        "Again",
        "Done",
    )
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
        additional_special_tokens=list(_CHAT_SPECIALS),
    )
    user, assistant = ("<u>", "<a>") if teacher else ("<user>", "<assistant>")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        f"{{{{ ({user!r} if message['role'] == 'user' else {assistant!r})"
        f" + message['content'] + {suffix!r} }}}}"
        "{% endfor %}"
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Rows, as kd_data_processor emits them
# ---------------------------------------------------------------------------

_SINGLE_TURN = [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"},
]
_MULTI_TURN = [
    *_SINGLE_TURN,
    {"role": "user", "content": "Again"},
    {"role": "assistant", "content": "Done"},
]


def _row(
    messages: list[dict[str, Any]],
    idx: int = 0,
    *,
    sample_id: Any = None,
    message_loss_mask: Optional[list[int]] = None,
) -> dict[str, Any]:
    """``kd_data_processor`` output; ``sample_id`` defaults to ``doc-{idx}``."""
    return kd_data_processor(
        {
            "messages": messages,
            "message_loss_mask": message_loss_mask,
            "sample_id": f"doc-{idx}" if sample_id is None else sample_id,
        },
        TaskDataSpec(),
        None,
        None,
        idx,
    )


def _text_row(text: str, idx: int = 0, **kwargs: Any) -> dict[str, Any]:
    return _row([{"role": "assistant", "content": text}], idx, **kwargs)


def _supervised(
    tokenizer: PreTrainedTokenizerFast,
    batch: Any,
    mask_key: str,
    *,
    ids_key: str = "input_ids",
    row: int = 0,
) -> list[str]:
    """Tokens of ``batch[ids_key][row]`` selected by ``batch[mask_key][row]``."""
    ids = batch[ids_key][row][batch[mask_key][row].bool()]
    return tokenizer.convert_ids_to_tokens(ids.tolist())


_ALIGNMENT_KEYS = {f"alignment_0_{f.name}" for f in dataclass_fields(AlignmentBatch)}
_STUDENT_KEYS = {
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
    "idx",
    "sample_id",
}
_TEACHER_0_KEYS = {
    "teacher_0_input_ids",
    "teacher_0_input_lengths",
    "teacher_0_token_mask",
    *_ALIGNMENT_KEYS,
}

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["text", "chat"])
def test_batch_is_the_shared_batch_plus_sample_ids(mode: str):
    # Teacher 0 is cross-tokenizer; teacher 1 shares the student tokenizer
    # and emits nothing extra.
    if mode == "text":
        config = _TEXT
        student, teacher = _text_pair()
        batch = [
            _text_row("alpha beta gamma", 0, sample_id="doc-b"),
            _text_row("delta", 1, sample_id=0),
        ]
    else:
        config = _CHAT
        student = _chat_tokenizer()
        teacher = _chat_tokenizer(teacher=True, suffix="\n<eot>")
        batch = [
            _row(_MULTI_TURN, 0, sample_id="doc-b"),
            _row(_SINGLE_TURN, 1, sample_id=0),
        ]

    out = _csd_collator(config, student, [teacher, None], make_seq_div_by_student=8)(
        batch
    )

    assert set(out.keys()) == _STUDENT_KEYS | _TEACHER_0_KEYS
    # Row identities keep their values (an integer 0 is a valid id) and order.
    assert out["sample_id"] == ["doc-b", 0]
    shared = _shared_collator(
        config, student, [teacher, None], make_seq_div_by_student=8
    )(batch)
    _assert_batches_equal({k: v for k, v in out.items() if k != "sample_id"}, shared)


def test_text_token_mask_is_the_attention_mask():
    student, teacher = _text_pair()
    out = _csd_collator(_TEXT, student, [teacher], make_seq_div_by_student=8)(
        [_text_row("alpha beta gamma", 0), _text_row("delta", 1)]
    )
    for prefix in ("", "teacher_0_"):
        width = out[f"{prefix}input_ids"].shape[1]
        attention = (
            torch.arange(width)[None, :] < out[f"{prefix}input_lengths"][:, None]
        )
        assert torch.equal(out[f"{prefix}token_mask"], attention.long()), prefix
    assert "kd_token_mask" not in out


def test_drop_first_flags_must_match_teacher_count():
    student, teacher = _text_pair()
    args = _collator_args(_TEXT, student, [teacher])
    args["drop_first_assistant_chunk_kl_by_teacher"] = []
    with pytest.raises(ValueError, match="drop_first_assistant_chunk_kl_by_teacher"):
        CSDCrossTokenizerCollator(
            **args, require_routed_experts=False, **_shared_options(_TEXT)
        )


# ---------------------------------------------------------------------------
# sample_id
# ---------------------------------------------------------------------------

_MISSING = object()


@pytest.mark.parametrize(
    "sample_id",
    [_MISSING, None, "", " \t\n"],
    ids=["missing", "none", "empty", "whitespace"],
)
def test_rows_without_a_sample_id_are_rejected_before_tokenization(sample_id):
    student, teacher = _text_pair()
    collator = _csd_collator(_TEXT, student, [teacher])
    batch = [_text_row("beta", 0), _text_row("delta", 1)]
    if sample_id is _MISSING:
        del batch[1]["sample_id"]
    else:
        batch[1]["sample_id"] = sample_id
    with (
        patch.object(CrossTokenizerCollator, "__call__") as shared_call,
        pytest.raises(
            ValueError, match=r"non-empty sample_id on every row.*\(batch row 1\)"
        ),
    ):
        collator(batch)
    shared_call.assert_not_called()


# ---------------------------------------------------------------------------
# Router replay
# ---------------------------------------------------------------------------


def _router_replay_collator(
    config: CrossTokenizerCollatorConfig, student: PreTrainedTokenizerFast
) -> CSDCrossTokenizerCollator:
    return _csd_collator(
        config,
        student,
        [None],
        make_seq_div_by_student=8,
        require_routed_experts=True,
    )


def test_router_replay_pads_rollout_routes_to_the_student_width():
    student, _ = _text_pair()
    text = "alpha beta gamma"
    # The rollout recorded the student tokenization, including its <s>.
    token_ids = student(text)["input_ids"]
    routes = torch.arange(len(token_ids) * 2 * 2).reshape(len(token_ids), 2, 2)
    row = _text_row(text)
    # Lists exercise the JSON-serialized rollout representation.
    row["message_log"][0]["token_ids"] = token_ids
    row["message_log"][0]["routed_experts"] = routes.tolist()

    out = _router_replay_collator(_TEXT, student)([row])

    assert out["input_ids"][0, : len(token_ids)].tolist() == token_ids
    assert out["routed_experts"].shape == (1, 8, 2, 2)
    assert torch.equal(out["routed_experts"][0, : len(token_ids)], routes)
    assert torch.count_nonzero(out["routed_experts"][0, len(token_ids) :]) == 0


def test_router_replay_fails_clearly_when_routes_are_unavailable():
    student, _ = _text_pair()
    row = _text_row("alpha beta")
    row["message_log"][0]["token_ids"] = student("alpha beta")["input_ids"]
    with pytest.raises(
        RuntimeError,
        match=r"policy\.router_replay\.enabled=true.*missing routed_experts",
    ):
        _router_replay_collator(_TEXT, student)([row])


def test_router_replay_rejects_routes_for_a_different_student_tokenization():
    student, _ = _text_pair()
    text = "alpha beta gamma"
    row = _text_row(text)
    # Recorded without the student's <s>, so positions would be off by one.
    token_ids = student(text, add_special_tokens=False)["input_ids"]
    row["message_log"][0]["token_ids"] = token_ids
    row["message_log"][0]["routed_experts"] = torch.zeros(len(token_ids), 2, 2)
    with pytest.raises(RuntimeError, match="do not match.*student tokenization"):
        _router_replay_collator(_TEXT, student)([row])


def test_router_replay_chat_concatenates_per_message_routes():
    student = _chat_tokenizer()
    messages: list[dict[str, Any]] = [dict(message) for message in _SINGLE_TURN]
    rendered = student.apply_chat_template(messages, tokenize=False)
    token_ids = student(rendered, add_special_tokens=False)["input_ids"]
    # <user> Question <eot> | <assistant> Answer <eot>
    split = 3
    routes = torch.arange(len(token_ids) * 2 * 2).reshape(len(token_ids), 2, 2)
    for message, part in zip(messages, (slice(None, split), slice(split, None))):
        message["token_ids"] = token_ids[part]
        message["routed_experts"] = routes[part]

    out = _router_replay_collator(_CHAT, student)([_row(messages)])

    assert out["routed_experts"].shape == (1, 8, 2, 2)
    assert torch.equal(out["routed_experts"][0, : len(token_ids)], routes)
    assert torch.count_nonzero(out["routed_experts"][0, len(token_ids) :]) == 0
    # The supervised content and EOT replay the final message's routes after
    # its <assistant> header.
    supervised = out["token_mask"][0].bool()
    assert torch.equal(out["routed_experts"][0, supervised], routes[split + 1 :])


# ---------------------------------------------------------------------------
# Text mode: special tokens, pad-to-longest, overlength clipping
# ---------------------------------------------------------------------------


def test_text_rows_carry_special_tokens_and_pad_to_the_longest_row():
    student, teacher = _text_pair()
    texts = ("alpha beta gamma", "delta")
    out = _csd_collator(
        _TEXT,
        student,
        [teacher],
        make_seq_div_by_student=8,
        make_seq_div_by_teachers=[4],
    )([_text_row(text, i) for i, text in enumerate(texts)])

    # Lengths count each tokenizer's marker: 1 + 3 words and 1 + 14 characters
    # for the longest row. Widths round the longest row up to the divisor,
    # not to the 64-token contexts.
    for prefix, tokenizer, lengths, width in (
        ("", student, [4, 2], 8),
        ("teacher_0_", teacher, [15, 6], 16),
    ):
        ids = out[f"{prefix}input_ids"]
        assert out[f"{prefix}input_lengths"].tolist() == lengths, prefix
        assert ids.shape == (2, width), prefix
        assert ids[:, 0].eq(tokenizer.bos_token_id).all(), prefix
        for row, (text, n) in enumerate(zip(texts, lengths)):
            assert ids[row, :n].tolist() == tokenizer(text)["input_ids"], prefix
            assert ids[row, n:].eq(tokenizer.pad_token_id).all(), prefix


@pytest.mark.parametrize(
    "sizes,text,prefix",
    [
        pytest.param(
            {"ctx_length_student": 4},
            "alpha beta gamma delta",
            "alpha beta gamma ",
            id="student",
        ),
        pytest.param(
            {"ctx_length_student": 6, "make_seq_div_by_student": 4},
            "alpha beta gamma delta",
            "alpha beta gamma ",
            id="student-after-rounding",
        ),
        pytest.param(
            {"ctx_length_teachers": [10, 64]},
            "alpha beta gamma",
            "alpha beta ",
            id="cross-tokenizer-teacher",
        ),
        pytest.param(
            {"ctx_length_teachers": [64, 4]},
            "alpha beta gamma delta",
            "alpha beta gamma ",
            id="same-tokenizer-teacher",
        ),
        pytest.param(
            {"ctx_length_teachers": [64, 6], "make_seq_div_by_teachers": [1, 4]},
            "alpha beta gamma delta",
            "alpha beta gamma ",
            id="same-tokenizer-teacher-after-rounding",
        ),
        pytest.param(
            {"ctx_length_teachers": [64, 6], "make_seq_div_by_student": 4},
            "alpha beta gamma delta",
            "alpha beta gamma ",
            id="same-tokenizer-teacher-after-student-rounding",
        ),
    ],
)
def test_text_overlength_rows_are_clipped_to_a_shared_prefix(sizes, text, prefix):
    # Token counts include each tokenizer's marker: "alpha beta gamma delta"
    # is 5 student tokens, "alpha beta gamma" 15 teacher tokens. Teacher 1
    # shares the student tokenization but has its own context and divisor.
    student, teacher = _text_pair()
    teachers = [teacher, None]
    out = _csd_collator(_TEXT, student, teachers, **sizes)(
        [_text_row("beta", 0), _text_row(text, 1)]
    )

    # The clipped batch is the one the shared collator builds from the prefix,
    # for the student and every teacher, and one more character does not fit.
    shared = _shared_collator(_TEXT, student, teachers, **sizes)
    expected = shared([_text_row("beta", 0), _text_row(prefix, 1)])
    _assert_batches_equal({k: v for k, v in out.items() if k != "sample_id"}, expected)
    with pytest.raises(ValueError, match="overlength rows are rejected$"):
        shared([_text_row(text[: len(prefix) + 1], 1)])


def test_text_row_without_a_fitting_prefix_raises():
    student, teacher = _text_pair()
    # Even one character tokenizes to the marker plus one token.
    collator = _csd_collator(_TEXT, student, [teacher], ctx_length_student=1)
    with pytest.raises(
        ValueError,
        match=r"^sample idx=0: no non-empty prefix of the text fits every "
        r"context length$",
    ):
        collator([_text_row("beta", 0)])


def test_clip_overlength_text_requires_text_mode():
    student, teacher = _chat_pair()
    with pytest.raises(ValueError, match="clip_overlength_text requires mode='text'"):
        CSDCrossTokenizerCollator(
            **_collator_args(_CHAT, student, [teacher]),
            require_routed_experts=False,
            clip_overlength_text=True,
        )


# ---------------------------------------------------------------------------
# Chat mode: CE and KD on the final assistant turn's content and EOT
# ---------------------------------------------------------------------------


def _chat_pair() -> tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    return _chat_tokenizer(), _chat_tokenizer(teacher=True, suffix="\n<eot>")


def _assert_kd_alignment_matches_masks(out: Any) -> None:
    """Aligned positions are exactly the KD positions on each side."""
    for side, mask_key in (
        ("student", "token_mask"),
        ("teacher", "teacher_0_token_mask"),
    ):
        chunks = out[f"alignment_0_{side}_chunk_id"][0]
        mask = out[mask_key][0]
        assert chunks[mask == 1].ge(0).all(), side
        assert chunks[mask == 0].eq(-1).all(), side


def test_chat_single_turn_masks():
    student, teacher = _chat_pair()
    out = _csd_collator(_CHAT, student, [teacher], make_seq_div_by_student=8)(
        [_row(_SINGLE_TURN)]
    )
    assert _supervised(student, out, "token_mask") == ["Answer", "<eot>"]
    assert "kd_token_mask" not in out
    assert _supervised(
        teacher, out, "teacher_0_token_mask", ids_key="teacher_0_input_ids"
    ) == ["Answer", "<eot>"]
    _assert_kd_alignment_matches_masks(out)


@pytest.mark.parametrize(
    "message_loss_mask",
    [None, [0, 0, 0, 1]],
    ids=["earlier-and-final-selected", "final-selected"],
)
def test_chat_multi_turn_supervises_only_the_final_turn(message_loss_mask):
    student, teacher = _chat_pair()
    row = _row(_MULTI_TURN, message_loss_mask=message_loss_mask)
    out = _csd_collator(_CHAT, student, [teacher])([row])

    # Earlier turns stay in the input as context.
    assert student.convert_ids_to_tokens(out["input_ids"][0].tolist()) == [
        "<user>",
        "Question",
        "<eot>",
        "<assistant>",
        "Answer",
        "<eot>",
        "<user>",
        "Again",
        "<eot>",
        "<assistant>",
        "Done",
        "<eot>",
    ]
    assert _supervised(student, out, "token_mask") == ["Done", "<eot>"]
    assert _supervised(
        teacher, out, "teacher_0_token_mask", ids_key="teacher_0_input_ids"
    ) == ["Done", "<eot>"]
    # The earlier assistant turn gets neither CE nor KD.
    assert not out["token_mask"][0, 3:6].any()
    _assert_kd_alignment_matches_masks(out)


@pytest.mark.parametrize(
    "message_loss_mask",
    [[0, 0, 0, 0], [0, 1, 0, 0]],
    ids=["nothing-selected", "earlier-turn-selected"],
)
def test_chat_selection_without_the_final_turn_is_rejected(message_loss_mask):
    student, teacher = _chat_pair()
    collator = _csd_collator(_CHAT, student, [teacher])
    with pytest.raises(
        ValueError,
        match=r"^student, sample idx=0: The final assistant message must be "
        r"selected for supervision$",
    ):
        collator([_row(_MULTI_TURN, message_loss_mask=message_loss_mask)])


def test_chat_same_tokenizer_teacher_bypasses_alignment():
    student = _chat_tokenizer()
    out = _csd_collator(_CHAT, student, [None])([_row(_SINGLE_TURN)])
    assert set(out.keys()) == _STUDENT_KEYS
    assert _supervised(student, out, "token_mask") == ["Answer", "<eot>"]


def test_chat_template_with_a_footer_supervises_content_and_eot():
    # A footer after the last message is context, not supervision.
    student = _chat_tokenizer()
    student.chat_template += "{{ '<eos>' }}"
    out = _csd_collator(_CHAT, student, [None])([_row(_SINGLE_TURN)])
    assert _supervised(student, out, "token_mask") == ["Answer", "<eot>"]


# ---------------------------------------------------------------------------
# Native thinking alignment
# ---------------------------------------------------------------------------


def test_native_thinking_uses_the_shared_native_path_and_adds_sample_ids():
    student, teacher, _ = _native_pair()
    sizes = {"ctx_length_student": 256, "ctx_length_teachers": [256]}
    row = {
        "idx": 7,
        "message_loss_mask": [0, 1],
        "message_log": _messages(),
        "loss_multiplier": 1.0,
        "sample_id": "doc-7",
    }
    out = _csd_collator(_NATIVE_CHAT, student, [teacher], **sizes)([row])
    shared = _shared_collator(_NATIVE_CHAT, student, [teacher], **sizes)([row])

    assert out["sample_id"] == ["doc-7"]
    assert "kd_token_mask" not in out
    _assert_batches_equal({k: v for k, v in out.items() if k != "sample_id"}, shared)
