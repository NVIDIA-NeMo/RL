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
"""Assistant terminator supervision using local, real fast tokenizers."""

import pytest
import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.cross_tokenizer.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)
from nemo_rl.data.cross_tokenizer.native_chat import _render_and_tokenize_chat
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import kd_data_processor
from tests.unit.data.test_cross_tokenizer_collate import (
    _assert_batches_equal,
    _batch_checksums,
)


def _tokenizer(
    *, suffix: str, teacher: bool = False, pad_is_eot: bool = False
) -> PreTrainedTokenizerFast:
    specials = [
        "[UNK]",
        "[PAD]",
        "<eos>",
        "<eot>",
        "<user>",
        "<assistant>",
        "<u>",
        "<a>",
    ]
    tokens = specials + [
        "Question",
        "Answer",
        "Again",
        "Done",
        "<think>",
        "Why",
        "</think>",
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="<eot>" if pad_is_eot else "[PAD]",
        eos_token="<eos>",
        additional_special_tokens=specials[3:],
    )
    role = (
        "('u' if message['role'] == 'user' else 'a')" if teacher else "message['role']"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<' + " + role + " + '>' + message['content'] + " + repr(suffix) + " }}"
        "{% endfor %}"
    )
    return tokenizer


def _collator(
    student: PreTrainedTokenizerFast,
    teacher: PreTrainedTokenizerFast | None,
    *,
    student_length: int = 64,
    teacher_length: int = 64,
    **options: bool,
) -> CrossTokenizerCollator:
    return CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(mode="chat"),
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[
            None
            if teacher is None
            else TokenAligner(student, teacher, projection_matrix_path="unused.pt")
        ],
        ctx_length_student=student_length,
        ctx_length_teachers=[teacher_length],
        drop_first_assistant_chunk_kl_by_teacher=[False],
        make_seq_div_by_student=8,
        make_seq_div_by_teachers=[8],
        **options,
    )


def _datum(*, multiple_turns: bool = False, trailing_content: str = "") -> dict:
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer" + trailing_content},
    ]
    if multiple_turns:
        messages.extend(
            [
                {"role": "user", "content": "Again"},
                {"role": "assistant", "content": "Done"},
            ]
        )
    return kd_data_processor({"messages": messages}, TaskDataSpec(), None, None, 0)


@pytest.mark.parametrize("separator", ["", " \n\n"])
@pytest.mark.parametrize("trailing_content", ["", "  \n"])
@pytest.mark.parametrize("pad_is_eot", [False, True])
def test_masks_include_only_final_assistant_content_and_eot(
    separator: str, trailing_content: str, pad_is_eot: bool
):
    student = _tokenizer(suffix=separator + "<eot>\n", pad_is_eot=pad_is_eot)
    teacher = _tokenizer(suffix="\n<eot>", teacher=True, pad_is_eot=pad_is_eot)
    batch = _collator(student, teacher)(
        [_datum(multiple_turns=True, trailing_content=trailing_content)]
    )
    # Every turn has three tokens: role, content, EOT. Only the final assistant
    # content/EOT pair is scored. Earlier turns and padding share the EOT
    # id in some tokenizers, so enabling every occurrence of that id is wrong.
    expected = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
    assert torch.equal(batch["token_mask"], expected)
    assert torch.equal(batch["teacher_0_token_mask"], expected)
    assert batch["alignment_0_pair_valid"].sum().item() == 2
    for side in ("student", "teacher"):
        chunks = batch[f"alignment_0_{side}_chunk_id"][0]
        assert chunks[[10, 11]].tolist() == [0, 1]
        assert chunks[expected[0] == 0].eq(-1).all()


@pytest.mark.parametrize("student_length,teacher_length", [(5, 6), (6, 5)])
def test_one_sided_overflow_is_rejected(student_length: int, teacher_length: int):
    student = _tokenizer(suffix="<eot>")
    teacher = _tokenizer(suffix="\n<eot>", teacher=True)
    side = "student" if student_length < 6 else "teacher 0"
    with pytest.raises(ValueError, match=side + ".*overlength"):
        _collator(
            student,
            teacher,
            student_length=student_length,
            teacher_length=teacher_length,
        )([_datum()])


def test_template_without_terminators_does_not_supervise_next_role_header():
    student = _tokenizer(suffix="")
    teacher = _tokenizer(suffix="", teacher=True)
    batch = _collator(student, teacher)([_datum(multiple_turns=True)])
    expected = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]])
    assert torch.equal(batch["token_mask"], expected)
    assert torch.equal(batch["teacher_0_token_mask"], expected)
    assert batch["alignment_0_pair_valid"].sum().item() == 1
    assert batch["alignment_0_student_chunk_id"][0, 4] == -1


def test_eot_mask_does_not_extend_assistant_content_spans():
    tokenizer = _tokenizer(suffix=" \n<eot>")
    messages = _datum()["message_log"]
    doc = _render_and_tokenize_chat(
        tokenizer,
        messages,
        64,
        message_loss_mask=[0, 1],
        include_thinking_in_loss=False,
        native_thinking_alignment=False,
    )
    assert len(doc.assistant_spans) == len(doc.eot_indices) == 1
    eot_idx = doc.eot_indices[0]
    assert doc.input_ids[eot_idx] == tokenizer.convert_tokens_to_ids("<eot>")
    assert doc.assistant_mask[eot_idx] == 1
    assert doc.offsets[eot_idx][0] > doc.assistant_spans[0][1]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    assert rendered[slice(*doc.assistant_spans[0])] == "Answer"


def test_response_repeated_inside_terminator_keeps_eot_supervision():
    tokenizer = _tokenizer(suffix="<eot>")
    messages = [{"role": "assistant", "content": "eot"}]
    doc = _render_and_tokenize_chat(
        tokenizer,
        messages,
        64,
        message_loss_mask=[1],
        include_thinking_in_loss=False,
        native_thinking_alignment=False,
    )
    assert doc.eot_indices == [2]
    assert doc.input_ids[2] == tokenizer.convert_tokens_to_ids("<eot>")
    assert doc.assistant_mask == [0, 1, 1]


@pytest.mark.parametrize("trailing_content", ["", " \n"])
def test_eot_token_can_absorb_preceding_whitespace(trailing_content: str):
    tokenizer = _tokenizer(suffix=" \n<eot>")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken("<eot>", lstrip=True)]}
    )
    messages = [{"role": "assistant", "content": "Answer" + trailing_content}]
    doc = _render_and_tokenize_chat(
        tokenizer,
        messages,
        64,
        message_loss_mask=[1],
        include_thinking_in_loss=False,
        native_thinking_alignment=False,
    )
    assert doc.eot_indices == [2]
    assert doc.input_ids[2] == tokenizer.convert_tokens_to_ids("<eot>")
    assert doc.assistant_mask == [0, 1, 1]
    assert doc.offsets[2][0] <= doc.assistant_spans[0][1]


def _chat_snapshot_batch():
    """Default chat batch with cross-tokenizer and same-tokenizer teachers."""
    student = _tokenizer(suffix=" \n\n<eot>\n")
    teacher = _tokenizer(suffix="\n<eot>", teacher=True)
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(mode="chat"),
        student_tokenizer=student,
        teacher_tokenizers=[teacher, student],
        aligners=[
            TokenAligner(student, teacher, projection_matrix_path="unused.pt"),
            None,
        ],
        ctx_length_student=64,
        ctx_length_teachers=[64, 64],
        drop_first_assistant_chunk_kl_by_teacher=[False, False],
        make_seq_div_by_student=8,
        make_seq_div_by_teachers=[4, 1],
    )
    return collator(
        [
            _datum(multiple_turns=True),
            {**_datum(trailing_content="  \n"), "idx": 1, "loss_multiplier": 0.5},
        ]
    )


# Captured from the collator before the optional overlength and CE-turn
# behaviors existed; the default batch must stay bitwise identical.
_CHAT_SNAPSHOT_CHECKSUMS = {
    "input_ids": 3045364916,
    "input_lengths": 1242912239,
    "token_mask": 1070972743,
    "sample_mask": 586538992,
    "idx": 2005179788,
    "teacher_0_input_ids": 798308162,
    "teacher_0_input_lengths": 1242912239,
    "teacher_0_token_mask": 2796794980,
    "alignment_0_pair_valid": 1323192758,
    "alignment_0_pair_is_correct": 1323192758,
    "alignment_0_student_chunk_id": 3606849464,
    "alignment_0_teacher_chunk_id": 2913205021,
}


def test_default_chat_batch_matches_pinned_output():
    assert _batch_checksums(_chat_snapshot_batch()) == _CHAT_SNAPSHOT_CHECKSUMS


def test_reject_overlength_text_leaves_chat_batches_unchanged():
    student = _tokenizer(suffix="<eot>\n")
    teacher = _tokenizer(suffix="\n<eot>", teacher=True)
    batch = [_datum(multiple_turns=True)]
    _assert_batches_equal(
        _collator(student, teacher, reject_overlength_text=True)(batch),
        _collator(student, teacher)(batch),
    )


def _supervised_tokens(tokenizer, batch, mask_key: str, row: int = 0) -> list[str]:
    mask = batch[mask_key][row].bool()
    return tokenizer.convert_ids_to_tokens(batch["input_ids"][row][mask].tolist())


@pytest.mark.parametrize("cross_tokenizer", [False, True])
def test_ce_covers_whole_final_assistant_turn(cross_tokenizer: bool):
    student = _tokenizer(suffix="<eot>\n")
    teacher = _tokenizer(suffix="\n<eot>", teacher=True) if cross_tokenizer else None
    # Rows whose final turns start at different positions.
    batch = [_datum(multiple_turns=True), {**_datum(), "idx": 1}]
    default = _collator(student, teacher)(batch)
    actual = _collator(student, teacher, ce_covers_assistant_turn=True)(batch)

    # Earlier turns, including the earlier assistant turn, remain context.
    assert _supervised_tokens(student, actual, "token_mask", row=0) == [
        "<assistant>",
        "Done",
        "<eot>",
    ]
    assert _supervised_tokens(student, actual, "token_mask", row=1) == [
        "<assistant>",
        "Answer",
        "<eot>",
    ]
    assert _supervised_tokens(student, actual, "kd_token_mask", row=0) == [
        "Done",
        "<eot>",
    ]
    assert torch.equal(actual["kd_token_mask"], default["token_mask"])
    assert set(actual.keys()) == set(default.keys()) | {"kd_token_mask"}
    assert ("alignment_0_pair_valid" in actual) is cross_tokenizer
    for key, value in default.items():
        if key == "token_mask":
            continue
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[key], value), key
        else:
            assert actual[key] == value, key


def test_ce_turn_covers_reasoning_that_kd_excludes():
    student = _tokenizer(suffix="<eot>")
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think> Why </think> Answer"},
    ]
    batch = [kd_data_processor({"messages": messages}, TaskDataSpec(), None, None, 0)]
    default = _collator(student, None)(batch)
    actual = _collator(student, None, ce_covers_assistant_turn=True)(batch)
    assert _supervised_tokens(student, default, "token_mask") == ["Answer", "<eot>"]
    assert torch.equal(actual["kd_token_mask"], default["token_mask"])
    assert _supervised_tokens(student, actual, "token_mask") == [
        "<assistant>",
        "<think>",
        "Why",
        "</think>",
        "Answer",
        "<eot>",
    ]


def test_ce_turn_token_may_absorb_history_whitespace():
    student = _tokenizer(suffix="<eot>\n")
    student.add_special_tokens(
        {"additional_special_tokens": [AddedToken("<assistant>", lstrip=True)]}
    )
    collator = _collator(student, None, ce_covers_assistant_turn=True)
    datum = _datum(multiple_turns=True)
    doc = collator._render_document(student, datum, 64, "student")
    history = student.apply_chat_template(datum["message_log"][:-1], tokenize=False)
    first = collator._final_turn_start(datum, doc)
    # The role token also covers the newline that ends the history.
    assert doc.offsets[first][0] < len(history)
    assert _supervised_tokens(student, collator([datum]), "token_mask") == [
        "<assistant>",
        "Done",
        "<eot>",
    ]


def test_ce_turn_without_history_is_the_whole_rendering():
    student = _tokenizer(suffix="<eot>")
    datum = {
        "message_log": [{"role": "assistant", "content": "Answer"}],
        "message_loss_mask": [1],
        "loss_multiplier": 1.0,
        "idx": 9,
    }
    batch = _collator(student, None, ce_covers_assistant_turn=True)([datum])
    assert _supervised_tokens(student, batch, "token_mask") == [
        "<assistant>",
        "Answer",
        "<eot>",
    ]
    assert _supervised_tokens(student, batch, "kd_token_mask") == ["Answer", "<eot>"]


def test_ce_turn_rejects_history_that_is_not_a_prefix():
    student = _tokenizer(suffix="<eot>")
    # A footer after the last message is not part of the shorter rendering.
    student.chat_template += "{{ '<eos>' }}"
    default = _collator(student, None)([_datum()])
    assert _supervised_tokens(student, default, "token_mask") == ["Answer", "<eot>"]
    with pytest.raises(
        ValueError,
        match=r"^student, sample idx=0: rendering the messages before the final "
        r"assistant turn does not give a prefix of the full rendering",
    ):
        _collator(student, None, ce_covers_assistant_turn=True)([_datum()])


def test_ce_turn_rejects_token_that_carries_history_text():
    student = _tokenizer(suffix="<eot>")
    collator = _collator(student, None, ce_covers_assistant_turn=True)
    datum = _datum()
    doc = collator._render_document(student, datum, 64, "student")
    # Tokens: <user> Question <eot> | <assistant> Answer <eot>. Make the first
    # turn token also cover the history's <eot> text.
    doc.offsets[3] = (doc.offsets[2][0], doc.offsets[3][1])
    with pytest.raises(
        ValueError,
        match=r"^student, sample idx=0: token 3 crosses the final assistant turn",
    ):
        collator._final_turn_start(datum, doc)


def test_ce_turn_rejects_targets_before_the_turn():
    student = _tokenizer(suffix="<eot>")
    collator = _collator(student, None, ce_covers_assistant_turn=True)
    datum = _datum()
    doc = collator._render_document(student, datum, 64, "student")
    assert collator._final_turn_start(datum, doc) == 3
    doc.assistant_mask[1] = 1
    with pytest.raises(
        ValueError, match="assistant targets precede the final assistant turn"
    ):
        collator._final_turn_start(datum, doc)
