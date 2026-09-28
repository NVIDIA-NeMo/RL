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
from nemo_rl.data.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
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
    tokens = specials + ["Question", "Answer", "Again", "Done"]
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
    return {"message_log": messages, "loss_multiplier": 1.0, "idx": 0}


@pytest.mark.parametrize("separator", ["", " \n\n"])
@pytest.mark.parametrize("trailing_content", ["", "  \n"])
@pytest.mark.parametrize("pad_is_eot", [False, True])
def test_masks_include_each_assistant_eot_only(
    separator: str, trailing_content: str, pad_is_eot: bool
):
    student = _tokenizer(suffix=separator + "<eot>\n", pad_is_eot=pad_is_eot)
    teacher = _tokenizer(suffix="\n<eot>", teacher=True, pad_is_eot=pad_is_eot)
    batch = _collator(student, teacher)(
        [_datum(multiple_turns=True, trailing_content=trailing_content)]
    )
    # Every turn has three tokens: role, content, EOT. Only the two assistant
    # content/EOT pairs are scored. User terminators and padding share the EOT
    # id in some tokenizers, so enabling every occurrence of that id is wrong.
    expected = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
    assert torch.equal(batch["token_mask"], expected)
    assert torch.equal(batch["teacher_0_token_mask"], expected)
    assert batch["alignment_0_pair_valid"].sum().item() == 4
    for side in ("student", "teacher"):
        chunks = batch[f"alignment_0_{side}_chunk_id"][0]
        assert chunks[[4, 5, 10, 11]].tolist() == [0, 1, 2, 3]
        assert chunks[expected[0] == 0].eq(-1).all()


@pytest.mark.parametrize("length,expected_count", [(4, 0), (5, 1), (6, 2)])
def test_same_tokenizer_mask_respects_truncation(length: int, expected_count: int):
    tokenizer = _tokenizer(suffix=" \n<eot>\n", pad_is_eot=True)
    ids, _, mask, _, _ = CrossTokenizerCollator._render_and_tokenize_chat(
        tokenizer, _datum()["message_log"], length
    )
    assert len(ids) == length
    assert sum(mask) == expected_count


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
    expected = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 1]])
    assert torch.equal(batch["token_mask"], expected)
    assert torch.equal(batch["teacher_0_token_mask"], expected)
    assert batch["alignment_0_pair_valid"].sum().item() == 2
    assert batch["alignment_0_student_chunk_id"][0, 4] == -1


def test_eot_mask_does_not_extend_assistant_content_spans():
    tokenizer = _tokenizer(suffix=" \n<eot>")
    messages = _datum()["message_log"]
    ids, offsets, mask, spans, eot_indices = (
        CrossTokenizerCollator._render_and_tokenize_chat(tokenizer, messages, 64)
    )
    assert len(spans) == len(eot_indices) == 1
    eot_idx = eot_indices[0]
    assert ids[eot_idx] == tokenizer.convert_tokens_to_ids("<eot>")
    assert mask[eot_idx] == 1
    assert offsets[eot_idx][0] > spans[0][1]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    assert rendered[slice(*spans[0])] == "Answer"


def test_response_repeated_inside_terminator_keeps_eot_supervision():
    tokenizer = _tokenizer(suffix="<eot>")
    messages = [{"role": "assistant", "content": "eot"}]
    ids, _, mask, _, eot_indices = CrossTokenizerCollator._render_and_tokenize_chat(
        tokenizer, messages, 64
    )
    assert eot_indices == [2]
    assert ids[2] == tokenizer.convert_tokens_to_ids("<eot>")
    assert mask == [0, 1, 1]


@pytest.mark.parametrize("trailing_content", ["", " \n"])
def test_eot_token_can_absorb_preceding_whitespace(trailing_content: str):
    tokenizer = _tokenizer(suffix=" \n<eot>")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken("<eot>", lstrip=True)]}
    )
    messages = [{"role": "assistant", "content": "Answer" + trailing_content}]
    ids, offsets, mask, spans, eot_indices = (
        CrossTokenizerCollator._render_and_tokenize_chat(tokenizer, messages, 64)
    )
    assert eot_indices == [2]
    assert ids[2] == tokenizer.convert_tokens_to_ids("<eot>")
    assert mask == [0, 1, 1]
    assert offsets[2][0] <= spans[0][1]
