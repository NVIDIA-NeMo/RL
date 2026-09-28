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

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)
from nemo_rl.data.native_chat import (
    _prepare_native_thinking_messages,
    _render_and_tokenize_chat,
    _render_chat_text,
)


class _NativeThinkingTokenizer:
    """Small fast-tokenizer stand-in with exact character offsets."""

    is_fast = True
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def __init__(self, style: str, merged_pieces: tuple[str, ...] = ()) -> None:
        self.style = style
        self.truncation_side = "right"
        self.merged_pieces = tuple(sorted(merged_pieces, key=len, reverse=True))
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self.seen_tools: list[Any] = []

    def _id(self, piece: str) -> int:
        if piece not in self._piece_to_id:
            token_id = len(self._piece_to_id) + 1
            self._piece_to_id[piece] = token_id
            self._id_to_piece[token_id] = piece
        return self._piece_to_id[piece]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        rendered = []
        tools = kwargs.get("tools")
        self.seen_tools.append(tools)
        if tools:
            rendered.append("<|im_start|>system\n<tools>schema</tools><|im_end|>\n")
        for message in messages:
            role = message["role"]
            content = str(message.get("content", "") or "").strip()
            if role == "assistant":
                serialized_calls = []
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function", tool_call)
                    name = function["name"]
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if self.style == "nano":
                        parameters = "".join(
                            f"<parameter={key}>\n{value}\n</parameter>\n"
                            for key, value in arguments.items()
                        )
                        serialized_calls.append(
                            "<tool_call>\n"
                            f"<function={name}>\n{parameters}</function>\n"
                            "</tool_call>"
                        )
                    elif self.style == "qwen":
                        payload = json.dumps(
                            {"name": name, "arguments": arguments}, sort_keys=True
                        )
                        serialized_calls.append(f"<tool_call>\n{payload}\n</tool_call>")
                    else:
                        raise ValueError(self.style)
                if serialized_calls:
                    content = "\n".join(filter(None, (content, *serialized_calls)))
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    if self.style == "nano":
                        content = f"<think>\n{reasoning}</think>{content}"
                    elif self.style == "qwen":
                        content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                    else:
                        raise ValueError(self.style)
            rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(rendered)

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        position = 0
        while position < len(text):
            piece = next(
                (
                    candidate
                    for candidate in self.merged_pieces
                    if text.startswith(candidate, position)
                ),
                text[position],
            )
            ids.append(self._id(piece))
            offsets.append((position, position + len(piece)))
            position += len(piece)
        if kwargs.get("truncation") and len(ids) > kwargs["max_length"]:
            max_length = int(kwargs["max_length"])
            if self.truncation_side == "left":
                ids = ids[-max_length:]
                offsets = offsets[-max_length:]
            else:
                ids = ids[:max_length]
                offsets = offsets[:max_length]
        return {"input_ids": ids, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, token_ids: int | list[int]):
        if isinstance(token_ids, list):
            return [self._id_to_piece[int(token_id)] for token_id in token_ids]
        return self._id_to_piece[int(token_ids)]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self._id_to_piece[int(token_id)] for token_id in token_ids)


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think>reason</think>answer"},
    ]


def _native_pair() -> tuple[
    _NativeThinkingTokenizer,
    _NativeThinkingTokenizer,
    TokenAligner,
]:
    special = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    student = _NativeThinkingTokenizer("nano", special)
    teacher = _NativeThinkingTokenizer("qwen", special + ("reason", "answer"))
    aligner = TokenAligner(
        student,
        teacher,
        projection_matrix_path="unused",
    )
    return student, teacher, aligner


def _decode_valid_spans(tokenizer, input_ids, chunk_ids, valid) -> str:
    return tokenizer.decode(
        [
            token
            for token, chunk in zip(input_ids, chunk_ids)
            if chunk >= 0 and valid[chunk]
        ]
    )


def test_prepare_native_thinking_messages_preserves_source() -> None:
    messages = _messages()
    prepared = _prepare_native_thinking_messages(messages)

    assert prepared[1]["reasoning_content"] == "reason"
    assert prepared[1]["content"] == "answer"
    assert "reasoning_content" not in messages[1]


def test_native_thinking_alignment_excludes_model_specific_whitespace() -> None:
    student, teacher, aligner = _native_pair()
    student_doc = _render_and_tokenize_chat(
        student,
        _messages(),
        256,
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    teacher_doc = _render_and_tokenize_chat(
        teacher,
        _messages(),
        256,
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    assert student_doc is not None
    assert teacher_doc is not None

    pairs = aligner.align_one_offset_per_asst(
        student_doc.input_ids,
        student_doc.offsets,
        student_doc.assistant_spans,
        teacher_doc.input_ids,
        teacher_doc.offsets,
        teacher_doc.assistant_spans,
        student_asst_mask=student_doc.assistant_mask,
        teacher_asst_mask=teacher_doc.assistant_mask,
        student_alignment_regions=student_doc.alignment_regions,
        teacher_alignment_regions=teacher_doc.alignment_regions,
        student_eot_indices=student_doc.eot_indices,
        teacher_eot_indices=teacher_doc.eot_indices,
    )
    student_text = "".join(
        student.decode(student_doc.input_ids[pair.s_start : pair.s_end])
        for pair in pairs
    )
    teacher_text = "".join(
        teacher.decode(teacher_doc.input_ids[pair.t_start : pair.t_end])
        for pair in pairs
    )

    assert student_text == "reason</think>answer<|im_end|>"
    assert teacher_text == student_text
    assert all(pair.is_correct for pair in pairs)


def test_native_thinking_alignment_flags_wrong_teacher_eot_position() -> None:
    student, teacher, aligner = _native_pair()
    student_doc = _render_and_tokenize_chat(
        student,
        _messages(),
        256,
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    teacher_doc = _render_and_tokenize_chat(
        teacher,
        _messages(),
        256,
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    assert student_doc is not None
    assert teacher_doc is not None

    pairs = aligner.align_one_offset_per_asst(
        student_doc.input_ids,
        student_doc.offsets,
        student_doc.assistant_spans,
        teacher_doc.input_ids,
        teacher_doc.offsets,
        teacher_doc.assistant_spans,
        student_asst_mask=student_doc.assistant_mask,
        teacher_asst_mask=teacher_doc.assistant_mask,
        student_alignment_regions=student_doc.alignment_regions,
        teacher_alignment_regions=teacher_doc.alignment_regions,
        student_eot_indices=student_doc.eot_indices,
        teacher_eot_indices=[teacher_doc.eot_indices[0] - 1],
    )

    assert (
        student.decode(student_doc.input_ids[pairs[-1].s_start : pairs[-1].s_end])
        == "<|im_end|>"
    )
    assert (
        teacher.decode(teacher_doc.input_ids[pairs[-1].t_start : pairs[-1].t_end])
        != "<|im_end|>"
    )
    assert not pairs[-1].is_correct


def test_native_collator_wires_semantic_regions_and_tools() -> None:
    student, teacher, aligner = _native_pair()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    collator = CrossTokenizerCollator(
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=256,
        ctx_length_teachers=[256],
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator(
        [
            {
                "idx": 7,
                "message_log": _messages(),
                "tools": tools,
                "loss_multiplier": 1.0,
            }
        ]
    )

    assert student.seen_tools[-1] is tools
    assert teacher.seen_tools[-1] is tools
    valid = result["alignment_0_pair_valid"][0]
    student_text = _decode_valid_spans(
        student,
        result["input_ids"][0].tolist(),
        result["alignment_0_student_chunk_id"][0].tolist(),
        valid.tolist(),
    )
    teacher_text = _decode_valid_spans(
        teacher,
        result["teacher_0_input_ids"][0].tolist(),
        result["alignment_0_teacher_chunk_id"][0].tolist(),
        valid.tolist(),
    )
    assert student_text == "reason</think>answer<|im_end|>"
    assert teacher_text == student_text


def test_native_collator_supervises_corrected_canonical_tool_call_only() -> None:
    student, teacher, aligner = _native_pair()
    call = (
        "<tool_call>\n<function=lookup>\n<parameter=query>\nstatus\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    messages = [
        {"role": "user", "content": "check"},
        {
            "role": "assistant",
            "content": f"<think>need tool</think>{call}",
        },
        {"role": "tool", "content": "secret tool result"},
        {"role": "assistant", "content": "<think>done</think>final"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    collator = CrossTokenizerCollator(
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=512,
        ctx_length_teachers=[512],
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator(
        [
            {
                "idx": 12,
                "message_log": messages,
                "tools": tools,
                "loss_multiplier": 1.0,
            }
        ]
    )

    supervised = "".join(
        student.decode([token_id])
        for token_id, include in zip(
            result["input_ids"][0].tolist(), result["token_mask"][0].tolist()
        )
        if include
    )
    assert "need tool" in supervised
    assert "<function=lookup>" in supervised
    assert "status" in supervised
    assert "final" in supervised
    assert "secret tool result" not in supervised
    assert "<tools>schema</tools>" not in supervised

    valid = result["alignment_0_pair_valid"][0]
    student_aligned = _decode_valid_spans(
        student,
        result["input_ids"][0].tolist(),
        result["alignment_0_student_chunk_id"][0].tolist(),
        valid.tolist(),
    )
    teacher_aligned = _decode_valid_spans(
        teacher,
        result["teacher_0_input_ids"][0].tolist(),
        result["alignment_0_teacher_chunk_id"][0].tolist(),
        valid.tolist(),
    )
    assert "<function=lookup>" in student_aligned
    assert "status" in student_aligned
    assert teacher_aligned == student_aligned


def test_kd_alignment_regions_filter_to_answer_and_eot() -> None:
    student, teacher, aligner = _native_pair()
    collator = CrossTokenizerCollator(
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=256,
        ctx_length_teachers=[256],
        config=CrossTokenizerCollatorConfig(
            mode="chat",
            include_thinking_in_loss=True,
            native_thinking_alignment=True,
            kd_alignment_regions=["answer", "eot"],
        ),
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator([{"idx": 7, "message_log": _messages(), "loss_multiplier": 1.0}])

    valid = result["alignment_0_pair_valid"][0]
    student_text = _decode_valid_spans(
        student,
        result["input_ids"][0].tolist(),
        result["alignment_0_student_chunk_id"][0].tolist(),
        valid.tolist(),
    )
    assert student_text == "answer<|im_end|>"


def test_native_message_loss_mask_excludes_context_only_turn() -> None:
    student, teacher, aligner = _native_pair()
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "context-only answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "selected answer"},
    ]
    collator = CrossTokenizerCollator(
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=512,
        ctx_length_teachers=[512],
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator(
        [
            {
                "idx": 8,
                "message_log": messages,
                "message_loss_mask": [0, 0, 0, 1],
                "loss_multiplier": 1.0,
            }
        ]
    )

    valid = result["alignment_0_pair_valid"][0]
    student_text = _decode_valid_spans(
        student,
        result["input_ids"][0].tolist(),
        result["alignment_0_student_chunk_id"][0].tolist(),
        valid.tolist(),
    )
    assert student_text == "selected answer<|im_end|>"


@pytest.mark.parametrize("style", ["nano", "qwen"])
def test_structured_tool_call_with_empty_content_is_supervised(style: str) -> None:
    tokenizer = _NativeThinkingTokenizer(
        style,
        ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"),
    )
    messages = [
        {"role": "user", "content": "check"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"query": "status"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "secret tool result"},
        {"role": "assistant", "content": "done"},
    ]
    document = _render_and_tokenize_chat(
        tokenizer,
        messages,
        512,
        message_loss_mask=[0, 1, 0, 0],
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    assert document is not None

    supervised = tokenizer.decode(
        [
            token_id
            for token_id, include in zip(document.input_ids, document.assistant_mask)
            if include
        ]
    )
    assert "<tool_call>" in supervised
    assert "lookup" in supervised
    assert "status" in supervised
    assert "</tool_call>" in supervised
    assert "secret tool result" not in supervised
    assert supervised.endswith("<|im_end|>")

    answer_start, answer_end = document.alignment_regions[0].answer
    rendered = tokenizer.decode(document.input_ids)
    assert rendered[answer_start:answer_end].startswith("<tool_call>")
    assert rendered[answer_start:answer_end].endswith("</tool_call>")


def test_chat_collator_rejects_overlength_instead_of_left_truncating() -> None:
    student, teacher, aligner = _native_pair()
    collator = CrossTokenizerCollator(
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=8,
        ctx_length_teachers=[8],
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )

    with pytest.raises(ValueError, match="overlength rows are rejected"):
        collator(
            [
                {
                    "idx": 99,
                    "message_log": _messages(),
                    "loss_multiplier": 1.0,
                }
            ]
        )


def test_tool_schema_is_not_silently_dropped_by_template_fallback() -> None:
    class NoToolTemplateTokenizer(_NativeThinkingTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            if kwargs.get("tools") is not None:
                raise TypeError("tools are unsupported")
            return super().apply_chat_template(*args, **kwargs)

    tokenizer = NoToolTemplateTokenizer("nano")
    with pytest.raises(TypeError, match="tools are unsupported"):
        _render_chat_text(
            tokenizer,
            [{"role": "user", "content": "check"}],
            preserve_thinking=True,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )


def test_native_alignment_rejects_invalid_region_selection() -> None:
    student, teacher, aligner = _native_pair()
    with pytest.raises(ValueError, match="non-empty subset"):
        CrossTokenizerCollator(
            student_tokenizer=student,
            teacher_tokenizers=[teacher],
            aligners=[aligner],
            ctx_length_student=256,
            ctx_length_teachers=[256],
            config=CrossTokenizerCollatorConfig(
                mode="chat",
                include_thinking_in_loss=True,
                native_thinking_alignment=True,
                kd_alignment_regions=[],
            ),
            drop_first_assistant_chunk_kl_by_teacher=[False],
        )


class _XmlToolTokenizer(_NativeThinkingTokenizer):
    """Model the deployed Nano/Qwen XML separators and scalar serializers."""

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str:
        assert not tokenize and not add_generation_prompt
        rendered = []
        for message in messages:
            role = message["role"]
            content = str(message.get("content") or "").strip()
            if role == "assistant":
                reasoning = message.get("reasoning_content") or ""
                if self.style == "nano":
                    prefix = f"<think>\n{reasoning}</think>"
                else:
                    prefix = f"<think>\n{reasoning}\n</think>\n\n"
                calls = []
                for call in message.get("tool_calls") or []:
                    function = call.get("function", call)
                    parameters = []
                    for key, value in (function.get("arguments") or {}).items():
                        if isinstance(value, (dict, list)) or (
                            self.style == "qwen" and not isinstance(value, str)
                        ):
                            value = json.dumps(
                                value, sort_keys=True, ensure_ascii=False
                            )
                        parameters.append(f"<parameter={key}>\n{value}\n</parameter>\n")
                    calls.append(
                        f"<tool_call>\n<function={function['name']}>\n"
                        + "".join(parameters)
                        + "</function>\n</tool_call>"
                    )
                if calls:
                    separator = "\n" if self.style == "nano" else "\n\n"
                    content += (separator if content else "") + "\n".join(calls)
                content = prefix + content
            rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(rendered)


@pytest.mark.parametrize("prose", ["", "Now apply the fix.", "Inspect this!\nNext:"])
@pytest.mark.parametrize("call_count", [1, 2])
@pytest.mark.parametrize(
    "arguments",
    [
        {"command": 'printf "<tool_call>fake</tool_call>"'},
        {"replaceAll": True, "disabled": False, "missing": None},
        {"nested": {"flag": True, "missing": None}, "items": [False, "日本語"]},
    ],
)
def test_native_tool_parts_preserve_ce_and_align_matching_payloads(
    prose: str, call_count: int, arguments: dict[str, Any]
) -> None:
    special = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    # Cross-piece punctuation/newline tokens must retain CE even when they
    # cannot be used as an exact KD event on both native surfaces.
    student = _XmlToolTokenizer("nano", special + (".\n", "True", "False", "None"))
    teacher = _XmlToolTokenizer("qwen", special + (".\n\n", "true", "false", "null"))
    calls = [
        {"type": "function", "function": {"name": "check", "arguments": arguments}}
        for _ in range(call_count)
    ]
    messages = [
        {"role": "user", "content": "Inspect the repository."},
        {"role": "assistant", "content": "Historical answer."},
        {"role": "tool", "content": "Historical result."},
        {
            "role": "assistant",
            "reasoning_content": "Inspect carefully.",
            "content": prose,
            "tool_calls": calls,
        },
    ]
    original = deepcopy(messages)
    selected = [0, 0, 0, 1]
    aligner = TokenAligner(student, teacher, None)
    collator = CrossTokenizerCollator(
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=4096,
        ctx_length_teachers=[4096],
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator(
        [
            {
                "idx": 0,
                "message_log": messages,
                "message_loss_mask": selected,
                "loss_multiplier": 1.0,
            }
        ]
    )
    assert messages == original
    valid = result["alignment_0_pair_valid"][0]
    assert valid.any()
    assert result["alignment_0_pair_is_correct"][0, valid].all()
    for tokenizer, prefix, span_key in (
        (student, "", "alignment_0_student_chunk_id"),
        (teacher, "teacher_0_", "alignment_0_teacher_chunk_id"),
    ):
        expected = _render_and_tokenize_chat(
            tokenizer,
            messages,
            4096,
            message_loss_mask=selected,
            include_thinking_in_loss=True,
            native_thinking_alignment=True,
            skip_overlength=True,
        )
        assert expected is not None
        length = len(expected.input_ids)
        assert result[prefix + "input_ids"][0, :length].tolist() == expected.input_ids
        assert (
            result[prefix + "token_mask"][0, :length].tolist()
            == expected.assistant_mask
        )
        kd_text = _decode_valid_spans(
            tokenizer,
            result[prefix + "input_ids"][0].tolist(),
            result[span_key][0].tolist(),
            valid.tolist(),
        )
        assert kd_text.count("<function=check>") == call_count
        assert "Historical" not in kd_text
        if "replaceAll" in arguments:
            for value in ("True", "False", "None", "true", "false", "null"):
                assert value not in kd_text
            ce_text = tokenizer.decode(
                [
                    token
                    for token, keep in zip(expected.input_ids, expected.assistant_mask)
                    if keep
                ]
            )
            assert ("True" if prefix == "" else "true") in ce_text


def test_native_parts_reject_unexpected_tool_serializer_difference() -> None:
    student = _XmlToolTokenizer("nano")
    # This older fake uses a JSON call body, which has no declared XML
    # correspondence. The collator must not silently align it by raw offsets.
    teacher = _NativeThinkingTokenizer("qwen")
    aligner = TokenAligner(student, teacher, None)
    collator = CrossTokenizerCollator(
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=4096,
        ctx_length_teachers=[4096],
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    messages = [
        {"role": "user", "content": "Call check."},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {"function": {"name": "check", "arguments": {"value": "payload"}}}
            ],
        },
    ]
    with pytest.raises(ValueError, match="different student/teacher text"):
        collator([{"idx": 0, "message_log": messages, "loss_multiplier": 1.0}])


@pytest.mark.parametrize("select_history", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_history_retention_and_supervision_are_independent(select_history, inline):
    student, teacher, aligner = _native_pair()
    messages = [
        {"role": "user", "content": "first question"},
        {
            "role": "assistant",
            "content": "earlier answer",
            "reasoning_content": "earlier reasoning",
        },
        {"role": "user", "content": "later question"},
        {
            "role": "assistant",
            "content": "latest answer",
            "reasoning_content": "latest reasoning",
        },
    ]
    if inline:
        for turn in (1, 3):
            messages[turn]["content"] = (
                f"<think>{messages[turn].pop('reasoning_content')}</think>"
                + messages[turn]["content"]
            )
    selected = [0, int(select_history), 0, 1]
    original = deepcopy(messages)
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=512,
        ctx_length_teachers=[512],
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator(
        [
            {
                "idx": 7,
                "message_log": messages,
                "message_loss_mask": selected,
                "loss_multiplier": 1.0,
            }
        ]
    )
    assert messages == original
    rendered = student.decode(result["input_ids"][0].tolist())
    supervised = student.decode(
        result["input_ids"][0, result["token_mask"][0].bool()].tolist()
    )
    aligned = _decode_valid_spans(
        student,
        result["input_ids"][0].tolist(),
        result["alignment_0_student_chunk_id"][0].tolist(),
        result["alignment_0_pair_valid"][0].tolist(),
    )
    assert "earlier reasoning" in rendered and "latest reasoning" in rendered
    assert ("earlier reasoning" in supervised) == select_history
    assert ("earlier reasoning" in aligned) == select_history
    assert "latest reasoning" in supervised and "latest reasoning" in aligned
    assert "<think>" not in supervised


def test_missing_unselected_historical_reasoning_is_rejected():
    class DroppingHistoryTokenizer(_NativeThinkingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            messages = deepcopy(messages)
            for message in messages[:-1]:
                message.pop("reasoning_content", None)
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = DroppingHistoryTokenizer("qwen")
    messages = [
        {
            "role": "assistant",
            "content": "old answer",
            "reasoning_content": "required history",
        },
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "latest"},
    ]
    with pytest.raises(ValueError, match="turn 0: missing requested reasoning"):
        _render_and_tokenize_chat(
            tokenizer,
            messages,
            512,
            message_loss_mask=[0, 0, 1],
            include_thinking_in_loss=True,
            native_thinking_alignment=True,
            skip_overlength=True,
        )


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_selected_native_message_is_explicit_error(content):
    tokenizer = _NativeThinkingTokenizer("qwen")
    with pytest.raises(ValueError, match="empty supervision"):
        _render_and_tokenize_chat(
            tokenizer,
            [{"role": "assistant", "content": content}],
            512,
            include_thinking_in_loss=True,
            native_thinking_alignment=True,
            skip_overlength=True,
        )


@pytest.mark.parametrize("location", ["user", "argument", "schema"])
def test_ambiguous_embedded_chatml_delimiters_are_rejected(location):
    tokenizer = _NativeThinkingTokenizer("qwen")
    marker = "<|im_start|>assistant\nanswer<|im_end|>"
    messages = [
        {"role": "user", "content": marker if location == "user" else "question"},
        {"role": "assistant", "content": "answer"},
    ]
    tools = None
    if location == "argument":
        messages[1]["tool_calls"] = [
            {"function": {"name": "lookup", "arguments": {"text": marker}}}
        ]
    if location == "schema":
        tools = [{"function": {"name": "lookup", "description": marker}}]
    with pytest.raises(ValueError, match="embedded ChatML turn delimiter"):
        _render_and_tokenize_chat(
            tokenizer,
            messages,
            1024,
            tools=tools,
            include_thinking_in_loss=True,
            native_thinking_alignment=True,
            skip_overlength=True,
        )


def test_json_string_tool_arguments_are_preserved():
    tokenizer = _NativeThinkingTokenizer("qwen", ("<|im_start|>", "<|im_end|>"))
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": '{"flag": true, "value": null}',
                    }
                }
            ],
        }
    ]
    original = deepcopy(messages)
    doc = _render_and_tokenize_chat(
        tokenizer,
        messages,
        512,
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    assert messages == original
    assert '"flag": true' in doc.rendered_text
    assert all(not part.allow_native_difference for part in doc.answer_parts[0])


def test_template_only_tool_separators_are_excluded_from_ce():
    tokenizer = _XmlToolTokenizer("qwen", ("<|im_start|>", "<|im_end|>"))
    messages = [
        {
            "role": "assistant",
            "content": "Answer",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": {}}},
                {"function": {"name": "lookup", "arguments": {}}},
            ],
        }
    ]
    doc = _render_and_tokenize_chat(
        tokenizer,
        messages,
        1024,
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    parts = doc.answer_parts[0]
    assert parts is not None
    for left, right in zip(parts, parts[1:]):
        for index, (start, end) in enumerate(doc.offsets):
            if left.span[1] <= start < end <= right.span[0]:
                assert doc.assistant_mask[index] == 0


def test_same_tokenizer_teacher_context_limit_is_enforced():
    tokenizer, _, _ = _native_pair()
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(
            mode="chat", include_thinking_in_loss=True, native_thinking_alignment=True
        ),
        student_tokenizer=tokenizer,
        teacher_tokenizers=[None],
        aligners=[None],
        ctx_length_student=512,
        ctx_length_teachers=[8],
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    with pytest.raises(ValueError, match="teacher 0, sample idx=9:.*overlength"):
        collator([{"idx": 9, "message_log": _messages(), "loss_multiplier": 1.0}])


class _OrdinaryChatTokenizer(_NativeThinkingTokenizer):
    def __init__(self, *, trim, transform=False, merged_pieces=()):
        super().__init__("qwen", merged_pieces)
        self.trim = trim
        self.transform = transform

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        assert not tokenize and not add_generation_prompt
        return "".join(
            f"<|im_start|>{message['role']}\n"
            + (
                message["content"].upper()
                if self.transform
                else message["content"].strip()
                if self.trim
                else message["content"]
            )
            + "<|im_end|>\n"
            for message in messages
        )


def test_ordinary_canonical_spans_bind_repeated_logical_turns():
    student = _OrdinaryChatTokenizer(trim=False)
    teacher = _OrdinaryChatTokenizer(trim=True)
    messages = [
        {"role": "user", "content": "  Same  "},
        {"role": "assistant", "content": "  Same  "},
        {"role": "user", "content": "  Same  "},
        {"role": "assistant", "content": "  Same  "},
    ]
    docs = [
        _render_and_tokenize_chat(
            tokenizer,
            messages,
            512,
            include_thinking_in_loss=False,
            native_thinking_alignment=False,
            skip_overlength=True,
        )
        for tokenizer in (student, teacher)
    ]
    for document in docs:
        assert [
            document.rendered_text[start:end] for start, end in document.assistant_spans
        ] == ["Same", "Same"]
        assert document.source_turn_indices == [1, 3]
        assert document.assistant_spans[0][0] > document.rendered_text.index("Same")
    aligner = TokenAligner(student, teacher, None)
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(mode="chat"),
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=512,
        ctx_length_teachers=[512],
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator([{"idx": 0, "message_log": messages, "loss_multiplier": 1.0}])
    aligned = _decode_valid_spans(
        student,
        result["input_ids"][0].tolist(),
        result["alignment_0_student_chunk_id"][0].tolist(),
        result["alignment_0_pair_valid"][0].tolist(),
    )
    assert aligned.count("Same") == 2
    assert result["alignment_0_pair_is_correct"][result["alignment_0_pair_valid"]].all()


def test_ordinary_transformed_turn_cannot_match_later_repeated_content():
    tokenizer = _OrdinaryChatTokenizer(trim=True, transform=True)
    with pytest.raises(ValueError, match="turn 0: missing canonical content span"):
        _render_and_tokenize_chat(
            tokenizer,
            [
                {"role": "user", "content": "same"},
                {"role": "assistant", "content": "SAME"},
            ],
            256,
            include_thinking_in_loss=False,
            native_thinking_alignment=False,
            skip_overlength=True,
        )


def test_ordinary_boundary_whitespace_token_keeps_ce_but_is_not_exact_kd():
    student = _OrdinaryChatTokenizer(trim=False, merged_pieces=(" Hello", "<|im_end|>"))
    teacher = _OrdinaryChatTokenizer(trim=True, merged_pieces=("Hello", "<|im_end|>"))
    aligner = TokenAligner(student, teacher, None)
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(mode="chat"),
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=256,
        ctx_length_teachers=[256],
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    result = collator(
        [
            {
                "idx": 0,
                "message_log": [{"role": "assistant", "content": " Hello"}],
                "loss_multiplier": 1.0,
            }
        ]
    )
    ce_text = student.decode(
        result["input_ids"][0, result["token_mask"][0].bool()].tolist()
    )
    assert ce_text == " Hello<|im_end|>"
    valid = result["alignment_0_pair_valid"][0]
    assert result["alignment_0_pair_is_correct"][0, valid].tolist() == [False, True]


def test_ordinary_content_matching_role_header_binds_body():
    tokenizer = _OrdinaryChatTokenizer(trim=True, merged_pieces=("<|im_end|>",))
    doc = _render_and_tokenize_chat(
        tokenizer,
        [{"role": "assistant", "content": "assistant"}],
        256,
        include_thinking_in_loss=False,
        native_thinking_alignment=False,
        skip_overlength=True,
    )
    start, end = doc.assistant_spans[0]
    assert start == len("<|im_start|>assistant\n")
    assert doc.rendered_text[start:end] == "assistant"
    assert (
        tokenizer.decode(
            [token for token, keep in zip(doc.input_ids, doc.assistant_mask) if keep]
        )
        == "assistant<|im_end|>"
    )


@pytest.mark.parametrize("select_history", [False, True])
def test_ordinary_separate_reasoning_loss_requires_native_alignment(select_history):
    tokenizer = _NativeThinkingTokenizer("qwen")
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(mode="chat", include_thinking_in_loss=True),
        student_tokenizer=tokenizer,
        teacher_tokenizers=[None],
        aligners=[None],
        ctx_length_student=512,
        ctx_length_teachers=[512],
        drop_first_assistant_chunk_kl_by_teacher=[False],
    )
    with pytest.raises(
        ValueError,
        match="student, sample idx=17: turn 1:.*native_thinking_alignment=true",
    ):
        collator(
            [
                {
                    "idx": 17,
                    "loss_multiplier": 1.0,
                    "message_loss_mask": [0, int(select_history), 0, 1],
                    "message_log": [
                        {"role": "user", "content": "first"},
                        {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "required historical reasoning",
                        },
                        {"role": "user", "content": "second"},
                        {"role": "assistant", "content": "final"},
                    ],
                }
            ]
        )


def test_ordinary_inline_reasoning_can_receive_loss():
    tokenizer = _OrdinaryChatTokenizer(
        trim=True, merged_pieces=("<think>", "</think>", "<|im_end|>")
    )
    doc = _render_and_tokenize_chat(
        tokenizer,
        [{"role": "assistant", "content": "<think>reason</think>answer"}],
        256,
        include_thinking_in_loss=True,
        native_thinking_alignment=False,
        skip_overlength=True,
    )
    supervised = tokenizer.decode(
        [token for token, keep in zip(doc.input_ids, doc.assistant_mask) if keep]
    )
    assert supervised == "reason</think>answer<|im_end|>"
