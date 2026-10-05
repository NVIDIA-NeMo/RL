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

import pytest

from nemo_rl.algorithms.x_token.mopd import (
    classify_thinking_state_before_generated_span,
    compute_offsets_manual,
    generated_think_tool_marker_error,
    generated_thinking_state_after_response,
    proven_template_only_teacher_token_indices,
    qwen3_assistant_content_transform,
    qwen3_leading_think_prefix_len,
    render_teacher_open_thinking_prefix,
    split_sampled_assistant_eot,
    trim_leading_text_tokens_for_alignment,
    trim_outer_whitespace_with_offsets_for_alignment,
)


class _QwenEOTTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 99


class _GroupedDecodeTokenizer:
    all_special_ids = []

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "x" if [int(token_id) for token_id in ids] == [10, 11] else ""


class _NoNativeOffsetsTokenizer:
    all_special_ids = []

    def __call__(
        self,
        text,
        *,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del return_tensors, add_special_tokens
        if return_offsets_mapping:
            raise NotImplementedError("offsets are unavailable")
        return {"input_ids": [10, 11] if text == " x " else []}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        mapping = {(10,): " ", (11,): "x ", (10, 11): " x "}
        return mapping.get(tuple(int(token_id) for token_id in ids), "")


def test_sampled_assistant_eot_is_removed_only_when_present():
    tokenizer = _QwenEOTTokenizer()

    assert split_sampled_assistant_eot(tokenizer, [1, 2, 99]) == ([1, 2], 2)
    assert split_sampled_assistant_eot(tokenizer, [1, 2]) == ([1, 2], None)


def test_think_and_tool_structure_accepts_proven_open_tool_transcript():
    transcript = (
        "reasoning</think>answer<tool_call>"
        '{"name":"weather","arguments":{"city":"SF"}}'
        "</tool_call>"
    )

    assert generated_think_tool_marker_error(transcript, initial_state="open") is None
    assert (
        generated_think_tool_marker_error(transcript, initial_state="unknown")
        == "unknown_thinking_state"
    )


def test_template_provenance_distinguishes_template_only_and_mixed_tokens():
    transformed = "r\n</think>\n\nx"
    # Token 2 contains sampled </think> plus an inserted newline. Tokens 1 and
    # 3 contain only template-inserted whitespace and can be excluded safely.
    offsets = [(0, 1), (1, 2), (2, 11), (11, 12), (12, 13)]

    assert proven_template_only_teacher_token_indices(
        "r</think>x", transformed, offsets
    ) == ({1, 3}, {2})


def test_manual_offsets_support_tokens_that_decode_only_as_a_group():
    assert compute_offsets_manual(_GroupedDecodeTokenizer(), [10, 11], "x") == [
        (0, 1),
        (0, 1),
    ]


def test_trim_outer_whitespace_reconstructs_offsets_when_backend_has_none():
    assert trim_outer_whitespace_with_offsets_for_alignment(
        _NoNativeOffsetsTokenizer(), [10, 11], " x "
    ) == ([11], 1, 0, "x", [(0, 1)])


class _ThinkingBoundaryTokenizer:
    """Split thinking markers across IDs to exercise text-based validation."""

    all_special_ids = []

    def __init__(self, generation_suffix="<|im_start|>assistant\n"):
        self.generation_suffix = generation_suffix

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        preserve_thinking,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is True
        assert preserve_thinking is True
        history = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        return history + self.generation_suffix


@pytest.mark.parametrize(
    ("suffix", "expects_thinking", "expected"),
    [
        ("<|im_start|>assistant\n", True, "waiting"),
        ("<|im_start|>assistant\n", False, "unknown"),
        ("<|im_start|>assistant\n<think>\n", True, "open"),
        ("<|im_start|>assistant\n<think>\n", False, "open"),
        ("<|im_start|>assistant\n<think></think>", True, "closed"),
        ("<|im_start|>assistant\n<think></think>", False, "closed"),
        ("<|im_start|>assistant\n<think>\n\n</think>\n\n", True, "closed"),
        ("<|im_start|>assistant\n<think>\n\n</think>\n\n", False, "closed"),
        ("<|im_start|>assistant\nother prefix", True, "unknown"),
        ("<|im_start|>assistant\n ", True, "unknown"),
        ("<|im_start|>user\n", True, "unknown"),
        ("unrecognized template", True, "unknown"),
    ],
)
def test_thinking_boundary_requires_supported_exact_prefix(
    suffix, expects_thinking, expected
):
    tokenizer = _ThinkingBoundaryTokenizer()
    prefix = tokenizer("history\n" + suffix)["input_ids"]
    response = tokenizer("<think>reasoning</think>answer")["input_ids"]

    assert (
        classify_thinking_state_before_generated_span(
            tokenizer,
            prefix + response,
            generated_start=len(prefix),
            expects_thinking=expects_thinking,
        )
        == expected
    )


@pytest.mark.parametrize("generated_start", [-1, 100])
def test_thinking_boundary_rejects_invalid_span(generated_start):
    assert (
        classify_thinking_state_before_generated_span(
            _ThinkingBoundaryTokenizer(),
            [],
            generated_start=generated_start,
            expects_thinking=True,
        )
        == "unknown"
    )


@pytest.mark.parametrize(
    ("initial_state", "response", "final_state", "error"),
    [
        ("waiting", "<think>reasoning</think>answer", "closed", None),
        ("waiting", "<think></think>answer", "closed", None),
        ("waiting", "<think>unfinished", "open", None),
        ("waiting", "<think>", "open", None),
        ("waiting", "answer", "waiting", "missing_think_open"),
        ("waiting", "", "waiting", "missing_think_open"),
        ("waiting", "text<think>reasoning", "waiting", "misplaced_think_open"),
        ("waiting", "\n<think>reasoning", "waiting", "misplaced_think_open"),
        ("waiting", "</think><think>reasoning", "waiting", "unexpected_think_close"),
        ("waiting", "<think><think>nested</think>", "open", "generated_think_open"),
        (
            "waiting",
            "<think>r</think><think>reopened",
            "closed",
            "generated_think_open",
        ),
        ("waiting", "<think>r</think></think>", "closed", "multiple_think_close"),
        ("open", "reasoning</think>answer", "closed", None),
        ("open", "</think>answer", "closed", None),
        ("open", "unfinished", "open", None),
        ("open", "<think>nested</think>", "open", "generated_think_open"),
        ("open", "r</think><think>reopened", "closed", "generated_think_open"),
        ("open", "r</think></think>", "closed", "multiple_think_close"),
        ("closed", "answer", "closed", None),
        ("closed", "</think>answer", "closed", "unexpected_think_close"),
        ("closed", "<think>reasoning</think>", "closed", "generated_think_open"),
        ("unknown", "<think>r</think>answer", "unknown", "unknown_thinking_state"),
    ],
)
def test_generated_thinking_tracks_marker_order_and_final_state(
    initial_state, response, final_state, error
):
    tokenizer = _ThinkingBoundaryTokenizer()
    sampled_ids = tokenizer(response)["input_ids"]
    sampled_text = tokenizer.decode(sampled_ids, skip_special_tokens=False)

    assert generated_thinking_state_after_response(
        sampled_text, initial_state=initial_state
    ) == (final_state, error)
    assert (
        generated_think_tool_marker_error(sampled_text, initial_state=initial_state)
        == error
    )


@pytest.mark.parametrize(
    ("suffix", "error"),
    [
        ('<tool_call>{"name":"weather"}</tool_call>', None),
        ("<tool_call><tool_call></tool_call></tool_call>", "nested_tool_call"),
        ("</tool_call>", "unmatched_tool_call_close"),
        ("<tool_call>", "unclosed_tool_call"),
        ("<tool_response>result</tool_response>", "generated_tool_response"),
    ],
)
def test_native_generated_thinking_preserves_tool_checks(suffix, error):
    assert generated_thinking_state_after_response(
        "<think>reasoning</think>" + suffix, initial_state="waiting"
    ) == ("closed", error)


@pytest.mark.parametrize(
    "suffix",
    ["<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n"],
)
def test_teacher_open_prefix_preserves_native_and_nano_templates(suffix):
    tokenizer = _ThinkingBoundaryTokenizer(generation_suffix=suffix)

    rendered = render_teacher_open_thinking_prefix(
        tokenizer, [{"role": "user", "content": "question"}]
    )

    assert rendered == (
        "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    assert rendered.count("<think>") == 1
    assert "</think>" not in rendered
    assert tokenizer.generation_suffix == suffix


@pytest.mark.parametrize(
    "suffix", ["unrecognized template", "<|im_start|>assistant\n<think></think>"]
)
def test_teacher_open_prefix_rejects_unproven_or_closed_template(suffix):
    tokenizer = _ThinkingBoundaryTokenizer(generation_suffix=suffix)
    with pytest.raises(ValueError, match="provable open-thinking"):
        render_teacher_open_thinking_prefix(
            tokenizer, [{"role": "user", "content": "question"}]
        )


class _NoncanonicalThinkTokenizer:
    all_special_ids = []
    pieces = {1: "<think>", 2: "<thi", 3: "nk>", 4: "<think>r"}

    def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
        del kwargs
        if return_offsets_mapping:
            raise NotImplementedError("offsets are unavailable")
        assert text.startswith("<think>")
        return {"input_ids": [1, *[ord(char) + 1000 for char in text[7:]]]}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(
            self.pieces[token_id] if token_id < 1000 else chr(token_id - 1000)
            for token_id in ids
        )


@pytest.mark.parametrize("newlines", ["", "\n", "\n\n"])
def test_leading_think_trim_preserves_original_noncanonical_token_positions(newlines):
    tokenizer = _NoncanonicalThinkTokenizer()
    prefix_ids = [2, 3, *[ord(char) + 1000 for char in newlines]]
    reasoning_ids = [ord(char) + 1000 for char in "reasoning"]
    original_ids = prefix_ids + reasoning_ids
    text = tokenizer.decode(original_ids)

    trimmed_ids, dropped, trimmed_text = trim_leading_text_tokens_for_alignment(
        tokenizer, original_ids, text, qwen3_leading_think_prefix_len(text)
    )

    assert trimmed_ids == reasoning_ids
    assert dropped == len(prefix_ids)
    assert trimmed_text == "reasoning"
    assert original_ids[dropped:] == trimmed_ids


def test_leading_think_trim_refuses_token_crossing_generated_prefix():
    tokenizer = _NoncanonicalThinkTokenizer()
    original_ids = [4, *[ord(char) + 1000 for char in "easoning"]]
    text = tokenizer.decode(original_ids)

    assert trim_leading_text_tokens_for_alignment(
        tokenizer, original_ids, text, qwen3_leading_think_prefix_len(text)
    ) == (original_ids, 0, text)


@pytest.mark.parametrize("sampled", ["</think>answer", "<think></think>answer"])
def test_empty_thinking_keeps_combined_template_whitespace_for_provenance(sampled):
    transformed, had_think = qwen3_assistant_content_transform(sampled)

    assert had_think is True
    assert transformed == "\n\n</think>\n\nanswer"
    assert proven_template_only_teacher_token_indices(
        "</think>answer",
        transformed,
        [(0, 2), (2, 10), (10, 12), (12, 18)],
    ) == ({0, 2}, set())
