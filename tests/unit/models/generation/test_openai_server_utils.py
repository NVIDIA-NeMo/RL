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
"""Tests for the shared on-policy prefix splice used by the vLLM/TRT-LLM OpenAI servers."""

import pytest

from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens
from nemo_rl.models.generation.openai_server_utils import (
    CompleteTokenInjectionError,
    build_complete_prompt_token_ids,
    find_latest_tokenized_assistant_ordinal,
)


class _CompleteTokenTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        token_ids = []
        while text:
            if text.startswith(self.eos_token):
                token_ids.append(self.eos_token_id)
                text = text[len(self.eos_token) :]
            elif text.startswith("next"):
                token_ids.append(40)
                text = text[len("next") :]
            elif text.startswith("GEN"):
                token_ids.append(99)
                text = text[len("GEN") :]
            else:
                raise AssertionError(f"Unexpected text to encode: {text!r}")
        return token_ids


def _render_marked_conversation(conversation, *, tokenize):
    token_ids = []
    rendered = ""
    for message in conversation:
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", tool_call)
            assert isinstance(function.get("arguments"), dict)

        role = message["role"]
        content = message.get("content")
        if role == "user" and content == "hello":
            token_ids.append(10)
            rendered += "hello"
        elif role == "user" and content == "next":
            token_ids.append(40)
            rendered += "next"
        elif role == "assistant":
            token_ids.extend([777, 2])
            rendered += f"{content}<eos>"
        else:
            token_ids.append(900)
            rendered += "other"
    token_ids.append(99)
    rendered += "GEN"
    return token_ids if tokenize else rendered


def test_build_complete_prompt_token_ids_preserves_prefix_and_suffix() -> None:
    tokenizer = _CompleteTokenTokenizer()
    conversation = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "older tool call",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ],
        },
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
    ]

    result = build_complete_prompt_token_ids(
        tokenizer=tokenizer,
        conversation=conversation,
        assistant_ordinal=1,
        model_prefix_token_ids=[10, 777, 2, 40, 31, 32, 2],
        render_prompt_text=lambda messages: _render_marked_conversation(
            messages, tokenize=False
        ),
    )

    assert result == [10, 777, 2, 40, 31, 32, 2, 40, 99]
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command":"pwd"}'
    )


def test_complete_token_injection_finds_latest_tokenized_assistant() -> None:
    messages = [
        {"role": "assistant", "content": "first"},
        {
            "role": "assistant",
            "content": "second",
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
        },
        {"role": "assistant", "content": "third"},
    ]

    assert find_latest_tokenized_assistant_ordinal(messages) == 1


def test_complete_token_injection_requires_assistant_eos() -> None:
    tokenizer = _CompleteTokenTokenizer()
    conversation = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
    ]

    with pytest.raises(CompleteTokenInjectionError, match="did not close"):
        build_complete_prompt_token_ids(
            tokenizer=tokenizer,
            conversation=conversation,
            assistant_ordinal=0,
            model_prefix_token_ids=[31, 32, 2],
            render_prompt_text=lambda messages: (
                "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E missing boundary close"
            ),
        )


def test_replace_prefix_tokens_empty_model_prefix_returns_template():
    """Turn 1 has no prior model output; the full template is returned unchanged."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = []
    template_prefix_token_ids = [9, 2]
    template_token_ids = [9, 2, 33, 44]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == template_token_ids


def test_replace_prefix_tokens_missing_eos_in_template_prefix_raises():
    """A template prefix with no EOS has no valid splice boundary and must raise."""

    class _T:
        eos_token_id = 2

        def decode(self, *args, **kwargs):
            pass

    tokenizer = _T()
    model_prefix_token_ids = [7, 2]
    template_prefix_token_ids = [9, 9, 9]
    template_token_ids = [9, 9, 9, 2, 10]
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )


def test_replace_prefix_tokens_tokenizer_without_eos_raises():
    """A tokenizer that has no EOS token cannot locate the splice boundary."""

    class _T:
        eos_token_id = None

    tokenizer = _T()
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=[1],
            template_prefix_token_ids=[1, 2],
            template_token_ids=[1, 2],
        )


def test_replace_prefix_tokens_uses_last_eos_in_template_prefix():
    """When the prefix contains multiple EOS tokens, the splice cuts at the last one."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = [100, 2]
    template_prefix_token_ids = [9, 2, 9, 2]
    template_token_ids = [9, 2, 9, 2, 77, 88]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == [100, 2, 77, 88]


def test_replace_prefix_tokens_qwen3_think_shift_picks_assistant_eos_not_user_eos():
    """Non-strict-prefix: Qwen3 strips <think> from history when the last message is a
    user turn, so the template's prefix region is shorter and a later user-turn EOS
    lands within the first len(template_prefix) positions. The count-based algorithm
    must cut at the assistant EOS and preserve the intervening user turn.
    """

    class _T:
        eos_token_id = 2

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids)

    tokenizer = _T()
    model_prefix_token_ids = [11, 12, 99, 99, 99, 55, 2]
    template_prefix_token_ids = [11, 12, 88, 88, 88, 56, 2]
    template_token_ids = [11, 12, 56, 2, 70, 71, 2, 40, 41]

    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )

    assert result == [11, 12, 99, 99, 99, 55, 2, 70, 71, 2, 40, 41]
    assert 70 in result and 71 in result
