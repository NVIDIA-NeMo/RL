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

import asyncio
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.trtllm.trtllm_http_server import (
    _build_prompt_token_ids,
    _compute_splice_inputs,
    _make_parse_tool_calls,
    _resolve_tool_parser_name,
)


class _FakeToolParser:
    def __init__(self, *, calls):
        self.calls = calls

    def has_tool_call(self, text: str) -> bool:
        return "<tool_call>" in text

    def detect_and_parse(self, text: str, tools: list):
        del text, tools
        return SimpleNamespace(
            normal_text="visible before visible after",
            calls=self.calls,
        )


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]


def test_explicit_tool_parser_overrides_model_auto_resolution():
    assert _resolve_tool_parser_name("qwen3_coder", "/missing/model") == "qwen3_coder"


def test_tool_parser_is_inferred_from_local_hf_config(tmp_path):
    pytest.importorskip("tensorrt_llm.serve.tool_parser.tool_parser_factory")
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_moe"}')

    assert _resolve_tool_parser_name(None, str(tmp_path)) == "qwen3"


def test_tool_parser_resolution_fails_loudly(monkeypatch):
    factory = pytest.importorskip("tensorrt_llm.serve.tool_parser.tool_parser_factory")
    monkeypatch.setattr(factory, "resolve_auto_tool_parser", lambda _: None)

    with pytest.raises(ValueError, match="set trtllm_cfg.tool_parser explicitly"):
        _resolve_tool_parser_name(None, "/missing/model")


def test_parse_tool_calls_uses_parser_normal_text():
    parser = _FakeToolParser(
        calls=[
            SimpleNamespace(
                name="run_command",
                parameters='{"command": "pwd"}',
            )
        ]
    )
    parse_tool_calls = _make_parse_tool_calls(parser)

    content, calls = parse_tool_calls(
        "visible before<tool_call>xml</tool_call>visible after",
        tools=None,
    )

    assert content == "visible before visible after"
    assert calls[0]["function"] == {
        "name": "run_command",
        "arguments": '{"command": "pwd"}',
    }


def test_parse_tool_calls_preserves_text_when_parser_returns_no_calls():
    parser = _FakeToolParser(calls=[])
    parse_tool_calls = _make_parse_tool_calls(parser)
    text = "visible before<tool_call>malformed</tool_call>visible after"

    content, calls = parse_tool_calls(text, tools=None)

    assert content == text
    assert calls == []


def _tool_call_messages():
    return [
        {"role": "user", "content": "List the current directory."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command": "pwd"}',
                    },
                }
            ],
            "prompt_token_ids": [10],
            "generation_token_ids": [11],
            "generation_log_probs": [-0.1],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "/workspace"},
    ]


def _parse_with_trtllm(messages):
    chat_utils = pytest.importorskip("tensorrt_llm.serve.chat_utils")
    transformers = pytest.importorskip("transformers")
    conversation, mm_coroutine, _ = chat_utils.parse_chat_messages_coroutines(
        messages, transformers.PretrainedConfig()
    )
    assert asyncio.run(mm_coroutine) == (None, None)
    return conversation


def test_trtllm_parser_normalizes_openai_tool_arguments_without_mutation():
    messages = _tool_call_messages()

    conversation = _parse_with_trtllm(messages)

    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "pwd"
    }
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command": "pwd"}'
    )
    assert "prompt_token_ids" not in conversation[1]
    assert "generation_token_ids" not in conversation[1]
    assert "generation_log_probs" not in conversation[1]


def test_prompt_and_prefix_render_use_normalized_tool_arguments():
    messages = _tool_call_messages()
    conversation = _parse_with_trtllm(messages)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {"type": "object"},
            },
        }
    ]
    tokenizer = _RecordingTokenizer()

    assert _build_prompt_token_ids(conversation, tokenizer, tools=tools) == [1, 2, 3]
    required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
        messages, conversation, tokenizer, tools, {}
    )

    assert required_prefix_ids == [10, 11]
    assert template_prefix_ids == [1, 2, 3]
    assert len(tokenizer.calls) == 2
    for rendered_messages, kwargs in tokenizer.calls:
        assert rendered_messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "command": "pwd"
        }
        assert kwargs["tools"] == tools


def test_trtllm_parser_rejects_invalid_tool_arguments():
    messages = _tool_call_messages()
    messages[1]["tool_calls"][0]["function"]["arguments"] = "not-json"

    with pytest.raises(ValueError, match="function.arguments"):
        _parse_with_trtllm(messages)
