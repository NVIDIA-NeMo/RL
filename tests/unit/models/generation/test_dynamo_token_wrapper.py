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
import json

import pytest
from transformers import AutoTokenizer

from nemo_rl.models.generation.dynamo.token_wrapper import (
    DynamoTokenWrapperServer,
    _inject_gym_token_metadata,
    _validate_engine_data,
    prepare_dynamo_chat_completion_request,
)


class _Tokenizer:
    eos_token_id = 2
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.calls = []

    def decode(self, token_ids):
        return repr(token_ids)

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

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        documents=None,
        chat_template=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        return_tensors=None,
        return_dict=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "tools": tools,
                "documents": documents,
                "chat_template": chat_template,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
                "tokenize": tokenize,
                "return_tensors": return_tensors,
                "return_dict": return_dict,
                "kwargs": kwargs,
            }
        )
        token_ids = []
        rendered = ""
        for index, message in enumerate(conversation):
            role = message["role"]
            content = message.get("content")
            for tool_call in message.get("tool_calls", []):
                function = tool_call.get("function", tool_call)
                arguments = function.get("arguments")
                if arguments is not None and not isinstance(arguments, dict):
                    raise TypeError("Can only get item pairs from a mapping.")
            if role == "user" and content == "hello":
                token_ids.extend([10])
                rendered += "hello"
            elif role == "assistant" and content == "first":
                # Model the Nemotron template's context-dependent history
                # truncation: the assistant is longer when rendered alone than
                # when a later user turn is present.
                has_later_user = any(
                    item["role"] == "user" for item in conversation[index + 1 :]
                )
                token_ids.extend(
                    [3, self.eos_token_id]
                    if has_later_user
                    else [300, 301, 302, self.eos_token_id]
                )
                rendered += f"first{self.eos_token}"
            elif role == "user" and content == "next":
                token_ids.extend([40])
                rendered += "next"
            elif role == "assistant" and isinstance(content, str):
                token_ids.extend([777, self.eos_token_id])
                rendered += f"{content}{self.eos_token}"
            else:
                token_ids.extend([900])
                rendered += "other"
        if add_generation_prompt:
            token_ids.extend([99])
            rendered += "GEN"
        return token_ids if tokenize else rendered


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def _tool_conversation() -> list[dict]:
    return [
        {"role": "user", "content": "What is the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"San Francisco"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "Sunny, 20C"},
        {"role": "user", "content": "What about tomorrow?"},
    ]


def test_prepare_dynamo_chat_completion_request_first_turn() -> None:
    tokenizer = _Tokenizer()
    body = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "nvext": {"extra_fields": ["timing"], "trace": "keep-me"},
        "chat_template_kwargs": {
            "enable_thinking": False,
            "force_nonempty_content": True,
        },
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=tokenizer,
        tokenizer_chat_template_kwargs={"enable_thinking": True},
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["messages"] == [{"role": "user", "content": "hello"}]
    assert "logprobs" not in prepared
    assert "return_tokens_as_token_ids" not in prepared
    assert prepared["chat_template_kwargs"] == {
        "enable_thinking": False,
        "force_nonempty_content": True,
    }
    assert prepared["nvext"] == {
        "extra_fields": ["timing", "engine_data"],
        "trace": "keep-me",
        "token_data": [10, 99],
    }
    assert tokenizer.calls[0]["kwargs"] == {
        "enable_thinking": False,
        "force_nonempty_content": True,
    }


def test_public_qwen3_tokenizer_first_turn_matches_chat_template(
    tiny_qwen3_model_path,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen3_model_path)
    messages = [{"role": "user", "content": "Solve 2 + 2."}]
    template_kwargs = {"enable_thinking": False}
    expected = tokenizer.apply_chat_template(
        messages,
        tools=None,
        documents=None,
        chat_template=None,
        add_generation_prompt=True,
        continue_final_message=False,
        tokenize=True,
        return_tensors=None,
        return_dict=False,
        **template_kwargs,
    )

    prepared = prepare_dynamo_chat_completion_request(
        {"model": "Qwen/Qwen3-0.6B", "messages": messages},
        tokenizer=tokenizer,
        tokenizer_chat_template_kwargs=template_kwargs,
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["nvext"]["token_data"] == expected


def test_public_qwen3_tool_conversation_matches_plain_chat_template(
    tiny_qwen3_model_path,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen3_model_path)
    messages = _tool_conversation()
    template_kwargs = {"enable_thinking": False}
    expected = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors=None,
        return_dict=False,
        **template_kwargs,
    )

    prepared = prepare_dynamo_chat_completion_request(
        {"model": "Qwen/Qwen3-0.6B", "messages": messages, "tools": TOOLS},
        tokenizer=tokenizer,
        tokenizer_chat_template_kwargs=template_kwargs,
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["nvext"]["token_data"] == expected


def test_public_qwen3_multiturn_prefix_splice_parity(
    tiny_qwen3_model_path,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen3_model_path)
    messages = _tool_conversation()
    template_kwargs = {"enable_thinking": False}
    expected = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors=None,
        return_dict=False,
        **template_kwargs,
    )
    assistant_prefix = tokenizer.apply_chat_template(
        messages[:2],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=False,
        return_tensors=None,
        return_dict=False,
        **template_kwargs,
    )
    first_prompt = tokenizer.apply_chat_template(
        messages[:1],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors=None,
        return_dict=False,
        **template_kwargs,
    )
    generation_token_ids = assistant_prefix[len(first_prompt) : -1]
    messages[1]["prompt_token_ids"] = first_prompt
    messages[1]["generation_token_ids"] = generation_token_ids
    messages[1]["generation_log_probs"] = [-0.1] * len(generation_token_ids)

    assistant_eos_index = [
        index
        for index, token_id in enumerate(expected)
        if token_id == tokenizer.eos_token_id
    ][2]
    expected_splice = assistant_prefix[:-2] + expected[assistant_eos_index:]

    prepared = prepare_dynamo_chat_completion_request(
        {"model": "Qwen/Qwen3-0.6B", "messages": messages, "tools": TOOLS},
        tokenizer=tokenizer,
        tokenizer_chat_template_kwargs=template_kwargs,
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["nvext"]["token_data"] == expected_splice


def test_prepare_dynamo_chat_completion_request_preserves_logprob_fields() -> None:
    tokenizer = _Tokenizer()
    body = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "logprobs": True,
        "return_tokens_as_token_ids": True,
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=tokenizer,
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["logprobs"] is True
    assert prepared["return_tokens_as_token_ids"] is True
    assert prepared["nvext"] == {
        "extra_fields": ["engine_data"],
        "token_data": [10, 99],
    }


def test_prepare_dynamo_chat_completion_request_preserves_prior_prefix() -> None:
    tokenizer = _Tokenizer()
    body = {
        "model": "dummy-model",
        "required_prefix_token_ids": [999],
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "first",
                "prompt_token_ids": [10],
                "generation_token_ids": [31, 32, 2],
                "generation_log_probs": [-0.1, -0.2, -0.3],
            },
            {"role": "user", "content": "next"},
        ],
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=tokenizer,
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["nvext"]["token_data"] == [10, 31, 32, 2, 40, 99]
    assert "required_prefix_token_ids" not in prepared
    assert "prompt_token_ids" not in prepared["messages"][1]
    assert "generation_token_ids" not in prepared["messages"][1]
    assert "generation_log_probs" not in prepared["messages"][1]
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[1]["add_generation_prompt"] is False
    assert tokenizer.calls[0]["tokenize"] is True
    assert tokenizer.calls[1]["tokenize"] is True


def test_prepare_dynamo_chat_completion_request_validates_extra_fields() -> None:
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "nvext": {"extra_fields": ["engine_data", "timing", "engine_data"]},
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=_Tokenizer(),
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["nvext"]["extra_fields"] == ["engine_data", "timing"]

    body["nvext"]["extra_fields"] = "timing"
    with pytest.raises(ValueError, match="extra_fields must be a JSON list"):
        prepare_dynamo_chat_completion_request(
            body,
            tokenizer=_Tokenizer(),
            exclude_tools_when_tool_choice_none=True,
        )


def test_tool_choice_none_honors_worker_exclusion_setting() -> None:
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": TOOLS,
        "tool_choice": "none",
    }
    excluded_tokenizer = _Tokenizer()
    included_tokenizer = _Tokenizer()

    prepare_dynamo_chat_completion_request(
        body,
        tokenizer=excluded_tokenizer,
        exclude_tools_when_tool_choice_none=True,
    )
    prepare_dynamo_chat_completion_request(
        body,
        tokenizer=included_tokenizer,
        exclude_tools_when_tool_choice_none=False,
    )

    assert excluded_tokenizer.calls[0]["tools"] is None
    assert included_tokenizer.calls[0]["tools"] == TOOLS


def test_prepare_dynamo_chat_completion_request_normalizes_prior_tool_arguments() -> (
    None
):
    tokenizer = _Tokenizer()
    body = {
        "model": "dummy-model",
        "messages": [
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
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "malformed",
                            "arguments": "not-json",
                        },
                    },
                ],
            },
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": "first",
                "prompt_token_ids": [10, 777, 2, 40],
                "generation_token_ids": [31, 32, 2],
                "generation_log_probs": [-0.1, -0.2, -0.3],
            },
            {"role": "user", "content": "next"},
        ],
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=tokenizer,
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["nvext"]["token_data"] == [10, 777, 2, 40, 31, 32, 2, 40, 99]
    assert prepared["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command":"pwd"}'
    )
    assert prepared["messages"][1]["tool_calls"][1]["function"]["arguments"] == (
        "not-json"
    )


def test_prepare_dynamo_chat_completion_request_normalizes_tools_without_prefix() -> (
    None
):
    tokenizer = _Tokenizer()
    body = {
        "messages": [
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
        ]
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=tokenizer,
        exclude_tools_when_tool_choice_none=True,
    )

    assert prepared["nvext"]["token_data"] == [10, 777, 2, 40, 99]
    assert prepared["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command":"pwd"}'
    )
    assert len(tokenizer.calls) == 2


def test_prepare_dynamo_chat_completion_request_rejects_stream() -> None:
    with pytest.raises(ValueError, match="stream=True"):
        prepare_dynamo_chat_completion_request(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            tokenizer=_Tokenizer(),
            exclude_tools_when_tool_choice_none=True,
        )


def test_prepare_dynamo_chat_completion_request_rejects_multiple_choices() -> None:
    with pytest.raises(ValueError, match="only n=1"):
        prepare_dynamo_chat_completion_request(
            {"messages": [{"role": "user", "content": "hello"}], "n": 2},
            tokenizer=_Tokenizer(),
            exclude_tools_when_tool_choice_none=True,
        )


def test_validate_engine_data_requires_prompt_completion_and_logprobs() -> None:
    _validate_engine_data(
        {
            "nvext": {
                "engine_data": {
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3],
                    "completion_logprobs": [-0.25],
                }
            }
        }
    )

    with pytest.raises(ValueError, match="engine_data"):
        _validate_engine_data({"nvext": {}})
    with pytest.raises(ValueError, match="prompt_token_ids"):
        _validate_engine_data(
            {
                "nvext": {
                    "engine_data": {
                        "completion_token_ids": [],
                        "completion_logprobs": [],
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="completion_token_ids"):
        _validate_engine_data(
            {
                "nvext": {
                    "engine_data": {
                        "prompt_token_ids": [],
                        "completion_logprobs": [],
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="completion_logprobs"):
        _validate_engine_data(
            {
                "nvext": {
                    "engine_data": {
                        "prompt_token_ids": [],
                        "completion_token_ids": [],
                    }
                }
            }
        )


def test_inject_gym_token_metadata_validates_and_populates_message() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "logprobs": None,
            }
        ],
        "nvext": {
            "engine_data": {
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4, 5],
                "completion_logprobs": [-0.25, -0.5],
            }
        },
    }

    _inject_gym_token_metadata(response)

    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "answer",
        "prompt_token_ids": [1, 2, 3],
        "generation_token_ids": [4, 5],
        "generation_log_probs": [-0.25, -0.5],
    }

    response["nvext"]["engine_data"]["completion_logprobs"] = [-0.25]
    with pytest.raises(ValueError, match="1 generation log probabilities for 2"):
        _inject_gym_token_metadata(response)


def test_forward_chat_completion_reuses_loop_bound_session() -> None:
    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def text(self):
            return json.dumps({"choices": []})

    class FakeSession:
        def __init__(self):
            self.calls = []

        def post(self, url, *, json, headers):
            self.calls.append((url, json, headers))
            return FakeResponse()

    server = DynamoTokenWrapperServer(
        dynamo_frontend_base_url="http://dynamo/v1",
        tokenizer=_Tokenizer(),
        tokenizer_chat_template_kwargs=None,
        exclude_tools_when_tool_choice_none=True,
        request_timeout_s=30,
    )
    session = FakeSession()
    server._client_session = session

    async def forward_twice():
        await server._forward_chat_completion({}, authorization=None)
        await server._forward_chat_completion({}, authorization="Bearer token")

    asyncio.run(forward_twice())

    assert len(session.calls) == 2
    assert session.calls[1][2]["Authorization"] == "Bearer token"
