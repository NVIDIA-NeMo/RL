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

from nemo_rl.models.generation.dynamo.token_wrapper import (
    _validate_engine_data,
    prepare_dynamo_chat_completion_request,
)


class _Tokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.calls = []

    def decode(self, token_ids):
        return repr(token_ids)

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
        for message in conversation:
            role = message["role"]
            content = message.get("content")
            if role == "user" and content == "hello":
                token_ids.extend([10])
            elif role == "assistant" and content == "first":
                token_ids.extend([300, self.eos_token_id])
            elif role == "user" and content == "next":
                token_ids.extend([40])
            else:
                token_ids.extend([900])
        if add_generation_prompt:
            token_ids.extend([99])
        return token_ids


def test_prepare_dynamo_chat_completion_request_first_turn() -> None:
    tokenizer = _Tokenizer()
    body = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "nvext": {"extra_fields": ["timing"], "trace": "keep-me"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=tokenizer,
        tokenizer_chat_template_kwargs={"enable_thinking": True},
    )

    assert prepared["messages"] == [{"role": "user", "content": "hello"}]
    assert "logprobs" not in prepared
    assert "return_tokens_as_token_ids" not in prepared
    assert prepared["nvext"] == {
        "extra_fields": ["engine_data"],
        "trace": "keep-me",
        "token_data": [10, 99],
    }
    assert tokenizer.calls[0]["kwargs"] == {"enable_thinking": False}


def test_prepare_dynamo_chat_completion_request_preserves_logprob_fields() -> None:
    tokenizer = _Tokenizer()
    body = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "logprobs": True,
        "return_tokens_as_token_ids": True,
    }

    prepared = prepare_dynamo_chat_completion_request(body, tokenizer=tokenizer)

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

    prepared = prepare_dynamo_chat_completion_request(body, tokenizer=tokenizer)

    assert prepared["nvext"]["token_data"] == [10, 31, 32, 2, 40, 99]
    assert "required_prefix_token_ids" not in prepared
    assert "prompt_token_ids" not in prepared["messages"][1]
    assert "generation_token_ids" not in prepared["messages"][1]
    assert "generation_log_probs" not in prepared["messages"][1]
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[1]["add_generation_prompt"] is False


def test_prepare_dynamo_chat_completion_request_rejects_stream() -> None:
    with pytest.raises(ValueError, match="stream=True"):
        prepare_dynamo_chat_completion_request(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            tokenizer=_Tokenizer(),
        )


def test_prepare_dynamo_chat_completion_request_rejects_multiple_choices() -> None:
    with pytest.raises(ValueError, match="only n=1"):
        prepare_dynamo_chat_completion_request(
            {"messages": [{"role": "user", "content": "hello"}], "n": 2},
            tokenizer=_Tokenizer(),
        )


def test_validate_engine_data_requires_prompt_and_completion_tokens() -> None:
    _validate_engine_data(
        {
            "nvext": {
                "engine_data": {
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3],
                }
            }
        }
    )

    with pytest.raises(ValueError, match="engine_data"):
        _validate_engine_data({"nvext": {}})
    with pytest.raises(ValueError, match="prompt_token_ids"):
        _validate_engine_data({"nvext": {"engine_data": {"completion_token_ids": []}}})
    with pytest.raises(ValueError, match="completion_token_ids"):
        _validate_engine_data({"nvext": {"engine_data": {"prompt_token_ids": []}}})
