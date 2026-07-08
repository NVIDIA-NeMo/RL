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

import aiohttp
import pytest
import uvicorn
from fastapi.testclient import TestClient

from nemo_rl.models.generation.dynamo import token_wrapper as _wrapper
from nemo_rl.models.generation.dynamo.token_wrapper import (
    DynamoTokenWrapperServer,
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
                    "completion_logprobs": [-0.1],
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
    with pytest.raises(ValueError, match="only integer token IDs"):
        _validate_engine_data(
            {
                "nvext": {
                    "engine_data": {
                        "prompt_token_ids": ["invalid"],
                        "completion_token_ids": [],
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "completion_logprobs",
    [None, "invalid", [], [-0.1, -0.2], [None]],
)
def test_validate_engine_data_requires_aligned_completion_logprobs(
    completion_logprobs,
) -> None:
    with pytest.raises(ValueError, match="completion_logprobs"):
        _validate_engine_data(
            {
                "nvext": {
                    "engine_data": {
                        "prompt_token_ids": [1, 2],
                        "completion_token_ids": [3],
                        "completion_logprobs": completion_logprobs,
                    }
                }
            }
        )


def test_prepare_dynamo_chat_completion_request_forwards_template_controls() -> None:
    tokenizer = _Tokenizer()
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function"}],
        "tool_choice": "none",
        "add_generation_prompt": False,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"request_option": "request"},
    }

    prepared = prepare_dynamo_chat_completion_request(
        body,
        tokenizer=tokenizer,
        tokenizer_chat_template_kwargs={"global_option": "global"},
    )

    assert prepared["nvext"]["token_data"] == [10]
    assert tokenizer.calls[0]["tools"] is None
    assert tokenizer.calls[0]["add_generation_prompt"] is False
    assert tokenizer.calls[0]["kwargs"] == {
        "global_option": "global",
        "request_option": "request",
        "reasoning_effort": "high",
    }


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"messages": "not-a-list"}, "messages"),
        (
            {"messages": [{"role": "user", "content": "hello"}], "nvext": []},
            "nvext",
        ),
        (
            {
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": [],
            },
            "chat_template_kwargs",
        ),
    ],
)
def test_prepare_dynamo_chat_completion_request_rejects_invalid_shapes(
    body, match
) -> None:
    with pytest.raises(ValueError, match=match):
        prepare_dynamo_chat_completion_request(body, tokenizer=_Tokenizer())


def test_prepare_dynamo_chat_completion_request_rejects_invalid_global_kwargs() -> None:
    with pytest.raises(ValueError, match="tokenizer chat_template_kwargs"):
        prepare_dynamo_chat_completion_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            tokenizer=_Tokenizer(),
            tokenizer_chat_template_kwargs=[],  # type: ignore[arg-type]
        )


class _ResultTokenizer:
    eos_token_id = 2

    def __init__(self, result) -> None:
        self.result = result

    def apply_chat_template(self, *args, **kwargs):
        return self.result


def test_prepare_dynamo_chat_completion_request_accepts_nested_token_ids() -> None:
    prepared = prepare_dynamo_chat_completion_request(
        {"messages": [{"role": "user", "content": "hello"}]},
        tokenizer=_ResultTokenizer([[10, 11]]),
    )

    assert prepared["nvext"]["token_data"] == [10, 11]


@pytest.mark.parametrize("token_ids", [[[10], [11]], [10, "invalid"]])
def test_prepare_dynamo_chat_completion_request_rejects_invalid_token_ids(
    token_ids,
) -> None:
    with pytest.raises(ValueError, match="prompt token IDs|one list"):
        prepare_dynamo_chat_completion_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            tokenizer=_ResultTokenizer(token_ids),
        )


def _start_wrapper(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, app, **kwargs) -> None:
            captured["app"] = app
            captured["config_kwargs"] = kwargs

    class FakeServer:
        def __init__(self, config) -> None:
            self.config = config
            self.should_exit = False

        def run(self) -> None:
            raise AssertionError("The mocked uvicorn thread must not execute.")

    class FakeThread:
        def __init__(self, *, target, name, daemon) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = False
            self.join_timeout = None

        def start(self) -> None:
            self.started = True

        def join(self, timeout) -> None:
            self.join_timeout = timeout

    monkeypatch.setattr(uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(_wrapper, "threading", SimpleNamespace(Thread=FakeThread))
    monkeypatch.setattr(_wrapper, "_get_node_ip_local", lambda: "127.0.0.1")
    monkeypatch.setattr(_wrapper, "_get_free_port_local", lambda: 23456)

    wrapper = DynamoTokenWrapperServer(
        dynamo_frontend_base_url="http://dynamo.example.com:8000/v1",
        tokenizer=_Tokenizer(),
        tokenizer_chat_template_kwargs=None,
        request_timeout_s=5.0,
    )
    assert wrapper.start() == "http://127.0.0.1:23456/v1"
    return wrapper, TestClient(captured["app"]), captured


def test_token_wrapper_routes_and_lifecycle(monkeypatch) -> None:
    wrapper, client, captured = _start_wrapper(monkeypatch)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "dynamo_frontend_base_url": "http://dynamo.example.com:8000/v1",
    }

    forwarded = {}

    async def forward_success(request_body, *, authorization):
        forwarded["request_body"] = request_body
        forwarded["authorization"] = authorization
        return 200, {
            "nvext": {
                "engine_data": {
                    "prompt_token_ids": [10],
                    "completion_token_ids": [11],
                    "completion_logprobs": [-0.1],
                }
            }
        }

    monkeypatch.setattr(wrapper, "_forward_chat_completion", forward_success)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer token"},
    )
    assert response.status_code == 200
    assert forwarded["authorization"] == "Bearer token"
    assert forwarded["request_body"]["nvext"]["token_data"] == [10, 99]

    async def forward_invalid_engine_data(request_body, *, authorization):
        return 200, {"nvext": {}}

    monkeypatch.setattr(
        wrapper, "_forward_chat_completion", forward_invalid_engine_data
    )
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 502
    assert "engine_data" in response.json()["error"]["message"]

    async def forward_upstream_error(request_body, *, authorization):
        return 429, {"error": {"message": "busy"}}

    monkeypatch.setattr(wrapper, "_forward_chat_completion", forward_upstream_error)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 429
    assert response.json() == {"error": {"message": "busy"}}

    invalid_json = client.post(
        "/v1/chat/completions",
        content="{",
        headers={"Content-Type": "application/json"},
    )
    assert invalid_json.status_code == 400
    assert client.post("/v1/chat/completions", json=[]).status_code == 400
    assert (
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ).status_code
        == 400
    )

    assert wrapper.thread.started is True
    assert captured["config_kwargs"] == {
        "host": "0.0.0.0",
        "port": 23456,
        "timeout_keep_alive": 120,
    }
    wrapper.shutdown()
    assert wrapper.server.should_exit is True
    assert wrapper.thread.join_timeout == 10
    client.close()


@pytest.mark.parametrize(
    ("response_text", "expected_body"),
    [
        ("", {}),
        ('{"ok": true}', {"ok": True}),
        ("not-json", {"raw": "not-json"}),
        ("[1, 2]", {"response": [1, 2]}),
    ],
)
def test_forward_chat_completion_normalizes_responses(
    monkeypatch, response_text, expected_body
) -> None:
    state = {}

    class FakeResponse:
        status = 201

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def text(self):
            return response_text

    class FakeSession:
        def __init__(self, *, timeout) -> None:
            state["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, *, json, headers):
            state["url"] = url
            state["json"] = json
            state["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    wrapper = DynamoTokenWrapperServer(
        dynamo_frontend_base_url="http://dynamo.example.com:8000/v1/",
        tokenizer=_Tokenizer(),
        tokenizer_chat_template_kwargs=None,
        request_timeout_s=4.5,
    )

    status, body = asyncio.run(
        wrapper._forward_chat_completion(
            {"messages": []},
            authorization="Bearer token",
        )
    )

    assert status == 201
    assert body == expected_body
    assert state["url"] == "http://dynamo.example.com:8000/v1/chat/completions"
    assert state["json"] == {"messages": []}
    assert state["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
    }
    assert state["timeout"].total == 4.5


@pytest.mark.parametrize(
    ("error", "expected_status", "message_fragment"),
    [
        (asyncio.TimeoutError(), 504, "Timed out"),
        (aiohttp.ClientConnectionError("connection failed"), 502, "connection failed"),
    ],
)
def test_forward_chat_completion_maps_client_errors(
    monkeypatch, error, expected_status, message_fragment
) -> None:
    class FailingSession:
        def __init__(self, *, timeout) -> None:
            assert timeout.total is None

        async def __aenter__(self):
            raise error

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", FailingSession)
    wrapper = DynamoTokenWrapperServer(
        dynamo_frontend_base_url="http://dynamo.example.com:8000/v1",
        tokenizer=_Tokenizer(),
        tokenizer_chat_template_kwargs=None,
        request_timeout_s=None,
    )

    status, body = asyncio.run(
        wrapper._forward_chat_completion({"messages": []}, authorization=None)
    )

    assert status == expected_status
    assert message_fragment in body["error"]["message"]


def test_token_wrapper_shutdown_before_start() -> None:
    wrapper = DynamoTokenWrapperServer(
        dynamo_frontend_base_url="http://dynamo.example.com:8000/v1",
        tokenizer=_Tokenizer(),
        tokenizer_chat_template_kwargs=None,
        request_timeout_s=None,
    )

    wrapper.shutdown()
