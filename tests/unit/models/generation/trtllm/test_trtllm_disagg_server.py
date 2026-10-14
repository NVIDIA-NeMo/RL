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
"""The disagg frontend's request-adapting middleware.

TRT-LLM's OpenAIDisaggServer validates bodies against ``extra="forbid"``
models, so any vLLM-only request field must be stripped before it reaches the
route. ``required_prefix_token_ids`` (the vLLM prefix-override extension the
MLPerf warmup sends) turned every warmup request into a 400 until it was added
to the strip list; the TRT-LLM engine adapter ignores that field anyway.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.trtllm.trtllm_disagg_server import (
    _GYM_ONLY_REQUEST_FIELDS,
    _DropGymOnlyRequestFields,
    attach_rollout_fields,
    generation_token_ids,
)

pytestmark = pytest.mark.trtllm


class _CaptureApp:
    """Downstream ASGI app recording the scope and body it is handed."""

    def __init__(self):
        self.scope = None
        self.body = b""

    async def __call__(self, scope, receive, send):
        self.scope = scope
        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        self.body = b"".join(chunks)


def _scope(path: str, body: bytes) -> dict:
    return {
        "type": "http",
        "path": path,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    }


def _drive(middleware, scope, body: bytes) -> None:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(_message):
        return None

    asyncio.run(middleware(scope, receive, send))


def _warmup_payload() -> dict:
    return {
        "model": "qwen",
        "messages": [{"role": "user", "content": "MLPerf warmup 0-0"}],
        "required_prefix_token_ids": [11, 22, 33],
        "return_token_ids": True,
        "max_tokens": 8,
        "temperature": 1.0,
    }


def test_required_prefix_token_ids_is_a_stripped_field():
    assert "required_prefix_token_ids" in _GYM_ONLY_REQUEST_FIELDS


def test_parallel_tool_calls_is_a_stripped_field():
    """A tool-use dataset can carry it; TRT-LLM's request model does not declare it.

    Unlike the other entries this is an OpenAI-standard chat-completions field,
    so it arrives from ordinary Gym traffic rather than a vLLM extension:
    responses_create_params carries ``parallel_tool_calls`` and the vllm_model
    proxy forwards it verbatim. TRT-LLM's ChatCompletionRequest stops at
    ``tools`` / ``tool_choice`` and is extra="forbid", so leaving it in turns
    every rollout request into a 400 and the whole run dies in rollout
    collection rather than at startup.
    """
    assert "parallel_tool_calls" in _GYM_ONLY_REQUEST_FIELDS


def test_a_tool_use_body_survives_stripping_with_its_tools_intact():
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "fix the bug"}],
        "tools": [{"type": "function", "function": {"name": "str_replace_editor"}}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": 8,
    }
    body = json.dumps(payload).encode()
    app = _CaptureApp()

    _drive(_DropGymOnlyRequestFields(app), _scope("/v1/chat/completions", body), body)

    got = json.loads(app.body)
    assert "parallel_tool_calls" not in got
    # The fields TRT-LLM does declare must not be collateral damage.
    assert got["tools"] == payload["tools"]
    assert got["tool_choice"] == "auto"
    headers = dict(app.scope["headers"])
    assert int(headers[b"content-length"]) == len(app.body)


def test_vllm_only_fields_are_stripped_and_content_length_follows():
    payload = _warmup_payload()
    body = json.dumps(payload).encode()
    app = _CaptureApp()

    _drive(_DropGymOnlyRequestFields(app), _scope("/v1/chat/completions", body), body)

    got = json.loads(app.body)
    assert "required_prefix_token_ids" not in got
    assert "return_token_ids" not in got
    # Everything the engine needs survives untouched.
    assert got["messages"] == payload["messages"]
    assert got["max_tokens"] == 8 and got["temperature"] == 1.0
    headers = dict(app.scope["headers"])
    assert int(headers[b"content-length"]) == len(app.body)


def test_frontend_tokenizer_never_sees_the_stripped_fields():
    seen = []

    async def tokenize_fn(payload):
        seen.append(dict(payload))
        return "QUJD"  # base64("ABC")

    payload = _warmup_payload()
    body = json.dumps(payload).encode()
    app = _CaptureApp()

    _drive(
        _DropGymOnlyRequestFields(app, tokenize_fn=tokenize_fn),
        _scope("/v1/chat/completions", body),
        body,
    )

    assert len(seen) == 1
    assert "required_prefix_token_ids" not in seen[0]
    assert json.loads(app.body)["prompt_token_ids_b64"] == "QUJD"


def test_non_adapted_paths_pass_through_untouched():
    body = b'{"required_prefix_token_ids": [1]}'
    scope = _scope("/v1/models", body)
    app = _CaptureApp()

    _drive(_DropGymOnlyRequestFields(app), scope, body)

    assert app.body == body
    assert app.scope is scope


def test_a_failing_frontend_tokenizer_falls_back_instead_of_failing_the_request():
    async def tokenize_fn(payload):
        raise RuntimeError("tokenizer died")

    payload = _warmup_payload()
    body = json.dumps(payload).encode()
    app = _CaptureApp()

    _drive(
        _DropGymOnlyRequestFields(app, tokenize_fn=tokenize_fn),
        _scope("/v1/chat/completions", body),
        body,
    )

    # No b64 ids, so the ctx adapter tokenizes from the messages itself --
    # degraded, not a 500.
    got = json.loads(app.body)
    assert "prompt_token_ids_b64" not in got
    assert "required_prefix_token_ids" not in got
    assert got["messages"] == payload["messages"]


def test_engine_facing_legs_are_not_retokenized():
    calls = []

    async def tokenize_fn(payload):
        calls.append(payload)
        return "QUJD"

    payload = _warmup_payload()
    payload["disaggregated_params"] = {"request_type": "generation_only"}
    body = json.dumps(payload).encode()
    app = _CaptureApp()

    _drive(
        _DropGymOnlyRequestFields(app, tokenize_fn=tokenize_fn),
        _scope("/v1/chat/completions", body),
        body,
    )

    # The generation leg's history may already be stripped, so re-deriving the
    # prompt ids from it would contradict the KV the context leg transferred.
    assert calls == []
    assert "prompt_token_ids_b64" not in json.loads(app.body)


def test_a_chunked_body_is_reassembled_before_the_fields_are_stripped():
    payload = _warmup_payload()
    body = json.dumps(payload).encode()
    half = len(body) // 2
    app = _CaptureApp()
    middleware = _DropGymOnlyRequestFields(app)

    parts = iter(
        [
            {"type": "http.request", "body": body[:half], "more_body": True},
            {"type": "http.request", "body": body[half:], "more_body": False},
        ]
    )

    async def receive():
        return next(parts)

    async def send(_message):
        return None

    asyncio.run(middleware(_scope("/v1/chat/completions", body), receive, send))

    got = json.loads(app.body)
    assert "required_prefix_token_ids" not in got
    assert got["messages"] == payload["messages"]


def test_a_non_json_body_is_forwarded_verbatim():
    body = b"not json at all"
    app = _CaptureApp()

    _drive(_DropGymOnlyRequestFields(app), _scope("/v1/completions", body), body)

    assert app.body == body


# -------------------------------------------------------------------------- #
#  Outbound: the rollout fields NeMo-Gym reads off choices[].message
# -------------------------------------------------------------------------- #


def _disagg_response(*, token_ids=None, logprobs_content=None) -> dict:
    """The shape trtllm_http_server emits on the disagg (non-aggregated) path."""
    choice: dict = {
        "index": 0,
        "message": {"role": "assistant", "content": "hi", "reasoning_content": None},
        "finish_reason": "stop",
    }
    if token_ids is not None:
        choice["token_ids"] = token_ids
    if logprobs_content is not None:
        choice["logprobs"] = {"content": logprobs_content}
    return {"choices": [choice], "prompt_token_ids": [7, 8, 9]}


def test_rollout_fields_move_onto_the_message_from_the_declared_token_ids_field():
    payload = attach_rollout_fields(
        _disagg_response(
            token_ids=[4, 5],
            logprobs_content=[
                {"token": "token_id:4", "logprob": -0.5},
                {"token": "token_id:5", "logprob": -1.5},
            ],
        )
    )

    message = payload["choices"][0]["message"]
    assert message["prompt_token_ids"] == [7, 8, 9]
    assert message["generation_token_ids"] == [4, 5]
    assert message["generation_log_probs"] == [-0.5, -1.5]


def test_generation_token_ids_fall_back_to_the_logprobs_encoding():
    # The build in use has no ChatCompletionResponseChoice.token_ids, so the ids
    # ride the declared logprobs strings instead.
    payload = attach_rollout_fields(
        _disagg_response(
            logprobs_content=[
                {"token": "token_id:11", "logprob": -0.1},
                {"token": "token_id:22", "logprob": -0.2},
            ]
        )
    )

    assert payload["choices"][0]["message"]["generation_token_ids"] == [11, 22]


def test_decoded_tokens_are_not_mistaken_for_ids():
    assert generation_token_ids({"logprobs": {"content": [{"token": "hello"}]}}) is None


def test_a_response_without_choices_is_returned_unchanged():
    assert attach_rollout_fields({"choices": []}) == {"choices": []}


def _engine_rejection(status: int, body: dict):
    import aiohttp

    # aiohttp's message is the engine's raw response body.
    return aiohttp.ClientResponseError(
        request_info=MagicMock(), history=(), status=status, message=json.dumps(body)
    )


def _adaptor_handling(exception) -> Exception:
    pytest.importorskip("tensorrt_llm.serve.openai_disagg_server")
    from nemo_rl.models.generation.trtllm.trtllm_disagg_server import (
        _build_adaptor_class,
    )

    server = SimpleNamespace(_perf_metrics_collector=MagicMock())
    with pytest.raises(Exception) as raised:
        _build_adaptor_class()._handle_exception(server, exception)
    return raised.value


def test_an_engine_context_length_rejection_keeps_its_text():
    # Gym classifies overflow by this text. The base handler would replace it,
    # because it only unwraps {"error": {"message": ...}} / {"message": ...}.
    body = {"error": "context length exceeded: prompt (9000 tokens) exhausted"}

    raised = _adaptor_handling(_engine_rejection(400, body))

    assert raised.status_code == 400
    assert "context length exceeded" in raised.detail


# The shape NeMo-Gym sends for interleaved reasoning: it writes the extracted
# <think> block into both fields, which makes TRT-LLM resolve the message to
# ReasoningAssistantMessage -- the case whose tool_calls dumped as [].
_REASONING_TOOL_CALL_MESSAGES = [
    {"role": "user", "content": "fix the bug"},
    {
        "role": "assistant",
        "content": "Let me look.",
        "reasoning": "check the repo first",
        "reasoning_content": "check the repo first",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "arguments": '{"command": "ls"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
]


def _dumped_tool_calls(dump: dict) -> list:
    return [m.get("tool_calls") for m in dump["messages"] if m["role"] == "assistant"]


def test_assistant_tool_calls_survive_every_dump_of_the_request():
    # The disagg server dumps one request once per leg, and the frontend
    # tokenizer dumps its own copy; every dump must carry the history's calls.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from nemo_rl.models.generation.trtllm.trtllm_http_server import (
        _chat_completion_request_cls,
    )

    request = _chat_completion_request_cls()(
        model="m", messages=_REASONING_TOOL_CALL_MESSAGES
    )

    for _ in range(3):
        dump = request.model_dump(mode="json", exclude_unset=True)
        [tool_calls] = _dumped_tool_calls(dump)
        assert [c["function"]["name"] for c in tool_calls] == ["execute_bash"]


def test_the_frontend_tokenizer_renders_the_history_tool_calls(monkeypatch):
    # With frontend_tokenize the ctx engine uses these ids verbatim, so a
    # dropped tool call here is a dropped tool call in the trained prompt.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    import transformers

    from nemo_rl.models.generation.trtllm import trtllm_http_server
    from nemo_rl.models.generation.trtllm.trtllm_disagg_server import (
        _build_frontend_tokenizer,
    )

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: MagicMock()
    )
    rendered = {}

    async def fake_build(messages, tools, tokenizer, model_config, template_kwargs):
        rendered["messages"] = messages
        return [1, 2, 3]

    monkeypatch.setattr(trtllm_http_server, "build_spliced_prompt_ids", fake_build)
    tokenize = _build_frontend_tokenizer("m", {})

    asyncio.run(tokenize({"model": "m", "messages": _REASONING_TOOL_CALL_MESSAGES}))

    [tool_calls] = _dumped_tool_calls({"messages": rendered["messages"]})
    assert [c["function"]["name"] for c in tool_calls] == ["execute_bash"]
