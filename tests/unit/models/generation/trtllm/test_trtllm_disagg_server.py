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
