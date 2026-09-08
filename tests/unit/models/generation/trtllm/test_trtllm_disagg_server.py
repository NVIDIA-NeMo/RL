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

from nemo_rl.models.generation.trtllm.trtllm_disagg_server import (
    _GYM_ONLY_REQUEST_FIELDS,
    _DropGymOnlyRequestFields,
)


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
