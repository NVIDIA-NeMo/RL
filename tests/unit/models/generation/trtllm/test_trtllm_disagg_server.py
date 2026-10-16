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
    assert b"x-nemo-fe-recv-us" in headers and b"x-nemo-fe-fwd-us" in headers


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


def _attach(payload: dict) -> dict:
    """Run the outbound adaptor's _attach_rollout_fields on *payload*."""
    import pytest

    pytest.importorskip("tensorrt_llm")
    from types import SimpleNamespace

    from fastapi.responses import JSONResponse

    from nemo_rl.models.generation.trtllm.trtllm_disagg_server import (
        _build_adaptor_class,
    )

    cls = _build_adaptor_class()
    self = SimpleNamespace(_generation_token_ids=cls._generation_token_ids)
    out = cls._attach_rollout_fields(self, JSONResponse(content=payload), None)
    return json.loads(out.body)


def _final_payload(**prompt_fields) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
                "token_ids": [7, 8],
                "logprobs": {
                    "content": [
                        {"token": "token_id:7", "logprob": -0.25},
                        {"token": "token_id:8", "logprob": -0.5},
                    ]
                },
            }
        ],
        **prompt_fields,
    }


def test_rollout_fields_decode_a_relayed_b64_prompt():
    import base64

    import numpy as np

    ids = [101, 202, 303, 150_000]
    b64 = base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()

    got = _attach(_final_payload(prompt_token_ids_b64=b64))

    # Gym sees exactly what the int-array path produced: a list, top level and
    # on the message, and no base64 field.
    assert got["prompt_token_ids"] == ids
    assert got["choices"][0]["message"]["prompt_token_ids"] == ids
    assert "prompt_token_ids_b64" not in got
    assert got["choices"][0]["message"]["generation_token_ids"] == [7, 8]
    assert got["choices"][0]["message"]["generation_log_probs"] == [-0.25, -0.5]


def test_rollout_fields_keep_an_int_list_prompt():
    got = _attach(_final_payload(prompt_token_ids=[1, 2, 3]))

    assert got["choices"][0]["message"]["prompt_token_ids"] == [1, 2, 3]
    assert got["prompt_token_ids"] == [1, 2, 3]
