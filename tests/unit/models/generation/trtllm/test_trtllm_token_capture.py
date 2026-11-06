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
"""Token capture hosted by the TRT-LLM PD disagg frontend."""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_rl.models.generation.trtllm import token_capture
from nemo_rl.models.generation.trtllm.token_capture import (
    TokenCaptureMiddleware,
    TrtllmCaptureHost,
)

pytestmark = pytest.mark.trtllm

# The TRT-LLM venv cannot install the nemo_gym extra; this is the same path
# the frontend takes before importing Gym's capture modules.
token_capture._ensure_nemo_gym_importable()

from nemo_gym.token_id_capture import (  # noqa: E402
    NG_CAPTURE_FIELD,
    NG_COMMIT_COORDS_FIELD,
)
from nemo_gym.token_id_capture.protocols import CAPTURE_REFUSAL_HEADER  # noqa: E402


def _b64(ids: list[int]) -> str:
    return base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()


def _request(**extra) -> dict:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True,
        **extra,
    }


def _response(prompt_ids: list[int]) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "prompt_token_ids": prompt_ids,
                    "generation_token_ids": [7, 8],
                    "generation_log_probs": [-0.1, -0.2],
                },
                "token_ids": [7, 8],
                "logprobs": {"content": []},
            }
        ],
        "prompt_token_ids": prompt_ids,
    }


class _App:
    """Downstream ASGI app: records the body and answers with a fixed response."""

    def __init__(self, status: int = 200, payload: dict | None = None):
        self.status = status
        self.payload = payload or {}
        self.body = None

    async def __call__(self, scope, receive, send):
        message = await receive()
        self.body = json.loads(message["body"])
        body = json.dumps(self.payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": self.status,
                "headers": [(b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _drive(middleware, payload: dict) -> tuple[int, dict, dict]:
    body = json.dumps(payload).encode()
    sent = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "headers": [(b"content-length", str(len(body)).encode())],
    }
    asyncio.run(middleware(scope, receive, send))
    start, out = sent
    headers = dict(start["headers"])
    assert int(headers[b"content-length"]) == len(out["body"])
    return start["status"], headers, json.loads(out["body"])


class _FakeCapture:
    """Stands in for Gym's RolloutTokenCapture."""

    def __init__(self):
        self.completed = []
        self.failed = []

    def begin_call(self, admission, *, prefix_token_ids=None):
        return SimpleNamespace(admission=admission, prefix=prefix_token_ids)

    def complete_call_from_response(self, call, response):
        self.completed.append(dict(response))
        return SimpleNamespace(model_dump=lambda mode: {"status": "staged"})

    def fail_call(self, call, *, reason):
        self.failed.append(reason)
        return SimpleNamespace(model_dump=lambda mode: {"status": "capture_failed"})


def _installed_host() -> tuple[TrtllmCaptureHost, _FakeCapture]:
    host = TrtllmCaptureHost()
    capture = _FakeCapture()
    host.install_token_capture(capture)
    host._chain_prefix = MagicMock()
    return host, capture


def _tokenize_to(ids: list[int]):
    seen = []

    async def tokenize(payload, *, required_prefix_ids=None):
        seen.append(required_prefix_ids)
        return _b64(ids)

    return tokenize, seen


_PARENT_CHAIN_HASH = "a" * 64


def _text_admission() -> dict:
    return {"capture_key": "r0", "model_call_id": "c1", "mode": "text"}


def test_uninstalled_capture_passes_requests_through_untouched():
    app = _App(payload={"ok": True})
    tokenize, seen = _tokenize_to([1])
    middleware = TokenCaptureMiddleware(
        app, host=TrtllmCaptureHost(), tokenize_fn=tokenize
    )

    status, _, out = _drive(middleware, _request(**{NG_CAPTURE_FIELD: {"x": 1}}))

    # Strict request validation downstream rejects the field: capture is never
    # silently skipped on a frontend that has none installed.
    assert app.body[NG_CAPTURE_FIELD] == {"x": 1}
    assert (status, out, seen) == (200, {"ok": True}, [])


def test_captured_call_pins_the_prompt_and_answers_with_coords():
    host, capture = _installed_host()
    app = _App(payload=_response([1, 2, 3]))
    tokenize, seen = _tokenize_to([1, 2, 3])
    middleware = TokenCaptureMiddleware(app, host=host, tokenize_fn=tokenize)
    request = _request(
        **{NG_CAPTURE_FIELD: _text_admission(), "required_prefix_token_ids": [9]}
    )

    status, _, out = _drive(middleware, request)

    assert status == 200
    # A text root renders with no splice, whatever the inline override says.
    assert seen == [[]]
    assert NG_CAPTURE_FIELD not in app.body
    assert "required_prefix_token_ids" not in app.body
    assert app.body["prompt_token_ids_b64"] == _b64([1, 2, 3])
    assert capture.completed[0]["prompt_token_ids"] == [1, 2, 3]
    assert out[NG_COMMIT_COORDS_FIELD] == {"status": "staged"}
    assert "prompt_token_ids" not in out
    [choice] = out["choices"]
    assert "logprobs" not in choice and "token_ids" not in choice
    assert set(choice["message"]) == {"role", "content"}


def test_an_engine_prompt_other_than_the_captured_one_poisons_the_call():
    host, capture = _installed_host()
    app = _App(payload=_response([1, 2, 4]))
    tokenize, _ = _tokenize_to([1, 2, 3])
    middleware = TokenCaptureMiddleware(app, host=host, tokenize_fn=tokenize)

    _, _, out = _drive(middleware, _request(**{NG_CAPTURE_FIELD: _text_admission()}))

    assert capture.completed == []
    assert capture.failed == ["engine prompt differs from the captured prompt"]
    assert out[NG_COMMIT_COORDS_FIELD] == {"status": "capture_failed"}


def test_a_context_length_refusal_fails_the_call_and_is_acknowledged():
    host, capture = _installed_host()
    app = _App(status=400, payload={"detail": "context length exceeded: 9 > 8"})
    tokenize, _ = _tokenize_to([1])
    middleware = TokenCaptureMiddleware(app, host=host, tokenize_fn=tokenize)

    status, headers, _ = _drive(
        middleware, _request(**{NG_CAPTURE_FIELD: _text_admission()})
    )

    assert status == 400
    assert capture.failed == ["context_length"]
    assert headers[CAPTURE_REFUSAL_HEADER.encode()] == b"c1"


def test_an_engine_error_fails_the_call_without_a_refusal_header():
    host, capture = _installed_host()
    app = _App(status=500, payload={"detail": "boom"})
    tokenize, _ = _tokenize_to([1])
    middleware = TokenCaptureMiddleware(app, host=host, tokenize_fn=tokenize)

    status, headers, _ = _drive(
        middleware, _request(**{NG_CAPTURE_FIELD: _text_admission()})
    )

    assert status == 500
    assert capture.failed == ["engine_error"]
    assert CAPTURE_REFUSAL_HEADER.encode() not in headers


def test_a_token_in_call_splices_the_resolved_prefix():
    host, _ = _installed_host()
    app = _App(payload=_response([5, 6, 1]))
    tokenize, seen = _tokenize_to([5, 6, 1])
    middleware = TokenCaptureMiddleware(app, host=host, tokenize_fn=tokenize)
    admission = {
        "capture_key": "r0",
        "model_call_id": "c2",
        "mode": "token_in",
        "parent_call_id": "c1",
        "prev_len": 2,
        "parent_chain_hash": _PARENT_CHAIN_HASH,
        "required_prefix_token_ids": [5, 6],
    }

    _drive(middleware, _request(**{NG_CAPTURE_FIELD: admission}))

    assert seen == [[5, 6]]


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"stream": True}, "non-streaming"),
        ({}, "authorized capture prefix"),
    ],
)
def test_unadmittable_calls_are_rejected_before_inference(extra, message):
    host, capture = _installed_host()
    app = _App()
    # Renders a prompt that does not start with the inline prefix.
    tokenize, _ = _tokenize_to([1, 2])
    middleware = TokenCaptureMiddleware(app, host=host, tokenize_fn=tokenize)
    admission = {
        "capture_key": "r0",
        "model_call_id": "c2",
        "mode": "token_in",
        "parent_call_id": "c1",
        "prev_len": 2,
        "parent_chain_hash": _PARENT_CHAIN_HASH,
        "required_prefix_token_ids": [5, 6],
    }

    status, _, out = _drive(
        middleware, _request(**{NG_CAPTURE_FIELD: admission}, **extra)
    )

    assert status == 400
    assert message in out["error"]["message"]
    assert app.body is None
    assert capture.failed == []


@pytest.mark.parametrize("version", [-1, 1.0, True])
def test_rollout_weight_version_must_be_a_non_negative_int(version):
    with pytest.raises(ValueError, match="non-negative int"):
        TrtllmCaptureHost().set_rollout_weight_version(version)
