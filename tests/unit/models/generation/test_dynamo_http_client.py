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

import aiohttp
import pytest

from nemo_rl.models.generation.dynamo import http_client
from nemo_rl.models.generation.dynamo.rollout_client import _is_retryable


class _AsyncResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def read(self) -> bytes:
        return self._body


class _AsyncSession:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def post(self, url, *, json):
        self.requests.append((url, json))
        if self._error is not None:
            raise self._error
        return self._response


@pytest.mark.parametrize(
    ("body", "status", "expected", "retryable"),
    [
        (b'{"status":"ok"}', 200, {"status": "ok"}, False),
        (
            b"not-json",
            200,
            {"status": "error", "json_decode_error": True, "raw": "not-json"},
            True,
        ),
        (
            b"busy",
            503,
            {"status": "error", "http_status": 503, "raw": "busy"},
            True,
        ),
    ],
)
def test_async_http_post_json_error_contract(
    monkeypatch, body, status, expected, retryable
) -> None:
    session = _AsyncSession(_AsyncResponse(body, status))
    monkeypatch.setattr(
        http_client.aiohttp,
        "ClientSession",
        lambda **kwargs: session,
    )

    response = asyncio.run(
        http_client.async_http_post_json(
            "http://worker/route", {"value": 1}, timeout_s=3
        )
    )

    assert response == expected
    assert _is_retryable(response) is retryable
    if response.get("status") == "error":
        assert expected["raw"] in http_client.format_dynamo_error(response)
    assert session.requests == [("http://worker/route", {"value": 1})]


@pytest.mark.parametrize(
    ("error", "key", "retryable"),
    [
        (aiohttp.ClientConnectionError("refused"), "transport_error", True),
        # One request_timeout_s bounds a rollout: a timeout is not retried.
        (TimeoutError("slow"), "timeout_error", False),
        (aiohttp.ServerTimeoutError("slow read"), "timeout_error", False),
    ],
)
def test_async_http_transport_errors(monkeypatch, error, key, retryable) -> None:
    session = _AsyncSession(error=error)
    monkeypatch.setattr(
        http_client.aiohttp,
        "ClientSession",
        lambda **kwargs: session,
    )

    response = asyncio.run(
        http_client.async_http_post_json(
            "http://worker/route", {"value": 1}, timeout_s=3
        )
    )

    assert response["status"] == "error"
    assert _is_retryable(response) is retryable
    assert key in response
    assert response[key] in http_client.format_dynamo_error(response)
