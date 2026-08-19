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
import io
import json
import urllib.error

import aiohttp
import pytest

from nemo_rl.models.generation.dynamo import dynamo_generation as generation_module
from nemo_rl.models.generation.dynamo import http_client


class _SyncResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return self._body


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


def _sync_post(monkeypatch, result):
    def fake_urlopen(request, timeout):
        if isinstance(result, BaseException):
            raise result
        return _SyncResponse(result)

    monkeypatch.setattr(http_client.urllib.request, "urlopen", fake_urlopen)
    return http_client.http_post_json("http://worker/route", {"value": 1}, 3)


def test_http_post_json_success_preserves_request_contract(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _SyncResponse(b'{"status":"ok"}')

    monkeypatch.setattr(http_client.urllib.request, "urlopen", fake_urlopen)

    response = http_client.http_post_json(
        "http://worker/route", {"value": 1}, timeout_s=3
    )

    request = captured["request"]
    assert response == {"status": "ok"}
    assert request.full_url == "http://worker/route"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"value": 1}
    assert captured["timeout"] == 3


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            b"not-json",
            {"status": "error", "json_decode_error": True, "raw": "not-json"},
        ),
        (b"[1, 2]", {"status": "error", "raw": "[1, 2]"}),
    ],
)
def test_http_post_json_rejects_invalid_json_shapes(
    monkeypatch, body, expected
) -> None:
    assert _sync_post(monkeypatch, body) == expected


@pytest.mark.parametrize(
    ("error", "retryable", "message"),
    [
        (
            urllib.error.HTTPError(
                "http://worker/route",
                503,
                "unavailable",
                hdrs=None,
                fp=io.BytesIO(b"try later"),
            ),
            True,
            "HTTP 503: try later",
        ),
        (
            urllib.error.HTTPError(
                "http://worker/route", 400, "bad request", hdrs=None, fp=None
            ),
            False,
            "HTTP 400: ",
        ),
        (urllib.error.URLError("refused"), True, "URLError"),
        (TimeoutError("slow"), True, "TimeoutError"),
    ],
)
def test_http_errors_round_trip_through_consumers(
    monkeypatch, error, retryable, message
) -> None:
    response = _sync_post(monkeypatch, error)

    assert response["status"] == "error"
    assert generation_module._is_retryable_http_response(response) is retryable
    assert message in http_client.format_dynamo_error(response)


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
def test_async_http_post_json_uses_same_error_contract(
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
    assert generation_module._is_retryable_http_response(response) is retryable
    assert session.requests == [("http://worker/route", {"value": 1})]


@pytest.mark.parametrize(
    "error", [aiohttp.ClientConnectionError("refused"), TimeoutError("slow")]
)
def test_async_http_transport_errors_are_retryable(monkeypatch, error) -> None:
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
    assert generation_module._is_retryable_http_response(response)
    assert "transport_error" in response
