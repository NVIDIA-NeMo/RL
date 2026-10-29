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

"""HTTP helpers for requests to the managed Dynamo frontend."""

import json
from typing import Any

import aiohttp


def _decode_json_object(body: bytes) -> dict[str, Any]:
    """Decode an HTTP response body using the shared Dynamo error contract."""
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "json_decode_error": True,
            "raw": body.decode("utf-8", "replace"),
        }
    if not isinstance(decoded, dict):
        return {"status": "error", "raw": repr(decoded)}
    return decoded


async def async_http_post_json(
    url: str, payload: dict[str, Any], timeout_s: float | None
) -> dict[str, Any]:
    """POST JSON without blocking the rollout actor event loop."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                body = await response.read()
                if response.status >= 400:
                    return {
                        "status": "error",
                        "http_status": response.status,
                        "raw": body.decode("utf-8", "replace"),
                    }
    except TimeoutError:
        # Before ClientError: aiohttp's ServerTimeoutError is both.
        return {
            "status": "error",
            "timeout_error": f"no response within {timeout_s}s",
        }
    except aiohttp.ClientError as error:
        return {
            "status": "error",
            "transport_error": f"{type(error).__name__}: {error}",
        }
    return _decode_json_object(body)


def format_dynamo_error(response: dict[str, Any]) -> str:
    """Format an error object returned by :func:`async_http_post_json`."""
    if "http_status" in response:
        return f"HTTP {response['http_status']}: {response.get('raw', '')}"
    if "transport_error" in response:
        return str(response["transport_error"])
    if "timeout_error" in response:
        return str(response["timeout_error"])
    if "raw" in response:
        return str(response["raw"])
    return str(response)
