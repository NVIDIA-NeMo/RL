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

"""Send NeMo RL rollouts through the Dynamo frontend's KV-aware router."""

import asyncio
import json
import logging
from typing import Any, Optional

import aiohttp

from nemo_rl.experience.failures import http_status_is_infra

LOGGER = logging.getLogger(__name__)

# Marks a request as a NeMo RL rollout for the engine (native_engine), which
# then samples it from the generation config. Defined here, not in the engine
# module, so the driver need not import ai-dynamo.
ROLLOUT_ANNOTATION = "nemo_rl:rollout:"

_HTTP_MAX_ATTEMPTS = 3
_HTTP_RETRY_DELAY_S = 1.0


async def _post_json(
    url: str, payload: dict[str, Any], timeout_s: float | None
) -> dict[str, Any]:
    """POST JSON; a failure comes back as a dict with ``"status": "error"``."""
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as session:
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
        return {"status": "error", "timeout_error": f"no response within {timeout_s}s"}
    except aiohttp.ClientError as error:
        return {
            "status": "error",
            "transport_error": f"{type(error).__name__}: {error}",
        }
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


def _format_error(response: dict[str, Any]) -> str:
    if "http_status" in response:
        return f"HTTP {response['http_status']}: {response.get('raw', '')}"
    for key in ("transport_error", "timeout_error", "raw"):
        if key in response:
            return str(response[key])
    return str(response)


def _is_retryable(response: dict[str, Any]) -> bool:
    # A timeout is final: the request may still be running, and a retry would
    # let one hung rollout hold its slot for several request_timeout_s.
    if "transport_error" in response or response.get("json_decode_error") is True:
        return True
    status = response.get("http_status")
    return isinstance(status, int) and http_status_is_infra(status)


def rollout_payload(
    *,
    model_name: str,
    prompt_token_ids: list[int],
    max_model_len: int,
    greedy: bool,
    stop_strings: Optional[list[str]],
) -> dict[str, Any]:
    """The completion request for one rollout.

    Sampling is the engine's: the annotation tells it to build the request as
    ``generate_async`` does, from the generation config. ``ignore_eos`` and a
    context-sized ``max_tokens`` keep the frontend from ending the stream on
    its own stop rules before the engine reports the result.
    """
    rollout = {"greedy": greedy, "stop_strings": stop_strings or None}
    return {
        "model": model_name,
        "prompt": prompt_token_ids,
        "max_tokens": max(1, max_model_len - len(prompt_token_ids)),
        "ignore_eos": True,
        "nvext": {
            "annotations": [ROLLOUT_ANNOTATION + json.dumps(rollout)],
            "extra_fields": ["engine_data"],
        },
    }


def parse_rollout_response(response: dict[str, Any], *, url: str) -> dict[str, Any]:
    """Return the engine's ``engine_data``, or raise with the frontend's error.

    It holds the rollout result (``request_pipeline.rollout_result_to_wire``)
    and the serving worker's index.
    """
    if response.get("status") == "error":
        raise RuntimeError(
            f"Dynamo rollout request to {url} failed: {_format_error(response)}"
        )
    engine_data = (response.get("nvext") or {}).get("engine_data")
    if not isinstance(engine_data, dict) or "rollout" not in engine_data:
        raise RuntimeError(
            f"Dynamo rollout response from {url} carried no rollout engine_data; "
            "is every engine NeMo RL's vLLM worker?"
        )
    return engine_data


async def post_rollout(
    url: str, payload: dict[str, Any], timeout_s: float | None
) -> dict[str, Any]:
    """POST one rollout, retrying transient errors; returns its ``engine_data``."""
    response: dict[str, Any] = {}
    for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
        response = await _post_json(url, payload, timeout_s)
        if not _is_retryable(response) or attempt == _HTTP_MAX_ATTEMPTS:
            break
        LOGGER.warning(
            "Dynamo rollout attempt %d/%d failed; retrying in %.1fs: %s",
            attempt,
            _HTTP_MAX_ATTEMPTS,
            _HTTP_RETRY_DELAY_S,
            _format_error(response),
        )
        await asyncio.sleep(_HTTP_RETRY_DELAY_S)
    if "timeout_error" in response:
        # Name the setting, as the native path's timeout message names its own.
        raise RuntimeError(
            f"Dynamo rollout request to {url} failed: {response['timeout_error']}. "
            "For longer sequences, increase "
            "policy.generation.dynamo_cfg.request_timeout_s, or set it to null."
        )
    return parse_rollout_response(response, url=url)
