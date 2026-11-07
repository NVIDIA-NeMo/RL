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

"""One native rollout through the Dynamo frontend's OpenAI completions endpoint.

The frontend tokenizes nothing here: the prompt is sent as token IDs and the
engine's sampled token IDs and log-probabilities come back in the response, so
the output matches what the engine would return to a direct ``generate_async``.
"""

import asyncio
import json
import logging
from typing import Any, Optional

import aiohttp
import torch

from nemo_rl.data.multimodal_utils import VLLM_PROMPT_KEYS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.megatron.config import MCoreGenerationConfig

LOGGER = logging.getLogger(__name__)

_HTTP_MAX_ATTEMPTS = 3
_HTTP_RETRY_DELAY_S = 1.0
_RETRYABLE_HTTP_STATUS_CODES = {408, 429}
# The frontend caps the stop strings of one request.
_MAX_STOP_STRINGS = 32


def _decode_json_object(body: bytes) -> dict[str, Any]:
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
    url: str, payload: dict[str, Any], timeout_s: Optional[float]
) -> dict[str, Any]:
    """POST JSON without blocking the rollout actor's event loop.

    Returns the decoded object, or an error object with ``status: error`` and
    one of ``http_status``, ``transport_error`` or ``json_decode_error``.
    """
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
    except (aiohttp.ClientError, TimeoutError) as error:
        return {
            "status": "error",
            "transport_error": f"{type(error).__name__}: {error}",
        }
    return _decode_json_object(body)


def format_error(response: dict[str, Any]) -> str:
    """Format an error object returned by :func:`async_http_post_json`."""
    if "http_status" in response:
        return f"HTTP {response['http_status']}: {response.get('raw', '')}"
    if "transport_error" in response:
        return str(response["transport_error"])
    if "raw" in response:
        return str(response["raw"])
    return str(response)


def is_retryable(response: Any) -> bool:
    """Whether an error object represents a transient failure."""
    if not isinstance(response, dict):
        return False
    if "transport_error" in response or response.get("json_decode_error") is True:
        return True
    status = response.get("http_status")
    return isinstance(status, int) and (
        status in _RETRYABLE_HTTP_STATUS_CODES or 500 <= status < 600
    )


def parse_completion_response(
    response: dict[str, Any], *, request_url: str
) -> tuple[list[int], list[float], bool]:
    """The generated token IDs, their log-probabilities, and whether the length cap hit."""
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} was not a JSON object."
        )
    if response.get("status") == "error":
        raise RuntimeError(
            f"Dynamo completion request to {request_url} failed: "
            f"{format_error(response)}"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include choices."
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} has invalid choice shape."
        )
    nvext = response.get("nvext")
    if not isinstance(nvext, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include nvext."
        )
    completion_token_ids = nvext.get("completion_token_ids")
    if not isinstance(completion_token_ids, list):
        raise RuntimeError(
            "Dynamo completion response did not include nvext.completion_token_ids. "
            "Ensure the Dynamo frontend is configured to return completion token IDs."
        )
    generated_token_ids = [int(token_id) for token_id in completion_token_ids]
    truncated = choice.get("finish_reason") == "length"
    if not generated_token_ids:
        return generated_token_ids, [], truncated

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include "
            "choice.logprobs."
        )
    token_logprobs = logprobs.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include "
            "choice.logprobs.token_logprobs."
        )
    if len(token_logprobs) != len(generated_token_ids):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} returned "
            f"{len(token_logprobs)} token logprobs for "
            f"{len(generated_token_ids)} generated tokens."
        )
    generated_logprobs = []
    for idx, logprob in enumerate(token_logprobs):
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise RuntimeError(
                f"Dynamo completion response from {request_url} returned invalid "
                f"logprob {logprob!r} for generated token {idx}."
            )
        generated_logprobs.append(float(logprob))
    return generated_token_ids, generated_logprobs, truncated


async def post_completion(
    url: str, payload: dict[str, Any], timeout_s: Optional[float]
) -> tuple[list[int], list[float], bool]:
    """POST one completion, retrying transient failures a few times."""
    response: dict[str, Any] = {}
    for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
        response = await async_http_post_json(url, payload, timeout_s)
        if not is_retryable(response) or attempt == _HTTP_MAX_ATTEMPTS:
            break
        LOGGER.warning(
            "Dynamo completion attempt %d/%d failed; retrying in %.1fs: %s",
            attempt,
            _HTTP_MAX_ATTEMPTS,
            _HTTP_RETRY_DELAY_S,
            format_error(response),
        )
        await asyncio.sleep(_HTTP_RETRY_DELAY_S)
    return parse_completion_response(response, request_url=url)


def merge_stop_strings(
    configured: Optional[list[str]], per_sample: Any
) -> Optional[list[str]]:
    """The configured stop strings plus one sample's, within the frontend's cap."""
    stop_set: set[str] = set(configured or ())
    if per_sample:
        if isinstance(per_sample, str):
            stop_set.add(per_sample)
        else:
            stop_set.update(per_sample)
    if len(stop_set) > _MAX_STOP_STRINGS:
        raise ValueError(
            f"Dynamo supports at most {_MAX_STOP_STRINGS} stop strings after merging "
            "configured and per-sample values"
        )
    return sorted(stop_set) if stop_set else None


def allowed_new_tokens(cfg: MCoreGenerationConfig, input_length: int) -> int:
    """The generation budget for a prompt, within the engines' context."""
    max_model_len = int(cfg["mcore_generation_config"]["max_model_len"])
    remaining = max_model_len - input_length
    if remaining <= 0:
        raise ValueError(
            f"Dynamo prompt length {input_length} must be less than "
            f"mcore_generation_config.max_model_len={max_model_len}"
        )
    return min(int(cfg["max_new_tokens"]), remaining)


def completion_payload(
    cfg: MCoreGenerationConfig,
    *,
    prompt_token_ids: list[int],
    greedy: bool,
    stop_strings: Optional[list[str]],
    max_new_tokens: int,
) -> dict[str, Any]:
    """The OpenAI completion request for one token-ID prompt."""
    top_k_cfg = cfg["top_k"]
    payload: dict[str, Any] = {
        "model": cfg["model_name"],
        "prompt": prompt_token_ids,
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0 if greedy else cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1),
        "n": 1,
        "logprobs": 0,
        "include_stop_str_in_output": True,
        "nvext": {"extra_fields": ["completion_token_ids"]},
    }
    if cfg["stop_token_ids"] is not None:
        payload["stop_token_ids"] = cfg["stop_token_ids"]
    if stop_strings is not None:
        payload["stop"] = stop_strings
    return payload


def single_sample_output(
    cfg: MCoreGenerationConfig,
    *,
    input_ids: torch.Tensor,
    input_length: int,
    generated_token_ids: list[int],
    generated_logprobs: list[float],
    truncated: bool,
) -> BatchedDataDict[GenerationOutputSpec]:
    """One sample's generation output, shaped as the direct path returns it."""
    output_length = input_length + len(generated_token_ids)
    max_model_len = int(cfg["mcore_generation_config"]["max_model_len"])
    if output_length > max_model_len:
        raise AssertionError(
            "Dynamo response length exceeded mcore_generation_config.max_model_len: "
            f"{output_length} > {max_model_len}"
        )
    output_ids = torch.full(
        (output_length,),
        cfg["_pad_token_id"],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    output_ids[:input_length] = input_ids[:input_length]
    if generated_token_ids:
        output_ids[input_length:output_length] = torch.tensor(
            generated_token_ids, dtype=input_ids.dtype, device=input_ids.device
        )
    logprobs = torch.zeros(
        (1, output_length), dtype=torch.float32, device=input_ids.device
    )
    for idx, logprob in enumerate(generated_logprobs[: len(generated_token_ids)]):
        logprobs[0, input_length + idx] = logprob
    return BatchedDataDict[GenerationOutputSpec](
        {
            "output_ids": output_ids.unsqueeze(0),
            "logprobs": logprobs,
            "generation_lengths": torch.tensor(
                [len(generated_token_ids)], dtype=torch.long, device=input_ids.device
            ),
            "unpadded_sequence_lengths": torch.tensor(
                [output_length], dtype=torch.long, device=input_ids.device
            ),
            "truncated": torch.tensor(
                [truncated], dtype=torch.bool, device=input_ids.device
            ),
        }
    )


async def generate_through_frontend(
    cfg: MCoreGenerationConfig,
    *,
    frontend_url: str,
    data: BatchedDataDict[GenerationDatumSpec],
    greedy: bool,
    request_timeout_s: Optional[float],
) -> BatchedDataDict[GenerationOutputSpec]:
    """Generate one token-ID prompt through the frontend's completions endpoint.

    Args:
        cfg: ``policy.generation``.
        frontend_url: The frontend's ``/v1`` URL.
        data: A single-sample batch with ``input_ids`` and ``input_lengths``.
        greedy: Greedy decoding instead of the configured sampling.
        request_timeout_s: Deadline for the HTTP request; None waits.
    """
    assert "input_ids" in data and "input_lengths" in data, (
        "input_ids and input_lengths are required in data for Dynamo generation"
    )
    if VLLM_PROMPT_KEYS & data.keys():
        raise NotImplementedError(
            "Megatron engines behind Dynamo serve token-ID prompts only; multimodal "
            "prompt data is not supported."
        )
    assert data.size == 1, (
        "generate_async is restricted to handle only single samples, "
        f"but received batch_size={data.size}. Please handle batching "
        "outside this method."
    )
    verify_right_padding(data, pad_value=cfg["_pad_token_id"])
    input_length = int(data["input_lengths"][0].item())
    input_ids = data["input_ids"][0]
    per_sample_stop_strings = data.get("stop_strings", [None])[0]
    payload = completion_payload(
        cfg,
        prompt_token_ids=input_ids[:input_length].tolist(),
        greedy=greedy,
        stop_strings=merge_stop_strings(
            cfg.get("stop_strings"), per_sample_stop_strings
        ),
        max_new_tokens=allowed_new_tokens(cfg, input_length),
    )
    generated_token_ids, generated_logprobs, truncated = await post_completion(
        f"{frontend_url.rstrip('/')}/completions", payload, request_timeout_s
    )
    return single_sample_output(
        cfg,
        input_ids=input_ids,
        input_length=input_length,
        generated_token_ids=generated_token_ids,
        generated_logprobs=generated_logprobs,
        truncated=truncated,
    )
