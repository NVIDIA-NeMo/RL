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
"""NeMo-Gym's OpenAI server for Megatron engines behind the Dynamo frontend.

Without Dynamo, a Megatron engine serves NeMo-Gym from Megatron-LM's own
text-generation server on its rank-zero worker: it renders the chat template,
runs the reasoning and tool parsers, splices an exact prior-turn prefix, and
hands token ids to the engine through an ``InferenceClient``. Behind Dynamo the
engines are reached through the frontend only, so that server cannot call the
coordinator. This module keeps everything above the client and replaces the
client: :class:`FrontendInferenceClient` posts each token-id prompt to the
frontend's ``/v1/completions`` and shapes the answer like the coordinator's
reply, so Megatron-LM's endpoints serve unchanged. The frontend routes the
request over the prefill and decode engines; nothing here knows about the
split.

Token capture is not served here yet: a capture admission travels to the
engine as request metadata, which the completion request does not carry
across the frontend, so such requests are refused before they are sent.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import torch

from nemo_rl.models.generation.megatron import dynamo_rollouts
from nemo_rl.models.generation.megatron.dynamo_rollouts import (
    async_http_post_json,
    format_error,
    is_retryable,
)

_ENDPOINTS_MODULE = (
    "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints"
)

if TYPE_CHECKING:
    from concurrent.futures import Future as ConcurrentFuture

    from megatron.core.inference.async_stream import AsyncStream
    from megatron.core.inference.sampling_params import SamplingParams

LOGGER = logging.getLogger(__name__)

# HTTP statuses the frontend answers a request it will never serve with; the
# endpoint maps them to a 400, as it does an engine's non-transient failure.
_NONTRANSIENT_HTTP_STATUS_CODES = range(400, 500)


def completion_request(
    prompt_token_ids: list[int],
    sampling_params: "SamplingParams",
    *,
    model_name: str,
) -> dict[str, Any]:
    """The frontend completion request for one of the endpoint's engine submissions.

    Megatron's ``SamplingParams`` disable ``top_p`` with ``0.0`` and ``top_k``
    with ``0``; the frontend takes ``1.0`` and ``-1`` for the same. Only the
    sampled tokens' log probabilities can come back (``logprobs: 0``), as the
    Megatron Dynamo adapter serves no top-N or prompt log probabilities.

    Raises:
        ValueError: For sampling the adapter cannot serve, or no token budget.
    """
    if sampling_params.top_n_logprobs:
        raise ValueError(
            "top_logprobs is not supported for Megatron engines behind Dynamo: the "
            "frontend returns the sampled tokens' log probabilities only"
        )
    max_tokens = sampling_params.num_tokens_to_generate
    if max_tokens is None and sampling_params.num_tokens_total is not None:
        max_tokens = sampling_params.num_tokens_total - len(prompt_token_ids)
    if max_tokens is None or max_tokens < 1:
        raise ValueError(
            "a request through the Dynamo frontend needs a positive token budget "
            f"(num_tokens_to_generate / num_tokens_total); got {max_tokens!r}"
        )
    top_p = float(sampling_params.top_p)
    top_k = int(sampling_params.top_k)
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": list(prompt_token_ids),
        "max_tokens": int(max_tokens),
        "temperature": float(sampling_params.temperature),
        "top_p": 1.0 if top_p <= 0.0 else top_p,
        "top_k": -1 if top_k <= 0 else top_k,
        "n": 1,
        "include_stop_str_in_output": bool(sampling_params.detokenize_stop_sequence),
        "nvext": {"extra_fields": ["completion_token_ids"]},
    }
    if sampling_params.return_log_probs:
        payload["logprobs"] = 0
    if sampling_params.stop_words:
        payload["stop"] = list(sampling_params.stop_words)
    stop_token_ids = list(getattr(sampling_params, "stop_token_ids", None) or [])
    if sampling_params.termination_id is not None:
        stop_token_ids.append(int(sampling_params.termination_id))
    if stop_token_ids:
        payload["stop_token_ids"] = sorted({int(t) for t in stop_token_ids})
    seed = getattr(sampling_params, "seed", None)
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def parse_completion(
    response: Any, *, want_log_probs: bool, request_url: str
) -> tuple[list[int], Optional[list[float]]]:
    """The generated token ids and, when asked for, their log probabilities.

    Like the rollout client's parser, but log probabilities are required only
    when the request asked for them; a NeMo-Gym call without ``logprobs`` gets
    none back from the frontend.

    Raises:
        RuntimeError: A reply without the fields the request needs.
    """
    if not isinstance(response, dict):
        raise RuntimeError(
            f"completion response from {request_url} was not a JSON object"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(
            f"completion response from {request_url} did not include choices"
        )
    nvext = response.get("nvext")
    token_ids = nvext.get("completion_token_ids") if isinstance(nvext, dict) else None
    if not isinstance(token_ids, list):
        raise RuntimeError(
            f"completion response from {request_url} did not include "
            "nvext.completion_token_ids"
        )
    generated = [int(token_id) for token_id in token_ids]
    if not want_log_probs:
        return generated, None
    logprobs = choices[0].get("logprobs")
    token_logprobs = (
        logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
    )
    if not isinstance(token_logprobs, list) or len(token_logprobs) != len(generated):
        raise RuntimeError(
            f"completion response from {request_url} did not include one "
            f"choice.logprobs.token_logprobs entry per generated token"
        )
    return generated, [float(logprob) for logprob in token_logprobs]


def completion_record(
    prompt_token_ids: list[int],
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    request_id: int,
    request_url: str,
    latency_s: float,
) -> dict[str, Any]:
    """The coordinator-shaped reply the endpoints read, from a frontend completion.

    The keys are those of a serialized ``DynamicInferenceRequest`` that the
    chat and completions endpoints consume: generated tokens and their log
    probabilities, the prompt, usage counts and the sampling budget the finish
    reason is derived from. Fields the frontend has no counterpart for (top-N
    log probabilities, routing indices, staged-payload metadata) are None.
    """
    generated, log_probs = parse_completion(
        response, want_log_probs="logprobs" in payload, request_url=request_url
    )
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "uid": str(response.get("id") or request_id),
        "request_id": request_id,
        "status": "COMPLETED",
        "events": [],
        "prompt_tokens": list(prompt_token_ids),
        "compact_prompt_tokens": None,
        "prompt_length": len(prompt_token_ids),
        "generated_tokens": generated,
        "generated_log_probs": log_probs,
        "generated_top_n_logprobs": None,
        "num_cached_tokens": int(details.get("cached_tokens") or 0),
        "payload_offloaded": False,
        "payload_stage_metadata": None,
        "routing_indices": None,
        "sampling_params": {"num_tokens_to_generate": payload["max_tokens"]},
        "latency": latency_s,
    }


def failed_record(
    prompt_token_ids: list[int],
    message: str,
    *,
    request_id: int,
    transient: bool,
) -> dict[str, Any]:
    """A failed reply, as the endpoints read one the engine rejected or dropped.

    A non-transient failure becomes the caller's 400, a transient one its 500.
    """
    return {
        "uid": str(request_id),
        "request_id": request_id,
        "status": "FAILED",
        "events": [
            {
                "type": "ERROR_TRANSIENT" if transient else "ERROR_NONTRANSIENT",
                "payload": message,
            }
        ],
        "prompt_tokens": list(prompt_token_ids),
        "prompt_length": len(prompt_token_ids),
        "generated_tokens": [],
        "generated_log_probs": None,
        "generated_top_n_logprobs": None,
        "num_cached_tokens": 0,
        "payload_offloaded": False,
        "payload_stage_metadata": None,
        "routing_indices": None,
        "sampling_params": {"num_tokens_to_generate": 0},
    }


class FrontendInferenceClient:
    """An ``InferenceClient`` for Megatron-LM's endpoints that talks to the Dynamo frontend.

    Implements the submission surface the endpoints use (``add_request``,
    ``add_request_with_id``, ``add_request_streaming``, ``abort_request``,
    ``start``, ``stop``). Each submission is one HTTP completion on the running
    event loop; the reply is reshaped by :func:`completion_record`, a failure by
    :func:`failed_record`, so the endpoints' own error handling applies.
    Streaming submissions deliver the whole completion as the one final item.

    Refused up front, as the frontend path cannot carry them: string prompts
    (the endpoints always tokenize), multimodal data, and request metadata (a
    token-capture admission).
    """

    def __init__(
        self,
        *,
        frontend_url: str,
        model_name: str,
        request_timeout_s: Optional[float],
    ) -> None:
        self._completions_url = f"{frontend_url.rstrip('/')}/completions"
        self._model_name = model_name
        self._request_timeout_s = request_timeout_s
        self._tasks: dict[int, asyncio.Task] = {}
        self.next_request_id = 0

    @property
    def completions_url(self) -> str:
        return self._completions_url

    def start(self) -> None:
        """Nothing to connect: every request is its own HTTP call."""

    def stop(self) -> None:
        """Cancel the requests still in flight."""
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()

    def add_request(
        self,
        prompt: Any,
        sampling_params: "SamplingParams",
        *,
        multi_modal_data: Any = None,
        request_metadata: Optional[dict] = None,
    ) -> asyncio.Future:
        return self.add_request_with_id(
            prompt,
            sampling_params,
            multi_modal_data=multi_modal_data,
            request_metadata=request_metadata,
        )[1]

    def add_request_with_id(
        self,
        prompt: Any,
        sampling_params: "SamplingParams",
        *,
        multi_modal_data: Any = None,
        request_metadata: Optional[dict] = None,
    ) -> tuple[int, asyncio.Future]:
        request_id, payload, prompt_token_ids = self._prepare(
            prompt,
            sampling_params,
            multi_modal_data=multi_modal_data,
            request_metadata=request_metadata,
        )
        task = asyncio.get_running_loop().create_task(
            self._complete(request_id, prompt_token_ids, payload)
        )
        self._track(request_id, task)
        return request_id, task

    def add_request_streaming(
        self,
        prompt: Any,
        sampling_params: "SamplingParams",
        *,
        multi_modal_data: Any = None,
        request_metadata: Optional[dict] = None,
    ) -> "AsyncStream[dict]":
        # Deferred: the stream class is Megatron-LM's.
        from megatron.core.inference.async_stream import AsyncStream

        request_id, payload, prompt_token_ids = self._prepare(
            prompt,
            sampling_params,
            multi_modal_data=multi_modal_data,
            request_metadata=request_metadata,
        )
        loop = asyncio.get_running_loop()
        stream: AsyncStream[dict] = AsyncStream(
            request_id, lambda: self.abort_request(request_id), loop=loop
        )

        async def _deliver() -> None:
            try:
                record = await self._complete(request_id, prompt_token_ids, payload)
            except asyncio.CancelledError:
                stream.finish(asyncio.CancelledError())
                raise
            stream.put({"final": record})
            stream.finish()

        self._track(request_id, loop.create_task(_deliver()))
        return stream

    def abort_request(self, request_id: int) -> None:
        """Drop a request still in flight; a finished or unknown id is not an error."""
        task = self._tasks.pop(request_id, None)
        if task is not None:
            task.cancel()

    def _prepare(
        self,
        prompt: Any,
        sampling_params: "SamplingParams",
        *,
        multi_modal_data: Any,
        request_metadata: Optional[dict],
    ) -> tuple[int, dict[str, Any], list[int]]:
        if request_metadata:
            raise NotImplementedError(
                "token capture (the ng_capture admission) is not yet served for "
                "Megatron engines behind Dynamo: request metadata does not travel "
                "through the frontend. Set token_capture.enabled=false."
            )
        if multi_modal_data:
            raise NotImplementedError(
                "Megatron engines behind Dynamo serve text prompts only; the "
                "request carried multimodal data"
            )
        if isinstance(prompt, torch.Tensor):
            prompt = prompt.tolist()
        if isinstance(prompt, str) or not isinstance(prompt, list):
            raise ValueError(
                "the frontend takes token ids; the endpoint must tokenize the prompt "
                f"first (got {type(prompt).__name__})"
            )
        prompt_token_ids = [int(token_id) for token_id in prompt]
        payload = completion_request(
            prompt_token_ids, sampling_params, model_name=self._model_name
        )
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id, payload, prompt_token_ids

    def _track(self, request_id: int, task: asyncio.Task) -> None:
        self._tasks[request_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(request_id, None))

    async def _complete(
        self, request_id: int, prompt_token_ids: list[int], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """One completion through the frontend, retried on transient failures."""
        started = time.perf_counter()
        response: dict[str, Any] = {}
        for attempt in range(1, dynamo_rollouts._HTTP_MAX_ATTEMPTS + 1):
            response = await async_http_post_json(
                self._completions_url, payload, self._request_timeout_s
            )
            if (
                not is_retryable(response)
                or attempt == dynamo_rollouts._HTTP_MAX_ATTEMPTS
            ):
                break
            LOGGER.warning(
                "Gym completion attempt %d/%d through the frontend failed; retrying in %.1fs: %s",
                attempt,
                dynamo_rollouts._HTTP_MAX_ATTEMPTS,
                dynamo_rollouts._HTTP_RETRY_DELAY_S,
                format_error(response),
            )
            await asyncio.sleep(dynamo_rollouts._HTTP_RETRY_DELAY_S)
        if isinstance(response, dict) and response.get("status") == "error":
            status = response.get("http_status")
            return failed_record(
                prompt_token_ids,
                format_error(response),
                request_id=request_id,
                transient=status not in _NONTRANSIENT_HTTP_STATUS_CODES,
            )
        try:
            return completion_record(
                prompt_token_ids,
                payload,
                response,
                request_id=request_id,
                request_url=self._completions_url,
                latency_s=time.perf_counter() - started,
            )
        except RuntimeError as error:
            # A reply the frontend produced but this client cannot read: the
            # engine did not fail, so retrying the call would not help.
            return failed_record(
                prompt_token_ids, str(error), request_id=request_id, transient=False
            )


def build_gym_proxy_app(
    client: FrontendInferenceClient,
    *,
    tokenizer: Any,
    parsers: Optional[list[str]],
    multimodal_prompt_config: Any,
    default_temperature: float,
    default_top_p: float,
    default_top_k: int,
) -> Any:
    """Megatron-LM's OpenAI endpoints as a Quart app over ``client``.

    Assembled as Megatron-LM's own server assembles them, so the chat and
    completions endpoints behave as on the native path; only the client under
    them differs.
    """
    # Deferred: Quart and the endpoints are Megatron-LM's server dependencies.
    from quart import Quart  # type: ignore[import-error]

    endpoints = importlib.import_module(_ENDPOINTS_MODULE)

    app = Quart("nemo_rl.megatron_dynamo_gym_proxy")
    # Large prompts; the same limit Megatron-LM's server sets.
    app.config["MAX_CONTENT_LENGTH"] = 2**30
    app.config["client"] = client
    app.config["tokenizer"] = tokenizer
    app.config["parsers"] = parsers
    app.config["verbose"] = False
    app.config["chat_template"] = None
    app.config["multimodal_prompt_config"] = multimodal_prompt_config
    app.config["default_temperature"] = default_temperature
    app.config["default_top_p"] = default_top_p
    app.config["default_top_k"] = default_top_k
    app.config["eval_mode"] = False
    # Template rendering is synchronous and O(prompt): off the loop, one worker.
    # HF tokenizers are not thread-safe, so the worker gets its own copy.
    app.config["tokenize_executor"] = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="gym-proxy-tokenize"
    )
    app.config["tokenizer_copy"] = copy.deepcopy(tokenizer)
    for blueprint in endpoints.__all__:
        app.register_blueprint(blueprint)
    return app


def bind_server_socket(port: int) -> socket.socket:
    """A listening socket on ``port`` that shares it with the driver's reservation.

    The driver's port holder binds the port with ``SO_REUSEPORT`` and keeps it
    until this process adopts it, so the port is never free in between. The
    server binds its own socket into the same reuse group, as Megatron-LM's
    replicas do, and the holder's socket is released after the handoff. ``0``
    picks a free port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(2**14)
    sock.setblocking(False)
    return sock


async def serve_app(app: Any, sock: socket.socket) -> None:
    """Serve ``app`` on ``sock`` until cancelled; the socket is closed on exit."""
    # Deferred: hypercorn is Megatron-LM's server dependency.
    from hypercorn.asyncio import serve  # type: ignore[import-error]
    from hypercorn.config import Config  # type: ignore[import-error]

    config = Config()
    config.keep_alive_timeout = 30.0
    config.backlog = 2**14
    config.h2_max_concurrent_streams = 2**14
    config.bind = [f"fd://{sock.fileno()}"]
    try:
        await serve(app, config)
    finally:
        sock.close()


def wait_until_healthy(
    base_url: str, *, server: "ConcurrentFuture", timeout_s: float
) -> None:
    """Block until ``base_url`` answers its health check, or the server task ends.

    Raises:
        TimeoutError: No healthy answer within ``timeout_s``.
        RuntimeError: The server task finished (or failed) before serving.
    """
    # Deferred: requests is only needed for this synchronous probe.
    import requests

    deadline = time.monotonic() + timeout_s
    with requests.Session() as session:
        while True:
            if server.done():
                message = "the NeMo-Gym proxy server stopped before serving"
                error = server.exception()
                if error is None:
                    raise RuntimeError(message)
                raise RuntimeError(message) from error
            try:
                if session.get(f"{base_url}/health", timeout=10).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"the NeMo-Gym proxy server at {base_url} did not become "
                    f"healthy within {timeout_s:.0f}s"
                )
            time.sleep(0.5)
