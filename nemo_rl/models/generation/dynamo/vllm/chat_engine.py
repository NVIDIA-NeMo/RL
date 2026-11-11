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

"""NeMo RL's chat engine inside Dynamo's frontend, for NeMo-Gym's calls.

Dynamo's frontend hands every chat call to this engine (``chat_engine_factory``)
instead of its Rust preprocessor, so NeMo-Gym's calls are rendered, spliced and
parsed by the same code as NeMo RL's native vLLM server:

1. the call's :class:`~nemo_rl.models.generation.gym_call.GymCallExtras` and
   on-policy sampling are checked;
2. a render-pool worker renders it (:mod:`render_pool`);
3. here, the prompt is prepared as natively
   (:func:`~nemo_rl.models.generation.vllm.prompt_and_capture.prepare_prompt`: the
   token-capture prefix read from TQ, the splice and the clamp);
4. the final token ids go through Dynamo's KV router to one engine, whose
   adapter runs the capture and generation
   (``adapter.DynamoVllmAdapter``);
5. a render-pool worker builds the reply with vLLM's own parsers.

The engine publishes its engine args and chat settings at registration
(``EngineConfig.runtime_data``), so this engine renders exactly as it would.
"""

from __future__ import annotations

import asyncio
import base64
import os
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Optional, TypedDict

from dynamo.llm import PythonAsyncEngine
from dynamo.llm.exceptions import HttpError, InvalidArgument

from nemo_rl.models.generation.dynamo.managed_runtime import (
    RAY_ADDRESS_ENV,
    RENDER_WORKERS_ENV,
    read_chat_capture,
)
from nemo_rl.models.generation.dynamo.vllm.render_pool import (
    RenderPool,
    RestoredReply,
)
from nemo_rl.models.generation.gym_call import (
    GYM_RESULT_NVEXT_KEY,
    GymCallExtras,
    GymCallResult,
    GymRequestFields,
)

# The key the engine publishes its chat settings under in runtime_data.
RUNTIME_DATA_KEY = "nemo_rl_chat"
# The annotation the adapter serves NeMo-Gym calls by; the ForwardedCall rides
# in extra_args under PLAN_KEY.
GYM_CALL_ANNOTATION = "nemo_gym:rollout"
PLAN_KEY = "nemo_rl"


class ForwardedCall(TypedDict):
    """What the chat engine sends the adapter in ``extra_args[PLAN_KEY]``.

    The call's ``PreparedPrompt.to_json()`` and what the engine needs that the
    native worker has in process: the rendered request, its media, and what
    vLLM's chat serving hands the engine besides the prompt.
    """

    prepared_prompt: dict[str, Any]
    # The rendered chat request, without messages.
    request: dict[str, Any]
    # The render's multimodal input, pickled then base64-encoded.
    multimodal: Optional[str]
    reasoning_ended: Optional[bool]
    reasoning_parser_kwargs: Optional[dict[str, Any]]


class GymCallRefusal(TypedDict):
    """A call the engine refused before it ran, with the native status."""

    status: int
    message: str
    # Whether Gym is told the capture call was refused (a native 400).
    acknowledge: bool


class GymEngineReply(TypedDict, total=False):
    """The adapter's ``engine_data`` for a Gym call: an output, or a refusal."""

    # A served call: the completion the reply is built from, and its
    # GymCallResult as JSON.
    output: dict[str, Any]
    result: dict[str, Any]
    # Only for a call resumed from a checkpoint cut.
    restored: RestoredReply
    refusal: GymCallRefusal


# Fields of Dynamo's chat request that are not vLLM request fields
# (chat_template_args is renamed to vLLM's chat_template_kwargs).
_DYNAMO_ONLY_FIELDS = {"nvext", "chat_template_args", "unsupported_fields"}


def _capture_refusal_ack(extras: GymCallExtras) -> str:
    """Gym's acknowledgement that a capture call was refused before it ran.

    The native route sends it as a header; Dynamo's frontend forwards only an
    error's status and message, so it ends the message instead.
    """
    if extras.ng_capture is None or extras.ng_measure:
        # Native's /tokenize never acknowledges a measure call.
        return ""
    # Deferred: Gym is installed wherever a call carries ng_capture.
    from nemo_gym.token_id_capture.external_capture import (
        dynamo_capture_refusal_suffix,
    )

    return dynamo_capture_refusal_suffix(extras.ng_capture["model_call_id"])


# Free-form JSON the client wrote (a tool's parameters, a response_format
# schema, processor and media kwargs) keeps its nulls, as natively: a schema's
# `"default": null` is rendered into the prompt.
_CLIENT_JSON_KEYS = frozenset(
    {"parameters", "schema", "mm_processor_kwargs", "media_io_kwargs"}
)


def _drop_none(value: Any) -> Any:
    """Drop the Nones Dynamo adds for unset struct fields, which vLLM rejects.

    An image part's unset ``detail`` arrives as None, for example.
    """
    if isinstance(value, dict):
        return {
            k: v if k in _CLIENT_JSON_KEYS else _drop_none(v)
            for k, v in value.items()
            if v is not None
        }
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


class _CaptureSource:
    """The read-only TQ client the token-capture prefix is read with.

    Built on the first captured call, from what ``setup_token_capture`` wrote
    for the frontend: capture starts after the engines register.
    """

    def __init__(self) -> None:
        self._prefix_cache: Any = None
        # Calls run resolve_prefix on several threads at once; connect once.
        self._connect_lock = threading.Lock()

    def resolve_prefix(self, serving_admission: Any) -> list[int]:
        """The admission's prefix, read from TQ (blocking; run off the loop)."""
        from nemo_rl.data_plane.tq_token_sink import resolve_admission_prefix

        if self._prefix_cache is None:
            with self._connect_lock:
                if self._prefix_cache is None:
                    self._prefix_cache = self._connect()
        return resolve_admission_prefix(serving_admission, self._prefix_cache)

    @staticmethod
    def _connect() -> Any:
        import ray

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache, TQTokenSource

        capture = read_chat_capture()
        if capture is None:
            raise RuntimeError(
                "a token-capture call reached the Dynamo frontend before token "
                "capture was set up for it"
            )
        if not ray.is_initialized():
            ray.init(address=os.environ[RAY_ADDRESS_ENV], log_to_driver=False)
        client = build_data_plane_client(
            capture["data_plane"], bootstrap=False, segment_size=0
        )
        cache = ChainPrefixCache()
        cache.install(
            TQTokenSource(
                client,
                staging_partition=capture["staging_partition"],
                capture_media=capture["capture_media"],
            )
        )
        return cache


class NemoRLChatEngine:
    """Serves NeMo-Gym's chat calls for one model's engines."""

    def __init__(
        self,
        *,
        pool: RenderPool,
        routed_engine: Any,
        spec: dict[str, Any],
        capture: _CaptureSource,
    ):
        self._pool = pool
        self._routed_engine = routed_engine
        self._generation = spec["generation"]
        self._capture = capture

    async def generate(
        self, request: dict[str, Any], context: Any = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Deferred: prompt_and_capture imports vLLM.
        from nemo_rl.models.generation.vllm.prompt_and_capture import error_reply

        extras = GymCallExtras.from_annotations(
            (request.get("nvext") or {}).get("annotations") or ()
        )
        if extras is None:
            raise HttpError(
                400,
                "NeMo RL's Dynamo frontend serves NeMo-Gym's calls (the "
                "nemo_rl:gym: annotation) on chat; NeMo RL's own rollouts use "
                "/v1/completions.",
            )
        try:
            chunk = await self._serve(request, extras, context)
        except HttpError:
            raise
        except Exception as error:
            # The steps before the engine refuse a call with these, as
            # natively; the frontend holds no capture state to fail.
            failure = error_reply(error)
            if failure is None:
                raise
            ack = _capture_refusal_ack(extras) if failure.acknowledge else ""
            raise HttpError(failure.status, failure.message + ack) from error
        yield chunk

    async def _serve(
        self, request: dict[str, Any], extras: GymCallExtras, context: Any
    ) -> dict[str, Any]:
        # Deferred: prompt_and_capture imports vLLM.
        from nemo_rl.models.generation.vllm.prompt_and_capture import prepare_prompt
        from nemo_rl.models.generation.vllm.utils import on_policy_sampling

        body = _drop_none(
            {k: v for k, v in request.items() if k not in _DYNAMO_ONLY_FIELDS}
        )
        if request.get("chat_template_args"):
            body["chat_template_kwargs"] = request["chat_template_args"]
        body.update(
            on_policy_sampling(
                temperature=body.get("temperature"),
                top_p=body.get("top_p"),
                top_k=body.get("top_k"),
                min_p=body.get("min_p"),
                repetition_penalty=body.get("repetition_penalty"),
                presence_penalty=body.get("presence_penalty"),
                frequency_penalty=body.get("frequency_penalty"),
                logit_bias=body.get("logit_bias"),
                n=body.get("n"),
                logprobs=body.get("logprobs"),
                # From the raw request: _drop_none drops an explicit null.
                top_logprobs=request.get("top_logprobs"),
                generation_config=self._generation,
            )
        )
        request_fields: set[str] = set(GymRequestFields.model_fields)
        body.update(extras.model_dump(include=request_fields))
        info = await self._pool.info()
        rendered = await self._pool.render(body, measure=extras.ng_measure)
        plan = await prepare_prompt(
            rendered,
            resolve_prefix=self._capture.resolve_prefix,
            turn_end_token_ids=info["turn_end_token_ids"],
            max_model_len=info["max_model_len"],
            clamp=not extras.ng_measure,
        )
        if extras.ng_measure:
            return self._chunk(
                request,
                GymCallResult(
                    prompt_token_count=len(plan.prompt_token_ids),
                    ng_context_measured=True,
                ),
                message={"role": "assistant", "content": ""},
                finish_reason="stop",
            )
        wire: ForwardedCall = {
            "prepared_prompt": plan.to_json(),
            "request": rendered.request,
            "multimodal": None
            if rendered.multimodal is None
            else base64.b64encode(rendered.multimodal).decode(),
            "reasoning_ended": rendered.reasoning_ended,
            "reasoning_parser_kwargs": rendered.reasoning_parser_kwargs,
        }
        preprocessed = {
            "model": request["model"],
            "token_ids": plan.prompt_token_ids,
            "stop_conditions": {"max_tokens": plan.max_tokens},
            "sampling_options": {
                "temperature": body["temperature"],
                "top_p": body["top_p"],
            },
            "output_options": {},
            "eos_token_ids": [],
            "annotations": [GYM_CALL_ANNOTATION],
            "extra_args": {PLAN_KEY: wire},
        }
        engine_data = await self._call_engine(preprocessed, context)
        if (refusal := engine_data.get("refusal")) is not None:
            # The engine refused the call before it ran, with the native status.
            ack = _capture_refusal_ack(extras) if refusal["acknowledge"] else ""
            raise HttpError(refusal["status"], refusal["message"] + ack)
        result = GymCallResult.model_validate(engine_data["result"])
        reply_request = rendered.request
        if result.ng_commit_coords is not None:
            # A captured call's logprobs were staged with it; as natively, its
            # reply carries none.
            reply_request = {**reply_request, "logprobs": False, "top_logprobs": None}
        response = await self._pool.respond(
            reply_request,
            plan.prompt_token_ids,
            engine_data["output"],
            engine_data.get("restored"),
        )
        (choice,) = response["choices"]
        return self._chunk(
            request,
            result,
            message=choice["message"],
            finish_reason=choice["finish_reason"],
            logprobs=choice.get("logprobs"),
            usage=response.get("usage"),
        )

    async def _call_engine(
        self, preprocessed: dict[str, Any], context: Any
    ) -> GymEngineReply:
        engine_data: Optional[GymEngineReply] = None
        stream = await self._routed_engine.generate(preprocessed, context=context)
        try:
            async for item in stream:
                data = item.data()
                if data and data.get("engine_data"):
                    engine_data = data["engine_data"]
        except (ValueError, InvalidArgument) as error:
            # Dynamo raises an engine's error as a ValueError, or from 1.7 the
            # engine adapter's InvalidArgument as itself (not a ValueError), and
            # would answer either with 400. The engine sends every status it means
            # as a refusal, so anything raised is an engine failure: 500, as natively.
            raise HttpError(500, str(error)) from error
        if engine_data is None:
            raise HttpError(500, "the engine returned no result for this call")
        return engine_data

    @staticmethod
    def _chunk(
        request: dict[str, Any],
        result: GymCallResult,
        *,
        message: dict[str, Any],
        finish_reason: str,
        logprobs: Optional[dict[str, Any]] = None,
        usage: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """The OpenAI chunk Dynamo's frontend aggregates into the reply."""
        delta: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content"),
        }
        # vLLM names it "reasoning"; Dynamo's OpenAI types "reasoning_content".
        reasoning = message.get("reasoning", message.get("reasoning_content"))
        if reasoning is not None:
            delta["reasoning_content"] = reasoning
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {**tool_call, "index": index}
                for index, tool_call in enumerate(message["tool_calls"])
            ]
        chunk: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "logprobs": logprobs,
                }
            ],
            "nvext": {GYM_RESULT_NVEXT_KEY: result.model_dump(exclude_none=True)},
        }
        if usage is not None:
            chunk["usage"] = usage
        return chunk


class ChatEngineFactory:
    """Dynamo's ``chat_engine_factory``: one chat engine per model's engines."""

    def __init__(self) -> None:
        self._capture = _CaptureSource()

    async def chat_engine_factory(
        self, instance_id: Any, mdc: Any, routed_engine: Any
    ) -> PythonAsyncEngine:
        del instance_id
        runtime_data = (mdc.runtime_config() or {}).get("runtime_data") or {}
        spec = runtime_data.get(RUNTIME_DATA_KEY)
        if spec is None:
            raise RuntimeError(
                f"model {mdc.name()!r} registered without NeMo RL's chat settings "
                f"(runtime_data[{RUNTIME_DATA_KEY!r}]); only NeMo RL's engines are "
                "served by this frontend"
            )
        pool = RenderPool(
            spec, mdc.local_dir(), workers=int(os.environ[RENDER_WORKERS_ENV])
        )
        engine = NemoRLChatEngine(
            pool=pool, routed_engine=routed_engine, spec=spec, capture=self._capture
        )
        return PythonAsyncEngine(engine.generate, asyncio.get_running_loop())
