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
3. here, the token-capture prefix is read from TQ and spliced in, and the
   output budget is clamped to the context;
4. the final token ids go through Dynamo's KV router to one engine, whose
   adapter runs the capture and generation (``adapter.DynamoVllmAdapter``);
5. a render-pool worker builds the reply with vLLM's own parsers.

The engine publishes its engine args and chat settings at registration
(``EngineConfig.runtime_data``), so this engine renders exactly as it would.
"""

from __future__ import annotations

import asyncio
import base64
import os
import pickle
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Optional, TypedDict

from dynamo.llm import PythonAsyncEngine
from dynamo.llm.exceptions import HttpError

from nemo_rl.models.generation.dynamo.managed_runtime import (
    CAPTURE_FILE_ENV,
    RAY_ADDRESS_ENV,
    RENDER_WORKERS_ENV,
)
from nemo_rl.models.generation.dynamo.vllm.render_pool import (
    RenderedCall,
    RenderPool,
)
from nemo_rl.models.generation.gym_call import (
    GYM_RESULT_NVEXT_KEY,
    GymCallExtras,
    GymCallResult,
    GymRequestFields,
)
from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens

# The key the engine publishes its chat settings under in runtime_data.
RUNTIME_DATA_KEY = "nemo_rl_chat"
# The annotation the adapter serves NeMo-Gym calls by; the capture plan rides
# in extra_args under PLAN_KEY.
GYM_CALL_ANNOTATION = "nemo_gym:rollout"
PLAN_KEY = "nemo_rl"


class GymCallPlan(TypedDict):
    """What the chat engine sends the adapter in ``extra_args[PLAN_KEY]``."""

    # The rendered chat request, without messages.
    request: dict[str, Any]
    # Capture admissions as JSON: what is stored, and how the prompt is served.
    admission: Optional[dict[str, Any]]
    serving_admission: Optional[dict[str, Any]]
    verify_retained_media: bool
    # The model's own prefix length at the start of the spliced prompt.
    prefix_len: Optional[int]
    template_token_ids: list[int]
    # PrefixSplice's (model_cut_end, template_cut_start), when spliced.
    splice: Optional[list[int]]
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

    def resolve_prefix(self, serving_admission: dict[str, Any]) -> list[int]:
        """The admission's prefix, read from TQ (blocking; run off the loop)."""
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        from nemo_rl.data_plane.tq_token_sink import resolve_admission_prefix

        if self._prefix_cache is None:
            with self._connect_lock:
                if self._prefix_cache is None:
                    self._prefix_cache = self._connect()
        return resolve_admission_prefix(
            CaptureAdmission.model_validate(serving_admission), self._prefix_cache
        )

    @staticmethod
    def _connect() -> Any:
        import ray

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache, TQTokenSource

        path = os.environ[CAPTURE_FILE_ENV]
        if not os.path.exists(path):
            raise RuntimeError(
                "a token-capture call reached the Dynamo frontend before token "
                "capture was set up for it"
            )
        with open(path, "rb") as capture_file:
            capture = pickle.load(capture_file)
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
        from vllm.exceptions import VLLMValidationError

        from nemo_rl.data.captured_media import MediaCaptureRejected
        from nemo_rl.models.generation.vllm.utils import OffPolicySamplingError

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
            yield await self._serve(request, extras, context)
        except (OffPolicySamplingError, MediaCaptureRejected) as error:
            raise HttpError(400, str(error)) from error
        except VLLMValidationError as error:
            # Refused before the engine ran, so nothing was staged: as natively.
            raise HttpError(400, str(error) + _capture_refusal_ack(extras)) from error

    async def _serve(
        self, request: dict[str, Any], extras: GymCallExtras, context: Any
    ) -> dict[str, Any]:
        # Deferred: chat_server imports vLLM's OpenAI server stack.
        from nemo_rl.models.generation.vllm.chat_server import clamp_max_tokens
        from nemo_rl.models.generation.vllm.utils import on_policy_sampling

        body = _drop_none(
            {k: v for k, v in request.items() if k not in _DYNAMO_ONLY_FIELDS}
        )
        if request.get("chat_template_args"):
            body["chat_template_kwargs"] = request["chat_template_args"]
        if (body.get("n") or 1) != 1:
            raise HttpError(400, "NeMo-Gym calls take n=1.")
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
                generation_config=self._generation,
            )
        )
        request_fields: set[str] = set(GymRequestFields.model_fields)
        body.update(extras.model_dump(include=request_fields))
        info = await self._pool.info()
        rendered = await self._pool.render(body, measure=extras.ng_measure)
        prompt_token_ids, splice, prefix = await self._splice(
            rendered, info["turn_end_token_ids"]
        )
        if extras.ng_measure:
            return self._chunk(
                request,
                GymCallResult(
                    prompt_token_count=len(prompt_token_ids), ng_context_measured=True
                ),
                message={"role": "assistant", "content": ""},
                finish_reason="stop",
            )
        max_tokens = clamp_max_tokens(
            rendered.request_max_tokens, len(prompt_token_ids), info["max_model_len"]
        )
        engine_request = dict(rendered.request)
        if max_tokens is not None:
            engine_request["max_tokens"] = max_tokens
            engine_request["max_completion_tokens"] = None
        plan: GymCallPlan = {
            "request": engine_request,
            "admission": rendered.admission,
            "serving_admission": rendered.serving_admission,
            "verify_retained_media": rendered.verify_retained_media,
            # The model's own prefix is the spliced prompt's first ids.
            "prefix_len": None if prefix is None else len(prefix),
            "template_token_ids": rendered.template_token_ids,
            "splice": None
            if splice is None
            else [splice.model_cut_end, splice.template_cut_start],
            "multimodal": None
            if rendered.multimodal is None
            else base64.b64encode(rendered.multimodal).decode(),
            "reasoning_ended": rendered.reasoning_ended,
            "reasoning_parser_kwargs": rendered.reasoning_parser_kwargs,
        }
        preprocessed = {
            "model": request["model"],
            "token_ids": prompt_token_ids,
            "stop_conditions": {"max_tokens": max_tokens},
            "sampling_options": {
                "temperature": body["temperature"],
                "top_p": body["top_p"],
            },
            "output_options": {},
            "eos_token_ids": [],
            "annotations": [GYM_CALL_ANNOTATION],
            "extra_args": {PLAN_KEY: plan},
        }
        engine_data = await self._call_engine(preprocessed, context)
        if (refusal := engine_data.get("refusal")) is not None:
            # The engine refused the call before it ran, with the native status.
            ack = _capture_refusal_ack(extras) if refusal["acknowledge"] else ""
            raise HttpError(refusal["status"], refusal["message"] + ack)
        result = GymCallResult.model_validate(engine_data["result"])
        reply_request = engine_request
        if result.ng_commit_coords is not None:
            # A captured call's logprobs were staged with it; as natively, its
            # reply carries none.
            reply_request = {**engine_request, "logprobs": False, "top_logprobs": None}
        response = await self._pool.respond(
            reply_request, prompt_token_ids, engine_data["output"]
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

    async def _splice(
        self, rendered: RenderedCall, turn_end_token_ids: list[int]
    ) -> tuple[list[int], Any, Any]:
        """The prompt the engine runs: the model's own prefix, then the template's."""
        prefix: Optional[list[int]] = None
        serving = rendered.serving_admission
        if serving is not None and serving["mode"] == "token_in":
            # A blocking data-plane read; keep it off the frontend's loop.
            prefix = await asyncio.to_thread(self._capture.resolve_prefix, serving)
        elif rendered.required_prefix_token_ids is not None:
            prefix = rendered.required_prefix_token_ids
        if prefix is None:
            return rendered.template_token_ids, None, None
        # The render includes the history whenever a prefix will be spliced.
        assert rendered.history_token_ids is not None
        splice = splice_prefix_tokens(
            tokenizer=None,
            model_prefix_token_ids=prefix,
            template_prefix_token_ids=rendered.history_token_ids,
            template_token_ids=rendered.template_token_ids,
            eos_token_id=turn_end_token_ids,
        )
        return splice.token_ids, splice, prefix

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
        except ValueError as error:
            # Dynamo raises an engine's error as a ValueError, which it would
            # answer with 400. The engine sends every status it means as a
            # refusal, so anything raised is an engine failure: 500, as natively.
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
