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

"""NeMo-Gym's chat calls on a Megatron engine behind Dynamo.

:class:`NemoRLMegatronLLMEngine` is Megatron-LM's Dynamo adapter
(``MegatronLLMEngine``) with NeMo RL's Gym call added, as
``vllm.adapter.DynamoVllmAdapter`` adds it to the vLLM engines. The frontend's
chat engine renders and prepares each call, and sends its final token ids
with the call's ``ForwardedCall`` in ``extra_args`` and the ``nemo_gym:rollout``
annotation. This adapter then:

1. on a prefill engine, runs Megatron-LM's prefill leg unchanged: it generates
   nothing, and the call's tokens come from the decode engine the frontend
   sends it to next, with the same annotations and ``extra_args``;
2. on a decode engine, builds the engine's sampling from the rendered request,
   as Megatron-LM's chat endpoint does (:func:`gym_sampling_request`);
3. runs Megatron-LM's decode leg, keeps the engine's final
   record (``_stream_chunks``), stages a captured call in this process (the
   engine's rank zero, which holds its token-capture stager), and answers
   with one chunk whose ``engine_data`` carries the output the frontend builds
   the reply from and the call's
   :class:`~nemo_rl.models.generation.gym_call.GymCallResult`.

Everything else (NeMo RL's rollouts, probes, the KV handoff and its
recovery) is Megatron-LM's own.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
from collections.abc import AsyncGenerator
from typing import Any, Optional, Protocol

from dynamo.llm.exceptions import InvalidArgument
from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine

from nemo_rl.models.generation.dynamo.chat_engine import (
    CALL_KEY,
    GYM_CALL_ANNOTATION,
    RUNTIME_DATA_KEY,
    ForwardedCall,
    GymEngineReply,
)
from nemo_rl.models.generation.gym_call import GymCallResult
from nemo_rl.models.generation.gym_prompt import PreparedPrompt

# The engine's final record of the Gym call this task is serving; set around
# the decode leg so _stream_chunks can hand it over.
_FINAL_RECORD: contextvars.ContextVar[Optional[dict[str, Any]]] = (
    contextvars.ContextVar("nemo_rl_megatron_gym_final_record", default=None)
)


class GymCallHost(Protocol):
    """What the adapter needs from the Megatron worker it runs in (rank zero)."""

    def dynamo_chat_spec(self) -> dict[str, Any]:
        """The chat settings the frontend renders with (``megatron_chat_spec``)."""
        ...

    def stage_dynamo_gym_call(
        self,
        admission: dict[str, Any],
        *,
        prompt_token_ids: list[int],
        generated_token_ids: list[int],
        generated_log_probs: list[float],
        final_record: dict[str, Any],
    ) -> dict[str, Any]:
        """Stage a captured call in one TQ write; returns its commit coordinates."""
        ...


def gym_sampling_request(
    request: dict[str, Any],
    wire: ForwardedCall,
    *,
    max_tokens: Optional[int],
    captured: bool,
) -> dict[str, Any]:
    """``request`` with the sampling Megatron-LM's chat endpoint builds for the call.

    Megatron-LM's adapter reads only ``sampling_options``, ``stop_conditions``
    and ``output_options``; the frontend's chat engine fills them for its
    router alone, so they are rebuilt here from the rendered chat request, as
    the native endpoint builds its ``SamplingParams``. A captured call always
    returns its sampled tokens' log probabilities, which it stages.
    """
    chat = wire["request"]
    stop = chat.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    return {
        **request,
        "sampling_options": {
            "temperature": chat.get("temperature"),
            "top_p": chat.get("top_p"),
            "top_k": chat.get("top_k"),
            "seed": chat.get("seed"),
            "n": 1,
            "include_stop_str_in_output": bool(chat.get("include_stop_str_in_output")),
        },
        "stop_conditions": {
            "max_tokens": max_tokens,
            "stop": stop,
            "ignore_eos": bool(chat.get("ignore_eos")),
        },
        "output_options": ({"logprobs": 0} if captured or chat.get("logprobs") else {}),
    }


class _FinalTee:
    """An engine stream that also records its final record in ``sink``."""

    def __init__(self, stream: Any, sink: dict[str, Any]) -> None:
        self._stream = stream
        self._sink = sink
        self.request_id = stream.request_id

    def __aiter__(self) -> "_FinalTee":
        return self

    async def __anext__(self) -> dict[str, Any]:
        item = await self._stream.__anext__()
        if "final" in item:
            self._sink.update(item["final"])
        return item

    async def aclose(self) -> None:
        await self._stream.aclose()


class NemoRLMegatronLLMEngine(MegatronLLMEngine):
    """Megatron-LM's Dynamo adapter, serving NeMo-Gym's chat calls too."""

    def __init__(self, config: Any, registration_model: str, *, host: GymCallHost):
        super().__init__(config, registration_model)
        self._host = host

    async def start(self, worker_id: int) -> Any:
        """Register as Megatron-LM does, with the chat settings the frontend renders with."""
        engine_config = await super().start(worker_id)
        # DRAFT: assumes Dynamo's EngineConfig is a dataclass, as the
        # keyword construction in Megatron-LM's adapter suggests.
        return dataclasses.replace(
            engine_config,
            runtime_data={
                **(engine_config.runtime_data or {}),
                RUNTIME_DATA_KEY: self._host.dynamo_chat_spec(),
            },
        )

    async def generate(
        self, request: dict[str, Any], context: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self.config.role == "prefill" or GYM_CALL_ANNOTATION not in (
            request.get("annotations") or ()
        ):
            # NeMo RL's rollouts, and every prefill leg: the prefill engine only
            # fills the KV cache the decode engine pulls, and generates nothing.
            # DRAFT: assumes Dynamo's prefill router hands the decode leg the
            # request's annotations and extra_args as the frontend sent them.
            async for chunk in super().generate(request, context):
                yield chunk
            return
        wire: Optional[ForwardedCall] = (request.get("extra_args") or {}).get(CALL_KEY)
        if wire is None:
            raise InvalidArgument(
                "NeMo-Gym calls reach NeMo RL's engines through NeMo RL's chat "
                f"engine in the Dynamo frontend (extra_args.{CALL_KEY})."
            )
        plan = PreparedPrompt.from_json(
            wire["prepared_prompt"],
            prompt_token_ids=list(request["token_ids"]),
            # Kept as JSON: the stager validates it where it stages.
            parse_admission=lambda admission: admission,
        )
        request = gym_sampling_request(
            request,
            wire,
            max_tokens=plan.max_tokens,
            captured=plan.admission is not None,
        )
        try:
            chunk = await self._serve_gym_call(request, wire, plan, context)
        except ValueError as error:
            # Megatron-LM's InferenceRequestRejected and its sampling checks:
            # the native endpoint answers these with 400.
            chunk = _refusal(400, str(error))
        if chunk is not None:
            yield chunk

    async def _serve_gym_call(
        self,
        request: dict[str, Any],
        wire: ForwardedCall,
        plan: PreparedPrompt,
        context: Any,
    ) -> Optional[dict[str, Any]]:
        """Run the decode leg, then stage or answer the call in one chunk."""
        final: dict[str, Any] = {}
        token_ids: list[int] = []
        log_probs: list[float] = []
        last: dict[str, Any] = {}
        reset = _FINAL_RECORD.set(final)
        try:
            async for chunk in super().generate(request, context):
                token_ids += chunk.get("token_ids") or []
                log_probs += chunk.get("log_probs") or []
                last = chunk
        finally:
            _FINAL_RECORD.reset(reset)
        finish_reason = last.get("finish_reason")
        if finish_reason == "cancelled":
            # Nothing was staged: capture begins and completes in one write.
            return None
        prompt_token_ids = plan.prompt_token_ids
        output: dict[str, Any] = {
            "token_ids": token_ids,
            "finish_reason": finish_reason,
        }
        if plan.admission is not None:
            admission = dict(plan.admission)
            if admission["mode"] == "token_in":
                # Gym verifies the engine's prompt against the model's own prefix,
                # as the native Megatron prompt preparer sets it.
                admission["required_prefix_token_ids"] = prompt_token_ids[
                    : plan.prefix_len
                ]
            coords = await asyncio.to_thread(
                self._host.stage_dynamo_gym_call,
                admission,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=token_ids,
                generated_log_probs=log_probs,
                final_record=final,
            )
            result = GymCallResult(ng_commit_coords=coords)
        else:
            if wire["request"].get("logprobs"):
                output["log_probs"] = log_probs
            # DRAFT: routing_indices for router replay would come from the final
            # record; router replay stays refused behind Dynamo
            # (MegatronDynamoGeneration.validate_settings).
            result = GymCallResult(
                prompt_token_ids=prompt_token_ids,
                generation_token_ids=token_ids,
                generation_log_probs=log_probs or None,
            )
        reply: GymEngineReply = {
            "output": output,
            "result": result.model_dump(exclude_none=True),
        }
        chunk: dict[str, Any] = {
            "token_ids": token_ids,
            "index": 0,
            "finish_reason": finish_reason,
            "engine_data": reply,
        }
        if last.get("completion_usage") is not None:
            chunk["completion_usage"] = last["completion_usage"]
        return chunk

    async def _stream_chunks(
        self, stream: Any, prompt_token_ids: list[int], params: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        sink = _FINAL_RECORD.get()
        if sink is not None:
            stream = _FinalTee(stream, sink)
        async for chunk in super()._stream_chunks(stream, prompt_token_ids, params):
            yield chunk


def _refusal(status: int, message: str) -> dict[str, Any]:
    """A Gym call the engine refused, answered in the chunk with ``status``."""
    reply: GymEngineReply = {
        "refusal": {"status": int(status), "message": message, "acknowledge": False}
    }
    return {"token_ids": [], "index": 0, "finish_reason": "stop", "engine_data": reply}
