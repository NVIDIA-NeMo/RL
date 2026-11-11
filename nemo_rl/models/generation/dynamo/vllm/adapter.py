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

"""Serve a NeMo RL vLLM worker's own AsyncLLM through Dynamo's backend SDK.

The engine process holds no Dynamo-owned vLLM code. ``DynamoVllmAdapter`` is the
vLLM backend of ``engine_adapter.DynamoEngineAdapter``, over the AsyncLLM that
``VllmAsyncGenerationWorkerImpl`` owns, so refit, pause, sleep, quantization and
speculative decoding are that worker's own.

It serves two request kinds, both token ids in: NeMo RL's own rollouts, and
NeMo-Gym's chat calls, whose prompt NeMo RL's chat engine in the frontend
rendered, prepared and routed (``chat_engine``). For those the adapter runs the
native server's steps after that (``prompt_and_capture``): ``start_capture``, the
generation with the SamplingParams the native server would build, and
``save_capture``'s one TQ write.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import math
import os
import pickle
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

import torch
from dynamo._core import Context
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LlmRegistration,
)
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, ZmqSource
from dynamo.llm.exceptions import InvalidArgument
from pydantic_core import to_jsonable_python
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.inputs import TokensPrompt
from vllm.v1.kv_cache_interface import KVCacheSpecKind
from vllm.v1.metrics.loggers import StatLoggerBase

from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo.engine_adapter import (
    DynamoEngineAdapter,
    RequestHandler,
    ServedRollout,
)
from nemo_rl.models.generation.dynamo.vllm.chat_engine import (
    GYM_CALL_ANNOTATION,
    PLAN_KEY,
    RUNTIME_DATA_KEY,
    ForwardedCall,
    GymEngineReply,
)
from nemo_rl.models.generation.dynamo.vllm.generation_cut import (
    CUT_PREFIXES_KEY,
    GENERATION_CUT_ANNOTATION,
    INVENTORY_KEY,
)
from nemo_rl.models.generation.gym_call import GymCallResult
from nemo_rl.models.generation.score_centering import (
    extract_score_centering_samples,
)
from nemo_rl.models.generation.vllm.prompt_and_capture import (
    PreparedPrompt,
    abort_request_capture,
    start_capture,
    abort_capture,
    save_capture,
)
from nemo_rl.models.generation.vllm.utils import (
    chat_generation_tokens,
    chat_routed_experts,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

# vLLM's main-attention kinds, as Dynamo's own vLLM backend counts them
# (dynamo/vllm/cache_info.py MAIN_ATTENTION_KV_CACHE_KINDS, which NeMo RL may
# not import); test_main_attention_kinds_match_dynamos checks the copy.
_MAIN_ATTENTION_KINDS = (
    KVCacheSpecKind.FULL_ATTENTION,
    KVCacheSpecKind.MLA_ATTENTION,
    KVCacheSpecKind.SINK_FULL_ATTENTION,
)


def dynamo_kv_events_config(
    cfg: DynamoCfg, *, prefix_caching: bool
) -> Optional[dict[str, Any]]:
    """The ``kv_events_config`` that feeds Dynamo's KV router.

    None unless the frontend routes by KV and the engine caches prefixes: vLLM
    publishes block events only for the blocks its prefix cache stores.
    """
    if cfg.frontend_args.router_mode != "kv" or not prefix_caching:
        return None
    return {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        # A random abstract-namespace socket per engine (dynamo-integration.md).
        "endpoint": f"ipc://@nemo-rl-kv-events-{os.urandom(8).hex()}",
    }


def chat_runtime_data(host: "VllmAsyncGenerationWorkerImpl") -> dict[str, Any]:
    """What the frontend's chat engine renders with, published at registration.

    The engine's own args (those a JSON payload can carry, which include every
    one ``ModelConfig`` reads), so the frontend builds the same tokenizer,
    template and processor; and all of ``policy.generation``, which the native
    server's chat helpers read.
    """
    engine_args: dict[str, Any] = {}
    for field in dataclasses.fields(host.llm_async_engine_args):
        value = getattr(host.llm_async_engine_args, field.name)
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = dataclasses.MISSING
        try:
            if value == default:
                continue
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        engine_args[field.name] = value
    # The frontend renders chat calls, so it always loads the tokenizer.
    engine_args["skip_tokenizer_init"] = False
    return {
        RUNTIME_DATA_KEY: {
            "engine_args": engine_args,
            # MasterConfig validates some blocks into pydantic models
            # (sampler_replay); anything else that is not JSON fails here.
            "generation": to_jsonable_python(host.cfg),
        }
    }


class SnapshotStatLogger(StatLoggerBase):
    """Push per-iteration KV usage into Dynamo's SnapshotPublisher (router load)."""

    # Set by Dynamo after vLLM built the logger, so shared through the class.
    publisher: Optional[Any] = None

    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.dp_rank = engine_index
        # vLLM sizes the KV cache before it builds its stat loggers.
        self.num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0

    def record(self, scheduler_stats, iteration_stats, *args, **kwargs) -> None:
        cls = type(self)
        if scheduler_stats is None or cls.publisher is None:
            return
        total = max(1, self.num_gpu_blocks)
        usage = scheduler_stats.kv_cache_usage
        hit_rate = None
        prefix = scheduler_stats.prefix_cache_stats
        if prefix.queries > 0:
            hit_rate = prefix.hits / prefix.queries
        cls.publisher.publish(
            self.dp_rank,
            ComponentSnapshot(
                kv_used_blocks=int(total * usage),
                kv_total_blocks=total,
                gpu_cache_usage=usage,
                kv_cache_hit_rate=hit_rate,
                dp_rank=self.dp_rank,
            ),
        )

    def log_engine_initialized(self) -> None:
        pass


class DynamoVllmAdapter(DynamoEngineAdapter):
    """The vLLM backend, over a running NeMo RL vLLM worker it does not own."""

    def __init__(
        self, host: "VllmAsyncGenerationWorkerImpl", *, leader_worker_idx: int
    ):
        super().__init__(leader_worker_idx=leader_worker_idx)
        self._host = host
        self._llm = host.llm
        self._max_model_len = self._llm.vllm_config.model_config.max_model_len

    def _handlers(self) -> dict[str, RequestHandler]:
        return {
            **super()._handlers(),
            GYM_CALL_ANNOTATION: self._generate_gym,
            GENERATION_CUT_ANNOTATION: self._generate_cut,
        }

    async def _generate_cut(
        self, request: GenerateRequest, annotation: str, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """Cut the calls of a checkpoint's inventory that this engine holds.

        The frontend sends every engine the whole inventory
        (:mod:`~nemo_rl.models.generation.dynamo.vllm.generation_cut`), so the
        answer names only the calls cut here.
        """
        del annotation, context
        # Deferred: Gym is optional outside framework-owned capture.
        from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

        inventory = GenerationCutInventory.model_validate(
            (request.get("extra_args") or {})[INVENTORY_KEY]
        )
        receipt = await self._host._prefix_recovery.run_cut(inventory)
        yield {
            "token_ids": [],
            "index": 0,
            "finish_reason": "stop",
            "engine_data": {
                CUT_PREFIXES_KEY: [
                    ack.model_dump(mode="json")
                    for ack in receipt.prefixes
                    if ack.disposition == "durable_prefix"
                ]
            },
        }

    async def serve_rollout(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
    ) -> ServedRollout:
        result, final_output = await self._host._serve_rollout(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            prompt_token_ids=prompt_token_ids,
            greedy=greedy,
            stop_strings=stop_strings,
            request_id=request_id,
            device=torch.device("cpu"),
        )
        if final_output is None:
            return ServedRollout(result, finish_reason="length", usage=None)
        return ServedRollout(
            result,
            finish_reason=final_output.outputs[0].finish_reason,
            usage=_usage(final_output, len(result.token_ids)),
        )

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        vllm_config = self._llm.vllm_config
        cache = vllm_config.cache_config
        scheduler = vllm_config.scheduler_config
        # The KV-event block size is the main-attention group's, which can
        # differ from cache_config.block_size on hybrid models.
        block_size = cache.block_size
        for group in await self._llm.engine_core.call_utility_async(
            "get_kv_cache_group_metadata"
        ):
            if group.get("kind") in _MAIN_ATTENTION_KINDS:
                block_size = group["block_size"]
                break
        model = self._host.model_name
        return EngineConfig(
            model=model,
            served_model_name=model,
            runtime_data=chat_runtime_data(self._host),
            # No token budget: the engine owns overflow (dynamo-integration.md).
            llm=LlmRegistration(
                context_length=self._max_model_len,
                kv_cache_block_size=block_size,
                total_kv_blocks=cache.num_gpu_blocks,
                max_num_seqs=scheduler.max_num_seqs,
                max_num_batched_tokens=scheduler.max_num_batched_tokens,
            ),
        )

    async def _generate_gym(
        self, request: GenerateRequest, annotation: str, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        # Deferred: chat_server imports vLLM's OpenAI server stack.
        from nemo_rl.models.generation.vllm.chat_server import (
            chat_request_from_render,
        )

        del annotation
        wire: Optional[ForwardedCall] = (request.get("extra_args") or {}).get(PLAN_KEY)
        if wire is None:
            raise InvalidArgument(
                "NeMo-Gym calls reach NeMo RL's engines through NeMo RL's chat "
                f"engine in the Dynamo frontend (extra_args.{PLAN_KEY})."
            )
        host = self._host
        plan = PreparedPrompt.from_json(
            wire["prepared_prompt"],
            prompt_token_ids=list(request["token_ids"]),
            parse_admission=host._parse_capture_admission,
        )
        chat = chat_request_from_render(wire["request"])
        try:
            chunk = await self._serve_gym_call(chat, plan, wire, context)
        except BaseException as error:
            failure = await abort_capture(host, chat, error)
            if failure is None:
                raise
            message = failure.message
            if failure.code != failure.status:
                # The frontend forwards only a status and a message.
                message = f"{message} ({failure.code})"
            chunk = _refusal(failure.status, message, acknowledge=failure.acknowledge)
        if chunk is not None:
            yield chunk

    async def _serve_gym_call(
        self, chat: Any, plan: PreparedPrompt, wire: ForwardedCall, context: Context
    ) -> Optional[GenerateChunk]:
        """One NeMo-Gym chat call, after the frontend prepared its prompt.

        The native server's steps after the plan, in its order: the queue
        check vLLM's serving runs first, :func:`start_capture` (a restore
        from a checkpoint cut, and ``begin_call``, which stamps the weight
        version), the generation, and :func:`save_capture`'s one TQ write.
        Returns one chunk whose ``engine_data`` carries the output the
        frontend builds the reply from and the call's :class:`GymCallResult`;
        None for a captured call whose client went away.
        """
        host = self._host
        # The worker's own engine client, as the native server uses it, with
        # the queue limits vLLM's serving checks before the render.
        engine_client = host._http_engine_client or self._llm
        engine_client.check_admission(1, context.id())
        engine_prompt: dict[str, Any]
        if wire["multimodal"] is not None:
            # The processed media, in the template's coordinates.
            engine_prompt = {
                **pickle.loads(base64.b64decode(wire["multimodal"])),
                "prompt_token_ids": list(plan.template_token_ids),
            }
        else:
            engine_prompt = TokensPrompt(prompt_token_ids=plan.prompt_token_ids)
        await start_capture(
            host, chat, engine_prompt, plan, max_model_len=self._max_model_len
        )
        captured = chat in host._capture_registry
        final_output = await self._generate(
            chat, engine_prompt, wire, engine_client, context
        )
        if plan.admission is not None and context.is_stopped():
            # Dynamo aborts the vLLM request on a disconnect but never
            # cancels this generator: fail the capture, as the native
            # endpoint's cancellation does, instead of staging it.
            await abort_request_capture(host, chat, "client_disconnect")
            return None
        completion = final_output.outputs[0]
        if completion.finish_reason == "error":
            # As vLLM's chat server does (_raise_if_error): never stage a failed call.
            raise RuntimeError(f"Generation failed for request {context.id()}")
        output: dict[str, Any] = {
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
        }
        if captured:
            result = await self._stage(chat, final_output)
        else:
            result = self._call_result(final_output)
            if chat.logprobs:
                output["logprobs"] = _wire_logprobs(completion.logprobs)
        reply: GymEngineReply = {
            "output": output,
            "result": result.model_dump(exclude_none=True),
        }
        if chat._restored_prompt_token_ids:
            # The frontend's reply puts the saved text back in front.
            terminal = chat._restored_prefix_terminal
            reply["restored"] = {
                "prompt_token_ids": list(chat._restored_prompt_token_ids),
                "token_ids": list(chat._restored_generation_token_ids),
                "terminal": None
                if terminal is None
                else {
                    "prefix_token_count": terminal.prefix_token_count,
                    "reason": terminal.reason,
                    "finish_reason": terminal.finish_reason,
                    "stop_reason": terminal.stop_reason,
                },
            }
        return {
            "token_ids": output["token_ids"],
            "index": 0,
            "finish_reason": completion.finish_reason,
            "completion_usage": _usage(final_output, len(completion.token_ids)),
            "engine_data": reply,
        }

    async def _generate(
        self,
        chat: Any,
        engine_prompt: Any,
        wire: ForwardedCall,
        engine_client: Any,
        context: Context,
    ) -> Any:
        """Generate as the native server does, with the SamplingParams it builds."""
        model_config = self._llm.model_config
        default_sampling_params = model_config.get_diff_sampling_param()
        override_max_tokens = (
            default_sampling_params.get("max_tokens")
            if model_config.generation_config not in ("auto", "vllm")
            else getattr(model_config, "override_generation_config", {}).get(
                "max_new_tokens"
            )
        )
        max_tokens = get_max_tokens(
            self._max_model_len,
            chat.max_completion_tokens
            if chat.max_completion_tokens is not None
            else chat.max_tokens,
            len(engine_prompt["prompt_token_ids"]),
            default_sampling_params,
            override_max_tokens,
            truncate_prompt_tokens=chat.truncate_prompt_tokens,
        )
        params = chat.to_sampling_params(max_tokens, default_sampling_params)
        if chat.ng_capture is not None:
            # As the native request does: a cut needs the call's resolved budget.
            self._host._prefix_recovery.on_sampling_params(chat, params)
        final_output = None
        async for output in engine_client.generate(
            engine_prompt,
            params,
            context.id(),
            reasoning_ended=wire["reasoning_ended"],
            reasoning_parser_kwargs=wire["reasoning_parser_kwargs"],
        ):
            # Recorded before asking for the next output, which releases
            # vLLM's finished state for this request.
            self._host._prefix_recovery.observe(chat, output)
            final_output = output
        if final_output is None:
            raise RuntimeError(f"No output received for request {context.id()}")
        return final_output

    def _routes(self, final_output: Any, *, encode_for_wire: bool) -> Any:
        """The call's routes, as the native endpoint attaches them; None if off.

        With router replay on, a call vLLM returned no routes for fails
        (``chat_routed_experts`` raises), captured or not.
        """
        if not self._host._return_routed_experts_enabled():
            return None
        completion = final_output.outputs[0]
        return chat_routed_experts(
            final_output,
            completion,
            prompt_token_count=len(final_output.prompt_token_ids),
            device=torch.device("cpu"),
            logger=LOGGER,
            routed_experts_dtype=self._host.routed_experts_dtype,
            encode_for_wire=encode_for_wire,
        )

    def _call_result(self, final_output: Any) -> GymCallResult:
        """An uncaptured call's tokens and routes, as the native message carries them."""
        completion = final_output.outputs[0]
        generation_log_probs = None
        if completion.logprobs is not None:
            _, generation_log_probs = chat_generation_tokens(completion)
            for index, log_prob in enumerate(generation_log_probs):
                if not math.isfinite(log_prob):
                    # Strict JSON has no NaN or inf (Dynamo sends null). The native
                    # server's JSONResponse fails this call with 500, which Gym retries.
                    raise RuntimeError(
                        f"Generation produced a non-finite log-prob ({log_prob}) at "
                        f"token {index} of {len(generation_log_probs)}"
                    )
        return GymCallResult(
            prompt_token_ids=list(final_output.prompt_token_ids),
            generation_token_ids=list(completion.token_ids),
            generation_log_probs=generation_log_probs,
            routed_experts=self._routes(final_output, encode_for_wire=True),
        )

    async def _stage(self, chat: Any, final_output: Any) -> GymCallResult:
        """Stage a captured call from this process, in one TQ write."""
        # Deferred: only captured calls reach here, and capture already imports nemo_gym.
        from nemo_gym.token_id_capture import NG_COMMIT_COORDS_FIELD

        # Off the loop, as natively: the sampler top-k scales with the
        # completion's length and K.
        content = await asyncio.to_thread(self._response_to_stage, final_output)
        content = await save_capture(self._host, chat, content)
        return GymCallResult(ng_commit_coords=content[NG_COMMIT_COORDS_FIELD])

    def _response_to_stage(self, final_output: Any) -> dict[str, Any]:
        """A captured call's response, with the fields the native endpoint stages."""
        completion = final_output.outputs[0]
        message: dict[str, Any] = {"generation_token_ids": list(completion.token_ids)}
        if completion.logprobs is not None:
            _, message["generation_log_probs"] = chat_generation_tokens(completion)
        routes = self._routes(final_output, encode_for_wire=False)
        if routes is not None:
            message["routed_experts"] = routes
        if self._host._capture_top_logprobs:
            # Score centering stages the sampler's top-k with the call.
            (
                message["sampler_topk_token_ids"],
                message["sampler_topk_logprobs"],
            ) = extract_score_centering_samples(
                completion, num_candidates=self._host._capture_top_logprobs
            )
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": completion.finish_reason,
                    "message": message,
                }
            ]
        }

    async def abort(self, context: Context) -> None:
        await self._llm.abort(context.id())

    async def kv_event_sources(self) -> list[KvEventSource]:
        kv_events = self._llm.vllm_config.kv_events_config
        if kv_events is None or not kv_events.enable_kv_cache_events:
            return []
        # A user's own kv_events_config can keep vLLM's wildcard default
        # (tcp://*:5557): vLLM binds it, but a subscriber cannot connect to "*".
        return [ZmqSource(endpoint=kv_events.endpoint.replace("*", "127.0.0.1"))]

    def component_metrics_dp_ranks(self) -> list[int]:
        return [0]

    def attach_snapshot_publisher(self, publisher: Any) -> None:
        SnapshotStatLogger.publisher = publisher

    async def cleanup(self) -> None:
        # The worker owns the AsyncLLM; its shutdown() tears the engine down.
        SnapshotStatLogger.publisher = None


def _wire_logprobs(logprobs: Any) -> Optional[list[dict[str, list[Any]]]]:
    """Per-position logprobs as JSON, for the frontend's OpenAI ``logprobs``."""
    if logprobs is None:
        return None
    return [
        {
            str(token_id): [entry.logprob, entry.rank, entry.decoded_token]
            for token_id, entry in position.items()
        }
        for position in logprobs
    ]


def _refusal(status: int, message: str, *, acknowledge: bool = False) -> GenerateChunk:
    """A Gym call the engine refused before it ran, answered in the chunk.

    Dynamo hands an error the engine raises to the frontend's chat engine
    without its status, so it could only answer 500; the chat engine answers
    this with ``status``, as the native server does.
    """
    reply: GymEngineReply = {
        "refusal": {
            "status": int(status),
            "message": message,
            "acknowledge": acknowledge,
        }
    }
    return {
        "token_ids": [],
        "index": 0,
        "finish_reason": "stop",
        "engine_data": reply,
    }


def _usage(output: Any, num_output_tokens: int) -> dict[str, Any]:
    prompt_tokens = len(output.prompt_token_ids or ())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": num_output_tokens,
        "total_tokens": prompt_tokens + num_output_tokens,
        "prompt_tokens_details": {"cached_tokens": output.num_cached_tokens or 0},
    }
