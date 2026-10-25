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

The engine process holds no Dynamo-owned vLLM code. This adapter implements
``dynamo.common.backend.LLMEngine``, which imports no vLLM, over the AsyncLLM
that ``VllmAsyncGenerationWorkerImpl`` owns, so refit, pause, sleep,
quantization and speculative decoding are that worker's own.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from dynamo._core import Context, get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
    Worker,
    WorkerConfig,
)
from dynamo.common.backend.logprobs import (
    extract_from_completion_output,
    extract_prompt_logprobs_from_completion_output,
    parse_logprob_options,
)
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, ZmqSource
from dynamo.llm.exceptions import HttpError, InvalidArgument
from vllm import envs as vllm_envs
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.exceptions import GracefulHTTPError, VLLMClientError
from vllm.inputs import TokensPrompt
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.transformers_utils.processor import get_video_processor_cls_name
from vllm.v1.kv_cache_interface import KVCacheSpecKind
from vllm.v1.metrics.loggers import StatLoggerBase

from nemo_rl.data.captured_media import MediaCaptureRejected
from nemo_rl.models.generation.dynamo.config import (
    DynamoCfg,
    dynamo_cfg,
    dynamo_chat_template,
    dynamo_parsers,
)
from nemo_rl.models.generation.dynamo.managed_runtime import (
    DYNAMO_COMPONENT,
    DYNAMO_ENDPOINT,
)
from nemo_rl.models.generation.dynamo.rollout_client import ROLLOUT_ANNOTATION
from nemo_rl.models.generation.openai_server_utils import (
    derive_prefix_eos_count,
    splice_prefix_tokens,
)
from nemo_rl.models.generation.vllm.request_pipeline import (
    RolloutResult,
    rollout_result,
    rollout_result_to_wire,
)
from nemo_rl.models.generation.vllm.utils import (
    chat_generation_tokens,
    chat_routed_experts,
    off_policy_sampling_error,
    remap_multimodal_placeholders,
    validate_rollout_prompt,
)

if TYPE_CHECKING:
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

LOGGER = logging.getLogger(__name__)

# Annotations NeMo-Gym's Dynamo dialect (vllm_model with dynamo_frontend, Gym#4817)
# sets on its requests; Dynamo forwards nvext.annotations to the engine verbatim.
GYM_ROLLOUT_ANNOTATION = "nemo_gym:rollout"
# A token-capture call's admission JSON.
GYM_CAPTURE_ANNOTATION_PREFIX = "ng_capture:"
# Without capture: the last echoed assistant turn's prompt + generation token
# IDs, which the native server reads from the messages instead.
GYM_PREFIX_ANNOTATION_PREFIX = "ng_prefix:"
_GYM_SAMPLING_KEYS = ("temperature", "top_p", "val_temperature", "val_top_p")

# The vLLM modality each key of Dynamo's multi_modal_data holds.
_DYNAMO_MEDIA_MODALITIES = {
    "image_url": "image",
    "video_url": "video",
    "audio_url": "audio",
}

# Filled from the model's generation config when a request leaves them unset,
# as vLLM's OpenAI server fills them.
_GENERATION_CONFIG_KEYS = (
    "repetition_penalty",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
)
_SAMPLING_KEYS = (
    *_GENERATION_CONFIG_KEYS,
    "seed",
    "presence_penalty",
    "frequency_penalty",
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
        # A Linux abstract-namespace socket (no port, no file): vLLM binds any
        # ipc:// endpoint and this process's Dynamo worker connects to it, so
        # engines on one node cannot collide. Random rather than the pid, since
        # containers sharing the host network share these names. vLLM rejects
        # ipc:// only at data_parallel_rank != 0, which Dynamo rejects at startup.
        "endpoint": f"ipc://@nemo-rl-kv-events-{os.urandom(8).hex()}",
    }


def _annotation_value(request: GenerateRequest, prefix: str) -> Optional[str]:
    """The value of the first ``<prefix><value>`` annotation, or None."""
    for annotation in request.get("annotations") or ():
        if annotation.startswith(prefix):
            return annotation[len(prefix) :]
    return None


def _rollout_request(request: GenerateRequest) -> Optional[dict[str, Any]]:
    """The rollout spec if ``request`` is one of NeMo RL's own rollouts.

    Those come from ``VllmGeneration.generate_async`` and are built here
    exactly as the native path builds them.
    """
    rollout = _annotation_value(request, ROLLOUT_ANNOTATION)
    return None if rollout is None else json.loads(rollout)


def _as_f32(value: Optional[float]) -> Optional[float]:
    """Dynamo's frontend carries sampling floats as f32 (0.7 arrives as 0.69999998)."""
    return None if value is None else float(np.float32(value))


def structured_outputs(guided_decoding: dict[str, Any]) -> StructuredOutputsParams:
    """Structured outputs for Dynamo's ``sampling_options.guided_decoding``.

    Dynamo sets at most one constraint. The decoding backend is the engine's
    own (``structured_outputs_config``), as for the native server.
    """
    structural_tag = guided_decoding.get("structural_tag")
    if isinstance(structural_tag, dict):
        structural_tag = json.dumps(structural_tag)
    return StructuredOutputsParams(
        json=guided_decoding.get("json"),
        regex=guided_decoding.get("regex"),
        choice=guided_decoding.get("choice"),
        grammar=guided_decoding.get("grammar"),
        whitespace_pattern=guided_decoding.get("whitespace_pattern"),
        structural_tag=structural_tag,
    )


def build_sampling_params(
    request: GenerateRequest,
    defaults: dict[str, Any],
    *,
    output_room: int,
    overrides: Optional[dict[str, Any]] = None,
    output_kind: RequestOutputKind = RequestOutputKind.DELTA,
) -> SamplingParams:
    """Map an OpenAI-shaped Dynamo request onto vLLM SamplingParams.

    As vLLM's OpenAI server builds them: the model's generation-config
    ``defaults`` for what the request leaves unset, and ``max_tokens`` capped
    by the context. Stop strings are applied by Dynamo's Rust detokenizer, so
    only token-level stops reach vLLM unless ``overrides`` sets them.

    Args:
        request: Dynamo's preprocessed request.
        defaults: ``ModelConfig.get_diff_sampling_param()``.
        output_room: The context left after the prompt the engine runs, which
            media expansion or a prefix splice can make longer than the
            request's ``token_ids``.
        overrides: SamplingParams fields that replace the request's.
        output_kind: How vLLM reports outputs.
    """
    sampling = request.get("sampling_options") or {}
    passthrough = (request.get("extra_args") or {}).get("sampling_options") or {}
    stop = request.get("stop_conditions") or {}
    n = sampling.get("n") or 1
    if sampling.get("best_of") not in (None, n):
        raise InvalidArgument("best_of is not supported by vLLM's V1 engine")

    kwargs: dict[str, Any] = {
        key: defaults[key] for key in _GENERATION_CONFIG_KEYS if key in defaults
    }
    kwargs.update(
        {key: sampling[key] for key in _SAMPLING_KEYS if sampling.get(key) is not None}
    )
    kwargs["n"] = n
    # As vLLM's get_max_tokens: the generation config's max_tokens applies only
    # when the request leaves max_tokens unset.
    requested_max_tokens = stop.get("max_tokens")
    if requested_max_tokens is None:
        requested_max_tokens = defaults.get("max_tokens")
    kwargs["max_tokens"] = min(
        limit
        for limit in (max(1, output_room), requested_max_tokens)
        if limit is not None
    )
    if stop.get("min_tokens") is not None:
        kwargs["min_tokens"] = stop["min_tokens"]
    if stop.get("ignore_eos") is not None:
        kwargs["ignore_eos"] = stop["ignore_eos"]
    stop_ids = set(stop.get("stop_token_ids") or ()) | set(
        stop.get("stop_token_ids_hidden") or ()
    )
    if stop_ids:
        kwargs["stop_token_ids"] = sorted(stop_ids)
    if sampling.get("guided_decoding"):
        kwargs["structured_outputs"] = structured_outputs(sampling["guided_decoding"])
    for key in ("allowed_token_ids", "logprob_token_ids"):
        if passthrough.get(key) is not None:
            kwargs[key] = passthrough[key]

    logprobs, prompt_logprobs = parse_logprob_options(
        request.get("output_options") or {}
    )
    # Explicit logprob_token_ids replace top-k selection; vLLM rejects both.
    if logprobs is not None and "logprob_token_ids" not in kwargs:
        kwargs["logprobs"] = logprobs
    if prompt_logprobs is not None:
        kwargs["prompt_logprobs"] = prompt_logprobs
    kwargs["detokenize"] = False
    kwargs.update(overrides or {})
    params = SamplingParams(**kwargs, output_kind=output_kind)
    if passthrough.get("bad_words_token_ids") is not None:
        # vLLM only tokenizes bad_words strings into this field; a token-in
        # client sends the token IDs themselves.
        if not hasattr(params, "_bad_words_token_ids"):
            raise RuntimeError(
                "vLLM SamplingParams has no _bad_words_token_ids; update the "
                "bad_words_token_ids passthrough for this vLLM"
            )
        params._bad_words_token_ids = passthrough["bad_words_token_ids"]
    return params


class SnapshotStatLogger(StatLoggerBase):
    """Push per-iteration KV usage into Dynamo's SnapshotPublisher (router load)."""

    publisher: Optional[Any] = None
    num_gpu_blocks: int = 0

    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.dp_rank = engine_index

    def record(self, scheduler_stats, iteration_stats, *args, **kwargs) -> None:
        cls = type(self)
        if scheduler_stats is None or cls.publisher is None:
            return
        total = max(1, cls.num_gpu_blocks)
        usage = scheduler_stats.kv_cache_usage
        hit_rate = None
        prefix = getattr(scheduler_stats, "prefix_cache_stats", None)
        if prefix is not None and prefix.queries > 0:
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


class NativeVllmEngine(LLMEngine):
    """LLMEngine over a running NeMo RL vLLM worker. Does not own its lifecycle."""

    def __init__(self, host: "VllmAsyncGenerationWorkerImpl", leader_worker_idx: int):
        self._host = host
        self._leader_worker_idx = leader_worker_idx
        self._llm = host.llm
        self._max_model_len = self._llm.vllm_config.model_config.max_model_len
        self._defaults = self._llm.model_config.get_diff_sampling_param()
        # The train and validation sampling NeMo-Gym rollouts must match, at the
        # f32 precision the frontend delivers.
        self._gym_sampling_profile = {
            key: _as_f32(host.cfg[key]) for key in _GYM_SAMPLING_KEYS
        }

    @classmethod
    async def from_args(cls, argv=None):
        raise NotImplementedError("constructed in-process by the vLLM worker")

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
            # vLLM's main-attention kinds, as Dynamo's own vLLM backend counts them.
            if group.get("kind") in (
                KVCacheSpecKind.FULL_ATTENTION,
                KVCacheSpecKind.MLA_ATTENTION,
                KVCacheSpecKind.SINK_FULL_ATTENTION,
            ):
                block_size = group["block_size"]
                break
        SnapshotStatLogger.num_gpu_blocks = cache.num_gpu_blocks or 0
        model = self._host.model_name
        return EngineConfig(
            model=model,
            served_model_name=model,
            # No token budget: the engine owns prompt overflow and the output
            # cap, as the native server does. A budget would make the frontend
            # fill an omitted max_tokens from the prompt it rendered, which a
            # NeMo-Gym prefix splice then changes, and refuse NeMo RL's own
            # rollouts that fill the context.
            llm=LlmRegistration(
                context_length=self._max_model_len,
                kv_cache_block_size=block_size,
                total_kv_blocks=cache.num_gpu_blocks,
                max_num_seqs=scheduler.max_num_seqs,
                max_num_batched_tokens=scheduler.max_num_batched_tokens,
            ),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        try:
            async for chunk in self._generate(request, context):
                yield chunk
        except VLLMClientError as error:
            # vLLM's request errors are not ValueErrors; the native server
            # answers them with a 4xx that carries the message.
            raise InvalidArgument(str(error)) from error

    async def _generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if request.get("prompt_embeds"):
            raise InvalidArgument(
                "Embedding prompts are not served through Dynamo by NeMo RL's "
                "vLLM engines."
            )
        rollout = _rollout_request(request)
        if rollout is not None:
            async for chunk in self._generate_rollout(request, rollout, context):
                yield chunk
            return
        annotations = request.get("annotations") or ()
        if GYM_ROLLOUT_ANNOTATION in annotations:
            async for chunk in self._generate_gym(request, context):
                yield chunk
            return
        engine_input = await self._engine_input(request)
        prompt_len = len(engine_input["prompt_token_ids"])
        self._check_prompt_fits(prompt_len)
        sampling_params = build_sampling_params(
            request, self._defaults, output_room=self._max_model_len - prompt_len
        )
        try:
            # vLLM's queue limits (max_num_queued_reqs / max_num_queued_tokens),
            # checked before every request as the native server checks them.
            self._llm.check_admission(sampling_params.n, context.id())
        except GracefulHTTPError as error:
            raise HttpError(int(error.http_status), error.message) from error
        logprobs_requested, prompt_logprobs_requested = parse_logprob_options(
            request.get("output_options") or {}
        )
        tokenizer = self._llm.tokenizer
        completion_tokens: dict[int, int] = {}
        prompt_output = None
        async for output in self._llm.generate(
            engine_input, sampling_params, context.id()
        ):
            # Under DELTA, vLLM sends every prompt logprob once, at the end of
            # prefill; later outputs carry an empty list.
            if getattr(output, "prompt_logprobs", None):
                prompt_output = output
            for completion in output.outputs:
                token_ids = list(completion.token_ids)
                completion_tokens[completion.index] = completion_tokens.get(
                    completion.index, 0
                ) + len(token_ids)
                if not token_ids and completion.finish_reason is None:
                    continue
                chunk: GenerateChunk = {
                    "token_ids": token_ids,
                    "index": completion.index,
                }
                log_probs, top_logprobs = extract_from_completion_output(
                    completion, 0, tokenizer=tokenizer, include_bytes=True
                )
                if log_probs is not None:
                    chunk["log_probs"] = log_probs
                    if logprobs_requested:
                        chunk["top_logprobs"] = top_logprobs
                if completion.finish_reason is not None:
                    chunk["finish_reason"] = completion.finish_reason
                    chunk["completion_usage"] = _usage(
                        output, sum(completion_tokens.values())
                    )
                    if prompt_logprobs_requested is not None and prompt_output:
                        prompt_logprobs = (
                            extract_prompt_logprobs_from_completion_output(
                                prompt_output, tokenizer=tokenizer
                            )
                        )
                        if prompt_logprobs is not None:
                            chunk["engine_data"] = {"prompt_logprobs": prompt_logprobs}
                yield chunk

    async def _generate_rollout(
        self, request: GenerateRequest, rollout: dict[str, Any], context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """One NeMo RL rollout, served exactly as ``generate_async`` serves it.

        The worker builds the SamplingParams and the shared request pipeline
        builds the result, routed experts included, from vLLM's final output.
        One chunk carries it all in ``engine_data``; the driver assembles the
        same output ``generate_async`` returns.
        """
        host = self._host
        prompt_token_ids = list(request["token_ids"])
        max_new_tokens = host._rollout_max_new_tokens(len(prompt_token_ids))
        final_output = None
        if max_new_tokens == 0:
            result = RolloutResult.nothing_generated()
        else:
            sampling_params = host._rollout_sampling_params(
                max_new_tokens=max_new_tokens,
                greedy=rollout["greedy"],
                stop_strings=rollout["stop_strings"],
            )
            async for output in self._llm.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params,
                context.id(),
            ):
                final_output = output
            if final_output is None:
                raise RuntimeError(f"No output received for request {context.id()}")
            validate_rollout_prompt(prompt_token_ids, final_output.prompt_token_ids)
            result = rollout_result(
                final_output,
                input_length=len(prompt_token_ids),
                return_routed_experts=host._return_routed_experts_enabled(),
                routed_experts_dtype=host.routed_experts_dtype,
                device=torch.device("cpu"),
            )
        chunk: GenerateChunk = {
            "token_ids": result.token_ids,
            "index": 0,
            "finish_reason": (
                "length"
                if final_output is None
                else final_output.outputs[0].finish_reason
            ),
            "engine_data": {
                "rollout": rollout_result_to_wire(result),
                "gen_leader_worker_idx": self._leader_worker_idx,
            },
        }
        if final_output is not None:
            chunk["completion_usage"] = _usage(final_output, len(result.token_ids))
        yield chunk

    async def _generate_gym(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """One NeMo-Gym model call, served as the native chat endpoint serves it.

        The request was rendered by the frontend. As on the native server, it must
        sample on-policy, its echoed or staged prefix replaces the rendered
        history, output is clamped to the context, and a token-capture call stages
        its delta from this process. Tokens, log probabilities and routed experts
        come from vLLM's final output through the native endpoint's helpers. One
        chunk carries the whole completion and, in ``engine_data``, the token
        arrays Gym's dialect reads, or a captured call's commit coordinates.
        """
        host = self._host
        sampling: dict[str, Any] = request.get("sampling_options") or {}
        sampling_error = off_policy_sampling_error(
            temperature=_as_f32(sampling.get("temperature")),
            top_p=_as_f32(sampling.get("top_p")),
            top_k=sampling.get("top_k"),
            generation_config=self._gym_sampling_profile,
        )
        if sampling_error is not None:
            raise InvalidArgument(sampling_error)
        if (sampling.get("n") or 1) != 1:
            raise InvalidArgument(
                "NeMo-Gym calls through Dynamo take n=1: engine_data carries one "
                "choice's tokens."
            )
        # The profile's own values, so vLLM samples exactly as natively.
        validation = (
            _as_f32(sampling["temperature"])
            != self._gym_sampling_profile["temperature"]
            or _as_f32(sampling["top_p"]) != self._gym_sampling_profile["top_p"]
        )
        temperature, top_p = (
            (host.cfg["val_temperature"], host.cfg["val_top_p"])
            if validation
            else (host.cfg["temperature"], host.cfg["top_p"])
        )

        capture_context = _annotation_value(request, GYM_CAPTURE_ANNOTATION_PREFIX)
        admission = (
            None
            if capture_context is None
            else host._parse_capture_admission(json.loads(capture_context))
        )
        prefix_token_ids = None
        if admission is not None:
            if admission.mode == "token_in":
                # A blocking data-plane read; keep it off the engine loop.
                prefix_token_ids = await asyncio.to_thread(
                    host._resolve_admission_prefix, admission
                )
        elif (
            echoed := _annotation_value(request, GYM_PREFIX_ANNOTATION_PREFIX)
        ) is not None:
            prefix_token_ids = json.loads(echoed)
        # As the native endpoint does: a captured call reads the processor's own
        # pixels, not cache references (#4124).
        engine_input = await self._engine_input(
            request, skip_mm_cache=admission is not None and host._capture_media
        )
        template_token_ids = engine_input["prompt_token_ids"]
        self._check_prompt_fits(len(template_token_ids))
        token_ids, splice = template_token_ids, None
        if prefix_token_ids is not None:
            # The frontend rendered the whole conversation; replace its history
            # with the exact tokens the model generated for it. The engine sees no
            # messages, so the prefix's EOS count stands in for the native
            # endpoint's render of the history.
            tokenizer = self._llm.tokenizer
            splice = splice_prefix_tokens(
                tokenizer=tokenizer,
                model_prefix_token_ids=prefix_token_ids,
                template_token_ids=template_token_ids,
                template_prefix_eos_count=derive_prefix_eos_count(
                    prefix_token_ids, eos_token_id=tokenizer.eos_token_id
                ),
            )
            token_ids = splice.token_ids
            self._check_prompt_fits(len(token_ids))
        try:
            # Off the loop: a retained-media check reads staged calls.
            media = await asyncio.to_thread(
                host._capture_request_media,
                engine_input,
                admission=admission,
                splice=splice,
            )
        except MediaCaptureRejected as error:
            raise InvalidArgument(f"{error} ({error.code})") from error
        if (
            media is None
            and splice is not None
            and (placeholders := engine_input.get("mm_placeholders"))
        ):
            # Media capture already moved them to spliced coordinates.
            engine_input["mm_placeholders"] = remap_multimodal_placeholders(
                template_token_ids=template_token_ids,
                final_token_ids=token_ids,
                mm_placeholders=placeholders,
                splice=splice,
            )
        engine_input["prompt_token_ids"] = token_ids

        stop = (request.get("stop_conditions") or {}).get("stop")
        overrides: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            # As the native endpoint does: Gym requests leave top_k to NeMo RL.
            "top_k": -1,
        }
        if stop:
            # vLLM stops on them as natively, so the token arrays match; the
            # frontend then cuts the text at the same place.
            overrides.update(stop=stop, detokenize=True)
        if admission is not None:
            # A captured call stages the sampled tokens' log probabilities.
            overrides["logprobs"] = 0
        sampling_params = build_sampling_params(
            request,
            self._defaults,
            output_room=self._max_model_len - len(token_ids),
            overrides=overrides,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        try:
            self._llm.check_admission(1, context.id())
        except GracefulHTTPError as error:
            raise HttpError(int(error.http_status), error.message) from error

        if admission is not None:
            host._begin_request_capture(
                request,
                token_ids,
                admission=admission,
                prefix_token_ids=prefix_token_ids,
                media=media,
            )
        staged = False
        try:
            final_output = None
            async for output in self._llm.generate(
                engine_input, sampling_params, context.id()
            ):
                final_output = output
            if final_output is None:
                raise RuntimeError(f"No output received for request {context.id()}")
            if admission is not None and context.is_stopped():
                # Dynamo aborts the vLLM request on a disconnect but never
                # cancels this generator: fail the capture (below), as the
                # native endpoint's cancellation does, instead of staging it.
                return
            completion = final_output.outputs[0]
            chunk: GenerateChunk = {
                "token_ids": list(completion.token_ids),
                "index": 0,
                "finish_reason": completion.finish_reason,
                "completion_usage": _usage(final_output, len(completion.token_ids)),
            }
            if admission is not None:
                chunk["engine_data"] = {
                    "ng_commit_coords": await self._stage_gym_call(
                        request, final_output
                    )
                }
                staged = True
            else:
                chunk["engine_data"] = self._gym_token_data(final_output)
                log_probs, top_logprobs = extract_from_completion_output(
                    completion, 0, tokenizer=self._llm.tokenizer, include_bytes=True
                )
                if log_probs is not None:
                    chunk["log_probs"] = log_probs
                    chunk["top_logprobs"] = top_logprobs
            yield chunk
        finally:
            if admission is not None and not staged:
                host._abort_request_capture(
                    request,
                    reason=(
                        "client_disconnect" if context.is_stopped() else "engine_error"
                    ),
                )

    async def _load_media(
        self, request: GenerateRequest
    ) -> Optional[tuple[dict[str, list[Any]], dict[str, list[Any]]]]:
        """The request's media and their UUIDs by modality, as vLLM's chat parser loads them.

        Dynamo passes each media URL through (data: URIs included); the engine
        fetches and decodes it with vLLM's own media connector and settings.
        """
        media = request.get("multi_modal_data")
        if not media:
            return None
        model_config = self._llm.model_config
        mm_config = model_config.multimodal_config
        connector = MEDIA_CONNECTOR_REGISTRY.load(
            vllm_envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=request.get("media_io_kwargs")
            or (mm_config.media_io_kwargs if mm_config else None),
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        fetch = {
            "image": connector.fetch_image_async,
            "audio": connector.fetch_audio_async,
        }
        if "video_url" in media:
            fetch["video"] = functools.partial(
                connector.fetch_video_async,
                video_processor=get_video_processor_cls_name(model_config),
            )
        uuids = request.get("multi_modal_uuids") or {}
        mm_data: dict[str, list[Any]] = {}
        mm_uuids: dict[str, list[Any]] = {}
        for key, items in media.items():
            modality = _DYNAMO_MEDIA_MODALITIES.get(key)
            if modality is None:
                raise InvalidArgument(f"Unsupported media type {key!r}")
            if any("Url" not in item for item in items):
                raise InvalidArgument(
                    f"{key} items must carry their URL: NeMo RL's vLLM engines load "
                    "media themselves"
                )
            try:
                mm_data[modality] = list(
                    await asyncio.gather(
                        *(fetch[modality](item["Url"]) for item in items)
                    )
                )
            except ValueError as error:
                raise InvalidArgument(f"Cannot load {key}: {error}") from error
            mm_uuids[modality] = list(uuids.get(key) or [None] * len(items))
        return mm_data, mm_uuids

    async def _engine_input(
        self, request: GenerateRequest, *, skip_mm_cache: bool = False
    ) -> dict[str, Any]:
        """The processed vLLM prompt for the request, as the native chat endpoint builds it.

        The frontend's tokens carry one unexpanded placeholder per media item;
        vLLM's processor expands them and processes the media, exactly as after
        the native endpoint renders and tokenizes the messages.
        """
        prompt = TokensPrompt(prompt_token_ids=list(request["token_ids"]))
        media = await self._load_media(request)
        if media is not None:
            prompt["multi_modal_data"], prompt["multi_modal_uuids"] = media
            if request.get("mm_processor_kwargs"):
                prompt["mm_processor_kwargs"] = request["mm_processor_kwargs"]
        return await self._llm.renderer.process_for_engine_async(
            prompt, time.time(), skip_mm_cache=skip_mm_cache
        )

    def _check_prompt_fits(self, prompt_len: int) -> None:
        """Refuse a prompt that leaves no room for output, as the native endpoint does.

        NeMo-Gym reads "context length" in a 400 as a length refusal rather than an
        engine failure.
        """
        if prompt_len >= self._max_model_len:
            raise InvalidArgument(
                f"Prompt length ({prompt_len}) fills or exceeds this model's "
                f"maximum context length ({self._max_model_len}). No room for "
                "output tokens."
            )

    def _gym_token_data(self, final_output: Any) -> dict[str, Any]:
        """The token fields Gym's dialect lifts onto the message, as the native endpoint reports them."""
        completion = final_output.outputs[0]
        token_data: dict[str, Any] = {
            "prompt_token_ids": list(final_output.prompt_token_ids),
            "completion_token_ids": list(completion.token_ids),
        }
        if completion.logprobs is not None:
            _, token_data["completion_logprobs"] = chat_generation_tokens(completion)
        if self._host._return_routed_experts_enabled():
            # Full prompt + generation rows, as the native endpoint attaches them.
            token_data["routed_experts"] = chat_routed_experts(
                final_output,
                completion,
                prompt_token_count=len(final_output.prompt_token_ids),
                device=torch.device("cpu"),
                logger=LOGGER,
                routed_experts_dtype=self._host.routed_experts_dtype,
            )
        return token_data

    async def _stage_gym_call(self, request: GenerateRequest, final_output: Any) -> Any:
        """Stage a captured call from this process; return its commit coordinates."""
        host = self._host
        completion = final_output.outputs[0]
        generation_token_ids, generation_log_probs = chat_generation_tokens(completion)
        message: dict[str, Any] = {
            "generation_token_ids": generation_token_ids,
            "generation_log_probs": generation_log_probs,
        }
        if host._return_routed_experts_enabled():
            message["routed_experts"] = chat_routed_experts(
                final_output,
                completion,
                prompt_token_count=len(final_output.prompt_token_ids),
                device=torch.device("cpu"),
                logger=LOGGER,
                routed_experts_dtype=host.routed_experts_dtype,
            )
        # The native endpoint's response shape; the prompt is the one recorded at
        # begin_call.
        content = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": completion.finish_reason,
                    "message": message,
                }
            ]
        }
        # The sink write is a blocking data-plane round trip.
        content = await asyncio.to_thread(
            host._finish_request_capture, request, content
        )
        return content["ng_commit_coords"]

    async def abort(self, context: Context) -> None:
        await self._llm.abort(context.id())

    async def kv_event_sources(self) -> list[KvEventSource]:
        kv_events = self._llm.vllm_config.kv_events_config
        if kv_events is None or not kv_events.enable_kv_cache_events:
            return []
        return [ZmqSource(endpoint=kv_events.endpoint)]

    def component_metrics_dp_ranks(self) -> list[int]:
        return [0]

    def attach_snapshot_publisher(self, publisher: Any) -> None:
        SnapshotStatLogger.publisher = publisher

    async def cleanup(self) -> None:
        # The worker owns the AsyncLLM; its shutdown() tears the engine down.
        SnapshotStatLogger.publisher = None


def _usage(output: Any, num_output_tokens: int) -> dict[str, Any]:
    prompt_tokens = len(output.prompt_token_ids or ())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": num_output_tokens,
        "total_tokens": prompt_tokens + num_output_tokens,
        "prompt_tokens_details": {"cached_tokens": output.num_cached_tokens or 0},
    }


def _check_parser_name(name: Optional[str], known: list[str], field: str) -> None:
    if name is not None and name not in known:
        raise ValueError(
            f"Dynamo has no {field.replace('_', ' ')} named {name!r}; set "
            f"policy.generation.dynamo_cfg.worker_args.{field} to one of "
            f"{sorted(known)}."
        )


def _chat_template_file(chat_template: Optional[str]) -> Optional[str]:
    """A file holding ``chat_template``, which Dynamo reads by path.

    Resolved as the native server resolves it (vLLM's ``load_chat_template``):
    a file, a vLLM built-in template name, or literal Jinja. Named by content,
    so every engine on a node, and every restart, shares one file.
    """
    if chat_template is None:
        return None
    content = load_chat_template(chat_template)
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"nemo_rl_chat_template_{digest}.jinja")
    if not os.path.exists(path):
        # Written aside and renamed, so a concurrent reader never sees a part.
        with tempfile.NamedTemporaryFile(
            "w", dir=os.path.dirname(path), suffix=".jinja", delete=False
        ) as template_file:
            template_file.write(content)
        os.replace(template_file.name, path)
    return path


def start_dynamo_endpoint(
    host: "VllmAsyncGenerationWorkerImpl",
    *,
    namespace: str,
    engine_env: dict[str, str],
    leader_worker_idx: int,
) -> asyncio.Task:
    """Register ``host``'s engine with Dynamo and serve it on the running loop.

    The Rust Worker serves until the process gets SIGTERM. It installs a chained
    handler, so Ray's actor handler still runs and the actor exits as usual;
    discovery drops the engine with its process.
    """
    # The Rust runtime reads these when the first Worker is constructed.
    os.environ.update(engine_env)
    kv_events = host.llm.vllm_config.kv_events_config
    publishes = kv_events is not None and kv_events.enable_kv_cache_events
    # The Dynamo functional overlays check the router applied these.
    print(f"  [Dynamo] engine publishes KV events: {publishes}", flush=True)
    worker_args = dynamo_cfg(host.cfg).worker_args
    tool_call_parser, reasoning_parser = dynamo_parsers(host.cfg)
    _check_parser_name(tool_call_parser, get_tool_parser_names(), "tool_call_parser")
    _check_parser_name(
        reasoning_parser, get_reasoning_parser_names(), "reasoning_parser"
    )
    config = WorkerConfig(
        namespace=namespace,
        component=DYNAMO_COMPONENT,
        endpoint=DYNAMO_ENDPOINT,
        model_name=host.model_name,
        served_model_name=host.model_name,
        endpoint_types=",".join(worker_args.endpoint_types),
        discovery_backend=engine_env["DYN_DISCOVERY_BACKEND"],
        request_plane=engine_env["DYN_REQUEST_PLANE"],
        event_plane=engine_env["DYN_EVENT_PLANE"],
        custom_jinja_template=_chat_template_file(dynamo_chat_template(host.cfg)),
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        exclude_tools_when_tool_choice_none=(
            worker_args.exclude_tools_when_tool_choice_none
        ),
        structural_tag_mode="on" if worker_args.enable_structural_tag else "off",
        structural_tag_scope=worker_args.structural_tag_scope,
        structural_tag_schema=worker_args.structural_tag_schema,
    )
    return asyncio.get_running_loop().create_task(
        Worker(NativeVllmEngine(host, leader_worker_idx), config).run()
    )
