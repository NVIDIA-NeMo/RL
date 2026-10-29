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
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

import torch
from dynamo._core import Context
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
    Worker,
    WorkerConfig,
)
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, ZmqSource
from dynamo.llm.exceptions import InvalidArgument
from vllm.inputs import TokensPrompt
from vllm.v1.kv_cache_interface import KVCacheSpecKind
from vllm.v1.metrics.loggers import StatLoggerBase

from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo.managed_runtime import (
    DYNAMO_COMPONENT,
    DYNAMO_ENDPOINT,
)
from nemo_rl.models.generation.dynamo.rollout_client import ROLLOUT_ANNOTATION
from nemo_rl.models.generation.vllm.request_pipeline import rollout_result_to_wire

if TYPE_CHECKING:
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

LOGGER = logging.getLogger(__name__)


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


class NativeVllmEngine(LLMEngine):
    """LLMEngine over a running NeMo RL vLLM worker. Does not own its lifecycle."""

    def __init__(self, host: "VllmAsyncGenerationWorkerImpl", leader_worker_idx: int):
        self._host = host
        self._leader_worker_idx = leader_worker_idx
        self._llm = host.llm
        self._max_model_len = self._llm.vllm_config.model_config.max_model_len

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
        model = self._host.model_name
        return EngineConfig(
            model=model,
            served_model_name=model,
            # No token budget: the engine owns overflow (dynamo-integration.md).
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
        rollout = _rollout_request(request)
        if rollout is None:
            raise InvalidArgument(
                "NeMo RL's vLLM engines serve only NeMo RL rollouts through Dynamo "
                f"(requests annotated {ROLLOUT_ANNOTATION!r})."
            )
        async for chunk in self._generate_rollout(request, rollout, context):
            yield chunk

    async def _generate_rollout(
        self, request: GenerateRequest, rollout: dict[str, Any], context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """One NeMo RL rollout, served by the worker's own ``_serve_rollout``.

        One chunk carries the result, routed experts included, in ``engine_data``;
        the driver assembles the same output ``generate_async`` returns.
        """
        prompt_token_ids = list(request["token_ids"])
        result, final_output = await self._host._serve_rollout(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            prompt_token_ids=prompt_token_ids,
            greedy=rollout["greedy"],
            stop_strings=rollout["stop_strings"],
            request_id=context.id(),
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


def _usage(output: Any, num_output_tokens: int) -> dict[str, Any]:
    prompt_tokens = len(output.prompt_token_ids or ())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": num_output_tokens,
        "total_tokens": prompt_tokens + num_output_tokens,
        "prompt_tokens_details": {"cached_tokens": output.num_cached_tokens or 0},
    }


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
    config = WorkerConfig(
        namespace=namespace,
        component=DYNAMO_COMPONENT,
        endpoint=DYNAMO_ENDPOINT,
        model_name=host.model_name,
        served_model_name=host.model_name,
        # NeMo RL's rollouts go to /v1/completions only.
        endpoint_types="completions",
        discovery_backend=engine_env["DYN_DISCOVERY_BACKEND"],
        request_plane=engine_env["DYN_REQUEST_PLANE"],
        event_plane=engine_env["DYN_EVENT_PLANE"],
    )
    return asyncio.get_running_loop().create_task(
        Worker(NativeVllmEngine(host, leader_worker_idx), config).run()
    )
