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
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
from dynamo._core import Context
from dynamo.common.backend import EngineConfig, LlmRegistration
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, ZmqSource
from vllm.inputs import TokensPrompt
from vllm.v1.kv_cache_interface import KVCacheSpecKind
from vllm.v1.metrics.loggers import StatLoggerBase

from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo.engine_adapter import (
    DynamoEngineAdapter,
    ServedRollout,
)

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
            # No token budget: the engine owns overflow (dynamo-integration.md).
            llm=LlmRegistration(
                context_length=self._max_model_len,
                kv_cache_block_size=block_size,
                total_kv_blocks=cache.num_gpu_blocks,
                max_num_seqs=scheduler.max_num_seqs,
                max_num_batched_tokens=scheduler.max_num_batched_tokens,
            ),
        )

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
