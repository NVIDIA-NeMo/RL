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

"""Serve a NeMo RL generation worker's own engine through Dynamo's backend SDK.

``DynamoEngineAdapter`` is the Dynamo side, written once for every backend: the
request kinds served, how errors reach the driver, and the endpoint's
registration. A backend implements four methods over its worker's engine
(``serve_rollout``, ``start``, ``abort`` and ``kv_event_sources``);
``vllm_adapter.DynamoVllmAdapter`` is the vLLM one. The engine only ever receives
token IDs, on ``/v1/completions``. This module imports no generation backend.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import abstractmethod
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, Optional

from dynamo._core import Context
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    Worker,
    WorkerConfig,
)
from dynamo.common.backend import worker as dynamo_worker
from dynamo.common.backend.publisher import KvEventSource
from dynamo.llm.exceptions import InvalidArgument

from nemo_rl.models.generation.dynamo.managed_runtime import (
    DYNAMO_COMPONENT,
    DYNAMO_ENDPOINT,
)
from nemo_rl.models.generation.dynamo.rollout_client import ROLLOUT_ANNOTATION
from nemo_rl.models.generation.rollout_result import (
    RolloutResult,
    rollout_result_to_wire,
)

# Serves one request kind: the request, its annotation's value, and the context.
RequestHandler = Callable[
    [GenerateRequest, str, Context], AsyncGenerator[GenerateChunk, None]
]


@dataclass(frozen=True)
class ServedRollout:
    """A backend's answer to one rollout."""

    result: RolloutResult
    # The OpenAI finish reason the frontend reports.
    finish_reason: str
    # The OpenAI completion usage; None when no new token fit.
    usage: Optional[dict[str, Any]]


def _keep_host_logging(*args: object, **kwargs: object) -> None:
    """Replaces Dynamo's ``configure_dynamo_logging`` (ai-dynamo/dynamo#18447).

    ``Worker.run()`` calls it first, and it would replace this process's Python
    logging: NeMo RL's root handler and level, and vLLM's handlers.
    """


def _annotation_value(request: GenerateRequest, prefix: str) -> Optional[str]:
    """The value of the first ``<prefix><value>`` annotation, or None."""
    for annotation in request.get("annotations") or ():
        if annotation.startswith(prefix):
            return annotation[len(prefix) :]
    return None


class DynamoEngineAdapter(LLMEngine):
    """Serves one NeMo RL generation worker through Dynamo. The worker owns the engine."""

    def __init__(self, *, leader_worker_idx: int):
        # The worker's index in its group; rollouts report it, as
        # generate_async results do.
        self._leader_worker_idx = leader_worker_idx

    @classmethod
    async def from_args(cls, argv=None):
        raise NotImplementedError("constructed in-process by the generation worker")

    def _handlers(self) -> dict[str, RequestHandler]:
        """The request kinds served, keyed by the annotation prefix that marks them."""
        return {ROLLOUT_ANNOTATION: self._generate_rollout}

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        handlers = self._handlers()
        for prefix, handler in handlers.items():
            value = _annotation_value(request, prefix)
            if value is not None:
                break
        else:
            raise InvalidArgument(
                "NeMo RL's engines serve only NeMo RL rollouts through Dynamo "
                f"(requests annotated {', '.join(map(repr, handlers))})."
            )
        try:
            async for chunk in handler(request, value, context):
                yield chunk
        except RuntimeError as error:
            # A 5xx reaches the driver as "Internal server error" and is re-sent;
            # a 4xx keeps the cause, and the rollout retry policy still applies.
            raise InvalidArgument(f"{type(error).__name__}: {error}") from error

    async def _generate_rollout(
        self, request: GenerateRequest, rollout_spec: str, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """One NeMo RL rollout, served by the backend's ``serve_rollout``.

        One chunk carries the result, routed experts included, in ``engine_data``;
        the driver assembles the same output ``generate_async`` returns.
        """
        rollout = json.loads(rollout_spec)
        served = await self.serve_rollout(
            list(request["token_ids"]),
            greedy=rollout["greedy"],
            stop_strings=rollout["stop_strings"],
            request_id=context.id(),
        )
        chunk: GenerateChunk = {
            "token_ids": served.result.token_ids,
            "index": 0,
            "finish_reason": served.finish_reason,
            "engine_data": {
                "rollout": rollout_result_to_wire(served.result),
                "gen_leader_worker_idx": self._leader_worker_idx,
            },
        }
        if served.usage is not None:
            chunk["completion_usage"] = served.usage
        yield chunk

    @abstractmethod
    async def serve_rollout(
        self,
        prompt_token_ids: list[int],
        *,
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
    ) -> ServedRollout:
        """Serve one rollout exactly as the worker's native ``generate_async`` does."""

    @abstractmethod
    async def start(self, worker_id: int) -> EngineConfig:
        """Register the engine's context length, KV block size and capacity."""

    @abstractmethod
    async def abort(self, context: Context) -> None:
        """Cancel the request ``context`` names; Dynamo calls it on disconnect."""

    @abstractmethod
    async def kv_event_sources(self) -> list[KvEventSource]:
        """Where the engine publishes the KV-cache events Dynamo's KV router indexes."""

    async def cleanup(self) -> None:
        """Nothing to release: the worker owns its engine and shuts it down."""


def start_dynamo_endpoint(
    adapter: DynamoEngineAdapter,
    *,
    model_path: str,
    served_model_name: str,
    namespace: str,
    engine_env: dict[str, str],
) -> asyncio.Task:
    """Register ``adapter``'s engine with Dynamo and serve it on the running loop.

    ``model_path`` is a local directory with the model's config and tokenizer
    files; the engine is served under ``served_model_name``.

    The Rust Worker serves until the process gets SIGTERM. It installs a chained
    handler, so Ray's actor handler still runs and the actor exits as usual;
    discovery drops the engine with its process.
    """
    # The Rust runtime reads these when the first Worker is constructed.
    os.environ.update(engine_env)
    # The generation worker keeps its own logging until WorkerConfig can opt out.
    dynamo_worker.configure_dynamo_logging = _keep_host_logging
    config = WorkerConfig(
        namespace=namespace,
        component=DYNAMO_COMPONENT,
        endpoint=DYNAMO_ENDPOINT,
        model_name=model_path,
        served_model_name=served_model_name,
        # NeMo RL's rollouts go to /v1/completions only.
        endpoint_types="completions",
        discovery_backend=engine_env["DYN_DISCOVERY_BACKEND"],
        request_plane=engine_env["DYN_REQUEST_PLANE"],
        event_plane=engine_env["DYN_EVENT_PLANE"],
    )
    return asyncio.get_running_loop().create_task(Worker(adapter, config).run())
