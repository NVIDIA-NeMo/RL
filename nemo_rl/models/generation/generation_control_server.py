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
"""Control plane server + DP shard router for disaggregated vLLM generation.

Wraps a local VllmGeneration instance and exposes:
  A) Control plane — weight sync triggers, lifecycle management, collective init.
  B) DP Shard Router — reverse proxy for vLLM's OpenAI-compatible HTTP API with
     intelligent routing across data-parallel shards.

Runs on a single port (default 8089). Training cluster discovers this one URL.
The training client sends generation requests directly to per-shard
``/v1/completions`` endpoints (round-robin); the reverse proxy here is for
external OpenAI-compatible clients (e.g. NemoGym).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import traceback
from enum import Enum
from typing import Any, Optional

import aiohttp
import ray
import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from nemo_rl.models.generation.interfaces import GenerationInterface


class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_PENDING = "least_pending"
    PREFIX_HASH = "prefix_hash"


class GenerationControlServer:
    """FastAPI server wrapping a VllmGeneration for disaggregated mode."""

    def __init__(
        self,
        generation: GenerationInterface,
        port: int = 8089,
        routing_strategy: str = "round_robin",
    ):
        self.generation = generation
        self.port = port
        self.routing_strategy = RoutingStrategy(routing_strategy)

        self.shard_urls: list[str] = [
            url
            for url in getattr(generation, "dp_openai_server_base_urls", [])
            if url is not None
        ]

        self._rr_index = 0
        self._pending_counts: dict[int, int] = {i: 0 for i in range(len(self.shard_urls))}
        self._lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None

        self._app = self._build_app()
        self._server_thread: Optional[threading.Thread] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create a persistent aiohttp session for the router."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600)
            )
        return self._session

    def _build_app(self):
        app = FastAPI(title="Generation Control Server")

        # =====================================================================
        # Control plane endpoints
        # =====================================================================

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/config")
        async def get_config():
            return dict(self.generation.cfg)

        @app.get("/dp_openai_server_base_urls")
        async def get_dp_urls():
            return self.shard_urls

        # All control plane endpoints use run_in_executor to avoid blocking
        # the uvicorn event loop. Blocking ray.get() or synchronous GPU operations
        # in async handlers deadlocks the NCCL warmup broadcast.
        async def _run_blocking(fn, *args):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, fn, *args)

        @app.post("/init_collective")
        async def init_collective(request: Request):
            body = await request.json()
            try:
                def _do():
                    futures = self.generation.init_collective(
                        ip=body["ip"],
                        port=body["port"],
                        world_size=body["world_size"],
                        train_world_size=body["train_world_size"],
                    )
                    ray.get(futures)
                await _run_blocking(_do)
                return {"success": True}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

        @app.post("/reset_collective")
        async def reset_collective():
            """Tear down the weight-sync NCCL group so a new training run can re-init.

            Idempotent — safe to call when no collective is currently held.
            """
            try:
                def _do():
                    futures = self.generation.reset_collective()
                    if futures:
                        ray.get(futures)
                await _run_blocking(_do)
                return {"success": True}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

        @app.post("/update_weights_from_collective")
        async def update_weights_from_collective():
            try:
                def _do():
                    futures = self.generation.update_weights_from_collective()
                    results = ray.get(futures)
                    success = all(r for r in results if r is not None)
                    if not success:
                        raise RuntimeError(
                            f"One or more workers failed to update weights. Results: {results}"
                        )
                    return success
                success = await _run_blocking(_do)
                return {"success": success}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

        @app.post("/prepare_for_generation")
        async def prepare_for_generation():
            try:
                result = await _run_blocking(self.generation.prepare_for_generation)
                return {"success": bool(result)}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

        @app.post("/finish_generation")
        async def finish_generation():
            try:
                result = await _run_blocking(self.generation.finish_generation)
                return {"success": bool(result)}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

        @app.post("/prepare_refit_info")
        async def prepare_refit_info(request: Request):
            try:
                body_bytes = await request.body()
                def _do():
                    state_dict_info = torch.load(io.BytesIO(body_bytes), weights_only=False)
                    self.generation.prepare_refit_info(state_dict_info)
                await _run_blocking(_do)
                return {"success": True}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

        @app.post("/invalidate_kv_cache")
        async def invalidate_kv_cache():
            try:
                result = await _run_blocking(self.generation.invalidate_kv_cache)
                return {"success": bool(result)}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

        @app.post("/clear_logger_metrics")
        async def clear_logger_metrics():
            self.generation.clear_logger_metrics()
            return {"success": True}

        @app.get("/get_logger_metrics")
        async def get_logger_metrics():
            return self.generation.get_logger_metrics()

        @app.post("/snapshot_step_metrics")
        async def snapshot_step_metrics():
            if hasattr(self.generation, "snapshot_step_metrics"):
                self.generation.snapshot_step_metrics()
            return {"success": True}

        @app.get("/get_step_metrics")
        async def get_step_metrics():
            if hasattr(self.generation, "get_step_metrics"):
                return self.generation.get_step_metrics()
            return {}

        # =====================================================================
        # DP Shard Router (OpenAI-compatible reverse proxy for external clients)
        # =====================================================================

        @app.post("/v1/completions")
        async def route_completions(request: Request):
            return await self._route_request(request, "/v1/completions")

        @app.post("/v1/chat/completions")
        async def route_chat_completions(request: Request):
            return await self._route_request(request, "/v1/chat/completions")

        @app.post("/tokenize")
        async def route_tokenize(request: Request):
            return await self._route_request(request, "/tokenize")

        return app

    # =====================================================================
    # Router implementation
    # =====================================================================

    def _select_shard(self, body: dict | None = None) -> int:
        """Select a DP shard index based on the routing strategy.

        Safe without a lock: FastAPI runs on a single-threaded asyncio loop,
        so _rr_index and _pending_counts mutations are non-concurrent.
        """
        if len(self.shard_urls) <= 1:
            return 0

        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self._rr_index
            self._rr_index = (self._rr_index + 1) % len(self.shard_urls)
            return idx

        if self.routing_strategy == RoutingStrategy.LEAST_PENDING:
            return min(self._pending_counts, key=self._pending_counts.get)

        if self.routing_strategy == RoutingStrategy.PREFIX_HASH:
            prompt_ids = body.get("prompt_token_ids") if body else None
            if prompt_ids:
                prefix = tuple(prompt_ids[:128])
                h = int(hashlib.md5(str(prefix).encode()).hexdigest(), 16)
                return h % len(self.shard_urls)
            return min(self._pending_counts, key=self._pending_counts.get)

        return 0

    async def _route_request(self, request: Request, path: str):
        """Forward an HTTP request to a selected DP shard."""
        body_bytes = await request.body()

        body_dict = None
        try:
            body_dict = json.loads(body_bytes)
        except Exception:
            pass

        shard_idx = self._select_shard(body_dict)
        shard_base_url = self.shard_urls[shard_idx]
        base = shard_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        target_url = f"{base}{path}"

        self._pending_counts[shard_idx] = self._pending_counts.get(shard_idx, 0) + 1
        try:
            is_streaming = body_dict and body_dict.get("stream", False)
            session = await self._get_session()

            async with session.post(
                target_url,
                data=body_bytes,
                headers={"Content-Type": request.headers.get("content-type", "application/json")},
            ) as resp:
                if is_streaming:
                    async def stream_response():
                        async for chunk in resp.content.iter_any():
                            yield chunk

                    return StreamingResponse(
                        stream_response(),
                        status_code=resp.status,
                        media_type=resp.headers.get("content-type", "text/event-stream"),
                    )
                else:
                    response_body = await resp.read()
                    return Response(
                        content=response_body,
                        status_code=resp.status,
                        media_type=resp.headers.get("content-type", "application/json"),
                    )
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=502, content={"error": f"Router failed to reach shard {shard_idx}: {e}"})
        finally:
            self._pending_counts[shard_idx] = max(0, self._pending_counts.get(shard_idx, 1) - 1)

    # =====================================================================
    # Server lifecycle
    # =====================================================================

    def start(self) -> None:
        """Start the server in a background thread."""
        import uvicorn

        config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, timeout_keep_alive=120)
        server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=server.run, daemon=True)
        self._server_thread.start()
        print(
            f"GenerationControlServer started on port {self.port} "
            f"(routing={self.routing_strategy.value}, shards={len(self.shard_urls)})"
        )

    def get_app(self):
        """Return the FastAPI app (for testing or custom server setup)."""
        return self._app
