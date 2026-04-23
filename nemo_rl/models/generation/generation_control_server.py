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
"""Control-plane server for disaggregated vLLM generation.

Wraps a local VllmGeneration instance and exposes:
  * Weight sync & lifecycle — init/reset/update_weights_from_collective,
    prepare_for_generation, finish_generation, prepare_refit_info,
    invalidate_kv_cache.
  * Shard discovery — ``GET /dp_openai_server_base_urls`` returns the list of
    per-shard OpenAI-compatible URLs that clients (training, NemoGym)
    round-robin across directly.

Runs on a single port (default 8089). Training discovers this one URL, pulls
the DP shard list once, and sends generation requests straight to the shards
via OpenAI ``/v1/completions``. The control server never sits on the
generation hot path.
"""

from __future__ import annotations

import asyncio
import io
import threading
import traceback
from typing import Optional

import ray
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nemo_rl.models.generation.interfaces import GenerationInterface


class GenerationControlServer:
    """FastAPI server wrapping a VllmGeneration for disaggregated mode."""

    def __init__(
        self,
        generation: GenerationInterface,
        port: int = 8089,
    ):
        self.generation = generation
        self.port = port

        self.shard_urls: list[str] = [
            url
            for url in getattr(generation, "dp_openai_server_base_urls", [])
            if url is not None
        ]

        self._app = self._build_app()
        self._server_thread: Optional[threading.Thread] = None

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
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

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
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

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
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/prepare_for_generation")
        async def prepare_for_generation():
            try:
                result = await _run_blocking(self.generation.prepare_for_generation)
                return {"success": bool(result)}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/finish_generation")
        async def finish_generation():
            try:
                result = await _run_blocking(self.generation.finish_generation)
                return {"success": bool(result)}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/prepare_refit_info")
        async def prepare_refit_info(request: Request):
            try:
                body_bytes = await request.body()

                def _do():
                    state_dict_info = torch.load(
                        io.BytesIO(body_bytes), weights_only=True
                    )
                    self.generation.prepare_refit_info(state_dict_info)

                await _run_blocking(_do)
                return {"success": True}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/invalidate_kv_cache")
        async def invalidate_kv_cache():
            try:
                result = await _run_blocking(self.generation.invalidate_kv_cache)
                return {"success": bool(result)}
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

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

        return app

    # =====================================================================
    # Server lifecycle
    # =====================================================================

    def start(self) -> None:
        """Start the server in a background thread."""
        import uvicorn

        config = uvicorn.Config(
            self._app, host="0.0.0.0", port=self.port, timeout_keep_alive=120
        )
        server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=server.run, daemon=True)
        self._server_thread.start()
        print(
            f"GenerationControlServer started on port {self.port} "
            f"(shards={len(self.shard_urls)})"
        )

    def get_app(self):
        """Return the FastAPI app (for testing or custom server setup)."""
        return self._app
