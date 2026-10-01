"""vLLM endpoint plugin for NeMo RL multi-turn prefix-token preservation."""

import asyncio
import os
from argparse import Namespace
from http import HTTPStatus
from importlib.metadata import version
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import State
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
try:  # vLLM >= 0.30 moved the SSE keep-alive helper under entrypoints.serve
    from vllm.entrypoints.serve.utils.sse_keep_alive import with_sse_keep_alive
except ImportError:  # vLLM 0.29
    from vllm.entrypoints.openai.sse_keep_alive import with_sse_keep_alive
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.exceptions import VLLMValidationError

from .capture_runtime import CaptureRuntime
from .renderer_patch import install_renderer_patch
from .route_override import (
    CHAT_COMPLETIONS_PATH,
    remove_existing_chat_completion_routes,
)

PLUGIN_NAME = "nemo_rl_prefix_api"

_EXECUTOR_READY = False


def _ensure_executor() -> None:
    """Widen the event loop's default thread pool once.

    Prefix fetches and commits run in ``asyncio.to_thread``; the default pool
    (min(32, cpus + 4)) caps how many bridge calls an engine can have in flight
    and queues every other request behind them under RL load.
    """
    global _EXECUTOR_READY
    if _EXECUTOR_READY:
        return
    _EXECUTOR_READY = True
    workers = int(os.environ.get("NEMO_RL_PREFIX_PLUGIN_THREADS", "128"))
    if workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nrl-prefix")
        )
PLUGIN_VERSION = "0.3.0"
SUPPORTED_VLLM_VERSIONS = ("0.29.0", "0.30.0")
SUPPORTED_VLLM_VERSION = SUPPORTED_VLLM_VERSIONS[-1]  # newest verified; kept for tooling
CAPABILITY_PATH = "/v1/nemo-rl/prefix-token-capability"


class NeMoRLChatCompletionRequest(ChatCompletionRequest):
    """Stock vLLM request plus the exact token prefix required by NeMo Gym."""

    required_prefix_token_ids: list[int] | None = None
    ng_capture: dict[str, Any] | None = None


class NeMoRLPrefixEndpointPlugin:
    """Extend vLLM's chat route without replacing its serving implementation."""

    name = PLUGIN_NAME
    required_tasks = ("generate",)

    def __init__(self) -> None:
        self._capture_runtime = CaptureRuntime()

    def attach_router(self, app: FastAPI) -> None:
        replaced_route_count = remove_existing_chat_completion_routes(app)
        router = APIRouter()

        @router.get(CAPABILITY_PATH)
        async def prefix_token_capability(raw_request: Request) -> dict[str, Any]:
            return {
                **raw_request.app.state.nemo_rl_prefix_token_capability,
                **self._capture_runtime.capability(),
            }

        @router.post("/v1/nemo-rl/token-capture/configure")
        async def configure_token_capture(
            body: dict[str, Any],
        ) -> dict[str, Any]:
            bridge_url = body.get("bridge_url")
            auth_token = body.get("auth_token")
            if not isinstance(bridge_url, str) or not isinstance(auth_token, str):
                return JSONResponse(
                    content={"error": "bridge_url and auth_token must be strings"},
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            self._capture_runtime.configure(
                bridge_url=bridge_url,
                auth_token=auth_token,
            )
            return {"status": "ok", "token_capture_configured": True}

        @router.post(
            CHAT_COMPLETIONS_PATH,
            dependencies=[Depends(validate_json_request)],
            responses={
                HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
                HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
            },
        )
        @with_cancellation
        @load_aware_call
        async def create_chat_completion(
            request: NeMoRLChatCompletionRequest,
            raw_request: Request,
        ):
            _ensure_executor()
            handler = raw_request.app.state.openai_serving_chat
            if handler is None:
                raise NotImplementedError(
                    "The model does not support Chat Completions API"
                )

            try:
                generator = await handler.create_chat_completion(request, raw_request)
            except BaseException:
                self._capture_runtime.discard(request)
                raise
            if isinstance(generator, ErrorResponse):
                self._capture_runtime.discard(request)
                return JSONResponse(
                    content=generator.model_dump(),
                    status_code=generator.error.code,
                )
            if isinstance(generator, ChatCompletionResponse):
                content = generator.model_dump()
                content = await asyncio.to_thread(
                    self._capture_runtime.finish,
                    request,
                    content,
                )
                metrics_header_format = raw_request.headers.get(
                    "endpoint-load-metrics-format", ""
                )
                return JSONResponse(
                    content=content,
                    headers=metrics_header(metrics_header_format),
                )

            self._capture_runtime.discard(request)
            args = getattr(raw_request.app.state, "args", None)
            keep_alive_interval = getattr(args, "sse_keep_alive_interval", 0)
            return StreamingResponse(
                content=with_sse_keep_alive(generator, float(keep_alive_interval)),
                media_type="text/event-stream",
            )

        # Endpoint plugins are attached after vLLM's stock OpenAI routes.
        # FastAPI uses the first matching route, so the stock route must be
        # removed above rather than leaving this as an unreachable duplicate.
        app.include_router(router)
        app.state.nemo_rl_prefix_stock_chat_routes_replaced = replaced_route_count

    async def init_state(
        self,
        engine_client: Any,
        state: State,
        args: Namespace,
    ) -> None:
        del engine_client, args
        installed_vllm_version = version("vllm")
        if installed_vllm_version not in SUPPORTED_VLLM_VERSIONS:
            raise RuntimeError(
                f"{PLUGIN_NAME} supports vLLM {', '.join(SUPPORTED_VLLM_VERSIONS)}, "
                f"but the server has {installed_vllm_version}"
            )

        serving_chat = getattr(state, "openai_serving_chat", None)
        if serving_chat is None:
            raise RuntimeError(
                f"{PLUGIN_NAME} requires the vLLM Chat Completions serving handler"
            )
        install_renderer_patch(
            serving_chat.online_renderer,
            VLLMValidationError,
            self._capture_runtime,
        )
        state.nemo_rl_prefix_token_capability = {
            "active": True,
            "plugin": PLUGIN_NAME,
            "plugin_version": PLUGIN_VERSION,
            "vllm_version": installed_vllm_version,
            "required_prefix_token_ids": True,
            "ng_capture": True,
            "external_staging": True,
            "return_token_ids": True,
            "stock_chat_routes_replaced": getattr(
                state, "nemo_rl_prefix_stock_chat_routes_replaced", 0
            ),
        }
