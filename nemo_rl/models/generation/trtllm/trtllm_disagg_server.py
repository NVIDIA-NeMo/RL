
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
"""One replica's disaggregation front-end, backed by TRT-LLM's
``OpenAIDisaggServer``.

Started the same way as :mod:`trtllm_http_server`: a uvicorn app in a daemon
thread, returning the URL NeMo-Gym will talk to. That server owns everything
below the replica boundary — which context engine runs the prefill, which
generation engine runs the decode, and the KV handshake between them. NeMo RL
only hands it the two address pools and the router policies.
"""

import logging
import threading
import time
from typing import Any, Optional

import ray

logger = logging.getLogger(__name__)

__all__ = [
    "DisaggServerActor",
    "DisaggServerActorImpl",
    "build_config",
    "start_server",
    "wait_ready",
]


def build_config(
    ctx_addrs: list[tuple[str, int]],
    gen_addrs: list[tuple[str, int]],
    *,
    node_id: int,
    ctx_router: str,
    gen_router: str,
    gen_tokids_ctxbytes: bool = False,
    gen_strip_message_history: bool = False,
) -> Any:
    """Assemble the ``DisaggServerConfig`` for one replica.

    Args:
        ctx_addrs / gen_addrs: ``(hostname, port)`` of every engine in this
            replica. An address is the only thing the disagg server is given
            about an engine.
        node_id: Must be distinct per replica. TRT-LLM's default is
            ``uuid.getnode() % 256``, documented as assuming a single disagg
            server per machine, and we run one per replica.
        ctx_router: Stateful, so a trajectory's turns keep reaching the context
            engine holding its prefix.
        gen_router: Stateless, so placement stays local to this replica and no
            coordinator process is needed.
    """
    from tensorrt_llm.llmapi.disagg_utils import (
        CtxGenServerConfig,
        DisaggServerConfig,
        RouterConfig,
    )

    server_configs = [
        CtxGenServerConfig(type="ctx", hostname=host, port=port)
        for host, port in ctx_addrs
    ] + [
        CtxGenServerConfig(type="gen", hostname=host, port=port)
        for host, port in gen_addrs
    ]

    config = DisaggServerConfig(
        server_configs=server_configs,
        ctx_router_config=RouterConfig(type=ctx_router),
        gen_router_config=RouterConfig(type=gen_router),
        node_id=node_id,
    )
    # Post-construction assignment, same as TRT-LLM's own YAML parser
    # (llmapi/disagg_utils.py extract_disagg_cfg): thins the gen leg so the
    # relay stops re-serializing a 30k-int id array and the full history.
    config.gen_tokids_ctxbytes = gen_tokids_ctxbytes
    config.gen_strip_message_history = gen_strip_message_history
    return config


# Request fields clients send for vLLM that TRT-LLM's ChatCompletionRequest
# does not declare. Its models are extra="forbid", so leaving them in means a
# 422/400 on every request. `required_prefix_token_ids` is the vLLM server's
# prefix-override extension (used by the MLPerf warmup); the TRT-LLM engine
# adapter derives the on-policy prefix from the assistant messages instead and
# ignores the top-level field, so dropping it here keeps both paths identical.
_GYM_ONLY_REQUEST_FIELDS = (
    "return_tokens_as_token_ids",
    "return_token_ids",
    "required_prefix_token_ids",
)

# Prefix vLLM uses when asked to report tokens as ids, which NeMo-Gym parses.
_TOKEN_ID_PREFIX = "token_id:"

# Paths whose bodies are validated against TRT-LLM's extra="forbid" models.
_ADAPTED_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})


class _DropGymOnlyRequestFields:
    """ASGI middleware stripping the vLLM-only fields from request bodies.

    Deliberately raw ASGI rather than a Starlette ``BaseHTTPMiddleware``: that
    class passes the downstream app its own captured receive channel, so
    reassigning ``request._receive`` never reaches FastAPI's validation and the
    request is rejected anyway. Replacing ``receive`` here is what the route
    actually reads.
    """

    def __init__(self, app: Any, tokenize_fn: Any = None) -> None:
        self.app = app
        # Phase-2 frontend tokenize: async callable(payload) -> b64 ids or
        # None. Runs here because this middleware already pays the
        # loads/dumps round trip for every body.
        self._tokenize_fn = tokenize_fn

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") not in _ADAPTED_PATHS:
            await self.app(scope, receive, send)
            return


        import json

        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)

        try:
            payload = json.loads(body)
        except ValueError:
            payload = None

        mutated = False
        if isinstance(payload, dict) and any(
            field in payload for field in _GYM_ONLY_REQUEST_FIELDS
        ):
            for field in _GYM_ONLY_REQUEST_FIELDS:
                payload.pop(field, None)
            mutated = True
        if (
            self._tokenize_fn is not None
            and isinstance(payload, dict)
            and payload.get("messages")
            and "prompt_token_ids_b64" not in payload
            and "prompt_token_ids" not in payload
            and payload.get("disaggregated_params") is None
        ):
            # Inbound gym request (the engine-facing legs carry
            # disaggregated_params and never traverse this app). Attach the
            # rendered+spliced prompt ids so the ctx adapter skips its
            # chat-template work entirely; prompt_token_ids_b64 is a declared
            # request field end-to-end.
            try:
                b64 = await self._tokenize_fn(payload)
            except Exception:  # noqa: BLE001 - fall back to adapter-side tokenize
                logger.exception(
                    "frontend tokenize failed; falling back to adapter-side"
                )
                b64 = None
            if b64 is not None:
                payload["prompt_token_ids_b64"] = b64
                mutated = True
        if mutated:
            body = json.dumps(payload).encode()
        # Content-Length must follow the body when it changed; a stale value
        # makes any proxy in front of this server truncate or hang.
        headers = [
            (name, value)
            for name, value in scope["headers"]
            if name.lower() != b"content-length" or not mutated
        ]
        if mutated:
            headers.append((b"content-length", str(len(body)).encode()))
        scope = {**scope, "headers": headers}

        delivered = False

        async def _receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, _receive, send)


def _build_adaptor_class() -> type:
    """Build ``OpenAIDisaggServerAdaptor`` lazily.

    Deferred so importing this module does not require tensorrt_llm.
    """
    import aiohttp
    from fastapi import HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
    from tensorrt_llm.serve.openai_protocol import UCompletionRequest

    class OpenAIDisaggServerAdaptor(OpenAIDisaggServer):
        """``OpenAIDisaggServer`` speaking NeMo-Gym's dialect at both edges.

        The disagg server validates strictly in both directions -- requests into
        ``ChatCompletionRequest``, engine responses into
        ``ChatCompletionResponse``, both ``extra="forbid"``. NeMo-Gym, which was
        written against vLLM, sends one request field TRT-LLM does not declare
        and reads rollout fields that are not part of the OpenAI schema. This
        subclass reconciles the two without touching either side:

        * inbound -- drop the vLLM-only request fields before FastAPI validates
        * outbound -- re-attach the rollout fields Gym reads off the message

        Note the outbound translation only moves fields that already survived
        the engine -> disagg-server hop. It cannot rescue a field the engine
        emitted that ``ChatCompletionResponse`` does not declare; those have to
        travel in a declared field (see ``_generation_token_ids``).
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._tokenize_fn = kwargs.pop("tokenize_fn", None)
            super().__init__(*args, **kwargs)
            self._uvicorn: Any = None
            self._install_request_adaptor()

        async def __call__(self, host: str, port: int, sockets: Any = None) -> None:
            """Serve, keeping a handle on uvicorn so shutdown can stop it.

            The base class builds its ``uvicorn.Server`` as a local, so there is
            nothing to signal ``should_exit`` on otherwise.
            """
            import uvicorn

            config = uvicorn.Config(
                self.app, host=host, port=port, log_level="info", timeout_keep_alive=10
            )
            self._uvicorn = uvicorn.Server(config)
            await self._uvicorn.serve(sockets=sockets)

        def request_shutdown(self) -> None:
            if self._uvicorn is not None:
                self._uvicorn.should_exit = True

        # -------------------------------------------------------------- #
        #  Inbound
        # -------------------------------------------------------------- #

        def _install_request_adaptor(self) -> None:
            # Pure ASGI, not @app.middleware("http"): BaseHTTPMiddleware hands
            # the downstream app the receive channel it captured itself, so
            # rewriting request._receive there is invisible to FastAPI's
            # validation and every request still fails with extra_forbidden.
            # Wrapping receive at the ASGI layer is what actually replaces the
            # body the route sees.
            self.app.add_middleware(
                _DropGymOnlyRequestFields, tokenize_fn=self._tokenize_fn
            )

        # -------------------------------------------------------------- #
        #  Errors
        # -------------------------------------------------------------- #

        def _handle_exception(self, exception: BaseException) -> None:
            """Pass a downstream 4xx through instead of masking it as a 500.

            The base implementation only re-raises ``HTTPException``; a 4xx
            from a context/generation engine arrives as an
            ``aiohttp.ClientResponseError`` and falls into the catch-all that
            turns it into ``500 Internal server error``. The aggregated server
            returns that same rejection to the caller as a 4xx, so without this
            a deterministic client error -- most commonly the
            ``context length exceeded`` guard in ``trtllm_http_server`` -- looks
            like a server fault under disaggregation only, and Gym's masking
            statistics stop being comparable between the two paths.
            """
            if (
                isinstance(exception, aiohttp.ClientResponseError)
                and 400 <= exception.status < 500
            ):
                self._perf_metrics_collector.http_exceptions.inc()
                raise HTTPException(
                    status_code=exception.status, detail=exception.message
                )
            super()._handle_exception(exception)

        # -------------------------------------------------------------- #
        #  Outbound
        # -------------------------------------------------------------- #

        def _wrap_entry_point(
            self, entry_point: Any, request_type: type = UCompletionRequest
        ) -> Any:
            inner = super()._wrap_entry_point(entry_point, request_type)

            async def wrapper(req: request_type, raw_req: Request) -> Response:  # type: ignore[valid-type]
                response = await inner(req, raw_req)
                if req.stream or not isinstance(response, JSONResponse):
                    return response
                return self._attach_rollout_fields(response, raw_req)

            return wrapper

        @staticmethod
        def _generation_token_ids(choice: dict[str, Any]) -> Optional[list[int]]:
            """Generated token ids, from whichever declared field carries them.

            ``ChatCompletionResponseChoice.token_ids`` is the clean home, but it
            does not exist upstream yet. Until it does they ride in
            ``logprobs.content[].token`` using vLLM's ``token_id:N`` encoding,
            which is a declared string field and therefore survives the hop.
            """
            if choice.get("token_ids"):
                return list(choice["token_ids"])

            content = (choice.get("logprobs") or {}).get("content") or []
            ids = []
            for entry in content:
                token = entry.get("token") or ""
                if not token.startswith(_TOKEN_ID_PREFIX):
                    return None
                ids.append(int(token[len(_TOKEN_ID_PREFIX) :]))
            return ids or None

        def _attach_rollout_fields(
            self, response: JSONResponse, raw_req: "Request | None" = None
        ) -> JSONResponse:
            """Re-attach the fields NeMo-Gym reads off ``choices[].message``."""
            import json

            payload = json.loads(response.body)

            choices = payload.get("choices") or []
            if not choices:
                return JSONResponse(
                    content=payload, status_code=response.status_code
                )

            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                return JSONResponse(
                    content=payload, status_code=response.status_code
                )

            if payload.get("prompt_token_ids") is not None:
                message["prompt_token_ids"] = payload["prompt_token_ids"]

            token_ids = self._generation_token_ids(choice)
            if token_ids is not None:
                message["generation_token_ids"] = token_ids

            content = (choice.get("logprobs") or {}).get("content")
            if content:
                message["generation_log_probs"] = [
                    entry.get("logprob") for entry in content
                ]

            return JSONResponse(content=payload, status_code=response.status_code)

    return OpenAIDisaggServerAdaptor


def _build_frontend_tokenizer(
    model_name: str, server_template_kwargs: dict[str, Any]
) -> Any:
    """Async ``payload -> b64 prompt ids`` using the adapters' exact pipeline.

    Loads the tokenizer and HF config the same way the engine adapters do
    (``AutoTokenizer/AutoConfig.from_pretrained(model_name,
    trust_remote_code=True)``) and renders through the shared
    ``build_spliced_prompt_ids`` -- the ids must be bit-identical to what the
    ctx adapter would compute, since a divergence silently trains on
    off-policy tokens (the adapter shadow-validates a sample under
    NRL_TRTLLM_TOKENIZE_SHADOW_RATE).
    """
    import base64

    import numpy as np
    from transformers import AutoConfig, AutoTokenizer

    from nemo_rl.models.generation.trtllm.trtllm_http_server import (
        build_spliced_prompt_ids,
    )

    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    async def _tokenize(payload: dict[str, Any]) -> str:
        # Normalize through the SAME pydantic model and dump flags the disagg
        # client uses toward the engines (model_dump(mode="json",
        # exclude_unset=True), openai_client.py). The ctx adapter renders the
        # POST-VALIDATION dicts -- pydantic inserts defaults and fixes key
        # order (e.g., tools entries become {"type": "function", "function":
        # ...}), and the chat template serializes them verbatim, so rendering
        # the raw gym payload instead diverges at the tools JSON on turn-1
        # prompts (caught by shadow validation).
        norm = ChatCompletionRequest(**payload).model_dump(
            mode="json", exclude_unset=True
        )
        per_request = norm.get("chat_template_kwargs") or {}
        effective = {**server_template_kwargs, **per_request}
        ids = await build_spliced_prompt_ids(
            norm.get("messages") or [],
            norm.get("tools"),
            tokenizer,
            model_config,
            effective,
        )
        return base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()

    return _tokenize


def start_server(
    config: Any,
    host: str = "0.0.0.0",
    port: int = 0,
    req_timeout_secs: int = 1800,
    tokenize_fn: Any = None,
) -> "tuple[threading.Thread, str, Any]":
    """Start the disagg server in a daemon thread and return (thread, base_url, server)."""
    import asyncio

    from nemo_rl.distributed.virtual_cluster import (
        _get_free_port_local,
        _get_node_ip_local,
    )

    if port == 0:
        port = _get_free_port_local()

    node_ip = _get_node_ip_local()
    base_url = f"http://{node_ip}:{port}/v1"

    # OpenAIDisaggServer.register_routes() builds a prometheus
    # MultiProcessCollector, which raises unless PROMETHEUS_MULTIPROC_DIR points
    # at a real directory. TRT-LLM's own entrypoint calls this helper first
    # (tensorrt_llm/commands/serve.py); we construct the server directly, so we
    # have to. It keeps the TemporaryDirectory alive in a module global, which
    # is also what stops it from being collected while the server runs.
    from tensorrt_llm._utils import set_prometheus_multiproc_dir

    set_prometheus_multiproc_dir()

    # coordinator_url=None: this server owns its routing state in-process.
    # Replicas are disjoint so there is nothing to coordinate across them, and
    # a stateless generation router places locally anyway.
    server = _build_adaptor_class()(
        config,
        req_timeout_secs=req_timeout_secs,
        coordinator_url=None,
        tokenize_fn=tokenize_fn,
    )

    def _run() -> None:
        # OpenAIDisaggServer.__call__ is a coroutine that runs uvicorn, so the
        # thread needs its own event loop. Frontend-saturation mitigations
        # mirror TRT-LLM's own serve entrypoint: uvloop when available, and
        # optionally GC off (tekit measured negligible growth over 200k
        # requests; the periodic gen-2 collections otherwise stall the whole
        # relay loop).
        import os

        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass
        if os.environ.get("TRTLLM_DISAGG_SERVER_DISABLE_GC", "0") == "1":
            import gc

            gc.disable()
        asyncio.run(server(host, port))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    logger.info("TRT-LLM disagg server starting on %s", base_url)

    return thread, base_url, server


class DisaggServerActorImpl:
    """Hosts one replica's disagg server.

    Its own process rather than an engine worker's: this is the request hot path
    for the whole replica, and sharing a process with an engine would couple the
    replica's routing latency to that one engine's load -- two uvicorn loops plus
    the engine's generate thread pool contending for a single GIL. It holds no
    GPUs, so the isolation is cheap.

    Held separately from the ``@ray.remote``-wrapped :class:`DisaggServerActor`
    so it can be exercised without Ray.
    """

    def __init__(
        self,
        replica_idx: int,
        frontend_idx: int = 0,
        num_frontends: int = 1,
        port: int = 0,
        serve_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self._replica_idx = replica_idx
        self._frontend_idx = frontend_idx
        self._num_frontends = num_frontends
        self._port = port
        self._thread = None
        self._server = None
        self._base_url: Optional[str] = None
        # Serving from the constructor is what makes crash recovery work: Ray
        # re-runs __init__ on actor restart but never replays method calls, so
        # with a hard node pin and a deterministic port the restarted frontend
        # comes back at the SAME URL and Gym's sticky clients recover after
        # their 5xx retries. The empty router trie is a one-time cache-miss.
        if serve_args is not None:
            self._start_serving(**serve_args)

    def _start_serving(
        self,
        ctx_addrs: list[tuple[str, int]],
        gen_addrs: list[tuple[str, int]],
        *,
        ctx_router: str,
        gen_router: str,
        gen_tokids_ctxbytes: bool = False,
        gen_strip_message_history: bool = False,
        frontend_tokenize: bool = False,
        model_name: Optional[str] = None,
        default_chat_template_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        # node_id keys the snowflake request-id mint (process_id is hardwired
        # to 0 without a coordinator, and time.monotonic() shares an origin
        # across processes on one node) -- distinct node_id per frontend is
        # what keeps ctx_request_ids disjoint. It also namespaces the router's
        # implicit conversation-id counter.
        config = build_config(
            ctx_addrs,
            gen_addrs,
            node_id=self._replica_idx * self._num_frontends + self._frontend_idx,
            ctx_router=ctx_router,
            gen_router=gen_router,
            gen_tokids_ctxbytes=gen_tokids_ctxbytes,
            gen_strip_message_history=gen_strip_message_history,
        )
        tokenize_fn = None
        if frontend_tokenize:
            assert model_name, "frontend_tokenize requires the model name"
            tokenize_fn = _build_frontend_tokenizer(
                model_name, default_chat_template_kwargs or {}
            )
        self._thread, self._base_url, self._server = start_server(
            config, port=self._port, tokenize_fn=tokenize_fn
        )

    def start(
        self,
        ctx_addrs: Optional[list[tuple[str, int]]] = None,
        gen_addrs: Optional[list[tuple[str, int]]] = None,
        *,
        ctx_router: str = "conversation",
        gen_router: str = "load_balancing",
        gen_tokids_ctxbytes: bool = False,
        gen_strip_message_history: bool = False,
    ) -> str:
        """Ensure the server is up and return the URL NeMo-Gym will talk to."""
        if self._base_url is None:
            assert ctx_addrs is not None and gen_addrs is not None, (
                "start() needs the address pools unless they were passed to "
                "the constructor via serve_args"
            )
            self._start_serving(
                ctx_addrs,
                gen_addrs,
                ctx_router=ctx_router,
                gen_router=gen_router,
                gen_tokids_ctxbytes=gen_tokids_ctxbytes,
                gen_strip_message_history=gen_strip_message_history,
            )
        wait_ready(self._base_url)

        logger.info(
            "disagg frontend %d/%d for replica %d ready at %s",
            self._frontend_idx,
            self._num_frontends,
            self._replica_idx,
            self._base_url,
        )
        return self._base_url

    def base_url(self) -> Optional[str]:
        return self._base_url

    def shutdown(self) -> bool:
        if self._server is not None:
            self._server.request_shutdown()
        self._server = None
        self._thread = None
        self._base_url = None
        return True


def wait_ready(base_url: str, timeout_s: float = 300.0) -> None:
    """Block until the disagg server answers /health.

    It reaches out to every engine in its pools on startup, so readiness lags
    the thread start by more than a socket bind.
    """
    import time

    import requests

    health = base_url.rsplit("/v1", 1)[0] + "/health"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if requests.get(health, timeout=5).status_code == 200:
                return
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"disagg server at {base_url} did not become ready within {timeout_s}s"
            )
        time.sleep(0.5)


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class DisaggServerActor(DisaggServerActorImpl):
    """Ray actor wrapper around :class:`DisaggServerActorImpl`."""

    pass
