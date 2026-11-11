# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
import contextvars
import copy
import gc
import logging
import os
import secrets
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from nemo_rl.data.captured_media import (
    CapturedMedia,
    CapturedMediaItem,
    MediaCaptureRejected,
    MediaColumnSpec,
    capture_processed_media,
    verified_media_items,
    verify_media_geometry,
)
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.data_plane.tq_token_sink import MediaMetadataIntegrityError
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.experience.route_assembly import boundary_route_rows
from nemo_rl.models.generation.dynamo.config import dynamo_cfg, dynamo_enabled
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.openai_server_utils import (
    PrefixSplice,
)
from nemo_rl.models.generation.prefix_recovery import (
    CaptureRegistry,
    NoPrefixRecovery,
    PrefixRecovery,
    PrefixRecoveryParts,
)
from nemo_rl.models.generation.prefix_recovery.restore import (
    _build_restored_prefix_terminal_output,
    restored_terminal,
)
from nemo_rl.models.generation.prefix_recovery.state import _RequestCaptureState
from nemo_rl.models.generation.rollout_result import RolloutResult
from nemo_rl.models.generation.score_centering import validate_score_centering_request
from nemo_rl.models.generation.vllm.checkpoint_engine import (
    VllmAsyncCheckpointEngineRpcMixin,
)
from nemo_rl.models.generation.vllm.collective_rpc import (
    resolve_collective_rpc_result,
)
from nemo_rl.models.generation.vllm.config import parse_nvfp4_pertoken_rollout
from nemo_rl.models.generation.vllm.progress_reader import VllmProgressReader
from nemo_rl.models.generation.vllm.request_pipeline import (
    rollout_output,
    rollout_result,
)
from nemo_rl.models.generation.vllm.utils import (
    format_prompt_for_vllm_generation,
    validate_rollout_prompt,
)
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker
from nemo_rl.telemetry.setup import shutdown_telemetry
from nemo_rl.utils.routed_experts_codec import routed_experts_tensor_metadata

LOGGER = logging.getLogger(__name__)


# The captured model call the current HTTP request serves. Set when capture
# begins the call, in the request's task, and read by the engine client when the
# same task submits the call to the engine, which links the engine request ID to
# the captured call.
_CAPTURED_MODEL_CALL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "nemo_rl_captured_model_call", default=None
)


# Paths the bearer-token check protects: the OpenAI-compatible routes this module
# registers, which NeMo-Gym's model server calls. The sparse-refit routes the
# receiver registers on the same app carry their own header token
# (vllm_sparse_refit.setup_api_server) and are not covered.
HTTP_SERVER_PROTECTED_PATH_PREFIXES = ("/v1/", "/tokenize")


def install_http_server_api_key_check(app: FastAPI, api_key: str) -> None:
    """Refuse OpenAI-route requests that do not carry ``Authorization: Bearer <api_key>``.

    Registers an HTTP middleware on ``app``. For a request whose path starts with one
    of ``HTTP_SERVER_PROTECTED_PATH_PREFIXES``, the ``Authorization`` header is
    compared in constant time with ``Bearer <api_key>``; a missing or different
    value is answered with HTTP 401 in OpenAI's error envelope and never reaches the
    route. Requests to other paths pass through unchanged.
    """
    expected = f"Bearer {api_key}".encode()

    @app.middleware("http")
    async def _require_bearer_token(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path.startswith(HTTP_SERVER_PROTECTED_PATH_PREFIXES):
            supplied = (request.headers.get("Authorization") or "").encode()
            if not secrets.compare_digest(supplied, expected):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "missing or invalid bearer token",
                            "type": "authentication_error",
                            "code": 401,
                        }
                    },
                )
        return await call_next(request)


from nemo_rl.distributed.refit_watchdog import RefitAborted, is_refit_abort


class _AsyncLLMHTTPClient:
    """Keep HTTP generation on the loop that owns AsyncLLM request state.

    The engine-client surface is explicit. Do not add a ``__getattr__`` fallback.
    Add each new member here and decide whether it must run on the engine loop.
    """

    def __init__(
        self,
        engine_client: Any,
        engine_loop: asyncio.AbstractEventLoop,
        *,
        flat_logprobs: bool,
    ):
        self._engine_client = engine_client
        self._engine_loop = engine_loop
        self._flat_logprobs = flat_logprobs
        # Engine request ID -> captured model call ID, while the request runs.
        self._captured_requests: dict[str, str] = {}
        # Called with each request ID once its consumer is done with it.
        self._request_done_listeners: list[Callable[[str], None]] = []
        self.model_config = engine_client.model_config
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        self.vllm_config = engine_client.vllm_config

    async def _run_on_engine_loop(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        if asyncio.get_running_loop() is self._engine_loop:
            return await operation()

        future = asyncio.run_coroutine_threadsafe(operation(), self._engine_loop)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        # Flat logprobs avoid per-token Logprob objects that cause long GC pauses.
        # Not for prompt logprobs: vLLM's -inf clamp edits copies of flat storage.
        if (
            self._flat_logprobs
            and sampling_params.logprobs is not None
            and sampling_params.prompt_logprobs is None
        ):
            sampling_params.flat_logprobs = True
        return self._generate(prompt, sampling_params, request_id, kwargs)

    async def _generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        terminal = restored_terminal()
        if terminal is not None:
            yield _build_restored_prefix_terminal_output(request_id, terminal)
            return

        iterator = None
        completed = False
        model_call_id = _CAPTURED_MODEL_CALL.get()
        if model_call_id is not None:
            self._captured_requests[request_id] = model_call_id

        async def next_output() -> Any:
            nonlocal iterator
            if iterator is None:
                iterator = self._engine_client.generate(
                    prompt, sampling_params, request_id, **kwargs
                )
            return await anext(iterator)

        try:
            while True:
                try:
                    yield await self._run_on_engine_loop(next_output)
                except StopAsyncIteration:
                    completed = True
                    return
        finally:
            try:
                self._captured_requests.pop(request_id, None)
                if not completed:
                    try:
                        await self._run_on_engine_loop(
                            lambda: self._engine_client.abort(request_id)
                        )
                    except Exception:
                        LOGGER.exception("Failed to abort vLLM request %s", request_id)
            finally:
                for listener in self._request_done_listeners:
                    listener(request_id)

    # Members a VllmProgressReader uses; they read this client's own state.
    @property
    def output_processor(self) -> Any:
        return self._engine_client.output_processor

    def captured_request_ids(self) -> dict[str, str]:
        """Engine request ID to captured model call ID, for running requests."""
        # One C-level copy under the GIL; the HTTP thread may add entries.
        return dict(self._captured_requests)

    def is_captured(self, request_id: str) -> bool:
        return request_id in self._captured_requests

    def add_request_done_listener(self, listener: Callable[[str], None]) -> None:
        """Call ``listener(request_id)`` once a request's consumer is done with it."""
        self._request_done_listeners.append(listener)

    # These members only read engine status or immutable configuration. Running
    # them on the engine loop added a cross-thread wait to each HTTP request.
    @property
    def errored(self) -> bool:
        return self._engine_client.errored

    @property
    def dead_error(self) -> BaseException:
        return self._engine_client.dead_error

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        """Queue-limit admission check vLLM >= 0.29 runs before every response.

        ``OpenAIServing._preflight`` calls this (vllm-project/vllm#49445,
        ``max_num_queued_reqs`` / ``max_num_queued_tokens``); without it every
        chat completion 500s with ``'_AsyncLLMHTTPClient' object has no
        attribute 'check_admission'``. It only reads scheduler config and
        unfinished-request counters, so it stays off the engine loop like the
        other status reads above. Raises vLLM's HTTP-mapped overflow errors.
        """
        self._engine_client.check_admission(n, request_id=request_id)

    async def is_tracing_enabled(self) -> bool:
        return await self._engine_client.is_tracing_enabled()

    async def captured_call_progress(self) -> list[list[Any]]:
        """Return ``[model_call_id, generated_tokens]`` for each running captured call.

        Reads vLLM's output-processor request state, which has no public API, on
        the engine loop. Returns nothing if this vLLM keeps that state elsewhere.
        """

        async def read() -> list[list[Any]]:
            states = getattr(
                getattr(self._engine_client, "output_processor", None),
                "request_states",
                None,
            )
            if states is None:
                return []
            progress = []
            for state in list(states.values()):
                model_call_id = self._captured_requests.get(state.external_req_id)
                if model_call_id is not None and state.detokenizer is not None:
                    progress.append(
                        [model_call_id, state.detokenizer.num_output_tokens()]
                    )
            return progress

        return await self._run_on_engine_loop(read)


class VllmAsyncGenerationWorkerImpl(
    VllmAsyncCheckpointEngineRpcMixin,
    BaseVllmGenerationWorker,
):
    def __init__(
        self,
        config,
        bundle_indices=None,
        fraction_of_gpus: float = 1.0,
        seed=None,
        extra_env_vars: Optional[list[str]] = None,
        defer_model_load: bool = False,
    ):
        """Initialize an async vLLM worker.

        When defer_model_load=True, only stores config and reserves a port for
        the HTTP server (if expose_http_server is enabled). Call load_model()
        later to perform the heavy model loading. This enables overlapping vLLM
        model loading with NeMo Gym init.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
            extra_env_vars: Additional environment variable names to forward into
                          the vLLM worker subprocess.
            defer_model_load: If True, skip model loading and only reserve port
        """
        # Deferred-loading state. Always initialized so every instance has a
        # consistent set of attributes regardless of init path.
        self._reserved_socket = None
        self._reserved_port = None
        self._reserved_node_ip = None
        self._deferred_bundle_indices = None
        self._deferred_seed = None

        # Defaults for HTTP server state; populated after the actor loop starts.
        self.server_thread = None
        self.base_url = None
        self.http_server = None
        self._engine_loop = None
        self._http_engine_client = None
        # The task serving this engine through Dynamo, when it is registered.
        self._dynamo_endpoint_task: Optional[asyncio.Task] = None

        self._init_token_capture_state()

        super().__init__(
            config,
            bundle_indices,
            fraction_of_gpus,
            seed,
            extra_env_vars,
            defer_model_load,
        )

        if not self.is_model_owner or not defer_model_load:
            return

        self._deferred_bundle_indices = bundle_indices
        self._deferred_seed = seed

        if self.cfg["vllm_cfg"].get("expose_http_server"):
            self._reserve_port()

        self.llm = None
        self.vllm_device_ids = None

    def _init_token_capture_state(self) -> None:
        """Token-capture and generation-cut state, dormant until setup_token_capture.

        Separate from ``__init__`` so tests can build a worker without vLLM:
        ``object.__new__`` plus this method.
        """
        # Ledger-authoritative token capture. The weight version is stamped
        # per model call at begin_call time and rotated by the
        # set_rollout_weight_version fan-out from the SC's _sync_weights.
        self.token_capture = None
        self._rollout_weight_version = 0
        # In-flight captured calls, by request and by Gym's model-call ID.
        self._capture_registry = CaptureRegistry()
        self._capture_sink: Any | None = None
        # Generation-prefix cuts and restores; a no-op unless setup_token_capture
        # enables them.
        self._prefix_recovery: PrefixRecovery | NoPrefixRecovery = NoPrefixRecovery()
        # Built once per engine: it wraps vLLM's output processor.
        self._progress_reader: Any | None = None
        self._capture_media = False
        self._capture_patch_size: int | None = None
        self._capture_image_token_id: int | None = None
        # TQTokenSource installed by setup_token_capture. The media path reads
        # parent-chain rows through it directly; prefix resolution goes through
        # the shared ChainPrefixCache below.
        self._staging_source: Any | None = None
        self._capture_dp_client: Any | None = None
        # Resolved staging-chain prefixes, shared implementation with the Megatron
        # preparer. Installed by setup_token_capture; fetch runs on executor threads.
        # Deferred import: tq_token_sink pulls in the data-plane stack.
        from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache

        self._chain_prefix = ChainPrefixCache()
        # policy.generation.top_logprobs: sampler top-k log-probs staged next to each
        # captured call's delta; 0 leaves captured calls as they are.
        self._capture_top_logprobs = 0

    async def start_gpu_profiling(self) -> None:
        """Start GPU profiling on the inner TP workers.

        The async engine's collective_rpc is a coroutine; the inherited sync
        start_gpu_profiling would drop it un-awaited, so the inner TP workers
        never receive cudaProfilerStart. Await it here so every inner worker starts.
        This outer actor deliberately does not call cudaProfilerStart itself, so its
        own nsys session never opens a capture and writes no report.
        """
        if self.llm is not None:
            await self.llm.collective_rpc("start_gpu_profiling", args=tuple())

    async def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling on the inner TP workers.

        Awaits collective_rpc("stop_gpu_profiling") so cudaProfilerStop reaches every
        inner worker and ends its capture.
        """
        if self.llm is not None:
            await self.llm.collective_rpc("stop_gpu_profiling", args=tuple())

    def _return_routed_experts_enabled(self) -> bool:
        engine_args = getattr(self, "llm_async_engine_args", None)
        if bool(getattr(engine_args, "enable_return_routed_experts", False)):
            return True
        return bool(
            self.cfg.get("vllm_kwargs", {}).get("enable_return_routed_experts", False)
        )

    def _reserve_port(self) -> None:
        """Bind and listen on a TCP socket to reserve a free port from the OS.

        The socket is held open in LISTENING state and later passed directly to
        uvicorn via the ``sockets=`` parameter in ``server.serve()``. The socket
        is never closed and re-opened, so there is zero gap where another process
        could steal the port.
        """
        import socket

        from nemo_rl.distributed.virtual_cluster import _get_node_ip_local

        self._reserved_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reserved_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._reserved_socket.bind(("", 0))
        self._reserved_socket.listen(128)
        self._reserved_socket.setblocking(False)
        self._reserved_port = self._reserved_socket.getsockname()[1]
        self._reserved_node_ip = _get_node_ip_local()
        print(
            f"Reserved port {self._reserved_port} on {self._reserved_node_ip} "
            f"for vLLM HTTP server"
        )

    def load_model(self) -> None:
        """Load the vLLM model and create the engine.

        Called after a deferred init to perform the heavy model loading.
        """
        if not self.is_model_owner:
            return
        self._load_model(self._deferred_bundle_indices, self._deferred_seed)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.metrics.loggers import PrometheusStatLogger

        # Workaround: convert compilation_config dict to CompilationConfig object
        # since AsyncEngineArgs doesn't handle the dict-to-pydantic conversion.
        if llm_kwargs.get("compilation_config", None):
            compilation_config = dict(llm_kwargs["compilation_config"])
            # use_inductor was removed in vLLM v0.12+ (https://github.com/vllm-project/vllm/pull/29323)
            # and replaced by the `backend` field: use_inductor=True -> backend="" (inductor),
            # use_inductor=False -> backend="eager".
            if "use_inductor" in compilation_config:
                use_inductor = compilation_config.pop("use_inductor")
                if "backend" not in compilation_config:
                    compilation_config["backend"] = "" if use_inductor else "eager"
                warnings.warn(
                    "compilation_config.use_inductor is deprecated in vLLM v0.12+. "
                    "Use compilation_config.backend instead: "
                    "use_inductor=True -> backend='inductor', "
                    "use_inductor=False -> backend='eager'.",
                    DeprecationWarning,
                    stacklevel=1,
                )
            llm_kwargs["compilation_config"] = CompilationConfig(**compilation_config)
        self.stat_loggers = (
            [PrometheusStatLogger]
            if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False)
            else []
        )
        if dynamo_enabled(self.cfg):
            from nemo_rl.models.generation.dynamo.vllm.adapter import (
                SnapshotStatLogger,
                dynamo_kv_events_config,
            )

            # Dynamo's KV-aware router indexes the engine's own KV-cache events;
            # the stat logger feeds its per-engine load signal.
            llm_kwargs.setdefault(
                "kv_events_config",
                dynamo_kv_events_config(
                    dynamo_cfg(self.cfg),
                    prefix_caching=llm_kwargs["enable_prefix_caching"],
                ),
            )
            self.stat_loggers.append(SnapshotStatLogger)
        # AsyncEngineArgs leaves kv_events_config dicts unconverted.
        if isinstance(llm_kwargs.get("kv_events_config"), dict):
            from vllm.config import KVEventsConfig

            llm_kwargs["kv_events_config"] = KVEventsConfig(
                **llm_kwargs["kv_events_config"]
            )

        self.llm_async_engine_args = AsyncEngineArgs(**llm_kwargs)
        self.llm = AsyncLLM.from_engine_args(
            self.llm_async_engine_args, stat_loggers=self.stat_loggers
        )
        # Keep vLLM's output loops from monopolizing this actor's event loop under a
        # generation backlog (see install_output_loop_yield).
        output_loop_yield_budget_ms = self.cfg["vllm_cfg"].get(
            "output_loop_yield_budget_ms"
        )
        if output_loop_yield_budget_ms is not None:
            from nemo_rl.models.generation.vllm.patches import (
                install_output_loop_yield,
            )

            install_output_loop_yield(
                self.llm, reader_budget_s=output_loop_yield_budget_ms / 1000
            )

        # vLLM Metrics Logger
        # Metrics logger only enabled for per-actor, model-owner only
        self._vllm_metrics_lock = threading.Lock()
        if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            self._start_vllm_metrics_logger()

    def _start_vllm_metrics_logger(self) -> None:
        """Start a background thread that periodically collects vLLM logger metrics.

        Controlled by the required vllm_metrics_logger_interval in vllm_cfg.
        Runs only on the model-owner actor.
        """
        from vllm.v1.metrics.reader import Counter, Gauge, get_metrics_snapshot

        assert self.cfg["vllm_cfg"].get("async_engine", False), (
            "vLLM metrics logger is only supported with async engine enabled"
        )
        # Run only on the model-owner actor
        if not getattr(self, "is_model_owner", False):
            return

        assert "vllm_metrics_logger_interval" in self.cfg["vllm_cfg"], (
            "vllm_metrics_logger_interval must be set in vllm_cfg if enable_vllm_metrics_logger is True"
        )
        interval_s = self.cfg["vllm_cfg"]["vllm_metrics_logger_interval"]
        assert interval_s > 0, (
            f"vllm_metrics_logger_interval must be a positive float, got {interval_s}"
        )

        # Lazy import inside thread target to avoid import overhead if disabled
        stop_event = threading.Event()
        self._vllm_metrics_logger_stop_event = stop_event

        self.inflight_batch_sizes: list[int] = []
        self.num_pending_samples: list[int] = []
        self.kv_cache_usage_perc: list[float] = []
        self.generation_tokens: list[int] = []

        def _logger_loop():
            # Delay a little to let engine settle
            time.sleep(min(2.0, interval_s))
            while True:
                try:
                    for m in get_metrics_snapshot():
                        with self._vllm_metrics_lock:
                            if isinstance(m, Gauge):
                                # Log the vllm inflight batch sizes
                                if m.name == "vllm:num_requests_running":
                                    self.inflight_batch_sizes.append(int(m.value))
                                # Log the vllm pending number of requests in the queue
                                elif m.name == "vllm:num_requests_waiting":
                                    self.num_pending_samples.append(int(m.value))
                                # Log the vllm kv cache usage
                                elif m.name == "vllm:kv_cache_usage_perc":
                                    self.kv_cache_usage_perc.append(float(m.value))
                            elif isinstance(m, Counter):
                                if m.name == "vllm:generation_tokens":
                                    self.generation_tokens.append(int(m.value))
                except Exception:
                    print(
                        "⚠️[vLLM Metric Logger] Exception in vLLM metrics logger",
                        flush=True,
                    )
                    pass
                time.sleep(interval_s)

        t = threading.Thread(
            target=_logger_loop, name="vllm-metrics-logger", daemon=True
        )
        t.start()
        self._vllm_metrics_logger_thread = t
        print(
            "📋[vLLM Metric Logger] vLLM metrics logger thread started",
            flush=True,
        )

    def get_vllm_logger_metrics(self) -> dict[str, Any]:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return {}

        with self._vllm_metrics_lock:
            metric = {
                "inflight_batch_sizes": copy.deepcopy(self.inflight_batch_sizes),
                "num_pending_samples": copy.deepcopy(self.num_pending_samples),
                "kv_cache_usage_perc": copy.deepcopy(self.kv_cache_usage_perc),
                "generation_tokens": copy.deepcopy(self.generation_tokens),
            }
        return metric

    def drain_latest_vllm_logger_metrics(self) -> dict[str, Any]:
        """Return latest samples and prune histories after a telemetry poll."""
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return {}

        with self._vllm_metrics_lock:
            histories = {
                "inflight_batch_sizes": self.inflight_batch_sizes,
                "num_pending_samples": self.num_pending_samples,
                "kv_cache_usage_perc": self.kv_cache_usage_perc,
                "generation_tokens": self.generation_tokens,
            }
            latest = {
                name: [values[-1]] if values else []
                for name, values in histories.items()
            }
            # Keep worker-owned histories distinct from the lists handed to Ray;
            # the sampling thread may append immediately after this lock exits.
            self.inflight_batch_sizes = list(latest["inflight_batch_sizes"])
            self.num_pending_samples = list(latest["num_pending_samples"])
            self.kv_cache_usage_perc = list(latest["kv_cache_usage_perc"])
            self.generation_tokens = list(latest["generation_tokens"])
            return {name: list(values) for name, values in latest.items()}

    def clear_vllm_logger_metrics(self) -> None:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return

        with self._vllm_metrics_lock:
            self.inflight_batch_sizes = []
            self.num_pending_samples = []
            self.kv_cache_usage_perc = []
            self.generation_tokens = []

    async def post_init_async(self):
        self._engine_loop = asyncio.get_running_loop()
        if self._sparse_refit_receiver is not None:
            self._sparse_refit_receiver.set_async_loop(self._engine_loop)
        if self.llm is not None:
            await self.llm.collective_rpc("bind_numa", args=tuple())
            if parse_nvfp4_pertoken_rollout(self.cfg) is not None:
                target_counts = await resolve_collective_rpc_result(
                    self.llm.collective_rpc(
                        "report_nvfp4_pertoken_target_count", args=tuple()
                    )
                )
                if not target_counts or sum(target_counts) == 0:
                    raise RuntimeError(
                        "generation.nvfp4_pertoken_rollout selected no "
                        "RoutedExperts targets across the vLLM model"
                    )
        self.vllm_device_ids = await self.report_device_id_async()
        if self._mtp_speculative_enabled:
            await self.llm.collective_rpc(
                "configure_mtp_drafter_weight_source",
                args=(self._mtp_weights_from_refit,),
            )
        if self._mtp_load_from_disk:
            await self.llm.collective_rpc(
                "load_mtp_weights_from_disk", args=(self.model_name,)
            )
        if self._sparse_refit_receiver is not None:
            hostnames = await self.llm.collective_rpc("report_node_hostname", args=())
            self._sparse_refit_receiver.set_worker_hostnames(hostnames)
        if self.llm is not None and self.cfg["vllm_cfg"].get("expose_http_server"):
            self._http_engine_client = _AsyncLLMHTTPClient(
                self.llm,
                self._engine_loop,
                flat_logprobs=bool(self.cfg["vllm_cfg"].get("flat_logprobs")),
            )
            self.server_thread, self.base_url, self.http_server = (
                self._setup_vllm_server()
            )

    async def start_dynamo_endpoint_async(
        self, *, namespace: str, engine_env: dict[str, str], leader_worker_idx: int
    ) -> None:
        """Register this engine with Dynamo and serve it from this actor's loop.

        The loop is the one that owns the AsyncLLM, so Dynamo requests, refit,
        pause and sleep interleave exactly as this worker's own requests do.
        ``leader_worker_idx`` is this worker's index in its group; rollouts the
        engine serves report it, as ``generate_async`` results do.
        """
        # ai-dynamo is optional: import it only for engines served through Dynamo.
        from dynamo._core import fetch_model

        from nemo_rl.models.generation.dynamo.engine_adapter import (
            start_dynamo_endpoint,
        )
        from nemo_rl.models.generation.dynamo.vllm.adapter import DynamoVllmAdapter

        # Dynamo fetches a Hub ID's whole checkpoint before it registers; the engine's
        # weights come from refit, so register the config and tokenizer files only.
        model_path = (
            self.model_name
            if os.path.exists(self.model_name)
            else await fetch_model(self.model_name, ignore_weights=True)
        )
        kv_events = self.llm.vllm_config.kv_events_config
        publishes = kv_events is not None and kv_events.enable_kv_cache_events
        # The Dynamo functional overlays check the router applied these.
        print(f"  [Dynamo] engine publishes KV events: {publishes}", flush=True)
        self._dynamo_endpoint_task = start_dynamo_endpoint(
            DynamoVllmAdapter(self, leader_worker_idx=leader_worker_idx),
            model_path=model_path,
            served_model_name=self.model_name,
            namespace=namespace,
            engine_env=engine_env,
        )

    async def dynamo_endpoint_error(self) -> Optional[str]:
        """Why this engine's Dynamo endpoint stopped serving; None while it serves."""
        task = self._dynamo_endpoint_task
        if task is None or not task.done():
            return None
        if task.cancelled():
            return "cancelled"
        error = task.exception()
        return "exited" if error is None else repr(error)

    async def get_reserved_url(self) -> Optional[str]:
        """Return the URL from the reserved socket, available before model loading."""
        if self._reserved_socket is not None:
            return f"http://{self._reserved_node_ip}:{self._reserved_port}/v1"
        return None

    async def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self.base_url

    def install_token_capture(self, capture: Any) -> None:
        """Gym's ``install_capture`` seam (the ``CaptureHost`` contract)."""
        self.token_capture = capture

    async def setup_token_capture(
        self,
        dp_cfg: dict[str, Any],
        staging_partition: str,
        *,
        capture_media: bool = False,
        top_logprobs: int = 0,
        generation_prefix_cuts_enabled: bool = False,
        generation_cut_control_token: str | None = None,
        generation_cut_control_timeout_s: float | None = None,
    ) -> MediaColumnSpec | None:
        """Host ledger-authoritative token capture in this worker.

        Fan-out target (token_capture.enabled only): builds the in-worker
        data-plane client and TQTokenSink, then makes the single
        ``install_capture`` call wiring Gym's engine-blind capture core +
        vLLM adapter into this worker. Returns the media column spec this
        worker pinned the staging column to (``None`` for text-only capture
        and on non-model-owner ranks, which serve no HTTP and install nothing).
        ``top_logprobs`` > 0 stages sampler top-k log-probs.
        """
        if not self.is_model_owner:
            return None
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
        from nemo_gym.token_id_capture.staging import install_capture

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource

        # Own no segment: a save would otherwise wait on this actor, whose event
        # loop is busy serving rollouts. Capture writes land in other segments.
        dp_client = build_data_plane_client(dp_cfg, bootstrap=False, segment_size=0)
        self._capture_dp_client = dp_client
        # The Omni processor emits pixels in the engine's model dtype; the
        # sink pins its media column to it so text-call sentinels never
        # introduce a second dtype (TQ keeps one dtype per field).
        pixel_dtype = self.llm.model_config.dtype if capture_media else None
        sink = TQTokenSink(
            dp_client,
            staging_partition=staging_partition,
            capture_media=capture_media,
            media_pixel_dtype=pixel_dtype,
        )
        self._capture_top_logprobs = int(top_logprobs)
        if generation_prefix_cuts_enabled and not generation_cut_control_token:
            raise ValueError(
                "generation-prefix cuts require a non-empty control bearer token"
            )
        if capture_media:
            # Omni-only: a new processor family must also change setup.py (driver
            # checks), captured_media.py (_processed_omni_tensors, pack_images,
            # capture_processed_media), tq_token_sink.py (MEDIA_TENSOR_COLUMNS,
            # validate_media_tensors, fetch_media) and rollout_reassembler.py
            # (_concat_media, _trainer_media).
            # Optional engine/Gym capabilities are checked only on VLM workers.
            from vllm.model_executor.models.nano_nemotron_vl import (
                NanoNemotronVLProcessingInfo,
            )

            info = self.llm.renderer.get_mm_processor().info
            if (
                not isinstance(info, NanoNemotronVLProcessingInfo)
                or not info.is_dynamic_tiler
            ):
                raise ValueError(
                    "Media capture requires vLLM's Omni dynamic-resolution processor"
                )
            if info.get_video_pruning_rate():
                raise ValueError(
                    "Omni media capture does not support video token pruning"
                )
            context_ids = info.get_hf_processor()._img_context_token_ids
            if len(context_ids) != 1:
                raise ValueError(
                    "Omni media capture requires one image-context token ID"
                )
            self._capture_image_token_id = int(context_ids[0])
            self._capture_patch_size = int(info.get_hf_config().patch_size)
        self._capture_media = capture_media
        self._capture_sink = sink
        if generation_prefix_cuts_enabled and self._http_engine_client is None:
            raise RuntimeError(
                "generation-prefix cuts need the vLLM HTTP server "
                "(vllm_cfg.expose_http_server: true)"
            )
        source = TQTokenSource(
            dp_client, staging_partition=staging_partition, capture_media=capture_media
        )
        self._staging_source = source
        self._chain_prefix.install(source)
        install_capture(
            self,
            sink=sink,
            weight_version_fn=lambda: self._rollout_weight_version,
            adapter=VLLMCaptureAdapter(),
        )
        self._prefix_recovery.shutdown()
        self._prefix_recovery = NoPrefixRecovery()
        if generation_prefix_cuts_enabled:
            if self._progress_reader is None:
                self._progress_reader = VllmProgressReader(self._http_engine_client)
            self._prefix_recovery = PrefixRecovery(
                PrefixRecoveryParts(
                    capture=self.token_capture,
                    sink=sink,
                    staging_source=source,
                    registry=self._capture_registry,
                    # Cuts must land on character boundaries.
                    tokenizer=self.llm.renderer.get_tokenizer(),
                    weight_version=lambda: self._rollout_weight_version,
                    control_token=generation_cut_control_token,
                    control_timeout_s=generation_cut_control_timeout_s,
                    progress_reader=self._progress_reader,
                )
            )
        if not capture_media:
            return None
        return MediaColumnSpec(
            pixel_dtype=pixel_dtype, patch_size=self._capture_patch_size
        )

    async def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run owner-local checkpoint I/O without blocking the actor event loop."""
        return await asyncio.to_thread(run_checkpoint_command, body)

    def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        """The capture client's data-plane counters, ``None`` when it has none."""
        from nemo_rl.data_plane.observability import is_metrics_client

        if not is_metrics_client(self._capture_dp_client):
            return None
        return self._capture_dp_client.snapshot(reset_step_window=True)

    async def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the weight version stamped on subsequent captured calls."""
        self._rollout_weight_version = int(version)

    async def captured_call_progress(self) -> Optional[dict[str, Any]]:
        """Report how far each running captured call got on the current weights.

        Called while generation is paused for a refit, so the counts are where
        each call switches to the new weights. ``None`` on ranks that serve no
        captured calls.
        """
        if self._http_engine_client is None or self.token_capture is None:
            return None
        return {
            "weight_version": self._rollout_weight_version,
            "calls": await self._http_engine_client.captured_call_progress(),
        }

    def _capture_admission(self, request: Any) -> Any | None:
        """Parse the ledger's ``ng_capture`` context into a ``CaptureAdmission``.

        Returns None unless capture is installed and the request carries the
        context. The dict itself is never mutated: the admission is the typed,
        read-only contract that the prefix resolution and ``begin_call`` share.
        """
        return self._parse_capture_admission(getattr(request, "ng_capture", None))

    def _parse_capture_admission(self, context: Any) -> Any | None:
        """``_capture_admission`` for a raw ``ng_capture`` context (Dynamo's annotation)."""
        if self.token_capture is None or not context:
            return None
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        return CaptureAdmission.model_validate(context)

    def _begin_request_capture(
        self,
        request: Any,
        prompt_token_ids: list[int],
        *,
        admission: Any | None = None,
        prefix_token_ids: list[int] | None = None,
        media: CapturedMedia | None = None,
        generation_cut: Any | None = None,
        resumed_generation_token_ids: list[int] | None = None,
    ) -> None:
        """Admit one ledger-forwarded call into the capture layer.

        Called from ``prompt_and_capture.start_capture`` once the exact engine prompt
        is known (post-splice in token-in mode, full render in text mode). No-op
        unless capture is installed and the request carries the ledger's
        ``ng_capture`` context.

        ``prefix_token_ids`` is the prefix resolved by
        :meth:`_resolve_admission_prefix`. Gym's ``begin_call`` requires it for
        a ``staging_chain`` admission and checks its length against
        ``prev_len``; violations raise ``CaptureError`` before any state is kept.
        """
        capture = self.token_capture
        if capture is None or getattr(request, "ng_measure_only", False):
            return
        if admission is None:
            admission = self._capture_admission(request)
            if admission is None:
                return
        if self._capture_top_logprobs:
            validate_score_centering_request(
                request, num_candidates=self._capture_top_logprobs
            )
        call = capture.begin_call(
            admission,
            prefix_token_ids=prefix_token_ids,
            generation_cut=generation_cut,
            generation_cut_staging_keys=(
                admission.generation_cut.staging_keys
                if admission.generation_cut is not None
                else None
            ),
            stream=bool(getattr(request, "stream", False)),
        )
        _CAPTURED_MODEL_CALL.set(call.model_call_id)
        state = _RequestCaptureState(
            call,
            prompt_token_ids,
            media=media,
            resumed_generation_token_ids=resumed_generation_token_ids or (),
            effective_output_limit=(
                admission.generation_cut.effective_output_limit
                if admission.generation_cut is not None
                else None
            ),
            staging_keys=(
                admission.generation_cut.staging_keys
                if admission.generation_cut is not None
                else ()
            ),
        )
        self._capture_registry.add(request, state)
        # Counted once Gym's begin_call has accepted a restored cut.
        self._prefix_recovery.on_begin(admission)

    def _capture_request_media(
        self,
        engine_prompt: dict[str, Any],
        *,
        admission: Any | None,
        splice: PrefixSplice | None = None,
        verify_retained_media: bool = False,
    ) -> CapturedMedia | None:
        """Run off-loop: resolve retained geometry and snapshot processed pixels."""
        if admission is None:
            return None
        if not self._capture_media:
            if engine_prompt.get("mm_placeholders") or engine_prompt.get("mm_kwargs"):
                raise MediaCaptureRejected(
                    "Multimodal token capture requires media capture setup"
                )
            return None
        retained: tuple[CapturedMediaItem, ...] = ()
        retained_tensors = () if verify_retained_media else None
        if admission.parent_call_id is not None:
            # Optional Gym dependency: this method only runs on captured calls.
            # Gym owns the media_spans key its adapter copies into the extras.
            from nemo_gym.token_id_capture.adapters.vllm import MEDIA_SPANS_FIELD
            from nemo_gym.token_id_capture.staging.digest import compute_chain_hash
            from nemo_gym.token_id_capture.staging.records import staging_key

            source = self._staging_source
            if source is None:
                raise RuntimeError("Media capture staging source is not initialized")
            try:
                if admission.staging_chain:
                    calls = source.fetch_for_finalization(
                        list(admission.staging_chain),
                        include_route_fragments=verify_retained_media,
                    )
                else:
                    # Inline token admissions still have receipt-owned parent keys.
                    calls, visited = [], set()
                    parent = admission.parent_call_id
                    while parent is not None:
                        if parent in visited:
                            raise MediaCaptureRejected("Cycle in retained media chain")
                        visited.add(parent)
                        call = source.fetch_for_finalization(
                            [staging_key(admission.capture_key, parent)],
                            include_route_fragments=verify_retained_media,
                        )[0]
                        calls.append(call)
                        parent = call.snapshot.parent_call_id
                    calls.reverse()
            except (KeyError, MediaMetadataIntegrityError) as error:
                raise MediaCaptureRejected(
                    f"Invalid retained media metadata: {error}"
                ) from error
            parent, length, chain_hash = None, 0, None
            for call in calls:
                snapshot = call.snapshot
                if (
                    snapshot.capture_key != admission.capture_key
                    or snapshot.parent_call_id != parent
                    or snapshot.prev_len != length
                    or snapshot.chain_hash
                    != compute_chain_hash(chain_hash, snapshot.token_ids_delta)
                ):
                    raise MediaCaptureRejected("Invalid retained media call chain")
                parent, length, chain_hash = (
                    snapshot.model_call_id,
                    snapshot.cum_len,
                    snapshot.chain_hash,
                )
            if (parent, length, chain_hash) != (
                admission.parent_call_id,
                admission.prev_len,
                admission.parent_chain_hash,
            ):
                raise MediaCaptureRejected(
                    "Retained image chain does not match capture admission"
                )
            descriptors = {}
            for call in calls:
                if verify_retained_media:
                    items = verified_media_items(call)
                else:
                    extras = call.extras or {}
                    if MEDIA_SPANS_FIELD not in extras:
                        raise MediaCaptureRejected(
                            "Retained vLLM media spans are missing"
                        )
                    items = tuple(
                        CapturedMediaItem.from_dict(value)
                        for value in extras[MEDIA_SPANS_FIELD]
                    )
                    for item in items:
                        item.verify_tokens(
                            call.snapshot.token_ids_delta, origin=call.snapshot.prev_len
                        )
                descriptors[call.snapshot.model_call_id] = items
            retained = tuple(item for items in descriptors.values() for item in items)
            if verify_retained_media:
                media_calls = [call for call in calls if call.media_present]
                retained_tensors = (
                    tuple(source.fetch_media(media_calls)) if media_calls else ()
                )
                for call, tensors in zip(media_calls, retained_tensors, strict=True):
                    verify_media_geometry(
                        descriptors[call.snapshot.model_call_id], tensors
                    )
        return capture_processed_media(
            engine_prompt,
            prev_len=admission.prev_len,
            retained=retained,
            splice=splice,
            image_token_id=self._capture_image_token_id,
            patch_size=self._capture_patch_size,
            retained_tensors=retained_tensors,
        )

    def _fetch_chain_prefix(self, staging_chain: list[str]) -> list[int]:
        """Resolve a staging chain through the shared, cached TQ read."""
        return self._chain_prefix.fetch(staging_chain)

    def _resolve_admission_prefix(self, admission: Any) -> list[int]:
        """Resolve a ``CaptureAdmission`` to the flat prefix the engine prompt starts with."""
        # Deferred import, matching setup_token_capture.
        from nemo_rl.data_plane.tq_token_sink import resolve_admission_prefix

        return resolve_admission_prefix(admission, self._chain_prefix)

    @staticmethod
    def _delta_align_routed_experts(
        payload: dict[str, Any], *, prev_len: int, prompt_len: int, generated_len: int
    ) -> torch.Tensor | None:
        """Slice native routes for the delta plus boundary row and bind metadata."""
        choices = payload.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            return None
        choice = dict(choices[0])
        message = dict(choice.get("message") or {})
        routed = message.get("routed_experts")
        if routed is None:
            return None
        try:
            if not isinstance(routed, torch.Tensor):
                raise TypeError("captured routed_experts must be a native tensor")
            expected_full_len = prompt_len + generated_len
            if routed.dim() != 3 or routed.shape[0] != expected_full_len:
                raise ValueError(
                    f"route shape {tuple(routed.shape)} does not match engine sequence "
                    f"length {expected_full_len}"
                )
            start = prev_len - boundary_route_rows(prev_len)
            # Keep this as a view: capture uses a local client, and SimpleStorage
            # serializes only the slice's bytes. Message routes sent through
            # Ray/pickle use slice_routed_experts() to copy only their token span.
            routed = routed[start:]
            message["routed_experts"] = routed_experts_tensor_metadata(routed)
        except (IndexError, TypeError, ValueError) as error:
            LOGGER.warning(
                "dropping invalid routed_experts from staged capture: %s", error
            )
            message.pop("routed_experts", None)
            routed = None
        choice["message"] = message
        payload["choices"] = [choice]
        return routed

    def _finish_request_capture(self, request: Any, content: dict) -> dict:
        """Stage one canonical terminal row and retire its prefix chunks.

        Tokens, routes, and the call's new media tensors go to TQ in one write
        (routes and media ride as opaque attachments beside the record), so
        ``staged`` coords vouch for all and any failure is ``capture_failed`` at call
        time. Token ids, logprobs, and routes are stripped after staging, so
        the worker->gate hop carries the completion and coords only.
        Sampler tensors ride as opaque attachments in that same staging write.
        """
        sampler_topk = None
        for choice in content.get("choices") or []:
            message = choice.get("message")
            if isinstance(message, dict) and (
                "sampler_topk_token_ids" in message
                or "sampler_topk_logprobs" in message
            ):
                sampler_topk = (
                    message.pop("sampler_topk_token_ids", None),
                    message.pop("sampler_topk_logprobs", None),
                )
        state = self._capture_registry.get(request)
        if state is None:
            return content
        with state.lifecycle() as owned:
            # A failed call is retried from its task or its previous boundary,
            # never from these rows, so they go whether or not its row stages.
            obsolete_staging_keys = owned.begin_terminal()
            content = self._finish_request_capture_with_lifecycle_owned(
                state, request, content, sampler_topk=sampler_topk
            )
        self._prefix_recovery.retire(obsolete_staging_keys)
        return content

    def _abort_request_capture(
        self, request: Any, *, reason: str, blocking: bool = True
    ) -> bool:
        """Drop the in-flight capture state for a request that errored.

        A generation cut or a terminal write holds the call's lifecycle lock
        across a TQ round trip. With ``blocking=False`` the abort does not wait
        for it and returns False instead; it returns True once the call needs
        no abort.
        """
        state = self._capture_registry.get(request)
        if state is None:
            return True
        obsolete_staging_keys: tuple[str, ...] = ()
        with state.lifecycle() if blocking else state.try_lifecycle() as owned:
            if owned is None:
                return False
            if not owned.terminal_started:
                staging_keys = owned.begin_terminal()
                popped = self._capture_registry.pop(request)
                if popped is state and self.token_capture is not None:
                    coords = self.token_capture.fail_call(state.call, reason=reason)
                    self._prefix_recovery.remember_completed(
                        state.call.model_call_id, coords, 0
                    )
                    obsolete_staging_keys = staging_keys
        self._prefix_recovery.retire(obsolete_staging_keys)
        return True

    def _finish_request_capture_with_lifecycle_owned(
        self,
        state: _RequestCaptureState,
        request: Any,
        content: dict,
        *,
        sampler_topk: tuple[Any, Any] | None = None,
    ) -> dict:
        """Stage a terminal call while its lifecycle is owned."""
        call, prompt_token_ids = state.call, state.prompt_token_ids
        payload = dict(content)
        # vLLM's OpenAI response carries no prompt ids; the adapter reads the
        # preprocess-time engine prompt off the payload (see
        # nemo_gym.token_id_capture.adapters.vllm.extract_prompt_ids).
        payload["prompt_token_ids"] = prompt_token_ids
        if state.media is not None:
            # Optional Gym dependency: only captured media calls reach here.
            from nemo_gym.token_id_capture.adapters.vllm import MEDIA_SPANS_FIELD

            # Placeholder metadata (offsets, token hashes, sizes) rides the
            # digest-covered extras; the pixels themselves are attachments.
            payload[MEDIA_SPANS_FIELD] = [item.to_dict() for item in state.media.items]
        adapter = self.token_capture.adapter
        routed_experts = None
        generated_token_ids: list[int] = []
        if adapter is not None:
            try:
                generated_token_ids, _ = adapter.extract_generation(payload)
            except Exception:  # capture core will report the authoritative failure
                generated_token_ids = []
            routed_experts = self._delta_align_routed_experts(
                payload,
                prev_len=call.admission.prev_len,
                prompt_len=len(prompt_token_ids),
                generated_len=len(generated_token_ids),
            )
        attachments = dict(state.media.tensors or {}) if state.media is not None else {}
        if routed_experts is not None:
            attachments["routed_experts"] = routed_experts
        if sampler_topk is not None:
            attachments.update(
                sampler_topk_token_ids=sampler_topk[0],
                sampler_topk_logprobs=sampler_topk[1],
            )
        if self._capture_top_logprobs > 0 and sampler_topk is None:
            coords = self.token_capture.fail_call(
                call, reason="missing sampler top-k tensors"
            )
        else:
            coords = self.token_capture.complete_call_from_response(
                call,
                payload,
                attachments=attachments or None,
            )
        total_generation_token_count = len(generated_token_ids)
        if call.generation_cut is not None:
            prefix_tokens = sum(
                mask == 1.0 for mask in call.generation_cut.token_mask_delta
            )
            total_generation_token_count += prefix_tokens
            LOGGER.info(
                "generation prefix completed: rollout_id=%s model_call_id=%s "
                "source_model_call_id=%s prefix_tokens=%d tail_tokens=%d "
                "total_generation_tokens=%d",
                call.capture_key,
                call.model_call_id,
                call.generation_cut.model_call_id,
                prefix_tokens,
                len(generated_token_ids),
                total_generation_token_count,
            )
        progress = state.snapshot()
        effective_output_limit = progress.effective_output_limit
        terminal_finish_reason = progress.terminal_finish_reason
        terminal_stop_reason = progress.terminal_stop_reason
        if terminal_finish_reason is None:
            choices = content.get("choices") or []
            finish_reason = choices[0].get("finish_reason") if choices else None
            if finish_reason in ("stop", "length"):
                terminal_finish_reason = finish_reason
                terminal_stop_reason = choices[0].get("stop_reason")
        self._prefix_recovery.remember_completed(
            call.model_call_id,
            coords,
            total_generation_token_count,
            effective_output_limit=effective_output_limit,
            terminal_finish_reason=terminal_finish_reason,
            terminal_stop_reason=terminal_stop_reason,
        )
        self._capture_registry.pop(request)
        for choice in content.get("choices") or []:
            choice.pop("logprobs", None)
            choice.pop("routed_experts", None)
            # Token arrays and delta-aligned routes were staged to TQ above;
            # remove the serializer's message fields before the worker->gate hop.
            message = choice.get("message")
            if isinstance(message, dict):
                for field in (
                    "prompt_token_ids",
                    "generation_token_ids",
                    "generation_log_probs",
                    "routed_experts",
                ):
                    message.pop(field, None)
        content["ng_commit_coords"] = coords.model_dump()
        return content

    def _context_refusal_headers(self, request: Any) -> dict[str, str]:
        """Acknowledge a validation refusal after draining capture state, before staging."""
        admission = self._capture_admission(request)
        if admission is None:
            return {}
        # Deferred: Gym is optional outside framework-owned capture.
        from nemo_gym.token_id_capture.protocols import CAPTURE_REFUSAL_HEADER

        return {CAPTURE_REFUSAL_HEADER: admission.model_call_id}

    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        # Deferred: the chat server imports vLLM's OpenAI server stack.
        from nemo_rl.models.generation.vllm import chat_server

        engine_client = self._http_engine_client
        if engine_client is None:
            raise RuntimeError("The HTTP engine client is not initialized.")
        server = chat_server.build_chat_server(
            self,
            engine_client=engine_client,
            model_config=self.llm_async_engine_args.create_model_config(),
            cfg=self.cfg,
        )
        chat_server.register_routes(app, server, host=self, generation_config=self.cfg)
        chat_server.install_log_filters()
        return app

    def _setup_vllm_server(self) -> "tuple[threading.Thread, str, uvicorn.Server]":
        import threading
        from logging import Filter as LoggingFilter
        from logging import LogRecord, getLogger

        import uvicorn
        from fastapi import FastAPI

        # We initialize the FastAPI app here in case we want to do some generic configuration before the subsequent server inits
        # e.g. last-run middleware.
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        app = self._setup_vllm_openai_api_server(app)
        if self._sparse_refit_receiver is not None:
            self._sparse_refit_receiver.setup_api_server(app)
        # Present only when the driver generated a key (vllm_cfg.http_server_api_key_required).
        http_server_api_key = self.cfg["vllm_cfg"].get("http_server_api_key")
        if http_server_api_key:
            install_http_server_api_key_check(app, http_server_api_key)

        ########################################
        # Server spinup
        ########################################

        if self._reserved_socket is not None:
            # Use the socket reserved during __init__ (deferred model load path).
            # Pass it directly to uvicorn via sockets= — zero gap, the socket is
            # never closed and re-opened, so no other process can steal the port.
            node_ip = self._reserved_node_ip
            free_port = self._reserved_port
            reserved_sock = self._reserved_socket
            self._reserved_socket = None  # Transfer ownership to uvicorn
        else:
            node_ip = _get_node_ip_local()
            port_range_low = self.cfg.get(
                "port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW
            )
            port_range_high = self.cfg.get(
                "port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH
            )
            free_port = _get_free_port_local(port_range_low, port_range_high)
            reserved_sock = None

        base_url = f"http://{node_ip}:{free_port}/v1"
        print(f"Starting server on {base_url}")

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=free_port,
            timeout_keep_alive=120,  # Keep connections alive longer (default is 5s), fix for this error: Hit an exception while making a request (try 1): <class 'aiohttp.client_exceptions.ClientOSError'>: [Errno 104] Connection reset by peer
        )
        server = uvicorn.Server(config=config)

        print(
            "Adding a uvicorn logging filter so that the logs aren't spammed with 200 OK messages. This is to help errors pop up better and filter out noise."
        )

        class No200Filter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                msg = record.getMessage()
                return not msg.strip().endswith("200")

        uvicorn_logger = getLogger("uvicorn.access")
        uvicorn_logger.addFilter(No200Filter())

        if reserved_sock is not None:
            # Hand the pre-bound listening socket directly to uvicorn's asyncio
            # server via server.serve(sockets=). No close-and-rebind needed.
            def _run_with_socket() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(server.serve(sockets=[reserved_sock]))

            thread = threading.Thread(target=_run_with_socket, daemon=True)
        else:
            thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        return thread, base_url, server

    async def init_collective_async(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        await self.llm.collective_rpc(
            "init_collective",
            args=(
                rank_prefix,
                ip,
                port,
                world_size,
                train_world_size,
            ),
        )

    def _rollout_max_new_tokens(self, input_length: int) -> int:
        """Token budget for one rollout prompt; 0 means nothing fits.

        The configured cap, clipped to the context window and to speculative
        decoding's lookahead.
        """
        max_model_len = int(self.cfg["vllm_cfg"]["max_model_len"])
        allowed_new_tokens = max(
            0, min(self.cfg["max_new_tokens"], max_model_len - input_length)
        )
        spec_cfg = self.cfg.get("vllm_kwargs", {}).get("speculative_config") or {}
        spec_lookahead = int(spec_cfg.get("num_speculative_tokens", 0))
        if allowed_new_tokens > 0 and spec_lookahead > 0:
            allowed_new_tokens = self._request_max_new_tokens(
                configured_max_new_tokens=allowed_new_tokens,
                input_length=input_length,
                max_model_len=max_model_len,
                cap_to_context=False,
                spec_lookahead=spec_lookahead,
            )
        return allowed_new_tokens

    async def _serve_rollout(
        self,
        prompt: Any,
        *,
        prompt_token_ids: list[int],
        greedy: bool,
        stop_strings: Optional[list[str]],
        request_id: str,
        device: torch.device,
    ) -> tuple[RolloutResult, Optional[Any]]:
        """Serve one rollout from this engine: its budget, sampling, generation and result.

        Args:
            prompt: The vLLM prompt to generate from.
            prompt_token_ids: The learner's prompt tokens, which vLLM must have run.
            greedy: Greedy decoding instead of the configured sampling.
            stop_strings: The rollout's own stop strings, merged with the
                configured ones.
            request_id: The vLLM request ID.
            device: Where to place the routed-experts tensor.

        Returns:
            The result, and vLLM's final RequestOutput (None when no new token fits).
            The final output supplies finish and usage metadata to Dynamo
            (``dynamo/vllm_adapter.py``).
        """
        max_new_tokens = self._rollout_max_new_tokens(len(prompt_token_ids))
        if max_new_tokens == 0:
            return RolloutResult.nothing_generated(), None
        sampling_params = self._build_sampling_params(
            greedy=greedy,
            stop_strings=self._merge_stop_strings(
                [stop_strings] if stop_strings else None
            ),
            max_new_tokens=max_new_tokens,
        )
        final_output = None
        async for output in self.llm.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id
        ):
            final_output = output
        if final_output is None:
            raise RuntimeError(f"No output received for request {request_id}")
        validate_rollout_prompt(prompt_token_ids, final_output.prompt_token_ids)
        result = rollout_result(
            final_output,
            input_length=len(prompt_token_ids),
            return_routed_experts=self._return_routed_experts_enabled(),
            routed_experts_dtype=self.routed_experts_dtype,
            device=device,
        )
        return result, final_output

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate one sample with vLLM's AsyncLLM, yielding it when it is ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths for one sample.
            greedy: Whether to use greedy decoding instead of sampling.

        Yields:
            (0, the sample's output conforming to GenerationOutputSpec).
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        batch_size = data["input_ids"].shape[0]
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        input_ids = data["input_ids"][0]
        input_length = int(data["input_lengths"][0].item())
        prompt = self._tokenize_prompt_with_bos(
            format_prompt_for_vllm_generation(data, 0)
        )
        result, _ = await self._serve_rollout(
            prompt,
            prompt_token_ids=input_ids[:input_length].tolist(),
            greedy=greedy,
            stop_strings=(data.get("stop_strings") or [None])[0],
            request_id=str(uuid.uuid4()),
            device=input_ids.device,
        )
        yield 0, rollout_output(result, input_ids=input_ids, input_length=input_length)

    async def generate_text_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate text responses asynchronously, yielding results as they are ready.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict containing single text response)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["prompts"]) == 0:
            return

        prompts = data["prompts"]
        batch_size = len(prompts)

        # Extract stop_strings if provided, else use default from config
        batch_stop_strings: list[list[str] | None] = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * batch_size
        )

        # Create tasks for each prompt
        async def process_single_prompt(prompt_idx):
            """Process a single prompt and return the result."""
            prompt = self._tokenize_prompt_with_bos(prompts[prompt_idx])

            # Get stop strings for this specific prompt
            per_prompt_stop_strings = None
            if batch_stop_strings and prompt_idx < len(batch_stop_strings):
                per_prompt_stop_strings = batch_stop_strings[prompt_idx]

            # Merge stop strings
            final_stop_strings = self._merge_stop_strings(
                [per_prompt_stop_strings] if per_prompt_stop_strings else None
            )

            # Create sampling parameters
            top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
            sampling_params = self.SamplingParams(
                temperature=self.cfg["temperature"] if not greedy else 0,
                top_p=self.cfg["top_p"],
                top_k=top_k if not greedy else 1,
                max_tokens=self.cfg["max_new_tokens"],
                stop_token_ids=self.cfg["stop_token_ids"],
                stop=final_stop_strings,
                include_stop_str_in_output=True,  # returning stop strings like hf
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Extract the generated text
            generated_text = final_request_output.outputs[0].text

            # Create result in BatchedDataDict format
            result_batch = BatchedDataDict[GenerationOutputSpec](
                {"texts": [generated_text]}
            )

            return (prompt_idx, result_batch)

        # Create tasks for all prompts and yield results as they complete
        prompt_tasks = [
            asyncio.create_task(process_single_prompt(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        try:
            for completed_task in asyncio.as_completed(prompt_tasks):
                result = await completed_task
                yield result
        finally:
            for task in prompt_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*prompt_tasks, return_exceptions=True)

    async def report_device_id_async(self) -> list[str]:
        """Async version of report_device_id."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id_async can only be used with async_engine=True. Use report_device_id instead."
            )

        result_or_coro = await self.llm.collective_rpc("report_device_id", args=tuple())

        if asyncio.iscoroutine(result_or_coro):
            list_of_worker_results = await result_or_coro
        else:
            list_of_worker_results = result_or_coro

        return cast(list[str], list_of_worker_results)

    async def prepare_refit_info_async(
        self, state_dict_info: dict[str, Any]
    ) -> Optional[list[str]]:
        """Async version of prepare_refit_info."""
        from nemo_rl.models.generation.vllm.quantization import fp8

        results = await self.llm.collective_rpc(
            "prepare_refit_info",
            args=(state_dict_info, fp8.serialize_fp8_config()),
        )
        # Union across the engine's TP/PP workers: with pipeline parallelism
        # each shard only classifies its local parameters as fp8-eligible.
        names = sorted({name for result in results if result for name in result})
        return names or None

    async def _reset_encoder_cache_after_weight_update(self) -> None:
        """Invalidate weight-dependent multimodal encoder outputs when enabled."""
        if not self.cfg["vllm_cfg"].get(
            "reset_encoder_cache_after_weight_update", False
        ):
            return
        assert self.llm is not None
        await self.llm.reset_encoder_cache()

    async def update_weights_via_ipc_zmq_async(
        self,
    ) -> bool:
        """Async version of update_weights_via_ipc_zmq."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_via_ipc_zmq_async can only be used with async_engine=True. Use update_weights_via_ipc_zmq instead."
                )

            # TODO: switch to update_weights_from_local_ipc_handles for better performance once collectively report_device_id is supported in asyncLLM initialization
            result_or_coro = await self.llm.collective_rpc(
                "update_weights_via_ipc_zmq",
                args=tuple(),
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_results = cast(list[bool], worker_results)

            if not worker_results or not all(worker_results):
                print(
                    f"Error: Worker failed to update weights. Results: {worker_results}"
                )
                return False
            await self._reset_encoder_cache_after_weight_update()
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def update_weights_from_collective_async(
        self, refit_timeout_s: float | None = None
    ) -> bool:
        """Async version of update_weights_from_collective."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective_async can only be used with async_engine=True. Use update_weights_from_collective instead."
                )

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_collective",
                args=(refit_timeout_s, self._refit_with_reload_api_enabled()),
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_results = cast(list[bool], worker_results)

            if not worker_results or not all(worker_results):
                print(
                    f"Error: Worker failed to update weights. Results: {worker_results}"
                )
                return False
            await self._reset_encoder_cache_after_weight_update()
            return True
        except Exception as e:
            # Propagate a deliberate abort instead of folding it into `return False`. It
            # is the controller's signal to rebuild over the survivors and retry; reported
            # as a generic failure it just ends the run, which is the wedge this exists to
            # replace.
            #
            # Matched by message, not by type, and that is not belt-and-braces. vLLM's
            # EngineCore RPC stringifies the worker exception and re-raises it client-side
            # as a bare Exception, so the RefitAborted raised inside the engine arrives
            # here as Exception(str) and a plain `except RefitAborted` never fires. Job
            # 6484412 is the proof: the deadline fired, the abort was named in the log, and
            # the run still wedged at step 4 because this handler did not match.
            if is_refit_abort(e):
                raise RefitAborted(str(e)) from e
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def init_nccl_reshard_comm_group_async(
        self,
        rank_prefix: int,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Async version of init_nccl_reshard_comm_group."""
        await self.llm.collective_rpc(
            "init_nccl_reshard_comm_group",
            args=(
                rank_prefix,
                pp_ips,
                pp_ports,
                pp_size,
                train_ranks_per_stage,
                sub_world_size,
            ),
        )

    async def prepare_nccl_reshard_refit_info_async(self, refit_info: dict) -> None:
        """Async version of prepare_nccl_reshard_refit_info."""
        await self.llm.collective_rpc(
            "prepare_nccl_reshard_refit_info", args=(refit_info,)
        )

    async def nccl_reshard_refit_async(
        self, refit_timeout_s: Optional[float] = None
    ) -> bool:
        """Async version of nccl_reshard_refit."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            result_or_coro = await self.llm.collective_rpc(
                "nccl_reshard_refit", args=(refit_timeout_s,)
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed nccl_reshard_refit. Result: {worker_result}"
                )
                return False
            await self._reset_encoder_cache_after_weight_update()
            return True
        except Exception as e:
            # Propagate a deliberate abort instead of folding it into `return False`. It
            # is the controller's signal to rebuild over the survivors and retry; reported
            # as a generic failure it just ends the run, which is the wedge this exists to
            # replace.
            #
            # Matched by message, not by type, and that is not belt-and-braces. vLLM's
            # EngineCore RPC stringifies the worker exception and re-raises it client-side
            # as a bare Exception, so the RefitAborted raised inside the engine arrives
            # here as Exception(str) and a plain `except RefitAborted` never fires. Job
            # 6484412 is the proof: the deadline fired, the abort was named in the log, and
            # the run still wedged at step 4 because this handler did not match.
            if is_refit_abort(e):
                raise RefitAborted(str(e)) from e
            print(f"Exception during nccl_reshard_refit: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return False

    async def reset_prefix_cache_async(self):
        """Async version of reset_prefix_cache."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache_async can only be used with async_engine=True. Use reset_prefix_cache instead."
            )

        await self.llm.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    async def pause_generation_async(self, *, clear_cache: bool) -> bool:
        """Pause vLLM generation for an in-flight weight update."""
        assert self.llm is not None, (
            "Attempting to pause generation with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "pause_generation_async can only be used with async_engine=True"
            )

        await self.llm.pause_generation(mode="keep", clear_cache=clear_cache)
        return True

    async def flush_obsolete_generation_cut_rows_async(self, epoch: int) -> bool:
        """Delete the chunk rows of calls that ended before snapshot ``epoch``.

        Returns once the delete is done: TQ must not clear rows while the
        snapshot is saved.
        """
        await self._prefix_recovery.flush(epoch)
        return True

    def get_generation_prefix_counts(self) -> dict[str, int]:
        """Generation-prefix restore and progress-read counts; see PrefixRecovery."""
        return self._prefix_recovery.counts()

    async def resume_generation_async(self) -> bool:
        """Resume vLLM generation after an in-flight weight update."""
        assert self.llm is not None, (
            "Attempting to resume generation with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "resume_generation_async can only be used with async_engine=True"
            )

        await self.llm.resume_generation()
        return True

    async def sleep_async(self):
        """Async version of sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep_async can only be used with async_engine=True. Use sleep instead."
            )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        await self.llm.reset_prefix_cache()
        # Reset the multimodal processor cache (sender side) so it stays in
        # sync with the receiver cache that vLLM clears internally during
        # sleep.  Without this, the sender thinks images are already cached on
        # the receiver and sends data=None, causing an assertion error.
        if hasattr(self.llm, "reset_mm_cache"):
            await self.llm.reset_mm_cache()
        await self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    async def wake_up_async(self, **kwargs):
        """Async version of wake_up."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up_async can only be used with async_engine=True. Use wake_up instead."
            )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        await self.llm.wake_up(**wake_up_args)

    async def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
            # Waiting for a wedged TQ write here would make actor shutdown hang.
            prefix_recovery = getattr(self, "_prefix_recovery", None)
            if prefix_recovery is not None:
                prefix_recovery.shutdown()
            if self.server_thread is not None:
                self.http_server.should_exit = True
                await asyncio.to_thread(self.server_thread.join)
                self.server_thread = None

            if self._sparse_refit_receiver is not None:
                await asyncio.to_thread(self._sparse_refit_receiver.shutdown)

            if self.llm is not None:
                # Clean up extension resources (e.g., ZMQ sockets)
                await self.llm.collective_rpc("cleanup", args=tuple())
                try:
                    self.llm.shutdown()
                except Exception as e_stop:
                    print(f"Error calling shutdown_background_loop: {e_stop}")

                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False
        finally:
            # Flush buffered spans before the actor goes away, off the event
            # loop: the export blocks for up to 5s and this async actor's
            # in-flight generate requests share the loop. Shielded because a
            # cancel here is cleanup being interrupted.
            try:
                await asyncio.shield(asyncio.to_thread(shutdown_telemetry))
            except asyncio.CancelledError:
                pass


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    pass
