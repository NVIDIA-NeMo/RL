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
import copy
import gc
import logging
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI

from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.checkpoint_engine import (
    VllmAsyncCheckpointEngineRpcMixin,
)
from nemo_rl.models.generation.vllm.collective_rpc import (
    resolve_collective_rpc_result,
)
from nemo_rl.models.generation.vllm.config import parse_nvfp4_pertoken_rollout
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    attach_token_information_to_chat_response_choices,
    extract_selected_token_logprobs,
    format_prompt_for_vllm_generation,
    model_dump_chat_response_with_dynamic_message_fields,
    pad_and_align_routed_expert_indices,
)
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker
from nemo_rl.models.generation.openai_server_utils import (
    replace_prefix_tokens,
)
from nemo_rl.telemetry.setup import shutdown_telemetry

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0


@dataclass
class _RequestCaptureBuffer:
    """One append-only generation segment owned by an in-flight request."""

    sequence: int
    generated_token_ids: list[int] = field(default_factory=list)
    generated_logprobs: list[float] = field(default_factory=list)


@dataclass
class _FrozenRequestCaptureBuffer:
    """An active buffer detached for a checkpoint write."""

    checkpoint_id: str
    buffer: _RequestCaptureBuffer


@dataclass
class _RequestCaptureState:
    """Append-only token buffers for one in-flight captured request.

    vLLM publishes delta progress. ``observe`` appends each delta to the current
    active buffer exactly once. A checkpoint swaps the active buffer under
    ``lock`` and performs its blocking TQ write after releasing the lock, so
    later observations can continue in a fresh buffer.

    Each successful cut stages only the detached buffer. The ordered TQ keys
    remain in Gym lineage, while the sealed token arrays are retained locally
    only to compute the cumulative digest advertised by the latest cut.
    """

    call: Any
    prompt_token_ids: list[int]
    resumed_generation_token_ids: list[int] = field(default_factory=list)
    observation_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Serializes lifecycle-changing TQ operations for this call. Token
    # observation deliberately uses ``lock`` instead so decoding can continue
    # while a detached generation chunk is staged.
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_started: bool = False
    sealed_generated_token_ids: list[int] = field(default_factory=list)
    sealed_generated_logprobs: list[float] = field(default_factory=list)
    generation_cut_staging_keys: list[str] = field(default_factory=list)
    active_buffer: _RequestCaptureBuffer = field(
        default_factory=lambda: _RequestCaptureBuffer(sequence=0)
    )
    frozen_buffer: _FrozenRequestCaptureBuffer | None = None
    observed_generation_token_count: int = 0
    next_buffer_sequence: int = 1
    # Set by ``observe`` when the active buffer passes the periodic-flush
    # threshold. Cleared by whichever flush actually detaches that buffer, so a
    # request is queued at most once per unstaged segment.
    periodic_flush_due: bool = False

    def unstaged_token_count(self) -> int:
        """Return tokens held only in memory for this call."""
        return len(self.active_buffer.generated_token_ids)

    def observe(
        self,
        generated_token_ids: list[int],
        generated_logprobs: list[float],
    ) -> None:
        """Append one immutable vLLM output delta."""
        if len(generated_token_ids) != len(generated_logprobs):
            self.observation_error = (
                "generated token IDs and log probabilities must have equal lengths"
            )
            return
        self.active_buffer.generated_token_ids.extend(generated_token_ids)
        self.active_buffer.generated_logprobs.extend(generated_logprobs)
        self.observed_generation_token_count += len(generated_token_ids)
        self.observation_error = None

    def refresh_periodic_flush_due(self, threshold: int) -> None:
        """Recompute whether the current active buffer needs a flush."""
        self.periodic_flush_due = bool(
            threshold > 0
            and self.observation_error is None
            and self.frozen_buffer is None
            and self.unstaged_token_count() >= threshold
        )

    def freeze_for_checkpoint(
        self, checkpoint_id: str
    ) -> tuple[str, int, list[int], list[float], list[int], list[float]]:
        """Swap the active buffer and return its delta plus the stable prefix."""
        if self.frozen_buffer is not None:
            raise RuntimeError(
                "cannot start a generation cut while another cut is in progress"
            )
        buffer = self.active_buffer
        self.active_buffer = _RequestCaptureBuffer(sequence=self.next_buffer_sequence)
        self.next_buffer_sequence += 1
        self.frozen_buffer = _FrozenRequestCaptureBuffer(
            checkpoint_id=checkpoint_id,
            buffer=buffer,
        )
        return (
            f"active/{checkpoint_id}/{buffer.sequence}",
            buffer.sequence,
            list(buffer.generated_token_ids),
            list(buffer.generated_logprobs),
            self.sealed_generated_token_ids + buffer.generated_token_ids,
            self.sealed_generated_logprobs + buffer.generated_logprobs,
        )

    def seal_frozen_buffer(self, checkpoint_id: str) -> None:
        """Adopt a successfully staged frozen buffer into the live prefix."""
        frozen = self._require_frozen_buffer(checkpoint_id)
        self.sealed_generated_token_ids.extend(frozen.generated_token_ids)
        self.sealed_generated_logprobs.extend(frozen.generated_logprobs)
        self.frozen_buffer = None

    def rollback_frozen_buffer(self, checkpoint_id: str) -> None:
        """Restore a failed cut ahead of progress collected after its swap."""
        frozen = self._require_frozen_buffer(checkpoint_id)
        self.active_buffer.generated_token_ids[:0] = frozen.generated_token_ids
        self.active_buffer.generated_logprobs[:0] = frozen.generated_logprobs
        self.frozen_buffer = None

    def _require_frozen_buffer(self, checkpoint_id: str) -> _RequestCaptureBuffer:
        frozen = self.frozen_buffer
        if frozen is None or frozen.checkpoint_id != checkpoint_id:
            raise RuntimeError(
                f"generation cut {checkpoint_id!r} does not own the frozen buffer"
            )
        return frozen.buffer


@dataclass(frozen=True)
class _CompletedCaptureState:
    """Time-bounded terminal evidence retained across the response/cut race."""

    coords: Any
    generation_token_count: int
    completed_at_monotonic: float


@dataclass
class _CompletionOutputDeltaAccumulator:
    """Linear-time accumulator for one vLLM completion index."""

    template: Any | None = None
    text_parts: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[Any] | None = None
    routed_expert_chunks: list[Any] = field(default_factory=list)

    def append(self, output: Any) -> None:
        if self.template is None:
            self.template = copy.copy(output)
        else:
            for name in (
                "cumulative_logprob",
                "finish_reason",
                "stop_reason",
                "lora_request",
            ):
                if hasattr(output, name):
                    setattr(self.template, name, getattr(output, name))
        self.text_parts.append(str(getattr(output, "text", "")))
        self.token_ids.extend(list(getattr(output, "token_ids", ()) or ()))
        delta_logprobs = getattr(output, "logprobs", None)
        if delta_logprobs is not None:
            if self.logprobs is None:
                self.logprobs = []
            self.logprobs.extend(list(delta_logprobs))
        routed = getattr(output, "routed_experts", None)
        if routed is not None:
            self.routed_expert_chunks.append(copy.deepcopy(routed))

    def build(self) -> Any:
        if self.template is None:
            raise RuntimeError("cannot build an empty completion output")
        self.template.text = "".join(self.text_parts)
        self.template.token_ids = self.token_ids
        self.template.logprobs = self.logprobs
        if self.routed_expert_chunks:
            self.template.routed_experts = torch.cat(
                [torch.as_tensor(chunk) for chunk in self.routed_expert_chunks],
                dim=0,
            )
        return self.template


@dataclass
class _RequestOutputDeltaAccumulator:
    """Reconstruct one final RequestOutput from immutable engine deltas."""

    template: Any | None = None
    completions: dict[int, _CompletionOutputDeltaAccumulator] = field(
        default_factory=dict
    )

    def append(self, delta: Any) -> None:
        if self.template is None:
            self.template = copy.copy(delta)
        else:
            previous = self.template
            self.template = copy.copy(delta)
            for name in (
                "prompt",
                "prompt_token_ids",
                "prompt_logprobs",
                "encoder_prompt",
                "encoder_prompt_token_ids",
                "lora_request",
                "num_cached_tokens",
                "prompt_routed_experts",
            ):
                if getattr(self.template, name, None) is None and hasattr(
                    previous, name
                ):
                    setattr(self.template, name, getattr(previous, name))
        for output in getattr(delta, "outputs", ()):
            self.completions.setdefault(
                output.index, _CompletionOutputDeltaAccumulator()
            ).append(output)

    def build(self) -> Any:
        if self.template is None:
            raise RuntimeError("cannot build an empty request output")
        self.template.outputs = [
            self.completions[index].build() for index in sorted(self.completions)
        ]
        return self.template


def _remaining_generation_limits_after_prefix(
    *,
    max_tokens: int | None,
    min_tokens: int | None,
    generation_token_count: int,
) -> tuple[int | None, int | None]:
    """Return output limits for the suffix after restoring generated tokens."""
    if generation_token_count < 0:
        raise ValueError("generation_token_count must be non-negative")
    return (
        None if max_tokens is None else max_tokens - generation_token_count,
        None if min_tokens is None else max(0, min_tokens - generation_token_count),
    )


class _CheckpointCaptureGate:
    """Drain active terminal writes and block new ones across a TQ snapshot."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._open = True
        self._active = 0

    def enter(self) -> None:
        with self._condition:
            while not self._open:
                self._condition.wait()
            self._active += 1

    def exit(self) -> None:
        with self._condition:
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    def close_and_wait(self) -> None:
        with self._condition:
            self._open = False
            while self._active:
                self._condition.wait()

    def reopen(self) -> None:
        with self._condition:
            self._open = True
            self._condition.notify_all()


from nemo_rl.distributed.refit_watchdog import RefitAborted, is_refit_abort


class _AsyncLLMHTTPClient:
    """Keep HTTP generation on the loop that owns AsyncLLM request state.

    The engine-client surface is explicit. Do not add a ``__getattr__`` fallback.
    Add each new member here and decide whether it must run on the engine loop.
    """

    def __init__(self, engine_client: Any, engine_loop: asyncio.AbstractEventLoop):
        self._engine_client = engine_client
        self._engine_loop = engine_loop
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
        return self._generate(prompt, sampling_params, request_id, kwargs)

    async def _generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        iterator = None
        completed = False

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
            if not completed:
                try:
                    await self._run_on_engine_loop(
                        lambda: self._engine_client.abort(request_id)
                    )
                except Exception:
                    LOGGER.exception("Failed to abort vLLM request %s", request_id)

    # These members only read engine status or immutable configuration. Running
    # them on the engine loop added a cross-thread wait to each HTTP request.
    @property
    def errored(self) -> bool:
        return self._engine_client.errored

    @property
    def dead_error(self) -> BaseException:
        return self._engine_client.dead_error

    async def is_tracing_enabled(self) -> bool:
        return await self._engine_client.is_tracing_enabled()


class VllmAsyncGenerationWorkerImpl(
    VllmAsyncCheckpointEngineRpcMixin, BaseVllmGenerationWorker
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

        # Ledger-authoritative token capture (dormant until the
        # setup_token_capture fan-out runs). The weight
        # version is stamped per model call at begin_call time and rotated by
        # the set_rollout_weight_version fan-out from the SC's _sync_weights.
        self.token_capture = None
        self._rollout_weight_version = 0
        # In-flight captured calls keyed by request and by Gym's stable call ID.
        self._capture_calls: dict[int, _RequestCaptureState] = {}
        self._capture_calls_by_model_call_id: dict[str, _RequestCaptureState] = {}
        self._completed_capture_calls: dict[str, _CompletedCaptureState] = {}
        self._generation_cut_receipts: dict[tuple[str, str], Any] = {}
        self._capture_registry_lock = threading.Lock()
        self._capture_sink: Any | None = None
        self._generation_prefix_cuts_enabled = False
        self._generation_cut_control_token: str | None = None
        self._generation_checkpoint_gate = _CheckpointCaptureGate()
        # Periodic chunk flushing. 0 disables it, leaving checkpoint cuts as the
        # only thing that makes an in-flight generation durable.
        self._generation_chunk_flush_tokens = 0
        self._generation_chunk_flush_thread: threading.Thread | None = None
        self._generation_chunk_flush_stop = threading.Event()
        self._generation_chunk_flush_sequence = 0
        # Terminal capture writes use asyncio's shared default executor. Keep
        # checkpoint control on an isolated thread so a closed gate cannot let
        # waiting completions consume every thread needed to create the cut.
        self._generation_checkpoint_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nrl-generation-checkpoint",
        )
        self._staging_source: Any | None = None
        # Guarded by _prefix_cache_lock: _fetch_chain_prefix runs on executor
        # threads (asyncio.to_thread), so lookups/evictions can be concurrent.
        self._prefix_cache: dict[str, list[int]] = {}
        self._prefix_cache_lock = threading.Lock()

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

        self.llm_async_engine_args = AsyncEngineArgs(**llm_kwargs)
        self.stat_loggers = (
            [PrometheusStatLogger]
            if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False)
            else []
        )
        self.llm = AsyncLLM.from_engine_args(
            self.llm_async_engine_args, stat_loggers=self.stat_loggers
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
        from vllm.v1.metrics.reader import Gauge, Counter, get_metrics_snapshot

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
            self._http_engine_client = _AsyncLLMHTTPClient(self.llm, self._engine_loop)
            self.server_thread, self.base_url, self.http_server = (
                self._setup_vllm_server()
            )

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
        generation_prefix_cuts_enabled: bool = False,
        generation_cut_control_token: str | None = None,
        generation_chunk_flush_tokens: int = 0,
    ) -> bool:
        """Host ledger-authoritative token capture in this worker.

        Fan-out target (token_capture.enabled only): builds the in-worker
        data-plane client and TQTokenSink, then makes the single
        ``install_capture`` call wiring Gym's engine-blind capture core +
        vLLM adapter into this worker. Returns whether capture was installed
        (False on non-model-owner ranks, which serve no HTTP).
        """
        if not self.is_model_owner:
            return False
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
        from nemo_gym.token_id_capture.staging import install_capture

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource

        dp_client = build_data_plane_client(dp_cfg, bootstrap=False)
        sink = TQTokenSink(dp_client, staging_partition=staging_partition)
        if generation_prefix_cuts_enabled and not generation_cut_control_token:
            raise ValueError(
                "generation-prefix cuts require a non-empty control bearer token"
            )
        if generation_chunk_flush_tokens < 0:
            raise ValueError("generation_chunk_flush_tokens must not be negative")
        if generation_chunk_flush_tokens and not generation_prefix_cuts_enabled:
            raise ValueError(
                "generation_chunk_flush_tokens requires generation-prefix cuts; "
                "without them a staged chunk has no restore path"
            )
        self._capture_sink = sink
        self._generation_prefix_cuts_enabled = generation_prefix_cuts_enabled
        self._generation_cut_control_token = generation_cut_control_token
        self._generation_chunk_flush_tokens = generation_chunk_flush_tokens
        self._staging_source = TQTokenSource(
            dp_client, staging_partition=staging_partition
        )
        self._prefix_cache.clear()
        install_capture(
            self,
            sink=sink,
            weight_version_fn=lambda: self._rollout_weight_version,
            adapter=VLLMCaptureAdapter(),
        )
        self._start_generation_chunk_flusher()
        return True

    async def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run owner-local checkpoint I/O without blocking the actor event loop."""
        return await asyncio.to_thread(run_checkpoint_command, body)

    async def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the weight version stamped on subsequent captured calls."""
        self._rollout_weight_version = int(version)

    def _capture_admission(self, request: Any) -> Any | None:
        """Parse the ledger's ``ng_capture`` context into a ``CaptureAdmission``.

        Returns None unless capture is installed and the request carries the
        context. The dict itself is never mutated: the admission is the typed,
        read-only contract that the prefix resolution and ``begin_call`` share.
        """
        context = getattr(request, "ng_capture", None)
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
        generation_cut: Any | None = None,
        resumed_generation_token_ids: list[int] | None = None,
    ) -> None:
        """Admit one ledger-forwarded call into the capture layer.

        Called from preprocess_chat once the exact engine prompt is known
        (post-splice in token-in mode, full render in text mode). No-op
        unless capture is installed and the request carries the ledger's
        ``ng_capture`` context.

        ``prefix_token_ids`` is the prefix resolved by
        :meth:`_resolve_admission_prefix`; Gym's ``begin_call`` checks it
        against the admission (length == ``prev_len``, equal to an inline
        prefix) and requires it for a ``staging_chain`` admission.
        """
        capture = self.token_capture
        if capture is None:
            return
        if admission is None:
            admission = self._capture_admission(request)
            if admission is None:
                return
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
        state = _RequestCaptureState(
            call=call,
            prompt_token_ids=list(prompt_token_ids),
            resumed_generation_token_ids=list(resumed_generation_token_ids or ()),
            generation_cut_staging_keys=(
                list(admission.generation_cut.staging_keys)
                if admission.generation_cut is not None
                else []
            ),
        )
        with self._capture_registry_lock:
            if call.model_call_id in self._capture_calls_by_model_call_id:
                raise RuntimeError(
                    f"model call {call.model_call_id!r} is already active in token capture"
                )
            self._capture_calls[id(request)] = state
            self._capture_calls_by_model_call_id[call.model_call_id] = state

    def _observe_request_capture(self, request: Any, request_output: Any) -> None:
        """Append one request's immutable vLLM output delta."""
        with self._capture_registry_lock:
            state = self._capture_calls.get(id(request))
        if state is None:
            return
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return
        try:
            generation_token_ids = list(getattr(outputs[0], "token_ids", ()) or ())
            generation_logprobs = extract_selected_token_logprobs(outputs[0])
        except (RuntimeError, TypeError, ValueError) as error:
            with state.lock:
                state.observation_error = f"{type(error).__name__}: {error}"
            return
        threshold = self._generation_chunk_flush_tokens
        with state.lock:
            state.observe(generation_token_ids, generation_logprobs)
            # Marking is deliberately all this hook does: it runs on the event
            # loop that drives generation, so the blocking staging write belongs
            # on the background flush thread.
            state.refresh_periodic_flush_due(threshold)

    def _restore_response_prefix(
        self, request: Any, request_output: Any, *, tokenizer: Any
    ) -> None:
        """Prepend the durable assistant prefix before vLLM parses the response."""
        state = self._get_request_capture(request)
        if state is None or not state.resumed_generation_token_ids:
            return
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return
        output = outputs[0]
        with state.lock:
            output.text = tokenizer.decode(
                state.resumed_generation_token_ids
                + list(getattr(output, "token_ids", ()) or ())
            )

    def _pop_request_capture(self, request: Any) -> _RequestCaptureState | None:
        with self._capture_registry_lock:
            state = self._capture_calls.pop(id(request), None)
            if state is not None:
                self._capture_calls_by_model_call_id.pop(state.call.model_call_id, None)
            return state

    def _get_request_capture(self, request: Any) -> _RequestCaptureState | None:
        with self._capture_registry_lock:
            return self._capture_calls.get(id(request))

    def _remember_completed_capture(
        self, model_call_id: str, coords: Any, generation_token_count: int
    ) -> None:
        now = time.monotonic()
        with self._capture_registry_lock:
            # Dict insertion order is completion order. Remove only entries old
            # enough that the response/inventory race has certainly elapsed;
            # never discard fresh evidence because the deployment crossed an
            # arbitrary call-count threshold.
            while self._completed_capture_calls:
                oldest_id = next(iter(self._completed_capture_calls))
                oldest = self._completed_capture_calls[oldest_id]
                if (
                    now - oldest.completed_at_monotonic
                    <= _COMPLETED_CAPTURE_RETENTION_S
                ):
                    break
                self._completed_capture_calls.pop(oldest_id)
            self._completed_capture_calls.pop(model_call_id, None)
            self._completed_capture_calls[model_call_id] = _CompletedCaptureState(
                coords=coords,
                generation_token_count=generation_token_count,
                completed_at_monotonic=now,
            )

    def _fetch_chain_prefix(self, staging_chain: list[str]) -> list[int]:
        """Assemble prefix token ids from staging_chain, with a worker-local LRU cache."""
        cache = self._prefix_cache
        with self._prefix_cache_lock:
            cached_ids: list[int] = []
            miss_start = 0
            for i, key in enumerate(staging_chain):
                if key in cache:
                    cached_ids = cache[key]
                    miss_start = i + 1
            miss_keys = staging_chain[miss_start:]
        if not miss_keys:
            return list(cached_ids)
        if self._staging_source is None:
            raise RuntimeError(
                "_staging_source not initialized; call setup_token_capture() first"
            )
        # TQ read stays outside the lock so concurrent fetches overlap.
        fetched = self._staging_source.fetch_prefix_token_ids(miss_keys)
        result = cached_ids + fetched
        last_key = staging_chain[-1]
        with self._prefix_cache_lock:
            cache[last_key] = result
            if len(cache) > 256:
                del cache[next(iter(cache))]
        return result

    def _resolve_admission_prefix(self, admission: Any) -> list[int]:
        """Resolve a ``CaptureAdmission`` to the flat prefix the engine prompt starts with.

        A ``staging_chain`` is fetched through the cached TransferQueue read;
        an inline ``required_prefix_token_ids`` is used as is; a text root has
        no prefix. Length checks are Gym's: ``begin_call`` rejects a prefix
        that does not match ``prev_len``.
        """
        if admission.mode == "text":
            return []
        if admission.staging_chain:
            return self._fetch_chain_prefix(list(admission.staging_chain))
        return list(admission.required_prefix_token_ids)

    def _resolve_generation_cut(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> Any | None:
        """Fetch and rebuild the cumulative staged prefix named by an admission."""
        continuation = admission.generation_cut
        if continuation is None:
            return None
        if self._staging_source is None:
            raise RuntimeError(
                "_staging_source not initialized; call setup_token_capture() first"
            )
        staging_keys = list(continuation.staging_keys)
        snapshots = self._staging_source.fetch(staging_keys)
        if len(snapshots) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        if not snapshots:
            raise RuntimeError("generation-cut continuation has no staged chunks")
        weight_versions = [snapshot.weight_version for snapshot in snapshots]
        if weight_versions != sorted(weight_versions):
            raise RuntimeError(
                "generation-cut chunk policy versions are not monotonically non-decreasing"
            )
        current_weight_version = self._rollout_weight_version
        if weight_versions[-1] > current_weight_version:
            raise RuntimeError(
                "generation-cut prefix contains policy version "
                f"{weight_versions[-1]}, newer than the current rollout version "
                f"{current_weight_version}"
            )
        token_ids_delta = [
            token_id for snapshot in snapshots for token_id in snapshot.token_ids_delta
        ]
        token_mask_delta = [
            mask for snapshot in snapshots for mask in snapshot.token_mask_delta
        ]
        generation_logprobs_delta = [
            logprob
            for snapshot in snapshots
            for logprob in snapshot.generation_log_probs_delta
        ]
        if any(
            mask == 0.0
            for snapshot in snapshots[1:]
            for mask in snapshot.token_mask_delta
        ):
            raise RuntimeError(
                "only the first generation-cut chunk may contain prompt tokens"
            )

        # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
        from nemo_gym.token_id_capture.staging.digest import (
            EXTRAS_DIGEST_VERSION,
            STAGING_DIGEST_VERSION,
            compute_chain_hash,
            compute_extras_digest,
            compute_staging_digest,
            hash_token_ids,
        )
        from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot

        delta_len = len(token_ids_delta)
        cum_len = admission.prev_len + delta_len
        # A cut chain may span retries performed after later weight updates.
        # Its token-level behavior logprobs remain attached to their original
        # chunks. The cumulative snapshot uses the oldest contributing version
        # so downstream replay-buffer staleness checks remain conservative.
        weight_version = weight_versions[0]
        extras_digest = compute_extras_digest(None)
        chain_hash = compute_chain_hash(admission.parent_chain_hash, token_ids_delta)
        cumulative_hash = hash_token_ids(prefix_token_ids + token_ids_delta)
        digest = compute_staging_digest(
            schema_version=admission.schema_version,
            digest_version=STAGING_DIGEST_VERSION,
            extras_digest_version=EXTRAS_DIGEST_VERSION,
            rollout_id=continuation.source_capture_key,
            model_call_id=continuation.source_model_call_id,
            parent_call_id=admission.parent_call_id,
            mode=admission.mode,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=cum_len,
            weight_version=weight_version,
            token_ids_delta=token_ids_delta,
            token_mask_delta=token_mask_delta,
            generation_log_probs_delta=generation_logprobs_delta,
            extras_digest=extras_digest,
            chain_hash=chain_hash,
            cumulative_hash=cumulative_hash,
        )
        generation_token_count = sum(mask == 1.0 for mask in token_mask_delta)
        if (
            generation_token_count != continuation.generation_token_count
            or digest != continuation.digest
        ):
            raise RuntimeError(
                "generation-cut checkpoint coordinates do not match the staged prefix"
            )
        snapshot = StagedCallBaseSnapshot(
            rollout_id=continuation.source_capture_key,
            model_call_id=continuation.source_model_call_id,
            parent_call_id=admission.parent_call_id,
            mode=admission.mode,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=cum_len,
            weight_version=weight_version,
            digest=digest,
            token_ids_delta=token_ids_delta,
            token_mask_delta=token_mask_delta,
            generation_log_probs_delta=generation_logprobs_delta,
            extras_digest=extras_digest,
            chain_hash=chain_hash,
            cumulative_hash=cumulative_hash,
        )
        LOGGER.info(
            "generation prefix restored: rollout_id=%s model_call_id=%s "
            "source_model_call_id=%s prefix_tokens=%d prefix_digest=%s "
            "weight_version_span=[%d,%d]",
            admission.rollout_id,
            admission.model_call_id,
            continuation.source_model_call_id,
            continuation.generation_token_count,
            continuation.digest,
            weight_versions[0],
            weight_versions[-1],
        )
        return snapshot

    def _enter_request_prefix(self, request: Any, prefix_token_ids: list[int]) -> None:
        """Attach the resolved prefix to the request through the capture adapter.

        ``VLLMCaptureAdapter.enter_prefix`` writes the engine-native field
        (``required_prefix_token_ids``) into a payload; the same fields are
        applied to the pydantic request so the existing prefix-splice branch
        of preprocess_chat handles staged and inline prefixes alike.
        """
        adapter = self.token_capture.adapter
        for field_name, value in adapter.enter_prefix({}, prefix_token_ids).items():
            setattr(request, field_name, value)

    @staticmethod
    def _delta_align_routed_experts(
        payload: dict[str, Any], *, prev_len: int, prompt_len: int, generated_len: int
    ) -> None:
        """Normalize optional vLLM routes to the exact staged token delta."""
        choices = payload.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            return
        choice = dict(choices[0])
        message = dict(choice.get("message") or {})
        routed = message.get("routed_experts")
        if routed is None:
            return
        try:
            from nemo_rl.utils.routed_experts_codec import (
                decode_routed_experts,
                encode_routed_experts,
            )

            if isinstance(routed, str):
                dtype_name = routed.split(":", 3)[1]
                dtype = {
                    "int8": torch.int8,
                    "int16": torch.int16,
                    "int32": torch.int32,
                }.get(dtype_name)
                if dtype is None:
                    raise ValueError(f"unsupported routed_experts dtype {dtype_name!r}")
            else:
                dtype = torch.int16
            experts = decode_routed_experts(routed, dtype)
            expected_full_len = prompt_len + generated_len
            if experts.dim() != 3 or experts.shape[0] != expected_full_len:
                raise ValueError(
                    f"route length {experts.shape[0]} does not match engine sequence "
                    f"length {expected_full_len}"
                )
            message["routed_experts"] = encode_routed_experts(experts[prev_len:])
        except (IndexError, TypeError, ValueError) as error:
            LOGGER.warning(
                "dropping invalid routed_experts from staged capture: %s", error
            )
            message.pop("routed_experts", None)
        choice["message"] = message
        payload["choices"] = [choice]

    # ── periodic generation-chunk flushing ──────────────────────────────────

    def _start_generation_chunk_flusher(self) -> None:
        """Run periodic chunk flushing on its own daemon thread."""
        if self._generation_chunk_flush_tokens <= 0:
            return
        if self._generation_chunk_flush_thread is not None:
            return
        self._generation_chunk_flush_stop.clear()
        thread = threading.Thread(
            target=self._generation_chunk_flush_loop,
            name="nemo-rl-generation-chunk-flush",
            daemon=True,
        )
        self._generation_chunk_flush_thread = thread
        thread.start()

    def stop_generation_chunk_flusher(self) -> None:
        """Stop the flush thread. Idempotent; safe before startup."""
        self._generation_chunk_flush_stop.set()
        thread = self._generation_chunk_flush_thread
        self._generation_chunk_flush_thread = None
        if thread is not None:
            thread.join(timeout=5.0)

    def _generation_chunk_flush_loop(self) -> None:
        while not self._generation_chunk_flush_stop.wait(0.25):
            try:
                self.flush_due_generation_chunks()
            except Exception:  # noqa: BLE001 — a flush failure must not end the loop
                LOGGER.exception("periodic generation-chunk flush pass failed")

    def flush_due_generation_chunks(self) -> int:
        """Stage every in-flight call whose unstaged segment passed the bound.

        Returns the number of calls staged. Mirrors the checkpoint-cut staging
        path, minus the acknowledgement it does not have to produce: freeze the
        active buffer, stage the detached segment, then seal it on success or
        roll it back so the next attempt retries the same tokens.
        """
        if self._generation_chunk_flush_tokens <= 0:
            return 0
        with self._capture_registry_lock:
            states = list(self._capture_calls.values())

        staged = 0
        for state in states:
            with state.lock:
                if not state.periodic_flush_due:
                    continue
                # A cut owns the frozen buffer, or the call regressed. Leave the
                # mark set so the next pass reconsiders it.
                if state.frozen_buffer is not None or state.observation_error:
                    continue
                if state.unstaged_token_count() < self._generation_chunk_flush_tokens:
                    state.periodic_flush_due = False
                    continue
            self._generation_chunk_flush_sequence += 1
            flush_id = f"periodic-{self._generation_chunk_flush_sequence:012d}"
            # Ride the same gate as terminal writes so a TQ snapshot drains and
            # then blocks periodic staging exactly as it does completions.
            self._generation_checkpoint_gate.enter()
            try:
                if self._flush_generation_chunk(state, flush_id):
                    staged += 1
            except Exception:  # noqa: BLE001 — one bad call must not stall the rest
                LOGGER.exception("periodic generation-chunk flush failed")
            finally:
                self._generation_checkpoint_gate.exit()
        return staged

    def _flush_generation_chunk(self, state: Any, flush_id: str) -> bool:
        """Stage one call's unstaged segment. Returns whether anything staged."""
        with state.lifecycle_lock:
            # The registry snapshot in ``flush_due_generation_chunks`` can be
            # stale. Once terminal processing owns the call, it will write the
            # canonical row and this detached prefix must not be staged.
            if state.terminal_started:
                return False
            return self._flush_generation_chunk_with_lifecycle_owned(state, flush_id)

    def _flush_generation_chunk_with_lifecycle_owned(
        self, state: Any, flush_id: str
    ) -> bool:
        """Stage one generation chunk while owning ``state.lifecycle_lock``."""
        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            return False

        with state.lock:
            if state.frozen_buffer is not None or state.observation_error:
                return False
            (
                _frozen_buffer_id,
                chunk_sequence,
                chunk_token_ids,
                chunk_logprobs,
                generated_token_ids,
                generated_logprobs,
            ) = state.freeze_for_checkpoint(flush_id)
            state.periodic_flush_due = False
            first_chunk = not state.generation_cut_staging_keys

        if not chunk_token_ids:
            with state.lock:
                state.rollback_frozen_buffer(flush_id)
                state.refresh_periodic_flush_due(
                    self._generation_chunk_flush_tokens
                )
            return False

        staged_key = None
        try:
            # The first staged row for a call is cumulative; later rows are
            # deltas. A cut and a periodic flush share that numbering, so the
            # two paths interleave without renumbering anything.
            if first_chunk:
                record = capture.build_prefix_record(
                    state.call,
                    prompt_token_ids=state.prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=generated_logprobs,
                )
            else:
                record = capture.build_generation_chunk_record(
                    state.call,
                    generated_token_ids=chunk_token_ids,
                    generated_logprobs=chunk_logprobs,
                )
            result = sink.stage_generation_prefix(
                record,
                checkpoint_id=flush_id,
                chunk_sequence=chunk_sequence,
            )
            staged_key = result.staging_key
            if not result.ok or staged_key is None:
                raise RuntimeError(
                    f"periodic generation-chunk staging failed: {result.error}"
                )
            with state.lock:
                state.seal_frozen_buffer(flush_id)
                state.generation_cut_staging_keys.append(staged_key)
                state.refresh_periodic_flush_due(
                    self._generation_chunk_flush_tokens
                )
        except Exception:
            with state.lock:
                frozen = state.frozen_buffer
                if frozen is not None and frozen.checkpoint_id == flush_id:
                    state.rollback_frozen_buffer(flush_id)
                    state.refresh_periodic_flush_due(
                        self._generation_chunk_flush_tokens
                    )
            if staged_key is not None:
                try:
                    sink.clear([staged_key])
                except Exception:  # noqa: BLE001 — preserve the flush failure
                    LOGGER.exception(
                        "failed to clear rejected periodic chunk row %s", staged_key
                    )
            raise
        return True

    def _finish_request_capture(self, request: Any, content: dict) -> dict:
        """Run terminal token staging outside an active checkpoint cut."""
        self._generation_checkpoint_gate.enter()
        try:
            return self._finish_request_capture_after_checkpoint_gate(request, content)
        finally:
            self._generation_checkpoint_gate.exit()

    def _finish_request_capture_after_checkpoint_gate(
        self, request: Any, content: dict
    ) -> dict:
        """Stage the finished call and ride its coords on the response.

        Fail-closed: the sink write happens inside complete_call —
        the coords exist only after the bytes are durable, and any capture
        failure degrades to capture_failed coords without breaking the
        completion. Token ids and logprobs are stripped: the staged delta is
        the only token store on this path, so the worker->gate hop carries
        text + delta ids + coords only.
        """
        state = self._get_request_capture(request)
        if state is None:
            return content
        with state.lifecycle_lock:
            if state.terminal_started:
                raise RuntimeError(
                    f"model call {state.call.model_call_id!r} already started "
                    "terminal token capture"
                )
            state.terminal_started = True
            content, obsolete_staging_keys = (
                self._finish_request_capture_with_lifecycle_owned(
                    state, request, content
                )
            )

        # The canonical terminal row now contains the complete call. Prefix
        # chunks are no longer needed in the live staging partition. Cleanup is
        # best-effort: a later checkpoint inventory pass also removes orphaned
        # generation-cut rows.
        sink = self._capture_sink
        if obsolete_staging_keys and sink is not None:
            try:
                sink.clear(list(obsolete_staging_keys))
            except Exception:  # noqa: BLE001 — do not lose a valid completion
                LOGGER.exception(
                    "failed to clear obsolete generation chunks for model call %s",
                    state.call.model_call_id,
                )
        return content

    def _finish_request_capture_with_lifecycle_owned(
        self, state: Any, request: Any, content: dict
    ) -> tuple[dict, tuple[str, ...]]:
        """Stage a terminal call while owning ``state.lifecycle_lock``."""
        call = state.call
        prompt_token_ids = state.prompt_token_ids
        payload = dict(content)
        # vLLM's OpenAI response carries no prompt ids; the adapter reads the
        # preprocess-time engine prompt off the payload (see
        # nemo_gym.token_id_capture.adapters.vllm.extract_prompt_ids).
        payload["prompt_token_ids"] = prompt_token_ids
        adapter = self.token_capture.adapter
        generated_token_ids: list[int] = []
        if adapter is not None:
            try:
                generated_token_ids, _ = adapter.extract_generation(payload)
            except Exception:  # capture core will report the authoritative failure
                generated_token_ids = []
            self._delta_align_routed_experts(
                payload,
                prev_len=call.admission.prev_len,
                prompt_len=len(prompt_token_ids),
                generated_len=len(generated_token_ids),
            )
        coords = self.token_capture.complete_call_from_response(call, payload)
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
                call.rollout_id,
                call.model_call_id,
                call.generation_cut.model_call_id,
                prefix_tokens,
                len(generated_token_ids),
                prefix_tokens + len(generated_token_ids),
            )
        self._remember_completed_capture(
            call.model_call_id,
            coords,
            total_generation_token_count,
        )
        self._pop_request_capture(request)
        obsolete_staging_keys = (
            tuple(state.generation_cut_staging_keys)
            if coords.disposition == "staged"
            else ()
        )
        for choice in content.get("choices") or []:
            choice.pop("logprobs", None)
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
        return content, obsolete_staging_keys

    def _abort_request_capture(self, request: Any, *, reason: str) -> None:
        """Drop the in-flight capture state for a request that errored."""
        state = self._get_request_capture(request)
        if state is None:
            return
        with state.lifecycle_lock:
            if state.terminal_started:
                return
            state.terminal_started = True
            popped = self._pop_request_capture(request)
            if popped is not state or self.token_capture is None:
                return
            coords = self.token_capture.fail_call(state.call, reason=reason)
            self._remember_completed_capture(state.call.model_call_id, coords, 0)

    async def _run_generation_checkpoint_control(
        self,
        operation: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """Run cut control independently of blocked terminal capture writes."""
        return await asyncio.get_running_loop().run_in_executor(
            self._generation_checkpoint_executor,
            operation,
            *args,
        )

    def _completed_generation_cut_ack(self, prefix: Any, checkpoint_id: str) -> Any:
        """Describe a call whose terminal path won the cut/abort race."""
        from nemo_gym._checkpoint.model_control_contracts import (
            GenerationCutPrefixAck,
        )
        from nemo_gym.token_id_capture.staging.records import staging_key

        with self._capture_registry_lock:
            completed = self._completed_capture_calls.get(prefix.model_call_id)
        expected_capture_key = (
            prefix.rollout_id
            if prefix.attempt_index == 0
            else f"{prefix.rollout_id}-a{prefix.attempt_index}"
        )
        if (
            completed is not None
            and completed.coords.rollout_id != expected_capture_key
        ):
            raise RuntimeError(
                "generation-prefix inventory identity does not match the "
                f"completed call: model_call_id={prefix.model_call_id!r}"
            )
        if completed is not None and completed.coords.disposition == "staged":
            return GenerationCutPrefixAck(
                **prefix.model_dump(mode="json"),
                disposition="durable_prefix",
                cut_kind="terminal_completion",
                frozen_buffer_id=f"terminal/{checkpoint_id}",
                staging_keys=(completed.coords.staging_key,),
                prefix_token_count=completed.generation_token_count,
                prefix_digest=completed.coords.digest,
            )
        # The in-memory cache is only an optimization. Its durable canonical
        # row remains authoritative if the race evidence aged out before a
        # delayed checkpoint inventory arrived.
        if completed is None and self._staging_source is not None:
            canonical_key = staging_key(expected_capture_key, prefix.model_call_id)
            try:
                snapshots = self._staging_source.fetch([canonical_key])
            except KeyError:
                snapshots = []
            if snapshots:
                if len(snapshots) != 1:
                    raise RuntimeError(
                        "terminal completion lookup returned an unexpected row count"
                    )
                snapshot = snapshots[0]
                if (
                    snapshot.rollout_id != expected_capture_key
                    or snapshot.model_call_id != prefix.model_call_id
                ):
                    raise RuntimeError(
                        "generation-prefix inventory identity does not match the "
                        f"durable terminal row: model_call_id={prefix.model_call_id!r}"
                    )
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_prefix",
                    cut_kind="terminal_completion",
                    frozen_buffer_id=f"terminal/{checkpoint_id}",
                    staging_keys=(canonical_key,),
                    prefix_token_count=sum(
                        mask == 1.0 for mask in snapshot.token_mask_delta
                    ),
                    prefix_digest=snapshot.digest,
                )
        return GenerationCutPrefixAck(
            **prefix.model_dump(mode="json"),
            disposition="durable_failure",
        )

    def _checkpoint_active_generation_cut(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
    ) -> Any | None:
        """Cut one live call, or return ``None`` if abort won the race."""
        from nemo_gym._checkpoint.model_control_contracts import (
            GenerationCutPrefixAck,
        )

        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")

        # Abort and terminal completion use the same lifecycle lock. Once the
        # cut owns it, the call cannot be removed until its detached buffer is
        # staged or rolled back. If abort won first, use its terminal evidence.
        with state.lifecycle_lock:
            with self._capture_registry_lock:
                current = self._capture_calls_by_model_call_id.get(
                    prefix.model_call_id
                )
            if current is not state or state.terminal_started:
                return None

            expected_capture_key = (
                prefix.rollout_id
                if prefix.attempt_index == 0
                else f"{prefix.rollout_id}-a{prefix.attempt_index}"
            )
            if state.call.rollout_id != expected_capture_key:
                raise RuntimeError(
                    "generation-prefix inventory identity does not match the "
                    f"active call: model_call_id={prefix.model_call_id!r}, "
                    f"inventory_rollout_id={prefix.rollout_id!r}, "
                    f"worker_rollout_id={state.call.rollout_id!r}"
                )

            with state.lock:
                observation_error = state.observation_error
                if observation_error is None:
                    (
                        frozen_buffer_id,
                        chunk_sequence,
                        chunk_token_ids,
                        chunk_logprobs,
                        generated_token_ids,
                        generated_logprobs,
                    ) = state.freeze_for_checkpoint(checkpoint_id)
            if observation_error is not None:
                raise RuntimeError(
                    f"cannot cut model call {prefix.model_call_id!r}: "
                    f"{observation_error}"
                )
            inherited_generation_token_count = (
                state.call.admission.generation_cut.generation_token_count
                if state.call.admission.generation_cut is not None
                else 0
            )
            total_generation_token_count = inherited_generation_token_count + len(
                generated_token_ids
            )
            if total_generation_token_count == 0:
                with state.lock:
                    state.rollback_frozen_buffer(checkpoint_id)
                    state.refresh_periodic_flush_due(
                        self._generation_chunk_flush_tokens
                    )
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )

            staged_key = None
            try:
                cumulative_record = capture.build_prefix_record(
                    state.call,
                    prompt_token_ids=state.prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=generated_logprobs,
                )
                if chunk_token_ids:
                    chunk_record = (
                        cumulative_record
                        if not state.generation_cut_staging_keys
                        else capture.build_generation_chunk_record(
                            state.call,
                            generated_token_ids=chunk_token_ids,
                            generated_logprobs=chunk_logprobs,
                        )
                    )
                    result = sink.stage_generation_prefix(
                        chunk_record,
                        checkpoint_id=checkpoint_id,
                        chunk_sequence=chunk_sequence,
                    )
                    staged_key = result.staging_key
                    if not result.ok:
                        raise RuntimeError(
                            "generation-prefix staging failed for "
                            f"{prefix.model_call_id!r}: {result.error}"
                        )
                    if staged_key is None:
                        raise RuntimeError(
                            "generation-prefix staging returned no staging key for "
                            f"{prefix.model_call_id!r}"
                        )
                with state.lock:
                    candidate_staging_keys = tuple(state.generation_cut_staging_keys)
                    if staged_key is not None:
                        candidate_staging_keys += (staged_key,)
                    acknowledgement = GenerationCutPrefixAck(
                        **prefix.model_dump(mode="json"),
                        disposition="durable_prefix",
                        cut_kind="active_prefix",
                        frozen_buffer_id=frozen_buffer_id,
                        staging_keys=candidate_staging_keys,
                        prefix_token_count=total_generation_token_count,
                        prefix_digest=cumulative_record.digest,
                    )
                    # Validate the complete acknowledgement before adopting the
                    # staged chunk. A failed acknowledgement must leave the
                    # frozen buffer available for rollback.
                    state.seal_frozen_buffer(checkpoint_id)
                    if staged_key is not None:
                        state.generation_cut_staging_keys.append(staged_key)
                    state.refresh_periodic_flush_due(
                        self._generation_chunk_flush_tokens
                    )
            except Exception:
                with state.lock:
                    frozen = state.frozen_buffer
                    if frozen is not None and frozen.checkpoint_id == checkpoint_id:
                        state.rollback_frozen_buffer(checkpoint_id)
                        state.refresh_periodic_flush_due(
                            self._generation_chunk_flush_tokens
                        )
                if staged_key is not None:
                    try:
                        sink.clear([staged_key])
                    except Exception:  # noqa: BLE001 — preserve the cut failure
                        LOGGER.exception(
                            "Failed to clear rejected generation-prefix row %s",
                            staged_key,
                        )
                raise
            return acknowledgement

    def _checkpoint_generation_cut(self, inventory: Any) -> Any:
        """Stage a stable prefix for every call named by Gym's frozen inventory."""
        from nemo_gym._checkpoint.model_control_contracts import (
            GenerationCutReceipt,
        )

        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")
        receipt_key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._capture_registry_lock:
            cached_receipt = self._generation_cut_receipts.get(receipt_key)
        if cached_receipt is not None:
            return cached_receipt
        acknowledgements = []
        for prefix in inventory.active_prefixes:
            with self._capture_registry_lock:
                state = self._capture_calls_by_model_call_id.get(prefix.model_call_id)
            if state is None:
                acknowledgements.append(
                    self._completed_generation_cut_ack(
                        prefix, inventory.checkpoint_id
                    )
                )
                continue
            acknowledgement = self._checkpoint_active_generation_cut(
                prefix, state, inventory.checkpoint_id
            )
            if acknowledgement is None:
                acknowledgement = self._completed_generation_cut_ack(
                    prefix, inventory.checkpoint_id
                )
            acknowledgements.append(acknowledgement)
        receipt = GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"worker-{inventory.inventory_digest}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"tq-{inventory.inventory_digest}",
            prefixes=tuple(acknowledgements),
        )
        with self._capture_registry_lock:
            self._generation_cut_receipts[receipt_key] = receipt
            if len(self._generation_cut_receipts) > 256:
                self._generation_cut_receipts.pop(
                    next(iter(self._generation_cut_receipts))
                )
        return receipt

    # ruff: noqa
    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        worker_self = self
        from copy import deepcopy
        from logging import Filter as LoggingFilter
        from logging import LogRecord
        from typing import List, Optional, Union

        from fastapi import Header, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.serve.tokenize.protocol import (
            TokenizeChatRequest,
            TokenizeCompletionRequest,
            TokenizeResponse,
        )
        from vllm.entrypoints.serve.tokenize.serving import (
            ServingTokenization,
        )
        from vllm.renderers.online_renderer import OnlineRenderer
        from vllm.sampling_params import RequestOutputKind
        from vllm.exceptions import VLLMValidationError
        from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        from vllm.v1.engine.async_llm import logger as vllm_async_llm_logger

        maybe_tool_parser_plugin = self.cfg["vllm_cfg"].get("tool_parser_plugin")
        if maybe_tool_parser_plugin:
            ToolParserManager.import_tool_parser(maybe_tool_parser_plugin)

        maybe_reasoning_parser_plugin = self.cfg["vllm_cfg"].get(
            "reasoning_parser_plugin"
        )
        if maybe_reasoning_parser_plugin:
            ReasoningParserManager.import_reasoning_parser(
                maybe_reasoning_parser_plugin
            )

        engine_client = self._http_engine_client
        if engine_client is None:
            raise RuntimeError("The HTTP engine client is not initialized.")
        model_config = self.llm_async_engine_args.create_model_config()
        base_model_paths = [
            BaseModelPath(
                name=model_config.served_model_name, model_path=model_config.model
            ),
            BaseModelPath(name=model_config.model, model_path=model_config.model),
        ]

        openai_serving_models_kwargs = dict(
            engine_client=engine_client,
            base_model_paths=base_model_paths,
            lora_modules=None,
        )
        openai_serving_models = OpenAIServingModels(**openai_serving_models_kwargs)

        class NeMoRLOpenAIChatRequestMixin:
            def model_post_init(self, context):
                # NeMo-Gym specific processing. This is just how NeMo-Gym returns the extra token information.
                if self.required_prefix_token_ids is None:
                    for message in reversed(self.messages):
                        if "prompt_token_ids" in message:
                            self.required_prefix_token_ids = (
                                message["prompt_token_ids"]
                                + message["generation_token_ids"]
                            )
                            break

                return super().model_post_init(context)

        class NeMoRLOpenAIServingMixin:
            @staticmethod
            def _set_max_tokens(request, max_tokens: int) -> None:
                """Set the request's max output tokens.

                Mutates the request in place. Handles both max_completion_tokens (newer OpenAI API)
                and max_tokens (deprecated but still supported by vLLM).
                """
                if request.max_completion_tokens is not None:
                    request.max_completion_tokens = max_tokens
                elif request.max_tokens is not None:
                    request.max_tokens = max_tokens

            def _clamp_max_tokens(
                self, request, request_max_tokens: int, prompt_token_ids: list[int]
            ) -> None:
                """Clamp the request's max output tokens so that input + output <= max_model_len."""
                remaining = self.model_config.max_model_len - len(prompt_token_ids)
                if remaining <= 0:
                    # preserve the literal "context length" in this message to match Gym's overflow handling
                    message = (
                        f"Prompt length ({len(prompt_token_ids)}) fills or exceeds "
                        f"this model's maximum context length ({self.model_config.max_model_len}). "
                        f"No room for output tokens."
                    )
                    LOGGER.warning("Prompt exceeds max_model_len: %s", message)
                    raise VLLMValidationError(
                        message,
                        parameter="input_tokens",
                        value=len(prompt_token_ids),
                    )
                max_tokens = min(request_max_tokens, remaining)
                self._set_max_tokens(request, max_tokens)

            # vLLM 0.25 moved chat preprocessing to
            # OnlineRenderer.preprocess_chat (tool_parser/reasoning_parser were
            # folded into a single `parser`), so this override now applies via
            # the renderer subclass.
            async def preprocess_chat(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=None,
                parser=None,
                *,
                skip_mm_cache: bool = False,
            ):
                for message in messages:
                    if message.get("tool_calls"):
                        message["tool_calls"] = list(message["tool_calls"])

                messages_for_replace_prefix_tokens = deepcopy(messages)

                # Temporarily set to 1 so vLLM's pre-tokenization length check passes;
                # the actual value will be set through _clamp_max_tokens later.
                actual_request_max_tokens = None
                if isinstance(request, NeMoRLChatCompletionRequest):
                    actual_request_max_tokens = (
                        request.max_completion_tokens
                        if request.max_completion_tokens is not None
                        else request.max_tokens
                    )
                    # If max_completion_tokens or max_tokens is not set, we don't need to do _clamp_max_tokens.
                    # So we don't need to set the request's max output tokens to 1 here.
                    if actual_request_max_tokens is not None:
                        self._set_max_tokens(request, 1)

                try:
                    res = await super().preprocess_chat(
                        request=request,
                        messages=messages,
                        default_template=default_template,
                        default_template_content_format=default_template_content_format,
                        default_template_kwargs=default_template_kwargs,
                        tool_dicts=tool_dicts,
                        parser=parser,
                        skip_mm_cache=skip_mm_cache,
                    )
                except (ValueError, VLLMValidationError) as e:
                    if "maximum context length" in str(e):
                        import logging

                        logging.getLogger(__name__).warning(
                            "Prompt exceeds max_model_len: %s", e
                        )
                    raise

                # Token capture: build the admission once, before branching,
                # and resolve its prefix from it (staging_chain -> cached TQ
                # read, inline ids, or nothing for a text root). The
                # ``ng_capture`` dict is never mutated. Off-loop: the chain
                # fetch is a blocking TQ read, and Gym's staging protocol
                # requires the serving host to move blocking staging I/O off
                # its event loop explicitly. The adapter then attaches the
                # prefix to the request, so the inline-prefix branch below is
                # the single splice path for staged and inline prefixes.
                admission = worker_self._capture_admission(request)
                capture_prefix_token_ids: list[int] | None = None
                generation_cut = None
                resumed_generation_token_ids: list[int] = []
                if admission is not None:
                    capture_prefix_token_ids = await asyncio.to_thread(
                        worker_self._resolve_admission_prefix, admission
                    )
                    generation_cut = await asyncio.to_thread(
                        worker_self._resolve_generation_cut,
                        admission,
                        capture_prefix_token_ids,
                    )
                    engine_prefix_token_ids = list(capture_prefix_token_ids)
                    if generation_cut is not None:
                        engine_prefix_token_ids.extend(generation_cut.token_ids_delta)
                        resumed_generation_token_ids = [
                            token_id
                            for token_id, mask in zip(
                                generation_cut.token_ids_delta,
                                generation_cut.token_mask_delta,
                            )
                            if mask == 1.0
                        ]
                        (
                            remaining_output_tokens,
                            remaining_min_tokens,
                        ) = _remaining_generation_limits_after_prefix(
                            max_tokens=actual_request_max_tokens,
                            min_tokens=getattr(request, "min_tokens", None),
                            generation_token_count=(
                                admission.generation_cut.generation_token_count
                            ),
                        )
                        if remaining_output_tokens is not None:
                            if remaining_output_tokens <= 0:
                                raise VLLMValidationError(
                                    "Durable generation prefix already exhausts max_tokens.",
                                    parameter="max_tokens",
                                    value=actual_request_max_tokens,
                                )
                            actual_request_max_tokens = remaining_output_tokens
                        if remaining_min_tokens is not None:
                            request.min_tokens = remaining_min_tokens
                    if engine_prefix_token_ids:
                        worker_self._enter_request_prefix(
                            request, engine_prefix_token_ids
                        )

                if (
                    not hasattr(request, "required_prefix_token_ids")
                    or request.required_prefix_token_ids is None
                ):
                    # Clamp the request's max output tokens so that input + output <= max_model_len.
                    if actual_request_max_tokens is not None:
                        self._clamp_max_tokens(
                            request,
                            actual_request_max_tokens,
                            res[1][0]["prompt_token_ids"],
                        )
                    # Token capture, text mode: the full render is the exact
                    # engine prompt.
                    worker_self._begin_request_capture(
                        request,
                        res[1][0]["prompt_token_ids"],
                        admission=admission,
                        prefix_token_ids=capture_prefix_token_ids,
                        generation_cut=generation_cut,
                        resumed_generation_token_ids=resumed_generation_token_ids,
                    )
                    return res

                model_prefix_token_ids = list(request.required_prefix_token_ids)

                # Token-in splice path — shared by staging_chain and direct prefix.
                last_assistant_message_idx = None
                for i in reversed(range(len(messages_for_replace_prefix_tokens))):
                    if messages_for_replace_prefix_tokens[i]["role"] == "assistant":
                        last_assistant_message_idx = i
                        break

                if last_assistant_message_idx is None:
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens
                    )
                else:
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens[
                            : last_assistant_message_idx + 1
                        ]
                    )

                modified_request = request.model_copy(
                    update={"add_generation_prompt": False}
                )

                corresponding_res = await super().preprocess_chat(
                    request=modified_request,
                    messages=messages_to_last_assistant_message,
                    default_template=default_template,
                    default_template_content_format=default_template_content_format,
                    default_template_kwargs=default_template_kwargs,
                    tool_dicts=tool_dicts,
                    parser=parser,
                    skip_mm_cache=skip_mm_cache,
                )
                actual_corresponding_token_ids = corresponding_res[1][0][
                    "prompt_token_ids"
                ]

                engine_prompt = res[1][0]

                if generation_cut is not None:
                    final_prompt_token_ids = model_prefix_token_ids
                else:
                    final_prompt_token_ids = replace_prefix_tokens(
                        tokenizer=self.renderer.tokenizer,
                        model_prefix_token_ids=model_prefix_token_ids,
                        template_prefix_token_ids=actual_corresponding_token_ids,
                        template_token_ids=engine_prompt["prompt_token_ids"],
                    )

                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

                # Clamp after prefix replacement since the prompt length may have changed.
                if actual_request_max_tokens is not None:
                    self._clamp_max_tokens(
                        request,
                        actual_request_max_tokens,
                        final_prompt_token_ids,
                    )

                # Token capture, token-in mode: the spliced prompt is the
                # exact engine prompt; begin_call re-checks the prefix it
                # was spliced from against the admission.
                worker_self._begin_request_capture(
                    request,
                    final_prompt_token_ids,
                    admission=admission,
                    prefix_token_ids=capture_prefix_token_ids,
                    generation_cut=generation_cut,
                    resumed_generation_token_ids=resumed_generation_token_ids,
                )

                return res

        ########################################
        # /v1/chat/completions endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > ChatCompletionRequest
        class NeMoRLChatCompletionRequest(
            NeMoRLOpenAIChatRequestMixin, ChatCompletionRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None
            # Ledger-authoritative token capture: the call identity the ledger
            # attaches (rollout_id, call_id, parent_call_id, prev_len, mode).
            ng_capture: Optional[dict[str, Any]] = None

            def to_sampling_params(self, *args, **kwargs):
                sampling_params = super().to_sampling_params(*args, **kwargs)
                if (
                    worker_self._generation_prefix_cuts_enabled
                    and self.ng_capture is not None
                ):
                    # Gym's public request remains non-streaming, but prefix
                    # cuts need vLLM to publish in-flight outputs to the
                    # internal full-response generator. DELTA keeps that work
                    # linear; the serving adapter reconstructs one cumulative
                    # result before invoking vLLM's non-streaming response path.
                    sampling_params.output_kind = RequestOutputKind.DELTA
                return sampling_params

        # vLLM 0.25 routes both /v1/chat/completions and /tokenize through
        # OnlineRenderer.preprocess_chat, so the prefix-token override
        # belongs on the renderer subclass.
        worker_self = self

        @app.post("/ng-control/v1/generation-cut")
        async def checkpoint_generation_cut(
            inventory: dict[str, Any],
            authorization: str | None = Header(default=None),
        ):
            """Persist one checkpoint's frozen active-call prefixes to TQ."""
            import secrets

            from nemo_gym._checkpoint.model_control_contracts import (
                GenerationCutInventory,
            )

            expected = worker_self._generation_cut_control_token
            supplied = (
                authorization.removeprefix("Bearer ")
                if authorization is not None and authorization.startswith("Bearer ")
                else ""
            )
            if not worker_self._generation_prefix_cuts_enabled or expected is None:
                raise HTTPException(
                    status_code=404, detail="generation-prefix cuts are disabled"
                )
            if not secrets.compare_digest(supplied, expected):
                raise HTTPException(status_code=401, detail="invalid control bearer")
            typed_inventory = GenerationCutInventory.model_validate(inventory)
            receipt = await worker_self._run_generation_checkpoint_control(
                worker_self._checkpoint_generation_cut, typed_inventory
            )
            return receipt.model_dump(mode="json")

        class NeMoRLOpenAIServingChatMixin:
            async def chat_completion_full_generator(
                self,
                request,
                result_generator,
                *args,
                **kwargs,
            ):
                return_as_token_id = (
                    request.return_tokens_as_token_ids
                    if request.return_tokens_as_token_ids is not None
                    else self.return_tokens_as_token_ids
                )
                if (
                    request.logprobs
                    and return_as_token_id
                    and request.top_logprobs is None
                ):
                    raise VLLMValidationError(
                        "`top_logprobs` must be set when requesting token "
                        "information from the NeMo-RL chat endpoint.",
                        parameter="top_logprobs",
                    )

                aggregate_deltas = bool(
                    worker_self._generation_prefix_cuts_enabled
                    and request.ng_capture is not None
                )
                final_res = None
                delta_accumulator = _RequestOutputDeltaAccumulator()

                async def capture_result_generator():
                    nonlocal final_res
                    async for res in result_generator:
                        worker_self._observe_request_capture(request, res)
                        if not aggregate_deltas:
                            final_res = res
                            worker_self._restore_response_prefix(
                                request,
                                res,
                                tokenizer=self.renderer.tokenizer,
                            )
                            yield res
                            continue

                        delta_accumulator.append(res)

                    if not aggregate_deltas or delta_accumulator.template is None:
                        return

                    final_res = delta_accumulator.build()
                    worker_self._restore_response_prefix(
                        request,
                        final_res,
                        tokenizer=self.renderer.tokenizer,
                    )
                    yield final_res

                response = await super().chat_completion_full_generator(
                    request,
                    capture_result_generator(),
                    *args,
                    **kwargs,
                )
                if (
                    not isinstance(response, ChatCompletionResponse)
                    or final_res is None
                ):
                    return response

                if request.logprobs and return_as_token_id:
                    response = attach_token_information_to_chat_response_choices(
                        response,
                        final_res,
                    )

                if worker_self._return_routed_experts_enabled():
                    response = attach_routed_experts_to_chat_response_choices(
                        response,
                        final_res,
                        device=torch.device("cpu"),
                        logger=LOGGER,
                        routed_experts_dtype=worker_self.routed_experts_dtype,
                    )

                return response

        class NeMoRLOpenAIServingChat(NeMoRLOpenAIServingChatMixin, OpenAIServingChat):
            pass

        class NeMoRLOnlineRenderer(NeMoRLOpenAIServingMixin, OnlineRenderer):
            pass

        serving_chat_default_kwargs = dict(
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
        )
        serving_chat_kwargs = serving_chat_default_kwargs | self.cfg["vllm_cfg"].get(
            "http_server_serving_chat_kwargs", dict()
        )
        # The embedded server is constructed directly instead of through
        # vLLM's CLI, where chat-template file paths are normally loaded.
        # OnlineRenderer expects literal Jinja content; passing a path makes
        # Transformers render the path itself and drops multimodal
        # placeholders such as <image>.
        configured_chat_template = serving_chat_kwargs.get("chat_template")
        if configured_chat_template is not None:
            serving_chat_kwargs["chat_template"] = load_chat_template(
                configured_chat_template
            )
        # Recipes may name the parameter either way: ``default_chat_template_kwargs``
        # is vLLM's own spelling, ``chat_template_kwargs`` is accepted for recipes
        # written against the older name. Normalize onto the native key rather
        # than popping it: OnlineRenderer, OpenAIServingChat and ServingTokenization
        # each keep their *own* copy and read it independently -- the chat serving
        # builds its reasoning parser from it, and the tokenize path passes its own
        # into preprocess_chat -- so the renderer's copy does not reach either.
        # vLLM's api_server hands the same value to all three for that reason.
        #
        # Popped separately, not `A or B`: short-circuiting on a truthy A would
        # leave B in the bag and OpenAIServingChat(**kwargs) would reject it.
        _legacy_chat_template_kwargs = serving_chat_kwargs.pop(
            "chat_template_kwargs", None
        )
        if serving_chat_kwargs.get("default_chat_template_kwargs") is None:
            serving_chat_kwargs["default_chat_template_kwargs"] = (
                _legacy_chat_template_kwargs
            )
        default_chat_template_kwargs: dict[str, Any] = (
            serving_chat_kwargs["default_chat_template_kwargs"] or {}
        )
        online_renderer = NeMoRLOnlineRenderer(
            model_config=engine_client.model_config,
            renderer=engine_client.renderer,
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            enable_auto_tools=serving_chat_kwargs["enable_auto_tools"],
            # Keep the renderer's parser consistent with any parser overrides
            # passed to OpenAIServingChat via http_server_serving_chat_kwargs.
            tool_parser=serving_chat_kwargs.get("tool_parser"),
            reasoning_parser=serving_chat_kwargs.get("reasoning_parser"),
            # vLLM merges these into every render, with request-supplied keys
            # winning (preprocess_chat's default_template_kwargs). The renderer
            # is shared by /v1/chat/completions and /tokenize, so setting it
            # here keeps the two endpoints rendering identical prompts.
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        serving_chat_kwargs.update(
            dict(
                engine_client=engine_client,
                models=openai_serving_models,
                online_renderer=online_renderer,
                return_tokens_as_token_ids=True,
            )
        )
        openai_serving_chat = NeMoRLOpenAIServingChat(**serving_chat_kwargs)

        generation_config = self.cfg

        # The create_chat_completion and tokenize methods are taken from vllm/entrypoints/openai/api_server.py
        @app.post("/v1/chat/completions")
        async def create_chat_completion(
            request: NeMoRLChatCompletionRequest, raw_request: Request
        ):
            # This needs to match the behavior in nemo_rl/models/generation/vllm/vllm_worker.py::BaseVllmGenerationWorker::_build_sampling_params
            # Right now we explicitly assert set this to -1.
            assert request.top_k in (None, -1), (
                f"Top k sampling parameter must be unset, empty, or -1. Got `{request.top_k}`"
            )
            request.top_k = -1

            # The request sampling params need to exactly match those as are set in NeMo RL.
            # If they do not match, the inference will be off policy and destroy training
            # stability. Validation rollouts are the one exception: they are stamped with
            # the validation sampling profile (generation.val_temperature / val_top_p),
            # which is metric-only and safe to serve — grpo.validate() is the only
            # caller that constructs a non-train GenerationSamplingParams. Multi-turn
            # agents issue their own requests, so this server-side check is the one
            # chokepoint they all pass.
            # vLLM resolves an unset top_p from the model's generation_config.json
            # (ModelConfig.generation_config defaults to "auto"), NOT to 1.0, so a
            # request omitting it would sample off-policy while passing this check.
            assert request.top_p is not None, (
                "top_p must be set explicitly on NeMo-RL requests; an unset top_p is "
                "resolved by vLLM from the model's generation_config.json and would "
                "bypass the on-policy sampling check."
            )
            request_top_p = request.top_p
            is_train_sampling = (
                request.temperature == generation_config["temperature"]
                and request_top_p == generation_config["top_p"]
            )
            is_val_sampling = (
                request.temperature == generation_config["val_temperature"]
                and request_top_p == generation_config["val_top_p"]
            )
            assert is_train_sampling or is_val_sampling, (
                f"request sampling (temperature={request.temperature}, "
                f"top_p={request.top_p}) matches neither the train sampling params "
                f"(temperature={generation_config['temperature']}, "
                f"top_p={generation_config['top_p']}) nor the validation sampling "
                f"params (val_temperature={generation_config['val_temperature']}, "
                f"val_top_p={generation_config['val_top_p']})"
            )

            try:
                generator = await openai_serving_chat.create_chat_completion(
                    request, raw_request
                )
            except VLLMValidationError as e:
                # vLLM raises VLLMValidationError for prompts exceeding
                # max_model_len during tokenization, instead of returning an
                # ErrorResponse. Convert to HTTP 400 so the Gym proxy can
                # detect context-length overflow and handle it gracefully.
                worker_self._abort_request_capture(request, reason="context_length")
                return JSONResponse(
                    content={
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "param": e.parameter,
                            "code": 400,
                        }
                    },
                    status_code=400,
                )
            except BaseException:
                worker_self._abort_request_capture(request, reason="engine_error")
                raise

            if isinstance(generator, ErrorResponse):
                worker_self._abort_request_capture(request, reason="error_response")
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )

            elif isinstance(generator, ChatCompletionResponse):
                content = model_dump_chat_response_with_dynamic_message_fields(
                    generator
                )
                # Token capture: stage the delta and ride the coords on the
                # response; strips logprobs/ids (no-op when capture is off).
                # Off-loop: the sink write inside complete_call is a blocking
                # TQ round trip (see the staging protocol's serving-host rule).
                content = await asyncio.to_thread(
                    worker_self._finish_request_capture, request, content
                )
                return JSONResponse(content=content)

            worker_self._abort_request_capture(request, reason="streaming_response")
            return StreamingResponse(content=generator, media_type="text/event-stream")

        ########################################
        # /tokenize endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
        class NeMoRLTokenizeChatRequest(
            NeMoRLOpenAIChatRequestMixin, TokenizeChatRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None

        NeMoRLTokenizeRequest = Union[
            TokenizeCompletionRequest, NeMoRLTokenizeChatRequest
        ]

        # Tokenize path delegates to OnlineRenderer.preprocess_chat,
        # where the prefix-token override lives.
        class NeMoRLServingTokenization(ServingTokenization):
            pass

        serving_tokenization_kwargs = dict(
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            models=serving_chat_kwargs["models"],
            online_renderer=online_renderer,
            # ServingTokenization reads its own copy in preprocess_chat rather
            # than the renderer's, so /tokenize would otherwise render with {}
            # and diverge from /v1/chat/completions under multi-turn.
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        openai_serving_tokenization = NeMoRLServingTokenization(
            **serving_tokenization_kwargs
        )

        @app.post("/tokenize")
        async def tokenize(request: NeMoRLTokenizeRequest, raw_request: Request):
            generator = await openai_serving_tokenization.create_tokenize(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )
            elif isinstance(generator, TokenizeResponse):
                return JSONResponse(content=generator.model_dump())

        ########################################
        # Logging
        ########################################
        print(
            "Adding a vLLM logging filter so that the logs aren't spammed with not useful messages like `Added request ...`. This is to help errors pop up better and filter out noise."
        )

        class CleanLoggingFilter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                msg = record.getMessage()

                # vLLM does not accept `strict` tool definitions and reporting it to the user is not useful either.
                return (
                    "Added request" not in msg
                    and "The following fields were present in the request but ignored: {'strict'}"
                    not in msg
                )

        vllm_async_llm_logger.addFilter(CleanLoggingFilter())

        from logging import getLogger as _getLogger

        _getLogger("vllm.entrypoints.openai.engine.protocol").addFilter(
            CleanLoggingFilter()
        )

        # Suppress the noisy vLLM traceback when a prompt exceeds max_model_len.
        # This is expected during multi-turn rollouts; we log a clean one-line
        # warning from _preprocess_chat instead.
        class MaxContextLengthFilter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                if record.exc_info and record.exc_info[1]:
                    if "maximum context length" in str(record.exc_info[1]):
                        return False
                return True

        _getLogger("vllm.entrypoints.openai.chat_completion.serving").addFilter(
            MaxContextLengthFilter()
        )

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

        app = self._setup_vllm_openai_api_server(app)
        if self._sparse_refit_receiver is not None:
            self._sparse_refit_receiver.setup_api_server(app)

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

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a batch of data using vLLM's AsyncLLMEngine, yielding results as they are ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]

        # Ensure generate_async only receives single samples (batch_size = 1)
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        batch_specific_stop_strings_list = data.get(
            "stop_strings", [[] for _ in range(batch_size)]
        )

        # Create tasks for each sample in the batch
        async def process_single_sample(sample_idx):
            """Process a single sample and return the result."""
            current_input_actual_length = input_lengths_batch[sample_idx].item()
            prompt = format_prompt_for_vllm_generation(data, sample_idx)

            per_sample_stop_strings = None
            if batch_specific_stop_strings_list and sample_idx < len(
                batch_specific_stop_strings_list
            ):
                per_sample_stop_strings = batch_specific_stop_strings_list[sample_idx]

            final_stop_strings_for_sample = self._merge_stop_strings(
                [per_sample_stop_strings] if per_sample_stop_strings else None
            )

            max_model_len = int(self.cfg["vllm_cfg"]["max_model_len"])
            remaining_ctx = max_model_len - current_input_actual_length
            allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

            spec_cfg = self.cfg.get("vllm_kwargs", {}).get("speculative_config") or {}
            spec_lookahead = int(spec_cfg.get("num_speculative_tokens", 0))
            if allowed_new_tokens > 0 and spec_lookahead > 0:
                allowed_new_tokens = self._request_max_new_tokens(
                    configured_max_new_tokens=allowed_new_tokens,
                    input_length=current_input_actual_length,
                    max_model_len=max_model_len,
                    cap_to_context=False,
                    spec_lookahead=spec_lookahead,
                )

            # Handle case where no tokens can be generated due to length constraints
            if allowed_new_tokens == 0:
                # Access the input data directly from the function parameters
                input_ids_single_row = input_ids_batch[sample_idx]

                # Create output tensors with just the input (no generated tokens)
                output_ids_single_item_batched = input_ids_single_row[
                    :current_input_actual_length
                ].unsqueeze(0)

                logprobs_single_item = torch.zeros(
                    (1, current_input_actual_length),
                    dtype=torch.float32,
                    device=input_ids_single_row.device,
                )

                generation_lengths_tensor = torch.tensor(
                    [0], dtype=torch.long, device=input_ids_single_row.device
                )

                unpadded_sequence_lengths_tensor = torch.tensor(
                    [current_input_actual_length],
                    dtype=torch.long,
                    device=input_ids_single_row.device,
                )

                # Not truncated since no generation was attempted (length constraint)
                truncated_tensor = torch.tensor(
                    [False], dtype=torch.bool, device=input_ids_single_row.device
                )

                result_batch = BatchedDataDict[GenerationOutputSpec](
                    {
                        "output_ids": output_ids_single_item_batched,
                        "logprobs": logprobs_single_item,
                        "generation_lengths": generation_lengths_tensor,
                        "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                        "truncated": truncated_tensor,
                    }
                )

                return (sample_idx, result_batch)

            sampling_params_for_request = self._build_sampling_params(
                greedy=greedy,
                stop_strings=final_stop_strings_for_sample,
                max_new_tokens=allowed_new_tokens,
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params_for_request,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Process the output
            generation_details = final_request_output.outputs[0]
            generated_token_ids = list(generation_details.token_ids)
            num_generated_tokens = len(generated_token_ids)
            return_routed_experts = self._return_routed_experts_enabled()

            original_input_ids_single_row = input_ids_batch[sample_idx]
            final_output_tensor_len = current_input_actual_length + num_generated_tokens

            # Create output_ids tensor for this single item
            output_ids_single_item = torch.full(
                (final_output_tensor_len,),
                self.cfg["_pad_token_id"],
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )
            # Copy original input (up to its actual length)
            output_ids_single_item[:current_input_actual_length] = (
                original_input_ids_single_row[:current_input_actual_length]
            )
            # Add generated tokens after the actual input
            output_ids_single_item[
                current_input_actual_length : current_input_actual_length
                + num_generated_tokens
            ] = torch.tensor(
                generated_token_ids,
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )

            # Reshape to (1, seq_len) for BatchedDataDict
            output_ids_single_item_batched = output_ids_single_item.unsqueeze(0)

            # Create logprobs tensor for this single item
            logprobs_single_item = torch.zeros(
                (1, final_output_tensor_len),
                dtype=torch.float32,
                device=original_input_ids_single_row.device,
            )
            if hasattr(generation_details, "logprobs") and generation_details.logprobs:
                for idx, logprob_dict_per_token in enumerate(
                    generation_details.logprobs
                ):
                    if logprob_dict_per_token and idx < len(generated_token_ids):
                        token_id_at_idx = generated_token_ids[idx]
                        if token_id_at_idx in logprob_dict_per_token:
                            logprob_value = logprob_dict_per_token[
                                token_id_at_idx
                            ].logprob
                            position_in_output_tensor = (
                                current_input_actual_length + idx
                            )
                            if position_in_output_tensor < final_output_tensor_len:
                                logprobs_single_item[0, position_in_output_tensor] = (
                                    logprob_value
                                )

            # Generation lengths
            generation_lengths_tensor = torch.tensor(
                [num_generated_tokens],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Unpadded sequence lengths (actual_input + actual_generated)
            unpadded_total_length = current_input_actual_length + num_generated_tokens
            unpadded_sequence_lengths_tensor = torch.tensor(
                [unpadded_total_length],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Check if response was truncated (hit max_tokens length limit)
            is_truncated = generation_details.finish_reason == "length"
            truncated_tensor = torch.tensor(
                [is_truncated],
                dtype=torch.bool,
                device=original_input_ids_single_row.device,
            )

            result_dict = {
                "output_ids": output_ids_single_item_batched,
                "logprobs": logprobs_single_item,
                "generation_lengths": generation_lengths_tensor,
                "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                "truncated": truncated_tensor,
            }
            routed_experts, r3_stats = pad_and_align_routed_expert_indices(
                final_request_output,
                generation_details,
                valid_length=unpadded_total_length,
                padded_length=final_output_tensor_len,
                device=original_input_ids_single_row.device,
                require_complete_routed_experts=return_routed_experts,
                return_stats=True,
                routed_experts_dtype=self.routed_experts_dtype,
            )
            if return_routed_experts and routed_experts is None:
                raise RuntimeError(
                    "vLLM was asked to return routed experts but the generation output "
                    "did not include routed_experts."
                )
            if return_routed_experts:
                if r3_stats["missing_routes"] > 0:
                    LOGGER.warning(
                        "R3 router replay fallback: vLLM returned incomplete "
                        "routed_experts for sample_idx=%d, missing_token_routes=%d, "
                        "actual_routes=%d, expected_routes=%d. Megatron will use its "
                        "own router for those missing token routes.",
                        sample_idx,
                        r3_stats["missing_routes"],
                        r3_stats["actual_routes"],
                        r3_stats["expected_routes"],
                    )
                result_dict["r3_routed_experts_missing_routes"] = torch.tensor(
                    [r3_stats["missing_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
                result_dict["r3_routed_experts_expected_routes"] = torch.tensor(
                    [r3_stats["expected_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
                result_dict["r3_routed_experts_actual_routes"] = torch.tensor(
                    [r3_stats["actual_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
            if routed_experts is not None:
                result_dict["routed_experts"] = routed_experts.unsqueeze(0)

            result_batch = BatchedDataDict[GenerationOutputSpec](result_dict)

            return (sample_idx, result_batch)

        # Create tasks for all samples and yield results as they complete
        sample_tasks = [
            asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        try:
            for completed_task in asyncio.as_completed(sample_tasks):
                result = await completed_task
                yield result
        finally:
            for task in sample_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*sample_tasks, return_exceptions=True)

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
            prompt = prompts[prompt_idx]

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

    async def prepare_refit_info_async(self, state_dict_info: dict[str, Any]) -> None:
        """Async version of prepare_refit_info."""
        await self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

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

    async def begin_generation_checkpoint_async(self) -> bool:
        """Fence terminal staging without pausing vLLM decoding."""
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "begin_generation_checkpoint_async requires async_engine=True"
            )
        await self._run_generation_checkpoint_control(
            self._generation_checkpoint_gate.close_and_wait
        )
        return True

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

    async def finish_generation_checkpoint_async(self) -> bool:
        """Release terminal token staging after the TQ snapshot is durable."""
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "finish_generation_checkpoint_async requires async_engine=True"
            )
        self._generation_checkpoint_gate.reopen()
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
            generation_checkpoint_gate = getattr(
                self, "_generation_checkpoint_gate", None
            )
            if generation_checkpoint_gate is not None:
                generation_checkpoint_gate.reopen()
            generation_checkpoint_executor = getattr(
                self, "_generation_checkpoint_executor", None
            )
            if generation_checkpoint_executor is not None:
                generation_checkpoint_executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
            # Flush buffered spans/metrics before the actor goes away. Off the
            # event loop: the flush blocks on a network export with a 5s
            # timeout, and this is an async actor whose other coroutines --
            # including in-flight generate requests -- share this loop. Same
            # reason the sparse-refit shutdown above is offloaded.
            await asyncio.to_thread(shutdown_telemetry)


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    pass
