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
import functools
import gc
import logging
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, AsyncGenerator, Literal, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI

from nemo_rl.data.captured_media import (
    CapturedMedia,
    CapturedMediaItem,
    MediaCaptureRejected,
    capture_processed_media,
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
from nemo_rl.models.generation.generation_cut_capture import (
    GenerationCutCaptureMixin,
    _CompletedCaptureState,
    _remaining_generation_limits_after_prefix,
    _RequestCaptureState,
    _TokenCaptureSnapshotGate,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.prefix_read_batcher import PrefixReadBatcher
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
    validate_rollout_prompt,
    model_dump_chat_response_with_dynamic_message_fields,
    pad_and_align_routed_expert_indices,
)
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker
from nemo_rl.models.generation.openai_server_utils import (
    PrefixSplice,
    splice_prefix_tokens,
)
from nemo_rl.telemetry.setup import shutdown_telemetry

LOGGER = logging.getLogger(__name__)
_RESTORED_PREFIX_TERMINAL_PROMPT_KEY = "__nemo_rl_restored_prefix_terminal__"


@dataclass(frozen=True)
class _RestoredPrefixTerminal:
    """A restored call whose durable output already reached a terminal limit."""

    prompt_token_ids: tuple[int, ...]
    prefix_token_count: int
    reason: str
    finish_reason: Literal["stop", "length"]
    stop_reason: str | int | None = None

    @property
    def original_prompt_token_count(self) -> int:
        return len(self.prompt_token_ids) - self.prefix_token_count


def _classify_restored_prefix_terminal(
    *,
    prompt_token_ids: list[int],
    prefix_token_count: int,
    requested_output_tokens: int | None,
    model_max_tokens: int,
    terminal_finish_reason: Literal["stop", "length"] | None = None,
    terminal_stop_reason: str | int | None = None,
) -> _RestoredPrefixTerminal | None:
    """Classify an exact terminal prefix or reject an incompatible restore."""
    prompt_token_count = len(prompt_token_ids)
    if prefix_token_count < 0:
        raise ValueError("prefix_token_count must be non-negative")
    if prefix_token_count > prompt_token_count:
        raise ValueError(
            "Durable generation prefix contains more generated tokens than the "
            "restored engine prompt."
        )
    if (
        requested_output_tokens is not None
        and prefix_token_count > requested_output_tokens
    ):
        raise ValueError(
            "Durable generation prefix token count "
            f"({prefix_token_count}) exceeds the restored request output "
            f"budget ({requested_output_tokens})."
        )
    if prompt_token_count > model_max_tokens:
        raise ValueError(
            "Durable generation prefix prompt length "
            f"({prompt_token_count}) exceeds restored model capacity "
            f"({model_max_tokens})."
        )
    if terminal_stop_reason is not None and terminal_finish_reason is None:
        raise ValueError("terminal_stop_reason requires terminal_finish_reason")
    if terminal_finish_reason is not None:
        return _RestoredPrefixTerminal(
            prompt_token_ids=tuple(prompt_token_ids),
            prefix_token_count=prefix_token_count,
            reason=f"observed_{terminal_finish_reason}",
            finish_reason=terminal_finish_reason,
            stop_reason=terminal_stop_reason,
        )
    reached_output_limit = (
        requested_output_tokens is not None
        and prefix_token_count == requested_output_tokens
    )
    reached_model_capacity = prompt_token_count == model_max_tokens
    if not reached_output_limit and not reached_model_capacity:
        return None
    return _RestoredPrefixTerminal(
        prompt_token_ids=tuple(prompt_token_ids),
        prefix_token_count=prefix_token_count,
        reason=(
            "output_and_model_limit"
            if reached_output_limit and reached_model_capacity
            else "output_limit"
            if reached_output_limit
            else "model_limit"
        ),
        finish_reason="length",
    )


def _build_restored_prefix_terminal_output(
    request_id: str, terminal: _RestoredPrefixTerminal
) -> Any:
    """Build the final vLLM output for a prefix that needs no more decoding."""
    from vllm.outputs import CompletionOutput, RequestOutput

    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=list(terminal.prompt_token_ids),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=[],
                finish_reason=terminal.finish_reason,
                stop_reason=terminal.stop_reason,
            )
        ],
        finished=True,
    )


@dataclass(frozen=True)
class _RestoredPrefixParser:
    """Give vLLM's parser the complete output while capture keeps tail deltas."""

    delegate: Any
    prefix_token_ids: tuple[int, ...]

    def parse(
        self,
        model_output: str,
        request: Any,
        *,
        enable_auto_tools: bool,
        model_output_token_ids: list[int],
    ) -> Any:
        return self.delegate.parse(
            model_output,
            request,
            enable_auto_tools=enable_auto_tools,
            model_output_token_ids=[
                *self.prefix_token_ids,
                *model_output_token_ids,
            ],
        )

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return self.delegate.count_reasoning_tokens(
            [*self.prefix_token_ids, *token_ids]
        )


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
        self._capture_active_routes = False

    def enable_active_route_capture(self) -> None:
        """Attach incremental route chunks to active prefix-capture outputs."""
        self._capture_active_routes = True

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
        terminal = (
            prompt.get(_RESTORED_PREFIX_TERMINAL_PROMPT_KEY)
            if isinstance(prompt, dict)
            else None
        )
        if isinstance(terminal, _RestoredPrefixTerminal):
            yield _build_restored_prefix_terminal_output(request_id, terminal)
            return

        iterator = None
        completed = False
        route_chunk_cursor = 0

        async def next_output() -> Any:
            nonlocal iterator, route_chunk_cursor
            if iterator is None:
                iterator = self._engine_client.generate(
                    prompt, sampling_params, request_id, **kwargs
                )
            output = await anext(iterator)
            completions = getattr(output, "outputs", ())
            active_decode = bool(
                completions and getattr(completions[0], "finish_reason", None) is None
            )
            if self._capture_active_routes and active_decode:
                try:
                    (
                        route_chunk_cursor,
                        route_chunks,
                    ) = self._snapshot_new_routed_expert_chunks_on_owner(
                        output.request_id,
                        after_chunk=route_chunk_cursor,
                    )
                    output.nemo_rl_routed_expert_chunk_count = route_chunk_cursor
                    output.nemo_rl_routed_expert_chunks = route_chunks
                except Exception as error:  # noqa: BLE001
                    # Token serving remains available. The HTTP capture layer
                    # marks this call ineligible for a prefix cut.
                    output.nemo_rl_routed_expert_error = error
            return output

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

    async def snapshot_new_routed_expert_chunks(
        self, request_id: str, *, after_chunk: int
    ) -> tuple[int, list[Any]]:
        """Copy newly accumulated vLLM route chunks on the engine owner loop.

        vLLM 0.29 keeps routed experts in ``OutputProcessor.RequestState``
        while decoding and only publishes them in ``CompletionOutput`` once a
        request finishes. Prefix checkpointing needs the active state, but it
        must not read that mutable list from the HTTP server thread.
        """

        async def snapshot() -> tuple[int, list[Any]]:
            return self._snapshot_new_routed_expert_chunks_on_owner(
                request_id, after_chunk=after_chunk
            )

        return await self._run_on_engine_loop(snapshot)

    def _snapshot_new_routed_expert_chunks_on_owner(
        self, request_id: str, *, after_chunk: int
    ) -> tuple[int, list[Any]]:
        """Copy one request's route delta while already on the engine loop."""
        output_processor = self._engine_client.output_processor
        internal_ids = list(output_processor.external_req_ids.get(request_id, ()))
        if len(internal_ids) != 1:
            raise RuntimeError(
                "router-replay prefix capture requires exactly one active "
                f"vLLM request for {request_id!r}; found {len(internal_ids)}"
            )
        state = output_processor.request_states.get(internal_ids[0])
        if state is None:
            raise RuntimeError(
                f"active vLLM request {request_id!r} has no output state"
            )
        chunks = state.routed_experts_chunks
        if after_chunk < 0 or after_chunk > len(chunks):
            raise RuntimeError(
                "routed-expert chunk cursor is outside vLLM state: "
                f"cursor={after_chunk} chunks={len(chunks)}"
            )
        return len(chunks), [copy.deepcopy(chunk) for chunk in chunks[after_chunk:]]


class VllmAsyncGenerationWorkerImpl(
    GenerationCutCaptureMixin,
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

        # Ledger-authoritative token capture (dormant until the
        # setup_token_capture fan-out runs). The weight
        # version is stamped per model call at begin_call time and rotated by
        # the set_rollout_weight_version fan-out from the SC's _sync_weights.
        self.token_capture = None
        self._rollout_weight_version = 0
        # In-flight calls are indexed by both the request object and Gym's
        # stable model-call ID. The second index is what a checkpoint inventory
        # uses while the original HTTP request remains active.
        self._capture_calls: dict[int, _RequestCaptureState] = {}
        self._capture_calls_by_model_call_id: dict[str, _RequestCaptureState] = {}
        self._completed_capture_calls: dict[str, _CompletedCaptureState] = {}
        self._generation_cut_receipts: dict[tuple[str, str], Any] = {}
        self._capture_registry_lock = threading.Lock()
        self._capture_sink: Any | None = None
        self._generation_prefix_cuts_enabled = False
        self._generation_cut_control_token: str | None = None
        self._generation_cut_control_timeout_s: float | None = None
        self._generation_cut_tokenizer: Any | None = None
        self._token_capture_snapshot_gate = _TokenCaptureSnapshotGate()
        # A dedicated executor prevents completions waiting on the closed gate
        # from consuming all threads needed to create the checkpoint cut.
        self._generation_cut_control_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nrl-generation-cut-control",
        )
        # Fence closes get their own thread so a stale cut still staging rows
        # cannot delay the next checkpoint's fence past the driver's timeout.
        self._token_capture_fence_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nrl-token-capture-fence",
        )
        self._capture_media = False
        self._capture_patch_size: int | None = None
        self._capture_image_token_id: int | None = None
        # TQTokenSource installed by setup_token_capture. The media path reads
        # parent-chain rows through it directly; prefix resolution goes through
        # the shared ChainPrefixCache below.
        self._staging_source: Any | None = None
        self._generation_prefix_reader: PrefixReadBatcher | None = None
        # Resolved staging-chain prefixes, shared implementation with the Megatron
        # preparer. Installed by setup_token_capture; fetch runs on executor threads.
        # Deferred import: tq_token_sink pulls in the data-plane stack.
        from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache

        self._chain_prefix = ChainPrefixCache()

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
        capture_media: bool = False,
        generation_prefix_cuts_enabled: bool = False,
        generation_cut_control_token: str | None = None,
        generation_cut_control_timeout_s: float | None = None,
        generation_prefix_batch_size: int = 256,
        generation_prefix_batch_max_tokens: int = 4_194_304,
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
        if generation_prefix_cuts_enabled and not generation_cut_control_token:
            raise ValueError(
                "generation-prefix cuts require a non-empty control bearer token"
            )
        GenerationCutCaptureMixin._configure_generation_prefix_batching(
            self,
            max_rows=generation_prefix_batch_size,
            max_tokens=generation_prefix_batch_max_tokens,
        )
        if generation_prefix_cuts_enabled and capture_media:
            raise ValueError(
                "generation-prefix recovery does not yet support multimodal capture"
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
        self._generation_prefix_cuts_enabled = generation_prefix_cuts_enabled
        self._generation_cut_control_token = generation_cut_control_token
        self._generation_cut_control_timeout_s = generation_cut_control_timeout_s
        if generation_prefix_cuts_enabled:
            # Cuts must land on character boundaries; see _ends_inside_character.
            self._generation_cut_tokenizer = self.llm.renderer.get_tokenizer()
        source = TQTokenSource(
            dp_client, staging_partition=staging_partition, capture_media=capture_media
        )
        self._staging_source = source
        if self._generation_prefix_reader is not None:
            await self._generation_prefix_reader.aclose()
        prefix_fetch = source.fetch
        if bool(getattr(self, "_return_routed_experts_enabled", lambda: False)()):
            prefix_fetch = partial(
                source.fetch_for_finalization, include_route_fragments=True
            )
        self._generation_prefix_reader = PrefixReadBatcher(
            prefix_fetch,
            max_rows=generation_prefix_batch_size,
            max_tokens=generation_prefix_batch_max_tokens,
        )
        if generation_prefix_cuts_enabled and bool(
            getattr(self, "_return_routed_experts_enabled", lambda: False)()
        ):
            if self._http_engine_client is None:
                raise RuntimeError(
                    "router-replay prefix capture requires the vLLM HTTP client"
                )
            self._http_engine_client.enable_active_route_capture()
        self._chain_prefix.install(source)
        install_capture(
            self,
            sink=sink,
            weight_version_fn=lambda: self._rollout_weight_version,
            adapter=VLLMCaptureAdapter(),
        )
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
        media: CapturedMedia | None = None,
        generation_cut: Any | None = None,
        generation_cut_routed_experts: torch.Tensor | None = None,
        resumed_generation_token_ids: list[int] | None = None,
    ) -> None:
        """Admit one ledger-forwarded call into the capture layer.

        Called from preprocess_chat once the exact engine prompt is known
        (post-splice in token-in mode, full render in text mode). No-op
        unless capture is installed and the request carries the ledger's
        ``ng_capture`` context.

        ``prefix_token_ids`` is the prefix resolved by
        :meth:`_resolve_admission_prefix`. Gym's ``begin_call`` requires it for
        a ``staging_chain`` admission and checks its length against
        ``prev_len``; violations raise ``CaptureError`` before any state is kept.
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
            capture_routed_experts=bool(
                getattr(self, "_return_routed_experts_enabled", lambda: False)()
            ),
            routed_experts_dtype=getattr(self, "routed_experts_dtype", torch.int16),
            restored_routed_experts=generation_cut_routed_experts,
            media=media,
            resumed_generation_token_ids=list(resumed_generation_token_ids or ()),
            effective_output_limit=(
                admission.generation_cut.effective_output_limit
                if admission.generation_cut is not None
                else None
            ),
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

    def _observe_request_capture(
        self,
        request: Any,
        request_output: Any,
        *,
        routed_expert_chunk_count: int | None = None,
        routed_expert_chunks: list[Any] | None = None,
    ) -> None:
        """Append one request's immutable vLLM output delta."""
        state = self._get_request_capture(request)
        if state is None:
            return
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return
        try:
            output = outputs[0]
            generation_token_ids = list(getattr(output, "token_ids", ()) or ())
            generation_logprobs = extract_selected_token_logprobs(output)
            finish_reason = getattr(output, "finish_reason", None)
            stop_reason = getattr(output, "stop_reason", None)
            complete_routed_experts = None
            if state.capture_routed_experts and finish_reason is not None:
                route_parts = [
                    value
                    for value in (
                        getattr(request_output, "prompt_routed_experts", None),
                        getattr(output, "routed_experts", None),
                    )
                    if value is not None
                ]
                if route_parts:
                    complete_routed_experts = torch.cat(
                        [torch.as_tensor(value) for value in route_parts], dim=0
                    )
        except (RuntimeError, TypeError, ValueError) as error:
            with state.lock:
                state.observation_error = f"{type(error).__name__}: {error}"
            return
        with state.lock:
            state.observe(
                generation_token_ids,
                generation_logprobs,
                routed_expert_chunks=routed_expert_chunks,
                routed_expert_chunk_count=routed_expert_chunk_count,
                complete_routed_experts=complete_routed_experts,
            )
            if finish_reason is not None:
                if finish_reason not in ("stop", "length"):
                    state.observation_error = (
                        f"unsupported terminal finish_reason {finish_reason!r}"
                    )
                elif (
                    state.terminal_finish_reason is not None
                    and state.terminal_finish_reason != finish_reason
                ):
                    state.observation_error = (
                        "conflicting terminal finish reasons: "
                        f"{state.terminal_finish_reason!r} and {finish_reason!r}"
                    )
                elif not isinstance(stop_reason, (str, int, type(None))):
                    state.observation_error = (
                        "terminal stop_reason must be a string, integer, or None"
                    )
                else:
                    state.terminal_finish_reason = finish_reason
                    state.terminal_stop_reason = stop_reason

    def _mark_request_capture_observation_error(
        self, request: Any, error: BaseException
    ) -> None:
        """Poison prefix cutting for one call without failing model serving."""
        state = self._get_request_capture(request)
        if state is None:
            return
        with state.lock:
            state.observation_error = f"{type(error).__name__}: {error}"

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
        # vLLM already detokenized the tail with the request's settings and
        # stripped EOS and any matched stop string from it. Re-decoding the
        # whole sequence would restore both, so decode only the prefix, with
        # the same settings, and prepend it to the processed tail.
        with state.lock:
            prefix_text = tokenizer.decode(
                state.resumed_generation_token_ids,
                skip_special_tokens=getattr(request, "skip_special_tokens", True),
                spaces_between_special_tokens=getattr(
                    request, "spaces_between_special_tokens", True
                ),
            )
        output.text = prefix_text + (getattr(output, "text", None) or "")

    def _record_request_effective_output_limit(
        self, request: Any, effective_output_limit: int
    ) -> None:
        """Remember vLLM's resolved total output budget for future cuts."""
        if effective_output_limit <= 0:
            raise ValueError("effective output limit must be positive")
        state = self._get_request_capture(request)
        if state is None:
            return
        with state.lock:
            if state.effective_output_limit is None:
                state.effective_output_limit = effective_output_limit

    def _capture_request_media(
        self,
        engine_prompt: dict[str, Any],
        *,
        admission: Any | None,
        splice: PrefixSplice | None = None,
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
                        list(admission.staging_chain), include_route_fragments=False
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
                            [staging_key(admission.rollout_id, parent)],
                            include_route_fragments=False,
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
                    snapshot.rollout_id != admission.rollout_id
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
            retained_items = []
            for call in calls:
                extras = call.extras or {}
                if MEDIA_SPANS_FIELD not in extras:
                    raise MediaCaptureRejected("Retained vLLM media spans are missing")
                for value in extras[MEDIA_SPANS_FIELD]:
                    item = CapturedMediaItem.from_dict(value)
                    item.verify_tokens(
                        call.snapshot.token_ids_delta, origin=call.snapshot.prev_len
                    )
                    retained_items.append(item)
            retained = tuple(retained_items)
        return capture_processed_media(
            engine_prompt,
            prev_len=admission.prev_len,
            retained=retained,
            splice=splice,
            image_token_id=self._capture_image_token_id,
            patch_size=self._capture_patch_size,
        )

    def _fetch_chain_prefix(self, staging_chain: list[str]) -> list[int]:
        """Resolve a staging chain through the shared, cached TQ read."""
        return self._chain_prefix.fetch(staging_chain)

    def _resolve_admission_prefix(self, admission: Any) -> list[int]:
        """Resolve a ``CaptureAdmission`` to the flat prefix the engine prompt starts with."""
        # Deferred import, matching setup_token_capture.
        from nemo_rl.data_plane.tq_token_sink import resolve_admission_prefix

        return resolve_admission_prefix(admission, self._chain_prefix)

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

    def _merge_generation_cut_routed_experts(
        self,
        payload: dict[str, Any],
        *,
        restored: torch.Tensor | None,
        generated_len: int,
    ) -> None:
        """Keep old prefix routes and current-policy boundary/suffix routes."""
        if restored is None:
            return
        choices = payload.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("router replay requires exactly one response choice")
        choice = dict(choices[0])
        message = dict(choice.get("message") or {})
        routed = message.get("routed_experts")
        if routed is None:
            raise ValueError("resumed generation did not return routed experts")

        from nemo_rl.utils.routed_experts_codec import (
            decode_routed_experts,
            encode_routed_experts,
        )

        dtype_name = routed.split(":", 3)[1] if isinstance(routed, str) else "int16"
        dtype = {
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"unsupported routed_experts dtype {dtype_name!r}")
        current = decode_routed_experts(routed, dtype)
        restored = restored.to(dtype=dtype)
        if current.dim() != 3 or tuple(current.shape[1:]) != tuple(restored.shape[1:]):
            raise ValueError("restored routed experts changed model dimensions")
        if current.shape[0] != restored.shape[0] + generated_len:
            raise ValueError(
                "resumed routed experts do not match prefix plus generated tail"
            )
        replace_len = restored.shape[0] if generated_len == 0 else restored.shape[0] - 1
        current[:replace_len] = restored[:replace_len]
        message["routed_experts"] = encode_routed_experts(current)
        choice["message"] = message
        payload["choices"] = [choice]

    def _finish_request_capture(self, request: Any, content: dict) -> dict:
        """Run terminal token staging outside an active checkpoint cut."""
        self._token_capture_snapshot_gate.enter()
        try:
            return self._finish_request_capture_after_snapshot_fence(request, content)
        finally:
            self._token_capture_snapshot_gate.exit()

    def _finish_request_capture_after_snapshot_fence(
        self, request: Any, content: dict
    ) -> dict:
        """Stage one canonical terminal row and retire its prefix chunks."""
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
        sink = self._capture_sink
        if obsolete_staging_keys and sink is not None:
            try:
                sink.clear(list(obsolete_staging_keys))
            except Exception:  # noqa: BLE001 - completion is already durable
                LOGGER.exception(
                    "failed to clear obsolete generation chunks for model call %s",
                    state.call.model_call_id,
                )
        return content

    def _finish_request_capture_with_lifecycle_owned(
        self, state: _RequestCaptureState, request: Any, content: dict
    ) -> tuple[dict, tuple[str, ...]]:
        """Stage a terminal call while owning ``state.lifecycle_lock``."""
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
            self._merge_generation_cut_routed_experts(
                payload,
                restored=state.restored_routed_experts,
                generated_len=len(generated_token_ids),
            )
        coords = self.token_capture.complete_call_from_response(
            call,
            payload,
            attachments=state.media.tensors if state.media is not None else None,
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
                call.rollout_id,
                call.model_call_id,
                call.generation_cut.model_call_id,
                prefix_tokens,
                len(generated_token_ids),
                total_generation_token_count,
            )
        with state.lock:
            effective_output_limit = state.effective_output_limit
            terminal_finish_reason = state.terminal_finish_reason
            terminal_stop_reason = state.terminal_stop_reason
        if terminal_finish_reason is None:
            choices = content.get("choices") or []
            finish_reason = choices[0].get("finish_reason") if choices else None
            if finish_reason in ("stop", "length"):
                terminal_finish_reason = finish_reason
                terminal_stop_reason = choices[0].get("stop_reason")
        self._remember_completed_capture(
            call.model_call_id,
            coords,
            total_generation_token_count,
            effective_output_limit=effective_output_limit,
            terminal_finish_reason=terminal_finish_reason,
            terminal_stop_reason=terminal_stop_reason,
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

    # ruff: noqa
    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        worker_self = self
        from copy import deepcopy
        from logging import Filter as LoggingFilter
        from logging import LogRecord
        from typing import List, Optional, Union

        from fastapi import Header, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import PrivateAttr
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )

        # vLLM 0.29 moved this out of the openai package (vllm-project/vllm#54492).
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
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
                # #4124: processor-only cache reads retain concrete pixels even
                # when the engine's sender cache would return references.
                if (
                    worker_self._capture_media
                    and worker_self._capture_admission(request) is not None
                ):
                    skip_mm_cache = True

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
                generation_cut_routed_experts = None
                resumed_generation_token_ids: list[int] = []
                restored_request_output_tokens: int | None = None
                if admission is not None:
                    capture_prefix_token_ids = await asyncio.to_thread(
                        worker_self._resolve_admission_prefix, admission
                    )
                    cut_snapshots = None
                    if admission.generation_cut is not None:
                        if worker_self._generation_prefix_reader is None:
                            raise RuntimeError(
                                "generation prefix reader not initialized"
                            )
                        cut_snapshots = (
                            await worker_self._generation_prefix_reader.fetch(
                                list(admission.generation_cut.staging_keys),
                                tokens_per_key=max(1, self.model_config.max_model_len),
                            )
                        )
                    resolved_generation_cut = await asyncio.to_thread(
                        worker_self._resolve_generation_cut,
                        admission,
                        capture_prefix_token_ids,
                        cut_snapshots,
                    )
                    if resolved_generation_cut is not None:
                        generation_cut = resolved_generation_cut.snapshot
                        generation_cut_routed_experts = (
                            resolved_generation_cut.routed_experts
                        )
                    engine_prefix_token_ids = list(capture_prefix_token_ids)
                    if generation_cut is not None:
                        restored_request_output_tokens = (
                            admission.generation_cut.effective_output_limit
                        )
                        engine_prefix_token_ids.extend(generation_cut.token_ids_delta)
                        resumed_generation_token_ids = [
                            token_id
                            for token_id, mask in zip(
                                generation_cut.token_ids_delta,
                                generation_cut.token_mask_delta,
                            )
                            if mask == 1.0
                        ]
                        request._restored_generation_token_ids = tuple(
                            resumed_generation_token_ids
                        )
                        (
                            remaining_output_tokens,
                            remaining_min_tokens,
                        ) = _remaining_generation_limits_after_prefix(
                            max_tokens=restored_request_output_tokens,
                            min_tokens=getattr(request, "min_tokens", None),
                            prefix_token_count=(
                                admission.generation_cut.prefix_token_count
                            ),
                        )
                        if (
                            remaining_output_tokens is not None
                            and remaining_output_tokens > 0
                        ):
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
                    media = await asyncio.to_thread(
                        worker_self._capture_request_media,
                        res[1][0],
                        admission=admission,
                    )
                    worker_self._begin_request_capture(
                        request,
                        res[1][0]["prompt_token_ids"],
                        admission=admission,
                        prefix_token_ids=capture_prefix_token_ids,
                        media=media,
                        generation_cut=generation_cut,
                        generation_cut_routed_experts=(generation_cut_routed_experts),
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

                splice = None
                if generation_cut is not None:
                    # The durable prefix is the exact engine prompt at the cut;
                    # rendering/splicing it again could add or drop tokens.
                    final_prompt_token_ids = model_prefix_token_ids
                    media = None
                else:
                    splice = splice_prefix_tokens(
                        tokenizer=self.renderer.tokenizer,
                        model_prefix_token_ids=model_prefix_token_ids,
                        template_prefix_token_ids=actual_corresponding_token_ids,
                        template_token_ids=engine_prompt["prompt_token_ids"],
                    )
                    final_prompt_token_ids = splice.token_ids
                    media = await asyncio.to_thread(
                        worker_self._capture_request_media,
                        engine_prompt,
                        admission=admission,
                        splice=splice,
                    )
                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

                restored_prefix_terminal = None
                if generation_cut is not None:
                    try:
                        restored_prefix_terminal = _classify_restored_prefix_terminal(
                            prompt_token_ids=final_prompt_token_ids,
                            prefix_token_count=(
                                admission.generation_cut.prefix_token_count
                            ),
                            requested_output_tokens=restored_request_output_tokens,
                            model_max_tokens=self.model_config.max_model_len,
                            terminal_finish_reason=(
                                admission.generation_cut.terminal_finish_reason
                            ),
                            terminal_stop_reason=(
                                admission.generation_cut.terminal_stop_reason
                            ),
                        )
                    except ValueError as error:
                        raise VLLMValidationError(
                            str(error),
                            parameter="generation_prefix",
                            value=admission.generation_cut.prefix_token_count,
                        ) from error

                # Clamp after prefix replacement since the prompt length may have changed.
                if (
                    restored_prefix_terminal is None
                    and actual_request_max_tokens is not None
                ):
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
                    media=media,
                    generation_cut=generation_cut,
                    generation_cut_routed_experts=generation_cut_routed_experts,
                    resumed_generation_token_ids=resumed_generation_token_ids,
                )

                if restored_prefix_terminal is not None:
                    request.min_tokens = 0
                    self._set_max_tokens(request, 1)
                    engine_prompt[_RESTORED_PREFIX_TERMINAL_PROMPT_KEY] = (
                        restored_prefix_terminal
                    )
                    if len(final_prompt_token_ids) == self.model_config.max_model_len:
                        engine_prompt["prompt_token_ids"] = final_prompt_token_ids[:-1]
                    request._restored_prefix_terminal = restored_prefix_terminal
                    LOGGER.info(
                        "generation prefix already terminal: "
                        "rollout_id=%s model_call_id=%s prefix_tokens=%d reason=%s",
                        admission.rollout_id,
                        admission.model_call_id,
                        restored_prefix_terminal.prefix_token_count,
                        restored_prefix_terminal.reason,
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
            _restored_prefix_terminal: _RestoredPrefixTerminal | None = PrivateAttr(
                default=None
            )
            _restored_generation_token_ids: tuple[int, ...] = PrivateAttr(default=())

            def to_sampling_params(self, *args, **kwargs):
                sampling_params = super().to_sampling_params(*args, **kwargs)
                if (
                    worker_self._generation_prefix_cuts_enabled
                    and self.ng_capture is not None
                ):
                    sampling_params.output_kind = RequestOutputKind.DELTA
                    worker_self._record_request_effective_output_limit(
                        self, int(sampling_params.max_tokens)
                    )
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

            from nemo_gym._checkpoint.generation_cut import GenerationCutInventory

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
            # Measured from arrival so time queued behind a stale cut counts.
            timeout_s = worker_self._generation_cut_control_timeout_s
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            receipt = await worker_self._run_generation_cut_control(
                functools.partial(
                    worker_self._checkpoint_generation_cut,
                    typed_inventory,
                    deadline=deadline,
                )
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
                        route_chunk_count = getattr(
                            res, "nemo_rl_routed_expert_chunk_count", None
                        )
                        route_chunks = getattr(
                            res, "nemo_rl_routed_expert_chunks", None
                        )
                        route_error = getattr(res, "nemo_rl_routed_expert_error", None)
                        if route_error is not None:
                            LOGGER.error(
                                "Failed to snapshot active vLLM routes for %s: %s: %s",
                                getattr(res, "request_id", "<unknown>"),
                                type(route_error).__name__,
                                route_error,
                            )
                            worker_self._mark_request_capture_observation_error(
                                request, route_error
                            )
                        worker_self._observe_request_capture(
                            request,
                            res,
                            routed_expert_chunk_count=route_chunk_count,
                            routed_expert_chunks=route_chunks,
                        )
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

                restored_parser_prefix = request._restored_generation_token_ids
                if restored_parser_prefix:
                    if len(args) >= 6 and args[5] is not None:
                        mutable_args = list(args)
                        mutable_args[5] = _RestoredPrefixParser(
                            delegate=args[5],
                            prefix_token_ids=restored_parser_prefix,
                        )
                        args = tuple(mutable_args)
                    elif kwargs.get("parser") is not None:
                        kwargs = dict(kwargs)
                        kwargs["parser"] = _RestoredPrefixParser(
                            delegate=kwargs["parser"],
                            prefix_token_ids=restored_parser_prefix,
                        )

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

                restored_prefix_terminal = request._restored_prefix_terminal
                if (
                    isinstance(restored_prefix_terminal, _RestoredPrefixTerminal)
                    and response.usage is not None
                ):
                    response.usage.prompt_tokens = (
                        restored_prefix_terminal.original_prompt_token_count
                    )
                    response.usage.completion_tokens = (
                        restored_prefix_terminal.prefix_token_count
                    )
                    response.usage.total_tokens = (
                        response.usage.prompt_tokens + response.usage.completion_tokens
                    )
                    if response.prompt_token_ids is not None:
                        response.prompt_token_ids = list(
                            restored_prefix_terminal.prompt_token_ids[
                                : restored_prefix_terminal.original_prompt_token_count
                            ]
                        )
                    request_metadata = (
                        args[4] if len(args) >= 5 else kwargs.get("request_metadata")
                    )
                    if request_metadata is not None:
                        request_metadata.final_usage_info = response.usage

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
            except MediaCaptureRejected as e:
                # Raised inside preprocess_chat before begin_call, so no capture
                # state exists yet and the abort below is a no-op kept for
                # symmetry. Return a 400 carrying a stable code so Gym and the
                # worker log can distinguish a retained-media re-tile from a
                # real engine error (which stays a 500 below).
                worker_self._abort_request_capture(request, reason=e.code)
                LOGGER.warning(
                    "Rejected captured call before inference (%s): %s", e.code, e
                )
                return JSONResponse(
                    content={
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "param": "messages",
                            "code": e.code,
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
            prompt = self._tokenize_prompt_with_bos(prompt)

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

            validate_rollout_prompt(
                input_ids_batch[sample_idx, :current_input_actual_length].tolist(),
                final_request_output.prompt_token_ids,
            )

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

    async def begin_token_capture_snapshot_fence_async(self) -> bool:
        """Fence terminal staging without pausing vLLM decoding."""
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "begin_token_capture_snapshot_fence_async requires async_engine=True"
            )
        gate = self._token_capture_snapshot_gate
        # Take the epoch on the loop, in call order with the matching end: if
        # the driver times out and releases first, the late close is a no-op.
        epoch = gate.begin_epoch()
        await asyncio.get_running_loop().run_in_executor(
            self._token_capture_fence_executor, gate.close_and_wait, epoch
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

    async def end_token_capture_snapshot_fence_async(self) -> bool:
        """Release terminal token staging after the TQ snapshot is durable."""
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "end_token_capture_snapshot_fence_async requires async_engine=True"
            )
        self._token_capture_snapshot_gate.reopen()
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
            generation_prefix_reader = getattr(self, "_generation_prefix_reader", None)
            if generation_prefix_reader is not None:
                await generation_prefix_reader.aclose()
            # A terminal write may be waiting at the TQ snapshot fence when the
            # actor is asked to stop. Release it before retiring the isolated
            # cut-control and fence executors; waiting for a wedged TQ write
            # here would otherwise make actor shutdown hang indefinitely.
            token_capture_snapshot_gate = getattr(
                self, "_token_capture_snapshot_gate", None
            )
            if token_capture_snapshot_gate is not None:
                token_capture_snapshot_gate.reopen()
            for executor_name in (
                "_generation_cut_control_executor",
                "_token_capture_fence_executor",
            ):
                executor = getattr(self, executor_name, None)
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
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
