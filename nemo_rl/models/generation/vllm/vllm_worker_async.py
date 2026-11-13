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
import os
import threading
import time
import uuid
import warnings
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch

from nemo_rl.data.captured_media import MediaColumnSpec
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.data_plane.interfaces import DataPlaneConfig, LocalDataPlaneConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.refit_watchdog import RefitAborted, is_refit_abort
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _bind_socket_in_range,
    _get_node_ip_local,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.dynamo.config import dynamo_cfg, dynamo_enabled
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.prefix_recovery import NoPrefixRecovery
from nemo_rl.models.generation.rollout_result import RolloutResult
from nemo_rl.models.generation.vllm.checkpoint_engine import (
    VllmAsyncCheckpointEngineRpcMixin,
)
from nemo_rl.models.generation.vllm.collective_rpc import (
    resolve_collective_rpc_result,
)
from nemo_rl.models.generation.vllm.config import (
    VllmConfig,
    parse_nvfp4_pertoken_rollout,
)
from nemo_rl.models.generation.vllm.frontend import VllmHTTPFrontend
from nemo_rl.models.generation.vllm.multi_api import MultiAPIServerPool
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

        # The task serving this engine through Dynamo, when it is registered.
        self._dynamo_endpoint_task: Optional[asyncio.Task] = None

        self._frontend: VllmHTTPFrontend | None = None
        self._engine_loop = None
        self._api_server_pool = None

        if bundle_indices is not None and config["vllm_cfg"].get("expose_http_server"):
            self._reserve_port(config)

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

        self.llm = None
        self.vllm_device_ids = None

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

    def _reserve_port(self, config: VllmConfig) -> None:
        """Reserve a listening socket within the configured generation port range.

        The socket is held open in LISTENING state and later passed directly to
        uvicorn via the ``sockets=`` parameter in ``server.serve()``. The socket
        is never closed and re-opened, so there is zero gap where another process
        could steal the port.
        """
        import socket

        self._reserved_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reserved_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _bind_socket_in_range(
            self._reserved_socket,
            config.get("port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW),
            config.get("port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH),
        )
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
            from nemo_rl.models.generation.dynamo.vllm_adapter import (
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

        # Initialize the vLLM engine
        api_server_count = self.cfg["vllm_cfg"].get("http_api_server_count", 1)
        if api_server_count > 1:
            # Multiple API servers require separate frontend and engine setup.
            self._api_server_pool = MultiAPIServerPool(self, api_server_count)
            self.llm = self._api_server_pool.create_engine(
                self.llm_async_engine_args, self.stat_loggers
            )
        else:
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
        if self.llm is not None:
            self._frontend = VllmHTTPFrontend(
                self.cfg,
                engine_client=self.llm,
                engine_args=self.llm_async_engine_args,
                engine_loop=self._engine_loop,
                routed_experts_dtype=self.routed_experts_dtype,
                listener=self._reserved_socket,
                node_ip=self._reserved_node_ip,
                register_routes=(
                    self._sparse_refit_receiver.setup_api_server
                    if self._sparse_refit_receiver is not None
                    else None
                ),
            )
            self._reserved_socket = None
            if self.cfg["vllm_cfg"].get("expose_http_server"):
                self._frontend.start()
                if self._api_server_pool is not None:
                    await self._api_server_pool.command("start")

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
        from nemo_rl.models.generation.dynamo.vllm_adapter import DynamoVllmAdapter

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
        return self._frontend.base_url if self._frontend is not None else None

    async def setup_token_capture(
        self,
        dp_cfg: DataPlaneConfig | LocalDataPlaneConfig,
        staging_partition: str,
        *,
        capture_media: bool = False,
        top_logprobs: int = 0,
        generation_prefix_cuts_enabled: bool = False,
        generation_cut_control_token: str | None = None,
        generation_cut_control_timeout_s: float | None = None,
    ) -> MediaColumnSpec | None:
        """Install capture on the local frontend and all child frontends.

        Returns:
            The pinned media column spec, or None for text-only capture or non-owner ranks.
        """
        if not self.is_model_owner:
            return None
        if generation_prefix_cuts_enabled and self._api_server_pool is not None:
            raise ValueError(
                "Multiple HTTP API servers do not support generation-prefix cuts"
            )
        if generation_prefix_cuts_enabled and not self.cfg["vllm_cfg"].get(
            "expose_http_server"
        ):
            raise RuntimeError(
                "generation-prefix cuts need the vLLM HTTP server "
                "(vllm_cfg.expose_http_server: true)"
            )
        assert self._frontend is not None
        media_spec = await self._frontend.setup_token_capture(
            dp_cfg,
            staging_partition,
            capture_media=capture_media,
            top_logprobs=top_logprobs,
            generation_prefix_cuts_enabled=generation_prefix_cuts_enabled,
            generation_cut_control_token=generation_cut_control_token,
            generation_cut_control_timeout_s=generation_cut_control_timeout_s,
        )
        if self._api_server_pool is not None:
            await self._api_server_pool.command(
                "capture", dp_cfg, staging_partition, capture_media, top_logprobs
            )
        return media_spec

    async def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run owner-local checkpoint I/O without blocking the actor event loop."""
        return await asyncio.to_thread(run_checkpoint_command, body)

    def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        """Return the local frontend's capture counters, when available."""
        return (
            self._frontend.get_data_plane_snapshot()
            if self._frontend is not None
            else None
        )

    async def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the weight version stamped on subsequent captured calls."""
        if self._api_server_pool is not None:
            await self._api_server_pool.command("version", int(version))
        if self._frontend is not None:
            await self._frontend.set_rollout_weight_version(version)

    async def get_data_plane_snapshots(self) -> list[dict[str, Any]]:
        """Collect each frontend's capture counters and reset its step window.

        Returns:
            One snapshot per frontend with data-plane observability enabled.
        """
        snapshots = []
        local_snapshot = self.get_data_plane_snapshot()
        if local_snapshot is not None:
            snapshots.append(local_snapshot)
        if self._api_server_pool is not None:
            snapshots.extend(
                snapshot
                for snapshot in await self._api_server_pool.command("snapshot")
                if snapshot is not None
            )
        return snapshots

    async def captured_call_progress(self) -> Optional[dict[str, Any]]:
        """Report how far each running captured call got on the current weights.

        Called while generation is paused for a refit, so the counts are where
        each call switches to the new weights. ``None`` on ranks that serve no
        captured calls.
        """
        calls = []
        capture_enabled = False
        weight_version = 0
        if self._frontend is not None:
            weight_version = self._frontend._rollout_weight_version
            local_progress = await self._frontend.captured_call_progress()
            if local_progress is not None:
                capture_enabled = True
                calls.extend(local_progress["calls"])
        if self._api_server_pool is not None:
            for progress in await self._api_server_pool.command("progress"):
                if progress is not None:
                    if progress["weight_version"] != weight_version:
                        raise RuntimeError(
                            "HTTP frontends have different weight versions"
                        )
                    capture_enabled = True
                    calls.extend(progress["calls"])
        if not capture_enabled:
            return None
        return {
            "weight_version": weight_version,
            "calls": calls,
        }

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
        if self._frontend is not None:
            await self._frontend.flush_obsolete_generation_cut_rows_async(epoch)
        return True

    def get_generation_prefix_counts(self) -> dict[str, int]:
        """Generation-prefix restore and progress-read counts; see PrefixRecovery."""
        if self._frontend is not None:
            return self._frontend.get_generation_prefix_counts()
        return NoPrefixRecovery().counts()

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
            if self._api_server_pool is not None:
                await self._api_server_pool.command("stop")
                self._api_server_pool.stop_children()
            if self._frontend is not None:
                await self._frontend.stop()

            if self._sparse_refit_receiver is not None:
                await asyncio.to_thread(self._sparse_refit_receiver.shutdown)

            if self.llm is not None:
                # Clean up extension resources (e.g., ZMQ sockets)
                if (
                    self._api_server_pool is None
                    or self._api_server_pool._failure is None
                ):
                    await self.llm.collective_rpc("cleanup", args=tuple())
                try:
                    self.llm.shutdown()
                except Exception as e_stop:
                    print(f"Error calling shutdown_background_loop: {e_stop}")

                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            if self._api_server_pool is not None:
                self._api_server_pool.close_engines()
                self._api_server_pool = None
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
