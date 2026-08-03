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

"""Ray actor wrapping ``tensorrt_llm._torch.async_llm.AsyncLLM``.

The sole TRT-LLM generation worker (the synchronous engine path was removed;
see :class:`TrtllmGeneration`, which asserts ``trtllm_cfg.async_engine=true``).
Every method that calls into ``AsyncLLM`` is exposed as ``async def`` with the
``_async`` suffix, so Ray's actor runtime runs them on the actor's own asyncio
loop; process-lifecycle / helper methods (e.g. ``shutdown``,
``configure_worker``) stay sync.

Weight updates flow through ``NcclExtension`` inside TRT-LLM's internal
``RayGPUWorker``, invoked via ``llm.collective_rpc()``.
"""

import asyncio
import copy
import gc
import json
import os
import threading
from typing import Any, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.trtllm.config import TrtllmConfig


class TrtllmAsyncGenerationWorkerImpl:
    """Plain (non-actor) implementation of the async TRT-LLM generation worker.

    Held separately from the ``@ray.remote``-wrapped
    :class:`TrtllmAsyncGenerationWorker` so it can be exercised without Ray.
    """

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
        num_gpus_per_node: Optional[int] = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
        # TRT-LLM with orchestrator_type="ray" creates its own internal
        # Ray actors that each need a GPU.  The outer actor therefore
        # gives up its GPU reservation.
        resources: dict[str, Any] = {"num_gpus": 0, "num_cpus": 0}
        # TRT-LLM's CudaRunner derives NVRTC -I include paths via
        # `popen("pip show tensorrt_llm")`. Pin the worker's actor python's
        # bin dir to the front of PATH so `pip` resolves to the interpreter
        # whose site-packages contain tensorrt_llm.
        worker_py_bin_dir = os.path.dirname(PY_EXECUTABLES.TRTLLM)
        worker_path = f"{worker_py_bin_dir}:" + os.environ.get("PATH", "")
        env_vars: dict[str, str] = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "PATH": worker_path,
        }
        init_kwargs: dict[str, Any] = {}

        if bundle_indices is not None:
            # Pass bundle_indices through __init__ kwargs; the worker resolves the
            # parent placement group via get_current_placement_group() and hands both
            # to TRT-LLM as ray_placement_config (instead of TRTLLM_RAY_BUNDLE_INDICES).
            init_kwargs["bundle_indices"] = bundle_indices[1]

        init_kwargs["fraction_of_gpus"] = num_gpus

        return resources, env_vars, init_kwargs, {}

    def __repr__(self) -> str:
        return "TrtllmAsyncGenerationWorker"

    def __init__(
        self,
        config: TrtllmConfig,
        bundle_indices: Optional[list[int]] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = config
        # Allow gen side to use a quantized checkpoint
        self.model_name = (
            self.cfg.get("trtllm_cfg", {}).get("model_name") or self.cfg["model_name"]
        )
        self.is_model_owner = bundle_indices is not None
        self._bundle_indices = bundle_indices
        self._fraction_of_gpus = fraction_of_gpus
        self._seed = seed
        self.llm = None
        self.TrtSamplingParams = None
        self._http_thread = None
        self._http_base_url: Optional[str] = None
        self._http_server = None

        # In-flight batching telemetry (mirrors the vLLM metrics logger).
        # Populated by a background asyncio task started in post_init_async
        # when trtllm_cfg.enable_trtllm_metrics_logger is set.
        self._trtllm_metrics_enabled = bool(
            self.cfg["trtllm_cfg"].get("enable_trtllm_metrics_logger")
        )
        self._trtllm_metrics_lock = threading.Lock()
        self._stats_task: Optional[asyncio.Task] = None
        self.inflight_batch_sizes: list[int] = []
        self.num_pending_samples: list[int] = []
        # KV-cache utilization as a 0-1 fraction (mirrors vLLM's
        # kv_cache_usage_perc) so both backends' figures are comparable.
        self.kv_cache_usage_perc: list[float] = []
        self._kv_stats_keys_logged = False

        if not self.is_model_owner:
            return

        from ray.util.placement_group import get_current_placement_group
        from tensorrt_llm import AsyncLLM
        from tensorrt_llm import SamplingParams as TrtSamplingParams
        from tensorrt_llm.llmapi.llm_args import (
            CapacitySchedulerPolicy,
            CudaGraphConfig,
            ExecutorMemoryType,
            KvCacheConfig,
            SchedulerConfig,
            SleepConfig,
        )

        self.TrtSamplingParams = TrtSamplingParams

        # Install the sync `fetch_stats_serialized` worker method BEFORE AsyncLLM
        # spawns its RayGPUWorker engine processes, so they import the patched
        # BaseWorker. See _ifb_stats_patch for the full rationale (the metric can
        # only be read from the real engine via collective_rpc, which needs a sync
        # picklable fetch). Applied here in-process + via a .pth drop so the
        # separate worker processes also run it.
        if self._trtllm_metrics_enabled:
            self._install_ifb_stats_patch()

        trtllm_cfg = self.cfg["trtllm_cfg"]
        tp_size = trtllm_cfg["tensor_parallel_size"]
        self._colocated = self.cfg["colocated"]["enabled"]

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        pg = get_current_placement_group()
        assert pg is not None, (
            "TrtllmAsyncGenerationWorker must be scheduled inside a Ray placement "
            "group; got None from get_current_placement_group()."
        )
        # AsyncLLM.__init__ has its own placement_groups /
        # placement_bundle_indices / per_worker_gpu_share named params that
        # unconditionally overwrite kwargs["ray_placement_config"] — so we
        # must pass these as top-level kwargs, not via ray_placement_config.
        print(
            f"[TrtllmAsyncWorker] bundle_indices={self._bundle_indices}, "
            f"pg={pg}, bundle_specs={pg.bundle_specs}",
            flush=True,
        )

        # TRT-LLM expects one bundle-index list per placement group. A unified
        # PG can contain bundles on multiple nodes, allowing one TP replica to
        # span them while Ray still pins every rank to a specific GPU bundle.
        placement_groups_list = [pg]
        placement_bundle_indices_list = [list(self._bundle_indices)]

        llm_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            backend="pytorch",
            tensor_parallel_size=tp_size,
            dtype=trtllm_cfg["precision"],
            max_seq_len=trtllm_cfg["max_model_len"],
            max_batch_size=trtllm_cfg["max_batch_size"],
            max_num_tokens=trtllm_cfg["max_num_tokens"],
            # vLLM accepts prompts up to max_model_len (no separate input cap; it clamps output so
            # input+output <= max_model_len). TRT-LLM defaults max_input_len=1024, which rejects long
            # SWE-agent prompts before any tokens generate -> NeMo Gym sees "no generation data".
            # Match the input cap to the context window so it isn't the bottleneck.
            max_input_len=trtllm_cfg["max_model_len"],
            orchestrator_type="ray",
            ray_worker_extension_cls="nemo_rl.models.generation.trtllm.trtllm_backend.NcclExtension",
            placement_groups=placement_groups_list,
            placement_bundle_indices=placement_bundle_indices_list,
            trust_remote_code=True,
            scheduler_config=SchedulerConfig(
                capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ),
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True,
                max_batch_size=trtllm_cfg["max_batch_size"],
            ),
        )

        # Pull the optional KvCacheConfig sub-dict out of trtllm_kwargs. Its
        # fields (free_gpu_memory_fraction, mamba_ssm_cache_dtype, ...) live on
        # KvCacheConfig, not on LlmArgs, so they can't be spread as top-level
        # AsyncLLM kwargs (which are validated against LlmArgs.model_fields and
        # reject unknown keys) — pass them via kv_cache_config instead. The rest
        # of trtllm_kwargs is spread as top-level AsyncLLM kwargs below.
        extra_trtllm_kwargs = dict(self.cfg.get("trtllm_kwargs") or {})
        kv_cache_kwargs = dict(extra_trtllm_kwargs.pop("kv_cache_config", None) or {})

        # gpu_memory_utilization is a dedicated trtllm_cfg knob for
        # free_gpu_memory_fraction; an explicit kv_cache_config value wins.
        gpu_mem_util = trtllm_cfg.get("gpu_memory_utilization")
        if gpu_mem_util is not None:
            kv_cache_kwargs.setdefault("free_gpu_memory_fraction", gpu_mem_util)
        if kv_cache_kwargs:
            llm_kwargs["kv_cache_config"] = KvCacheConfig(**kv_cache_kwargs)

        moe_tp = trtllm_cfg.get("moe_tensor_parallel_size")
        moe_ep = trtllm_cfg.get("moe_expert_parallel_size")
        if moe_tp is not None:
            llm_kwargs["moe_tensor_parallel_size"] = moe_tp
        if moe_ep is not None:
            llm_kwargs["moe_expert_parallel_size"] = moe_ep

        # Colocated: share each bundle's GPU 0.5/0.5 with the policy actor.
        # RayWorkerWrapper does ray.get_gpu_ids()[0], so num_gpus must be > 0.
        if self._colocated:
            # NeMo-RL's validation does sleep+wakeup without a refit, so freeing
            # the weights ("NONE") leaves them unrestored. Drop this memory
            # optimization for now.
            llm_kwargs["sleep_config"] = SleepConfig(
                restore_modes={
                    # ExecutorMemoryType.MODEL_WEIGHTS_MAIN: "NONE",
                    ExecutorMemoryType.KV_CACHE: "NONE",
                }
            )
            llm_kwargs["per_worker_gpu_share"] = self._fraction_of_gpus

        # Enable per-iteration performance stats so get_stats_async() yields
        # inflightBatchingStats (the TRT-LLM analog of vLLM's
        # vllm:num_requests_running). Set before the extra-kwargs spread so the
        # user can still override it via trtllm_kwargs.
        if self._trtllm_metrics_enabled:
            llm_kwargs["enable_iter_perf_stats"] = True

        # Escape hatch: spread remaining user-provided TRT-LLM kwargs last so
        # they can override anything above for advanced tuning.
        llm_kwargs.update(extra_trtllm_kwargs)

        # Propagate the nsight runtime_env down to TRT-LLM's internal Ray GPU
        # workers.  The outer actor's @ray.remote nsight config does NOT inherit
        # into TRT-LLM's RayExecutor workers (ray_executor.py sets an explicit
        # runtime_env); passing ray_worker_nsight_options is the sanctioned hook
        # (ray_executor.py:120,147). Returns {} when profiling is off → no-op.
        _nsight = get_nsight_config_if_pattern_matches(
            "trtllm_async_generation_worker"
        ).get("nsight")
        if _nsight and "ray_worker_nsight_options" not in llm_kwargs:
            llm_kwargs["ray_worker_nsight_options"] = _nsight

        # Defer __await__ (which fires setup_async) to post_init_async so
        # AsyncLLM setup runs on the Ray actor's asyncio loop.
        self.llm = AsyncLLM(**llm_kwargs)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def is_alive(self) -> bool:
        return True

    async def post_init_async(self) -> None:
        """Finish async engine setup on the Ray actor's asyncio loop and (optionally) start HTTP server."""
        if not self.is_model_owner or self.llm is None:
            return

        print("[TrtllmAsyncWorker] post_init_async: awaiting setup_async…", flush=True)
        await self.llm.setup_async()
        print("[TrtllmAsyncWorker] AsyncLLM ready", flush=True)

        if self.cfg["trtllm_cfg"].get("expose_http_server"):
            self.start_http_server()

        if self._trtllm_metrics_enabled:
            self._stats_task = asyncio.create_task(self._drain_stats_loop())
            print(
                "📋[TRT-LLM Metric Logger] stats drain task started",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    #  In-flight batching telemetry
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_stat_field(obj: Any, *keys: str) -> Any:
        """Read a nested field from a stats record that may be a dict or an
        attribute-style object (TRT-LLM has returned both shapes across
        versions). Returns None if any level is missing."""
        cur = obj
        for k in keys:
            if cur is None:
                return None
            if isinstance(cur, dict):
                cur = cur.get(k)
            else:
                cur = getattr(cur, k, None)
        return cur

    def _install_ifb_stats_patch(self) -> None:
        """Expose a SYNC, Ray-picklable ``fetch_stats_serialized`` on TRT-LLM's
        BaseWorker so the drain can read iteration stats via collective_rpc.

        The metric can only be read from the real engine worker via
        collective_rpc (the RPC-client get_stats path hits a stats-less engine);
        collective_rpc returns unpicklable IterationStats objects and the
        built-in serialized fetches are async (unusable through the sync
        RayGPUWorker.call_worker_method). We add a plain sync serializing method.
        Applied both in this actor's process and — because the engine runs in
        separate RayGPUWorker processes that do not import nemo_rl — by appending
        a module-level patch to the shared-venv base_worker.py source those
        processes import. See _ifb_stats_patch for the full rationale.
        """
        try:
            from nemo_rl.models.generation.trtllm import _ifb_stats_patch

            _ifb_stats_patch.apply()
        except Exception as e:  # pragma: no cover
            print(
                f"[TrtllmAsyncWorker] ifb in-process patch skipped: {e!r}",
                flush=True,
            )

        marker = "# --- nemo-rl IFB metric patch ---"
        block = (
            "\n\n" + marker + "\n"
            "def _nemorl_fetch_stats_serialized(self):\n"
            "    return [self._stats_serializer(s) for s in self.fetch_stats()]\n"
            "try:\n"
            "    BaseWorker.fetch_stats_serialized = _nemorl_fetch_stats_serialized\n"
            "except Exception:\n"
            "    pass\n"
            "# --- end nemo-rl IFB metric patch ---\n"
        )
        try:
            import fcntl

            import tensorrt_llm.executor.base_worker as _bw

            path = os.path.abspath(_bw.__file__)
            # a+ appends regardless of seek; flock serializes the co-located
            # replica actors that share this node's container venv.
            with open(path, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    already = marker in f.read()
                    if not already:
                        f.write(block)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            print(
                f"[TrtllmAsyncWorker] IFB stats patch "
                f"{'present' if already else 'appended'} in {path}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[TrtllmAsyncWorker] ifb base_worker append skipped: {e!r}",
                flush=True,
            )

    async def _drain_stats_loop(self) -> None:
        """Continuously drain per-iteration stats from the engine and record
        the in-flight batching size (context + generation requests scheduled
        this iteration)."""
        assert self.llm is not None
        interval = float(
            self.cfg["trtllm_cfg"]["trtllm_metrics_logger_interval"]
        )
        _err_logged = 0
        while True:
            try:
                # Read iteration stats from the real engine workers via
                # collective_rpc("fetch_stats_serialized") — a sync serializing
                # method injected onto TRT-LLM's BaseWorker (see
                # _install_ifb_stats_patch). On the _torch AsyncLLM Ray backend the
                # RPC-proxy get_stats()/get_stats_async() reach a stats-less engine
                # (always empty), and raw collective_rpc("fetch_stats") returns
                # unpicklable IterationStats objects; fetch_stats_serialized returns
                # JSON strings (camelCase inflightBatchingStats.numScheduledRequests).
                try:
                    rank_results = await self.llm.collective_rpc(
                        "fetch_stats_serialized"
                    )
                except Exception as e:
                    rank_results = None
                    if _err_logged < 5:
                        _err_logged += 1
                        print(
                            "⚠️[TRT-LLM Metric Logger] "
                            f"collective_rpc(fetch_stats_serialized) error: {e!r}",
                            flush=True,
                        )
                # Every TP rank of an engine accumulates the SAME per-iteration
                # stats, so take a single rank (the first non-empty one) to avoid
                # double-counting the timeline.
                records: list[Any] = []
                for _rank_res in rank_results or []:
                    if _rank_res:
                        records = _rank_res
                        break
                # Append EXACTLY ONE sample per drain interval to mirror vLLM's
                # _logger_loop (which samples a single gauge value per interval).
                # The plot's x-axis assumes one point per interval (x = index *
                # timeline_interval), so appending every per-iteration engine
                # record (tens of thousands/step) would both over-densify the
                # series and massively overstate wall-clock. Take the LAST drained
                # record as the instantaneous sample; when the engine is idle and
                # no records are returned, append 0 to both (mirrors vLLM logging
                # the current 0-gauge) so every interval yields one point.
                ifb_sample = 0
                ctx_sample = 0
                kv_sample = 0.0
                if records:
                    stats = records[-1]
                    if isinstance(stats, (str, bytes, bytearray)):
                        try:
                            stats = json.loads(stats)
                        except (ValueError, TypeError):
                            stats = None
                    if stats is not None:
                        ifb = self._get_stat_field(
                            stats, "inflightBatchingStats", "numScheduledRequests"
                        )
                        ctx = self._get_stat_field(
                            stats, "inflightBatchingStats", "numContextRequests"
                        )
                        ifb_sample = int(ifb) if ifb is not None else 0
                        # numContextRequests is prefill-only; approximates
                        # queued/pending prefill work. Keep the same two-series
                        # shape the consumer asserts on.
                        ctx_sample = int(ctx) if ctx is not None else 0
                        # KV-cache utilization as a 0-1 fraction, mirroring
                        # vLLM's kv_cache_usage_perc. TRT-LLM's serialized
                        # IterationStats exposes a kvCacheStats sub-object with
                        # block counts (usedNumBlocks / maxNumBlocks /
                        # freeNumBlocks) — not a ready percent — so derive it:
                        #   kv = usedNumBlocks / maxNumBlocks
                        # (fallback 1 - freeNumBlocks/maxNumBlocks), guarded for
                        # maxNumBlocks == 0 -> 0.0. One-time log of the observed
                        # kvCacheStats keys to ease field-name debugging.
                        kv_stats = self._get_stat_field(stats, "kvCacheStats")
                        if kv_stats is not None and not self._kv_stats_keys_logged:
                            self._kv_stats_keys_logged = True
                            _keys = (
                                list(kv_stats.keys())
                                if isinstance(kv_stats, dict)
                                else dir(kv_stats)
                            )
                            print(
                                "📋[TRT-LLM Metric Logger] kvCacheStats keys: "
                                f"{_keys}",
                                flush=True,
                            )
                        max_blocks = self._get_stat_field(kv_stats, "maxNumBlocks")
                        used_blocks = self._get_stat_field(kv_stats, "usedNumBlocks")
                        free_blocks = self._get_stat_field(kv_stats, "freeNumBlocks")
                        try:
                            max_blocks = float(max_blocks) if max_blocks is not None else 0.0
                            if max_blocks > 0:
                                if used_blocks is not None:
                                    kv_sample = float(used_blocks) / max_blocks
                                elif free_blocks is not None:
                                    kv_sample = 1.0 - float(free_blocks) / max_blocks
                                # Clamp into [0, 1] to match vLLM's fraction.
                                kv_sample = min(1.0, max(0.0, kv_sample))
                        except (TypeError, ValueError):
                            kv_sample = 0.0
                with self._trtllm_metrics_lock:
                    self.inflight_batch_sizes.append(ifb_sample)
                    self.num_pending_samples.append(ctx_sample)
                    self.kv_cache_usage_perc.append(kv_sample)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if _err_logged < 5:
                    _err_logged += 1
                    print(
                        f"⚠️[TRT-LLM Metric Logger] stats drain error: {e}",
                        flush=True,
                    )
            # Always back off between drain cycles, not just on error. get_stats()
            # already blocks up to `interval` collecting queued records; this extra
            # sleep guards against a tight loop if it ever returns immediately
            # (e.g. before prompts are submitted) so we never busy-spin.
            await asyncio.sleep(interval)

    def get_trtllm_logger_metrics(self) -> dict[str, Any]:
        if not self._trtllm_metrics_enabled:
            return {}
        with self._trtllm_metrics_lock:
            return {
                "inflight_batch_sizes": copy.deepcopy(self.inflight_batch_sizes),
                "num_pending_samples": copy.deepcopy(self.num_pending_samples),
                "kv_cache_usage_perc": copy.deepcopy(self.kv_cache_usage_perc),
            }

    def clear_trtllm_logger_metrics(self) -> None:
        if not self._trtllm_metrics_enabled:
            return
        with self._trtllm_metrics_lock:
            self.inflight_batch_sizes = []
            self.num_pending_samples = []
            self.kv_cache_usage_perc = []

    def shutdown(self) -> bool:
        if self._stats_task is not None:
            self._stats_task.cancel()
            self._stats_task = None
        try:
            self.stop_http_server()
            if self.llm is not None:
                del self.llm
                self.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"Error during TRT-LLM shutdown: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  HTTP server for NeMo Gym
    # ------------------------------------------------------------------ #

    def start_http_server(self, port: int = 0) -> str:
        """Start an OpenAI-compatible HTTP server backed by ``self.llm``."""
        if self._http_base_url is not None:
            return self._http_base_url

        from transformers import AutoTokenizer

        from nemo_rl.models.generation.trtllm.trtllm_http_server import start_server

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._http_thread, self._http_base_url, self._http_server = start_server(
            llm=self.llm,
            tokenizer=tokenizer,
            model_name=self.model_name,
            port=port,
            max_seq_len=self.cfg["trtllm_cfg"]["max_model_len"],
            sampling_config={
                "temperature": self.cfg["temperature"],
                "top_p": self.cfg["top_p"],
                "top_k": self.cfg["top_k"],
            },
            stop_token_ids=list(self.cfg.get("stop_token_ids") or []),
            default_chat_template_kwargs=self.cfg["trtllm_cfg"].get(
                "default_chat_template_kwargs"
            ),
            tool_parser=self.cfg["trtllm_cfg"].get("tool_parser"),
            reasoning_parser=self.cfg["trtllm_cfg"].get("reasoning_parser"),
        )
        print(
            f"[TrtllmAsyncWorker] HTTP server started: {self._http_base_url}",
            flush=True,
        )
        return self._http_base_url

    def stop_http_server(self) -> None:
        if self._http_server is not None:
            self._http_server.should_exit = True
            self._http_server = None
            self._http_thread = None
            self._http_base_url = None

    async def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self._http_base_url

    # ------------------------------------------------------------------ #
    #  Collective RPC / refit
    # ------------------------------------------------------------------ #

    async def init_collective_async(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        assert self.llm is not None
        await self.llm.collective_rpc(
            "init_collective",
            args=(rank_prefix, ip, port, world_size, train_world_size),
        )

    async def prepare_refit_info_async(self, state_dict_info: dict[str, Any]) -> None:
        assert self.llm is not None
        await self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    async def update_weights_from_collective_async(
        self, *, drain: bool = True, recompute_kv: bool = False
    ) -> bool:
        """Async version of ``update_weights_from_collective``.

        Args:
            drain: If False, run the refit at a scheduler step boundary
                without draining in-flight requests (in-flight weight
                update). Default True preserves the original drain-first
                behavior.
            recompute_kv: If True (and ``drain=False``), preempt all
                in-flight requests after the refit so the scheduler
                re-prefills them under the new weights.
        """
        assert self.llm is not None
        try:
            results = await self.llm.collective_rpc(
                "update_weights_from_collective",
                kwargs={"drain": drain, "recompute_kv": recompute_kv},
            )
            worker_result = results[0] if results else True
            if not worker_result:
                print(
                    f"Error: TRT-LLM worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during TRT-LLM async collective weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def update_weights_via_ipc_zmq_async(self) -> bool:
        assert self.llm is not None
        try:
            results = await self.llm.collective_rpc("update_weights_via_ipc_zmq")
            worker_result = results[0] if results else True
            if not worker_result:
                print(
                    f"Error: TRT-LLM worker failed to update weights via IPC. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during TRT-LLM async IPC weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def report_device_id_async(self) -> list[str]:
        assert self.llm is not None
        return await self.llm.collective_rpc("report_device_id")

    @classmethod
    def _weights_tags(cls) -> list[str]:
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        return [
            t.value
            for t in ExecutorMemoryType
            if t is not ExecutorMemoryType.KV_CACHE and not t.value.startswith("_")
        ]

    @classmethod
    def _all_sleep_tags(cls) -> list[str]:
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        return cls._weights_tags() + [ExecutorMemoryType.KV_CACHE.value]

    def _resolve_wake_tags(self, tags: Optional[list[str]]) -> list[str]:
        if not tags:
            return self._all_sleep_tags()
        out: list[str] = []
        for t in tags:
            if t == "weights":
                out.extend(self._weights_tags())
            elif t == "kv_cache":
                from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

                out.append(ExecutorMemoryType.KV_CACHE.value)
            else:
                out.append(t)
        return out

    async def sleep_async(self, **kwargs: Any) -> bool:
        # reset_prefix_cache before release: TRT-LLM's release() frees
        # kv_cache memory but doesn't invalidate the prefix-reuse index, so
        # the next wake-up would point at stale entries.
        if self.llm is None:
            return True
        await self.reset_prefix_cache_async()
        await self.llm.release(self._all_sleep_tags())
        gc.collect()
        torch.cuda.empty_cache()
        return True

    async def wake_up_async(self, **kwargs: Any) -> bool:
        if self.llm is None:
            return True
        tags = self._resolve_wake_tags(kwargs.get("tags"))
        await self.llm.resume(tags)
        return True

    async def reset_prefix_cache_async(self, **kwargs: Any) -> bool:
        if self.llm is None:
            return True
        await self.llm.collective_rpc("reset_prefix_cache")
        return True

    # ------------------------------------------------------------------ #
    #  GPU profiling (nsys capture-range trigger)
    # ------------------------------------------------------------------ #

    async def start_gpu_profiling_async(self) -> None:
        # Outer actor is CPU-only; broadcast to the internal GPU workers.
        if self.llm is not None:
            await self.llm.collective_rpc("start_gpu_profiling")

    async def stop_gpu_profiling_async(self) -> None:
        if self.llm is not None:
            await self.llm.collective_rpc("stop_gpu_profiling")

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if len(data["input_ids"]) == 0:
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": torch.zeros((0, 0), dtype=torch.long),
                    "logprobs": torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths": torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
                }
            )

        assert self.llm is not None
        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        padded_input_length = input_ids.size(1)

        prompts = []
        for i in range(len(input_ids)):
            length = input_lengths[i].item()
            token_ids = input_ids[i, :length].tolist()
            prompts.append({"prompt_token_ids": token_ids})

        sampling_params = self._build_sampling_params(greedy=greedy)

        # Fan all prompts out concurrently; AsyncLLM batches them in-flight.
        outputs = await asyncio.gather(
            *[
                self.llm.generate_async(inputs=p, sampling_params=sampling_params)
                for p in prompts
            ]
        )

        output_ids_list = []
        logprobs_list = []
        generation_lengths = []
        unpadded_sequence_lengths = []

        max_gen_len = max(len(o.outputs[0].token_ids) for o in outputs)

        for i, output in enumerate(outputs):
            seq_len = input_lengths[i].item()
            gen = output.outputs[0]
            gen_tokens = list(gen.token_ids)
            total_length = padded_input_length + max_gen_len

            full_output = torch.full(
                (total_length,),
                self.cfg["_pad_token_id"],
                dtype=input_ids.dtype,
            )
            full_output[:seq_len] = input_ids[i][:seq_len]
            full_output[seq_len : seq_len + len(gen_tokens)] = torch.tensor(gen_tokens)
            output_ids_list.append(full_output)

            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if gen.logprobs:
                for idx, lp in enumerate(gen.logprobs):
                    pos = seq_len + idx
                    if pos < total_length:
                        if isinstance(lp, (int, float)):
                            full_logprobs[pos] = float(lp)
                        elif isinstance(lp, dict):
                            full_logprobs[pos] = next(iter(lp.values())).logprob
                        else:
                            full_logprobs[pos] = float(lp)
            logprobs_list.append(full_logprobs)

            resp_len = seq_len + len(gen_tokens)
            generation_lengths.append(len(gen_tokens))
            unpadded_sequence_lengths.append(resp_len)

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.stack(output_ids_list),
                "logprobs": torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(
                    generation_lengths, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths, dtype=torch.long
                ),
            }
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _build_sampling_params(self, *, greedy: bool):
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else 0)
        temperature = 0.0 if greedy else self.cfg["temperature"]

        stop_ids = list(self.cfg.get("stop_token_ids") or [])

        return self.TrtSamplingParams(
            temperature=temperature,
            top_p=self.cfg["top_p"],
            top_k=top_k_val,
            max_tokens=self.cfg["max_new_tokens"],
            stop_token_ids=stop_ids or None,
            # Keep the EOS / stop token in the returned token_ids so that the
            # response sequence matches HF / vLLM behavior. Required for
            # logprob alignment with training-side Megatron.
            include_stop_str_in_output=True,
            logprobs=True,
            logprobs_simple_format=True,
        )


@ray.remote(
    num_cpus=0,
    runtime_env={
        **get_nsight_config_if_pattern_matches("trtllm_async_generation_worker")
    },
)  # pragma: no cover
class TrtllmAsyncGenerationWorker(TrtllmAsyncGenerationWorkerImpl):
    """Ray actor wrapper around :class:`TrtllmAsyncGenerationWorkerImpl`."""

    pass
