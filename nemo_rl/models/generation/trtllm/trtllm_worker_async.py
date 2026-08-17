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
import gc
import os
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


def _tag_nsys_output(output_spec: str, tag: str) -> str:
    """Append *tag* to the filename in an nsys ``-o`` spec.

    Operates on the filename rather than substituting a known worker name,
    because the spec may be the built-in default or anything the user put in
    ``NRL_NSYS_EXTRA_OPTIONS`` -- including a directory, which must be left
    where it is so the reports still land where the user asked. The value
    arrives quoted from ``get_nsight_config_if_pattern_matches``; a
    user-supplied one may not be, so the quoting is preserved either way.

    Appended rather than prefixed so the documented report names stay
    prefix-matchable: ``trtllm_async_generation_worker_*`` keeps finding every
    report (docs/nsys-profiling.md), disaggregated or not.
    """
    quote = ""
    if len(output_spec) >= 2 and output_spec[0] == output_spec[-1]:
        if output_spec[0] in "'\"":
            quote, output_spec = output_spec[0], output_spec[1:-1]

    head, sep, name = output_spec.rpartition("/")
    return f"{quote}{head}{sep}{name}_{tag}{quote}"


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
            # The placement-group index completes the engine's identity, which
            # the worker uses to find its own entry in trtllm_cfg's
            # _engine_overrides map (see TrtllmGeneration._engine_key). Local
            # bundle indices alone are ambiguous across per-node PGs, which
            # each restart numbering at 0.
            init_kwargs["bundle_pg_idx"] = bundle_indices[0]

        init_kwargs["fraction_of_gpus"] = num_gpus

        return resources, env_vars, init_kwargs, {}

    def __repr__(self) -> str:
        return "TrtllmAsyncGenerationWorker"

    def _engine_overrides(self) -> dict[str, Any]:
        """This engine's entry in the driver-built ``_engine_overrides`` map.

        Keyed by ``(placement group index, local bundle indices)`` -- the same
        tuple ``TrtllmGeneration`` used to create this worker, so the two sides
        agree without per-worker init kwargs. Returns ``{}`` for uniform runs,
        which carry no map at all.
        """
        overrides = self.cfg["trtllm_cfg"].get("_engine_overrides")
        if not overrides or self._bundle_indices is None:
            return {}
        key = f"{self._bundle_pg_idx}:" + ",".join(
            str(i) for i in self._bundle_indices
        )
        entry = overrides.get(key)
        if entry is None:
            raise RuntimeError(
                f"No engine override entry for {key!r}. Known keys: "
                f"{sorted(overrides)}. TrtllmGeneration._engine_key() and this "
                f"lookup must stay in sync."
            )
        return entry

    def __init__(
        self,
        config: TrtllmConfig,
        bundle_indices: Optional[list[int]] = None,
        bundle_pg_idx: Optional[int] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self._bundle_pg_idx = bundle_pg_idx
        # Allow gen side to use a quantized checkpoint
        self.model_name = (
            self.cfg.get("trtllm_cfg", {}).get("model_name") or self.cfg["model_name"]
        )
        self.is_model_owner = bundle_indices is not None
        self._bundle_indices = bundle_indices
        # This engine's effective config: the shared trtllm_cfg with this
        # engine's role overrides merged over it. Everything that describes the
        # engine -- constructor kwargs, the HTTP server's limits -- must read
        # this, never the raw trtllm_cfg, or a role's overrides apply to some
        # settings and not others. Equal to trtllm_cfg without disaggregation.
        self.engine_cfg: dict[str, Any] = {
            **self.cfg["trtllm_cfg"],
            **self._engine_overrides(),
        }
        self._fraction_of_gpus = fraction_of_gpus
        self._seed = seed
        self.llm = None
        self.TrtSamplingParams = None
        self._http_thread = None
        self._http_base_url: Optional[str] = None
        self._http_addr: Optional[tuple[str, int]] = None
        self._http_server = None

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

        engine_cfg = self.engine_cfg
        # This engine's TP is the number of bundles TrtllmGeneration tied to it,
        # not the config value: under PD disaggregation with asymmetric TP the
        # context and generation engines are different widths, and the config
        # holds only the default. Identical to engine_cfg["tensor_parallel_size"]
        # whenever the layout is uniform.
        tp_size = len(self._bundle_indices)
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
            dtype=engine_cfg["precision"],
            max_seq_len=engine_cfg["max_model_len"],
            max_batch_size=engine_cfg["max_batch_size"],
            max_num_tokens=engine_cfg["max_num_tokens"],
            # vLLM accepts prompts up to max_model_len (no separate input cap; it clamps output so
            # input+output <= max_model_len). TRT-LLM defaults max_input_len=1024, which rejects long
            # SWE-agent prompts before any tokens generate -> NeMo Gym sees "no generation data".
            # Match the input cap to the context window so it isn't the bottleneck.
            max_input_len=engine_cfg["max_model_len"],
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
                max_batch_size=engine_cfg["max_batch_size"],
            ),
        )

        # Pull the optional KvCacheConfig sub-dict out of trtllm_kwargs. Its
        # fields (free_gpu_memory_fraction, mamba_ssm_cache_dtype, ...) live on
        # KvCacheConfig, not on LlmArgs, so they can't be spread as top-level
        # AsyncLLM kwargs (which are validated against LlmArgs.model_fields and
        # reject unknown keys) — pass them via kv_cache_config instead. The rest
        # of trtllm_kwargs is spread as top-level AsyncLLM kwargs below.
        # A role may override kv_cache_config too: prefill and decode engines
        # usually want different KV pool sizes.
        extra_trtllm_kwargs = dict(self.cfg.get("trtllm_kwargs") or {})
        extra_trtllm_kwargs.update(engine_cfg.get("trtllm_kwargs") or {})
        kv_cache_kwargs = dict(extra_trtllm_kwargs.pop("kv_cache_config", None) or {})

        # gpu_memory_utilization is a dedicated engine_cfg knob for
        # free_gpu_memory_fraction; an explicit kv_cache_config value wins.
        gpu_mem_util = engine_cfg.get("gpu_memory_utilization")
        if gpu_mem_util is not None:
            kv_cache_kwargs.setdefault("free_gpu_memory_fraction", gpu_mem_util)
        if kv_cache_kwargs:
            llm_kwargs["kv_cache_config"] = KvCacheConfig(**kv_cache_kwargs)

        # PD disaggregation: every engine gets a cache transceiver. The
        # context/generation role is per *request* in TRT-LLM
        # (DisaggregatedParams.request_type), not per engine, so engines are
        # symmetric here; which pool an engine lands in is decided by the
        # replica's disagg server from its address alone.
        disagg_cfg = engine_cfg.get("disaggregation") or {}
        if disagg_cfg.get("enabled"):
            from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

            transceiver_kwargs: dict[str, Any] = {
                "backend": disagg_cfg["cache_transceiver_backend"],
            }
            if disagg_cfg.get("max_tokens_in_buffer") is not None:
                transceiver_kwargs["max_tokens_in_buffer"] = disagg_cfg[
                    "max_tokens_in_buffer"
                ]
            llm_kwargs["cache_transceiver_config"] = CacheTransceiverConfig(
                **transceiver_kwargs
            )

        moe_tp = engine_cfg.get("moe_tensor_parallel_size")
        moe_ep = engine_cfg.get("moe_expert_parallel_size")
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

        # Escape hatch: spread remaining user-provided TRT-LLM kwargs last so
        # they can override anything above for advanced tuning.
        llm_kwargs.update(extra_trtllm_kwargs)

        if engine_cfg["precision"] == "fp8":
            # Import only in TRT-LLM actors so the base NeMo-RL environment
            # does not need the optional TRT-LLM dependency.
            from tensorrt_llm.llmapi.llm_args import MoeConfig
            from transformers import AutoConfig

            from nemo_rl.models.generation.trtllm.quantization.fp8 import (
                configure_fp8_llm_kwargs,
                configure_fp8_moe_backend,
            )

            hf_config = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            is_mx = bool(engine_cfg.get("is_mx", False))
            configure_fp8_llm_kwargs(
                llm_kwargs,
                model_type=hf_config.model_type,
                is_mx=is_mx,
            )

            # Block-FP8: DeepGEMM resmooths 128x128 FP32 block scales to E8M0
            # on Blackwell, while the TRTLLM MoE backend retains the requested
            # FP32 buffers. MXFP8: only the CUTLASS backend implements it.
            # Either way, preserve any other user MoeConfig fields.
            configure_fp8_moe_backend(llm_kwargs, MoeConfig, is_mx=is_mx)

        # Propagate the nsight runtime_env down to TRT-LLM's internal Ray GPU
        # workers.  The outer actor's @ray.remote nsight config does NOT inherit
        # into TRT-LLM's RayExecutor workers (ray_executor.py sets an explicit
        # runtime_env); passing ray_worker_nsight_options is the sanctioned hook
        # (ray_executor.py:120,147). Returns {} when profiling is off → no-op.
        _nsight = get_nsight_config_if_pattern_matches(
            "trtllm_async_generation_worker"
        ).get("nsight")
        if _nsight and "ray_worker_nsight_options" not in llm_kwargs:
            _disagg_role = self.engine_cfg.get("_disagg_role")
            if _disagg_role:
                # Under disaggregation every engine reports the same worker
                # name, so the report name distinguishes them only by the %p
                # pid -- which means matching a report to the context or
                # generation side is a search through the driver log. Tag the
                # filename with the role and its ordinal (a layout can run
                # several engines of one role); %p still separates the ranks
                # within an engine. Aggregated runs have no role and are left
                # alone.
                _nsight = dict(_nsight)
                _nsight["o"] = _tag_nsys_output(
                    _nsight["o"],
                    f"{_disagg_role}{self.engine_cfg['_disagg_role_ordinal']}",
                )
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
            # Every engine serves HTTP, context and generation alike: under
            # disaggregation an address is the only way the disagg server can
            # reach one.
            self.start_http_server()

    def shutdown(self) -> bool:
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
            # engine_cfg, not trtllm_cfg: the server must not accept requests
            # longer than the engine it fronts was built for, and a role may
            # have overridden that length.
            max_seq_len=self.engine_cfg["max_model_len"],
            sampling_config={
                "temperature": self.cfg["temperature"],
                "top_p": self.cfg["top_p"],
                "top_k": self.cfg["top_k"],
            },
            # Validation rollouts are stamped with this second profile by
            # grpo.validate(); without it the server would reject them as
            # off-policy, which is why validation previously had to be pinned to
            # the train sampling params on this backend.
            val_sampling_config={
                "temperature": self.cfg["val_temperature"],
                "top_p": self.cfg["val_top_p"],
                "top_k": self.cfg["val_top_k"],
            },
            stop_token_ids=list(self.cfg.get("stop_token_ids") or []),
            default_chat_template_kwargs=self.engine_cfg.get(
                "default_chat_template_kwargs"
            ),
            tool_parser=self.engine_cfg.get("tool_parser"),
            reasoning_parser=self.engine_cfg.get("reasoning_parser"),
        )
        # host:port of the URL just returned; the disagg server is configured
        # with addresses, not URLs.
        _hostport = self._http_base_url.rsplit("/v1", 1)[0].rsplit("//", 1)[1]
        _host, _port = _hostport.rsplit(":", 1)
        self._http_addr = (_host, int(_port))

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
            self._http_addr = None

    async def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self._http_base_url

    async def report_http_addr(self) -> Optional[dict[str, Any]]:
        """This engine's ``{host, port, node_id}``.

        Under disaggregation an address is the only thing the replica's disagg
        server is given about an engine, so host/port is what feeds its context
        and generation pools. The node id lets the driver put that server's
        actor beside its engines.
        """
        if self._http_addr is None:
            return None
        host, port = self._http_addr
        return {
            "host": host,
            "port": port,
            "node_id": ray.get_runtime_context().get_node_id(),
        }

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
            if not results:
                print("Error: TRT-LLM weight update returned no worker results.")
                return False
            failed_workers = [
                (rank, result)
                for rank, result in enumerate(results)
                if not result
            ]
            if failed_workers:
                print(
                    "Error: TRT-LLM workers failed to update weights. "
                    f"Results: {failed_workers}"
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
            if not results:
                print("Error: TRT-LLM IPC weight update returned no worker results.")
                return False
            failed_workers = [
                (rank, result)
                for rank, result in enumerate(results)
                if not result
            ]
            if failed_workers:
                print(
                    "Error: TRT-LLM workers failed to update weights via IPC. "
                    f"Results: {failed_workers}"
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
