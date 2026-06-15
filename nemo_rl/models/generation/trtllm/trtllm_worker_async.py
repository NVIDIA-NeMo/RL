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


class TrtllmAsyncGenerationWorkerImpl:
    """Plain (non-actor) implementation of the async TRT-LLM generation worker.

    Held separately from the ``@ray.remote``-wrapped
    :class:`TrtllmAsyncGenerationWorker` so it can be exercised without Ray.
    """

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
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

        return resources, env_vars, init_kwargs, {}

    def __repr__(self) -> str:
        return "TrtllmAsyncGenerationWorker"

    def __init__(
        self,
        config: TrtllmConfig,
        bundle_indices: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.model_name = self.cfg["model_name"]
        self.is_model_owner = bundle_indices is not None
        self._bundle_indices = bundle_indices
        self._seed = seed
        self.llm = None
        self.TrtSamplingParams = None

        if not self.is_model_owner:
            return

        from tensorrt_llm import AsyncLLM, SamplingParams as TrtSamplingParams
        from tensorrt_llm.llmapi.llm_args import (
            CapacitySchedulerPolicy,
            CudaGraphConfig,
            KvCacheConfig,
            SchedulerConfig,
            SleepConfig,
            ExecutorMemoryType,
        )
        from ray.util.placement_group import get_current_placement_group

        self.TrtSamplingParams = TrtSamplingParams

        trtllm_cfg = self.cfg["trtllm_cfg"]
        tp_size = trtllm_cfg["tensor_parallel_size"]
        self._colocated = bool(self.cfg.get("colocated", {}).get("enabled", False))

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

        precision = trtllm_cfg.get("precision", "bfloat16")

        # One PG entry per worker (placement_groups=[pg]*N,
        # placement_bundle_indices=[[i0], [i1], ...]) so the expansion
        # generalises to multi-PG layouts (cross-node TP) when
        # configure_worker surfaces per-rank PGs.
        n_workers = len(self._bundle_indices)
        placement_groups_list = [pg] * n_workers
        placement_bundle_indices_list = [[i] for i in self._bundle_indices]

        llm_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            backend="pytorch",
            tensor_parallel_size=tp_size,
            dtype=precision,
            max_seq_len=trtllm_cfg["max_model_len"],
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
                max_batch_size=trtllm_cfg["max_batch_size"] if "max_batch_size" in trtllm_cfg else 0,
            ),
            sleep_config=SleepConfig(
                restore_modes={
                    ExecutorMemoryType.MODEL_WEIGHTS_MAIN: "NONE",
                    ExecutorMemoryType.KV_CACHE: "NONE",
                }
            )
        )
        if "max_batch_size" in trtllm_cfg:
            llm_kwargs["max_batch_size"] = trtllm_cfg["max_batch_size"]
        if "max_num_tokens" in trtllm_cfg:
            llm_kwargs["max_num_tokens"] = trtllm_cfg["max_num_tokens"]

        # Extract KvCacheConfig-level fields from trtllm_kwargs before
        # spreading the rest as top-level AsyncLLM kwargs.  AsyncLLM validates
        # its kwargs against LlmArgs.model_fields and rejects unknown keys;
        # mamba_ssm_cache_dtype and friends live on KvCacheConfig, not LlmArgs.
        _KV_CACHE_FIELDS = {
            "mamba_ssm_cache_dtype",
            "mamba_ssm_stochastic_rounding",
            "mamba_ssm_philox_rounds",
        }
        extra_trtllm_kwargs = dict(self.cfg.get("trtllm_kwargs") or {})
        kv_cache_kwargs = {k: extra_trtllm_kwargs.pop(k) for k in _KV_CACHE_FIELDS if k in extra_trtllm_kwargs}

        gpu_mem_util = trtllm_cfg.get("gpu_memory_utilization")
        if gpu_mem_util is not None or kv_cache_kwargs:
            llm_kwargs["kv_cache_config"] = KvCacheConfig(
                **({"free_gpu_memory_fraction": gpu_mem_util} if gpu_mem_util is not None else {}),
                **kv_cache_kwargs,
            )

        moe_tp = trtllm_cfg.get("moe_tensor_parallel_size")
        moe_ep = trtllm_cfg.get("moe_expert_parallel_size")
        if moe_tp is not None:
            llm_kwargs["moe_tensor_parallel_size"] = moe_tp
        if moe_ep is not None:
            llm_kwargs["moe_expert_parallel_size"] = moe_ep

        # Colocated: share each bundle's GPU 0.5/0.5 with the policy actor.
        # RayWorkerWrapper does ray.get_gpu_ids()[0], so num_gpus must be > 0.
        if self._colocated:
            llm_kwargs["sleep_config"] = SleepConfig()
            llm_kwargs["per_worker_gpu_share"] = 0.5

        # Escape hatch: spread remaining user-provided TRT-LLM kwargs last so
        # they can override anything above for advanced tuning.
        llm_kwargs.update(extra_trtllm_kwargs)

        # Defer __await__ (which fires setup_async) to post_init_async so
        # AsyncLLM setup runs on the Ray actor's asyncio loop.
        self.llm = AsyncLLM(**llm_kwargs)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def is_alive(self) -> bool:
        return True

    async def post_init_async(self) -> None:
        """Finish async-side engine setup on the Ray actor's asyncio loop."""
        if not self.is_model_owner or self.llm is None:
            return

        print("[TrtllmAsyncWorker] post_init_async: awaiting setup_async…", flush=True)
        await self.llm.setup_async()
        print("[TrtllmAsyncWorker] AsyncLLM ready", flush=True)

    def shutdown(self) -> bool:
        try:
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
            t.value for t in ExecutorMemoryType
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
        # AsyncLLM doesn't expose reset_prefix_cache directly; dispatch via
        # collective_rpc to invoke WorkerExtension.reset_prefix_cache on each
        # Ray worker (which calls PyExecutor.reset_prefix_cache locally).
        await self.llm.collective_rpc("reset_prefix_cache")
        return True

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if len(data["input_ids"]) == 0:
            return BatchedDataDict[GenerationOutputSpec]({
                "output_ids": torch.zeros((0, 0), dtype=torch.long),
                "logprobs": torch.zeros((0, 0), dtype=torch.float),
                "generation_lengths": torch.zeros(0, dtype=torch.long),
                "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
            })

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
                (total_length,), self.cfg["_pad_token_id"], dtype=input_ids.dtype,
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

        return BatchedDataDict[GenerationOutputSpec]({
            "output_ids": torch.stack(output_ids_list),
            "logprobs": torch.stack(logprobs_list),
            "generation_lengths": torch.tensor(generation_lengths, dtype=torch.long),
            "unpadded_sequence_lengths": torch.tensor(
                unpadded_sequence_lengths, dtype=torch.long
            ),
        })

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _resolve_end_id(self) -> Optional[int]:
        """Resolve end_id from model config.json, cached after first call.

        Mirrors vLLM engine which reads eos_token_id from config.json automatically
        at startup. TRT-LLM requires it to be passed explicitly as end_id.
        """
        if hasattr(self, "_end_id_cache"):
            return self._end_id_cache
        end_id: Optional[int] = None
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            eos_id = getattr(hf_config, "eos_token_id", None)
            if eos_id is not None:
                end_id = eos_id[0] if isinstance(eos_id, list) else eos_id
        except Exception as e:
            print(f"[TrtllmAsyncWorker] AutoConfig load failed: {e}", flush=True)
        self._end_id_cache = end_id
        return end_id

    def _build_sampling_params(self, *, greedy: bool):
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else 0)
        temperature = 0.0 if greedy else self.cfg["temperature"]

        end_id = self._resolve_end_id()
        stop_ids = list(self.cfg.get("stop_token_ids") or [])

        return self.TrtSamplingParams(
            temperature=temperature,
            top_p=self.cfg["top_p"],
            top_k=top_k_val,
            max_tokens=self.cfg["max_new_tokens"],
            end_id=end_id,
            stop_token_ids=stop_ids or None,
            # Keep the EOS / stop token in the returned token_ids so that the
            # response sequence matches HF / vLLM behavior. Required for
            # logprob alignment with training-side Megatron.
            include_stop_str_in_output=True,
            logprobs=True,
        )


@ray.remote(
    num_cpus=0,
    runtime_env={**get_nsight_config_if_pattern_matches("trtllm_async_generation_worker")},
)  # pragma: no cover
class TrtllmAsyncGenerationWorker(TrtllmAsyncGenerationWorkerImpl):
    """Ray actor wrapper around :class:`TrtllmAsyncGenerationWorkerImpl`."""

    pass
