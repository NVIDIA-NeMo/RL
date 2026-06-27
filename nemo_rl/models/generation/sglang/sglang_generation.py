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

import asyncio
import logging
import os
from typing import Any, AsyncGenerator

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    get_reordered_bundle,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_router import _start_router
from nemo_rl.models.generation.sglang.sglang_worker import SGLangGenerationWorker
from nemo_rl.models.generation.sglang.utils.async_utils import AsyncLoopThread
from nemo_rl.models.generation.sglang.utils.http_utils import HttpClient
from nemo_rl.models.generation.sglang.utils.ip_port_utils import (
    _allocate_rollout_engine_addr_and_ports_normal,
)
from nemo_rl.models.generation.sglang.utils.ray_utils import (
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.venvs import make_actor_runtime_env

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class SGLangGeneration(GenerationInterface):
    """The class to run rollout and convert rollout data to training data.

    This class owns the full rollout server topology: the placement group,
    the router subprocess, and every ``SGLangGenerationWorker`` Ray actor.
    The former ``ServerGroup`` dataclass has been folded in so there is a
    single source of truth for engine state.

    TODO: one sglang router(router ip, router port) --> different server group(eg: PD, different tp size, ...; each server group multiple engines/servsers with same settings)
    router + [[p, ..., p] [d, ..., d]] or router + [[tp = 2, ... tp = 2], ..., [tp = 8, ..., tp = 8]]
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        sglang_cfg: SGLangConfig,
    ):
        self.cluster = cluster
        self.sglang_cfg = sglang_cfg
        self._async_loop: AsyncLoopThread | None = AsyncLoopThread()
        self._http_client: HttpClient | None = None

        pgs = cluster._init_placement_groups(
            strategy="PACK",
            use_unified_pg=True,
        )
        self.pg = pgs[0]
        self.pg_reordered_bundle_indices, self.pg_reordered_gpu_ids, _ = (
            get_reordered_bundle(self.pg)
        )
        self._http_client = HttpClient(sglang_cfg)

        # --- Engine topology (formerly ``ServerGroup``) ------------------
        sglang_server_cfg = sglang_cfg["sglang_cfg"]["sglang_server_config"]
        gpus_per_engine = sglang_server_cfg["num_gpus_per_engine"]
        num_gpus_per_node = cluster.num_gpus_per_node
        num_gpu_per_engine_local = min(gpus_per_engine, num_gpus_per_node)
        num_engines = sglang_server_cfg["num_gpus"] // num_gpu_per_engine_local

        self.num_gpus_per_engine: int = gpus_per_engine
        self.num_gpus_per_node: int = num_gpus_per_node
        self.all_engines: list = [None] * num_engines
        # It will be useful for future features which involve pd disaggregation, mixture sglang config setup
        self.rank_offset: int = 0
        self.gpu_offset: int = 0
        self.needs_offload: bool = sglang_server_cfg["needs_offload"]
        self.model_path: str | None = sglang_cfg["sglang_cfg"]["model_path"]

        # --- Router bootstrap --------------------------------------------
        # Resolved router endpoint is held only on the instance; we don't
        # mutate the caller's config dict. Workers receive these as explicit
        # ``router_ip`` / ``router_port`` kwargs in ``init.remote(...)``.
        router_ip, router_port, router_actor = _start_router(
            sglang_cfg["sglang_cfg"].get("sglang_router_config") or {}
        )
        self.router_ip: str = router_ip
        self.router_port: int = router_port
        # Only set when ``_start_router`` actually spawned the router (i.e.
        # sglang_router_ip was not already configured). Kept so ``shutdown``
        # can terminate it cleanly.
        self._router_actor: ray.actor.ActorHandle | None = router_actor

        # --- Start engines -----------------------------------------------
        init_handles, _ = self._start_engines({})
        if init_handles:
            ray.get(init_handles)

    # ------------------------------------------------------------------
    # Engine topology properties (formerly ``ServerGroup``)
    # ------------------------------------------------------------------
    @property
    def nodes_per_engine(self) -> int:
        return max(1, self.num_gpus_per_engine // self.num_gpus_per_node)

    @property
    def engines(self) -> list:
        """Node-0 engines only (one entry per logical engine)."""
        return self.all_engines[:: self.nodes_per_engine]

    @property
    def rollout_engines(self) -> list:
        """Alias for ``engines`` — node-0 engines across all servers / models."""
        return self.engines

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count, parallel to ``engines``."""
        return [self.num_gpus_per_engine for _ in self.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [
            self.gpu_offset + j * self.num_gpus_per_engine
            for j in range(len(self.engines))
        ]

    def get_rollout_engine_urls(self) -> list[str]:
        """Resolve node-0 engine HTTP base URLs once on the driver."""
        return ray.get([e.get_base_url.remote() for e in self.rollout_engines])

    def _start_engines(
        self, port_cursors: dict[int, int] | None = None
    ) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index -> next free port.
        """
        if port_cursors is None:
            port_cursors = {}

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.num_gpus_per_node)
        pg = self.pg
        reordered_bundle_indices = self.pg_reordered_bundle_indices
        reordered_gpu_ids = self.pg_reordered_gpu_ids

        # Resolve the SGLang venv ONCE (mirrors the once-per-node RayWorkerBuilder path).
        sglang_runtime_env_base = make_actor_runtime_env(
            "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
        )
        sglang_runtime_env_base.update(
            get_nsight_config_if_pattern_matches("sglang_generation_worker")
        )

        local_all_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = min(0.2, 1 / self.cluster.max_colocated_worker_groups)
            num_cpus = num_gpus

            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                }.items()
            }

            # Explicitly pass CUDA_VISIBLE_DEVICES through to the engine actor so
            # all engines see the same global value (Ray would otherwise remap it
            # because we set the NOSET_* flags above).
            global_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if global_cvd:
                env_vars["CUDA_VISIBLE_DEVICES"] = global_cvd

            # Resolve the SGLang venv on every node and pin VIRTUAL_ENV /
            # UV_PROJECT_ENVIRONMENT so the actor (and its spawned children)
            # actually run inside the sglang venv instead of inheriting the
            # raylet's base venv (e.g. /opt/nemo_rl_venv inside the container).
            sglang_runtime_env = {
                **sglang_runtime_env_base,
                "env_vars": {**sglang_runtime_env_base["env_vars"], **env_vars},
            }

            actor_options = {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
                "scheduling_strategy": scheduling_strategy,
                "runtime_env": sglang_runtime_env,
            }
            init_args = (self.num_gpus_per_node, self.sglang_cfg)
            init_kwargs = {
                "rank": global_rank,
                "base_gpu_id": base_gpu_id,
                "num_gpus_per_engine": self.num_gpus_per_engine,
            }

            # Create worker actor directly — sglang_worker.py uses lazy imports
            # so it's importable in SYSTEM env; the actor runs in sglang env.
            engine = SGLangGenerationWorker.options(**actor_options).remote(
                *init_args, **init_kwargs
            )

            local_all_engines.append((global_rank, engine))
            self.all_engines[i] = engine

        if len(local_all_engines) == 0:
            return [], port_cursors

        base_port = max(port_cursors.values()) if port_cursors else 15000
        addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
            gpus_per_node=self.num_gpus_per_node,
            sglang_cfg=self.sglang_cfg,
            local_all_engines=local_all_engines,
            rank_offset=self.rank_offset,
            base_port=base_port,
        )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in local_all_engines
        ]
        return init_handles, port_cursors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_weights(self, action: str):
        """All node-0 engines across all servers / models."""
        return ray.get(
            [
                engine.check_weights.remote(action=action)
                for engine in self.engines
                if engine is not None
            ]
        )

    def shutdown(self) -> bool:
        ok = True
        engines = [e for e in self.all_engines if e is not None]
        if engines:
            try:
                ray.get([e.shutdown.remote() for e in engines])
            except Exception as e:
                logger.warning(f"Engine shutdown failed: {e}")
                ok = False
            self.all_engines = [None] * len(self.all_engines)

        if self._router_actor is not None:
            try:
                ray.get(self._router_actor.stop.remote())
                ray.kill(self._router_actor)
            except Exception as e:
                logger.warning(f"Router terminate failed: {e}")
                ok = False
            self._router_actor = None

        if self._http_client is not None:
            try:
                if self._async_loop is not None:
                    self._async_loop.run(self._http_client.aclose())
                else:
                    self._http_client.shutdown()
            except Exception as e:
                logger.warning(f"HTTP client shutdown failed: {e}")
                ok = False
            self._http_client = None

        if self._async_loop is not None:
            try:
                self._async_loop.close()
            except Exception as e:
                logger.warning(f"AsyncLoopThread close failed: {e}")
                ok = False
            self._async_loop = None

        return ok

    def __del__(self) -> None:
        self.shutdown()

    def _merge_stop_strings(self, batch_stop_strings) -> list[list[str]]:
        """Merge stop strings from config and batch.

        Args:
            batch_stop_strings: List of stop strings from batch (one per sample)

        Returns:
            List of merged stop strings (one per sample)
        """
        stop_set: set[str] = set()

        # Add stop strings from config
        if self.sglang_cfg.get("stop_strings"):
            stop_set.update(self.sglang_cfg["stop_strings"])

        # Merge stop strings from batch
        merged_stop_strings = []
        for sample_ss in batch_stop_strings:
            sample_stop_set = stop_set.copy()
            if sample_ss:
                if isinstance(sample_ss, str):
                    sample_stop_set.add(sample_ss)
                elif isinstance(sample_ss, list):
                    sample_stop_set.update(sample_ss)

            merged_stop_strings.append(list(sample_stop_set))

        return merged_stop_strings

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        max_new_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build sampling parameters dictionary for SGLang API.

        Args:
            greedy: Whether to use greedy decoding (temperature=0.0)
            max_new_tokens: Max new tokens for this sample (already clamped by caller
                against ``context_length - input_length - 1``).
            stop_strings: Merged stop strings for this sample.

        Returns:
            Dictionary of sampling parameters compatible with SGLang API.
        """
        temperature = 0.0 if greedy else self.sglang_cfg["temperature"]
        top_k_cfg = self.sglang_cfg.get("top_k")
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        # Build sampling params dict first, then patch in optional fields so we
        # never reference ``sampling_params`` before it's bound.
        sampling_params: dict[str, Any] = {
            "temperature": temperature,
            "top_p": self.sglang_cfg.get("top_p", 1.0),
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

        if top_k_val != -1:
            sampling_params["top_k"] = top_k_val

        stop_token_ids = self.sglang_cfg.get("stop_token_ids")
        if stop_token_ids is not None:
            sampling_params["stop_token_ids"] = stop_token_ids

        if stop_strings is not None and len(stop_strings) > 0:
            sampling_params["stop"] = stop_strings

        return sampling_params

    @wrap_with_nvtx_name("sglang_genertion/generate")
    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using Sglang generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        # Handle empty input case
        if len(data["input_ids"]) == 0:
            # Return empty BatchedDataDict with all required fields
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": torch.zeros((0, 0), dtype=torch.long),
                    "logprobs": torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths": torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
                    "truncated": torch.zeros(0, dtype=torch.bool),
                }
            )

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)

        batch_size = len(input_lengths)
        padded_input_length = input_ids.size(1)
        context_length = self.sglang_cfg["sglang_cfg"]["context_length"]

        # verify inputs have correct padding
        verify_right_padding(data, pad_value=self.sglang_cfg["_pad_token_id"])

        # Build per-sample requests (each sample gets its own sampling params because
        # max_new_tokens is adjusted against the per-sample input length).
        sample_requests: list[tuple[int, dict[str, Any], list[int]]] = []
        skip_results: set[int] = set()
        skip_max_length = 0
        for i in range(batch_size):
            input_length = input_lengths[i].item()
            valid_input_ids = input_ids[i, :input_length].tolist()

            if context_length is not None:
                max_new_tokens = min(
                    self.sglang_cfg["max_new_tokens"],
                    context_length - input_length - 1,
                )
            else:
                max_new_tokens = self.sglang_cfg["max_new_tokens"]
            max_new_tokens = max(0, max_new_tokens)

            if max_new_tokens == 0:
                skip_results.add(i)
                skip_max_length = max(skip_max_length, input_length)
                continue

            sample_sampling_params = self._build_sampling_params(
                greedy=greedy,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings[i] if i < len(stop_strings) else None,
            )
            sample_requests.append((i, sample_sampling_params, valid_input_ids))

        # Dispatch concurrently to the SGLang router with bounded concurrency.
        # Max concurrency = per-engine concurrency * number of engines.
        sglang_server_cfg = self.sglang_cfg["sglang_cfg"]["sglang_server_config"]
        max_concurrency = (
            sglang_server_cfg["sglang_server_concurrency"]
            * sglang_server_cfg["num_gpus"]
            // sglang_server_cfg["num_gpus_per_engine"]
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_generate_one_sample(
            idx: int, sp: dict[str, Any], ids: list[int]
        ):
            async with semaphore:
                return await self.generate_one_sample(sp, ids, idx)

        async def _dispatch_all() -> dict[int, tuple[list[int], list[float], bool]]:
            gathered = await asyncio.gather(
                *(
                    _bounded_generate_one_sample(idx, sp, ids)
                    for idx, sp, ids in sample_requests
                )
            )
            # generate_one_sample returns (index, tokens, logprobs, truncated).
            # Re-key by the original sample index so downstream code can look up
            # results directly without sorting.
            return {
                returned_idx: (new_tokens, new_logprobs, is_truncated)
                for returned_idx, new_tokens, new_logprobs, is_truncated in gathered
            }

        router_results: dict[int, tuple[list[int], list[float], bool]] = (
            self._async_loop.run(_dispatch_all()) if sample_requests else {}
        )

        # Process the outputs - preserve the original input padding structure.
        pad_token_id = self.sglang_cfg["_pad_token_id"]
        output_ids_list: list[torch.Tensor] = []
        logprobs_list: list[torch.Tensor] = []
        generation_lengths_list: list[int] = []
        unpadded_sequence_lengths_list: list[int] = []
        truncated_list: list[bool] = []

        # First pass: compute total_length as the max over all samples of
        # (input_length + generation_length). Skipped samples contribute only
        # their input_length (already tracked in ``skip_max_length``).
        max_length = skip_max_length
        for returned_idx, (returned_tokens, _, _) in router_results.items():
            sample_input_length = input_lengths[returned_idx].item()
            max_length = max(max_length, sample_input_length + len(returned_tokens))
        total_length = max(max_length, padded_input_length)

        # Second pass: materialize the output tensors, using a single set of
        # local variable names (``generation_length`` / ``unpadded_length`` are
        # always Python ints; tensor promotion happens only at the final stack).
        for i in range(batch_size):
            input_length = input_lengths[i].item()
            full_output = torch.full(
                (total_length,), pad_token_id, dtype=input_ids.dtype
            )
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            full_output[:input_length] = input_ids[i][:input_length]

            if i in skip_results:
                generation_length = 0
                is_truncated = False
            else:
                new_tokens, new_logprobs, is_truncated = router_results[i]
                generation_length = len(new_tokens)
                if new_tokens:
                    full_output[input_length : input_length + generation_length] = (
                        torch.tensor(new_tokens, dtype=input_ids.dtype)
                    )
                if new_logprobs:
                    full_logprobs[input_length : input_length + len(new_logprobs)] = (
                        torch.tensor(new_logprobs, dtype=torch.float32)
                    )

            unpadded_length = input_length + generation_length
            output_ids_list.append(full_output)
            logprobs_list.append(full_logprobs)
            generation_lengths_list.append(generation_length)
            unpadded_sequence_lengths_list.append(unpadded_length)
            truncated_list.append(bool(is_truncated))

        return_data = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.stack(output_ids_list),
                "logprobs": torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(
                    generation_lengths_list, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths_list, dtype=torch.long
                ),
                "truncated": torch.tensor(truncated_list, dtype=torch.bool),
            }
        )

        return return_data

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a single sample using SGLang, yielding the result when ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths (batch_size must be 1)
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec)
        """
        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.sglang_cfg["_pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]

        # Restrict to single-sample batches, matching the vLLM async contract.
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        sample_idx = 0
        input_length = input_lengths_batch[sample_idx].item()
        original_input_ids_single_row = input_ids_batch[sample_idx]
        device = original_input_ids_single_row.device
        dtype = original_input_ids_single_row.dtype
        pad_token_id = self.sglang_cfg["_pad_token_id"]

        # Clamp max_new_tokens against the per-sample remaining context window,
        # mirroring the logic in ``generate``.
        context_length = self.sglang_cfg["sglang_cfg"].get("context_length")
        if context_length is not None:
            max_new_tokens = min(
                self.sglang_cfg["max_new_tokens"],
                context_length - input_length - 1,
            )
        else:
            max_new_tokens = self.sglang_cfg["max_new_tokens"]
        max_new_tokens = max(0, max_new_tokens)

        # Short-circuit when there is no room left in the context window. Yield
        # a pure-input row (generation_length=0, truncated=False) without
        # touching the SGLang router.
        if max_new_tokens == 0:
            output_ids_single_item_batched = original_input_ids_single_row[
                :input_length
            ].unsqueeze(0)
            logprobs_single_item = torch.zeros(
                (1, input_length), dtype=torch.float32, device=device
            )
            empty_result = BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": output_ids_single_item_batched,
                    "logprobs": logprobs_single_item,
                    "generation_lengths": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "unpadded_sequence_lengths": torch.tensor(
                        [input_length], dtype=torch.long, device=device
                    ),
                    "truncated": torch.tensor([False], dtype=torch.bool, device=device),
                }
            )
            yield (sample_idx, empty_result)
            return

        # Merge stop strings for this single sample.
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)
        per_sample_stop_strings = (
            stop_strings[sample_idx] if sample_idx < len(stop_strings) else None
        )

        sampling_params = self._build_sampling_params(
            greedy=greedy,
            max_new_tokens=max_new_tokens,
            stop_strings=per_sample_stop_strings,
        )

        valid_input_ids = original_input_ids_single_row[:input_length].tolist()

        # batch_size == 1, so no task fan-out / as_completed is needed. Just
        # await the single coroutine directly.
        _, new_tokens, new_logprobs, is_truncated = await self.generate_one_sample(
            sampling_params,
            valid_input_ids,
            sample_idx,
        )

        # Build the single-sample output tensor: [input | generated].
        generation_length = len(new_tokens)
        unpadded_length = input_length + generation_length

        output_ids_single_item = torch.full(
            (unpadded_length,), pad_token_id, dtype=dtype, device=device
        )
        output_ids_single_item[:input_length] = original_input_ids_single_row[
            :input_length
        ]
        # Logprobs: zeros for input tokens, raw floats at generated positions.
        logprobs_single_item = torch.zeros(
            (1, unpadded_length), dtype=torch.float32, device=device
        )

        if new_tokens:
            output_ids_single_item[input_length:unpadded_length] = torch.tensor(
                new_tokens, dtype=dtype, device=device
            )
        if new_logprobs:
            logprobs_single_item[0, input_length : input_length + len(new_logprobs)] = (
                torch.tensor(new_logprobs, dtype=torch.float32, device=device)
            )

        result_batch = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids_single_item.unsqueeze(0),
                "logprobs": logprobs_single_item,
                "generation_lengths": torch.tensor(
                    [generation_length], dtype=torch.long, device=device
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    [unpadded_length], dtype=torch.long, device=device
                ),
                "truncated": torch.tensor(
                    [bool(is_truncated)], dtype=torch.bool, device=device
                ),
            }
        )

        yield (sample_idx, result_batch)

    # ---------------------------------------------------------------------------
    # Compatible with parent class or old interfaces
    # ---------------------------------------------------------------------------
    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        return []

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        pass

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        return []

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        return []

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake workers up for colocated inference."""
        if not self.needs_offload:
            return
        tags = kwargs.get("tags", None)
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        ray.get([e.resume_memory_occupation.remote(tags=tags) for e in engines])

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers and reset prefix cache."""
        if not self.needs_offload:
            return
        tags = kwargs.get("tags", None)
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        ray.get([e.release_memory_occupation.remote(tags=tags) for e in engines])

    def invalidate_kv_cache(self) -> bool:
        """Invalidate KV cache before weight updates (Megatron-style).

        Flushes the cache on every node-0 engine so stale KV entries are
        discarded before new weights land. Returns ``True`` iff every engine
        reports success.
        """
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return True
        try:
            ray.get([e.invalidate_kv_cache.remote() for e in engines])
        except Exception as e:
            logger.error(f"[sglang refit] Error flushing SGLang caches: {e}")
            return False

        logger.info("[sglang refit] All SGLang server caches flushed successfully")
        return True

    # ---------------------------------------------------------------------------
    # Generate one sample helper
    # ---------------------------------------------------------------------------
    async def generate_one_sample(
        self,
        sampling_params,
        input_ids,
        index: int,
    ):
        """Generate using traditional SGLang router with token-based workflow."""
        url = f"http://{self.router_ip}:{self.router_port}/generate"

        # Prepare payload for sglang server
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
            "input_ids": input_ids,
        }

        output = await self._http_client.post(url, payload)

        if "output_token_logprobs" in output["meta_info"]:
            response_tokens = [
                item[1] for item in output["meta_info"]["output_token_logprobs"]
            ]
            response_log_probs = [
                item[0] for item in output["meta_info"]["output_token_logprobs"]
            ]
        else:
            response_tokens, response_log_probs = [], []

        # SGLang reports the termination reason under meta_info.finish_reason.type;
        # "length" means the decoder hit max_new_tokens before EOS.
        finish_reason = output["meta_info"].get("finish_reason") or {}
        response_truncated = finish_reason.get("type") == "length"

        return index, response_tokens, response_log_probs, response_truncated
