import asyncio
import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    get_reordered_bundle_and_gpu_ids,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.redesign.async_utils import run
from nemo_rl.models.generation.redesign.config import SGLangConfig
from nemo_rl.models.generation.redesign.http_utils import (
    init_http_client,
    post,
)
from nemo_rl.models.generation.redesign.misc import (
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST,
    run_router,
)
from nemo_rl.models.generation.redesign.ray_utils import (
    Lock,
    _wrap_ipv6,
    find_available_port,
    get_host_info,
)
from nemo_rl.models.generation.redesign.sglang_worker import SGLangEngine

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_gpus_per_node: int
    num_new_engines: int
    rank_offset: int = 0
    gpu_offset: int = 0
    needs_offload: bool = False
    model_path: str | None = None
    router_ip: str | None = None
    router_port: int | None = None
    cluster_cfg: Any = None
    sglang_cfg: Any = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [self.num_gpus_per_engine for _ in self.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        offsets = []
        for j in range(len(self.engines)):
            offsets.append(self.gpu_offset + j * self.num_gpus_per_engine)
        return offsets

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index -> next free port.
        """
        if port_cursors is None:
            port_cursors = {}

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.num_gpus_per_node)
        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
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

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.cluster_cfg,
                self.sglang_cfg,
                rank=global_rank,
                base_gpu_id=base_gpu_id,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        base_port = max(port_cursors.values()) if port_cursors else 15000
        addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
            cluster_cfg=self.cluster_cfg,
            sglang_cfg=self.sglang_cfg,
            rollout_engines=rollout_engines,
            rank_offset=self.rank_offset,
            base_port=base_port,
        )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors
    
    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        dead_indices = [i for i, engine in enumerate(self.all_engines) if engine is None]

        port_cursors: dict[int, int] = {}
        handles = self.start_engines(port_cursors)
        if handles:
            ray.get(handles)

        release_handles = []
        updatable_new_engines = []

        assert self.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
        if self.needs_offload and dead_indices:
            new_engines = [self.all_engines[i] for i in dead_indices]
            release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
            updatable_new_engines.extend(new_engines)

        if release_handles:
            ray.get(release_handles)
            all_resume_engines = updatable_new_engines[:]
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights(self):
        if not self.needs_offload:
            return
        handles = self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        return ray.get(handles) if handles else []

    def onload_kv(self):
        handles = self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
        return ray.get(handles) if handles else []
    
    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups."""
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]
    

class SGLangGeneration(GenerationInterface):
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, cluster: RayVirtualCluster, cluster_cfg: ClusterConfig, sglang_cfg: SGLangConfig):
        self.cluster = cluster
        self.cluster_cfg = cluster_cfg
        self.sglang_cfg = sglang_cfg
        self.pg = cluster._init_placement_groups(
            strategy="PACK",
            use_unified_pg=True,
        )
        self.pg_reordered_bundle_indices, self.pg_reordered_gpu_ids = get_reordered_bundle_and_gpu_ids(self.pg)

        init_http_client(sglang_cfg)
        self.server_group = start_rollout_servers(
            sglang_cfg,
            cluster_cfg,
            (self.pg, self.pg_reordered_bundle_indices, self.pg_reordered_gpu_ids),
        )

        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for e in self.server_group.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        server_group = self.server_group
        engines = server_group.engines if server_group else []
        gpu_counts = server_group.engine_gpu_counts if server_group else []
        gpu_offsets = server_group.engine_gpu_offsets if server_group else []
        num_new = server_group.num_new_engines if server_group else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def offload(self, tags: list[str] | None = None):
        if tags is not None:
            handles = [
                engine.release_memory_occupation.remote(tags=tags)
                for engine in self.rollout_engines
                if engine is not None
            ]
            return ray.get(handles) if handles else []
        else:
            handles = self.server_group.offload()
            return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        handles = self.server_group.onload(tags)
        return ray.get(handles) if handles else []

    def onload_weights(self):
        self.server_group.onload_weights()

    def onload_kv(self):
        self.server_group.onload_kv()

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        server_group = self.server_group

        if server_group is None:
            engines = server_group.engines if server_group else []
            gpu_counts = server_group.engine_gpu_counts if server_group else []
            gpu_offsets = server_group.engine_gpu_offsets if server_group else []
            return engines, self.rollout_engine_lock, (server_group.num_new_engines if server_group else 0), gpu_counts, gpu_offsets

        server_group.recover()
        return (
            server_group.engines,
            self.rollout_engine_lock,
            server_group.num_new_engines,
            server_group.engine_gpu_counts,
            server_group.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        if self.server_group:
            self.server_group.num_new_engines = 0

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

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
                against ``context_length - input_length``).
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


    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM generation.

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
                    self.sglang_cfg["max_new_tokens"], context_length - input_length
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
        sglang_server_cfg = self.sglang_cfg["sglang_server"]
        max_concurrency = (
            sglang_server_cfg["sglang_server_concurrency"]
            * sglang_server_cfg["num_gpus"]
            // sglang_server_cfg["num_gpus_per_engine"]
        )

        router_ip = self.sglang_cfg["sglang_router"]["sglang_router_ip"]
        router_port = self.sglang_cfg["sglang_router"]["sglang_router_port"]

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_generate_one_sample(
            idx: int, sp: dict[str, Any], ids: list[int]
        ):
            async with semaphore:
                return await generate_one_sample(
                    router_ip, router_port, sp, ids, idx
                )

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
            run(_dispatch_all()) if sample_requests else {}
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
                self.sglang_cfg["max_new_tokens"], context_length - input_length
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
                    "truncated": torch.tensor(
                        [False], dtype=torch.bool, device=device
                    ),
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

        router_ip = self.sglang_cfg["sglang_router"]["sglang_router_ip"]
        router_port = self.sglang_cfg["sglang_router"]["sglang_router_port"]
        valid_input_ids = original_input_ids_single_row[:input_length].tolist()

        # batch_size == 1, so no task fan-out / as_completed is needed. Just
        # await the single coroutine directly.
        _, new_tokens, new_logprobs, is_truncated = await generate_one_sample(
            router_ip,
            router_port,
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
        if new_tokens:
            output_ids_single_item[input_length:unpadded_length] = torch.tensor(
                new_tokens, dtype=dtype, device=device
            )

        # Logprobs: zeros for input tokens, raw floats at generated positions.
        logprobs_single_item = torch.zeros(
            (1, unpadded_length), dtype=torch.float32, device=device
        )
        if new_logprobs:
            logprobs_single_item[
                0, input_length : input_length + len(new_logprobs)
            ] = torch.tensor(new_logprobs, dtype=torch.float32, device=device)

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
# Generate one sample helper
# ---------------------------------------------------------------------------
async def generate_one_sample(sglang_router_ip, sglang_router_port, sampling_params, input_ids, index: int):
    """Generate using traditional SGLang router with token-based workflow"""
    url = f"http://{sglang_router_ip}:{sglang_router_port}/generate"

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
        "input_ids": input_ids,
    }

    output = await post(url, payload)

    if "output_token_logprobs" in output["meta_info"]:
        response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        response_tokens, response_log_probs = [], []

    response_truncated = False
    if output["meta_info"]["output_token_logprobs"] is not None and output["meta_info"]["output_token_logprobs"] == "length":
        response_truncated = True

    return index, response_tokens, response_log_probs, response_truncated



# ---------------------------------------------------------------------------
# Port allocation helpers
# ---------------------------------------------------------------------------
def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    cluster_cfg,
    sglang_cfg,
    rollout_engines,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size

    sglang_dp_size = sglang_cfg["sglang_cfg"]["dp_size"]
    num_gpus_per_engine = sglang_cfg["sglang_server"]["num_gpus_per_engine"]
    num_gpus_per_node = cluster_cfg["gpus_per_node"]

    _gpus_per_engine = num_gpus_per_engine 
    num_engines_per_node = max(1, num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different server groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

        if _gpus_per_engine > num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                dist_init_addr = f"{get_addr()}:{get_port(30 + sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor

# ---------------------------------------------------------------------------
# Router + server bootstrap
# ---------------------------------------------------------------------------

def _start_router(args: SGLangConfig) -> tuple[str, int]:
    """Start sgl router return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set and ``force_new`` is False,
    skip launching and return the existing values.
    """
    if args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    router_port = args.sglang_router_port
    if router_port is None:
        router_port = find_available_port(random.randint(3000, 4000))

    from sglang_router.launch_router import RouterArgs

    # pass from 
    router_args = RouterArgs()
    router_args.host = router_ip
    router_args.port = router_port
    if args["sglang_router"]["router_policy"] is not None:
        router_args.router_policy = args["sglang_router"]["router_policy"]
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port

def start_rollout_servers(sglang_cfg, cluster_cfg, pg) -> ServerGroup:
    """Start rollout servers: one per model, each with its own router.

    Returns a dict mapping model name -> ``RolloutServer``.
    """

    engine_offset = 0
    gpu_offset = 0

    router_ip, router_port = _start_router(sglang_cfg)

    sglang_cfg["sglang_router"]["sglang_router_ip"] = router_ip
    sglang_cfg["sglang_router"]["sglang_router_port"] = router_port

    all_init_handles: list = []
    port_cursors: dict[int, int] = {}

    gpus_per_engine = sglang_cfg["sglang_server"]["num_gpus_per_engine"]
    num_gpu_per_engine_local = min(gpus_per_engine, cluster_cfg["gpus_per_node"])
    num_engines = sglang_cfg["sglang_server"]["num_gpus"] // num_gpu_per_engine_local
    needs_offload = sglang_cfg["sglang_server"]["needs_offload"]
    num_gpus_per_node = cluster_cfg["gpus_per_node"]
    model_path= sglang_cfg["sglang_cfg"]["model_path"]

    server_group = ServerGroup(
        pg=pg,
        all_engines=[None] * num_engines,
        num_gpus_per_engine=gpus_per_engine,
        num_gpus_per_node=num_gpus_per_node,
        num_new_engines=0,
        rank_offset=engine_offset,
        gpu_offset=gpu_offset,
        needs_offload=needs_offload,
        model_path=model_path,
        router_ip=router_ip,
        router_port=router_port,
        cluster_cfg=cluster_cfg,
        sglang_cfg=sglang_cfg,
    )

    handles, port_cursors = server_group.start_engines(port_cursors)
    all_init_handles.extend(handles)

    if all_init_handles:
        ray.get(all_init_handles)

    return server_group
