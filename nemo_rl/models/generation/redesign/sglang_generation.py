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
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from miles.backends.sglang_utils.sglang_engine import SGLangEngine
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils import tracking_utils
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.health_monitor import RolloutHealthMonitor
from miles.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from miles.utils.iter_utils import group_by
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from miles.utils.misc import load_function
from miles.utils.ray_utils import Box
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ServerGroup / RolloutServer abstractions
# ---------------------------------------------------------------------------

# use_unified_pg = True for Nemo
#         if use_unified_pg:
#             # Create a single unified placement group for cross-node model parallelism
#             all_bundles = []
#             for bundle_count in self._bundle_ct_per_node_list:
#                 for _ in range(bundle_count):
#                     all_bundles.append(
#                         {"CPU": num_cpus_per_bundle, "GPU": num_gpus_per_bundle}
#                     )

#             placement_groups = [
#                 placement_group(
#                     bundles=all_bundles, strategy=strategy, name=f"{self.name}-unified"
#                 )
#             ]

#         pg = cluster._init_placement_groups(strategy="PACK", use_unified_pg=True)[0]
#         pg_reordered_bundle_indices   = cluster._get_sorted_bundle_indices()


class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

# Create one global instance
async_loop = None

def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop

def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)


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
    if response_truncated["meta_info"]["output_token_logprobs"] is not None and output["meta_info"]["output_token_logprobs"] == "length":
        response_truncated = True

    return index, response_tokens, response_log_probs, response_truncated


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

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]
    
    @num_new_engines.setter
    def num_new_engines(self, value):
        self.num_new_engines = value

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
                self.args,
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
            rollout_engines=rollout_engines,
            num_gpus_per_engine=self.num_gpus_per_engine,
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
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
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

# ---------------------------------------------------------------------------
# SGLangGeneration
# ---------------------------------------------------------------------------

class SGLangGeneration(GenerationInterface):
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, cluster: RayVirtualCluster, cluster_cfg: ClusterConfig, sglang_cfg: SGLangConfig):
        configure_logger()

        self.cluster = cluster
        self.cluster_cfg = cluster_cfg
        self.sglang_cfg = sglang_cfg
        self.pg = cluster._init_placement_groups(
            strategy="PACK",
            use_unified_pg=True,
        )
        self.pg_reordered_bundle_indices, self.pg_reordered_gpu_ids = get_reordered_bundle_and_gpu_ids(self.pg)

        init_http_client(args)
        self.server_group = start_rollout_servers(args, (self.pg, self.pg_reordered_bundle_indices, self.pg_reordered_gpu_ids))

        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        
    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()

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
        if self.rollout_id == -1 or server_group is None:
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
        stop_strings: Optinoal[list[str]] = None,
        input_len: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build sampling parameters dictionary for SGLang API.

        Args:
            greedy: Whether to use greedy decoding (temperature=0.0)
            stop_strings: Merged stop strings (not used here, handled per sample)
            max_new_tokens: Override max_new_tokens from config if provided
            input_len: Input length for this sample (used for context_length adjustment)

        Returns:
            Dictionary of sampling parameters compatible with SGLang API
        """
        top_k_cfg = self.sglang_cfg.get("top_k")
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)
        if top_k_val != -1:
            sampling_params["top_k"] = top_k_val
        temperature = 0.0 if greedy else self.sglang_cfg["temperature"]

        # Build sampling params dict
        sampling_params = {
            "temperature": temperature,
            "top_p": self.sglang_cfg.get("top_p", 1.0),
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

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
        skip_results = []
        for i in range(batch_size):
            input_len = input_lengths[i].item()
            valid_input_ids = input_ids[i, :input_len].tolist()

            if context_length is not None:
                max_new_tokens = min(self.sglang_cfg["max_new_tokens"], context_length - input_len)
            else:
                max_new_tokens = self.sglang_cfg["max_new_tokens"]
            max_new_tokens = max(0, max_new_tokens)

            if max_new_tokens == 0:
                skip_results.append(i)
            else: 
                sample_sampling_params = self._build_sampling_params(
                    greedy=greedy,
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings[i] if i < len(stop_strings) else None,
                    input_len=input_len,
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

        async def _dispatch_all():
            return await asyncio.gather(
                *(
                    _bounded_generate_one_sample(idx, sp, ids)
                    for idx, sp, ids in sample_requests
                )
            )
            
        router_results = run(_dispatch_all())
        
        # Process the outputs - preserve the original input padding structure.
        pad_token_id = self.sglang_cfg["_pad_token_id"]
        output_ids_list: list[torch.Tensor] = []
        logprobs_list: list[torch.Tensor] = []
        generation_lengths_list: list[int] = []
        unpadded_sequence_lengths_list: list[int] = []
        truncated_list: list[bool] = []

        # First pass: compute max unpadded (input + generation) length across the batch
        # so every sample can be right-padded to the same width.
        max_length = 0
        for i, (_, new_tokens, _, _) in enumerate(router_results):
            input_len = input_lengths[i].item()
            max_length = max(max_length, input_len + len(new_tokens))
        total_length = max(max_length, padded_input_length)
        if len(skip_results):
            total_length = max(total_length, context_length)

        for i in range(batch_size):
            input_len = input_lengths[i].item()
            full_output = torch.full(
                (total_length,), pad_token_id, dtype=input_ids.dtype
            )
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if i in skip_results:

            else:


        for i, (_, new_tokens, new_logprobs, is_truncated) in enumerate(router_results):
            input_len = input_lengths[i].item()
            generation_length = len(new_tokens)
            unpadded_length = input_len + generation_length

            # Build full output: [input tokens | generated tokens | pad]
            full_output = torch.full(
                (total_length,), pad_token_id, dtype=input_ids.dtype
            )
            full_output[:input_len] = input_ids[i][:input_len]
            if new_tokens:
                full_output[input_len : input_len + generation_length] = torch.tensor(
                    new_tokens, dtype=input_ids.dtype
                )

            # Logprobs: zeros for input tokens, actual logprobs at generated positions.
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if new_logprobs:
                for idx, logprob in enumerate(new_logprobs):
                    full_logprobs[input_len + idx] = logprob

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

        This mirrors ``VllmGenerationWorker.generate_async``: it is restricted to
        single-sample batches (``batch_size == 1``), so the surrounding
        ``asyncio.create_task(...) / asyncio.as_completed(...)`` fan-out used in
        the vLLM version is unnecessary here — we simply ``await`` the single
        ``generate_one_sample`` call directly and yield its output.

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
        input_len = input_lengths_batch[sample_idx].item()
        valid_input_ids = input_ids_batch[sample_idx, :input_len].tolist()

        # Merge stop strings for this single sample.
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)
        per_sample_stop_strings = (
            stop_strings[sample_idx] if sample_idx < len(stop_strings) else None
        )

        context_length = self.sglang_cfg["sglang_cfg"].get("context_length")
        remaining_ctx = (
            (context_length - input_len) if context_length is not None else None
        )

        # Short-circuit when the prompt already fills the context window.
        if remaining_ctx is not None and remaining_ctx <= 0:
            device = input_ids_batch.device
            output_ids_single_item_batched = input_ids_batch[
                sample_idx, :input_len
            ].unsqueeze(0)
            logprobs_single_item = torch.zeros(
                (1, input_len), dtype=torch.float32, device=device
            )
            empty_result = BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": output_ids_single_item_batched,
                    "logprobs": logprobs_single_item,
                    "generation_lengths": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "unpadded_sequence_lengths": torch.tensor(
                        [input_len], dtype=torch.long, device=device
                    ),
                    "truncated": torch.tensor(
                        [False], dtype=torch.bool, device=device
                    ),
                }
            )
            yield (sample_idx, empty_result)
            return

        sampling_params = self._build_sampling_params(
            greedy=greedy,
            stop_strings=per_sample_stop_strings,
            input_len=input_len,
        )

        router_ip = self.sglang_cfg["sglang_router"]["sglang_router_ip"]
        router_port = self.sglang_cfg["sglang_router"]["sglang_router_port"]

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
        pad_token_id = self.sglang_cfg["_pad_token_id"]
        original_input_ids_single_row = input_ids_batch[sample_idx]
        device = original_input_ids_single_row.device
        dtype = original_input_ids_single_row.dtype

        num_generated_tokens = len(new_tokens)
        final_output_tensor_len = input_len + num_generated_tokens

        output_ids_single_item = torch.full(
            (final_output_tensor_len,), pad_token_id, dtype=dtype, device=device
        )
        output_ids_single_item[:input_len] = original_input_ids_single_row[:input_len]
        if new_tokens:
            output_ids_single_item[input_len:final_output_tensor_len] = torch.tensor(
                new_tokens, dtype=dtype, device=device
            )
        output_ids_single_item_batched = output_ids_single_item.unsqueeze(0)

        # Logprobs: zeros for input tokens, raw floats at generated positions.
        logprobs_single_item = torch.zeros(
            (1, final_output_tensor_len), dtype=torch.float32, device=device
        )
        if new_logprobs:
            for idx, logprob in enumerate(new_logprobs):
                logprobs_single_item[0, input_len + idx] = logprob

        generation_lengths_tensor = torch.tensor(
            [num_generated_tokens], dtype=torch.long, device=device
        )
        unpadded_sequence_lengths_tensor = torch.tensor(
            [final_output_tensor_len], dtype=torch.long, device=device
        )
        truncated_tensor = torch.tensor(
            [bool(is_truncated)], dtype=torch.bool, device=device
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

        yield (sample_idx, result_batch)

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
    """Start sgl router or miles router and return (router_ip, router_port).

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
        model_path= model_path,
        router_ip=router_ip,
        router_port=router_port,
    )

    handles, port_cursors = server_group.start_engines(port_cursors)
    all_init_handles.extend(handles)

    if all_init_handles:
        ray.get(all_init_handles)

    return server_group
