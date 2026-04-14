import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    get_reordered_bundle_and_gpu_ids,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
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
from nemo_rl.distributed.ray_actor_environment_registry import SGLANG_EXECUTABLE
from nemo_rl.models.generation.redesign.misc import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from nemo_rl.models.generation.redesign.ray_utils import Lock
from nemo_rl.models.generation.redesign.sglang_router import RouterActor
from nemo_rl.models.generation.redesign.sglang_worker import SGLangGenerationWorker
from nemo_rl.models.generation.redesign.fault_tolerance import RolloutHealthMonitor

from nemo_rl.utils.nsys import wrap_with_nvtx_name

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class SGLangGeneration(GenerationInterface):
    """The class to run rollout and convert rollout data to training data.

    This class owns the full rollout server topology: the placement group,
    the router subprocess, and every ``SGLangGenerationWorker`` Ray actor.
    The former ``ServerGroup`` dataclass has been folded in so there is a
    single source of truth for engine state.
    """

    def __init__(self, cluster: RayVirtualCluster, cluster_cfg: ClusterConfig, sglang_cfg: SGLangConfig):
        self.cluster = cluster
        self.cluster_cfg = cluster_cfg
        self.sglang_cfg = sglang_cfg

        pgs = cluster._init_placement_groups(
            strategy="PACK",
            use_unified_pg=True,
        )
        self.pg = pgs[0]
        self.pg_reordered_bundle_indices, self.pg_reordered_gpu_ids = get_reordered_bundle_and_gpu_ids(self.pg)

        init_http_client(sglang_cfg)

        # --- Engine topology (formerly ``ServerGroup``) ------------------
        gpus_per_engine = sglang_cfg["sglang_server"]["num_gpus_per_engine"]
        num_gpus_per_node = cluster_cfg["gpus_per_node"]
        num_gpu_per_engine_local = min(gpus_per_engine, num_gpus_per_node)
        num_engines = sglang_cfg["sglang_server"]["num_gpus"] // num_gpu_per_engine_local

        self.num_gpus_per_engine: int = gpus_per_engine
        self.num_gpus_per_node: int = num_gpus_per_node
        self.all_engines: list = [None] * num_engines
        self.num_new_engines: int = 0
        self.rank_offset: int = 0
        self.gpu_offset: int = 0
        self.needs_offload: bool = sglang_cfg["sglang_server"]["needs_offload"]
        self.model_path: str | None = sglang_cfg["sglang_cfg"]["model_path"]

        # --- Router bootstrap --------------------------------------------
        router_ip, router_port, router_actor = _start_router(sglang_cfg)
        sglang_cfg["sglang_router"]["sglang_router_ip"] = router_ip
        sglang_cfg["sglang_router"]["sglang_router_port"] = router_port
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

        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        monitor = None
        if sglang_cfg["sglang_cfg"].get("use_fault_tolerance"):
            monitor = RolloutHealthMonitor(self, sglang_cfg)
            monitor.start()
        self._health_monitor = monitor

    # ------------------------------------------------------------------
    # Engine topology properties (formerly ``ServerGroup``)
    # ------------------------------------------------------------------
    @property
    def nodes_per_engine(self) -> int:
        return max(1, self.num_gpus_per_engine // self.num_gpus_per_node)

    @property
    def engines(self) -> list:
        """Node-0 engines only (one entry per logical engine).

        For multi-node TP, ``all_engines`` contains ``nodes_per_engine``
        consecutive actors per logical engine; this slice returns just the
        node-0 representative for each.
        """
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

    # ------------------------------------------------------------------
    # Engine lifecycle (formerly ``ServerGroup.start_engines`` / ``recover``)
    # ------------------------------------------------------------------
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

        local_all_engines = []
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

            # Explicitly pass CUDA_VISIBLE_DEVICES through to the engine actor so
            # all engines see the same global value (Ray would otherwise remap it
            # because we set the NOSET_* flags above).
            global_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if global_cvd:
                env_vars["CUDA_VISIBLE_DEVICES"] = global_cvd

            actor_options = {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
                "scheduling_strategy": scheduling_strategy,
                "runtime_env": {
                    "py_executable": SGLANG_EXECUTABLE,
                    "env_vars": env_vars,
                    **get_nsight_config_if_pattern_matches("sglang_generation_worker"),
                },
            }
            init_args = (self.cluster_cfg, self.sglang_cfg)
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

        self.num_new_engines = len(local_all_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        base_port = max(port_cursors.values()) if port_cursors else 15000
        addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
            cluster_cfg=self.cluster_cfg,
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

    def _recover(self) -> None:
        """Recover dead engines, overlapping init."""
        dead_indices = [i for i, engine in enumerate(self.all_engines) if engine is None]

        port_cursors: dict[int, int] = {}
        handles, _ = self._start_engines(port_cursors)
        if handles:
            ray.get(handles)

        assert self.num_new_engines == len(dead_indices), (
            "num_new_engines does not match dead_indices length"
        )

        if self.needs_offload and dead_indices:
            new_engines = [self.all_engines[i] for i in dead_indices]
            ray.get([engine.release_memory_weights.remote() for engine in new_engines])
            ray.get([engine.resume_memory_weights.remote() for engine in new_engines])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        return (
            self.engines,
            self.rollout_engine_lock,
            self.num_new_engines,
            self.engine_gpu_counts,
            self.engine_gpu_offsets,
        )

    def offload_weights(self):
        if not self.needs_offload:
            return
        handles = [
            engine.release_memory_weights.remote()
            for engine in self.engines
            if engine is not None
        ]
        if handles:
            ray.get(handles)

    def offload_kv(self):
        handles = [
            engine.release_memory_kv_cache_and_cuda_graph.remote()
            for engine in self.engines
            if engine is not None
        ]
        if handles:
            ray.get(handles)

    def onload_weights(self):
        if not self.needs_offload:
            return
        handles = [
            engine.resume_memory_weights.remote()
            for engine in self.engines
            if engine is not None
        ]
        if handles:
            ray.get(handles)

    def onload_kv(self):
        handles = [
            engine.resume_memory_kv_cache_and_cuda_graph.remote()
            for engine in self.engines
            if engine is not None
        ]
        if handles:
            ray.get(handles)

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update ``num_new_engines``
        for weight-update detection.
        """
        self._recover()
        return (
            self.engines,
            self.rollout_engine_lock,
            self.num_new_engines,
            self.engine_gpu_counts,
            self.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        self.num_new_engines = 0

    def check_weights(self, action: str):
        """All node-0 engines across all servers / models."""
        return ray.get(
            [
                engine.check_weights.remote(action=action)
                for engine in self.engines
                if engine is not None
            ]
        )

    def health_monitoring_pause(self) -> None:
        if self._health_monitor:
            self._health_monitor.pause()

    def health_monitoring_resume(self) -> None:
        if self._health_monitor:
            self._health_monitor.resume()

    def shutdown(self) -> bool:
        if self._health_monitor:
            self._health_monitor.stop()

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
        # Logprobs: zeros for input tokens, raw floats at generated positions.
        logprobs_single_item = torch.zeros(
            (1, unpadded_length), dtype=torch.float32, device=device
        )
        
        if new_tokens:
            output_ids_single_item[input_length:unpadded_length] = torch.tensor(
                new_tokens, dtype=dtype, device=device
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
        pass

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers and reset prefix cache."""
        pass
    
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
            results = ray.get([e.invalidate_kv_cache.remote() for e in engines])
        except Exception as e:
            logger.error(f"[sglang refit] Error flushing SGLang caches: {e}")
            return False

        success = all(results)
        if success:
            logger.info("[sglang refit] All SGLang server caches flushed successfully")
        else:
            logger.warning(
                "[sglang refit] WARNING - Some SGLang server caches failed to flush"
            )
        return success

    def get_sglang_server_urls(self) -> list[str]:
        """Return the base URLs of all SGLang servers (one per logical engine).

        Returns:
            List of base URLs, e.g. ``["http://host-a:30000", "http://host-b:30001"]``.
        """
        engines = [e for e in self.engines if e is not None]
        if not engines:
            raise RuntimeError("No rollout engines initialized")

        urls = ray.get([e.get_base_url.remote() for e in engines])
        return list({u for u in urls if u is not None})

    def get_sglang_url_to_gpu_uuids(self) -> dict[str, list[str]]:
        """Return a mapping of SGLang server URL to the GPU UUIDs it owns.

        For multi-node TP, a single logical engine spans ``nodes_per_engine``
        consecutive actors in ``all_engines``; we key by the node-0 URL and
        concatenate the local-node UUID slices from every peer in the group.

        Returns:
            Dict mapping base URL to list of GPU UUIDs,
            e.g. ``{"http://host-a:30000": ["GPU-aaa", "GPU-bbb"], ...}``.
        """
        if not any(e is not None for e in self.all_engines):
            raise RuntimeError("No rollout engines initialized")

        nodes_per_engine = self.nodes_per_engine

        # Fan out to every actor (not just node-0) so peer ranks can report
        # their own GPU slice. ``None`` slots mean the engine was killed by
        # the health monitor and not yet recovered; skip them.
        url_refs = [
            e.get_base_url.remote() if e is not None else None
            for e in self.all_engines
        ]
        uuid_refs = [
            e.get_gpu_uuids.remote() if e is not None else None
            for e in self.all_engines
        ]
        urls_live = ray.get([r for r in url_refs if r is not None])
        uuids_live = ray.get([r for r in uuid_refs if r is not None])

        # Re-key results back to their original slot index so gaps (None
        # engines) stay aligned with ``all_engines``.
        urls_by_slot: list[str | None] = []
        uuids_by_slot: list[list[str] | None] = []
        u_iter = iter(urls_live)
        g_iter = iter(uuids_live)
        for engine in self.all_engines:
            if engine is None:
                urls_by_slot.append(None)
                uuids_by_slot.append(None)
            else:
                urls_by_slot.append(next(u_iter))
                uuids_by_slot.append(next(g_iter))

        url_to_uuids: dict[str, list[str]] = {}
        for group_start in range(0, len(self.all_engines), nodes_per_engine):
            node0_url = urls_by_slot[group_start]
            if node0_url is None:
                continue
            aggregated: list[str] = []
            for i in range(group_start, group_start + nodes_per_engine):
                slot_uuids = uuids_by_slot[i]
                if slot_uuids:
                    aggregated.extend(slot_uuids)
            if aggregated:
                url_to_uuids[node0_url] = aggregated
        return url_to_uuids
    
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

    # SGLang reports the termination reason under meta_info.finish_reason.type;
    # "length" means the decoder hit max_new_tokens before EOS.
    finish_reason = output["meta_info"].get("finish_reason") or {}
    response_truncated = finish_reason.get("type") == "length"

    return index, response_tokens, response_log_probs, response_truncated



# ---------------------------------------------------------------------------
# Port allocation helpers
# ---------------------------------------------------------------------------
def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    cluster_cfg,
    sglang_cfg,
    local_all_engines,
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
    for rank, engine in local_all_engines:
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

    for i, _ in local_all_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor

def _start_router(
    sglang_cfg: SGLangConfig,
) -> tuple[str, int, ray.actor.ActorHandle | None]:
    """Start sgl router, returning ``(router_ip, router_port, actor_handle)``.

    If ``sglang_router.sglang_router_ip`` is already set, reuse it and return
    ``actor_handle=None`` (we do not own that router and must not terminate it).
    Otherwise spawn a ``RouterActor`` in sglang env to own the router process.
    """
    router_cfg = sglang_cfg["sglang_router"]
    if router_cfg["sglang_router_ip"] is not None:
        return router_cfg["sglang_router_ip"], router_cfg["sglang_router_port"], None

    router_actor = RouterActor.options(
        runtime_env={"py_executable": SGLANG_EXECUTABLE},
    ).remote()
    router_ip, router_port = ray.get(router_actor.start.remote(dict(router_cfg)))
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port, router_actor
