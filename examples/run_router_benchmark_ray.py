# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ray-based router benchmark using NeMo-RL's vLLM infrastructure.

This benchmark follows the standalone benchmark pattern:
- Workers and router are recreated for each configuration
- Ensures clean state and fair comparison
- First X iterations are warmup (CUDA graphs, kernel JIT) - not counted in metrics
- Remaining iterations are measured for benchmark results

This benchmark supports two routing modes:

1. MANUAL MODE (default): Separate VllmGeneration + KvRouter
   - Uses the same VllmGeneration setup as GRPO training
   - Manually coordinates routing and execution
   - More flexible but requires more code

2. INTEGRATED MODE (--use-integrated-routing): RoutedVllmWorkerGroup
   - Cleaner API with routing + execution in one call
   - Better encapsulation (router lifecycle tied to worker group)
   - Simpler code (~90 lines → ~10 lines)

Supports multi-node clusters:
- Single node: uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 1 --gpus-per-node 4 --warmup-iterations 1 --num-iterations 5
- Multi-node: uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 2 --gpus-per-node 8 --warmup-iterations 2 --num-iterations 5
- Integrated: uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 2 --gpus-per-node 8 --use-integrated-routing --warmup-iterations 1 --num-iterations 5
"""

import argparse
import asyncio
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray

from dynamo._core import compute_block_hash_for_seq_py
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.standalone_router import KvRouter
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.timer import Timer

logger = logging.getLogger(__name__)


@dataclass
class RouterBenchmarkConfig:
    name: str
    route_type: str  # "kv_integrated", "kv_manual", "round_robin"
    batch_size: int
    seq_len: int
    num_iterations: int = 5
    warmup_iterations: int = 0  # Number of warmup iterations (not counted in metrics)
    generation_max_tokens: int = 128

    @property
    def requests_per_iteration(self) -> int:
        return self.batch_size

    @property
    def total_requests(self) -> int:
        return self.requests_per_iteration * (self.num_iterations + self.warmup_iterations)
    
    @property
    def is_kv_routing(self) -> bool:
        return self.route_type in ["kv_integrated", "kv_manual"]
    
    @property
    def is_integrated(self) -> bool:
        return self.route_type == "kv_integrated"


@dataclass
class RouterBenchmarkResult:
    config_name: str
    total_attempted: int
    total_success: int
    total_errors: int
    total_time: float
    throughput: float
    router_latency: "LatencyStats"
    end_to_end_latency: "LatencyStats"
    generation_latency: Optional["LatencyStats"]
    worker_distribution: Dict[int, int]
    batch_routing_stats: Optional["BatchRoutingStats"] = None
    time_breakdown: Optional["TimeBreakdown"] = None


@dataclass
class BatchRoutingStats:
    """Statistics for batch routing performance."""
    total_routing_time_ms: float = 0.0
    hash_compute_time_ms: float = 0.0
    route_decision_time_ms: float = 0.0
    num_requests: int = 0
    
    @property
    def avg_routing_time_per_request_ms(self) -> float:
        return self.total_routing_time_ms / self.num_requests if self.num_requests > 0 else 0.0


@dataclass
class TimeBreakdown:
    """Wall-clock time breakdown for the entire benchmark."""
    total_time_s: float = 0.0
    routing_phase_s: float = 0.0
    generation_phase_s: float = 0.0
    overhead_s: float = 0.0
    
    @property
    def routing_pct(self) -> float:
        return (self.routing_phase_s / self.total_time_s * 100) if self.total_time_s > 0 else 0.0
    
    @property
    def generation_pct(self) -> float:
        return (self.generation_phase_s / self.total_time_s * 100) if self.total_time_s > 0 else 0.0
    
    @property
    def overhead_pct(self) -> float:
        return (self.overhead_s / self.total_time_s * 100) if self.total_time_s > 0 else 0.0


@dataclass
class LatencyStats:
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def create_temp_benchmark_dataset(max_samples: int, max_seq_length: int, seed: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    logger.info(
        "Creating synthetic dataset: %s samples, max sequence length %s",
        max_samples,
        max_seq_length,
    )

    rng = np.random.default_rng(seed)
    input_ids = rng.integers(1, 1000, size=(max_samples, max_seq_length), dtype=np.int64)
    base_length = min(max_seq_length, 150)
    input_lengths = np.array(
        [max(1, min(max_seq_length, base_length - (i % 3))) for i in range(max_samples)],
        dtype=np.int64,
    )

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
    }


def get_batch_tokens(dataset: Dict[str, np.ndarray], start_idx: int, batch_size: int, seq_len: int) -> List[List[int]]:
    input_ids = dataset["input_ids"]
    input_lengths = dataset["input_lengths"]

    end_idx = start_idx + batch_size
    if end_idx > len(input_ids):
        raise ValueError(
            "Synthetic dataset exhausted. Increase --dataset-samples or adjust benchmark settings."
        )

    tokens_batch: List[List[int]] = []
    for idx in range(start_idx, end_idx):
        length = int(min(seq_len, input_lengths[idx]))
        length = max(length, 1)
        tokens = input_ids[idx, :length].tolist()
        tokens_batch.append(tokens)

    return tokens_batch


def compute_latency_stats(latencies: List[float]) -> LatencyStats:
    if not latencies:
        return LatencyStats()

    lat_array = np.array(latencies)
    return LatencyStats(
        avg=float(lat_array.mean()),
        min=float(lat_array.min()),
        max=float(lat_array.max()),
        p50=float(np.percentile(lat_array, 50)),
        p95=float(np.percentile(lat_array, 95)),
        p99=float(np.percentile(lat_array, 99)),
    )


# ============================================================================
# ROUTE 1: KV-Aware Integrated (RoutedVllmWorkerGroup)
# ============================================================================

async def execute_batch_kv_integrated(
    routed_worker_group,  # RoutedVllmWorkerGroup
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float]:
    """KV-Aware routing with integrated RoutedVllmWorkerGroup.
    
    Uses built-in batch routing API for efficient routing decisions.
    """
    import torch
    from dynamo._core import compute_block_hash_for_seq_py
    
    router_latencies: List[float] = []
    total_latencies: List[float] = []
    generation_latencies: List[float] = []
    worker_counts: Counter[int] = Counter()
    failed_requests = 0
    timer = Timer()
    
    try:
        # Phase 1: Batch route all requests
        with timer.time("routing_total"):
            # Compute hashes for KV-aware routing
            with timer.time("hash_compute"):
                batch_hashes = []
                batch_num_tokens = []
                for prompt_token_ids in tokens_batch:
                    local_hashes = compute_block_hash_for_seq_py(prompt_token_ids, block_size)
                    batch_hashes.append(local_hashes)
                    batch_num_tokens.append(len(prompt_token_ids))
            
            # Batch routing decision
            with timer.time("route_decision"):
                worker_ids = await routed_worker_group.get_best_worker_ids_batch(
                    batch_hashes=batch_hashes,
                    batch_num_tokens=batch_num_tokens
                )
        
        routing_elapsed = timer.get_latest_elapsed("routing_total")
        hash_compute_time = timer.get_latest_elapsed("hash_compute")
        route_decision_time = timer.get_latest_elapsed("route_decision")
        routing_phase_time = routing_elapsed
        
        logger.info(
            f"KV-Aware Integrated routing: total={routing_elapsed*1000:.2f}ms "
            f"(hash={hash_compute_time*1000:.2f}ms, decision={route_decision_time*1000:.2f}ms)"
        )
        
        # Phase 2: Execute all requests in parallel
        async def execute_single_request(idx: int, prompt_token_ids: List[int], worker_id: int):
            req_timer = Timer()
            try:
                with req_timer.time("total"):
                    data = BatchedDataDict[GenerationDatumSpec]({
                        "input_ids": torch.tensor([prompt_token_ids], dtype=torch.int64),
                        "input_lengths": torch.tensor([len(prompt_token_ids)], dtype=torch.int64),
                    })
                    
                    worker_gen = routed_worker_group.run_single_worker_single_data(
                        method_name="generate_async",
                        worker_idx=worker_id,
                        data=data,
                    )
                    
                    async for sample_result_ref in worker_gen:
                        await sample_result_ref
                
                # Get timing AFTER context manager exits
                return {
                    "success": True,
                    "total_elapsed": req_timer.get_latest_elapsed("total"),
                    "worker_id": worker_id,
                }
            except Exception as exc:  # noqa: BLE001
                logger.error("Request %s failed on worker %s: %s", idx, worker_id, exc, exc_info=True)
                return {"success": False}
        
        # Sort by worker_id for deterministic submission
        request_tuples = [
            (idx, tokens, worker_id)
            for idx, (tokens, worker_id) in enumerate(zip(tokens_batch, worker_ids))
        ]
        request_tuples.sort(key=lambda x: x[2])
        
        tasks = [
            execute_single_request(idx, tokens, worker_id)
            for idx, tokens, worker_id in request_tuples
        ]
        
        with timer.time("generation"):
            results = await asyncio.gather(*tasks, return_exceptions=False)
        generation_phase_time = timer.get_latest_elapsed("generation")
        
        # Collect metrics
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                router_latencies.append(routing_elapsed / len(tokens_batch))
                total_latencies.append(result["total_elapsed"])
                generation_latencies.append(result["total_elapsed"] - (routing_elapsed / len(tokens_batch)))
                worker_counts[result["worker_id"]] += 1
            else:
                failed_requests += 1
        
        routing_stats = BatchRoutingStats(
            total_routing_time_ms=routing_elapsed * 1000,
            hash_compute_time_ms=hash_compute_time * 1000,
            route_decision_time_ms=route_decision_time * 1000,
            num_requests=len(tokens_batch),
        )
                
    except Exception as exc:  # noqa: BLE001
        logger.error("KV-Aware Integrated routing failed: %s", exc, exc_info=True)
        failed_requests += len(tokens_batch)
        routing_stats = None
        routing_phase_time = 0.0
        generation_phase_time = 0.0
    
    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts, routing_stats, routing_phase_time, generation_phase_time


# ============================================================================
# ROUTE 2: KV-Aware Manual (Separate Router + VllmGeneration)
# ============================================================================

async def execute_batch_kv_manual(
    router,  # KvRouter
    policy_generation,  # VllmGeneration
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float]:
    """KV-Aware routing with separate router and vLLM workers.
    
    Manually coordinates routing and generation execution.
    """
    import torch
    from collections import defaultdict
    from dynamo._core import compute_block_hash_for_seq_py
    
    router_latencies: List[float] = []
    total_latencies: List[float] = []
    generation_latencies: List[float] = []
    worker_counts: Counter[int] = Counter()
    failed_requests = 0
    worker_batches: Dict[int, List[Tuple[int, List[int], float]]] = defaultdict(list)
    timer = Timer()
    
    # Phase 1: Route all requests
    with timer.time("routing_total"):
        for idx, prompt_token_ids in enumerate(tokens_batch):
            try:
                # Compute hash
                with timer.time("hash"):
                    local_hashes = compute_block_hash_for_seq_py(prompt_token_ids, block_size)
                
                # Route decision
                with timer.time("decision"):
                    worker_id = await router.get_best_worker(local_hashes, len(prompt_token_ids))
                
                # Get per-request routing time (hash + decision for this request)
                routing_elapsed = timer.get_latest_elapsed("hash") + timer.get_latest_elapsed("decision")
                
                worker_batches[worker_id].append((idx, prompt_token_ids, routing_elapsed))
                
            except Exception as exc:  # noqa: BLE001
                failed_requests += 1
                logger.error("KV-Aware Manual routing failed for request %d: %s", idx, exc, exc_info=True)
    
    total_routing_time = timer.get_latest_elapsed("routing_total")
    routing_phase_time = total_routing_time
    
    # Use Timer's built-in aggregation - sum all hash and decision times
    total_hash_compute_time = timer.reduce("hash", "sum") if "hash" in timer._timers else 0.0
    total_route_decision_time = timer.reduce("decision", "sum") if "decision" in timer._timers else 0.0
    
    logger.info(
        f"KV-Aware Manual routing: total={total_routing_time*1000:.2f}ms "
        f"(hash={total_hash_compute_time*1000:.2f}ms, decision={total_route_decision_time*1000:.2f}ms)"
    )
    
    # Phase 2: Execute all requests in parallel
    async def execute_single_request(worker_id: int, prompt_token_ids: List[int], routing_time: float):
        gen_timer = Timer()
        try:
            with gen_timer.time("generation"):
                data = BatchedDataDict[GenerationDatumSpec]({
                    "input_ids": torch.tensor([prompt_token_ids], dtype=torch.int64),
                    "input_lengths": torch.tensor([len(prompt_token_ids)], dtype=torch.int64),
                })
                
                worker_gen = policy_generation.worker_group.run_single_worker_single_data(
                    method_name="generate_async",
                    worker_idx=worker_id,
                    data=data,
                )
                
                async for sample_result_ref in worker_gen:
                    await sample_result_ref
            
            # Get timing AFTER context manager exits
            generation_time = gen_timer.get_latest_elapsed("generation")
            
            return {
                "success": True,
                "routing_time": routing_time,
                "total_elapsed": routing_time + generation_time,
                "generation_time": generation_time,
                "worker_id": worker_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("KV-Aware Manual request failed on worker %s: %s", worker_id, exc, exc_info=True)
            return {"success": False}
    
    try:
        tasks = []
        for worker_id in sorted(worker_batches.keys()):
            requests = worker_batches[worker_id]
            for idx, prompt_token_ids, routing_time in requests:
                tasks.append(execute_single_request(worker_id, prompt_token_ids, routing_time))
        
        with timer.time("generation"):
            results = await asyncio.gather(*tasks, return_exceptions=False)
        generation_phase_time = timer.get_latest_elapsed("generation")
        
        # Collect metrics
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                router_latencies.append(result["routing_time"])
                total_latencies.append(result["total_elapsed"])
                generation_latencies.append(result["generation_time"])
                worker_counts[result["worker_id"]] += 1
            else:
                failed_requests += 1
        
        routing_stats = BatchRoutingStats(
            total_routing_time_ms=total_routing_time * 1000,
            hash_compute_time_ms=total_hash_compute_time * 1000,
            route_decision_time_ms=total_route_decision_time * 1000,
            num_requests=len(tokens_batch),
        )
        
    except Exception as exc:  # noqa: BLE001
        logger.error("KV-Aware Manual generation failed: %s", exc, exc_info=True)
        failed_requests += len(tokens_batch)
        routing_stats = None
        generation_phase_time = 0.0
    
    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts, routing_stats, routing_phase_time, generation_phase_time


# ============================================================================
# ROUTE 3: Round-Robin (No KV Hashing)
# ============================================================================

async def execute_batch_round_robin(
    router,  # RoutedVllmWorkerGroup or KvRouter
    policy_generation,  # VllmGeneration or None
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
    use_integrated: bool,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float]:
    """Round-robin routing (no KV awareness).
    
    Works with both integrated and manual modes.
    """
    import torch
    from collections import defaultdict
    
    router_latencies: List[float] = []
    total_latencies: List[float] = []
    generation_latencies: List[float] = []
    worker_counts: Counter[int] = Counter()
    failed_requests = 0
    timer = Timer()
    
    try:
        # Phase 1: Route all requests with round-robin
        with timer.time("routing_total"):
            # No hash computation for round-robin
            batch_hashes = []
            batch_num_tokens = []
            for prompt_token_ids in tokens_batch:
                batch_hashes.append([])  # Empty (unused)
                batch_num_tokens.append(len(prompt_token_ids))
            hash_compute_time = 0.0
            
            # Route with round-robin
            with timer.time("route_decision"):
                worker_ids = []
                for local_hashes, num_tokens in zip(batch_hashes, batch_num_tokens):
                    worker_id = await router.get_worker_round_robin(local_hashes, num_tokens)
                    worker_ids.append(worker_id)
        
        routing_elapsed = timer.get_latest_elapsed("routing_total")
        route_decision_time = timer.get_latest_elapsed("route_decision")
        routing_phase_time = routing_elapsed
        
        logger.info(
            f"Round-Robin routing: total={routing_elapsed*1000:.2f}ms "
            f"(decision={route_decision_time*1000:.2f}ms)"
        )
        
        # Phase 2: Execute all requests in parallel
        async def execute_single_request(idx: int, prompt_token_ids: List[int], worker_id: int):
            req_timer = Timer()
            try:
                with req_timer.time("total"):
                    data = BatchedDataDict[GenerationDatumSpec]({
                        "input_ids": torch.tensor([prompt_token_ids], dtype=torch.int64),
                        "input_lengths": torch.tensor([len(prompt_token_ids)], dtype=torch.int64),
                    })
                    
                    if use_integrated:
                        worker_gen = router.run_single_worker_single_data(
                            method_name="generate_async",
                            worker_idx=worker_id,
                            data=data,
                        )
                    else:
                        worker_gen = policy_generation.worker_group.run_single_worker_single_data(
                            method_name="generate_async",
                            worker_idx=worker_id,
                            data=data,
                        )
                    
                    async for sample_result_ref in worker_gen:
                        await sample_result_ref
                
                # Get timing AFTER context manager exits
                return {
                    "success": True,
                    "total_elapsed": req_timer.get_latest_elapsed("total"),
                    "worker_id": worker_id,
                }
            except Exception as exc:  # noqa: BLE001
                logger.error("Request %s failed on worker %s: %s", idx, worker_id, exc, exc_info=True)
                return {"success": False}
        
        # Sort by worker_id for deterministic submission
        request_tuples = [
            (idx, tokens, worker_id)
            for idx, (tokens, worker_id) in enumerate(zip(tokens_batch, worker_ids))
        ]
        request_tuples.sort(key=lambda x: x[2])
        
        tasks = [
            execute_single_request(idx, tokens, worker_id)
            for idx, tokens, worker_id in request_tuples
        ]
        
        with timer.time("generation"):
            results = await asyncio.gather(*tasks, return_exceptions=False)
        generation_phase_time = timer.get_latest_elapsed("generation")
        
        # Collect metrics
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                router_latencies.append(routing_elapsed / len(tokens_batch))
                total_latencies.append(result["total_elapsed"])
                generation_latencies.append(result["total_elapsed"] - (routing_elapsed / len(tokens_batch)))
                worker_counts[result["worker_id"]] += 1
            else:
                failed_requests += 1
        
        routing_stats = BatchRoutingStats(
            total_routing_time_ms=routing_elapsed * 1000,
            hash_compute_time_ms=hash_compute_time * 1000,
            route_decision_time_ms=route_decision_time * 1000,
            num_requests=len(tokens_batch),
        )
                
    except Exception as exc:  # noqa: BLE001
        logger.error("Round-Robin routing failed: %s", exc, exc_info=True)
        failed_requests += len(tokens_batch)
        routing_stats = None
        routing_phase_time = 0.0
        generation_phase_time = 0.0
    
    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts, routing_stats, routing_phase_time, generation_phase_time


# ============================================================================
# Shared Route Dispatcher
# ============================================================================

async def execute_batch_with_route(
    config: RouterBenchmarkConfig,
    router,  # Union[KvRouter, RoutedVllmWorkerGroup]
    policy_generation,  # Union[VllmGeneration, None]
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float]:
    """Dispatch to appropriate route based on config."""
    if config.route_type == "kv_integrated":
        return await execute_batch_kv_integrated(
            router, tokens_batch, block_size, max_gen_tokens
        )
    elif config.route_type == "kv_manual":
        return await execute_batch_kv_manual(
            router, policy_generation, tokens_batch, block_size, max_gen_tokens
        )
    elif config.route_type == "round_robin":
        return await execute_batch_round_robin(
            router, policy_generation, tokens_batch, block_size, max_gen_tokens, config.is_integrated
        )
    else:
        raise ValueError(f"Unknown route type: {config.route_type}")


# ============================================================================
# SHARED BENCHMARK RUNNER
# ============================================================================
# run_router_benchmark now uses the clean dispatcher above


async def run_router_benchmark(
    router,  # Union[KvRouter, RoutedVllmWorkerGroup]
    policy_generation,  # Union[VllmGeneration, None]
    config: RouterBenchmarkConfig,
    dataset: Dict[str, np.ndarray],
    block_size: int,
) -> RouterBenchmarkResult:
    """Run a single benchmark configuration."""
    print(f"\nBenchmark: {config.name}")
    print(f"  Route type:             {config.route_type}")
    print(f"  Target batch size:      {config.batch_size}")
    print(f"  Requests per iteration: {config.requests_per_iteration}")
    print(f"  Sequence length:        {config.seq_len}")
    print(f"  Generation max tokens:  {config.generation_max_tokens}")

    total_attempted = 0
    total_success = 0
    total_errors = 0
    total_time = 0.0
    all_router_latencies: List[float] = []
    all_total_latencies: List[float] = []
    all_generation_latencies: List[float] = []
    aggregated_workers: Counter[int] = Counter()
    
    # Aggregate batch routing stats across iterations
    total_routing_time_ms = 0.0
    total_hash_compute_time_ms = 0.0
    total_route_decision_time_ms = 0.0
    total_routed_requests = 0
    
    # Aggregate wall-clock phase times
    total_routing_phase_time_s = 0.0
    total_generation_phase_time_s = 0.0

    sample_offset = 0

    # Prepare generation (only needed for manual mode)
    if not config.is_integrated and policy_generation is not None:
        policy_generation.prepare_for_generation()

    try:
        total_iterations = config.warmup_iterations + config.num_iterations
        
        # Run warmup iterations first (not counted in metrics)
        if config.warmup_iterations > 0:
            print(f"  Running {config.warmup_iterations} warmup iteration(s)...")
        
        for iteration in range(total_iterations):
            is_warmup = iteration < config.warmup_iterations
            requests_this_iteration = config.requests_per_iteration
            tokens_batch = get_batch_tokens(
                dataset,
                sample_offset,
                requests_this_iteration,
                config.seq_len,
            )
            
            # Only count attempts for non-warmup iterations
            if not is_warmup:
                total_attempted += requests_this_iteration

            iter_timer = Timer()
            with iter_timer.time("iteration"):
                (
                    router_latencies,
                    total_latencies,
                    generation_latencies,
                    failed_requests,
                    worker_counts,
                    batch_routing_stats,
                    routing_phase_time,
                    generation_phase_time,
                ) = await execute_batch_with_route(
                    config,
                    router,
                    policy_generation,
                    tokens_batch,
                    block_size,
                    config.generation_max_tokens,
                )
            iteration_time = iter_timer.get_latest_elapsed("iteration")

            successes = len(total_latencies)
            
            # Only aggregate metrics for non-warmup iterations
            if not is_warmup:
                total_success += successes
                total_errors += failed_requests
                total_time += iteration_time
                all_router_latencies.extend(router_latencies)
                all_total_latencies.extend(total_latencies)
                if generation_latencies:
                    all_generation_latencies.extend(generation_latencies)
                aggregated_workers.update(worker_counts)
                
                # Aggregate routing stats
                if batch_routing_stats:
                    total_routing_time_ms += batch_routing_stats.total_routing_time_ms
                    total_hash_compute_time_ms += batch_routing_stats.hash_compute_time_ms
                    total_route_decision_time_ms += batch_routing_stats.route_decision_time_ms
                    total_routed_requests += batch_routing_stats.num_requests
                
                # Aggregate phase times
                total_routing_phase_time_s += routing_phase_time
                total_generation_phase_time_s += generation_phase_time

                print(
                    f"    Iteration {iteration - config.warmup_iterations + 1}/{config.num_iterations}: "
                    f"{iteration_time:.3f}s, successes={successes}, errors={failed_requests}"
                )
            else:
                print(
                    f"    Warmup {iteration + 1}/{config.warmup_iterations}: "
                    f"{iteration_time:.3f}s, successes={successes}, errors={failed_requests}"
                )

    finally:
        # Cleanup (only needed for manual mode)
        if not config.is_integrated and policy_generation is not None:
            policy_generation.finish_generation()

    if aggregated_workers:
        distribution = ", ".join(
            f"{worker_id}:{count}" for worker_id, count in sorted(aggregated_workers.items())
        )
        print(f"    Worker assignment totals: {distribution}")

    router_latency_stats = compute_latency_stats(all_router_latencies)
    end_to_end_latency_stats = compute_latency_stats(all_total_latencies)
    generation_latency_stats = compute_latency_stats(all_generation_latencies)

    throughput = (total_success / total_time) if total_time > 0 else 0.0
    
    # Create aggregated batch routing stats
    aggregated_batch_routing_stats = None
    if total_routed_requests > 0:
        aggregated_batch_routing_stats = BatchRoutingStats(
            total_routing_time_ms=total_routing_time_ms,
            hash_compute_time_ms=total_hash_compute_time_ms,
            route_decision_time_ms=total_route_decision_time_ms,
            num_requests=total_routed_requests,
        )
    
    # Create time breakdown
    overhead_time_s = total_time - total_routing_phase_time_s - total_generation_phase_time_s
    time_breakdown = TimeBreakdown(
        total_time_s=total_time,
        routing_phase_s=total_routing_phase_time_s,
        generation_phase_s=total_generation_phase_time_s,
        overhead_s=max(0.0, overhead_time_s),  # Ensure non-negative
    )

    return RouterBenchmarkResult(
        config_name=config.name,
        total_attempted=total_attempted,
        total_success=total_success,
        total_errors=total_errors,
        total_time=total_time,
        throughput=throughput,
        router_latency=router_latency_stats,
        end_to_end_latency=end_to_end_latency_stats,
        generation_latency=generation_latency_stats,
        worker_distribution=dict(sorted(aggregated_workers.items())),
        batch_routing_stats=aggregated_batch_routing_stats,
        time_breakdown=time_breakdown,
    )


async def initialize_workers_and_router(
    args: argparse.Namespace,
    inference_cluster: RayVirtualCluster,
    vllm_config: VllmConfig,
    total_workers: int,
):
    """Initialize workers and router for a single benchmark configuration.
    
    Returns:
        Tuple of (router, policy_generation, routed_worker_group) where only one of
        policy_generation or routed_worker_group will be non-None depending on mode.
    """
    router = None
    policy_generation = None
    routed_worker_group = None
    
    if args.use_integrated_routing:
        # Integrated mode: Use RoutedVllmWorkerGroup
        from nemo_rl.distributed.worker_groups import RayWorkerBuilder
        from nemo_rl.distributed.named_sharding import NamedSharding
        from nemo_rl.models.generation.dynamo.routed_worker_group import (
            RoutedVllmWorkerGroup, RouterConfig
        )
        import numpy as np
        
        logger.info("Initializing RoutedVllmWorkerGroup (integrated mode)...")
        
        # Initialize placement groups (same as VllmGeneration does)
        # This is critical for proper GPU allocation and performance
        inference_cluster._init_placement_groups(
            strategy="PACK",  # Use PACK for non-colocated inference
            use_unified_pg=False,  # We're not using cross-node model parallelism
        )
        
        # Create worker builder (same pattern as VllmGeneration line 161)
        worker_builder = RayWorkerBuilder(
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
            vllm_config,
        )
        
        # Create router config
        router_config = RouterConfig(
            block_size=args.block_size,
            base_kv_events_port=args.base_kv_events_port,
            base_metrics_port=args.base_metrics_port,
            enabled=True,
        )
        
        # Create sharding annotations (same pattern as VllmGeneration line 131-136)
        # Extract parallelism settings from vllm_config
        tp_size = vllm_config["vllm_cfg"]["tensor_parallel_size"]
        pp_size = vllm_config["vllm_cfg"]["pipeline_parallel_size"]
        ep_size = vllm_config["vllm_cfg"]["expert_parallel_size"]
        model_parallel_size = tp_size * pp_size * ep_size
        dp_size = total_workers // model_parallel_size
        
        sharding_annotations = NamedSharding(
            layout=np.arange(total_workers).reshape(dp_size, pp_size, tp_size),
            names=["data_parallel", "pipeline_parallel", "tensor_parallel"],
        )
        
        # Set up environment variables (same as VllmGeneration)
        env_vars = {}
        # Explicitly set NCCL_CUMEM_ENABLE to 1 for non-colocated inference
        env_vars["NCCL_CUMEM_ENABLE"] = "1"
        
        # Create routed worker group
        routed_worker_group = RoutedVllmWorkerGroup(
            cluster=inference_cluster,
            remote_worker_builder=worker_builder,
            router_config=router_config,
            sharding_annotations=sharding_annotations,
            name_prefix="router_benchmark_vllm",
            env_vars=env_vars,
        )
        
        # Start router background tasks
        await routed_worker_group.start_router()
        logger.info(f"✓ RoutedVllmWorkerGroup initialized with {total_workers} workers")
        
        # Reset prefix cache on all workers (same as manual mode's finish_generation())
        # This ensures clean cache state before benchmarking
        logger.info("Resetting prefix cache on all workers...")
        reset_refs = routed_worker_group.run_all_workers_single_data(
            method_name="reset_prefix_cache_async",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        ray.get(reset_refs)
        logger.info("✓ Prefix cache reset complete")
        
        # Set router to the worker group for the benchmark
        router = routed_worker_group
        
    else:
        # Manual mode: Separate VllmGeneration + KvRouter
        logger.info("Initializing VllmGeneration + KvRouter (manual mode)...")
        
        # Create vLLM generation policy (similar to tiny_exp.py line 122)
        policy_generation = VllmGeneration(
            cluster=inference_cluster,
            config=vllm_config,
            name_prefix="router_benchmark_vllm",
        )
        policy_generation.finish_generation()  # Initialize workers
        logger.info("✓ VllmGeneration initialized")

        # Initialize router with total workers across all nodes
        router = KvRouter(
            block_size=args.block_size,
            num_workers=total_workers,
            base_kv_events_port=args.base_kv_events_port,
            base_metrics_port=args.base_metrics_port,
        )
        await router.start_background_tasks()
        logger.info(f"✓ KvRouter initialized with {total_workers} workers and background tasks started")

    # Give router time to collect initial metrics
    logger.info("Waiting for router to collect initial metrics...")
    await asyncio.sleep(2.0)
    
    return router, policy_generation, routed_worker_group


async def cleanup_workers_and_router(
    args: argparse.Namespace,
    router,
    policy_generation,
    routed_worker_group,
):
    """Cleanup workers and router after a benchmark configuration."""
    if args.use_integrated_routing:
        logger.info("Shutting down RoutedVllmWorkerGroup...")
        if routed_worker_group:
            await routed_worker_group.shutdown_async()
    else:
        logger.info("Shutting down router...")
        if router:
            await router.shutdown()
        
        logger.info("Shutting down VllmGeneration workers...")
        if policy_generation:
            policy_generation.shutdown()
    
    # Give time for cleanup
    await asyncio.sleep(1.0)


async def execute_benchmarks(
    configs: List[RouterBenchmarkConfig],
    dataset: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> List[RouterBenchmarkResult]:
    """Execute all benchmark configurations.
    
    Following the standalone benchmark pattern, we recreate workers and router
    for each configuration to ensure a clean state and fair comparison.
    
    Supports two modes:
    1. Manual mode (--use-integrated-routing=False): Separate router + VllmGeneration
    2. Integrated mode (--use-integrated-routing=True): RoutedVllmWorkerGroup
    """
    if not configs:
        raise ValueError("No benchmark configurations provided")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized")

    # Setup virtual cluster (similar to tiny_exp.py)
    # Support multi-node setup: each node has gpus_per_node GPUs
    num_nodes = args.num_nodes
    gpus_per_node = args.gpus_per_node
    total_workers = num_nodes * gpus_per_node
    
    # Each node gets gpus_per_node bundles
    bundle_ct_per_node_list = [gpus_per_node] * num_nodes
    
    inference_cluster = RayVirtualCluster(
        name="router_benchmark_cluster",
        bundle_ct_per_node_list=bundle_ct_per_node_list,
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=1,
    )
    logger.info(
        f"✓ Ray cluster initialized: {num_nodes} node(s) × {gpus_per_node} GPU(s) = {total_workers} total workers"
    )

    # Load tokenizer to configure generation config
    from transformers import AutoTokenizer
    from nemo_rl.models.generation import configure_generation_config
    
    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Setup vLLM configuration (similar to GRPO setup)
    vllm_config: VllmConfig = {
        "backend": "vllm",
        "model_name": args.model,
        "max_new_tokens": args.generation_max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "colocated": {"enabled": False, "resources": {}},
        "vllm_cfg": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "async_engine": True,  # Required for best performance with PP
            "skip_tokenizer_init": True,
            "enforce_eager": False,
            "precision": "bfloat16",  # Use bfloat16 instead of auto
            "max_model_len": args.max_seq_length,
            "load_format": "auto",  # Required by vLLM worker
            "use_deep_gemm": False,
            "num_last_layers_in_bf16": 0,
            "num_first_layers_in_bf16": 0,
            "hf_overrides": {},
        },
        "vllm_kwargs": {},  # Additional vLLM kwargs (optional)
    }
    
    # Configure vLLM with tokenizer to set _pad_token_id
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)

    results: List[RouterBenchmarkResult] = []

    try:
        # Following standalone benchmark pattern: recreate workers/router for each config
        for idx, config in enumerate(configs):
            logger.info(f"\nStarting fresh workers and router for: {config.name}")
            
            router = None
            policy_generation = None
            routed_worker_group = None
            
            try:
                # Initialize workers and router
                router, policy_generation, routed_worker_group = await initialize_workers_and_router(
                    args, inference_cluster, vllm_config, total_workers
                )
                
                # Run benchmark with fresh instances
                result = await run_router_benchmark(
                    router,
                    policy_generation,
                    config,
                    dataset,
                    args.block_size,
                )
                results.append(result)
            
            finally:
                # Cleanup workers and router
                await cleanup_workers_and_router(
                    args, router, policy_generation, routed_worker_group
                )

    finally:
        # Cleanup cluster
        logger.info("Shutting down cluster...")
        inference_cluster.shutdown()

        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

    return results


def print_benchmark_results(results: List[RouterBenchmarkResult]) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("RAY-BASED ROUTER BENCHMARK RESULTS")
    print("=" * 70)

    for result in results:
        print(f"\n{result.config_name}")
        print("-" * 70)
        print(f"  Requests attempted:   {result.total_attempted}")
        print(f"  Successful requests:  {result.total_success}")
        print(f"  Errors:               {result.total_errors}")
        print(f"  Total time:           {result.total_time:.3f}s")
        print(f"  Throughput:           {result.throughput:.2f} req/s")
        router_stats = result.router_latency
        end_to_end_stats = result.end_to_end_latency
        print("  Router latency (ms):")
        print(
            f"    avg={router_stats.avg * 1000:.2f}, "
            f"p95={router_stats.p95 * 1000:.2f}, "
            f"p99={router_stats.p99 * 1000:.2f}"
        )
        print(
            f"  End-to-end latency (ms): avg={end_to_end_stats.avg * 1000:.2f}, "
            f"p95={end_to_end_stats.p95 * 1000:.2f}, p99={end_to_end_stats.p99 * 1000:.2f}"
        )
        if result.generation_latency is not None:
            gen_stats = result.generation_latency
            print(
                f"  Generation latency (ms): avg={gen_stats.avg * 1000:.2f}, "
                f"p95={gen_stats.p95 * 1000:.2f}, p99={gen_stats.p99 * 1000:.2f}"
            )
        if result.time_breakdown is not None:
            td = result.time_breakdown
            print("  Wall-clock time breakdown:")
            print(f"    Total time:       {td.total_time_s:.3f}s (100%)")
            print(f"    - Routing phase:  {td.routing_phase_s:.3f}s ({td.routing_pct:.1f}%)")
            print(f"    - Generation:     {td.generation_phase_s:.3f}s ({td.generation_pct:.1f}%)")
            print(f"    - Overhead:       {td.overhead_s:.3f}s ({td.overhead_pct:.1f}%)")
        if result.batch_routing_stats is not None:
            batch_stats = result.batch_routing_stats
            print("  Batch routing breakdown:")
            print(
                f"    Total routing time: {batch_stats.total_routing_time_ms:.2f}ms "
                f"({batch_stats.avg_routing_time_per_request_ms:.2f}ms/req)"
            )
            print(f"    - Hash computation: {batch_stats.hash_compute_time_ms:.2f}ms")
            print(f"    - Route decision:   {batch_stats.route_decision_time_ms:.2f}ms")
            print(f"    Total requests:     {batch_stats.num_requests}")
        if result.worker_distribution:
            print("  Worker distribution:")
            for worker_id, count in result.worker_distribution.items():
                print(f"    Worker {worker_id}: {count}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        best_throughput = max(results, key=lambda r: r.throughput)
        print(
            f"  Best throughput: {best_throughput.config_name} "
            f"({best_throughput.throughput:.2f} req/s)"
        )

        latency_candidates = [r for r in results if r.end_to_end_latency.avg > 0]
        if latency_candidates:
            best_latency = min(latency_candidates, key=lambda r: r.end_to_end_latency.avg)
            print(
                f"  Lowest avg latency: {best_latency.config_name} "
                f"({best_latency.end_to_end_latency.avg * 1000:.2f} ms)"
            )

    print("=" * 70)


def build_benchmark_configs(args: argparse.Namespace) -> List[RouterBenchmarkConfig]:
    """Build 3 independent benchmark configurations based on routing mode."""
    
    if args.use_integrated_routing:
        # Integrated mode: KV-Aware Integrated + Round-Robin Integrated
        configs = [
            RouterBenchmarkConfig(
                name="KV-Aware (Integrated)",
                route_type="kv_integrated",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
            ),
            RouterBenchmarkConfig(
                name="Round-Robin (Integrated)",
                route_type="round_robin",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
            ),
        ]
    else:
        # Manual mode: KV-Aware Manual + Round-Robin Manual
        configs = [
            RouterBenchmarkConfig(
                name="KV-Aware (Manual)",
                route_type="kv_manual",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
            ),
            RouterBenchmarkConfig(
                name="Round-Robin (Manual)",
                route_type="round_robin",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
            ),
        ]
    
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark router with Ray-based vLLM workers (NeMo-RL infrastructure)."
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Number of warmup iterations (not counted in metrics, used for CUDA graph capture, kernel JIT)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    # Multi-node cluster configuration
    parser.add_argument(
        "--num-nodes", 
        type=int, 
        default=1,
        help="Number of nodes in the cluster"
    )
    parser.add_argument(
        "--gpus-per-node", 
        type=int, 
        default=8,
        help="Number of GPUs per node (default: 8 for DGX nodes)"
    )
    parser.add_argument("--base-kv-events-port", type=int, default=5557)
    parser.add_argument("--base-metrics-port", type=int, default=5657)
    parser.add_argument("--generation-max-tokens", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--dataset-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--use-integrated-routing",
        action="store_true",
        default=False,
        help="Use RoutedVllmWorkerGroup with integrated routing (cleaner API, experimental)"
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    configs = build_benchmark_configs(args)

    samples_needed = max(config.total_requests for config in configs)
    if args.dataset_samples is not None and args.dataset_samples < samples_needed:
        raise ValueError(
            "Provided --dataset-samples is smaller than the number of samples required by the benchmark."
        )

    max_samples = args.dataset_samples or samples_needed
    dataset = create_temp_benchmark_dataset(max_samples, args.max_seq_length, args.seed)

    print("Ray-Based Router Benchmarking Suite")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Block size:      {args.block_size}")
    print(f"Cluster:         {args.num_nodes} node(s) × {args.gpus_per_node} GPU(s) = {args.num_nodes * args.gpus_per_node} workers")
    print(f"Routing mode:    {'Integrated (RoutedVllmWorkerGroup)' if args.use_integrated_routing else 'Manual (VllmGeneration + KvRouter)'}")
    print(f"Warmup iters:    {args.warmup_iterations} (not counted in metrics)")
    print(f"Counted iters:   {args.num_iterations} (counted in metrics)")
    print(f"Benchmark configs: {len(configs)}")
    print(f"Note: Workers/router recreated for each config (following standalone pattern)")
    print("=" * 70)

    results = await execute_benchmarks(configs, dataset, args)
    print_benchmark_results(results)

    success = all(result.total_errors == 0 for result in results)
    return 0 if success else 1


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        exit_code = 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Benchmark failed: %s", exc)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# WARMUP PATTERN (following standalone benchmark):
# -------------------------------------------------
# Workers and router are recreated for each configuration
# First X iterations are warmup (CUDA graphs, kernel JIT) - not counted
# Remaining iterations are measured for benchmark results
#
# MANUAL MODE (default): Separate VllmGeneration + KvRouter
# ----------------------------------------------------------
# Single node with 8 GPUs (typical DGX node):
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 1 --gpus-per-node 8 --warmup-iterations 1 --num-iterations 5
#
# Multi-node with 2 nodes × 8 GPUs = 16 total workers:
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 2 --gpus-per-node 8 --model "Qwen/Qwen2.5-Math-1.5B-Instruct" --batch-size 128 --seq-len 256 --warmup-iterations 2 --num-iterations 10 --generation-max-tokens 256
#
# Multi-node with 4 nodes × 8 GPUs = 32 total workers:
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 4 --gpus-per-node 8 --warmup-iterations 1
#
# Small test with 2 nodes × 2 GPUs (no warmup for quick testing):
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 2 --gpus-per-node 2 --batch-size 32 --warmup-iterations 0 --num-iterations 3
#
# INTEGRATED MODE: RoutedVllmWorkerGroup (cleaner API)
# -----------------------------------------------------
# Use --use-integrated-routing flag to enable integrated routing:
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 1 --gpus-per-node 4 --use-integrated-routing --warmup-iterations 1
#
# Comparison between manual and integrated mode:
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 1 --gpus-per-node 4 --warmup-iterations 1
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 1 --gpus-per-node 4 --use-integrated-routing --warmup-iterations 1
#
# Benefits of integrated mode:
#   - Cleaner API: routing + execution in one call
#   - Better encapsulation: router lifecycle tied to worker group
#   - Simpler code: ~90 lines of routing logic → ~10 lines
#
# ADVANCED OPTIONS
# ----------------
# Large-scale benchmark with custom model and extended warmup:
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 8 --gpus-per-node 8 --model "Qwen/Qwen2.5-Math-1.5B-Instruct" --batch-size 128 --seq-len 256 --warmup-iterations 3 --num-iterations 10 --generation-max-tokens 256
#
# With custom ports (to avoid conflicts):
#   uv run --extra vllm examples/run_router_benchmark_ray.py --num-nodes 2 --gpus-per-node 8 --base-kv-events-port 6000 --base-metrics-port 6100 --warmup-iterations 1
#
# NOTE: Port allocation for multi-node setups:
#   - Each worker i uses: base_kv_events_port + i and base_metrics_port + i
#   - With 2 nodes × 8 GPUs = 16 workers, you need 16 ports for KV events
#     and 16 ports for metrics
#   - Ensure ports [base_port, base_port+total_workers) are available
#
# WARMUP ITERATIONS:
#   - Default: 1 warmup iteration (recommended for production benchmarks)
#   - 0 warmup: Quick testing (but first measured iteration will be slow)
#   - 2-3 warmup: Conservative approach for very large models
#   - Warmup triggers: CUDA graph capture, kernel JIT, memory allocation
#   - Each config gets fresh workers, so each pays warmup cost once
#
# Expected benefits with multi-node:
#   - Router can balance load across more workers
#   - Better cache hit rates when requests share prefixes
#   - Higher throughput with more parallel workers
#   - More interesting routing decisions to observe
#
