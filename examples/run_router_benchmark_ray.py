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

Built on top of sechoi/dynamo_router (b97785e), this benchmark compares routing approaches:

**vLLM V1 KV Events Status**: This benchmark uses VllmV1KvEventListener to properly
consume KV cache events from vLLM V1's msgpack-based protocol (vs V0's JSON protocol).
However, as of vLLM 0.6.x, the V1 AsyncLLM API does not emit KV cache events, even when
configured with `enable_kv_cache_events=True`. This means KV-aware routing will degrade
to load-based routing (overlap=0.000). Run `python test_vllm_v1_kv_events.py` to verify
if your vLLM version supports KV event emission. See TEST_VLLM_V1_KV_EVENTS.md for details.

This benchmark follows the standalone benchmark pattern:
- Workers and router are recreated for each configuration
- Ensures clean state and fair comparison
- First X iterations are warmup (CUDA graphs, kernel JIT) - not counted in metrics
- Remaining iterations are measured for benchmark results

This benchmark supports THREE routing modes:

1. SIMPLE MODE (default): VllmGeneration with built-in router (sechoi's approach)
   - Router integrated directly into VllmGeneration via router_cfg
   - Completely transparent - just set config and it works
   - Minimal code changes, best for standard GRPO workflows
   - Routing happens automatically in generate_async()

2. MANUAL MODE: Separate VllmGeneration + KvRouter
   - Manually coordinate routing and execution
   - More flexible, explicit control over routing decisions
   - Useful for custom routing strategies

3. INTEGRATED MODE (--use-integrated-routing): RoutedVllmWorkerGroup
   - Dedicated RoutedVllmWorkerGroup class with batch routing APIs
   - Cleaner API for advanced use cases
   - Better for benchmarking and testing
   - Includes batch routing: get_best_worker_ids_batch()

Supports multi-node clusters:
- Simple mode:     uv run --extra dynamo examples/run_router_benchmark_ray.py --num-nodes 1 --gpus-per-node 4 --warmup-iterations 1 --num-iterations 5
- Manual mode:     uv run --extra dynamo examples/run_router_benchmark_ray.py --num-nodes 2 --gpus-per-node 8 --use-manual-routing --warmup-iterations 2 --num-iterations 5 --num-generations-per-prompt 8
- Integrated mode: uv run --extra dynamo examples/run_router_benchmark_ray.py --num-nodes 2 --gpus-per-node 8 --use-integrated-routing --warmup-iterations 1 --num-iterations 5

Comparison:
- Simple mode: Best integration, minimal code, transparent operation (RECOMMENDED for most users)
- Manual mode: Maximum flexibility, explicit routing control
- Integrated mode: Clean API for advanced use, batch routing support
"""

import argparse
import asyncio
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from transformers import AutoTokenizer

from dynamo._core import compute_block_hash_for_seq_py
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.standalone_router import KvRouter
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.timer import Timer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset, load_eval_dataset
from nemo_rl.data.interfaces import TaskDataSpec, TaskDataProcessFnCallable
from nemo_rl.data.llm_message_utils import message_log_to_flat_messages, batched_message_log_to_flat_message
from nemo_rl.data.processors import math_hf_data_processor

logger = logging.getLogger(__name__)


@dataclass
class RouterBenchmarkConfig:
    name: str
    route_type: str  # "kv_simple", "kv_integrated", "kv_manual", "round_robin"
    batch_size: int
    seq_len: int
    num_iterations: int = 5
    warmup_iterations: int = 0  # Number of warmup iterations (not counted in metrics)
    generation_max_tokens: int = 128
    is_integrated: bool = False  # Explicit flag for integrated vs manual mode
    shuffle_batch: bool = False  # Shuffle requests before routing to prevent alignment artifacts

    @property
    def requests_per_iteration(self) -> int:
        return self.batch_size

    @property
    def total_requests(self) -> int:
        return self.requests_per_iteration * (self.num_iterations + self.warmup_iterations)
    
    @property
    def is_kv_routing(self) -> bool:
        return self.route_type in ["kv_simple", "kv_integrated", "kv_manual"]
    
    @property
    def is_simple(self) -> bool:
        return self.route_type == "kv_simple"


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
    worker_prefix_distribution: Optional[Dict[int, List[int]]] = None  # Worker ID -> list of prefix indices
    worker_prefix_distribution: Optional[Dict[int, List[int]]] = None  # Worker ID -> list of prefix indices


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


def setup_benchmark_data(
    dataset_name: str, 
    tokenizer_name: str, 
    max_samples: Optional[int] = None, 
    seed: int = 42,
    prompt_file: str = "examples/prompts/cot.txt"
) -> Tuple[AllTaskProcessedDataset, Any]:
    """Setup benchmark data following run_grpo_math.py pattern EXACTLY.
    
    This function mirrors GRPO's dataset setup (run_grpo_math.py lines 65-124):
    1. Create TaskDataSpec (like line 77-81)
    2. Load dataset using load_response_dataset() (like line 84)
    3. Create task_data_processors dict (like line 87-90)
    4. Create AllTaskProcessedDataset (like line 102-108)
    5. Return dataset for batched processing (like GRPO grpo_train loop)
    
    Args:
        dataset_name: Name of the dataset (e.g., "OpenMathInstruct-2", "DeepScaler", "DAPOMath17K")
        tokenizer_name: Name of the tokenizer/model to use
        max_samples: Maximum number of samples to load (None = load all)
        seed: Random seed for dataset loading
        
    Returns:
        Tuple of (AllTaskProcessedDataset, tokenizer) for batched processing
    """
    logger.info(f"\n▶ Setting up benchmark data from {dataset_name}...")
    
    # Step 1: Create TaskDataSpec - same as run_grpo_math.py line 77-81
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=prompt_file,
        system_prompt_file=None,
    )
    
    # Step 2: Prepare data config and load dataset - same as run_grpo_math.py
    data_config = {
        "dataset_name": dataset_name,
        "prompt_file": prompt_file,
        "system_prompt_file": None,
    }
    
    # Load dataset using repo function - same as run_grpo_math.py line 84
    logger.info(f"Loading {dataset_name} using load_response_dataset()...")
    data: Any = load_response_dataset(data_config, seed)
    
    # Step 3: Create task_data_processors - same as run_grpo_math.py line 87-90
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (math_task_spec, math_hf_data_processor))
    )
    task_data_processors["math"] = (math_task_spec, math_hf_data_processor)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Step 4: Create AllTaskProcessedDataset - same as run_grpo_math.py line 102-108
    logger.info("Creating AllTaskProcessedDataset...")
    full_dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=16384,  # Default max_seq_length for benchmarking
    )
    
    # Limit dataset size if requested - use Subset like GRPO would
    if max_samples is not None and max_samples < len(full_dataset):
        from torch.utils.data import Subset
        indices = list(range(max_samples))
        dataset = Subset(full_dataset, indices)
        logger.info(f"✓ Dataset created with {len(dataset)} samples (subset of {len(full_dataset)})")
    else:
        dataset = full_dataset
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
    
    return dataset, tokenizer


def create_temp_benchmark_dataset(max_samples: int, max_seq_length: int, seed: int, num_prefixes: int = 1) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    logger.info(
        "Creating synthetic dataset: %s samples, max sequence length %s, %s prefixes",
        max_samples,
        max_seq_length,
        num_prefixes,
    )

    rng = np.random.default_rng(seed)
    
    # generate random input_ids
    input_ids = rng.integers(1, 1000, size=(max_samples, max_seq_length), dtype=np.int64)
    
    # Track which prefix index each sample uses
    prefix_indices = np.zeros(max_samples, dtype=np.int64)
    
    # If using multiple prefixes, overwrite the beginning of each sequence
    if num_prefixes > 1:
        # Use 80% of sequence as prefix
        prefix_len = int(max_seq_length * 0.8)
        logger.info(f"Generating {num_prefixes} distinct prefixes of length {prefix_len}...")
        
        # Generate the distinct prefixes
        prefixes = rng.integers(1, 1000, size=(num_prefixes, prefix_len), dtype=np.int64)
        
        # Assign prefixes to samples round-robin
        for i in range(max_samples):
            prefix_idx = i % num_prefixes
            input_ids[i, :prefix_len] = prefixes[prefix_idx]
            prefix_indices[i] = prefix_idx  # Track the prefix index
            
    # For synthetic data, leave room for generation
    # If we use full max_seq_length as prompt, vLLM has no room to generate tokens
    # and will return immediately (0ms latency).
    prompt_len = max(1, max_seq_length - 128)  # Leave 128 tokens for generation
    input_lengths = np.full(max_samples, prompt_len, dtype=np.int64)

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "prefix_indices": prefix_indices,  # Add prefix tracking
    }


def get_batch_tokens(dataset: Dict[str, np.ndarray], start_idx: int, batch_size: int, seq_len: int) -> List[List[int]]:
    input_ids = dataset["input_ids"]
    input_lengths = dataset["input_lengths"]
    total_samples = len(input_ids)

    tokens_batch: List[List[int]] = []
    for i in range(batch_size):
        # Use modulo to wrap around dataset if we run out of samples
        idx = (start_idx + i) % total_samples
        
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
# ROUTE 0: KV-Aware Simple (VllmGeneration with router_cfg) - sechoi's approach
# ============================================================================

async def execute_batch_kv_simple(
    policy_generation,  # VllmGeneration with router_cfg enabled
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
    sequential: bool = False,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float, Optional[Dict[int, List[int]]]]:
    """KV-Aware routing using VllmGeneration's built-in router (sechoi's approach).
    
    This is the simplest integration - routing happens automatically inside VllmGeneration.
    The router_cfg is set during VllmGeneration initialization, and generate_async()
    automatically routes requests to the best worker.
    
    This demonstrates the transparent integration approach from sechoi/dynamo_router.
    """
    import torch
    
    router_latencies: List[float] = []
    total_latencies: List[float] = []
    generation_latencies: List[float] = []
    worker_counts: Counter[int] = Counter()
    failed_requests = 0
    timer = Timer()
    
    try:
        # Phase 1: Execute all requests (routing happens automatically inside)
        async def execute_single_request(idx: int, prompt_token_ids: List[int]):
            req_timer = Timer()
            try:
                with req_timer.time("total"):
                    data = BatchedDataDict[GenerationDatumSpec]({
                        "input_ids": torch.tensor([prompt_token_ids], dtype=torch.int64),
                        "input_lengths": torch.tensor([len(prompt_token_ids)], dtype=torch.int64),
                    })
                    
                    # VllmGeneration.generate_async() automatically:
                    # 1. Computes block hashes from input_ids
                    # 2. Calls router.get_best_worker() or router.get_worker_round_robin()
                    # 3. Routes to selected worker
                    # All of this is transparent to the user!
                    async for sample_result in policy_generation.generate_async(data, greedy=False):
                        idx, result_batch = sample_result  # Unpack the tuple (idx, BatchedDataDict)
                
                return {
                    "success": True,
                    "total_elapsed": req_timer.get_latest_elapsed("total"),
                }
            except Exception as exc:
                logger.error(f"Request {idx} failed: {exc}", exc_info=True)
                return {"success": False}
        
        tasks = [
            execute_single_request(idx, tokens)
            for idx, tokens in enumerate(tokens_batch)
        ]
        
        with timer.time("generation"):
            if sequential:
                # Process requests one at a time
                results = []
                for task in tasks:
                    result = await task
                    results.append(result)
            else:
                # Process requests in parallel
                results = await asyncio.gather(*tasks, return_exceptions=False)
        generation_phase_time = timer.get_latest_elapsed("generation")
        
        # Collect metrics
        # Note: We can't track individual routing latency or worker assignments in simple mode
        # because routing is internal to VllmGeneration
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                total_latencies.append(result["total_elapsed"])
                generation_latencies.append(result["total_elapsed"])
            else:
                failed_requests += 1
        
        # No batch routing stats available in simple mode (routing is internal)
        routing_stats = None
        routing_phase_time = 0.0
                
    except Exception as exc:
        logger.error(f"KV-Aware Simple routing failed: {exc}", exc_info=True)
        failed_requests += len(tokens_batch)
        routing_stats = None
        routing_phase_time = 0.0
        generation_phase_time = 0.0
    
    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts, routing_stats, routing_phase_time, generation_phase_time, None


# ============================================================================
# ROUTE 1: KV-Aware Integrated (RoutedVllmWorkerGroup)
# ============================================================================

async def execute_batch_kv_integrated(
    routed_worker_group,  # RoutedVllmWorkerGroup
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
    sequential: bool = False,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float, Optional[Dict[int, List[int]]]]:
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
            if sequential:
                # Process requests one at a time
                results = []
                for task in tasks:
                    result = await task
                    results.append(result)
            else:
                # Process requests in parallel
                results = await asyncio.gather(*tasks, return_exceptions=False)
        generation_phase_time = timer.get_latest_elapsed("generation")
        
        # Collect metrics
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                router_latencies.append(routing_elapsed / len(tokens_batch))
                total_latencies.append(result["total_elapsed"])
                generation_latencies.append(result["total_elapsed"] - (routing_elapsed / len(tokens_batch)))
                worker_counts[result["worker_id"]] += 1
                
                # DEBUG: Print details of the first successful request
                if not hasattr(logger, "_debug_printed_req_details"):
                    logger.info(f"[DEBUG] Request details (Sample):")
                    logger.info(f"  - Worker ID: {result['worker_id']}")
                    logger.info(f"  - Prompt len: {len(tokens_batch[0])} tokens")
                    logger.info(f"  - Generated tokens: {result.get('num_generated_tokens', 'N/A')}")
                    logger.info(f"  - Generated text: {result.get('generated_text', 'N/A')!r}")
                    logger.info(f"  - Total time: {result['total_elapsed']*1000:.2f} ms")
                    logger._debug_printed_req_details = True
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
    
    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts, routing_stats, routing_phase_time, generation_phase_time, None


# ============================================================================
# ROUTE 2: KV-Aware Manual (Separate Router + VllmGeneration)
# ============================================================================

async def execute_batch_kv_manual(
    router,  # KvRouter
    policy_generation,  # VllmGeneration
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
    sequential: bool = False,
    prefix_indices: Optional[List[int]] = None,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float, Optional[Dict[int, List[int]]]]:
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
    
    # Track worker -> prefix indices mapping if prefix_indices provided
    worker_to_prefixes_result = None
    if prefix_indices is not None:
        from collections import defaultdict as dd
        worker_to_prefixes = dd(list)
        for worker_id, requests in worker_batches.items():
            for idx, _, _ in requests:
                if idx < len(prefix_indices):
                    prefix_idx = prefix_indices[idx]
                    worker_to_prefixes[worker_id].append(prefix_idx)
        
        # Convert to regular dict for result
        worker_to_prefixes_result = dict(worker_to_prefixes)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"KV-AWARE ROUTING: Worker → Sample IDs (repetitions = cache hits)")
        logger.info(f"{'='*70}")
        for worker_id in sorted(worker_to_prefixes.keys()):
            prefixes = worker_to_prefixes[worker_id]
            # Show first 20 sample IDs for this worker
            prefix_str = str(prefixes[:20])
            if len(prefixes) > 20:
                prefix_str = prefix_str[:-1] + f", ... ({len(prefixes)-20} more)]"
            logger.info(f"  Worker {worker_id:2d} → Samples {prefix_str}")
        logger.info(f"{'='*70}\n")
    
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
                
                generated_text = ""
                num_tokens = 0
                
                async for sample_result_ref in worker_gen:
                    # sample_result_ref is likely a tuple (idx, result_dict) or BatchedDataDict
                    # We need to resolve the Ray object ref if it is one, but run_single_worker_single_data
                    # usually returns an async generator yielding refs or values.
                    # Given the existing code awaits it, it's likely a coroutine or ref.
                    batch_result = await sample_result_ref
                    
                    # Inspect the result structure to extract text/tokens
                    # Usually it's (idx, BatchedDataDict)
                    if isinstance(batch_result, tuple) and len(batch_result) == 2:
                        _, result_data = batch_result
                        # result_data should be BatchedDataDict or dict
                        if hasattr(result_data, "get"):
                            # Try standard keys
                            if "generated_text" in result_data:
                                val = result_data["generated_text"]
                                if isinstance(val, (list, tuple)) and len(val) > 0:
                                    generated_text = val[0]
                                else:
                                    generated_text = str(val)
                            
                            if "num_generated_tokens" in result_data:
                                val = result_data["num_generated_tokens"]
                                if hasattr(val, "item"): num_tokens = val.item()
                                elif isinstance(val, (list, tuple)) and len(val) > 0: num_tokens = val[0]
                                else: num_tokens = int(val)
                            
                            # vLLM worker might return 'output_ids'
                            if num_tokens == 0 and "output_ids" in result_data:
                                val = result_data["output_ids"]
                                if hasattr(val, "shape"): num_tokens = val.shape[-1]
                                elif isinstance(val, list): num_tokens = len(val[0]) if len(val)>0 else 0
            
            # Get timing AFTER context manager exits
            generation_time = gen_timer.get_latest_elapsed("generation")
            
            return {
                "success": True,
                "routing_time": routing_time,
                "total_elapsed": routing_time + generation_time,
                "generation_time": generation_time,
                "worker_id": worker_id,
                "generated_text": generated_text,
                "num_generated_tokens": num_tokens,
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
            if sequential:
                # Process requests one at a time
                results = []
                for task in tasks:
                    result = await task
                    results.append(result)
            else:
                # Process requests in parallel
                results = await asyncio.gather(*tasks, return_exceptions=False)
        generation_phase_time = timer.get_latest_elapsed("generation")
        
        # Collect metrics
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                router_latencies.append(routing_elapsed / len(tokens_batch))
                total_latencies.append(result["total_elapsed"])
                generation_latencies.append(result["total_elapsed"] - (routing_elapsed / len(tokens_batch)))
                worker_counts[result["worker_id"]] += 1
                
                # DEBUG: Print details of the first successful request in the batch
                if not hasattr(execute_batch_kv_manual, "_debug_printed"):
                    logger.info(f"[DEBUG] Request details:")
                    logger.info(f"  - Worker ID: {result['worker_id']}")
                    logger.info(f"  - Prompt len: {len(tokens_batch[0])} tokens")
                    logger.info(f"  - Total time: {result['total_elapsed']*1000:.2f} ms")
                    # We assume successful result has tokens, but we can't see them here easily 
                    # without modifying the worker return signature.
                    execute_batch_kv_manual._debug_printed = True
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
        worker_to_prefixes_result = None
    
    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts, routing_stats, routing_phase_time, generation_phase_time, worker_to_prefixes_result


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
    sequential: bool = False,
    prefix_indices: Optional[List[int]] = None,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float, Optional[Dict[int, List[int]]]]:
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
                    
                    generated_text = ""
                    num_tokens = 0
                    
                    async for sample_result_ref in worker_gen:
                        batch_result = await sample_result_ref
                        
                        # Inspect the result structure to extract text/tokens
                        if isinstance(batch_result, tuple) and len(batch_result) == 2:
                            _, result_data = batch_result
                            if hasattr(result_data, "get"):
                                if "generated_text" in result_data:
                                    val = result_data["generated_text"]
                                    if isinstance(val, (list, tuple)) and len(val) > 0:
                                        generated_text = val[0]
                                    else:
                                        generated_text = str(val)
                                
                                if "num_generated_tokens" in result_data:
                                    val = result_data["num_generated_tokens"]
                                    if hasattr(val, "item"): num_tokens = val.item()
                                    elif isinstance(val, (list, tuple)) and len(val) > 0: num_tokens = val[0]
                                    else: num_tokens = int(val)
                                
                                if num_tokens == 0 and "output_ids" in result_data:
                                    val = result_data["output_ids"]
                                    if hasattr(val, "shape"): num_tokens = val.shape[-1]
                                    elif isinstance(val, list): num_tokens = len(val[0]) if len(val)>0 else 0
                
                # Get timing AFTER context manager exits
                return {
                    "success": True,
                    "total_elapsed": req_timer.get_latest_elapsed("total"),
                    "worker_id": worker_id,
                    "generated_text": generated_text,
                    "num_generated_tokens": num_tokens,
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
        
        # Track worker -> prefix indices mapping if prefix_indices provided
        worker_to_prefixes_result = None
        if prefix_indices is not None:
            from collections import defaultdict
            worker_to_prefixes = defaultdict(list)
            for original_idx, (_, _, worker_id) in enumerate(request_tuples):
                prefix_idx = prefix_indices[original_idx]
                worker_to_prefixes[worker_id].append(prefix_idx)
            
            # Convert to regular dict for result
            worker_to_prefixes_result = dict(worker_to_prefixes)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"ROUND-ROBIN ROUTING: Worker → Sample IDs (repetitions = cache hits)")
            logger.info(f"{'='*70}")
            for worker_id in sorted(worker_to_prefixes.keys()):
                prefixes = worker_to_prefixes[worker_id]
                # Show first 20 sample IDs for this worker
                prefix_str = str(prefixes[:20])
                if len(prefixes) > 20:
                    prefix_str = prefix_str[:-1] + f", ... ({len(prefixes)-20} more)]"
                logger.info(f"  Worker {worker_id:2d} → Samples {prefix_str}")
            logger.info(f"{'='*70}\n")
        
        tasks = [
            execute_single_request(idx, tokens, worker_id)
            for idx, tokens, worker_id in request_tuples
        ]
        
        with timer.time("generation"):
            if sequential:
                # Process requests one at a time
                results = []
                for task in tasks:
                    result = await task
                    results.append(result)
            else:
                # Process requests in parallel
                results = await asyncio.gather(*tasks, return_exceptions=False)
        generation_phase_time = timer.get_latest_elapsed("generation")
        
        # Collect metrics
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                router_latencies.append(routing_elapsed / len(tokens_batch))
                total_latencies.append(result["total_elapsed"])
                generation_latencies.append(result["total_elapsed"] - (routing_elapsed / len(tokens_batch)))
                worker_counts[result["worker_id"]] += 1
                
                # DEBUG: Print details of the first successful request
                if not hasattr(logger, "_debug_printed_req_details"):
                    logger.info(f"[DEBUG] Request details (Sample):")
                    logger.info(f"  - Worker ID: {result['worker_id']}")
                    logger.info(f"  - Prompt len: {len(tokens_batch[0])} tokens")
                    logger.info(f"  - Generated tokens: {result.get('num_generated_tokens', 'N/A')}")
                    logger.info(f"  - Generated text: {result.get('generated_text', 'N/A')!r}")
                    logger.info(f"  - Total time: {result['total_elapsed']*1000:.2f} ms")
                    logger._debug_printed_req_details = True
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
        worker_to_prefixes_result = None
    
    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts, routing_stats, routing_phase_time, generation_phase_time, worker_to_prefixes_result


# ============================================================================
# Shared Route Dispatcher
# ============================================================================

async def execute_batch_with_route(
    config: RouterBenchmarkConfig,
    router,  # Union[KvRouter, RoutedVllmWorkerGroup, None]
    policy_generation,  # Union[VllmGeneration, None]
    tokens_batch: List[List[int]],
    block_size: int,
    max_gen_tokens: int,
    sequential: bool = False,
    prefix_indices: Optional[List[int]] = None,
) -> Tuple[List[float], List[float], List[float], int, Counter, Optional[BatchRoutingStats], float, float, Optional[Dict[int, List[int]]]]:
    """Dispatch to appropriate route based on config."""
    if config.route_type == "kv_simple":
        return await execute_batch_kv_simple(
            policy_generation, tokens_batch, block_size, max_gen_tokens, sequential
        )
    elif config.route_type == "kv_integrated":
        return await execute_batch_kv_integrated(
            router, tokens_batch, block_size, max_gen_tokens, sequential
        )
    elif config.route_type == "kv_manual":
        return await execute_batch_kv_manual(
            router, policy_generation, tokens_batch, block_size, max_gen_tokens, sequential, prefix_indices
        )
    elif config.route_type == "round_robin":
        return await execute_batch_round_robin(
            router, policy_generation, tokens_batch, block_size, max_gen_tokens, config.is_integrated, sequential, prefix_indices
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
    sequential: bool = False,
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
    
    # Aggregate worker -> prefix distribution across iterations
    aggregated_worker_prefix_dist: Dict[int, List[int]] = {}

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
            
            # Extract prefix_indices or generate sample IDs
            # For synthetic dataset: use prefix_indices (shows which shared prefix)
            # For real dataset: use sample indices (shows which unique prompt)
            batch_prefix_indices = None
            if "prefix_indices" in dataset:
                # Synthetic dataset with shared prefixes
                prefix_indices_array = dataset["prefix_indices"]
                end_idx = min(sample_offset + requests_this_iteration, len(prefix_indices_array))
                batch_prefix_indices = [
                    int(prefix_indices_array[i % len(prefix_indices_array)])
                    for i in range(sample_offset, end_idx)
                ]
            else:
                # Real dataset: track sample/prompt IDs
                total_samples = len(dataset["input_ids"])
                batch_prefix_indices = [
                    (sample_offset + i) % total_samples
                    for i in range(requests_this_iteration)
                ]
            
            # Shuffle batch to prevent Round-Robin from accidental alignment
            # when batch_size is a multiple of num_workers.
            # Without shuffle, RR sends the same static request to the same worker every iteration.
            if config.shuffle_batch:
                import random
                if batch_prefix_indices is not None:
                    # Shuffle both tokens_batch and prefix_indices together
                    combined = list(zip(tokens_batch, batch_prefix_indices))
                    random.shuffle(combined)
                    tokens_batch, batch_prefix_indices = zip(*combined)
                    tokens_batch = list(tokens_batch)
                    batch_prefix_indices = list(batch_prefix_indices)
                else:
                    random.shuffle(tokens_batch)
            
            # Advance sample offset for next iteration (wrapping handled in get_batch_tokens)
            sample_offset += requests_this_iteration
            
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
                    worker_prefix_dist,
                ) = await execute_batch_with_route(
                    config,
                    router,
                    policy_generation,
                    tokens_batch,
                    block_size,
                    config.generation_max_tokens,
                    sequential,
                    batch_prefix_indices,
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
                
                # Aggregate worker prefix distribution
                if worker_prefix_dist:
                    for worker_id, prefixes in worker_prefix_dist.items():
                        if worker_id not in aggregated_worker_prefix_dist:
                            aggregated_worker_prefix_dist[worker_id] = []
                        aggregated_worker_prefix_dist[worker_id].extend(prefixes)

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
        worker_prefix_distribution=dict(sorted(aggregated_worker_prefix_dist.items())) if aggregated_worker_prefix_dist else None,
    )


async def initialize_workers_and_router(
    args: argparse.Namespace,
    inference_cluster: RayVirtualCluster,
    vllm_config: VllmConfig,
    total_workers: int,
    config: RouterBenchmarkConfig,
):
    """Initialize workers and router for a single benchmark configuration.
    
    Args:
        config: The benchmark config to determine routing mode
    
    Returns:
        Tuple of (router, policy_generation, routed_worker_group) where only one of
        policy_generation or routed_worker_group will be non-None depending on mode.
        
    Three modes:
    - Simple: VllmGeneration with router_cfg enabled (default, sechoi's approach)
    - Manual: Separate VllmGeneration + KvRouter (--use-manual-routing)
    - Integrated: RoutedVllmWorkerGroup (--use-integrated-routing)
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
        # Only enable KV indexer for KV-aware routing, not for round-robin
        enable_kv_indexer = config.is_kv_routing
        await routed_worker_group.start_router(enable_kv_indexer=enable_kv_indexer)
        mode_str = "KV-aware" if enable_kv_indexer else "round-robin"
        logger.info(f"✓ RoutedVllmWorkerGroup initialized with {total_workers} workers ({mode_str} mode)")
        
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
        
    elif args.use_manual_routing:
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

        # Get worker addresses and ports for distributed setup
        worker_address_tuples = policy_generation._get_worker_addresses()
        worker_addresses = [addr for addr, _ in worker_address_tuples]
        worker_ports = [port for _, port in worker_address_tuples]
        
        logger.info(f"Manual mode: Got {len(worker_addresses)} worker addresses for router")
        if len(set(worker_addresses)) > 1:
            logger.info(f"Distributed setup: workers on {len(set(worker_addresses))} different nodes")

        # Initialize router with total workers across all nodes
        router = KvRouter(
            block_size=args.block_size,
            num_workers=total_workers,
            base_kv_events_port=args.base_kv_events_port,
            base_metrics_port=args.base_metrics_port,
            worker_addresses=worker_addresses,
            worker_ports=worker_ports,
        )
        # Only enable KV indexer for KV-aware routing, not for round-robin
        enable_kv_indexer = config.is_kv_routing
        await router.start_background_tasks(enable_kv_indexer=enable_kv_indexer)
        mode_str = "KV-aware" if enable_kv_indexer else "round-robin"
        logger.info(f"✓ KvRouter initialized with {total_workers} workers ({mode_str} mode)")
    
    else:
        # Simple mode (default): VllmGeneration with router_cfg enabled (sechoi's approach)
        logger.info("Initializing VllmGeneration with built-in routing (simple mode)...")
        
        # router_cfg should already be set in vllm_config before calling this function
        # Router is integrated directly into VllmGeneration, no separate router needed
        policy_generation = VllmGeneration(
            cluster=inference_cluster,
            config=vllm_config,
            name_prefix="router_benchmark_vllm",
        )
        
        # Start router (integrated into VllmGeneration)
        # Simple mode only does KV-aware routing, so always enable KV indexer
        enable_kv_indexer = config.is_kv_routing  # Should always be True for simple mode
        await policy_generation.start_router(enable_kv_indexer=enable_kv_indexer)
        logger.info("✓ VllmGeneration with built-in router initialized (KV-aware mode)")
        
        # Reset prefix cache
        policy_generation.finish_generation()  # Initialize workers
        logger.info("✓ Prefix cache reset complete")

    # Give router time to collect initial metrics
    # logger.info("Waiting for router to collect initial metrics...")
    # await asyncio.sleep(2.0)
    
    return router, policy_generation, routed_worker_group


async def cleanup_workers_and_router(
    args: argparse.Namespace,
    router,
    policy_generation,
    routed_worker_group,
):
    """Cleanup workers and router after a benchmark configuration."""
    if args.use_integrated_routing:
        # Integrated mode: Shutdown RoutedVllmWorkerGroup
        logger.info("Shutting down RoutedVllmWorkerGroup...")
        if routed_worker_group:
            await routed_worker_group.shutdown_async()
    elif args.use_manual_routing:
        # Manual mode: Shutdown separate router + VllmGeneration
        logger.info("Shutting down router...")
        if router:
            await router.shutdown()
        
        logger.info("Shutting down VllmGeneration workers...")
        if policy_generation:
            policy_generation.shutdown()
    else:
        # Simple mode: Shutdown VllmGeneration with built-in router
        logger.info("Shutting down VllmGeneration with built-in router...")
        if policy_generation:
            await policy_generation.stop_router()
            policy_generation.shutdown()
    
    # Give time for cleanup
    # await asyncio.sleep(1.0)


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
    
    # Add router_cfg for ALL modes - workers need to publish KV events
    # - Simple mode: Enables built-in routing in VllmGeneration + workers publish events
    # - Integrated mode: Workers publish events for RoutedVllmWorkerGroup's router
    # - Manual mode: Workers publish events for external KvRouter
    # The difference is only in how the router is created/managed, not worker config
    vllm_config["router_cfg"] = {
        "enabled": True,
        "mode": "best_worker",  # Used by simple mode's built-in routing
        "block_size": args.block_size,
        "base_kv_events_port": args.base_kv_events_port,
        "base_metrics_port": args.base_metrics_port,
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
                    args, inference_cluster, vllm_config, total_workers, config
                )
                
                # Run benchmark with fresh instances
                result = await run_router_benchmark(
                    router,
                    policy_generation,
                    config,
                    dataset,
                    args.block_size,
                    args.sequential,
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
        
        if result.worker_prefix_distribution:
            print("  Worker → Sample distribution (repetitions = cache hits):")
            for worker_id, prefixes in result.worker_prefix_distribution.items():
                # Show first 20 sample IDs
                prefix_str = str(prefixes[:20])
                if len(prefixes) > 20:
                    prefix_str = prefix_str[:-1] + f", ... ({len(prefixes)-20} more)]"
                # Count unique samples and repetitions
                unique_prefixes = len(set(prefixes))
                print(f"    Worker {worker_id:2d}: {len(prefixes)} total, {unique_prefixes} unique → {prefix_str}")

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
        
        # Add note about KV-aware routing benefits
        print("\n  Note: KV-aware routing benefits increase with:")
        print("    - Larger models (slower inference → cache reuse matters more)")
        print("    - Longer prompts with shared prefixes (more cache to reuse)")
        print("    - GRPO-style workloads (multiple generations per prompt)")

    print("=" * 70)


def build_benchmark_configs(args: argparse.Namespace) -> List[RouterBenchmarkConfig]:
    """Build benchmark configurations based on routing mode."""
    
    if args.use_integrated_routing:
        # Integrated mode: KV-Aware Integrated + Round-Robin Integrated
        # Uses RoutedVllmWorkerGroup with batch routing APIs
        configs = [
            RouterBenchmarkConfig(
                name="KV-Aware Routing (Integrated)",
                route_type="kv_integrated",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
                is_integrated=True,
                shuffle_batch=args.shuffle_batch,
            ),
            RouterBenchmarkConfig(
                name="Round-Robin Routing (Integrated)",
                route_type="round_robin",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
                is_integrated=True,
                shuffle_batch=args.shuffle_batch,
            ),
        ]
    elif args.use_manual_routing:
        # Manual mode: KV-Aware Manual + Round-Robin Manual
        # Uses separate VllmGeneration + KvRouter with manual coordination
        configs = [
            RouterBenchmarkConfig(
                name="KV-Aware Routing (Manual)",
                route_type="kv_manual",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
                is_integrated=False,
                shuffle_batch=args.shuffle_batch,
            ),
            RouterBenchmarkConfig(
                name="Round-Robin Routing (Manual)",
                route_type="round_robin",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
                is_integrated=False,
                shuffle_batch=args.shuffle_batch,
            ),
        ]
    else:
        # Simple mode (default): KV-Aware Simple (sechoi's approach)
        # Uses VllmGeneration with router_cfg - routing is completely automatic
        configs = [
            RouterBenchmarkConfig(
                name="KV-Aware Routing (Simple)",
                route_type="kv_simple",
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                generation_max_tokens=args.generation_max_tokens,
                is_integrated=False,
                shuffle_batch=args.shuffle_batch,
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
        default="agentica-org/DeepScaleR-1.5B-Preview",
        help="HuggingFace model name (default: DeepScaleR-1.5B-Preview for math reasoning)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenMathInstruct-2",
        choices=["OpenMathInstruct-2", "DeepScaler", "DAPOMath17K", "synthetic"],
        help="Dataset to use for benchmarking. OpenMathInstruct-2 (default), DeepScaler, DAPOMath17K (includes AIME 2024 validation), or synthetic"
    )
    parser.add_argument(
        "--num-synthetic-prefixes",
        type=int,
        default=1,
        help="For synthetic dataset: number of distinct shared prefixes to generate. If > 1, creates cache contention scenarios."
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="examples/prompts/cot.txt",
        help="Path to prompt template file (default: examples/prompts/cot.txt)"
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
    parser.add_argument("--num-generations-per-prompt", type=int, default=1, 
                        help="Number of times to repeat each prompt (like GRPO's num_generations_per_prompt)")
    parser.add_argument("--sequential", action="store_true", default=False,
                        help="Process requests sequentially (one at a time) instead of in parallel batches. "
                             "Better for observing KV cache indexing behavior.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--shuffle-batch",
        action="store_true",
        default=False,
        help="Shuffle request batch before routing to prevent alignment artifacts (useful for round-robin experiments)"
    )
    
    # Routing mode selection (mutually exclusive)
    routing_group = parser.add_mutually_exclusive_group()
    routing_group.add_argument(
        "--use-manual-routing",
        action="store_true",
        default=False,
        help="Use manual routing mode: separate VllmGeneration + KvRouter with manual coordination"
    )
    routing_group.add_argument(
        "--use-integrated-routing",
        action="store_true",
        default=False,
        help="Use integrated routing mode: RoutedVllmWorkerGroup with batch routing APIs"
    )
    # Default (neither flag): Simple mode with VllmGeneration + router_cfg (sechoi's approach)
    
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    configs = build_benchmark_configs(args)

    samples_needed = max(config.total_requests for config in configs)
    
    # Allow dataset reuse (looping) if we have enough samples for at least one full batch
    if args.dataset_samples is not None:
        if args.dataset_samples < args.batch_size:
             raise ValueError(
                f"Provided --dataset-samples ({args.dataset_samples}) is smaller than batch size ({args.batch_size})."
            )
        logger.info(f"Dataset reuse enabled: {args.dataset_samples} samples provided, {samples_needed} total requests needed.")
    
    max_samples = args.dataset_samples or samples_needed
    
    # Load dataset based on selection (following run_grpo_math.py pattern)
    if args.dataset == "synthetic":
        logger.info("Using synthetic dataset")
        benchmark_data = create_temp_benchmark_dataset(
            max_samples, 
            args.max_seq_length, 
            args.seed,
            num_prefixes=args.num_synthetic_prefixes
        )
        tokenizer = None
    else:
        # Setup data using repo's dataset infrastructure - same pattern as run_grpo_math.py
        processed_dataset, tokenizer = setup_benchmark_data(
            dataset_name=args.dataset,
            tokenizer_name=args.model,
            max_samples=max_samples,
            seed=args.seed,
            prompt_file=args.prompt_file
        )
        # Create dataloader like GRPO does (see grpo.py line 257-264)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            processed_dataset,
            batch_size=32,  # Process in batches
            shuffle=False,
            collate_fn=rl_collate_fn,
            drop_last=False,
        )
        
        # Extract data from dataloader in batches - following GRPO pattern
        all_input_ids = []
        all_input_lengths = []
        
        logger.info("Extracting tokenized data from dataloader...")
        for batch in dataloader:
            # Repeat batch items like GRPO does (grpo.py line 1038-1042)
            if args.num_generations_per_prompt > 1:
                repeated_batch = batch.repeat_interleave(args.num_generations_per_prompt)
                print(f"{repeated_batch=}")
            else:
                repeated_batch = batch
            
            # Extract individual samples from message_log (without batch padding yet)
            for message_log in repeated_batch["message_log"]:
                flat_messages = message_log_to_flat_messages(message_log)
                input_ids = flat_messages['token_ids']
                
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.numpy()
                
                all_input_ids.append(input_ids)
                all_input_lengths.append(len(input_ids))
        
        # Now pad to global max length (same as before)
        max_length = max(all_input_lengths)
        num_samples = len(all_input_ids)
        
        input_ids_array = np.zeros((num_samples, max_length), dtype=np.int64)
        for i, tokens in enumerate(all_input_ids):
            input_ids_array[i, :len(tokens)] = tokens
        
        input_lengths_array = np.array(all_input_lengths, dtype=np.int64)
        
        benchmark_data = {
            "input_ids": input_ids_array,
            "input_lengths": input_lengths_array,
        }
        
        logger.info(f"✓ Processed {len(input_ids_array)} samples from dataset")
        if args.num_generations_per_prompt > 1:
            logger.info(f"  Repeated {args.num_generations_per_prompt}x per prompt (like GRPO)")
            unique_prompts = len(input_ids_array) // args.num_generations_per_prompt
            logger.info(f"  Original prompts: {unique_prompts}")
            
            # Debug: Check if repetitions are actually identical
            first_two_match = np.array_equal(input_ids_array[0], input_ids_array[1])
            if args.num_generations_per_prompt > 1:
                expected_match_idx = args.num_generations_per_prompt
                if expected_match_idx < len(input_ids_array):
                    first_and_repeat_match = np.array_equal(input_ids_array[0], input_ids_array[expected_match_idx])
                    logger.info(f"  Repetition check: sample[0] vs sample[1]: {'SAME' if first_two_match else 'DIFFERENT'}")
                    logger.info(f"  Repetition check: sample[0] vs sample[{expected_match_idx}]: {'SAME' if first_and_repeat_match else 'DIFFERENT'}")
        logger.info(f"  Max prompt length: {input_lengths_array.max()} tokens")
        logger.info(f"  Avg prompt length: {input_lengths_array.mean():.1f} tokens")

    print("Ray-Based Router Benchmarking Suite")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Dataset:         {args.dataset}")
    total_samples = len(benchmark_data['input_ids'])
    print(f"Samples:         {total_samples}", end="")
    if args.num_generations_per_prompt > 1:
        print(f" ({total_samples // args.num_generations_per_prompt} unique × {args.num_generations_per_prompt} repeats)")
    else:
        print()
    print(f"Block size:      {args.block_size}")
    print(f"Cluster:         {args.num_nodes} node(s) × {args.gpus_per_node} GPU(s) = {args.num_nodes * args.gpus_per_node} workers")
    print(f"Routing mode:    {'Integrated (RoutedVllmWorkerGroup)' if args.use_integrated_routing else 'Manual (VllmGeneration + KvRouter)'}")
    print(f"Warmup iters:    {args.warmup_iterations} (not counted in metrics)")
    print(f"Counted iters:   {args.num_iterations} (counted in metrics)")
    print(f"Benchmark configs: {len(configs)}")
    print(f"Note: Workers/router recreated for each config (following standalone pattern)")
    print("=" * 70)

    results = await execute_benchmarks(configs, benchmark_data, args)
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
