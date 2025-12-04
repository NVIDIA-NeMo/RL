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
Routed Worker Group for vLLM with integrated KV-aware routing.

This module provides an ADVANCED alternative to the simple VllmGeneration integration.
It offers:
- Explicit router lifecycle management
- Batch routing APIs
- Fine-grained control over routing decisions
- Better separation of concerns for testing

For simple use cases, use VllmGeneration with router_cfg instead.
For advanced routing control and benchmarking, use RoutedVllmWorkerGroup.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import ray

from nemo_rl.distributed.worker_groups import (
    RayWorkerBuilder,
    RayWorkerGroup,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.dynamo.standalone_router import KvRouter

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for the KV-aware router.
    
    Attributes:
        block_size: Size of KV cache blocks (must match vLLM config)
        base_kv_events_port: Base port for ZMQ KV event publishers (one per worker)
        base_metrics_port: Base port for ZMQ metrics publishers (one per worker)
        enabled: Whether routing is enabled (default: True)
    """
    block_size: int = 64
    base_kv_events_port: int = 5557
    base_metrics_port: int = 5657
    enabled: bool = True


class RoutedVllmWorkerGroup(RayWorkerGroup):
    """Worker group with integrated KV-aware dynamic routing for vLLM workers.
    
    This is an ADVANCED alternative to VllmGeneration's built-in routing.
    
    **When to use this vs VllmGeneration with router_cfg:**
    
    Use VllmGeneration + router_cfg for:
    - Simple, transparent integration
    - Minimal code changes
    - Standard GRPO training pipelines
    
    Use RoutedVllmWorkerGroup for:
    - Explicit router control
    - Batch routing APIs (get_best_worker_ids_batch)
    - Custom routing strategies
    - Performance benchmarking
    - Testing and debugging routing logic
    
    The router continuously monitors worker state via ZMQ and maintains a RadixTree
    index of cached prefixes to make optimal routing decisions.
    
    Example:
        ```python
        from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
        from nemo_rl.distributed.worker_groups import RayWorkerBuilder
        from nemo_rl.models.generation.dynamo.routed_worker_group import (
            RoutedVllmWorkerGroup, RouterConfig
        )
        
        # Create cluster and worker builder
        cluster = RayVirtualCluster(num_nodes=1, num_gpus_per_node=4)
        worker_builder = RayWorkerBuilder(
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
            vllm_config
        )
        
        # Create routed worker group
        router_config = RouterConfig(block_size=64)
        worker_group = RoutedVllmWorkerGroup(
            cluster=cluster,
            remote_worker_builder=worker_builder,
            router_config=router_config,
            name_prefix="vllm_routed"
        )
        
        # Start router background tasks
        await worker_group.start_router()
        
        # Generate with routing
        results = await worker_group.generate_with_routing(
            method_name="generate",
            local_hashes=[hash1, hash2, ...],
            num_tokens=100,
            data=generation_data,
            greedy=False
        )
        ```
    """
    
    def __init__(
        self,
        cluster: RayVirtualCluster,
        remote_worker_builder: RayWorkerBuilder,
        router_config: Optional[RouterConfig] = None,
        workers_per_node: Optional[Union[int, list[int]]] = None,
        name_prefix: str = "vllm_routed",
        bundle_indices_list: Optional[list[tuple[int, list[int]]]] = None,
        sharding_annotations: Optional[NamedSharding] = None,
        env_vars: dict[str, str] = {},
    ):
        """Initialize a routed worker group for vLLM inference.
        
        Args:
            cluster: RayVirtualCluster managing GPU resources
            remote_worker_builder: Builder for creating Ray actor workers
            router_config: Configuration for the KV-aware router. If None, uses defaults.
            workers_per_node: Number of workers per node (inherited from RayWorkerGroup)
            name_prefix: Prefix for worker names
            bundle_indices_list: Explicit bundle placement (inherited from RayWorkerGroup)
            sharding_annotations: Named sharding for parallelism (inherited from RayWorkerGroup)
            env_vars: Environment variables for workers
        """
        # Initialize parent RayWorkerGroup
        super().__init__(
            cluster=cluster,
            remote_worker_builder=remote_worker_builder,
            workers_per_node=workers_per_node,
            name_prefix=name_prefix,
            bundle_indices_list=bundle_indices_list,
            sharding_annotations=sharding_annotations,
            env_vars=env_vars,
        )
        
        # Initialize router configuration
        self.router_config = router_config or RouterConfig()
        self.router: Optional[KvRouter] = None
        self._router_started = False
        
        # Only initialize router if enabled
        if self.router_config.enabled:
            self._init_router()
        else:
            logger.info("Router is disabled. Requests will need manual worker selection.")
    
    def _init_router(self) -> None:
        """Initialize the KV-aware router instance."""
        try:
            self.router = KvRouter(
                block_size=self.router_config.block_size,
                num_workers=len(self._workers),
                base_kv_events_port=self.router_config.base_kv_events_port,
                base_metrics_port=self.router_config.base_metrics_port,
            )
            logger.info(
                f"KvRouter initialized with {len(self._workers)} workers, "
                f"block_size={self.router_config.block_size}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize KvRouter: {e}")
            raise
    
    async def start_router(self, enable_kv_indexer: bool = True) -> None:
        """Start the router's background monitoring tasks.
        
        This starts two async tasks:
        1. Load monitoring: Polls worker metrics (KV cache usage, queue depth)
        2. Index monitoring: Tracks KV cache events to maintain the RadixTree index (optional)
        
        Args:
            enable_kv_indexer: If False, skip KV event tracking (useful for round-robin routing)
        
        Must be called before using routing-aware methods like generate_with_routing().
        
        Raises:
            RuntimeError: If router is not enabled or already started
        """
        if not self.router_config.enabled:
            raise RuntimeError(
                "Cannot start router when routing is disabled. "
                "Set router_config.enabled=True."
            )
        
        if self._router_started:
            logger.warning("Router background tasks already started")
            return
        
        if self.router is None:
            raise RuntimeError("Router not initialized")
        
        try:
            await self.router.start_background_tasks(enable_kv_indexer=enable_kv_indexer)
            self._router_started = True
            mode_str = "KV-aware" if enable_kv_indexer else "round-robin"
            logger.info(f"Router background tasks started successfully ({mode_str} mode)")
        except Exception as e:
            logger.error(f"Failed to start router background tasks: {e}")
            raise
    
    async def get_best_worker_id(
        self, 
        local_hashes: list[int], 
        num_tokens: int
    ) -> int:
        """Select the best worker for a request using the KV-aware router.
        
        The router computes a score for each worker based on:
        - Prefix overlap (higher is better): fraction of cached blocks
        - KV cache usage (lower is better): current memory utilization
        - Queue depth (lower is better): number of waiting requests
        
        Score formula: logit = 2 * overlap - usage - normalized_waiting
        
        Args:
            local_hashes: Block hashes for the input prompt (computed from token IDs)
            num_tokens: Total number of tokens in the prompt
            
        Returns:
            int: Index of the best worker to handle this request
            
        Raises:
            RuntimeError: If router is not enabled or not started
            ValueError: If num_tokens <= 0
        """
        if not self.router_config.enabled:
            raise RuntimeError(
                "Routing is disabled. Use run_single_worker_single_data() "
                "to manually specify worker_id."
            )
        
        if not self._router_started:
            raise RuntimeError(
                "Router not started. Call await start_router() first."
            )
        
        if self.router is None:
            raise RuntimeError("Router not initialized")
        
        try:
            worker_id = await self.router.get_best_worker(local_hashes, num_tokens)
            logger.debug(
                f"Router selected worker {worker_id} for request "
                f"with {num_tokens} tokens and {len(local_hashes)} block hashes"
            )
            return worker_id
        except Exception as e:
            logger.error(f"Error selecting best worker: {e}")
            raise
    
    async def get_best_worker_ids_batch(
        self,
        batch_hashes: list[list[int]],
        batch_num_tokens: list[int]
    ) -> list[int]:
        """Select best workers for multiple requests in a single batch call.
        
        This is much more efficient than calling get_best_worker_id() repeatedly
        because it:
        1. Reduces function call overhead (1 call vs N calls)
        2. Allows the router to make batch-aware routing decisions
        3. Better handles concurrent request bursts
        
        Args:
            batch_hashes: List of block hashes for each request
            batch_num_tokens: List of token counts for each request
            
        Returns:
            list[int]: Worker IDs for each request (same order as input)
            
        Raises:
            RuntimeError: If router is not enabled or not started
            ValueError: If batch_hashes and batch_num_tokens have different lengths
            
        Example:
            ```python
            # Route 64 requests in one call instead of 64 individual calls
            worker_ids = await routed_worker_group.get_best_worker_ids_batch(
                batch_hashes=[hashes1, hashes2, ..., hashes64],
                batch_num_tokens=[len1, len2, ..., len64]
            )
            ```
        """
        if not self.router_config.enabled:
            raise RuntimeError(
                "Routing is disabled. Use run_single_worker_single_data() "
                "to manually specify worker_id."
            )
        
        if not self._router_started:
            raise RuntimeError(
                "Router not started. Call await start_router() first."
            )
        
        if self.router is None:
            raise RuntimeError("Router not initialized")
        
        if len(batch_hashes) != len(batch_num_tokens):
            raise ValueError(
                f"batch_hashes (len={len(batch_hashes)}) and batch_num_tokens "
                f"(len={len(batch_num_tokens)}) must have the same length"
            )
        
        try:
            # Route all requests in parallel
            # This matches the standalone HTTP router behavior where multiple
            # concurrent requests are handled simultaneously
            routing_tasks = [
                self.router.get_best_worker(local_hashes, num_tokens)
                for local_hashes, num_tokens in zip(batch_hashes, batch_num_tokens)
            ]
            worker_ids = await asyncio.gather(*routing_tasks)
            
            logger.debug(
                f"Batch routed {len(worker_ids)} requests: "
                f"worker distribution={dict(zip(*list(zip(*[(w, worker_ids.count(w)) for w in set(worker_ids)]))))}"
            )
            return list(worker_ids)
        except Exception as e:
            logger.error(f"Error in batch routing: {e}")
            raise
    
    async def get_worker_round_robin(
        self,
        local_hashes: list[int],
        num_tokens: int
    ) -> int:
        """Select a worker using round-robin load balancing (ignores KV cache).
        
        This is a simpler routing strategy that can be used for comparison
        or when prefix caching is not beneficial.
        
        Args:
            local_hashes: Block hashes (unused in round-robin)
            num_tokens: Number of tokens (unused in round-robin)
            
        Returns:
            int: Index of the worker selected by round-robin
            
        Raises:
            RuntimeError: If router is not enabled or not started
        """
        if not self.router_config.enabled:
            raise RuntimeError("Routing is disabled")
        
        if not self._router_started:
            raise RuntimeError("Router not started. Call await start_router() first.")
        
        if self.router is None:
            raise RuntimeError("Router not initialized")
        
        try:
            worker_id = await self.router.get_worker_round_robin(local_hashes, num_tokens)
            logger.debug(f"Round-robin selected worker {worker_id}")
            return worker_id
        except Exception as e:
            logger.error(f"Error in round-robin selection: {e}")
            raise
    
    async def generate_with_routing(
        self,
        method_name: str,
        local_hashes: list[int],
        num_tokens: int,
        use_round_robin: bool = False,
        **method_kwargs: Any,
    ) -> ray.ObjectRef:
        """Generate using automatic worker selection via KV-aware routing.
        
        This is the main entry point for routed inference. It:
        1. Selects the best worker based on KV cache state
        2. Dispatches the generation request to that worker
        3. Returns a Ray ObjectRef for the result
        
        Args:
            method_name: Name of the method to call on the worker (e.g., "generate")
            local_hashes: Block hashes computed from input token IDs
            num_tokens: Total number of tokens in the input
            use_round_robin: If True, use round-robin instead of KV-aware routing
            **method_kwargs: Arguments to pass to the worker method
            
        Returns:
            ray.ObjectRef: Future for the generation result
            
        Example:
            ```python
            from dynamo._core import compute_block_hash_for_seq_py
            
            # Compute block hashes from tokens
            tokens = [1, 2, 3, 4, 5, ...]
            local_hashes = compute_block_hash_for_seq_py(tokens, block_size=64)
            
            # Generate with routing
            result_ref = await worker_group.generate_with_routing(
                method_name="generate",
                local_hashes=local_hashes,
                num_tokens=len(tokens),
                data=generation_data,
                greedy=False
            )
            
            # Get result
            result = ray.get(result_ref)
            ```
        """
        # Select best worker
        if use_round_robin:
            best_worker_id = await self.get_worker_round_robin(local_hashes, num_tokens)
        else:
            best_worker_id = await self.get_best_worker_id(local_hashes, num_tokens)
        
        # Execute on selected worker
        logger.info(
            f"Dispatching {method_name} to worker {best_worker_id} "
            f"(tokens={num_tokens}, hashes={len(local_hashes)})"
        )
        
        return self.run_single_worker_single_data(
            method_name,
            best_worker_id,
            **method_kwargs
        )
    
    def shutdown(
        self,
        cleanup_method: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        force: bool = False,
    ) -> bool:
        """Shutdown the router and all workers.
        
        This is an async-compatible shutdown that handles both router and workers.
        Note: This is a synchronous wrapper. For async contexts, use shutdown_async().
        
        Args:
            cleanup_method: Optional cleanup method to call on workers
            timeout: Timeout for graceful shutdown
            force: Whether to force-kill workers
            
        Returns:
            bool: True if shutdown was successful
        """
        # Try to run async shutdown if in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, schedule the coroutine
            task = loop.create_task(self.shutdown_async(cleanup_method, timeout, force))
            return True  # Return immediately, let the task complete in background
        except RuntimeError:
            # No running loop, do synchronous shutdown
            if self.router and self._router_started:
                # Can't await in sync context, just log warning
                logger.warning(
                    "Shutdown called in sync context but router requires async shutdown. "
                    "Router may not shut down cleanly. Use shutdown_async() instead."
                )
            
            # Shutdown workers synchronously
            return super().shutdown(cleanup_method, timeout, force)
    
    async def shutdown_async(
        self,
        cleanup_method: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        force: bool = False,
    ) -> bool:
        """Async shutdown of router and workers.
        
        Properly shuts down:
        1. Router background tasks
        2. Router ZMQ sockets and context
        3. All worker actors
        
        Args:
            cleanup_method: Optional cleanup method to call on workers
            timeout: Timeout for graceful shutdown
            force: Whether to force-kill workers
            
        Returns:
            bool: True if shutdown was successful
        """
        success = True
        
        # Shutdown router first
        if self.router and self._router_started:
            try:
                logger.info("Shutting down KvRouter...")
                await self.router.shutdown()
                self._router_started = False
                logger.info("KvRouter shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down router: {e}")
                success = False
        
        # Shutdown workers
        try:
            logger.info("Shutting down workers...")
            worker_success = super().shutdown(cleanup_method, timeout, force)
            success = success and worker_success
            logger.info("Workers shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down workers: {e}")
            success = False
        
        return success

