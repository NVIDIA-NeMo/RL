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

"""Ray Compiled Graph support for high-performance distributed training.

This module provides Ray Compiled Graph infrastructure for optimized distributed execution.

IMPLEMENTATION (Wrapper Architecture):
--------------------------------------
✅ Uses a wrapper class pattern to solve Ray's ActorMethod signature hiding issue:

1. **NeMoRayWorkerWrapper**: Simple wrapper class that gets ray.remote() applied
   - train_compiled() explicitly defined → Ray can inspect signature ✅
   - execute_method() for delegating all other method calls to wrapped worker

2. **Ray Compiled Graph**: Calls worker.train_compiled.bind() directly
   - Works because train_compiled is a real method on the wrapper (not hidden by ActorMethod)

3. **Standard Ray calls**: Use worker.execute_method.remote(method_name, ...)
   - Forwards to wrapped worker's actual methods

ARCHITECTURE:
```
ray.remote()(NeMoRayWorkerWrapper).remote(worker_class, *args, **kwargs)
    ├─> train_compiled(train_input) ← Explicit method (Ray can inspect!) ✅
    │       └─> self.worker.train(...)
    └─> execute_method(name, *args, **kwargs) ← Generic delegation
            └─> getattr(self.worker, name)(*args, **kwargs)
```

vs. Previous (broken) approach:
```
ray.remote()(MegatronPolicyWorker).remote(*args, **kwargs)
    └─> train_compiled() ← Hidden by ActorMethod wrapper ❌
```
"""

import logging
from typing import Any, Optional

import ray
from ray.dag import InputNode, MultiOutputNode

from nemo_rl.distributed.named_sharding import NamedSharding

logger = logging.getLogger(__name__)

# Configure Ray's compiled DAG logger to suppress noisy teardown messages
# These INFO messages during cleanup are expected (workers shutting down) but verbose
_ray_dag_logger = logging.getLogger("ray.dag.compiled_dag_node")
_ray_dag_logger.setLevel(logging.WARNING)  # Suppress INFO, only show WARNING and above


class CompiledGraphExecutor:
    """Executor for Ray Compiled Graph with pipeline and tensor parallelism support.

    This class creates and executes a compiled DAG for distributed training:
    - Organizes workers into PP (pipeline parallel) and TP (tensor parallel) groups
    - Constructs a static DAG that includes all PP stages as nodes
    - Note: Megatron handles inter-stage PP communication internally via send/recv,
      while Ray orchestrates when each stage executes

    Example DAG structure (PP=2, TP=4):
        Ray DAG: Input -> [W0, W1, W2, W3] (PP stage 0)
                       -> [W4, W5, W6, W7] (PP stage 1) -> Outputs
        Where each PP stage has TP workers executing in SPMD fashion.
        Inter-stage tensors are communicated by Megatron, not through Ray DAG edges.
    """

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        sharding_annotations: NamedSharding,
        method_name: str = "train",
        dp_rank: int = 0,
        overlap_communication: bool = False,
    ):
        """Initialize the CompiledGraphExecutor for a single DP shard.

        Args:
            workers: List of Ray actor handles for all workers
            sharding_annotations: NamedSharding describing worker organization (PP, TP, CP, DP)
            method_name: Name of the method to call on workers (e.g., "train")
            dp_rank: Which DP shard this executor handles (default 0 for DP=1 case)
            overlap_communication: Overlap GPU compute and communication (experimental)
        """
        self.workers = workers
        self.sharding_annotations = sharding_annotations
        self.method_name = method_name
        self.dp_rank = dp_rank
        self.overlap_communication = overlap_communication

        # Extract parallelism dimensions
        self.pp_size = sharding_annotations.get_axis_size("pipeline_parallel")
        self.tp_size = sharding_annotations.get_axis_size("tensor_parallel")
        self.cp_size = sharding_annotations.get_axis_size("context_parallel")
        self.dp_size = sharding_annotations.get_axis_size("data_parallel")

        # Organize workers into PP-TP groups for this DP shard
        self.pp_tp_workers = self._organize_workers_pp_tp()

        # Build and compile the DAG
        try:
            self.compiled_dag = self._build_and_compile_dag()
        except Exception as e:
            logger.error(
                f"DAG compilation failed for DP shard {dp_rank}, PP={self.pp_size}, "
                f"TP={self.tp_size}, CP={self.cp_size}, method={method_name}. "
                f"Error: {e}"
            )
            raise

    def _organize_workers_pp_tp(self) -> list[list[ray.actor.ActorHandle]]:
        """Organize workers into PP stages with TP groups.

        This organizes workers WITHIN A SINGLE DP SHARD.
        For training with DP>1, we'll need to handle multiple DP shards separately.

        Returns:
            List of PP stages, where each stage contains TP×CP workers.

        Example: For PP=2, TP=4, CP=1 (within one DP shard):
            [[worker_0, worker_1, worker_2, worker_3],     # PP stage 0
             [worker_4, worker_5, worker_6, worker_7]]     # PP stage 1

        Note: This only handles ONE DP shard. For DP>1, call this per DP group.
        """
        pp_tp_workers = []
        # Organize workers for the specified DP shard
        # self.dp_rank is set during initialization
        for pp_rank in range(self.pp_size):
            # For each PP stage, collect all TP×CP workers in this DP shard
            workers_for_pp_stage = []

            # Organize by CP -> TP for proper worker ordering
            # CP and TP workers execute in SPMD fashion (same input, different tensor slices)
            for cp_rank in range(self.cp_size):
                for tp_rank in range(self.tp_size):
                    # Get the global worker rank for this coordinate
                    worker_rank = self.sharding_annotations.get_ranks(
                        pipeline_parallel=pp_rank,
                        data_parallel=self.dp_rank,
                        context_parallel=cp_rank,
                        tensor_parallel=tp_rank,
                    )
                    if isinstance(worker_rank, int):
                        workers_for_pp_stage.append(self.workers[worker_rank])
                    else:
                        raise ValueError(
                            f"Expected single worker rank, got {worker_rank} for "
                            f"PP={pp_rank}, DP={self.dp_rank}, CP={cp_rank}, TP={tp_rank}"
                        )

            pp_tp_workers.append(workers_for_pp_stage)

        return pp_tp_workers

    def _build_and_compile_dag(self) -> Any:
        """Build and compile the Ray DAG for distributed execution.

        Creates a DAG for efficient distributed training:
        1. Input node receives data (as dict with 'data' and 'common_kwargs')
        2. All PP stage workers are included as DAG nodes
        3. Each PP stage processes inputs in SPMD fashion (TP workers)
        4. Megatron handles inter-stage PP communication internally (not Ray)
        5. Final PP stage produces outputs that Ray collects
        6. DAG is compiled for optimized execution

        Returns:
            Compiled Ray DAG ready for execution
        """
        # Wrapper architecture:
        # ====================
        # We apply ray.remote() to NeMoRayWorkerWrapper (simple class) instead of
        # the complex worker classes. The train_compiled() method is defined on the
        # wrapper, allowing Ray's compiled graph to properly inspect its signature.

        logger.info(f"Attempting DAG compilation (PP={self.pp_size})...")

        with InputNode() as input_dict:
            # NeMo-RL's train() method expects keyword arguments:
            #   train(data=..., loss_fn=..., eval_mode=..., gbs=..., mbs=...)
            outputs = []

            # Loop through all PP stages
            for pp_stage_idx in range(self.pp_size):
                tp_cp_workers = self.pp_tp_workers[pp_stage_idx]

                # All workers (across all PP stages and TP×CP groups) get the SAME input_dict from Ray.
                # TP/CP workers execute in SPMD fashion - same code, different tensor slices.
                # PP stages also get the same Ray input because Megatron handles inter-stage flow internally.
                for idx, worker in enumerate(tp_cp_workers):
                    if self.method_name == "train":
                        dag_node = worker.train_compiled.bind(input_dict)
                    else:
                        # For other methods, look for a _compiled variant
                        method = getattr(worker, f"{self.method_name}_compiled", None)
                        if method is not None:
                            dag_node = method.bind(input_dict)
                        else:
                            raise NotImplementedError(
                                f"Method '{self.method_name}' does not have a '_compiled' variant. "
                                "Add a wrapper method with signature: def {self.method_name}_compiled(self, train_input: dict)"
                            )
                    outputs.append(dag_node)

            forward_dag = MultiOutputNode(outputs)

        try:
            compiled_dag = forward_dag.experimental_compile(
                _overlap_gpu_communication=self.overlap_communication,
            )
            logger.info("✅ DAG compilation successful!")
        except TypeError as e:
            logger.info(
                f"DAG compilation failed with TypeError (expected due to ActorMethod wrapper): {e}"
            )
            raise

        return compiled_dag

    def execute(self, *args, **kwargs) -> Any:
        """Execute the compiled DAG with the given inputs.

        Args:
            *args: Positional arguments to pass to the DAG input
            **kwargs: Keyword arguments to pass to the DAG input

        Returns:
            The execution result reference
        """
        # For compiled graphs, we typically pass a single input that gets distributed
        # The DAG handles routing to all workers
        if args:
            input_data = args[0]
        else:
            input_data = kwargs

        # Execute the compiled DAG synchronously
        return self.compiled_dag.execute(input_data)

    def teardown(self, suppress_logging: bool = True):
        """Clean up the compiled DAG resources.

        Args:
            suppress_logging: If True, suppress Ray's internal logging. Set to False
                when called from a parent teardown that's already suppressing.
        """
        if hasattr(self.compiled_dag, "teardown"):
            if suppress_logging:
                # Temporarily suppress ALL Ray internal logging during teardown
                # to avoid noisy INFO/ERROR messages (workers already shutting down)
                import logging

                ray_loggers = [
                    logging.getLogger("ray"),
                    logging.getLogger("ray.dag"),
                    logging.getLogger("ray.dag.compiled_dag_node"),
                ]
                original_levels = [lg.level for lg in ray_loggers]
                for lg in ray_loggers:
                    lg.setLevel(logging.CRITICAL + 1)  # Suppress everything

            try:
                self.compiled_dag.teardown()
            except Exception as e:
                # Suppress expected teardown errors (workers already shutting down)
                # These are harmless ActorDiedError during cleanup
                logger.debug(f"Compiled DAG teardown warning (expected): {e}")

            if suppress_logging:
                # Restore original logging levels
                for lg, level in zip(ray_loggers, original_levels):
                    lg.setLevel(level)


class MultiDPCompiledGraphExecutor:
    """Executor for Ray Compiled Graph with multiple DP shards.

    This class creates and manages ONE compiled DAG per DP shard.
    Each DP shard gets different data and executes independently.

    Example (PP=2, TP=4, DP=2):
        DP_shard_0: Input_0 -> [W0,W1,W2,W3] -> [W4,W5,W6,W7] -> Output_0
        DP_shard_1: Input_1 -> [W8,W9,W10,W11] -> [W12,W13,W14,W15] -> Output_1
    """

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        sharding_annotations: NamedSharding,
        method_name: str,
        overlap_communication: bool = False,
    ):
        """Initialize multi-DP compiled graph executor.

        Args:
            workers: List of all Ray actor handles
            sharding_annotations: Sharding configuration
            method_name: Name of the method to execute
            overlap_communication: Overlap communication for all DAGs
        """
        self.workers = workers
        self.sharding_annotations = sharding_annotations
        self.method_name = method_name

        # Get DP size
        self.dp_size = sharding_annotations.get_axis_size("data_parallel")
        self.pp_size = sharding_annotations.get_axis_size("pipeline_parallel")

        # Create one CompiledGraphExecutor per DP shard
        self.dp_executors: list[CompiledGraphExecutor] = []
        for dp_rank in range(self.dp_size):
            executor = CompiledGraphExecutor(
                workers=workers,
                sharding_annotations=sharding_annotations,
                method_name=method_name,
                dp_rank=dp_rank,
                overlap_communication=overlap_communication,
            )
            self.dp_executors.append(executor)

        logger.info(
            f"🚀 MultiDP Compiled Graph created: {self.dp_size} independent DAGs "
            f"(PP={self.pp_size}, method={method_name})"
        )

    def execute(self, sharded_data: dict[int, Any]) -> list[Any]:
        """Execute all DP shards' DAGs with their respective data.

        Args:
            sharded_data: Dictionary mapping dp_rank -> input_dict for that shard
                         input_dict = {"data": ..., "loss_fn": ..., ...}

        Returns:
            Flattened list of results from all DP shards' workers
        """
        all_refs = []
        for dp_rank, executor in enumerate(self.dp_executors):
            input_dict = sharded_data[dp_rank]
            refs_for_this_dp = executor.execute(input_dict)
            # MultiOutputNode typically returns a list, but handle both cases defensively
            if isinstance(refs_for_this_dp, list):
                all_refs.extend(refs_for_this_dp)
            else:
                all_refs.append(refs_for_this_dp)

        return all_refs

    def teardown(self, suppress_logging: bool = True):
        """Clean up all DP shards' compiled DAG resources.

        Args:
            suppress_logging: If True, suppress Ray's internal logging. Set to False
                when called from a parent teardown that's already suppressing.
        """
        if suppress_logging:
            # Suppress ALL Ray internal logging ONCE for all nested teardowns
            import logging

            ray_loggers = [
                logging.getLogger("ray"),
                logging.getLogger("ray.dag"),
                logging.getLogger("ray.dag.compiled_dag_node"),
            ]
            original_levels = [lg.level for lg in ray_loggers]
            for lg in ray_loggers:
                lg.setLevel(logging.CRITICAL + 1)  # Suppress everything

        try:
            for executor in self.dp_executors:
                try:
                    # Don't let nested teardown restore logging (suppress_logging=False)
                    executor.teardown(suppress_logging=False)
                except Exception as e:
                    # Suppress expected teardown errors during cleanup
                    logger.debug(f"DP executor teardown warning (expected): {e}")
            self.dp_executors.clear()
        finally:
            if suppress_logging:
                # Restore original logging levels once at the end
                for lg, level in zip(ray_loggers, original_levels):
                    lg.setLevel(level)


def should_use_compiled_graph(config: Optional[dict[str, Any]]) -> bool:
    """Check if Ray Compiled Graph should be used.

    Args:
        config: RCG config dict from policy config, or None if not configured.

    Returns:
        True if compiled graph should be used, False otherwise
    """
    if config is None:
        return False
    return config.get("enabled", False)


def get_compiled_graph_config(
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Get configuration for Ray Compiled Graph.

    Args:
        config: RCG config dict from policy config, or None if not configured.

    Returns:
        Dictionary with compiled graph configuration options (all defaults if config is None)
    """
    if config is None:
        # Return defaults when RCG is not configured
        return {
            "enabled": False,
            "warmup_seq_len": None,
            "warmup_gbs": None,
            "overlap_communication": False,
        }

    # Use provided config with defaults
    return {
        "enabled": config.get("enabled", False),
        "warmup_seq_len": config.get("warmup_seq_len", None),
        "warmup_gbs": config.get("warmup_gbs", None),
        "overlap_communication": config.get("overlap_communication", False),
    }


class CompiledGraphWorkerGroup:
    """Wrapper around RayWorkerGroup that uses compiled graphs when enabled.

    This class provides a drop-in replacement for standard RayWorkerGroup method calls,
    but uses Ray Compiled Graph for improved performance when enabled.
    """

    def __init__(
        self,
        worker_group: "RayWorkerGroup",  # type: ignore # noqa: F821
        compiled_graph_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the compiled graph worker group wrapper.

        Args:
            worker_group: The underlying RayWorkerGroup to wrap
            compiled_graph_config: Configuration for compiled graph (or None to use defaults)
        """
        self.worker_group = worker_group
        self.config = get_compiled_graph_config(compiled_graph_config)
        self.compiled_executors: dict[
            str, CompiledGraphExecutor | MultiDPCompiledGraphExecutor
        ] = {}

        if self.config["enabled"]:
            logger.info("Ray Compiled Graph is ENABLED for this worker group")
        else:
            logger.info(
                "Ray Compiled Graph is DISABLED, using standard Ray remote calls"
            )

    def _get_or_create_executor(
        self, method_name: str
    ) -> CompiledGraphExecutor | MultiDPCompiledGraphExecutor | None:
        """Get or create a compiled graph executor for a method.

        Args:
            method_name: Name of the method to execute

        Returns:
            CompiledGraphExecutor or MultiDPCompiledGraphExecutor if compiled graphs are enabled, None otherwise
        """
        if not self.config["enabled"]:
            return None

        if method_name not in self.compiled_executors:
            # Create a new compiled executor for this method
            try:
                executor = CompiledGraphExecutor(
                    workers=self.worker_group._workers,
                    sharding_annotations=self.worker_group.sharding_annotations,
                    method_name=method_name,
                    overlap_communication=self.config["overlap_communication"],
                )
                self.compiled_executors[method_name] = executor
                logger.info(
                    f"✅ Created compiled graph executor for method '{method_name}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to create compiled graph executor for '{method_name}': {e}"
                )
                logger.warning("Falling back to standard Ray remote calls")
                return None

        return self.compiled_executors[method_name]

    def run_all_workers_sharded_data(
        self,
        method_name: str,
        *args,
        in_sharded_axes: list[str] | None = None,
        replicate_on_axes: list[str] | None = None,
        output_is_replicated: list[str] | None = None,
        make_dummy_calls_to_free_axes: bool = False,
        common_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """Run a method on all workers with sharded data.

        This method uses compiled graphs when:
        1. Compiled graphs are enabled
        2. Sharding annotations are available (works with any PP/TP/CP/DP configuration)

        Implementation:
        - Creates a compiled DAG that includes all PP×TP×CP workers
        - Ray orchestrates execution; Megatron handles inter-stage PP communication
        - For DP>1: Each DP group gets different data and executes independently
        - TP/CP execute in SPMD fashion (same input, different tensor slices)

        Args:
            method_name: Name of the method to call on workers
            in_sharded_axes: Axes along which data is sharded
            replicate_on_axes: Axes along which data is replicated
            output_is_replicated: Axes along which output is replicated
            make_dummy_calls_to_free_axes: Whether to make dummy calls to free axes
            common_kwargs: Common keyword arguments for all workers
            **kwargs: Sharded keyword arguments

        Returns:
            MultiWorkerFuture containing the execution results
        """
        # Get parallelism info for logging
        pp_size = (
            self.worker_group.sharding_annotations.get_axis_size("pipeline_parallel")
            if self.worker_group.sharding_annotations
            else 1
        )
        dp_size = (
            self.worker_group.sharding_annotations.get_axis_size("data_parallel")
            if self.worker_group.sharding_annotations
            else 1
        )

        # Check if we should use compiled graph
        should_use_compiled = (
            self.config["enabled"]
            and self.worker_group.sharding_annotations is not None
        )

        if should_use_compiled:
            # Use print to ensure visibility (logger.info might be filtered)
            print(
                f"🚀 Executing '{method_name}' with Ray Compiled Graph (PP={pp_size}, DP={dp_size})"
            )
            logger.info(
                f"🚀 Executing '{method_name}' with Ray Compiled Graph (PP={pp_size}, DP={dp_size})"
            )

        if not should_use_compiled:
            # Fall back to standard implementation
            print(
                f"⚠️  Compiled graph disabled for '{method_name}' (enabled={self.config['enabled']}, has_sharding={self.worker_group.sharding_annotations is not None})"
            )
            return self.worker_group.run_all_workers_sharded_data(
                method_name,
                *args,
                in_sharded_axes=in_sharded_axes,
                replicate_on_axes=replicate_on_axes,
                output_is_replicated=output_is_replicated,
                make_dummy_calls_to_free_axes=make_dummy_calls_to_free_axes,
                common_kwargs=common_kwargs,
                **kwargs,
            )

        # ========================================
        # USE RAY COMPILED GRAPH!
        # TP/CP/PP communication happens inside workers (Megatron), Ray orchestrates execution
        # ========================================

        # Build compiled DAG on first call
        executor_key = f"{method_name}_compiled"
        if executor_key not in self.compiled_executors:
            logger.info(f"Building compiled DAG for '{method_name}' (first call)...")

            try:
                if dp_size == 1:
                    # Single DP: create one DAG
                    executor = CompiledGraphExecutor(
                        workers=self.worker_group._workers,
                        sharding_annotations=self.worker_group.sharding_annotations,
                        method_name=method_name,
                        dp_rank=0,
                        overlap_communication=self.config["overlap_communication"],
                    )
                else:
                    # DP>1: Create one DAG per DP shard (each with its own PP×TP×CP sub-grid)
                    logger.info(
                        f"Creating {dp_size} independent DAGs (one per DP shard)..."
                    )
                    executor = MultiDPCompiledGraphExecutor(
                        workers=self.worker_group._workers,
                        sharding_annotations=self.worker_group.sharding_annotations,
                        method_name=method_name,
                        overlap_communication=self.config["overlap_communication"],
                    )

                self.compiled_executors[executor_key] = executor
                logger.info(f"✅ Compiled DAG ready for '{method_name}'")
            except Exception as e:
                logger.error(f"Failed to build compiled DAG for '{method_name}': {e}")
                logger.warning("Falling back to standard Ray execution")
                return self.worker_group.run_all_workers_sharded_data(
                    method_name,
                    *args,
                    in_sharded_axes=in_sharded_axes,
                    replicate_on_axes=replicate_on_axes,
                    output_is_replicated=output_is_replicated,
                    make_dummy_calls_to_free_axes=make_dummy_calls_to_free_axes,
                    common_kwargs=common_kwargs,
                    **kwargs,
                )

        executor = self.compiled_executors[executor_key]

        # Extract sharded data from kwargs
        if "data" not in kwargs:
            logger.error("No 'data' argument found for compiled graph execution")
            return self.worker_group.run_all_workers_sharded_data(
                method_name,
                *args,
                in_sharded_axes=in_sharded_axes,
                replicate_on_axes=replicate_on_axes,
                output_is_replicated=output_is_replicated,
                make_dummy_calls_to_free_axes=make_dummy_calls_to_free_axes,
                common_kwargs=common_kwargs,
                **kwargs,
            )

        sharded_data_dict = kwargs["data"]  # {dp_rank: data_for_that_rank}

        if dp_size == 1:
            # Single DP: extract data for dp_rank=0
            if isinstance(sharded_data_dict, dict) and 0 in sharded_data_dict:
                data_for_workers = sharded_data_dict[0]
            elif isinstance(sharded_data_dict, list):
                data_for_workers = (
                    sharded_data_dict[0]
                    if len(sharded_data_dict) == 1
                    else sharded_data_dict
                )
            else:
                logger.warning(
                    f"Expected dict with key 0 or list for DP=1, got: {type(sharded_data_dict)}"
                )
                data_for_workers = sharded_data_dict

            # All workers (PP/TP/CP) get same input; Megatron handles PP inter-stage flow
            input_dict = {"data": data_for_workers}
            if common_kwargs:
                input_dict.update(common_kwargs)
            refs = executor.execute(input_dict)

            # Compute return_from_workers for result deduplication based on output_is_replicated
            if output_is_replicated is None:
                output_is_replicated = []

            return_from_workers = []
            for worker_idx in range(len(self.worker_group._workers)):
                worker_coords = (
                    self.worker_group.sharding_annotations.get_worker_coords(worker_idx)
                )
                return_from_this_worker = True
                for axis in output_is_replicated:
                    if axis in worker_coords and worker_coords[axis] != 0:
                        return_from_this_worker = False
                        break
                if return_from_this_worker:
                    return_from_workers.append(worker_idx)

            # Wrap in MultiWorkerFuture
            from nemo_rl.distributed.worker_groups import MultiWorkerFuture

            return MultiWorkerFuture(
                futures=refs if isinstance(refs, list) else [refs],
                return_from_workers=return_from_workers,
                called_workers=list(range(len(refs))),  # All workers were called
            )
        else:
            # DP>1: Execute one DAG per DP shard with shard-specific data
            if isinstance(sharded_data_dict, list):
                if len(sharded_data_dict) != dp_size:
                    logger.error(
                        f"Data list size {len(sharded_data_dict)} doesn't match DP size {dp_size}"
                    )
                    raise ValueError(
                        f"Expected {dp_size} data elements for DP={dp_size}, got {len(sharded_data_dict)}"
                    )
                sharded_data_dict = {i: sharded_data_dict[i] for i in range(dp_size)}

            # Build input dict for each DP shard
            input_dicts_per_dp = {}
            for dp_rank in range(dp_size):
                if dp_rank not in sharded_data_dict:
                    logger.error(
                        f"Missing data for DP rank {dp_rank}. Available keys: {list(sharded_data_dict.keys())}"
                    )
                    raise ValueError(f"Missing data for DP rank {dp_rank}")

                input_dict = {"data": sharded_data_dict[dp_rank]}
                if common_kwargs:
                    input_dict.update(common_kwargs)
                input_dicts_per_dp[dp_rank] = input_dict

            refs = executor.execute(input_dicts_per_dp)

            # Compute return_from_workers for result deduplication
            # Refs are ordered by DP (major) then PP->CP->TP within each DP shard
            if output_is_replicated is None:
                output_is_replicated = []

            return_from_workers = []
            # Compute which ref indices to return (refs are in DP-major order)
            pp_size = self.worker_group.sharding_annotations.get_axis_size(
                "pipeline_parallel"
            )
            cp_size = self.worker_group.sharding_annotations.get_axis_size(
                "context_parallel"
            )
            tp_size = self.worker_group.sharding_annotations.get_axis_size(
                "tensor_parallel"
            )

            ref_idx = 0
            for dp_rank in range(dp_size):
                for pp_rank in range(pp_size):
                    for cp_rank in range(cp_size):
                        for tp_rank in range(tp_size):
                            # Only return from rank 0 of replicated axes
                            should_return = (
                                ("context_parallel" not in output_is_replicated or cp_rank == 0)
                                and ("tensor_parallel" not in output_is_replicated or tp_rank == 0)
                                and ("pipeline_parallel" not in output_is_replicated or pp_rank == 0)
                            )
                            if should_return:
                                return_from_workers.append(ref_idx)
                            ref_idx += 1

            # Wrap in MultiWorkerFuture
            from nemo_rl.distributed.worker_groups import MultiWorkerFuture

            return MultiWorkerFuture(
                futures=refs if isinstance(refs, list) else [refs],
                return_from_workers=return_from_workers,
                called_workers=list(range(len(refs))),  # All workers were called
            )

    def run_method_compiled(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """Run a method using compiled graph if enabled, otherwise fall back to standard.

        Args:
            method_name: Name of the method to call on workers
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Result of the method execution
        """
        executor = self._get_or_create_executor(method_name)

        if executor is not None:
            # Use compiled graph for execution
            return executor.execute(*args, **kwargs)
        else:
            # Fall back to standard worker group execution
            return self.worker_group.run_all_workers(method_name, *args, **kwargs)

    def teardown(self):
        """Clean up all compiled graph resources."""
        # Suppress ALL Ray internal logging ONCE for all nested teardowns
        import logging

        ray_loggers = [
            logging.getLogger("ray"),
            logging.getLogger("ray.dag"),
            logging.getLogger("ray.dag.compiled_dag_node"),
        ]
        original_levels = [lg.level for lg in ray_loggers]
        for lg in ray_loggers:
            lg.setLevel(logging.CRITICAL + 1)  # Suppress everything
        try:
            for executor in self.compiled_executors.values():
                try:
                    # Don't let nested teardowns restore logging (pass suppress_logging=False)
                    # Both CompiledGraphExecutor and MultiDPCompiledGraphExecutor support this
                    if hasattr(executor, "teardown"):
                        executor.teardown(suppress_logging=False)
                except Exception as e:
                    # Suppress expected teardown errors during cleanup
                    logger.debug(
                        f"Compiled graph executor teardown warning (expected): {e}"
                    )
            self.compiled_executors.clear()
        finally:
            # Restore original logging levels once at the end
            for lg, level in zip(ray_loggers, original_levels):
                lg.setLevel(level)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying worker group."""
        return getattr(self.worker_group, name)
