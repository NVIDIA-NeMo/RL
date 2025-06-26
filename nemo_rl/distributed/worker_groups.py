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
import importlib
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.batched_data_dict import SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_group_utils import recursive_merge_options
from nemo_rl.utils.venvs import create_local_venv_on_each_node


@dataclass
class MultiWorkerFuture:
    """Container for Ray futures with associated worker information."""

    futures: list[ray.ObjectRef]
    return_from_workers: Optional[list[int]] = None
    called_workers: Optional[list[int]] = None

    def get_results(
        self, worker_group: "RayWorkerGroup", return_generators_as_proxies: bool = False
    ) -> list[Any]:
        """Get results from the futures, optionally respecting tied workers.

        The method uses worker_group.worker_to_tied_group_index to identify which tied
        worker group each worker belongs to, then selects only the first result from each group.

        Args:
            worker_group: The RayWorkerGroup that spawned the futures.  The
                mapping contained in worker_group.worker_to_tied_group_index
                is required for the deduplication path.
            return_generators_as_proxies: If True, and a future is an ObjectRefGenerator,
                                          return the ObjectRefGenerator itself instead of consuming it.

        Returns:
            List of results
        """
        from ray import ObjectRef, ObjectRefGenerator

        if return_generators_as_proxies:
            # Directly return the futures, which are expected to be ObjectRefGenerators (or other proxies).
            # No ray.get() is called on them. The consumer is responsible for handling the proxies.
            if self.return_from_workers is None:
                return self.futures

            if self.called_workers is not None:
                map_called_worker_to_future_idx = {
                    global_idx: i for i, global_idx in enumerate(self.called_workers)
                }
                final_proxies = []
                for global_worker_to_return in self.return_from_workers:
                    if global_worker_to_return in map_called_worker_to_future_idx:
                        future_idx = map_called_worker_to_future_idx[
                            global_worker_to_return
                        ]
                        if future_idx < len(self.futures):
                            final_proxies.append(self.futures[future_idx])
                return final_proxies
            else:
                return [
                    self.futures[i]
                    for i in self.return_from_workers
                    if i < len(self.futures)
                ]

        object_refs: list[ObjectRef] = []
        has_generator = False
        for idx, fut in enumerate(self.futures):
            if isinstance(fut, ObjectRefGenerator):
                # ray.get cannot be called directly on the generator object – it must be iterated to obtain the individual ObjectRef instances first.
                for generated_ref in fut:
                    object_refs.append(generated_ref)
                    has_generator = True
            else:
                object_refs.append(fut)

        # Retrieve the concrete results.
        all_results = ray.get(object_refs)

        # If expanded generator was present we are in streaming mode.
        # Every ObjectRef now corresponds to a unique, ordered chunk of data
        if has_generator:
            return all_results

        if self.return_from_workers is not None:
            if self.called_workers is not None:
                # Create a mapping from global worker indices to local indices in all_results
                worker_to_result_idx = {
                    worker: idx for idx, worker in enumerate(self.called_workers)
                }
                # # Filter return_from_workers to only include workers that were actually called
                valid_return_workers = [
                    w for w in self.return_from_workers if w in worker_to_result_idx
                ]
                # Map global worker indices to local result indices and get results
                return [
                    all_results[worker_to_result_idx[worker]]
                    for worker in valid_return_workers
                ]
            else:
                return [all_results[worker] for worker in self.return_from_workers]

        return all_results


class RayWorkerBuilder:
    @ray.remote
    class IsolatedWorkerInitializer:
        def __init__(self, ray_actor_class_fqn: str, *init_args, **init_kwargs):
            self.ray_actor_class_fqn = ray_actor_class_fqn
            self.init_args = init_args
            self.init_kwargs = init_kwargs

        def create_worker(
            self,
            placement_group: PlacementGroup,
            placement_group_bundle_index: int,
            num_gpus: int,
            bundle_indices: Optional[tuple] = None,
            **extra_options: Optional[dict[str, Any]],
        ):
            """Create a Ray worker with the specified configuration.

            Order of precedence for worker options configuration (from lowest to highest):
            1. Options passed by the user to __call__ (extra_options)
            2. Options required by the worker via configure_worker (may override user options with warning)
            3. Options set by the RayWorkerBuilder.__call__ (specifically scheduling strategy)

            If the worker needs to override user-provided options, it should log a warning
            to inform the user about the change and the reason for it.

            Args:
                placement_group: Ray placement group for resource allocation
                placement_group_bundle_index: Index of the bundle in the placement group
                num_gpus: Number of GPUs to allocate to this worker
                bundle_indices: Tuple of (node_idx, local_bundle_indices) for tensor parallelism (if applicable)
                extra_options: Additional options to pass to the Ray actor (may be overridden by actor's configure_worker(...) method)

            Returns:
                A Ray actor reference to the created worker
            """
            # Set up worker arguments and resources
            module_name, class_name = self.ray_actor_class_fqn.rsplit(".", 1)
            module = importlib.import_module(module_name)
            worker_class = getattr(module, class_name)
            worker_kwargs = dict(self.init_kwargs)
            default_options = getattr(worker_class, "_default_options", {})
            options = recursive_merge_options(default_options, extra_options)

            # Use the worker's configuration interface if available
            if hasattr(worker_class, "configure_worker"):
                # Get complete worker configuration from the worker class
                resources, env_vars, init_kwargs = worker_class.configure_worker(
                    num_gpus=num_gpus,
                    bundle_indices=bundle_indices,
                )

                # Apply resource configuration
                if resources and "num_gpus" in resources:
                    num_gpus = resources["num_gpus"]

                # Apply environment variables if provided
                if env_vars:
                    if "runtime_env" not in options:
                        options["runtime_env"] = {"env_vars": {}}
                    if "env_vars" not in options["runtime_env"]:  # type: ignore
                        options["runtime_env"]["env_vars"] = {}  # type: ignore
                    for k, v in env_vars.items():
                        options["runtime_env"]["env_vars"][k] = v  # type: ignore

                # Apply initialization parameters
                if init_kwargs:
                    worker_kwargs.update(init_kwargs)

            # Create options for Ray actor
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=placement_group_bundle_index,
                placement_group_capture_child_tasks=True,
            )
            options["num_gpus"] = num_gpus
            worker = worker_class.options(**options).remote(
                *self.init_args, **worker_kwargs
            )
            return worker

    def __init__(self, ray_actor_class_fqn: str, *args, **kwargs):
        self.ray_actor_class_fqn = ray_actor_class_fqn
        self.args = args
        self.kwargs = kwargs

    def create_worker_async(
        self,
        placement_group: PlacementGroup,
        placement_group_bundle_index: int,
        num_gpus: float | int,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
        **extra_options: Any,
    ) -> tuple[ray.ObjectRef, ray.actor.ActorHandle]:
        """Create a Ray worker asynchronously, returning futures.

        This method returns immediately with futures that can be awaited later.

        Args:
            placement_group: Ray placement group for resource allocation
            placement_group_bundle_index: Index of the bundle in the placement group
            num_gpus: Number of GPUs to allocate to this worker (can be fractional)
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for tensor parallelism (if applicable)
            extra_options: Additional options to pass to the Ray actor

        Returns:
            Tuple of (worker_future, initializer_actor):
                - worker_future: A Ray ObjectRef that will resolve to the worker actor
                - initializer_actor: The initializer actor (needed to prevent GC)
        """
        # Set up worker arguments and resources
        options = deepcopy(extra_options)
        initializer_options = {"runtime_env": options["runtime_env"]}
        isolated_initializer = self.IsolatedWorkerInitializer.options(  # type: ignore # @ray.remote call
            **initializer_options
        ).remote(self.ray_actor_class_fqn, *self.args, **self.kwargs)

        # Return the future and the initializer actor
        worker_future = isolated_initializer.create_worker.remote(
            placement_group,
            placement_group_bundle_index,
            num_gpus,
            bundle_indices,
            **options,
        )

        return worker_future, isolated_initializer

    def __call__(
        self,
        placement_group: PlacementGroup,
        placement_group_bundle_index: int,
        num_gpus: float | int,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
        **extra_options: Any,
    ) -> ray.actor.ActorHandle:
        """Create a Ray worker with the specified configuration.

        Order of precedence for worker options configuration (from lowest to highest):
        1. Options passed by the user to __call__ (extra_options)
        2. Options required by the worker via configure_worker (may override user options with warning)
        3. Options set by the RayWorkerBuilder.__call__ (specifically scheduling strategy)

        If the worker needs to override user-provided options, it should log a warning
        to inform the user about the change and the reason for it.

        Args:
            placement_group: Ray placement group for resource allocation
            placement_group_bundle_index: Index of the bundle in the placement group
            num_gpus: Number of GPUs to allocate to this worker (can be fractional)
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for tensor parallelism (if applicable)
            extra_options: Additional options to pass to the Ray actor (may be overridden by actor's configure_worker(...) method)

        Returns:
            A Ray actor reference to the created worker
        """
        # Use the async method and then block on the result
        worker_future, isolated_initializer = self.create_worker_async(
            placement_group,
            placement_group_bundle_index,
            num_gpus,
            bundle_indices,
            **extra_options,
        )

        # Block to get the worker
        worker = ray.get(worker_future)

        # We hold onto a reference to the initializer actor to avoid gc (would kill the child, 'real' actor)
        worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = isolated_initializer
        return worker


class RayWorkerGroup:
    """Manages a group of distributed Ray worker/actor processes that execute tasks in parallel.

    This class creates and manages Ray actor instances that run on resources
    allocated by a RayVirtualCluster. It handles:
    - Worker creation and placement on specific GPU resources
    - Setting up distributed training environment variables (rank, world size, etc.)
    - Executing methods across all workers in parallel
    - Collecting and aggregating results
    - Support for tied worker groups where multiple workers process the same data
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        remote_worker_builder: RayWorkerBuilder,
        workers_per_node: Optional[Union[int, list[int]]] = None,
        name_prefix: str = "",
        bundle_indices_list: Optional[list[tuple[int, list[int]]]] = None,
        sharding_annotations: Optional[NamedSharding] = None,
    ):
        """Initialize a group of distributed Ray workers.

        Args:
            cluster: RayVirtualCluster
            remote_worker_builder: Callable that launches a ray worker and has updatable options
            workers_per_node: Defaults to launch one worker per bundle in the cluster.
                          Alternatively specify an int or list to launch a different number of workers per node.
            name_prefix: Optional prefix for the names of the workers
            bundle_indices_list: Explicit list of (node_idx, [local_bundle_indices]) tuples.
                               Each tuple defines a tied group of workers placed on the same node.
                               If provided, workers_per_node is ignored.
            sharding_annotations: NamedSharding object representing mapping of named axes to ranks (i.e. for TP, PP, etc.)
        """
        self._workers: list[ray.actor.ActorHandle] = []
        self._worker_metadata: list[dict[str, Any]] = []
        self.cluster = cluster
        self.name_prefix = name_prefix
        self.sharding_annotations = sharding_annotations
        self.dp_leader_worker_indices: list[int] = []

        # If explicit bundle indices are provided, use those
        if bundle_indices_list is None:
            # Create bundle_indices_list from workers_per_node specification
            # In this case, each worker is its own group (no tied workers)
            bundle_indices_list = []

            # Get placement groups
            placement_groups = self.cluster.get_placement_groups()
            if len(placement_groups) == 1:
                # Single unified placement group
                pg = placement_groups[0]
                workers_per_group = [pg.bundle_count]
            else:
                # Multiple per-node placement groups
                workers_per_group = [pg.bundle_count for pg in placement_groups]

            # Determine how many workers per node/placement group
            if workers_per_node is None:
                workers_per_group = [pg.bundle_count for pg in placement_groups]
            elif isinstance(workers_per_node, int):
                workers_per_group = [workers_per_node] * len(placement_groups)
            elif isinstance(workers_per_node, list):
                if len(workers_per_node) == 1 and len(placement_groups) == 1:
                    workers_per_group = workers_per_node
                elif len(workers_per_node) != len(placement_groups):
                    raise ValueError(
                        f"workers_per_node list length ({len(workers_per_node)}) must match "
                        f"number of placement groups ({len(placement_groups)})"
                    )
                else:
                    workers_per_group = workers_per_node
            else:
                raise ValueError(
                    "workers_per_node must be None (for default distribution), an int, or a list"
                )

            # Validate workers_per_group
            for i, (pg, worker_count) in enumerate(
                zip(placement_groups, workers_per_group)
            ):
                if worker_count > pg.bundle_count:
                    raise ValueError(
                        f"Placement group {i} has {pg.bundle_count} bundles, "
                        f"but {worker_count} workers were requested"
                    )

                for bundle_idx in range(worker_count):
                    # Each worker is its own single-element group
                    # The first element is the PG index (node_idx in the context of tied workers)
                    bundle_indices_list.append((i, [bundle_idx]))

        # Create workers based on the bundle_indices_list
        self._create_workers_from_bundle_indices(
            remote_worker_builder, bundle_indices_list
        )

    def get_dp_leader_worker_idx(self, dp_shard_idx: int) -> int:
        """Returns the index of the primary worker for a given data parallel shard."""
        if not 0 <= dp_shard_idx < len(self.dp_leader_worker_indices):
            raise IndexError(
                f"Data parallel shard index {dp_shard_idx} is out of range. "
                f"Valid range is [0, {len(self.dp_leader_worker_indices) - 1}]"
            )
        return self.dp_leader_worker_indices[dp_shard_idx]

    def _create_workers_from_bundle_indices(
        self,
        remote_worker_builder: RayWorkerBuilder,
        bundle_indices_list: list[tuple[int, list[int]]],
    ) -> None:
        """Create workers based on explicit bundle indices for tied worker groups.

        Args:
            remote_worker_builder: Builder function for Ray actors

            bundle_indices_list: List of (node_idx, local_bundle_indices) tuples, where each tuple
                                specifies a tied group with its node and local bundle indices. If the local_bundle_indices
                                spans multiple nodes, the node_idx will be the first node's index in the tied group.
        """
        self.master_address, self.master_port = (
            self.cluster.get_master_address_and_port()
        )

        actor_python_env = get_actor_python_env(
            remote_worker_builder.ray_actor_class_fqn
        )
        if actor_python_env.startswith("uv"):
            # If the py_executable begins with uv it signals that we need to create a
            #  local venv first and then replace the py_executable with the local venv's python.
            #  The directory the venv will be created in is controlled by the env var
            #  NEMO_RL_VENV_DIR and defaults to $GIT_ROOT/venvs/.
            py_executable = create_local_venv_on_each_node(
                py_executable=actor_python_env,
                venv_name=remote_worker_builder.ray_actor_class_fqn,
            )
        else:
            py_executable = actor_python_env

        # Count total workers
        self.world_size = sum(len(indices) for _, indices in bundle_indices_list)
        global_rank = 0

        # Collect all async creation calls
        worker_futures = []
        worker_info = []  # Store metadata for each worker

        # Get all placement groups
        placement_groups = self.cluster.get_placement_groups()

        for group_idx, (pg_idx, local_bundle_indices) in enumerate(bundle_indices_list):
            current_group = []

            if len(placement_groups) == 1:
                pg = placement_groups[0]
            else:
                pg = placement_groups[pg_idx]

            is_parallel_group = len(local_bundle_indices) > 1

            for local_rank, bundle_idx in enumerate(local_bundle_indices):
                # Set up basic distributed environment variables
                env_vars = dict(os.environ)
                env_vars.update(
                    {
                        "RANK": str(global_rank),
                        "LOCAL_RANK": str(bundle_idx),
                        "WORLD_SIZE": str(self.world_size),
                        "MASTER_ADDR": self.master_address,
                        "MASTER_PORT": str(self.master_port),
                        "NODE_RANK": str(pg_idx),
                    }
                )
                env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)

                # Only the first worker in each group gets bundle_indices
                # This ensures only one worker per group is the model owner
                worker_bundle_indices = None
                if local_rank == 0:
                    worker_bundle_indices = (pg_idx, local_bundle_indices)
                    self.dp_leader_worker_indices.append(global_rank)

                # Create a descriptive name based on group structure
                name = (
                    f"{self.name_prefix}-grp{group_idx}-{local_rank}"
                    if is_parallel_group
                    else f"{self.name_prefix}-{pg_idx}-{bundle_idx}"
                )

                # Calculate GPU resources
                num_gpus = (
                    1 / self.cluster.max_colocated_worker_groups
                    if self.cluster.use_gpus
                    else 0
                )

                # Pass these options to the remote_worker_builder
                runtime_env = {"env_vars": env_vars, "py_executable": py_executable}
                runtime_env["env_vars"]["VIRTUAL_ENV"] = py_executable
                runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"] = py_executable

                extra_options = {"runtime_env": runtime_env, "name": name}

                # start worker creation asynchronously
                worker_future, initializer = remote_worker_builder.create_worker_async(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    num_gpus=num_gpus,
                    bundle_indices=worker_bundle_indices,
                    **extra_options,
                )

                # Store the future and metadata
                worker_idx = len(worker_futures)
                worker_futures.append((worker_future, initializer))
                worker_info.append(
                    {
                        "group_idx": group_idx,
                        "worker_idx": worker_idx,
                        "node_idx": pg_idx,
                        "local_rank": local_rank,
                        "global_rank": global_rank,
                        "name": name,
                        "bundle_indices": worker_bundle_indices,
                        "dp_shard_idx": group_idx,
                    }
                )
                current_group.append(worker_idx)

                global_rank += 1

        print(
            f"Waiting for {len(worker_futures)} workers to finish initializing...",
            flush=True,
        )
        worker_refs = [future for future, _ in worker_futures]
        workers = ray.get(worker_refs)

        for idx, (worker, (_, initializer)) in enumerate(zip(workers, worker_futures)):
            worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = initializer
            self._workers.append(worker)

            # Get the corresponding metadata
            info = worker_info[idx]
            self._worker_metadata.append(
                {
                    "node_idx": info["node_idx"],
                    "local_rank": info["local_rank"],
                    "global_rank": info["global_rank"],
                    "name": info["name"],
                    "bundle_indices": info["bundle_indices"],
                    "dp_shard_idx": info["group_idx"],
                }
            )

    @property
    def workers(self) -> list[ray.actor.ActorHandle]:
        return self._workers

    @property
    def worker_metadata(self) -> list[dict[str, Any]]:
        return self._worker_metadata

    @property
    def dp_size(self) -> int:
        """Number of data parallel shards."""
        return len(self.dp_leader_worker_indices)

    def run_single_worker_single_data(
        self,
        method_name: str,
        worker_idx: int,
        *args,
        **kwargs,
    ) -> ray.ObjectRef:
        """Run a method on a single, specific worker.

        Args:
            method_name: Name of the method to call on the worker.
            worker_idx: The index of the worker to run the method on.
            *args, **kwargs: Arguments to pass to the method.

        Returns:
            ray.ObjectRef: A Ray future for the result.
        """
        worker = self.workers[worker_idx]
        method = getattr(worker, method_name)
        return method.remote(*args, **kwargs)

    def run_all_workers_multiple_data(
        self,
        method_name: str,
        data: list[Any],
        run_rank_0_only_axes: list[str] | None = None,
        common_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[ray.ObjectRef]:
        """Run a method on all workers in parallel with different data.

        Args:
            method_name: Name of the method to call on each worker
            data: List of data to pass to workers/groups
            run_rank_0_only_axes: List of named axes for which only rank 0 should run the method.
            common_kwargs: Additional keyword arguments to pass to all workers

        Returns:
            list[ray.ObjectRef]: A list of ray futures
        """
        if run_rank_0_only_axes is None:
            assert len(data) == len(self.workers), (
                "data length should be equal to the number of workers: "
                f"data length = {len(data)}, number of workers = {len(self.workers)}"
            )

        futures = []

        if run_rank_0_only_axes is None:
            run_rank_0_only_axes = []
        if common_kwargs is None:
            common_kwargs = {}

        data_idx = 0
        for worker_idx, worker in enumerate(self.workers):
            worker_coords = self.sharding_annotations.get_worker_coords(worker_idx)

            # Determine if this worker should receive data
            should_run = True
            for axis in self.sharding_annotations.names:
                if axis not in worker_coords:
                    continue
                if axis in run_rank_0_only_axes and worker_coords[axis] != 0:
                    should_run = False
                    break

            if should_run:
                method = getattr(worker, method_name)
                futures.append(method.remote(data=data[data_idx], **common_kwargs))
                data_idx += 1

        assert data_idx == len(data), (
            "data length should be equal to the number of workers started: "
            f"data length = {len(data)}, number of workers started = {data_idx}"
        )

        return futures

    def run_all_workers_single_data(
        self,
        method_name: str,
        *args,
        run_rank_0_only_axes: list[str] | None = None,
        **kwargs,
    ) -> list[ray.ObjectRef]:
        """Run a method on all workers in parallel with the same data.

        Args:
            method_name: Name of the method to call on each worker
            *args, **kwargs: Arguments to pass to the method
            run_rank_0_only_axes: List of named axes for which only rank 0 should run the method.

        Returns:
            list[ray.ObjectRef]: A list of ray futures
        """
        futures = []

        if run_rank_0_only_axes is None:
            run_rank_0_only_axes = []

        for worker_idx, worker in enumerate(self.workers):
            worker_coords = self.sharding_annotations.get_worker_coords(worker_idx)

            # Determine if this worker should receive data
            should_run = True
            for axis in self.sharding_annotations.names:
                if axis not in worker_coords:
                    continue
                if axis in run_rank_0_only_axes and worker_coords[axis] != 0:
                    should_run = False
                    break

            if should_run:
                method = getattr(worker, method_name)
                futures.append(method.remote(*args, **kwargs))

        return futures

    def run_all_workers_sharded_data(
        self,
        method_name: str,
        data: Iterable[SlicedDataDict],  # arbitrary nested iterables of SlicedDataDicts
        in_sharded_axes: list[str] | None = None,
        replicate_on_axes: list[str] | None = None,
        output_is_replicated: list[str] | None = None,
        make_dummy_calls_to_free_axes: bool = False,
        common_kwargs: Optional[dict[str, Any]] = None,
    ) -> MultiWorkerFuture:
        """Run a method on all workers in parallel with sharded data.

        Axes in in_sharded_axes: Data is already split across these axes, so we just send the appropriate slice to each worker (along this axis)
        Axes in replicate_on_axes: Data is replicated to all workers along these dimensions
        Free axes (axes not in either list): Data is only sent to workers at index 0 of these axes

        Args:
            method_name: Name of the method to call on each worker
            data: Iterable of SlicedDataDicts to pass to workers/groups
            in_sharded_axes: List of axes that are sharded
            replicate_on_axes: List of axes that are to be replicated
            output_is_replicated: List of axes along which the output is replicated (and we should just return the first result).
                                  We also just return from rank 0 of free axes.
            make_dummy_calls_to_free_axes: Whether to make dummy calls (with None) to workers that
                                           aren't rank 0 on 'free axes' (axes not in in_sharded_axes or replicate_on_axes).
            common_kwargs: Additional keyword arguments to pass to all workers
        Returns:
            MultiWorkerFuture: Object containing futures and their associated worker information
        """
        if self.sharding_annotations is None:
            raise ValueError(
                "Sharding annotations must be provided to use sharded data distribution"
            )

        if common_kwargs is None:
            common_kwargs = {}
        if in_sharded_axes is None:
            in_sharded_axes = []
        if replicate_on_axes is None:
            replicate_on_axes = []
        if output_is_replicated is None:
            output_is_replicated = []

        futures = []

        # Validate axes
        for axis in in_sharded_axes + replicate_on_axes:
            if axis not in self.sharding_annotations.names:
                raise ValueError(
                    f"Axis '{axis}' not found in sharding annotations. Valid axes: {self.sharding_annotations.names}"
                )

        # Check for overlapping axes
        overlap = set(in_sharded_axes).intersection(set(replicate_on_axes))
        if overlap:
            raise ValueError(f"Axes cannot be both sharded and replicated: {overlap}")

        called_workers = []
        return_from_workers = []
        # For each worker, determine what data it should receive
        for worker_idx, worker in enumerate(self._workers):
            # Get the worker's coordinates in the sharding space
            worker_coords = self.sharding_annotations.get_worker_coords(worker_idx)

            # Determine if this worker should receive data
            should_receive_data = True
            return_from_this_worker = True
            for axis in self.sharding_annotations.names:
                if axis not in worker_coords:
                    continue
                # We call axes not in in_sharded_axes or replicate_on_axes free axes.
                if (
                    axis not in in_sharded_axes
                    and axis not in replicate_on_axes
                    and worker_coords[axis] != 0
                ):
                    # For free axes, only workers at index 0 receive data
                    should_receive_data = False
                    return_from_this_worker = False
                    break
                if axis in output_is_replicated:
                    if worker_coords[axis] != 0:
                        return_from_this_worker = False
            if return_from_this_worker:
                return_from_workers.append(worker_idx)

            if should_receive_data:
                # Find the appropriate data slice for this worker
                worker_data = data
                for axis in in_sharded_axes:
                    if axis in worker_coords:
                        # Select the appropriate slice for this axis
                        worker_data = worker_data[worker_coords[axis]]

                # Call the method on the worker with its data slice
                future = getattr(worker, method_name).remote(
                    data=worker_data, **common_kwargs
                )
                futures.append(future)
                called_workers.append(worker_idx)
            else:
                # If this worker doesn't need data:
                if make_dummy_calls_to_free_axes:
                    # If make_dummy_calls_to_free_axes is True, just call the method with None
                    future = getattr(worker, method_name).remote(
                        data=None, **common_kwargs
                    )
                    futures.append(future)
                    called_workers.append(worker_idx)
                else:
                    # Else, don't call the method at all
                    pass

        return MultiWorkerFuture(
            futures=futures,
            called_workers=called_workers,
            return_from_workers=return_from_workers,
        )

    def get_all_worker_results(
        self,
        future_bundle: MultiWorkerFuture,
        return_generators_as_proxies: bool = False,
    ) -> list[Any]:
        """Get results from all workers, optionally filtering to get just one result per tied worker group.

        Args:
            future_bundle: MultiWorkerFuture containing futures and worker information.
            return_generators_as_proxies: If True, and a future in the bundle is an ObjectRefGenerator,
                                          return the ObjectRefGenerator itself instead of consuming it.

        Returns:
            List of results, deduplicated as specified in the future_bundle
        """
        return future_bundle.get_results(
            self, return_generators_as_proxies=return_generators_as_proxies
        )

    def shutdown(
        self,
        cleanup_method: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        force: bool = False,
    ) -> bool:
        """Shutdown all workers in the worker group.

        Args:
            cleanup_method: Optional method name to call on each worker before termination.
                            If provided, this method will be called on each worker to allow
                            for graceful cleanup.
            timeout: Timeout in seconds for graceful shutdown. Only applicable if cleanup_method is provided.
                     If None, wait indefinitely for workers to complete their cleanup.
            force: If True, forcefully terminate workers with ray.kill() even if cleanup_method is provided.
                   If cleanup_method is None, workers are always forcefully terminated.

        Returns:
            bool: True if all workers were successfully shut down
        """
        if not self._workers:
            return True

        success = True

        # First attempt graceful shutdown if cleanup method is provided and force=False
        if cleanup_method is not None and not force:
            try:
                # Call cleanup method on all workers
                futures = self.run_all_workers_single_data(cleanup_method)

                # Wait for all cleanup operations to complete with timeout
                if timeout is not None:
                    ray.get(futures, timeout=timeout)
                else:
                    ray.get(futures)

            except (ray.exceptions.RayTaskError, ray.exceptions.GetTimeoutError) as e:
                success = False
                print(
                    f"Error during graceful shutdown: {e}. Falling back to force termination."
                )
                force = True

        # Force kill any remaining workers
        if force or cleanup_method is None:
            initializers_to_kill = []
            for worker in self._workers:
                if hasattr(worker, "_RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC"):
                    # Store the initializer ref before the main worker is killed,
                    # as killing the worker might affect accessibility of this attribute later.
                    initializer = getattr(
                        worker, "_RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC", None
                    )
                    if initializer:
                        initializers_to_kill.append(initializer)
                try:
                    ray.kill(worker)
                except Exception as e:
                    success = False
                    print(f"Error killing worker: {e}")

            # Now, explicitly kill the initializer actors
            # This makes their termination more deterministic than relying solely on Ray's GC.
            for initializer in initializers_to_kill:
                try:
                    ray.kill(initializer)
                except Exception as e:
                    print(f"Error killing initializer actor for a worker: {e}")

        # Clear worker lists
        self._workers = []
        self._worker_metadata = []

        return success
