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
import logging
import os
import sys
import time
from typing import Optional, TypedDict

import ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int


# Get the directory path of the current module and the root of the package
dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))


class PY_EXECUTABLES:
    SYSTEM = sys.executable

    # Use NeMo-RL direct dependencies.
    BASE = f"uv run --locked --directory {git_root}"

    # Use NeMo-RL direct dependencies and vllm.
    VLLM = f"uv run --locked --extra vllm --directory {git_root}"

    # Use NeMo-RL direct dependencies and fsdp.
    FSDP = f"uv run --locked --extra fsdp --directory {git_root}"

    # Use NeMo-RL direct dependencies and nemo-automodel.
    AUTOMODEL = f"uv run --locked --extra automodel --directory {git_root}"

    # Use NeMo-RL direct dependencies and Megatron.
    MCORE = f"uv run --locked --extra mcore --directory {git_root}"

    # Use NeMo-Gym dependencies
    NEMO_GYM = f"uv run --locked --extra nemo_gym --directory {git_root}"


@ray.remote  # pragma: no cover
def _get_node_ip_and_free_port() -> tuple[str, int]:
    return _get_node_ip_local(), _get_free_port_local()


def _get_node_ip_local() -> str:
    # Get the IP address of the current node
    node_ip = ray._private.services.get_node_ip_address()

    return node_ip


def _get_free_port_local() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to port 0 to get a random free port
        s.listen(1)
        port = s.getsockname()[1]

    return port


def init_ray(log_dir: Optional[str] = None) -> None:
    """Initialise Ray.

    Try to attach to an existing local cluster.
    If that cluster uses the same CUDA_VISIBLE_DEVICES or Slurm managed tag we will reuse it.
    Otherwise, we will detach and start a fresh local cluster.

    Args:
        log_dir: Optional directory to store Ray logs and temp files.
    """
    # Set up runtime environment
    env_vars = dict(os.environ)
    env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    runtime_env = {
        "env_vars": env_vars,  # Pass thru all user environment variables
    }

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    # sort cvd to ensure consistent tag
    cvd = ",".join(sorted(cvd.split(",")))
    cvd_tag_prefix = "nrl_tag_"
    cvd_tag = f"{cvd_tag_prefix}{cvd.replace(',', '_')}"

    # Try to attach to an existing cluster
    try:
        ray.init(
            address="auto",
            log_to_driver=True,
            include_dashboard=False,
            runtime_env=runtime_env,
            _temp_dir=os.path.abspath(log_dir) if log_dir else None,
        )

        cluster_res = ray.cluster_resources()

        # Check reusability for NeMo-RL managed local clusters
        if any(k.startswith(cvd_tag_prefix) for k in cluster_res):
            # Reuse if the driver's cvd_tag matches a tag in the cluster.
            # This is for reusing a previously self-started local cluster.
            if cvd_tag in cluster_res:
                logger.info(
                    f"Connected to existing Ray cluster (driver CVD_TAG '{cvd_tag}' matched): {cluster_res}"
                )
                return

            # If neither reuse condition is met, but we connected to *something*
            logger.info(
                f"Existing Ray cluster found ({cluster_res}) but it does not meet reuse criteria. "
                f"Driver's cvd_tag: '{[k for k in cluster_res if k.startswith(cvd_tag_prefix)][0]}'. Expected cvd_tag: '{cvd_tag}'. "
                "Starting a new local cluster..."
            )
            ray.shutdown()

            # Clear driver-side package cache so working_dir is re-uploaded
            import importlib

            import ray._private.runtime_env.packaging as _pkg

            importlib.reload(_pkg)

        # Always reuse if it's an externally managed cluster.
        else:
            logger.info(f"Connected to existing Ray cluster: {cluster_res}")
            return

    except ConnectionError:
        logger.debug("No existing Ray cluster found, will start a new one.")
        # If ConnectionError, proceed to start a new local cluster without further action here.
        # Clear driver-side package cache so working_dir is re-uploaded
        ray.shutdown()
        pass

    # Start a brand-new local cluster
    # Reuse `runtime_env` but drop `working_dir` to avoid packaging the whole repo (prevents ray OSError: Failed to download runtime_env file package issue)
    local_runtime_env = dict(runtime_env)
    local_runtime_env.pop("working_dir", None)

    ray.init(
        log_to_driver=True,
        include_dashboard=True,
        runtime_env=local_runtime_env,
        _temp_dir=os.path.abspath(log_dir) if log_dir else None,
        resources={cvd_tag: 1},
    )
    logger.info(
        f"Started local cluster with tag '{cvd_tag}': {ray.cluster_resources()}"
    )


@ray.remote(num_gpus=1)
class GetGPUIDActor:  # pragma: no cover
    """Util actor class to return GPU id of the current worker."""

    def get_ip_and_gpu_id(self):
        return ray._private.services.get_node_ip_address(), int(ray.get_gpu_ids()[0])


class ResourceInsufficientError(Exception):
    """Exception raised when the cluster does not have enough resources to satisfy the requested configuration."""


class RayVirtualCluster:
    """Creates a virtual distributed cluster using Ray placement groups.

    This class simplifies distributed training setup by:
    - Creating placement groups that represent logical compute nodes
    - Allocating GPU and CPU resources for distributed workers
    - Managing communication between distributed processes

    - Bundle: A resource allocation unit (ex: 4 GPUs on a single node)
    - Worker: A process that performs computation (model training/inference)
    - Node: A physical or virtual machine containing multiple bundles
    """

    def __init__(
        self,
        bundle_ct_per_node_list: list[int],
        use_gpus: bool = True,
        max_colocated_worker_groups: int = 1,
        num_gpus_per_node: int = 8,
        name: str = "",
        placement_group_strategy: str = "PACK",
    ):
        """Initialize a virtual cluster using Ray placement groups.

        Args:
            bundle_ct_per_node_list: List specifying GPU bundles per node
                                    (e.g., [2,2] creates 2 nodes with 2 GPU bundles each)
            use_gpus: Whether to allocate GPU resources
            max_colocated_worker_groups: Maximum number of worker groups that can be colocated
            num_gpus_per_node: Number of GPUs per node
            name: Name prefix for placement groups
            placement_group_strategy: Ray placement group strategy ("STRICT_PACK", "PACK", or "SPREAD")
        """
        self._bundle_ct_per_node_list = bundle_ct_per_node_list
        self._world_size = sum(self._bundle_ct_per_node_list)
        self._unified_placement_group: Optional[PlacementGroup] = None
        self._bundle_ids_sorted_by_ip_and_gpu: Optional[list[int]] = None
        self._node_ids_sorted_by_ip_and_gpu: Optional[list[int]] = None
        self.num_gpus_per_node = num_gpus_per_node
        self.use_gpus = use_gpus
        if use_gpus:
            assert num_gpus_per_node > 0, (
                "num_gpus_per_node must be greater than 0 if using GPUs"
            )
        self.max_colocated_worker_groups = max_colocated_worker_groups
        self.name = name
        self.placement_group_strategy = placement_group_strategy

        self._init_placement_group()

    def _init_placement_group(self) -> None:
        """Creates a unified placement group based on whether cross-node model parallelism is needed.

        Args:
            strategy: Ray placement group strategy (defaults to self.placement_group_strategy)
            use_unified_pg: If True, create a single unified placement group.
                          If False, create per-node placement groups.

        Returns:
            List of placement groups
        """
        if self._unified_placement_group is not None:
            return

        # Add retry logic that was previously in __init__
        max_retries = int(os.environ.get("NRL_VIRTUAL_CLUSTER_MAX_RETRIES", 6))
        assert max_retries > 0, (
            f"NRL_VIRTUAL_CLUSTER_MAX_RETRIES={max_retries} must be an integer greater than 0"
        )

        for i in range(max_retries):
            try:
                self._unified_placement_group = self._create_placement_group_internal(
                    self.placement_group_strategy
                )
                (
                    self._bundle_ids_sorted_by_ip_and_gpu,
                    self._node_ids_sorted_by_ip_and_gpu,
                ) = self._get_sorted_bundle_and_node_ids()
                return
            except ResourceInsufficientError as e:
                print(e)
                print(
                    f"Retrying placement group creation... {i + 1}/{max_retries}. Next retry in {2**i} seconds."
                )
                time.sleep(2**i)
                continue
        raise ResourceInsufficientError(
            f"Maximum number of retries reached ({max_retries}). Cluster resources may be insufficient or cluster itself is highly unstable. Please check your cluster configuration and your cluster logs."
        )

    def _create_placement_group_internal(self, strategy: str) -> PlacementGroup:
        """Internal method to create placement groups without retry logic."""
        # Check available resources in the Ray cluster
        cluster_resources = ray.cluster_resources()
        total_available_gpus = int(cluster_resources.get("GPU", 0))
        total_available_cpus = int(cluster_resources.get("CPU", 0))

        # Calculate required resources
        total_requested_gpus = (
            sum(self._bundle_ct_per_node_list) if self.use_gpus else 0
        )
        total_requested_cpus = (
            sum(self._bundle_ct_per_node_list) * self.max_colocated_worker_groups
        )

        # Validate resources
        if self.use_gpus and total_requested_gpus > total_available_gpus:
            raise ResourceInsufficientError(
                f"Not enough GPUs available. Requested {total_requested_gpus} GPUs, but only {total_available_gpus} are available in the cluster."
            )

        if total_requested_cpus > total_available_cpus:
            raise ResourceInsufficientError(
                f"Not enough CPUs available. Requested {total_requested_cpus} CPUs, but only {total_available_cpus} are available in the cluster."
            )

        num_cpus_per_bundle = self.max_colocated_worker_groups
        # num_gpus_per_bundle == 1 indicates that there is 1 GPU per process
        num_gpus_per_bundle = 1 if self.use_gpus else 0

        # Create a single unified placement group for cross-node model parallelism
        all_bundles = []
        for bundle_count in self._bundle_ct_per_node_list:
            for _ in range(bundle_count):
                all_bundles.append(
                    {"CPU": num_cpus_per_bundle, "GPU": num_gpus_per_bundle}
                )

        pg = placement_group(
            bundles=all_bundles, strategy=strategy, name=f"{self.name}-unified"
        )

        # Add timeout to prevent hanging indefinitely
        try:
            ray.get(pg.ready(), timeout=180)  # 3-minute timeout
        except (TimeoutError, ray.exceptions.GetTimeoutError):
            # Clean up any created placement groups
            try:
                remove_placement_group(pg)
            except Exception:
                pass
            raise TimeoutError(
                "Timed out waiting for placement groups to be ready. The cluster may not have enough resources "
                "to satisfy the requested configuration, or the resources may be busy with other tasks."
            )

        return pg

    def get_unified_placement_group(self) -> PlacementGroup:
        assert self._unified_placement_group is not None, (
            "Placement group must be initialized before calling get_unified_placement_group"
        )
        return self._unified_placement_group

    def get_placement_groups(self) -> list[PlacementGroup]:
        """Keeping for backward compatibility."""
        return [self._unified_placement_group]

    def world_size(self) -> int:
        return self._world_size

    def node_count(self) -> int:
        return sum(1 for count in self._bundle_ct_per_node_list if count > 0)

    def get_available_address_and_port(self, bundle_idx: int) -> tuple[str, int]:
        """Gets an available address and port for the given placement group index and bundle index.

        Returns:
            Tuple of (address, port)
        """
        # Get placement groups if not already created
        pg = self.get_unified_placement_group()

        return ray.get(
            _get_node_ip_and_free_port.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=bundle_idx
                ),
                num_cpus=0,
            ).remote()
        )

    def get_master_address_and_port(self) -> tuple[str, int]:
        """Gets the master address and port for the distributed training setup.

        Returns:
            Tuple of (address, port)
        """
        assert self._bundle_ids_sorted_by_ip_and_gpu is not None, (
            "self._bundle_ids_sorted_by_ip_and_gpu should be created before calling get_master_address_and_port."
        )
        # Get placement groups if not already created
        pg = self.get_unified_placement_group()
        bundle_ids = self._bundle_ids_sorted_by_ip_and_gpu  # Type guard for mypy
        return ray.get(
            _get_node_ip_and_free_port.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_ids[0],
                ),
                num_cpus=0,
            ).remote()
        )

    def _get_sorted_bundle_and_node_ids(self) -> tuple[list[int], list[int]]:
        """Gets the sorted bundle indices for the placement groups."""
        pg = self.get_unified_placement_group()
        pg_data = placement_group_table(pg)
        num_bundles = len(pg_data["bundles"])
        bundle_to_node_ids = pg_data["bundles_to_node_id"]
        sorted_node_ids = sorted(set(bundle_to_node_ids.values()))
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}

        # use info actor to get the GPU id
        info_actors = []
        for i in range(num_bundles):
            info_actors.append(
                GetGPUIDActor.options(
                    num_cpus=0.01,  # set both num_cpus and num_gpus to be small values to enable assignment in colocated case
                    num_gpus=0.01,
                    resources=None,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    ),
                ).remote()
            )

        ip_and_gpu_ids = ray.get(
            [actor.get_ip_and_gpu_id.remote() for actor in info_actors]
        )
        for actor in info_actors:
            ray.kill(actor)

        # original index, node_id, gpu_id
        ip_map = lambda ip_address: tuple(int(x) for x in ip_address.split("."))
        bundle_infos = [
            (i, ip_map(ip_and_gpu_ids[i][0]), ip_and_gpu_ids[i][1])
            for i in range(num_bundles)
        ]
        sorted_bundle_ids = []
        sorted_node_ids = []
        for bundle_info in sorted(bundle_infos, key=lambda x: (x[1], x[2])):
            bundle_index = bundle_info[0]
            nid = bundle_to_node_ids[bundle_index]
            node_index = node_id_to_index[nid]
            sorted_bundle_ids.append(bundle_index)
            sorted_node_ids.append(node_index)
        return sorted_bundle_ids, sorted_node_ids

    def get_bundle_indices_list_for_worker_group(
        self,
        tied_worker_group_size: int,
        workers_per_node: Optional[int],
        num_nodes: Optional[int],
        cluster_node_offset: Optional[int],
        cluster_gpu_offset_each_node: Optional[int],
    ) -> list[tuple[int, list[int]]]:
        """Gets the bundle indices list for a worker group.

        Args:
            tied_worker_group_size: Number of workers in a tied worker group
            workers_per_node: Number of workers per node
            num_nodes: Number of nodes in the cluster
            cluster_node_offset: Offset of the first node in the cluster
            cluster_gpu_offset_each_node: Offset of the first GPU on each node in the cluster

        Returns:
            List of (node_idx, bundle_indices) tuples, where each tuple specifies a tied group with its node and local bundle indices.
        """
        assert self._bundle_ids_sorted_by_ip_and_gpu is not None, (
            "self._bundle_ids_sorted_by_ip_and_gpu should be created before calling get_bundle_indices_list_for_worker_group."
        )
        assert self._node_ids_sorted_by_ip_and_gpu is not None, (
            "self._node_ids_sorted_by_ip_and_gpu should be created before calling get_bundle_indices_list_for_worker_group."
        )

        # Type guards for mypy
        bundle_ids = self._bundle_ids_sorted_by_ip_and_gpu
        node_ids = self._node_ids_sorted_by_ip_and_gpu

        # default values for arguments
        num_nodes = num_nodes or self.node_count()
        workers_per_node = workers_per_node or self.num_gpus_per_node
        cluster_node_offset = cluster_node_offset or 0
        cluster_gpu_offset_each_node = cluster_gpu_offset_each_node or 0

        # sanity check
        assert cluster_node_offset + num_nodes <= self.node_count(), (
            f"cluster_node_offset={cluster_node_offset} + num_nodes={num_nodes} must be less than or equal to the number of nodes in the cluster ({self.node_count()})"
        )
        assert (
            cluster_gpu_offset_each_node + workers_per_node <= self.num_gpus_per_node
        ), (
            f"cluster_gpu_offset_each_node={cluster_gpu_offset_each_node} + workers_per_node={workers_per_node} must be less than or equal to the number of GPUs per node in the cluster ({self.num_gpus_per_node})"
        )

        flattened_bundle_indices = []
        flattened_node_indices = []
        for x in range(cluster_node_offset, cluster_node_offset + num_nodes):
            for y in range(
                cluster_gpu_offset_each_node,
                cluster_gpu_offset_each_node + workers_per_node,
            ):
                flattened_bundle_indices.append(
                    bundle_ids[x * self.num_gpus_per_node + y]
                )
                flattened_node_indices.append(node_ids[x * self.num_gpus_per_node + y])
        bundle_indices_list = []
        for group_idx in range(len(flattened_bundle_indices) // tied_worker_group_size):
            bundle_indices_this_group = flattened_bundle_indices[
                group_idx * tied_worker_group_size : (group_idx + 1)
                * tied_worker_group_size
            ]
            first_worker_node_idx = flattened_node_indices[
                group_idx * tied_worker_group_size
            ]
            bundle_indices_list.append(
                (first_worker_node_idx, bundle_indices_this_group)
            )
        return bundle_indices_list

    def shutdown(self) -> bool:
        """Cleans up and releases all resources associated with this virtual cluster.

        This includes removing all placement groups and resetting the internal state.

        This method is idempotent and can be safely called multiple times.
        """
        if self._unified_placement_group is not None:
            # Remove all placement groups
            try:
                remove_placement_group(self._unified_placement_group)
            except Exception as e:
                # Log but continue if a placement group can't be removed
                print(
                    f"Error removing placement group {self._unified_placement_group.id}: {e}"
                )

            # Reset internal state
            self._unified_placement_group = None

        return True

    def __del__(self) -> None:
        """Shutsdown the virtual cluster when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown and the pointer to
        the cluster is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()
