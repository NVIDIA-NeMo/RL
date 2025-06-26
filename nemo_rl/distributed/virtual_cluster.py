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
from typing import Any, Optional, TypedDict

import ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
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
    BASE = "uv run --locked"

    # Use NeMo-RL direct dependencies and vllm.
    VLLM = "uv run --locked --extra vllm"

    # Megatron-core (and nemo dependencies)
    # We always run with --reinstall to avoid issues where someone runs "uv run ... --extra mcore ..."
    # but the submodules are not downloaded yet. This results in errors where it appears Megatron/Nemo
    # aren't installed. Simple workaround is to always run the mcore py_executable with --reinstall.
    MCORE = "uv run --reinstall --extra mcore"


@ray.remote
def _get_node_ip_and_free_port() -> tuple[str, int]:
    import socket

    # Get the IP address of the current node
    node_ip = ray._private.services.get_node_ip_address()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to port 0 to get a random free port
        s.listen(1)
        port = s.getsockname()[1]
    return node_ip, port


def init_ray(log_dir: Optional[str] = None) -> None:
    """Initialise Ray.

    Try to attach to an existing local cluster.
    If that cluster uses the same CUDA_VISIBLE_DEVICES or Slurm managed tag we will reuse it.
    Otherwise, we will detach and start a fresh local cluster.
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
        placement_group_strategy: str = "SPREAD",
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
        self._node_placement_groups: Optional[list[PlacementGroup]] = None

        self.num_gpus_per_node = num_gpus_per_node
        self.use_gpus = use_gpus
        if use_gpus:
            assert num_gpus_per_node > 0, (
                "num_gpus_per_node must be greater than 0 if using GPUs"
            )
        self.max_colocated_worker_groups = max_colocated_worker_groups
        self.name = name
        self.placement_group_strategy = placement_group_strategy

    def _init_placement_groups(
        self, strategy: str | None = None, use_unified_pg: bool | None = None
    ) -> list[PlacementGroup]:
        """Creates placement groups based on whether cross-node model parallelism is needed.

        Args:
            strategy: Ray placement group strategy (defaults to self.placement_group_strategy)
            use_unified_pg: If True, create a single unified placement group.
                          If False, create per-node placement groups.

        Returns:
            List of placement groups
        """
        if self._node_placement_groups is not None:
            return self._node_placement_groups

        if strategy is None:
            strategy = self.placement_group_strategy

        # Add retry logic that was previously in __init__
        max_retries = int(os.environ.get("NRL_VIRTUAL_CLUSTER_MAX_RETRIES", 6))
        assert max_retries > 0, (
            f"NRL_VIRTUAL_CLUSTER_MAX_RETRIES={max_retries} must be an integer greater than 0"
        )

        for i in range(max_retries):
            try:
                self._node_placement_groups = self._create_placement_groups_internal(
                    strategy, use_unified_pg
                )
                return self._node_placement_groups
            except ResourceInsufficientError as e:
                print(e)
                print(
                    f"Retrying placement group creation... {i + 1}/{max_retries}. Next retry in {2**i} seconds."
                )
                time.sleep(2**i)
                continue
        else:
            raise ResourceInsufficientError(
                f"Maximum number of retries reached ({max_retries}). Cluster resources may be insufficient or cluster itself is highly unstable. Please check your cluster configuration and your cluster logs."
            )

    def _create_placement_groups_internal(
        self, strategy: str, use_unified_pg: bool = False
    ) -> list[PlacementGroup]:
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

        placement_groups = []
        if use_unified_pg:
            # Create a single unified placement group for cross-node model parallelism
            all_bundles = []
            for bundle_count in self._bundle_ct_per_node_list:
                for _ in range(bundle_count):
                    all_bundles.append(
                        {"CPU": num_cpus_per_bundle, "GPU": num_gpus_per_bundle}
                    )

            placement_groups = [
                placement_group(
                    bundles=all_bundles, strategy=strategy, name=f"{self.name}-unified"
                )
            ]
        else:
            # Create per-node placement groups to respect bundle_ct_per_node_list
            for node_idx, bundle_count in enumerate(self._bundle_ct_per_node_list):
                if bundle_count > 0:
                    node_bundles = [
                        {"CPU": num_cpus_per_bundle, "GPU": num_gpus_per_bundle}
                        for _ in range(bundle_count)
                    ]
                    pg = placement_group(
                        bundles=node_bundles,
                        strategy="PACK",  # Use PACK to keep bundles together
                        name=f"{self.name}-node{node_idx}",
                    )
                    placement_groups.append(pg)

        # Add timeout to prevent hanging indefinitely
        try:
            ray.get(
                [pg.ready() for pg in placement_groups], timeout=180
            )  # 3-minute timeout
        except (TimeoutError, ray.exceptions.GetTimeoutError):
            # Clean up any created placement groups
            for pg in placement_groups:
                try:
                    remove_placement_group(pg)
                except Exception:
                    pass
            raise TimeoutError(
                "Timed out waiting for placement groups to be ready. The cluster may not have enough resources "
                "to satisfy the requested configuration, or the resources may be busy with other tasks."
            )

        return placement_groups

    def get_placement_groups(self) -> list[PlacementGroup]:
        # Initialize placement groups if not already created
        if self._node_placement_groups is None:
            self._init_placement_groups()

        assert self._node_placement_groups is not None, (
            "Placement groups must be initialized before calling get_placement_groups"
        )
        return [pg for pg in self._node_placement_groups if pg.bundle_specs]

    def world_size(self) -> int:
        return self._world_size

    def node_count(self) -> int:
        return sum(1 for count in self._bundle_ct_per_node_list if count > 0)

    def get_master_address_and_port(self) -> tuple[str, int]:
        """Gets the master address and port for the distributed training setup.

        Returns:
            Tuple of (address, port)
        """
        # Get placement groups if not already created
        if not self._node_placement_groups:
            self.get_placement_groups()

        # Use the first bundle of the first placement group
        # This works for both unified PG and per-node PGs
        pg = self.get_placement_groups()[0]
        if pg.bundle_specs:
            # Launch port finder on the first bundle of this placement group
            addr, port = ray.get(
                _get_node_ip_and_free_port.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=0
                    ),
                    # Need to explicitly set to 0 since it's possible for this to be unschedulable if all CPUs are already in use.
                    num_cpus=0,
                ).remote()
            )
            return addr, port

        raise RuntimeError("No valid placement groups found to get master address")

    def shutdown(self) -> bool:
        """Cleans up and releases all resources associated with this virtual cluster.

        This includes removing all placement groups and resetting the internal state.

        This method is idempotent and can be safely called multiple times.
        """
        if self._node_placement_groups is not None:
            # Remove all placement groups
            for pg in self._node_placement_groups:
                try:
                    remove_placement_group(pg)
                except Exception as e:
                    # Log but continue if a placement group can't be removed
                    print(f"Error removing placement group {pg.id}: {e}")

            # Reset internal state
            self._node_placement_groups = None

        return True

    def _create_visualization_grid(
        self, worker_groups: Optional[Any] = None, is_global_view: bool = False
    ) -> dict[str, Any]:
        """Create a visualization grid for the cluster with optional worker groups.

        Args:
            worker_groups: Single worker group, list of worker groups, or None
            is_global_view: Whether this is a global view (multiple worker groups) or single view

        Returns:
            dict: A dictionary containing the grid data for display
        """
        # Convert single worker group to list for uniform processing
        if worker_groups is not None and not isinstance(worker_groups, list):
            worker_groups = [worker_groups]
        elif worker_groups is None:
            worker_groups = []

        # Find the maximum number of GPUs per node for grid layout
        max_gpus_per_node = (
            max(self._bundle_ct_per_node_list) if self._bundle_ct_per_node_list else 0
        )
        if max_gpus_per_node == 0:
            return {"empty": True}

        # Number of nodes with GPUs
        active_nodes = sum(1 for count in self._bundle_ct_per_node_list if count > 0)

        # Determine cell width based on view type
        cell_width = 12 if is_global_view else 7

        # Create horizontal divider based on max GPUs per node
        h_divider = "+" + "+".join(["-" * cell_width] * max_gpus_per_node) + "+"

        # Build the grid data
        grid_data = {
            "active_nodes": active_nodes,
            "total_gpus": self.world_size(),
            "worker_groups": worker_groups,
            "max_gpus_per_node": max_gpus_per_node,
            "cell_width": cell_width,
            "h_divider": h_divider,
            "is_global_view": is_global_view,
            "rows": [],
        }

        # For each node, create its row in the grid
        for node_idx, bundle_count in enumerate(self._bundle_ct_per_node_list):
            if bundle_count == 0:
                continue

            # Initialize row data
            node_row = {
                "node_idx": node_idx,
                "bundle_count": bundle_count,
                "gpu_cells": [],
                "worker_cells": [],
            }

            # Initialize worker cells arrays (one per worker group)
            for i in range(len(worker_groups)):
                node_row["worker_cells"].append([])  # type: ignore

            # Process each GPU position in the row
            for gpu_idx in range(max_gpus_per_node):
                if gpu_idx < bundle_count:
                    # This is a real GPU
                    gpu_cell = f" {node_idx}.{gpu_idx} "

                    # Process worker assignments for this GPU
                    worker_cells = self._get_worker_cells(
                        node_idx, gpu_idx, worker_groups, cell_width, is_global_view
                    )
                else:
                    # Empty cell (no GPU)
                    gpu_cell = " " * cell_width
                    worker_cells = [" " * cell_width] * len(worker_groups)

                # Add cells to the row
                node_row["gpu_cells"].append(gpu_cell)  # type: ignore
                for i, cell in enumerate(worker_cells):
                    if i < len(node_row["worker_cells"]):  # type: ignore
                        node_row["worker_cells"][i].append(cell)  # type: ignore

            # Add the completed row to the grid
            grid_data["rows"].append(node_row)

        return grid_data

    def _get_worker_cells(
        self,
        node_idx: int,
        gpu_idx: int,
        worker_groups: list[Any],
        cell_width: int,
        is_global_view: bool,
    ) -> list[str]:
        """Get the worker cell content for each worker group at a specific GPU location.

        Args:
            node_idx: The node index
            gpu_idx: The GPU index within the node
            worker_groups: List of worker groups to check
            cell_width: Width of each cell for formatting
            is_global_view: Whether this is a global view with multiple worker groups

        Returns:
            list: List of formatted worker cells, one per worker group
        """
        worker_cells = []

        for wg_idx, worker_group in enumerate(worker_groups):
            # Default empty worker cell
            worker_cell = " " * cell_width

            # Find workers from this group assigned to this GPU
            for worker_id, metadata in enumerate(worker_group.worker_metadata):
                if (
                    metadata["node_idx"] == node_idx
                    and metadata["local_rank"] == gpu_idx
                ):
                    if is_global_view:
                        # Use group numbering in global view
                        worker_cell = f" G{wg_idx}:W{worker_id:<2d} "
                    else:
                        # Use simple worker IDs in single group view
                        worker_cell = f" W {worker_id:<2d} "
                    break

            worker_cells.append(worker_cell)

        return worker_cells

    def _print_visualization(self, grid_data: dict[str, Any]) -> None:
        """Print the visualization based on the grid data.

        Args:
            grid_data: The grid data generated by _create_visualization_grid
        """
        if grid_data.get("empty", False):
            print("\nEmpty Ray Cluster (no GPUs)")
            return

        # Print header
        if grid_data["is_global_view"]:
            # Global view header
            wg_summary = ""
            if grid_data["worker_groups"]:
                wg_summary = f", Worker Groups: {len(grid_data['worker_groups'])}"

            print(
                f"\nRay Cluster Global View: {grid_data['active_nodes']} nodes, {grid_data['total_gpus']} GPUs{wg_summary}"
            )
        else:
            # Single view header
            wg_info = ""
            if grid_data["worker_groups"]:
                worker_group = grid_data["worker_groups"][0]
                wg_name = getattr(worker_group, "name_prefix", "Default") or "Default"
                wg_info = (
                    f", Worker Group: {wg_name} ({worker_group.world_size} workers)"
                )

            print(
                f"\nRay Cluster: {grid_data['active_nodes']} nodes, {grid_data['total_gpus']} GPUs{wg_info}"
            )

        # Print the top border
        print(grid_data["h_divider"])

        # Print each row of the grid
        for row in grid_data["rows"]:
            # Print GPU row
            gpu_row = ["|"]
            for cell in row["gpu_cells"]:
                gpu_row.append(cell.ljust(grid_data["cell_width"]))
                gpu_row.append("|")
            print("".join(gpu_row))

            # Print worker rows
            for wg_idx, worker_cells in enumerate(row["worker_cells"]):
                worker_row = ["|"]
                for cell in worker_cells:
                    worker_row.append(cell.ljust(grid_data["cell_width"]))
                    worker_row.append("|")
                print("".join(worker_row))

            # Print divider between nodes
            print(grid_data["h_divider"])

        # Print legend
        self._print_legend(grid_data)

    def _print_legend(self, grid_data: dict[str, Any]) -> None:
        """Print the legend for the visualization."""
        if grid_data["is_global_view"]:
            # Legend for global view
            if grid_data["worker_groups"]:
                print("Legend:")
                for wg_idx, wg in enumerate(grid_data["worker_groups"]):
                    wg_name = getattr(wg, "name_prefix", "unnamed") or "unnamed"
                    wg_count = wg.world_size
                    print(f"G{wg_idx}: {wg_name} ({wg_count} workers)")
                print("W##: Worker ID within its group")
        else:
            # Legend for single worker group view
            if grid_data["worker_groups"]:
                wg_name = (
                    getattr(grid_data["worker_groups"][0], "name_prefix", "") or ""
                )
                print(f"W## = Worker ID in '{wg_name}' worker group")

        print("#.#: Node.GPU identifier")

    def print_cluster_grid(self, worker_group: Optional[Any] = None) -> None:
        """Prints a compact grid visualization of the virtual cluster, similar to JAX's visualize_array_sharding.

        If a worker_group is provided, it will also show worker assignments on each device.

        Args:
            worker_group: Optional RayWorkerGroup instance to visualize worker assignments
        """
        grid_data = self._create_visualization_grid(worker_group, is_global_view=False)
        self._print_visualization(grid_data)

    def print_all_worker_groups(
        self, worker_groups: Optional[list[Any]] = None
    ) -> None:
        """Prints a visualization showing all worker groups in the cluster.

        This provides a global view of all workers across all worker groups.

        Args:
            worker_groups: List of RayWorkerGroup instances to visualize. If None,
                          no worker assignments will be shown.
        """
        grid_data = self._create_visualization_grid(worker_groups, is_global_view=True)
        self._print_visualization(grid_data)

    def __del__(self) -> None:
        """Shutsdown the virtual cluster when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown and the pointer to
        the cluster is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()
