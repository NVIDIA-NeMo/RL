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
"""Resource management for cluster allocation.

This module provides a unified interface for managing cluster resources
including GPU allocation, worker counts, and memory limits.

The ResourceManager serves as the single source of truth for resource
allocation and integrates with RayVirtualCluster for distributed allocation.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of allocatable resources."""

    GPU = "GPU"
    CPU = "CPU"
    MEMORY = "MEMORY"


class AllocationError(Exception):
    """Error raised when resource allocation fails.

    Attributes:
        resource_type: Type of resource that failed to allocate.
        requested: Amount of resource requested.
        available: Amount of resource available.
    """

    def __init__(
        self,
        message: str,
        resource_type: ResourceType | None = None,
        requested: int | float = 0,
        available: int | float = 0,
    ):
        self.resource_type = resource_type
        self.requested = requested
        self.available = available
        super().__init__(message)


@dataclass
class Resource:
    """Represents a resource with total and allocated amounts.

    Attributes:
        total: Total amount of the resource.
        allocated: Amount currently allocated.
        resource_type: Type of the resource.
    """

    total: int | float
    allocated: int | float = 0
    resource_type: ResourceType = ResourceType.GPU

    @property
    def available(self) -> int | float:
        """Amount of resource currently available."""
        return self.total - self.allocated

    @property
    def utilization(self) -> float:
        """Resource utilization as a fraction (0.0 to 1.0)."""
        if self.total == 0:
            return 0.0
        return self.allocated / self.total

    def can_allocate(self, amount: int | float) -> bool:
        """Check if the requested amount can be allocated."""
        return amount <= self.available

    def allocate(self, amount: int | float) -> None:
        """Allocate the specified amount of resource.

        Raises:
            AllocationError: If insufficient resources are available.
        """
        if not self.can_allocate(amount):
            raise AllocationError(
                f"Cannot allocate {amount} {self.resource_type.value}. "
                f"Only {self.available} available out of {self.total} total.",
                resource_type=self.resource_type,
                requested=amount,
                available=self.available,
            )
        self.allocated += amount

    def release(self, amount: int | float) -> None:
        """Release the specified amount of resource.

        Args:
            amount: Amount to release. Will be clamped to allocated amount.
        """
        amount = min(amount, self.allocated)
        self.allocated -= amount


@dataclass
class ResourceAllocation:
    """Represents an allocation of resources.

    Used to track allocations that can be released later.

    Attributes:
        allocation_id: Unique identifier for this allocation.
        gpus: Number of GPUs allocated.
        cpus: Number of CPUs allocated.
        memory_gb: Amount of memory allocated in GB.
        worker_group_id: ID of the worker group (if applicable).
        metadata: Additional allocation metadata.
    """

    allocation_id: str
    gpus: int = 0
    cpus: int = 0
    memory_gb: float = 0.0
    worker_group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ResourceManager:
    """Manages cluster resources for distributed training.

    The ResourceManager is the single source of truth for:
    - GPU allocation across the cluster
    - Worker counts and distribution
    - Memory limits and utilization

    It integrates with RayVirtualCluster for distributed allocation.

    Attributes:
        total_gpus: Total number of GPUs in the cluster.
        total_cpus: Total number of CPUs in the cluster.
        total_memory_gb: Total memory in the cluster (GB).
        gpus_per_node: Number of GPUs per node.
        num_nodes: Number of nodes in the cluster.

    Example:
        >>> manager = ResourceManager(num_nodes=2, gpus_per_node=8)
        >>> allocation = manager.allocate(gpus=4, cpus=32)
        >>> print(manager.gpu_utilization)
        0.25
        >>> manager.release(allocation)
        >>> print(manager.available_gpus)
        16
    """

    def __init__(
        self,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        cpus_per_node: int | None = None,
        memory_per_node_gb: float | None = None,
    ):
        """Initialize the ResourceManager.

        Args:
            num_nodes: Number of nodes in the cluster.
            gpus_per_node: Number of GPUs per node.
            cpus_per_node: Number of CPUs per node (auto-detected if None).
            memory_per_node_gb: Memory per node in GB (auto-detected if None).
        """
        self._num_nodes = num_nodes
        self._gpus_per_node = gpus_per_node

        # Auto-detect CPUs if not specified
        if cpus_per_node is None:
            cpus_per_node = os.cpu_count() or 8
        self._cpus_per_node = cpus_per_node

        # Auto-detect memory if not specified
        if memory_per_node_gb is None:
            memory_per_node_gb = self._detect_memory_gb()
        self._memory_per_node_gb = memory_per_node_gb

        # Initialize resource tracking
        self._gpus = Resource(
            total=num_nodes * gpus_per_node,
            resource_type=ResourceType.GPU,
        )
        self._cpus = Resource(
            total=num_nodes * cpus_per_node,
            resource_type=ResourceType.CPU,
        )
        self._memory = Resource(
            total=num_nodes * memory_per_node_gb,
            resource_type=ResourceType.MEMORY,
        )

        # Track allocations for release
        self._allocations: dict[str, ResourceAllocation] = {}

        # Virtual cluster reference (lazily initialized)
        self._virtual_cluster: RayVirtualCluster | None = None

        logger.info(
            f"ResourceManager initialized: {self._num_nodes} nodes × "
            f"{self._gpus_per_node} GPUs = {self.total_gpus} total GPUs"
        )

    @staticmethod
    def _detect_memory_gb() -> float:
        """Detect available system memory in GB."""
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Default to 64GB if psutil is not available
            return 64.0

    @classmethod
    def from_cluster_config(
        cls, cluster_config: "ClusterConfig"
    ) -> "ResourceManager":
        """Create ResourceManager from a ClusterConfig.

        Args:
            cluster_config: Cluster configuration object.

        Returns:
            ResourceManager instance.
        """
        return cls(
            num_nodes=cluster_config.num_nodes,
            gpus_per_node=cluster_config.gpus_per_node,
        )

    @classmethod
    def auto_detect(cls) -> "ResourceManager":
        """Auto-detect cluster resources.

        Detects available GPUs, CPUs, and memory from the system
        and Ray cluster (if available).

        Returns:
            ResourceManager with detected resources.
        """
        # Try to detect from Ray cluster first
        try:
            import ray

            if ray.is_initialized():
                resources = ray.cluster_resources()
                total_gpus = int(resources.get("GPU", 0))
                total_cpus = int(resources.get("CPU", os.cpu_count() or 8))

                # Estimate nodes from GPU count (assume 8 GPUs per node)
                gpus_per_node = 8
                num_nodes = max(1, total_gpus // gpus_per_node)
                if total_gpus > 0 and total_gpus < gpus_per_node:
                    gpus_per_node = total_gpus

                return cls(
                    num_nodes=num_nodes,
                    gpus_per_node=gpus_per_node,
                    cpus_per_node=total_cpus // num_nodes,
                )
        except (ImportError, Exception):
            pass

        # Fall back to local detection
        try:
            import torch

            gpus = torch.cuda.device_count()
        except (ImportError, RuntimeError):
            gpus = 0

        # Check CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            gpus = len(cuda_devices.split(","))

        return cls(
            num_nodes=1,
            gpus_per_node=max(1, gpus),
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the cluster."""
        return self._num_nodes

    @property
    def gpus_per_node(self) -> int:
        """Number of GPUs per node."""
        return self._gpus_per_node

    @property
    def cpus_per_node(self) -> int:
        """Number of CPUs per node."""
        return self._cpus_per_node

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs in the cluster."""
        return int(self._gpus.total)

    @property
    def total_cpus(self) -> int:
        """Total number of CPUs in the cluster."""
        return int(self._cpus.total)

    @property
    def total_memory_gb(self) -> float:
        """Total memory in the cluster (GB)."""
        return self._memory.total

    @property
    def allocated_gpus(self) -> int:
        """Number of GPUs currently allocated."""
        return int(self._gpus.allocated)

    @property
    def allocated_cpus(self) -> int:
        """Number of CPUs currently allocated."""
        return int(self._cpus.allocated)

    @property
    def allocated_memory_gb(self) -> float:
        """Amount of memory currently allocated (GB)."""
        return self._memory.allocated

    @property
    def available_gpus(self) -> int:
        """Number of GPUs currently available."""
        return int(self._gpus.available)

    @property
    def available_cpus(self) -> int:
        """Number of CPUs currently available."""
        return int(self._cpus.available)

    @property
    def available_memory_gb(self) -> float:
        """Amount of memory currently available (GB)."""
        return self._memory.available

    @property
    def gpu_utilization(self) -> float:
        """GPU utilization as a fraction (0.0 to 1.0)."""
        return self._gpus.utilization

    @property
    def cpu_utilization(self) -> float:
        """CPU utilization as a fraction (0.0 to 1.0)."""
        return self._cpus.utilization

    @property
    def memory_utilization(self) -> float:
        """Memory utilization as a fraction (0.0 to 1.0)."""
        return self._memory.utilization

    # =========================================================================
    # Allocation Methods
    # =========================================================================

    def can_allocate(
        self,
        gpus: int = 0,
        cpus: int = 0,
        memory_gb: float = 0.0,
    ) -> bool:
        """Check if the requested resources can be allocated.

        Args:
            gpus: Number of GPUs to allocate.
            cpus: Number of CPUs to allocate.
            memory_gb: Amount of memory to allocate (GB).

        Returns:
            True if allocation is possible.
        """
        return (
            self._gpus.can_allocate(gpus)
            and self._cpus.can_allocate(cpus)
            and self._memory.can_allocate(memory_gb)
        )

    def allocate(
        self,
        gpus: int = 0,
        cpus: int = 0,
        memory_gb: float = 0.0,
        worker_group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResourceAllocation:
        """Allocate resources.

        Args:
            gpus: Number of GPUs to allocate.
            cpus: Number of CPUs to allocate.
            memory_gb: Amount of memory to allocate (GB).
            worker_group_id: Optional worker group identifier.
            metadata: Optional allocation metadata.

        Returns:
            ResourceAllocation object that can be used to release resources.

        Raises:
            AllocationError: If insufficient resources are available.
        """
        # Check all resources before allocating any
        if not self._gpus.can_allocate(gpus):
            raise AllocationError(
                f"Insufficient GPUs: requested {gpus}, available {self.available_gpus}",
                resource_type=ResourceType.GPU,
                requested=gpus,
                available=self.available_gpus,
            )
        if not self._cpus.can_allocate(cpus):
            raise AllocationError(
                f"Insufficient CPUs: requested {cpus}, available {self.available_cpus}",
                resource_type=ResourceType.CPU,
                requested=cpus,
                available=self.available_cpus,
            )
        if not self._memory.can_allocate(memory_gb):
            raise AllocationError(
                f"Insufficient memory: requested {memory_gb}GB, available {self.available_memory_gb}GB",
                resource_type=ResourceType.MEMORY,
                requested=memory_gb,
                available=self.available_memory_gb,
            )

        # Allocate resources
        self._gpus.allocate(gpus)
        self._cpus.allocate(cpus)
        self._memory.allocate(memory_gb)

        # Create allocation record
        allocation = ResourceAllocation(
            allocation_id=str(uuid.uuid4()),
            gpus=gpus,
            cpus=cpus,
            memory_gb=memory_gb,
            worker_group_id=worker_group_id,
            metadata=metadata or {},
        )
        self._allocations[allocation.allocation_id] = allocation

        logger.debug(
            f"Allocated {gpus} GPUs, {cpus} CPUs, {memory_gb}GB memory "
            f"(allocation_id={allocation.allocation_id})"
        )

        return allocation

    def release(self, allocation: ResourceAllocation | str) -> None:
        """Release previously allocated resources.

        Args:
            allocation: ResourceAllocation object or allocation_id string.
        """
        if isinstance(allocation, str):
            allocation_id = allocation
            allocation = self._allocations.get(allocation_id)
            if allocation is None:
                logger.warning(f"Unknown allocation_id: {allocation_id}")
                return
        else:
            allocation_id = allocation.allocation_id

        if allocation_id not in self._allocations:
            logger.warning(f"Allocation already released: {allocation_id}")
            return

        # Release resources
        self._gpus.release(allocation.gpus)
        self._cpus.release(allocation.cpus)
        self._memory.release(allocation.memory_gb)

        # Remove allocation record
        del self._allocations[allocation_id]

        logger.debug(
            f"Released {allocation.gpus} GPUs, {allocation.cpus} CPUs, "
            f"{allocation.memory_gb}GB memory (allocation_id={allocation_id})"
        )

    def release_all(self) -> None:
        """Release all allocations."""
        for allocation_id in list(self._allocations.keys()):
            self.release(allocation_id)

    # =========================================================================
    # Virtual Cluster Integration
    # =========================================================================

    def get_virtual_cluster(
        self,
        bundle_ct_per_node_list: list[int] | None = None,
        max_colocated_worker_groups: int = 1,
        name: str = "",
        placement_group_strategy: str = "SPREAD",
    ) -> "RayVirtualCluster":
        """Get or create a RayVirtualCluster.

        Creates a RayVirtualCluster that uses the allocated resources.

        Args:
            bundle_ct_per_node_list: GPUs per node (defaults to [gpus_per_node] * num_nodes).
            max_colocated_worker_groups: Max worker groups that can be colocated.
            name: Name prefix for placement groups.
            placement_group_strategy: Ray placement group strategy.

        Returns:
            RayVirtualCluster instance.
        """
        from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

        if bundle_ct_per_node_list is None:
            bundle_ct_per_node_list = [self._gpus_per_node] * self._num_nodes

        return RayVirtualCluster(
            bundle_ct_per_node_list=bundle_ct_per_node_list,
            use_gpus=True,
            max_colocated_worker_groups=max_colocated_worker_groups,
            num_gpus_per_node=self._gpus_per_node,
            name=name,
            placement_group_strategy=placement_group_strategy,
        )

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get current resource status.

        Returns:
            Dictionary with resource status information.
        """
        return {
            "total_gpus": self.total_gpus,
            "allocated_gpus": self.allocated_gpus,
            "available_gpus": self.available_gpus,
            "gpu_utilization": self.gpu_utilization,
            "total_cpus": self.total_cpus,
            "allocated_cpus": self.allocated_cpus,
            "available_cpus": self.available_cpus,
            "cpu_utilization": self.cpu_utilization,
            "total_memory_gb": self.total_memory_gb,
            "allocated_memory_gb": self.allocated_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "memory_utilization": self.memory_utilization,
            "num_allocations": len(self._allocations),
            "num_nodes": self._num_nodes,
            "gpus_per_node": self._gpus_per_node,
        }

    def __repr__(self) -> str:
        return (
            f"ResourceManager("
            f"total_gpus={self.total_gpus}, "
            f"allocated_gpus={self.allocated_gpus}, "
            f"available_gpus={self.available_gpus})"
        )
