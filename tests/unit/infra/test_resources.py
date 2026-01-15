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
"""Tests for resource management."""

import pytest

from nemo_rl.infra.resources import (
    AllocationError,
    Resource,
    ResourceAllocation,
    ResourceManager,
    ResourceType,
)


class TestResource:
    """Tests for Resource class."""

    def test_init_defaults(self):
        """Test resource initialization with defaults."""
        resource = Resource(total=8)
        assert resource.total == 8
        assert resource.allocated == 0
        assert resource.available == 8
        assert resource.resource_type == ResourceType.GPU

    def test_available_property(self):
        """Test available property calculation."""
        resource = Resource(total=8, allocated=3)
        assert resource.available == 5

    def test_utilization_property(self):
        """Test utilization property calculation."""
        resource = Resource(total=8, allocated=4)
        assert resource.utilization == 0.5

    def test_utilization_zero_total(self):
        """Test utilization with zero total."""
        resource = Resource(total=0)
        assert resource.utilization == 0.0

    def test_can_allocate_true(self):
        """Test can_allocate returns True when sufficient resources."""
        resource = Resource(total=8, allocated=3)
        assert resource.can_allocate(5) is True

    def test_can_allocate_false(self):
        """Test can_allocate returns False when insufficient resources."""
        resource = Resource(total=8, allocated=3)
        assert resource.can_allocate(6) is False

    def test_allocate_success(self):
        """Test successful allocation."""
        resource = Resource(total=8)
        resource.allocate(3)
        assert resource.allocated == 3
        assert resource.available == 5

    def test_allocate_fails_when_insufficient(self):
        """Test allocation fails when insufficient resources."""
        resource = Resource(total=8, allocated=6, resource_type=ResourceType.GPU)
        with pytest.raises(AllocationError) as exc_info:
            resource.allocate(3)
        assert exc_info.value.resource_type == ResourceType.GPU
        assert exc_info.value.requested == 3
        assert exc_info.value.available == 2

    def test_release(self):
        """Test resource release."""
        resource = Resource(total=8, allocated=5)
        resource.release(3)
        assert resource.allocated == 2
        assert resource.available == 6

    def test_release_clamps_to_allocated(self):
        """Test release clamps to allocated amount."""
        resource = Resource(total=8, allocated=3)
        resource.release(10)  # Try to release more than allocated
        assert resource.allocated == 0


class TestResourceAllocation:
    """Tests for ResourceAllocation class."""

    def test_create_allocation(self):
        """Test creating an allocation."""
        allocation = ResourceAllocation(
            allocation_id="test-123",
            gpus=4,
            cpus=32,
            memory_gb=64.0,
        )
        assert allocation.allocation_id == "test-123"
        assert allocation.gpus == 4
        assert allocation.cpus == 32
        assert allocation.memory_gb == 64.0

    def test_default_values(self):
        """Test allocation default values."""
        allocation = ResourceAllocation(allocation_id="test")
        assert allocation.gpus == 0
        assert allocation.cpus == 0
        assert allocation.memory_gb == 0.0
        assert allocation.worker_group_id is None
        assert allocation.metadata == {}


class TestResourceManager:
    """Tests for ResourceManager class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        manager = ResourceManager()
        assert manager.num_nodes == 1
        assert manager.gpus_per_node == 8
        assert manager.total_gpus == 8

    def test_init_custom(self):
        """Test initialization with custom values."""
        manager = ResourceManager(num_nodes=4, gpus_per_node=4)
        assert manager.num_nodes == 4
        assert manager.gpus_per_node == 4
        assert manager.total_gpus == 16

    def test_total_gpus(self):
        """Test total_gpus calculation."""
        manager = ResourceManager(num_nodes=2, gpus_per_node=8)
        assert manager.total_gpus == 16

    def test_initial_available_gpus(self):
        """Test available_gpus is total initially."""
        manager = ResourceManager(num_nodes=2, gpus_per_node=8)
        assert manager.available_gpus == 16
        assert manager.allocated_gpus == 0

    def test_can_allocate_true(self):
        """Test can_allocate returns True when resources available."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        assert manager.can_allocate(gpus=4) is True

    def test_can_allocate_false(self):
        """Test can_allocate returns False when insufficient resources."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        assert manager.can_allocate(gpus=10) is False

    def test_allocate_success(self):
        """Test successful allocation."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        allocation = manager.allocate(gpus=4, cpus=16)

        assert allocation.gpus == 4
        assert allocation.cpus == 16
        assert manager.allocated_gpus == 4
        assert manager.available_gpus == 4

    def test_allocate_fails_when_insufficient(self):
        """Test allocation fails when insufficient resources."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        with pytest.raises(AllocationError) as exc_info:
            manager.allocate(gpus=10)

        assert exc_info.value.resource_type == ResourceType.GPU
        assert exc_info.value.requested == 10
        assert exc_info.value.available == 8

    def test_release_by_allocation(self):
        """Test release by allocation object."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        allocation = manager.allocate(gpus=4)

        assert manager.available_gpus == 4
        manager.release(allocation)
        assert manager.available_gpus == 8

    def test_release_by_id(self):
        """Test release by allocation ID."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        allocation = manager.allocate(gpus=4)

        manager.release(allocation.allocation_id)
        assert manager.available_gpus == 8

    def test_release_unknown_id(self):
        """Test release with unknown ID doesn't crash."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        manager.release("unknown-id")  # Should not raise

    def test_release_already_released(self):
        """Test releasing already released allocation."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        allocation = manager.allocate(gpus=4)
        manager.release(allocation)
        manager.release(allocation)  # Should not raise

    def test_release_all(self):
        """Test release_all releases all allocations."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        manager.allocate(gpus=2)
        manager.allocate(gpus=3)

        assert manager.allocated_gpus == 5
        manager.release_all()
        assert manager.allocated_gpus == 0
        assert manager.available_gpus == 8

    def test_multiple_allocations(self):
        """Test multiple sequential allocations."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)

        alloc1 = manager.allocate(gpus=2)
        alloc2 = manager.allocate(gpus=3)

        assert manager.allocated_gpus == 5
        assert manager.available_gpus == 3

        manager.release(alloc1)
        assert manager.allocated_gpus == 3
        assert manager.available_gpus == 5

    def test_gpu_utilization(self):
        """Test GPU utilization calculation."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        manager.allocate(gpus=4)
        assert manager.gpu_utilization == 0.5

    def test_get_status(self):
        """Test get_status returns correct information."""
        manager = ResourceManager(num_nodes=2, gpus_per_node=4)
        manager.allocate(gpus=4)

        status = manager.get_status()
        assert status["total_gpus"] == 8
        assert status["allocated_gpus"] == 4
        assert status["available_gpus"] == 4
        assert status["gpu_utilization"] == 0.5
        assert status["num_nodes"] == 2
        assert status["gpus_per_node"] == 4
        assert status["num_allocations"] == 1

    def test_repr(self):
        """Test string representation."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        manager.allocate(gpus=3)
        repr_str = repr(manager)
        assert "total_gpus=8" in repr_str
        assert "allocated_gpus=3" in repr_str
        assert "available_gpus=5" in repr_str

    def test_from_cluster_config(self):
        """Test creating ResourceManager from ClusterConfig."""
        from nemo_rl.config.cluster import ClusterConfig

        cluster_config = ClusterConfig(num_nodes=4, gpus_per_node=8)
        manager = ResourceManager.from_cluster_config(cluster_config)

        assert manager.num_nodes == 4
        assert manager.gpus_per_node == 8
        assert manager.total_gpus == 32

    def test_allocate_with_metadata(self):
        """Test allocation with metadata."""
        manager = ResourceManager(num_nodes=1, gpus_per_node=8)
        allocation = manager.allocate(
            gpus=4,
            worker_group_id="policy-workers",
            metadata={"purpose": "training"},
        )

        assert allocation.worker_group_id == "policy-workers"
        assert allocation.metadata["purpose"] == "training"


class TestAllocationError:
    """Tests for AllocationError class."""

    def test_basic_error(self):
        """Test basic error message."""
        error = AllocationError("Not enough GPUs")
        assert "Not enough GPUs" in str(error)

    def test_error_with_details(self):
        """Test error with resource details."""
        error = AllocationError(
            "Insufficient resources",
            resource_type=ResourceType.GPU,
            requested=10,
            available=8,
        )
        assert error.resource_type == ResourceType.GPU
        assert error.requested == 10
        assert error.available == 8
