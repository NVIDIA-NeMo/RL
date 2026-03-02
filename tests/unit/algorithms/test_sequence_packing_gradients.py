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
"""Test script to debug high gradients with sequence packing + context parallelism."""

import pytest
import ray
import torch

from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from tests.unit.algorithms.sequence_packing_gradient_actor import (
    SequencePackingGradientTestActor,
)

SEQUENCE_PACKING_GRADIENT_TEST_ACTOR_FQN = (
    f"{SequencePackingGradientTestActor.__module__}.SequencePackingGradientTestActor"
)


@pytest.fixture
def register_sequence_packing_gradient_test_actor():
    """Register the SequencePackingGradientTestActor for use in tests."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(
        SEQUENCE_PACKING_GRADIENT_TEST_ACTOR_FQN
    )
    ACTOR_ENVIRONMENT_REGISTRY[SEQUENCE_PACKING_GRADIENT_TEST_ACTOR_FQN] = (
        PY_EXECUTABLES.MCORE
    )

    yield SEQUENCE_PACKING_GRADIENT_TEST_ACTOR_FQN

    # Clean up registry
    if SEQUENCE_PACKING_GRADIENT_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[SEQUENCE_PACKING_GRADIENT_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[SEQUENCE_PACKING_GRADIENT_TEST_ACTOR_FQN] = (
                original_registry_value
            )


@pytest.fixture(scope="function")
def cluster_fixture(request):
    """Create and teardown a virtual cluster for CP tests."""
    cp_size = request.node.callspec.params["cp_size"]

    # Skip if not enough GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
        pytest.skip(
            f"Not enough GPUs available. Need {cp_size}, got {torch.cuda.device_count()}"
        )

    # Mysteriously, Ray is not initialized in this test, so we need to initialize it here.
    if not ray.is_initialized():
        print("Ray not initialized, initializing now...")
        from nemo_rl.distributed.virtual_cluster import init_ray

        init_ray()
        print("Ray initialized successfully")
    else:
        print("Ray is already initialized")

    cluster_name = f"test-sequence-packing-cp{cp_size}"
    print(f"Creating virtual cluster '{cluster_name}' for {cp_size} GPUs...")

    cluster = RayVirtualCluster(
        name=cluster_name, bundle_ct_per_node_list=[cp_size], use_gpus=True
    )
    yield cluster
    print(f"Shutting down cluster '{cluster_name}'...")
    cluster.shutdown()


@pytest.mark.parametrize("cp_size", [1, 2])
def test_sequence_packing_gradients_with_cp(
    cluster_fixture, register_sequence_packing_gradient_test_actor, cp_size
):
    """Test sequence packing gradients with context parallelism."""
    cluster = cluster_fixture
    actor_fqn = register_sequence_packing_gradient_test_actor

    # For CP, all ranks are in a single group
    sharding = NamedSharding(layout=list(range(cp_size)), names=["cp"])
    builder = RayWorkerBuilder(actor_fqn, cp_size)

    worker_group = RayWorkerGroup(
        cluster=cluster,
        remote_worker_builder=builder,
        workers_per_node=None,
        sharding_annotations=sharding,
    )

    # Run the test on all workers
    futures = worker_group.run_all_workers_single_data(
        "test_sequence_packing_gradients"
    )
    _ = ray.get(futures)
    worker_group.shutdown(force=True)
