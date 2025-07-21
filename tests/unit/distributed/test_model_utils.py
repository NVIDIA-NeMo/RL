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
import os

import pytest
import ray
import torch

from nemo_rl.distributed.model_utils import (
    _get_tokens_on_this_cp_rank,
    allgather_cp_sharded_tensor,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


@ray.remote(num_gpus=1)
class ModelUtilsTestActor:
    def __init__(self, tp_size, cp_size, sharding):
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.sharding = sharding
        self.env_vars = dict(os.environ)

    def test_packed_sequences_equivalence(self):
        """Test that packed and unpacked functions return the same results."""
        # Initialize worker groups
        torch.distributed.init_process_group(backend="nccl")

        tp_rank = int(os.environ["RANK"]) % self.tp_size
        cp_rank = int(os.environ["RANK"]) // self.tp_size
        tp_ranks = self.sharding.get_ranks(tp=tp_rank)
        if type(tp_ranks) != int:
            tp_ranks = tp_ranks.layout.tolist()
        else:
            tp_ranks = [tp_ranks]
        cp_ranks = self.sharding.get_ranks(cp=cp_rank)
        if type(cp_ranks) != int:
            cp_ranks = cp_ranks.layout.tolist()
        else:
            cp_ranks = [cp_ranks]

        tp_group = torch.distributed.new_group(ranks=cp_ranks)
        cp_group = torch.distributed.new_group(ranks=tp_ranks)  # this is correct

        # Test parameters
        batch_size = 4
        seq_len = 32
        vocab_size = 1024

        if self.cp_size > 1 and seq_len % (2 * self.cp_size) != 0:
            seq_len = (seq_len // (2 * self.cp_size) + 1) * (2 * self.cp_size)

        vocab_part_size = vocab_size // self.tp_size
        vocab_start_index = tp_rank * vocab_part_size
        vocab_end_index = (tp_rank + 1) * vocab_part_size

        unpacked_seq_len = seq_len

        # Create random data
        torch.manual_seed(42)  # For reproducibility
        unpacked_logits = torch.randn(
            batch_size, unpacked_seq_len, vocab_part_size, device="cuda"
        )
        unpacked_target_ids = (
            torch.arange(batch_size * seq_len).reshape(batch_size, seq_len).to("cuda")
        )

        # 1. Get expected logprobs from non-packed function
        expected_logprobs = from_parallel_logits_to_logprobs(
            unpacked_logits,
            unpacked_target_ids,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            cp_group=None,
        )

        # 1.5 get with_cp logprobs
        with_cp_logprobs = from_parallel_logits_to_logprobs(
            _get_tokens_on_this_cp_rank(
                unpacked_logits, cp_rank, self.cp_size, seq_dim=1
            ),
            unpacked_target_ids,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            cp_group=cp_group,
        )

        torch.testing.assert_close(
            with_cp_logprobs, expected_logprobs, rtol=1e-5, atol=1e-5
        )

        # 2. Prepare inputs for packed function
        # For simplicity, all sequences have the same length
        seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seq_lengths, dim=0, dtype=torch.int32), (1, 0)
        ).to("cuda")

        # Pack the logits and target_ids
        packed_logits = _get_tokens_on_this_cp_rank(
            unpacked_logits, cp_rank, self.cp_size, seq_dim=1
        ).reshape(1, -1, vocab_part_size)
        packed_target_ids = unpacked_target_ids.reshape(1, -1)

        # 3. Get actual logprobs from packed function
        actual_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
            packed_logits,
            packed_target_ids,
            cu_seqlens,
            seq_len,  # unpacked_seqlen
            vocab_start_index,
            vocab_end_index,
            tp_group,
            cp_group=cp_group,
        )

        # 4. Compare results
        torch.testing.assert_close(
            actual_logprobs, expected_logprobs, rtol=1e-5, atol=1e-5
        )
        return {"success": True, "error": None}


MODEL_UTILS_TEST_ACTOR_FQN = f"{ModelUtilsTestActor.__module__}.ModelUtilsTestActor"


@pytest.fixture
def register_model_utils_test_actor():
    """Register the ModelUtilsTestActor for use in tests."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(MODEL_UTILS_TEST_ACTOR_FQN)
    ACTOR_ENVIRONMENT_REGISTRY[MODEL_UTILS_TEST_ACTOR_FQN] = PY_EXECUTABLES.SYSTEM

    yield MODEL_UTILS_TEST_ACTOR_FQN

    # Clean up registry
    if MODEL_UTILS_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[MODEL_UTILS_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[MODEL_UTILS_TEST_ACTOR_FQN] = (
                original_registry_value
            )


@pytest.fixture
def virtual_cluster_2_gpus():
    """Create a virtual cluster with 2 GPU bundles."""
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2], use_gpus=True)
    yield cluster
    cluster.shutdown()


@pytest.fixture
def virtual_cluster_4_gpus():
    """Create a virtual cluster with 4 GPU bundles."""
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[4], use_gpus=True)
    yield cluster
    cluster.shutdown()


import numpy as np


@pytest.mark.parametrize(
    "tp_cp_config",
    [
        (2, 1),  # TP=2, CP=1
        (1, 2),  # TP=1, CP=2
    ],
)
def test_from_parallel_logits_to_logprobs_packed_sequences(
    register_model_utils_test_actor, tp_cp_config
):
    """Test packed sequences function against unpacked version."""
    tp_size, cp_size = tp_cp_config
    world_size = tp_size * cp_size

    # Skip if not enough GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(
            f"Not enough GPUs available. Need {world_size}, got {torch.cuda.device_count()}"
        )

    # Create appropriate virtual cluster
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2], use_gpus=True)

    try:
        actor_fqn = register_model_utils_test_actor

        sharding = NamedSharding(
            layout=np.arange(world_size).reshape(tp_size, cp_size), names=["tp", "cp"]
        )
        builder = RayWorkerBuilder(actor_fqn, tp_size, cp_size, sharding)

        worker_group = RayWorkerGroup(
            cluster=cluster,
            remote_worker_builder=builder,
            workers_per_node=None,
            sharding_annotations=sharding,
        )

        # Run the test on all workers
        futures = worker_group.run_all_workers_single_data(
            "test_packed_sequences_equivalence"
        )
        results = ray.get(futures)

        # Check that all workers succeeded
        for i, result in enumerate(results):
            assert result["success"], f"Worker {i} failed: {result['error']}"

        worker_group.shutdown(force=True)

    finally:
        cluster.shutdown()


@ray.remote(num_gpus=1)
class AllGatherCPTestActor:
    def __init__(self, cp_size):
        self.cp_size = cp_size
        self.env_vars = dict(os.environ)

    def test_allgather_cp_tensor(self):
        """Test that allgather_cp_sharded_tensor correctly reconstructs tensors."""
        # Initialize process group
        torch.distributed.init_process_group(backend="nccl")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Create CP group - all ranks participate in CP
        cp_group = torch.distributed.new_group(ranks=list(range(world_size)))

        # Test parameters
        batch_size = 2
        original_seq_len = 8
        hidden_size = 16

        # Ensure sequence length is compatible with CP load balancing
        if original_seq_len % (2 * self.cp_size) != 0:
            original_seq_len = (original_seq_len // (2 * self.cp_size) + 1) * (
                2 * self.cp_size
            )

        # Create original tensor (same on all ranks for testing)
        torch.manual_seed(42)  # Same seed for reproducibility
        original_tensor = (
            torch.arange(
                batch_size * original_seq_len * hidden_size, dtype=torch.float32
            )
            .reshape(batch_size, original_seq_len, hidden_size)
            .to("cuda")
        )
        original_tensor.requires_grad = True

        # Shard the tensor using CP logic
        sharded_tensor = _get_tokens_on_this_cp_rank(
            original_tensor, rank, self.cp_size, seq_dim=1
        )

        # Test 1: Gather sharded tensor and verify it matches original
        gathered_tensor = allgather_cp_sharded_tensor(
            sharded_tensor, cp_group, seq_dim=1
        )

        # Verify shapes match
        if gathered_tensor.shape != original_tensor.shape:
            return {
                "success": False,
                "error": f"Shape mismatch: expected {original_tensor.shape}, got {gathered_tensor.shape}",
            }

        # Verify content matches (should be identical)
        torch.testing.assert_close(
            gathered_tensor, original_tensor, rtol=1e-5, atol=1e-5
        )

        # test backward
        def loss_fn(x):
            return torch.sum(x**2)

        loss = loss_fn(gathered_tensor)
        loss.backward()
        grad = original_tensor.grad / self.cp_size
        grad_sharded = _get_tokens_on_this_cp_rank(grad, rank, self.cp_size, seq_dim=1)

        torch.testing.assert_close(
            grad_sharded,
            _get_tokens_on_this_cp_rank(
                2 * original_tensor, rank, self.cp_size, seq_dim=1
            ),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            _get_tokens_on_this_cp_rank(
                grad, (rank + 1) % self.cp_size, self.cp_size, seq_dim=1
            ),
            torch.zeros_like(sharded_tensor),
            rtol=1e-5,
            atol=1e-5,
        )

        # Test 2: Test with different sequence dimension (seq_dim=0)
        # Create a tensor with sequence dimension at dim=0
        original_tensor_dim0 = torch.randn(
            original_seq_len, batch_size, hidden_size, device="cuda"
        )

        sharded_tensor_dim0 = _get_tokens_on_this_cp_rank(
            original_tensor_dim0, rank, self.cp_size, seq_dim=0
        )

        gathered_tensor_dim0 = allgather_cp_sharded_tensor(
            sharded_tensor_dim0, cp_group, seq_dim=0
        )

        # Verify shapes and content match
        if gathered_tensor_dim0.shape != original_tensor_dim0.shape:
            return {
                "success": False,
                "error": f"Shape mismatch for seq_dim=0: expected {original_tensor_dim0.shape}, got {gathered_tensor_dim0.shape}",
            }

        torch.testing.assert_close(
            gathered_tensor_dim0, original_tensor_dim0, rtol=1e-5, atol=1e-5
        )

        # Test 3: Test with different tensor shapes
        # Test with 2D tensor
        original_2d = torch.randn(original_seq_len, hidden_size, device="cuda")
        sharded_2d = _get_tokens_on_this_cp_rank(
            original_2d, rank, self.cp_size, seq_dim=0
        )
        gathered_2d = allgather_cp_sharded_tensor(sharded_2d, cp_group, seq_dim=0)

        torch.testing.assert_close(gathered_2d, original_2d, rtol=1e-5, atol=1e-5)

        return {"success": True, "error": None}


ALLGATHER_CP_TEST_ACTOR_FQN = f"{AllGatherCPTestActor.__module__}.AllGatherCPTestActor"


@pytest.fixture
def register_allgather_cp_test_actor():
    """Register the AllGatherCPTestActor for use in tests."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(
        ALLGATHER_CP_TEST_ACTOR_FQN
    )
    ACTOR_ENVIRONMENT_REGISTRY[ALLGATHER_CP_TEST_ACTOR_FQN] = PY_EXECUTABLES.SYSTEM

    yield ALLGATHER_CP_TEST_ACTOR_FQN

    # Clean up registry
    if ALLGATHER_CP_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[ALLGATHER_CP_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[ALLGATHER_CP_TEST_ACTOR_FQN] = (
                original_registry_value
            )


@pytest.mark.parametrize("cp_size", [2])
def test_allgather_cp_sharded_tensor(register_allgather_cp_test_actor, cp_size):
    """Test allgather_cp_sharded_tensor function."""
    # Skip if not enough GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
        pytest.skip(
            f"Not enough GPUs available. Need {cp_size}, got {torch.cuda.device_count()}"
        )

    # Create virtual cluster
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[cp_size], use_gpus=True)

    try:
        actor_fqn = register_allgather_cp_test_actor

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
        futures = worker_group.run_all_workers_single_data("test_allgather_cp_tensor")
        results = ray.get(futures)

        # Check that all workers succeeded
        for i, result in enumerate(results):
            assert result["success"], f"Worker {i} failed: {result['error']}"

        worker_group.shutdown(force=True)

    finally:
        cluster.shutdown()
