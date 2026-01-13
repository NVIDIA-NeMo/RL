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

import os

import pytest
import ray
import torch

from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossFn,
    SequencePackingLossWrapper,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


@ray.remote(num_gpus=1)
class SequencePackingGradientTestActor:
    def __init__(self, cp_size):
        self.cp_size = cp_size
        self.env_vars = dict(os.environ)

    def test_sequence_packing_gradients(self):
        from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
        from nemo_rl.models.megatron.data import (
            _pack_sequences_for_megatron,
            make_processed_microbatch_iterator,
        )
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            forward_with_post_processing_fn,
        )

        # Initialize process group
        torch.distributed.init_process_group(backend="nccl")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Create CP group - all ranks participate in CP
        cp_group = torch.distributed.new_group(ranks=list(range(world_size)))

        # Patch get_context_parallel_group to always return cp_group
        # (Assume it's imported from nemo_rl.models.megatron.common)
        import megatron.core.parallel_state as parallel_state

        parallel_state._CONTEXT_PARALLEL_GROUP = cp_group
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = torch.distributed.new_group(
            ranks=[rank]
        )

        # Test parameters
        batch_size = 4
        max_seq_len = 512
        vocab_size = 1000
        cp_size = self.cp_size

        # Ensure sequence length is compatible with CP load balancing
        if max_seq_len % (2 * cp_size) != 0:
            max_seq_len = (max_seq_len // (2 * cp_size) + 1) * (2 * cp_size)

        # Create test data with varying sequence lengths
        torch.manual_seed(42)  # For reproducibility
        seq_lengths = torch.tensor(
            [
                max_seq_len // 4,
                max_seq_len * 1 // 4,
                max_seq_len // 4,
                max_seq_len * 3 // 4,
            ]
        )

        # Create input data
        input_ids = torch.zeros(
            batch_size, max_seq_len, dtype=torch.long, device="cuda"
        )
        token_mask = torch.zeros(
            batch_size, max_seq_len, dtype=torch.float, device="cuda"
        )

        # Fill with random tokens up to seq_length
        for i in range(batch_size):
            length = seq_lengths[i]
            input_ids[i, :length] = torch.randint(
                0, vocab_size, (length,), device="cuda"
            )
            token_mask[i, :length] = 1.0

        # Create other required tensors
        sample_mask = torch.ones(batch_size, dtype=torch.float, device="cuda")
        advantages = torch.randn(batch_size, max_seq_len, device="cuda")
        prev_logprobs = torch.randn(batch_size, max_seq_len, device="cuda")
        generation_logprobs = torch.randn(batch_size, max_seq_len, device="cuda")
        reference_policy_logprobs = generation_logprobs.clone()

        original_data = {
            "input_ids": input_ids,
            "input_lengths": seq_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": advantages,
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
            "reference_policy_logprobs": reference_policy_logprobs,
        }

        # ===== TEST 1: Baseline (no sequence packing) =====
        print(f"Rank {rank}: Testing baseline (no sequence packing)")

        baseline_logits = torch.randn(
            batch_size, max_seq_len, vocab_size, requires_grad=True, device="cuda"
        )

        loss_config = {
            "reference_policy_kl_penalty": 0.1,
            "reference_policy_kl_type": "k3",
            "kl_input_clamp_value": 20.0,
            "kl_output_clamp_value": 10.0,
            "ratio_clip_min": 0.2,
            "ratio_clip_max": 0.2,
            "ratio_clip_c": 3.0,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "truncated_importance_sampling_ratio": None,
            "sequence_level_importance_ratios": False,
            "token_level_loss": True,
            "force_on_policy_ratio": False,
        }

        base_loss_fn = ClippedPGLossFn(loss_config)
        data_dict = BatchedDataDict(original_data)

        global_valid_toks = torch.tensor(
            sum(seq_lengths).item(), dtype=torch.float, device="cuda"
        )
        global_valid_seqs = torch.tensor(batch_size, dtype=torch.float, device="cuda")

        # Forward pass
        baseline_loss, baseline_metrics = base_loss_fn(
            baseline_logits,
            data_dict,
            global_valid_seqs,
            global_valid_toks,
        )

        # Backward pass
        baseline_loss.backward()

        # Check baseline gradients
        baseline_grad_norm = torch.norm(baseline_logits.grad).item()
        baseline_grad_max = torch.max(torch.abs(baseline_logits.grad)).item()
        baseline_grad_mean = torch.mean(torch.abs(baseline_logits.grad)).item()
        baseline_grad_store = baseline_logits.grad.clone()
        baseline_logits.grad.zero_()

        print(
            f"Rank {rank}: Baseline gradient stats - norm: {baseline_grad_norm:.4f}, max: {baseline_grad_max:.4f}, mean: {baseline_grad_mean:.4f}"
        )

        # ===== TEST 2: Sequence packing with context parallelism =====
        print(f"Rank {rank}: Testing with sequence packing + CP")

        # Pack sequences
        pad_to_multiple = cp_size * 2  # Common requirement for CP
        (
            packed_input_ids,
            packed_input_ids_cp,
            packed_seq_params,
            cu_seqlens,
            cu_seqlens_padded,
        ) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths,
            pad_individual_seqs_to_multiple_of=pad_to_multiple,
            pad_packed_seq_to=max_seq_len * batch_size if cp_size > 1 else None,
            cp_rank=rank,
            cp_size=cp_size,
        )

        # For CP, logits are sharded across context parallel ranks
        def make_packed_logits(logits):
            packed_logits = torch.zeros(
                1, packed_input_ids_cp.shape[1], vocab_size, device="cuda"
            )
            run_seq = 0
            for i, seq_len in enumerate(seq_lengths):
                padded_seqlen = cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]
                if padded_seqlen > baseline_logits.shape[1]:
                    # pad the logits with zeros
                    tmp_logits = torch.zeros(
                        1, padded_seqlen, vocab_size, device="cuda"
                    )
                    tmp_logits[:, :seq_len] = baseline_logits[i : i + 1, :seq_len]
                else:
                    tmp_logits = baseline_logits[i : i + 1, :padded_seqlen]
                packed_logits[
                    :, run_seq // cp_size : (run_seq + padded_seqlen) // cp_size, :
                ] = _get_tokens_on_this_cp_rank(tmp_logits, rank, cp_size)
                run_seq += padded_seqlen
            return packed_logits

        packed_logits = make_packed_logits(baseline_logits)

        # Create sequence packing wrapper
        wrapper = SequencePackingLossWrapper(
            loss_fn=base_loss_fn,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
        )

        # Create data dict for packed sequences
        packed_data_dict = BatchedDataDict(original_data)

        tp_group = torch.distributed.new_group(ranks=[rank])

        # Forward pass
        packed_loss, packed_metrics = wrapper(
            packed_logits,
            packed_data_dict,
            global_valid_seqs,
            global_valid_toks,
            vocab_parallel_rank=0,
            vocab_parallel_group=tp_group,
            context_parallel_group=cp_group,
        )

        # Backward pass
        packed_loss /= cp_size
        packed_loss.backward()

        # Check packed gradients
        packed_grad = baseline_logits.grad.clone()
        # all-reduce across cp ranks
        torch.distributed.all_reduce(packed_grad, op=torch.distributed.ReduceOp.SUM)

        packed_grad_norm = torch.norm(packed_grad).item()
        packed_grad_max = torch.max(torch.abs(packed_grad)).item()
        packed_grad_mean = torch.mean(torch.abs(packed_grad)).item()
        # print(f"max grad on dims {torch.max(torch.abs(packed_grad), dim=0)}, {torch.max(torch.abs(packed_grad), dim=1)}, {torch.max(torch.abs(packed_grad), dim=2)}")

        print(
            f"Rank {rank}: Packed gradient stats - norm: {packed_grad_norm:.4f}, max: {packed_grad_max:.4f}, mean: {packed_grad_mean:.4f}"
        )

        # ===== ANALYSIS =====
        gradient_ratio_norm = (
            packed_grad_norm / baseline_grad_norm
            if baseline_grad_norm > 0
            else float("inf")
        )
        gradient_ratio_max = (
            packed_grad_max / baseline_grad_max
            if baseline_grad_max > 0
            else float("inf")
        )
        gradient_ratio_mean = (
            packed_grad_mean / baseline_grad_mean
            if baseline_grad_mean > 0
            else float("inf")
        )

        print(
            f"Rank {rank}: Gradient ratios - norm: {gradient_ratio_norm:.4f}, max: {gradient_ratio_max:.4f}, mean: {gradient_ratio_mean:.4f}"
        )

        print(
            f"differences by token: {torch.sum(torch.abs(packed_grad - baseline_grad_store), dim=-1)}"
        )

        torch.testing.assert_close(
            packed_grad, baseline_grad_store, atol=1e-5, rtol=1e-5
        )

        # test 3: with forward_with_post_processing_fn
        # reset grad
        baseline_logits.grad.zero_()
        packed_logits = make_packed_logits(baseline_logits)

        # mock model forward
        class MockModel:
            def __init__(self):
                self.logits = packed_logits

            def __call__(self, *args, **kwargs):
                return self.logits

            def forward(
                self, input_ids, position_ids, attention_mask, packed_seq_params=None
            ):
                return self.logits

        cfg = {
            "sequence_packing": {"enabled": True},
            "dynamic_batching": {"enabled": False},
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": cp_size,
            },
        }

        post_processor = LossPostProcessor(
            loss_fn=base_loss_fn,
            cfg=cfg,
            cp_normalize=True,
        )

        output_tensor, wrapped_loss_fn = forward_with_post_processing_fn(
            data_iterator=make_processed_microbatch_iterator(
                iter([packed_data_dict]),
                cfg=cfg,
                seq_length_key="input_lengths",
                pad_individual_seqs_to_multiple_of=pad_to_multiple,
                pad_packed_seq_to_multiple_of=1,
                pad_full_seq_to=max_seq_len * batch_size if cp_size > 1 else None,
            ),
            model=MockModel(),
            cfg=cfg,
            post_processing_fn=post_processor,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )
        loss, metrics = wrapped_loss_fn(output_tensor)

        loss.backward()

        # Check packed gradients
        packed_grad = baseline_logits.grad.clone()
        # all-reduce across cp ranks
        torch.distributed.all_reduce(packed_grad, op=torch.distributed.ReduceOp.SUM)

        packed_grad_norm = torch.norm(packed_grad).item()
        packed_grad_max = torch.max(torch.abs(packed_grad)).item()
        packed_grad_mean = torch.mean(torch.abs(packed_grad)).item()
        print(
            f"Rank {rank}: Packed gradient stats - norm: {packed_grad_norm:.4f}, max: {packed_grad_max:.4f}, mean: {packed_grad_mean:.4f}"
        )

        gradient_ratio_norm = (
            packed_grad_norm / baseline_grad_norm
            if baseline_grad_norm > 0
            else float("inf")
        )
        gradient_ratio_max = (
            packed_grad_max / baseline_grad_max
            if baseline_grad_max > 0
            else float("inf")
        )

        print(
            f"Rank {rank}: Gradient ratios - norm: {gradient_ratio_norm:.4f}, max: {gradient_ratio_max:.4f}"
        )
        print(
            f"differences by token: {torch.sum(torch.abs(packed_grad - baseline_grad_store), dim=-1)}"
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
