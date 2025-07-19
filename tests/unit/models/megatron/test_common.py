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

from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


@ray.remote(num_gpus=1)
class PackSequencesTestActor:
    def __init__(self, cp_size):
        self.cp_size = cp_size
        self.env_vars = dict(os.environ)

    def run_all_pack_sequences_tests(self):
        """Run all sequence packing tests in a single call to avoid expensive reinitializations."""
        from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
        from nemo_rl.models.megatron.common import _pack_sequences_for_megatron

        # Initialize process group if CP > 1
        if self.cp_size > 1:
            torch.distributed.init_process_group(backend="nccl")
            rank = int(os.environ["RANK"])
        else:
            rank = 0

        results = {}

        # Test 1: Basic packing functionality
        results["basic"] = self._test_basic_packing(_pack_sequences_for_megatron)
        if not results["basic"]["success"]:
            return results["basic"]

        # Test 2: Variable sequence lengths
        results["variable_lengths"] = self._test_variable_lengths(
            _pack_sequences_for_megatron
        )
        if not results["variable_lengths"]["success"]:
            return results["variable_lengths"]

        # Test 3: Content preservation and consistency
        results["consistency"] = self._test_consistency(_pack_sequences_for_megatron)
        if not results["consistency"]["success"]:
            return results["consistency"]

        # Test 4: Edge cases
        results["edge_cases"] = self._test_edge_cases(_pack_sequences_for_megatron)
        if not results["edge_cases"]["success"]:
            return results["edge_cases"]

        # Test 5: Context parallelism (only if CP > 1)
        if self.cp_size > 1:
            results["context_parallel"] = self._test_context_parallel(
                _pack_sequences_for_megatron, _get_tokens_on_this_cp_rank, rank
            )
            if not results["context_parallel"]["success"]:
                return results["context_parallel"]
        else:
            results["context_parallel"] = {
                "success": True,
                "error": None,
                "skipped": "CP=1",
            }

        return {"success": True, "error": None, "detailed_results": results}

    def _test_basic_packing(self, _pack_sequences_for_megatron):
        """Test basic sequence packing without context parallelism."""
        try:
            # Test parameters
            batch_size = 3
            max_seq_len = 10
            vocab_size = 100

            # Create test data with variable sequence lengths
            input_ids = torch.randint(
                0, vocab_size, (batch_size, max_seq_len), device="cuda"
            )
            seq_lengths = torch.tensor([8, 5, 7], device="cuda")

            # Test 1: Basic packing without CP
            packed_input_ids, _, packed_seq_params, cu_seqlens, cu_seqlens_padded = (
                _pack_sequences_for_megatron(
                    input_ids, seq_lengths, cp_rank=0, cp_size=1
                )
            )

            # Verify shapes
            expected_total_tokens = seq_lengths.sum().item()
            if packed_input_ids.shape != (1, expected_total_tokens):
                return {
                    "success": False,
                    "error": f"Basic packing shape mismatch: expected (1, {expected_total_tokens}), got {packed_input_ids.shape}",
                }

            # Verify cu_seqlens
            expected_cu_seqlens = torch.tensor(
                [0, 8, 13, 20], device="cuda", dtype=torch.int32
            )
            if not torch.equal(cu_seqlens, expected_cu_seqlens):
                return {
                    "success": False,
                    "error": f"cu_seqlens mismatch: expected {expected_cu_seqlens}, got {cu_seqlens}",
                }

            # Verify PackedSeqParams
            if packed_seq_params.qkv_format != "thd":
                return {
                    "success": False,
                    "error": f"Wrong qkv_format: expected 'thd', got {packed_seq_params.qkv_format}",
                }

            if packed_seq_params.max_seqlen_q != 8:
                return {
                    "success": False,
                    "error": f"Wrong max_seqlen_q: expected 8, got {packed_seq_params.max_seqlen_q}",
                }

            # Test 2: Packing with individual sequence padding
            (
                packed_input_ids_pad,
                _,
                packed_seq_params_pad,
                cu_seqlens_pad,
                cu_seqlens_padded_pad,
            ) = _pack_sequences_for_megatron(
                input_ids,
                seq_lengths,
                pad_individual_seqs_to_multiple_of=4,
                cp_rank=0,
                cp_size=1,
            )

            # With padding to multiple of 4: [8, 5, 7] -> [8, 8, 8] = 24 tokens
            expected_total_tokens_pad = 24
            if packed_input_ids_pad.shape != (1, expected_total_tokens_pad):
                return {
                    "success": False,
                    "error": f"Padded packing shape mismatch: expected (1, {expected_total_tokens_pad}), got {packed_input_ids_pad.shape}",
                }

            # Verify padded cu_seqlens
            expected_cu_seqlens_padded = torch.tensor(
                [0, 8, 16, 24], device="cuda", dtype=torch.int32
            )
            if not torch.equal(cu_seqlens_padded_pad, expected_cu_seqlens_padded):
                return {
                    "success": False,
                    "error": f"Padded cu_seqlens mismatch: expected {expected_cu_seqlens_padded}, got {cu_seqlens_padded_pad}",
                }

            return {"success": True, "error": None}

        except Exception as e:
            return {"success": False, "error": f"Basic packing test failed: {str(e)}"}

    def _test_variable_lengths(self, _pack_sequences_for_megatron):
        """Test sequence packing with variable sequence lengths."""
        try:
            # Test parameters
            batch_size = 4
            max_seq_len = 12
            vocab_size = 50

            # Create test data with highly variable sequence lengths
            input_ids = torch.randint(
                0, vocab_size, (batch_size, max_seq_len), device="cuda"
            )
            seq_lengths = torch.tensor([12, 3, 8, 1], device="cuda")

            # Test 1: Variable lengths without padding
            packed_input_ids, _, packed_seq_params, cu_seqlens, cu_seqlens_padded = (
                _pack_sequences_for_megatron(
                    input_ids, seq_lengths, cp_rank=0, cp_size=1
                )
            )

            # Verify total tokens
            expected_total_tokens = seq_lengths.sum().item()  # 12 + 3 + 8 + 1 = 24
            if packed_input_ids.shape != (1, expected_total_tokens):
                return {
                    "success": False,
                    "error": f"Variable lengths shape mismatch: expected (1, {expected_total_tokens}), got {packed_input_ids.shape}",
                }

            # Verify cu_seqlens
            expected_cu_seqlens = torch.tensor(
                [0, 12, 15, 23, 24], device="cuda", dtype=torch.int32
            )
            if not torch.equal(cu_seqlens, expected_cu_seqlens):
                return {
                    "success": False,
                    "error": f"Variable lengths cu_seqlens mismatch: expected {expected_cu_seqlens}, got {cu_seqlens}",
                }

            # Test 2: Variable lengths with padding
            (
                packed_input_ids_pad,
                _,
                packed_seq_params_pad,
                cu_seqlens_pad,
                cu_seqlens_padded_pad,
            ) = _pack_sequences_for_megatron(
                input_ids,
                seq_lengths,
                pad_individual_seqs_to_multiple_of=4,
                cp_rank=0,
                cp_size=1,
            )

            # With padding to multiple of 4: [12, 3, 8, 1] -> [12, 4, 8, 4] = 28 tokens
            expected_total_tokens_pad = 28
            if packed_input_ids_pad.shape != (1, expected_total_tokens_pad):
                return {
                    "success": False,
                    "error": f"Variable lengths padded shape mismatch: expected (1, {expected_total_tokens_pad}), got {packed_input_ids_pad.shape}",
                }

            # Verify padded cu_seqlens
            expected_cu_seqlens_padded = torch.tensor(
                [0, 12, 16, 24, 28], device="cuda", dtype=torch.int32
            )
            if not torch.equal(cu_seqlens_padded_pad, expected_cu_seqlens_padded):
                return {
                    "success": False,
                    "error": f"Variable lengths padded cu_seqlens mismatch: expected {expected_cu_seqlens_padded}, got {cu_seqlens_padded_pad}",
                }

            # Verify max_seqlen
            if packed_seq_params.max_seqlen_q != 12:
                return {
                    "success": False,
                    "error": f"Variable lengths wrong max_seqlen_q: expected 12, got {packed_seq_params.max_seqlen_q}",
                }

            if packed_seq_params_pad.max_seqlen_q != 12:
                return {
                    "success": False,
                    "error": f"Variable lengths padded wrong max_seqlen_q: expected 12, got {packed_seq_params_pad.max_seqlen_q}",
                }

            return {"success": True, "error": None}

        except Exception as e:
            return {
                "success": False,
                "error": f"Variable lengths test failed: {str(e)}",
            }

    def _test_consistency(self, _pack_sequences_for_megatron):
        """Test that packing produces consistent results and that content is preserved."""
        try:
            # Test parameters
            batch_size = 2
            seq_len = 8
            vocab_size = 20

            # Create deterministic test data
            torch.manual_seed(123)
            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device="cuda"
            )
            seq_lengths = torch.tensor([6, 4], device="cuda")

            # Test consistency between multiple calls
            (
                packed_input_ids_1,
                _,
                packed_seq_params_1,
                cu_seqlens_1,
                cu_seqlens_padded_1,
            ) = _pack_sequences_for_megatron(
                input_ids, seq_lengths, cp_rank=0, cp_size=1
            )

            (
                packed_input_ids_2,
                _,
                packed_seq_params_2,
                cu_seqlens_2,
                cu_seqlens_padded_2,
            ) = _pack_sequences_for_megatron(
                input_ids, seq_lengths, cp_rank=0, cp_size=1
            )

            # Verify consistency
            if not torch.equal(packed_input_ids_1, packed_input_ids_2):
                return {
                    "success": False,
                    "error": "Inconsistent packed_input_ids between calls",
                }

            if not torch.equal(cu_seqlens_1, cu_seqlens_2):
                return {
                    "success": False,
                    "error": "Inconsistent cu_seqlens between calls",
                }

            # Verify content preservation
            # Extract the first sequence (length 6) and compare with original
            first_seq_packed = packed_input_ids_1[0, :6]
            first_seq_original = input_ids[0, :6]

            if not torch.equal(first_seq_packed, first_seq_original):
                return {
                    "success": False,
                    "error": "Content not preserved in first sequence",
                }

            # Extract the second sequence (length 4) and compare with original
            second_seq_packed = packed_input_ids_1[0, 6:10]
            second_seq_original = input_ids[1, :4]

            if not torch.equal(second_seq_packed, second_seq_original):
                return {
                    "success": False,
                    "error": "Content not preserved in second sequence",
                }

            return {"success": True, "error": None}

        except Exception as e:
            return {"success": False, "error": f"Consistency test failed: {str(e)}"}

    def _test_edge_cases(self, _pack_sequences_for_megatron):
        """Test edge cases and error conditions."""
        try:
            # Test 1: Single sequence
            batch_size = 1
            seq_len = 10
            vocab_size = 50

            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device="cuda"
            )
            seq_lengths = torch.tensor([seq_len], device="cuda")

            packed_input_ids, _, packed_seq_params, cu_seqlens, cu_seqlens_padded = (
                _pack_sequences_for_megatron(
                    input_ids, seq_lengths, cp_rank=0, cp_size=1
                )
            )

            # Verify single sequence packing
            if packed_input_ids.shape != (1, seq_len):
                return {
                    "success": False,
                    "error": f"Single sequence shape mismatch: expected (1, {seq_len}), got {packed_input_ids.shape}",
                }

            expected_cu_seqlens = torch.tensor(
                [0, seq_len], device="cuda", dtype=torch.int32
            )
            if not torch.equal(cu_seqlens, expected_cu_seqlens):
                return {
                    "success": False,
                    "error": f"Single sequence cu_seqlens mismatch: expected {expected_cu_seqlens}, got {cu_seqlens}",
                }

            # Test 2: Empty sequences (length 0)
            batch_size = 3
            max_seq_len = 5
            input_ids = torch.randint(
                0, vocab_size, (batch_size, max_seq_len), device="cuda"
            )
            seq_lengths = torch.tensor([3, 0, 2], device="cuda")

            packed_input_ids, _, packed_seq_params, cu_seqlens, cu_seqlens_padded = (
                _pack_sequences_for_megatron(
                    input_ids, seq_lengths, cp_rank=0, cp_size=1
                )
            )

            # Should handle empty sequences gracefully
            expected_total_tokens = 5  # 3 + 0 + 2
            if packed_input_ids.shape != (1, expected_total_tokens):
                return {
                    "success": False,
                    "error": f"Empty sequence shape mismatch: expected (1, {expected_total_tokens}), got {packed_input_ids.shape}",
                }

            expected_cu_seqlens = torch.tensor(
                [0, 3, 3, 5], device="cuda", dtype=torch.int32
            )
            if not torch.equal(cu_seqlens, expected_cu_seqlens):
                return {
                    "success": False,
                    "error": f"Empty sequence cu_seqlens mismatch: expected {expected_cu_seqlens}, got {cu_seqlens}",
                }

            # Test 3: Large padding values
            batch_size = 2
            seq_len = 4
            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device="cuda"
            )
            seq_lengths = torch.tensor([3, 2], device="cuda")

            packed_input_ids, _, packed_seq_params, cu_seqlens, cu_seqlens_padded = (
                _pack_sequences_for_megatron(
                    input_ids,
                    seq_lengths,
                    pad_individual_seqs_to_multiple_of=8,
                    cp_rank=0,
                    cp_size=1,
                )
            )

            # With padding to multiple of 8: [3, 2] -> [8, 8] = 16 tokens
            expected_total_tokens = 16
            if packed_input_ids.shape != (1, expected_total_tokens):
                return {
                    "success": False,
                    "error": f"Large padding shape mismatch: expected (1, {expected_total_tokens}), got {packed_input_ids.shape}",
                }

            return {"success": True, "error": None}

        except Exception as e:
            return {"success": False, "error": f"Edge cases test failed: {str(e)}"}

    def _test_context_parallel(
        self, _pack_sequences_for_megatron, _get_tokens_on_this_cp_rank, rank
    ):
        """Test sequence packing with context parallelism."""
        # Test parameters
        batch_size = 2
        seq_len = 16  # Ensure divisible by cp_size * 2
        vocab_size = 100

        # Ensure sequence length is compatible with CP
        if seq_len % (2 * self.cp_size) != 0:
            seq_len = (seq_len // (2 * self.cp_size) + 1) * (2 * self.cp_size)

        # Create test data
        torch.manual_seed(42)  # For reproducibility
        input_ids = torch.arange(seq_len * batch_size, device="cuda").reshape(
            batch_size, seq_len
        )
        seq_lengths = torch.tensor([seq_len, seq_len], device="cuda")

        # Test 1: CP packing with individual sequence padding
        (
            packed_input_ids,
            packed_input_ids_cp_sharded,
            packed_seq_params,
            cu_seqlens,
            cu_seqlens_padded,
        ) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths,
            pad_individual_seqs_to_multiple_of=self.cp_size * 2,
            cp_rank=rank,
            cp_size=self.cp_size,
        )

        # Verify the packed tensor shape
        expected_tokens_per_rank = seq_len // self.cp_size
        expected_total_tokens = batch_size * expected_tokens_per_rank
        if packed_input_ids_cp_sharded.shape != (1, expected_total_tokens):
            return {
                "success": False,
                "error": f"CP packing shape mismatch: expected (1, {expected_total_tokens}), got {packed_input_ids_cp_sharded.shape}",
            }

        # Verify cu_seqlens for original sequences
        expected_cu_seqlens = torch.tensor(
            [0, seq_len, seq_len * 2], device="cuda", dtype=torch.int32
        )
        if not torch.equal(cu_seqlens, expected_cu_seqlens):
            return {
                "success": False,
                "error": f"CP cu_seqlens mismatch: expected {expected_cu_seqlens}, got {cu_seqlens}",
            }

        # Verify PackedSeqParams
        if packed_seq_params.qkv_format != "thd":
            return {
                "success": False,
                "error": f"CP wrong qkv_format: expected 'thd', got {packed_seq_params.qkv_format}",
            }

        # Test 2: CP packing with full sequence padding
        pad_full_seq_to = (batch_size * seq_len) + 8  # Add some padding
        (
            packed_input_ids_full,
            packed_input_ids_cp_sharded,
            packed_seq_params_full,
            cu_seqlens_full,
            cu_seqlens_padded_full,
        ) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths,
            pad_individual_seqs_to_multiple_of=self.cp_size * 2,
            pad_packed_seq_to=pad_full_seq_to,
            cp_rank=rank,
            cp_size=self.cp_size,
        )

        # Verify the packed tensor shape with full padding
        expected_tokens_per_rank_full = pad_full_seq_to // self.cp_size
        if packed_input_ids_cp_sharded.shape != (1, expected_tokens_per_rank_full):
            return {
                "success": False,
                "error": f"CP full padding shape mismatch: expected (1, {expected_tokens_per_rank_full}), got {packed_input_ids_cp_sharded.shape}",
            }

        # Verify cu_seqlens_padded for full padding
        expected_cu_seqlens_padded_full = torch.tensor(
            [0, seq_len, pad_full_seq_to], device="cuda", dtype=torch.int32
        )
        if not torch.equal(cu_seqlens_padded_full, expected_cu_seqlens_padded_full):
            return {
                "success": False,
                "error": f"CP full padding cu_seqlens_padded mismatch: expected {expected_cu_seqlens_padded_full}, got {cu_seqlens_padded_full}",
            }

        correct_ids_0 = torch.tensor(
            [0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0, 0, 0],
            device="cuda",
        )
        correct_ids_1 = torch.tensor(
            [4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 0],
            device="cuda",
        )

        if (
            rank == 0
            and torch.sum(torch.abs(packed_input_ids_cp_sharded - correct_ids_0)).item()
            != 0
        ):
            return {
                "success": False,
                "error": f"CP full padding ids mismatch: expected {correct_ids_0}, got {packed_input_ids_cp_sharded[0, :20]}",
            }
        if (
            rank == 1
            and torch.sum(torch.abs(packed_input_ids_cp_sharded - correct_ids_1)).item()
            != 0
        ):
            return {
                "success": False,
                "error": f"CP full padding ids mismatch: expected {correct_ids_1}, got {packed_input_ids_cp_sharded[0, 20:]}",
            }

        return {"success": True, "error": None}


PACK_SEQUENCES_TEST_ACTOR_FQN = (
    f"tests.unit.models.{PackSequencesTestActor.__module__}.PackSequencesTestActor"
)


@pytest.fixture
def register_pack_sequences_test_actor():
    """Register the PackSequencesTestActor for use in tests."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(
        PACK_SEQUENCES_TEST_ACTOR_FQN
    )
    ACTOR_ENVIRONMENT_REGISTRY[PACK_SEQUENCES_TEST_ACTOR_FQN] = PY_EXECUTABLES.MCORE

    yield PACK_SEQUENCES_TEST_ACTOR_FQN

    # Clean up registry
    if PACK_SEQUENCES_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[PACK_SEQUENCES_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[PACK_SEQUENCES_TEST_ACTOR_FQN] = (
                original_registry_value
            )


@pytest.fixture
def pack_sequences_setup(request):
    """Setup and teardown for pack sequences tests - creates a virtual cluster and reusable actor."""
    # Get parameters from request
    if hasattr(request, "param") and request.param is not None:
        cp_size = request.param
    else:
        cp_size = 1

    cluster = None
    worker_group = None

    try:
        # Skip if not enough GPUs
        if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
            pytest.skip(
                f"Not enough GPUs available. Need {cp_size}, got {torch.cuda.device_count()}"
            )

        cluster_name = f"test-pack-sequences-cp{cp_size}"
        print(f"Creating virtual cluster '{cluster_name}' for {cp_size} GPUs...")

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[cp_size],
            use_gpus=True,
            max_colocated_worker_groups=1,
        )

        actor_fqn = PACK_SEQUENCES_TEST_ACTOR_FQN

        # Register the actor
        original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(actor_fqn)
        ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.MCORE

        try:
            # For CP tests
            sharding = NamedSharding(layout=list(range(cp_size)), names=["cp"])
            builder = RayWorkerBuilder(actor_fqn, cp_size)

            worker_group = RayWorkerGroup(
                cluster=cluster,
                remote_worker_builder=builder,
                workers_per_node=None,
                sharding_annotations=sharding,
            )

            yield worker_group

        finally:
            # Clean up registry
            if actor_fqn in ACTOR_ENVIRONMENT_REGISTRY:
                if original_registry_value is None:
                    del ACTOR_ENVIRONMENT_REGISTRY[actor_fqn]
                else:
                    ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = original_registry_value

    finally:
        print("Cleaning up pack sequences test resources...")
        if worker_group:
            worker_group.shutdown(force=True)
        if cluster:
            cluster.shutdown()


@pytest.mark.unit
@pytest.mark.parametrize("pack_sequences_setup", [1], indirect=True, ids=["cp1"])
def test_pack_sequences_comprehensive(pack_sequences_setup):
    """Comprehensive test of pack sequences functionality without context parallelism."""
    worker_group = pack_sequences_setup

    # Run all tests in a single call to the actor
    futures = worker_group.run_all_workers_single_data("run_all_pack_sequences_tests")
    results = ray.get(futures)

    # Check that all workers succeeded
    for i, result in enumerate(results):
        assert result["success"], f"Worker {i} failed: {result['error']}"

        # Print detailed results for debugging
        if "detailed_results" in result:
            detailed = result["detailed_results"]
            print(f"Worker {i} detailed results:")
            for test_name, test_result in detailed.items():
                status = "PASSED" if test_result["success"] else "FAILED"
                print(f"  {test_name}: {status}")
                if not test_result["success"]:
                    print(f"    Error: {test_result['error']}")


@pytest.mark.unit
@pytest.mark.parametrize("pack_sequences_setup", [2], indirect=True, ids=["cp2"])
def test_pack_sequences_with_context_parallel(pack_sequences_setup):
    """Test pack sequences functionality with context parallelism."""
    worker_group = pack_sequences_setup

    # Run all tests including CP tests
    futures = worker_group.run_all_workers_single_data("run_all_pack_sequences_tests")
    results = ray.get(futures)

    # Check that all workers succeeded
    for i, result in enumerate(results):
        assert result["success"], f"Worker {i} failed: {result['error']}"

        # Print detailed results for debugging
        if "detailed_results" in result:
            detailed = result["detailed_results"]
            print(f"Worker {i} detailed results:")
            for test_name, test_result in detailed.items():
                if "skipped" in test_result:
                    print(f"  {test_name}: SKIPPED ({test_result['skipped']})")
                else:
                    status = "PASSED" if test_result["success"] else "FAILED"
                    print(f"  {test_name}: {status}")
                    if not test_result["success"]:
                        print(f"    Error: {test_result['error']}")
