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

"""
Unit tests for Megatron data utilities.

This module tests the data processing functions in nemo_rl.models.megatron.data,
focusing on:
- Microbatch processing and iteration
- Sequence packing and unpacking
- Global batch processing
- Sequence dimension validation
"""

from unittest.mock import MagicMock, patch

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from tests.unit.models.megatron.megatron_data_actors import (
    GetPackSequenceParametersTestActor,
    PackSequencesTestActor,
)


@pytest.mark.mcore
class TestProcessedMicrobatchDataclass:
    """Tests for ProcessedMicrobatch dataclass."""

    def test_processed_microbatch_fields(self):
        """Test that ProcessedMicrobatch has all expected fields."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch

        mock_data_dict = MagicMock()
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_input_ids_cp_sharded = torch.tensor([[1, 2, 3]])
        mock_attention_mask = torch.tensor([[1, 1, 1]])
        mock_position_ids = torch.tensor([[0, 1, 2]])
        mock_packed_seq_params = MagicMock()
        mock_cu_seqlens_padded = torch.tensor([0, 3])

        microbatch = ProcessedMicrobatch(
            data_dict=mock_data_dict,
            input_ids=mock_input_ids,
            input_ids_cp_sharded=mock_input_ids_cp_sharded,
            attention_mask=mock_attention_mask,
            position_ids=mock_position_ids,
            packed_seq_params=mock_packed_seq_params,
            cu_seqlens_padded=mock_cu_seqlens_padded,
        )

        assert microbatch.data_dict == mock_data_dict
        assert torch.equal(microbatch.input_ids, mock_input_ids)
        assert torch.equal(microbatch.input_ids_cp_sharded, mock_input_ids_cp_sharded)
        assert torch.equal(microbatch.attention_mask, mock_attention_mask)
        assert torch.equal(microbatch.position_ids, mock_position_ids)
        assert microbatch.packed_seq_params == mock_packed_seq_params
        assert torch.equal(microbatch.cu_seqlens_padded, mock_cu_seqlens_padded)


@pytest.mark.mcore
class TestGetAndValidateSeqlen:
    """Tests for get_and_validate_seqlen function."""

    def test_get_and_validate_seqlen_valid(self):
        """Test get_and_validate_seqlen with valid data."""
        from nemo_rl.models.megatron.data import get_and_validate_seqlen

        # Create mock data with consistent sequence dimension
        data = MagicMock()
        data.__getitem__ = MagicMock(
            side_effect=lambda k: torch.zeros(2, 10) if k == "input_ids" else None
        )
        data.items = MagicMock(
            return_value=[
                ("input_ids", torch.zeros(2, 10)),
                ("attention_mask", torch.zeros(2, 10)),
            ]
        )

        sequence_dim, seq_dim_size = get_and_validate_seqlen(data)

        assert sequence_dim == 1
        assert seq_dim_size == 10

    def test_get_and_validate_seqlen_mismatch(self):
        """Test get_and_validate_seqlen with mismatched sequence dimensions."""
        from nemo_rl.models.megatron.data import get_and_validate_seqlen

        # Create mock data with mismatched sequence dimension
        data = MagicMock()
        data.__getitem__ = MagicMock(
            side_effect=lambda k: torch.zeros(2, 10) if k == "input_ids" else None
        )
        data.items = MagicMock(
            return_value=[
                ("input_ids", torch.zeros(2, 10)),
                ("other_tensor", torch.zeros(2, 15)),  # Mismatched!
            ]
        )

        with pytest.raises(AssertionError) as exc_info:
            get_and_validate_seqlen(data)

        assert "Dim 1 must be the sequence dim" in str(exc_info.value)

    def test_get_and_validate_seqlen_skips_1d_tensors(self):
        """Test that get_and_validate_seqlen skips 1D tensors."""
        from nemo_rl.models.megatron.data import get_and_validate_seqlen

        # Create mock data with 1D tensor (should be skipped)
        data = MagicMock()
        data.__getitem__ = MagicMock(
            side_effect=lambda k: torch.zeros(2, 10) if k == "input_ids" else None
        )
        data.items = MagicMock(
            return_value=[
                ("input_ids", torch.zeros(2, 10)),
                ("seq_lengths", torch.zeros(2)),  # 1D tensor, should be skipped
            ]
        )

        # Should not raise
        sequence_dim, seq_dim_size = get_and_validate_seqlen(data)
        assert seq_dim_size == 10


@pytest.mark.mcore
class TestProcessMicrobatch:
    """Tests for process_microbatch function."""

    @patch("nemo_rl.models.megatron.data.get_ltor_masks_and_position_ids")
    def test_process_microbatch_no_packing(self, mock_get_masks):
        """Test process_microbatch without sequence packing."""
        from nemo_rl.models.megatron.data import process_microbatch

        # Setup mock
        mock_attention_mask = torch.ones(2, 10)
        mock_position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        mock_get_masks.return_value = (mock_attention_mask, None, mock_position_ids)

        # Create test data
        data_dict = MagicMock()
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [6, 7, 8, 9, 10, 11, 12, 0, 0, 0]]
        )
        data_dict.__getitem__ = MagicMock(return_value=input_ids)

        result = process_microbatch(
            data_dict, pack_sequences=False, straggler_timer=MagicMock()
        )

        # Verify results
        assert torch.equal(result.input_ids, input_ids)
        assert torch.equal(result.input_ids_cp_sharded, input_ids)
        assert result.attention_mask is not None
        assert result.position_ids is not None
        assert result.packed_seq_params is None
        assert result.cu_seqlens_padded is None

        # Verify get_ltor_masks_and_position_ids was called
        mock_get_masks.assert_called_once()

    @patch("nemo_rl.models.megatron.data.get_context_parallel_rank", return_value=0)
    @patch(
        "nemo_rl.models.megatron.data.get_context_parallel_world_size", return_value=1
    )
    @patch("nemo_rl.models.megatron.data._pack_sequences_for_megatron")
    def test_process_microbatch_with_packing(
        self, mock_pack, mock_cp_world, mock_cp_rank
    ):
        """Test process_microbatch with sequence packing."""
        from nemo_rl.models.megatron.data import process_microbatch

        # Setup mocks
        mock_packed_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        mock_packed_seq_params = MagicMock()
        mock_cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)
        mock_cu_seqlens_padded = torch.tensor([0, 5, 8], dtype=torch.int32)
        mock_pack.return_value = (
            mock_packed_input_ids,
            mock_packed_input_ids,
            mock_packed_seq_params,
            mock_cu_seqlens,
            mock_cu_seqlens_padded,
        )

        # Create test data
        data_dict = MagicMock()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]])
        seq_lengths = torch.tensor([5, 3])
        data_dict.__getitem__ = MagicMock(
            side_effect=lambda k: input_ids if k == "input_ids" else seq_lengths
        )
        data_dict.__contains__ = MagicMock(return_value=True)

        result = process_microbatch(
            data_dict,
            seq_length_key="input_lengths",
            pack_sequences=True,
            straggler_timer=MagicMock(),
        )

        # Verify results
        assert torch.equal(result.input_ids, mock_packed_input_ids)
        assert result.packed_seq_params == mock_packed_seq_params
        # For packed sequences, attention_mask and position_ids are None
        assert result.attention_mask is None
        assert result.position_ids is None
        assert result.cu_seqlens_padded is not None

        # Verify pack was called
        mock_pack.assert_called_once()

    def test_process_microbatch_packing_requires_seq_length_key(self):
        """Test that packing requires seq_length_key."""
        from nemo_rl.models.megatron.data import process_microbatch

        data_dict = MagicMock()
        input_ids = torch.tensor([[1, 2, 3]])
        data_dict.__getitem__ = MagicMock(return_value=input_ids)

        with pytest.raises(AssertionError) as exc_info:
            process_microbatch(
                data_dict,
                seq_length_key=None,
                pack_sequences=True,
                straggler_timer=MagicMock(),
            )

        assert "seq_length_key must be provided" in str(exc_info.value)

    def test_process_microbatch_packing_requires_seq_length_in_data(self):
        """Test that packing requires seq_length_key to be in data_dict."""
        from nemo_rl.models.megatron.data import process_microbatch

        data_dict = MagicMock()
        input_ids = torch.tensor([[1, 2, 3]])
        data_dict.__getitem__ = MagicMock(return_value=input_ids)
        data_dict.__contains__ = MagicMock(return_value=False)

        with pytest.raises(AssertionError) as exc_info:
            process_microbatch(
                data_dict,
                seq_length_key="input_lengths",
                pack_sequences=True,
                straggler_timer=MagicMock(),
            )

        assert "input_lengths not found in data_dict" in str(exc_info.value)


@pytest.mark.mcore
class TestProcessGlobalBatch:
    """Tests for process_global_batch function."""

    def test_process_global_batch_basic(self):
        """Test basic process_global_batch functionality."""
        from nemo_rl.models.megatron.data import process_global_batch

        # Create mock data
        sample_mask = torch.tensor([1.0, 1.0, 0.0])
        input_ids = torch.zeros(3, 10)
        mock_batch = BatchedDataDict(
            {
                "sample_mask": sample_mask,
                "input_ids": input_ids,
            }
        )

        mock_data = MagicMock()
        mock_data.get_batch.return_value = mock_batch

        mock_dp_group = MagicMock()

        # Mock torch.distributed.all_reduce
        with patch("torch.distributed.all_reduce") as mock_all_reduce:
            result = process_global_batch(
                data=mock_data,
                loss_fn=MagicMock(),
                dp_group=mock_dp_group,
                batch_idx=0,
                batch_size=3,
            )

            batch = result["batch"]
            assert torch.equal(batch["sample_mask"], mock_batch["sample_mask"])
            assert torch.equal(batch["input_ids"], mock_batch["input_ids"])

            # Verify get_batch was called
            mock_data.get_batch.assert_called_once_with(batch_idx=0, batch_size=3)

            # Verify all_reduce was called
            mock_all_reduce.assert_called_once()

    def test_process_global_batch_requires_sample_mask_in_data(self):
        """Test that process_global_batch requires sample_mask."""
        from nemo_rl.models.megatron.data import process_global_batch

        # Create mock data without sample_mask
        mock_batch = MagicMock()
        mock_batch.__contains__ = MagicMock(return_value=False)

        mock_data = MagicMock()
        mock_data.get_batch.return_value = mock_batch

        with pytest.raises(AssertionError) as exc_info:
            process_global_batch(
                data=mock_data,
                loss_fn=MagicMock(),
                dp_group=MagicMock(),
                batch_idx=0,
                batch_size=3,
            )

        assert "sample_mask must be present in the data!" in str(exc_info.value)


@pytest.mark.mcore
class TestGetMicrobatchIterator:
    """Tests for get_microbatch_iterator function."""

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_dynamic_batching(
        self, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test get_microbatch_iterator with dynamic batching."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 128)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.make_microbatch_iterator_with_dynamic_shapes.return_value = iter([])
        mock_data.get_microbatch_iterator_dynamic_shapes_len.return_value = 5

        cfg = {
            "dynamic_batching": {"enabled": True},
            "sequence_packing": {"enabled": False},
        }

        (
            iterator,
            data_iterator_len,
            micro_batch_size,
            seq_dim_size,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=4,
            straggler_timer=MagicMock(),
        )

        # Verify dynamic batching path was taken
        mock_data.make_microbatch_iterator_with_dynamic_shapes.assert_called_once()
        mock_data.get_microbatch_iterator_dynamic_shapes_len.assert_called_once()

        assert data_iterator_len == 5
        assert seq_dim_size == 128

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    @patch("nemo_rl.models.megatron.data._get_pack_sequence_parameters_for_megatron")
    def test_get_microbatch_iterator_sequence_packing(
        self, mock_get_params, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test get_microbatch_iterator with sequence packing."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 256)
        mock_get_params.return_value = (8, 16, None)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.make_microbatch_iterator_for_packable_sequences.return_value = iter(
            []
        )
        mock_data.get_microbatch_iterator_for_packable_sequences_len.return_value = (
            10,
            512,
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            },
        }

        (
            iterator,
            data_iterator_len,
            micro_batch_size,
            seq_dim_size,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=4,
            straggler_timer=MagicMock(),
        )

        # Verify sequence packing path was taken
        mock_data.make_microbatch_iterator_for_packable_sequences.assert_called_once()

        # With sequence packing, micro_batch_size should be 1
        assert micro_batch_size == 1
        assert data_iterator_len == 10

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_regular(
        self, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test get_microbatch_iterator with regular batching."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 64)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.size = 16
        mock_data.make_microbatch_iterator.return_value = iter([])

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
        }

        mbs = 4

        (
            iterator,
            data_iterator_len,
            micro_batch_size,
            seq_dim_size,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=mbs,
            straggler_timer=MagicMock(),
        )

        # Verify regular batching path was taken
        mock_data.make_microbatch_iterator.assert_called_once_with(mbs)

        assert micro_batch_size == mbs
        assert data_iterator_len == 16 // mbs
        assert seq_dim_size == 64

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_auto_detects_seq_length_key(
        self, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test that get_microbatch_iterator auto-detects seq_length_key for packing."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 128)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.make_microbatch_iterator_for_packable_sequences.return_value = iter(
            []
        )
        mock_data.get_microbatch_iterator_for_packable_sequences_len.return_value = (
            5,
            256,
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            },
        }

        get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=4,
            straggler_timer=MagicMock(),
            seq_length_key=None,  # Should be auto-detected
        )

        # Verify make_processed_microbatch_iterator was called with "input_lengths"
        call_kwargs = mock_make_iterator.call_args[1]
        assert call_kwargs["seq_length_key"] == "input_lengths"


@pytest.mark.mcore
class TestMakeProcessedMicrobatchIterator:
    """Tests for make_processed_microbatch_iterator function."""

    @patch("nemo_rl.models.megatron.data.process_microbatch")
    def test_make_processed_microbatch_iterator_basic(self, mock_process):
        """Test make_processed_microbatch_iterator yields ProcessedMicrobatch."""
        from nemo_rl.models.megatron.data import (
            ProcessedInputs,
            ProcessedMicrobatch,
            make_processed_microbatch_iterator,
        )

        # Setup mocks
        mock_input_ids = MagicMock()
        mock_input_ids_cp_sharded = MagicMock()
        mock_attention_mask = MagicMock()
        mock_position_ids = MagicMock()
        mock_packed_seq_params = None
        mock_cu_seqlens_padded = None

        mock_process.return_value = ProcessedInputs(
            input_ids=mock_input_ids,
            input_ids_cp_sharded=mock_input_ids_cp_sharded,
            attention_mask=mock_attention_mask,
            position_ids=mock_position_ids,
            packed_seq_params=mock_packed_seq_params,
            cu_seqlens_padded=mock_cu_seqlens_padded,
        )

        # Create mock data dict
        mock_data_dict = MagicMock()
        mock_data_dict.to.return_value = mock_data_dict

        raw_iterator = iter([mock_data_dict])

        cfg = {"sequence_packing": {"enabled": False}}

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            cfg=cfg,
            seq_length_key=None,
            pad_individual_seqs_to_multiple_of=1,
            pad_packed_seq_to_multiple_of=1,
            straggler_timer=MagicMock(),
            pad_full_seq_to=None,
        )

        # Get first item from iterator
        microbatch = next(processed_iterator)

        # Verify it's a ProcessedMicrobatch
        assert isinstance(microbatch, ProcessedMicrobatch)
        assert microbatch.data_dict == mock_data_dict
        assert microbatch.input_ids == mock_input_ids

        # Verify data was moved to CUDA
        mock_data_dict.to.assert_called_once_with("cuda")

    @patch("nemo_rl.models.megatron.data.process_microbatch")
    def test_make_processed_microbatch_iterator_with_packing(self, mock_process):
        """Test make_processed_microbatch_iterator with sequence packing."""
        from nemo_rl.models.megatron.data import (
            ProcessedInputs,
            make_processed_microbatch_iterator,
        )

        # Setup mocks
        mock_process.return_value = ProcessedInputs(
            input_ids=MagicMock(),
            input_ids_cp_sharded=MagicMock(),
            attention_mask=None,  # None for packed
            position_ids=None,  # None for packed
            packed_seq_params=MagicMock(),
            cu_seqlens_padded=MagicMock(),
        )

        mock_data_dict = MagicMock()
        mock_data_dict.to.return_value = mock_data_dict

        raw_iterator = iter([mock_data_dict])

        cfg = {"sequence_packing": {"enabled": True}}

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            cfg=cfg,
            seq_length_key="input_lengths",
            pad_individual_seqs_to_multiple_of=8,
            pad_packed_seq_to_multiple_of=16,
            straggler_timer=MagicMock(),
            pad_full_seq_to=1024,
        )

        microbatch = next(processed_iterator)

        # Verify process_microbatch was called with pack_sequences=True
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["pack_sequences"] is True
        assert call_kwargs["seq_length_key"] == "input_lengths"
        assert call_kwargs["pad_individual_seqs_to_multiple_of"] == 8
        assert call_kwargs["pad_packed_seq_to_multiple_of"] == 16
        assert call_kwargs["pad_full_seq_to"] == 1024


PACK_SEQUENCES_TEST_ACTOR_FQN = (
    f"{PackSequencesTestActor.__module__}.PackSequencesTestActor"
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


GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN = f"{GetPackSequenceParametersTestActor.__module__}.GetPackSequenceParametersTestActor"


@pytest.fixture
def register_get_pack_sequence_parameters_test_actor():
    """Register the GetPackSequenceParametersTestActor for use in tests."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(
        GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN
    )
    ACTOR_ENVIRONMENT_REGISTRY[GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN] = (
        PY_EXECUTABLES.MCORE
    )

    yield GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN

    # Clean up registry
    if GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN] = (
                original_registry_value
            )


@pytest.fixture
def get_pack_sequence_parameters_setup(request):
    """Setup and teardown for get pack sequence parameters tests - creates a virtual cluster and reusable actor."""
    cluster = None
    worker_group = None

    try:
        # Skip if not enough GPUs
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            pytest.skip(
                f"Not enough GPUs available. Need 1, got {torch.cuda.device_count()}"
            )

        cluster_name = "test-get-pack-sequence-parameters"
        print(f"Creating virtual cluster '{cluster_name}'...")

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            max_colocated_worker_groups=1,
        )

        actor_fqn = GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN

        # Register the actor
        original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(actor_fqn)
        ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.MCORE

        try:
            # For CP tests
            sharding = NamedSharding(layout=list(range(1)), names=["cp"])
            builder = RayWorkerBuilder(actor_fqn)

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
        print("Cleaning up get pack sequence parameters test resources...")
        if worker_group:
            worker_group.shutdown(force=True)
        if cluster:
            cluster.shutdown()


@pytest.mark.parametrize(
    "get_pack_sequence_parameters_setup", [1], indirect=True, ids=["cp1"]
)
def test_get_pack_sequence_parameters_for_megatron(get_pack_sequence_parameters_setup):
    """Comprehensive test of pack sequences functionality without context parallelism."""
    worker_group = get_pack_sequence_parameters_setup

    # Run all tests in a single call to the actor
    futures = worker_group.run_all_workers_single_data(
        "run_all_get_pack_sequence_parameters_for_megatron_tests"
    )
    results = ray.get(futures)

    # Check that all workers succeeded
    for i, result in enumerate(results):
        assert result["success"], f"Worker {i} failed: {result['error']}"
