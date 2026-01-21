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

import os
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


class TestCheckSequenceDim:
    """Tests for check_sequence_dim function."""

    def test_check_sequence_dim_valid(self):
        """Test check_sequence_dim with valid data."""
        from nemo_rl.models.megatron.data import check_sequence_dim

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

        sequence_dim, seq_dim_size = check_sequence_dim(data)

        assert sequence_dim == 1
        assert seq_dim_size == 10

    def test_check_sequence_dim_mismatch(self):
        """Test check_sequence_dim with mismatched sequence dimensions."""
        from nemo_rl.models.megatron.data import check_sequence_dim

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
            check_sequence_dim(data)

        assert "Dim 1 must be the sequence dim" in str(exc_info.value)

    def test_check_sequence_dim_skips_1d_tensors(self):
        """Test that check_sequence_dim skips 1D tensors."""
        from nemo_rl.models.megatron.data import check_sequence_dim

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
        sequence_dim, seq_dim_size = check_sequence_dim(data)
        assert seq_dim_size == 10


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

        (
            result_input_ids,
            input_ids_cp_sharded,
            attention_mask,
            position_ids,
            packed_seq_params,
            cu_seqlens_padded,
        ) = process_microbatch(data_dict, pack_sequences=False)

        # Verify results
        assert torch.equal(result_input_ids, input_ids)
        assert torch.equal(input_ids_cp_sharded, input_ids)
        assert attention_mask is not None
        assert position_ids is not None
        assert packed_seq_params is None
        assert cu_seqlens_padded is None

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

        (
            result_input_ids,
            input_ids_cp_sharded,
            attention_mask,
            position_ids,
            packed_seq_params,
            cu_seqlens_padded,
        ) = process_microbatch(
            data_dict, seq_length_key="input_lengths", pack_sequences=True
        )

        # Verify results
        assert torch.equal(result_input_ids, mock_packed_input_ids)
        assert packed_seq_params == mock_packed_seq_params
        # For packed sequences, attention_mask and position_ids are None
        assert attention_mask is None
        assert position_ids is None
        assert cu_seqlens_padded is not None

        # Verify pack was called
        mock_pack.assert_called_once()

    def test_process_microbatch_packing_requires_seq_length_key(self):
        """Test that packing requires seq_length_key."""
        from nemo_rl.models.megatron.data import process_microbatch

        data_dict = MagicMock()
        input_ids = torch.tensor([[1, 2, 3]])
        data_dict.__getitem__ = MagicMock(return_value=input_ids)

        with pytest.raises(AssertionError) as exc_info:
            process_microbatch(data_dict, seq_length_key=None, pack_sequences=True)

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
                data_dict, seq_length_key="input_lengths", pack_sequences=True
            )

        assert "input_lengths not found in data_dict" in str(exc_info.value)


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
            batch, global_valid_seqs, global_valid_toks = process_global_batch(
                data=mock_data,
                batch_idx=0,
                batch_size=3,
                loss_fn=MagicMock(),
                dp_group=mock_dp_group,
            )

            assert torch.equal(batch["sample_mask"], mock_batch["sample_mask"])
            assert torch.equal(batch["input_ids"], mock_batch["input_ids"])

            # Verify get_batch was called
            mock_data.get_batch.assert_called_once_with(batch_idx=0, batch_size=3)

            # Verify all_reduce was called
            mock_all_reduce.assert_called_once()

    def test_process_global_batch_requires_sample_mask(self):
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
                batch_idx=0,
                batch_size=3,
                loss_fn=MagicMock(),
                dp_group=MagicMock(),
            )

        assert "sample_mask must be present" in str(exc_info.value)


class TestGetMicrobatchIterator:
    """Tests for get_microbatch_iterator function."""

    @patch("nemo_rl.models.megatron.data.check_sequence_dim")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_dynamic_batching(
        self, mock_make_iterator, mock_check_seq_dim
    ):
        """Test get_microbatch_iterator with dynamic batching."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_check_seq_dim.return_value = (1, 128)
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
        )

        # Verify dynamic batching path was taken
        mock_data.make_microbatch_iterator_with_dynamic_shapes.assert_called_once()
        mock_data.get_microbatch_iterator_dynamic_shapes_len.assert_called_once()

        assert data_iterator_len == 5
        assert seq_dim_size == 128

    @patch("nemo_rl.models.megatron.data.check_sequence_dim")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    @patch("nemo_rl.models.megatron.data._get_pack_sequence_parameters_for_megatron")
    def test_get_microbatch_iterator_sequence_packing(
        self, mock_get_params, mock_make_iterator, mock_check_seq_dim
    ):
        """Test get_microbatch_iterator with sequence packing."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_check_seq_dim.return_value = (1, 256)
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
        )

        # Verify sequence packing path was taken
        mock_data.make_microbatch_iterator_for_packable_sequences.assert_called_once()

        # With sequence packing, micro_batch_size should be 1
        assert micro_batch_size == 1
        assert data_iterator_len == 10

    @patch("nemo_rl.models.megatron.data.check_sequence_dim")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_regular(
        self, mock_make_iterator, mock_check_seq_dim
    ):
        """Test get_microbatch_iterator with regular batching."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_check_seq_dim.return_value = (1, 64)
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
        )

        # Verify regular batching path was taken
        mock_data.make_microbatch_iterator.assert_called_once_with(mbs)

        assert micro_batch_size == mbs
        assert data_iterator_len == 16 // mbs
        assert seq_dim_size == 64

    @patch("nemo_rl.models.megatron.data.check_sequence_dim")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_auto_detects_seq_length_key(
        self, mock_make_iterator, mock_check_seq_dim
    ):
        """Test that get_microbatch_iterator auto-detects seq_length_key for packing."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_check_seq_dim.return_value = (1, 128)
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
            seq_length_key=None,  # Should be auto-detected
        )

        # Verify make_processed_microbatch_iterator was called with "input_lengths"
        call_kwargs = mock_make_iterator.call_args[1]
        assert call_kwargs["seq_length_key"] == "input_lengths"


class TestMakeProcessedMicrobatchIterator:
    """Tests for make_processed_microbatch_iterator function."""

    @patch("nemo_rl.models.megatron.data.process_microbatch")
    def test_make_processed_microbatch_iterator_basic(self, mock_process):
        """Test make_processed_microbatch_iterator yields ProcessedMicrobatch."""
        from nemo_rl.models.megatron.data import (
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

        mock_process.return_value = (
            mock_input_ids,
            mock_input_ids_cp_sharded,
            mock_attention_mask,
            mock_position_ids,
            mock_packed_seq_params,
            mock_cu_seqlens_padded,
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
        from nemo_rl.models.megatron.data import make_processed_microbatch_iterator

        # Setup mocks
        mock_process.return_value = (
            MagicMock(),  # input_ids
            MagicMock(),  # input_ids_cp_sharded
            None,  # attention_mask (None for packed)
            None,  # position_ids (None for packed)
            MagicMock(),  # packed_seq_params
            MagicMock(),  # cu_seqlens_padded
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


@ray.remote(num_gpus=1)
class PackSequencesTestActor:
    def __init__(self, cp_size):
        self.cp_size = cp_size
        self.env_vars = dict(os.environ)

    def run_all_pack_sequences_tests(self):
        """Run all sequence packing tests in a single call to avoid expensive reinitializations."""
        from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
        from nemo_rl.models.megatron.data import _pack_sequences_for_megatron

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


@ray.remote(num_gpus=1)
class GetPackSequenceParametersTestActor:
    def __init__(self):
        pass

    def run_all_get_pack_sequence_parameters_for_megatron_tests(self):
        """Test _get_pack_sequence_parameters_for_megatron function with various configurations."""
        from nemo_rl.models.megatron.data import (
            _get_pack_sequence_parameters_for_megatron,
        )

        # Test 1: Basic configuration - no parallelism, no FP8
        megatron_cfg = {
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        }
        max_seq_len = 1024

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 1 or pad_packed != 1 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=1, pad_packed=1, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 2: Context parallelism only
        megatron_cfg = {
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 4,
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 4 * 2 or pad_packed != 1 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=4*2, pad_packed=1, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 3: Tensor parallelism with sequence parallelism
        megatron_cfg = {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        expected_individual = 2  # tp_size when SP is enabled
        if pad_individual != 2 or pad_packed != 1 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=2, pad_packed=1, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 4: Tensor parallelism without sequence parallelism
        megatron_cfg = {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 1 or pad_packed != 1 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=1, pad_packed=1, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 5: Pipeline parallelism
        megatron_cfg = {
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 4,
            "context_parallel_size": 1,
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 1 or pad_packed != 1 or pad_to != max_seq_len:
            return {
                "success": False,
                "error": f"Expected pad_individual=1, pad_packed=1, pad_to={max_seq_len}, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 6: Combined CP and TP with SP
        megatron_cfg = {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 4,
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        expected_individual = 4 * 2 * 2  # cp_size * 2 * tp_size
        if (
            pad_individual != expected_individual
            or pad_packed != 1
            or pad_to is not None
        ):
            return {
                "success": False,
                "error": f"Expected pad_individual={expected_individual}, pad_packed=1, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 7: FP8 enabled with default recipe
        megatron_cfg = {
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "fp8_cfg": {"enabled": True},
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 1 or pad_packed != 16 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=1, pad_packed=16, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 8: FP8 enabled with blockwise recipe
        megatron_cfg = {
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "fp8_cfg": {"enabled": True, "fp8_recipe": "blockwise"},
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 1 or pad_packed != 128 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=1, pad_packed=128, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 9: FP8 with CP and TP+SP
        megatron_cfg = {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 4,
            "fp8_cfg": {"enabled": True, "fp8_recipe": "blockwise"},
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        expected_individual = 4 * 2 * 2  # cp_size * 2 * tp_size
        expected_packed = 128 * 4 * 2 * 2  # divisor * cp_size * 2 * tp_size
        if (
            pad_individual != expected_individual
            or pad_packed != expected_packed
            or pad_to is not None
        ):
            return {
                "success": False,
                "error": f"Expected pad_individual={expected_individual}, pad_packed={expected_packed}, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 10: All parallelism types with FP8 and PP
        megatron_cfg = {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 4,
            "context_parallel_size": 2,
            "fp8_cfg": {"enabled": True, "fp8_recipe": "other"},
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        expected_individual = 2 * 2 * 2  # cp_size * 2 * tp_size
        expected_packed = 16 * 2 * 2 * 2  # divisor * cp_size * 2 * tp_size
        if (
            pad_individual != expected_individual
            or pad_packed != expected_packed
            or pad_to != max_seq_len
        ):
            return {
                "success": False,
                "error": f"Expected pad_individual={expected_individual}, pad_packed={expected_packed}, pad_to={max_seq_len}, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 11: FP8 disabled explicitly
        megatron_cfg = {
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "fp8_cfg": {"enabled": False},
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 1 or pad_packed != 1 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=1, pad_packed=1, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 12: Missing fp8_cfg (should default to disabled)
        megatron_cfg = {
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            # No fp8_cfg key
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        if pad_individual != 1 or pad_packed != 1 or pad_to is not None:
            return {
                "success": False,
                "error": f"Expected pad_individual=1, pad_packed=1, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 13: Edge case - very large parallelism values
        megatron_cfg = {
            "tensor_model_parallel_size": 8,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 8,
            "fp8_cfg": {"enabled": True, "fp8_recipe": "blockwise"},
        }

        pad_individual, pad_packed, pad_to = _get_pack_sequence_parameters_for_megatron(
            megatron_cfg, max_seq_len
        )

        expected_individual = 8 * 2 * 8  # cp_size * 2 * tp_size = 128
        expected_packed = 128 * 8 * 2 * 8  # divisor * cp_size * 2 * tp_size = 16384
        if (
            pad_individual != expected_individual
            or pad_packed != expected_packed
            or pad_to is not None
        ):
            return {
                "success": False,
                "error": f"Expected pad_individual={expected_individual}, pad_packed={expected_packed}, pad_to=None, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
            }

        # Test 14: Edge case - different max_seq_len values with PP
        for test_seq_len in [512, 2048, 4096]:
            megatron_cfg = {
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 2,
                "context_parallel_size": 1,
            }

            pad_individual, pad_packed, pad_to = (
                _get_pack_sequence_parameters_for_megatron(megatron_cfg, test_seq_len)
            )

            if pad_individual != 1 or pad_packed != 1 or pad_to != test_seq_len:
                return {
                    "success": False,
                    "error": f"Expected pad_individual=1, pad_packed=1, pad_to={test_seq_len}, got pad_individual={pad_individual}, pad_packed={pad_packed}, pad_to={pad_to}",
                }

        return {"success": True, "error": None}


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
