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

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.interfaces import LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    ProcessedInputs,
    ProcessedMicrobatch,
    get_microbatch_iterator,
    make_processed_microbatch_iterator,
    process_global_batch,
    process_microbatch,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_loss_fn():
    loss_fn = MagicMock()
    loss_fn.loss_type = LossType.SEQUENCE_LEVEL
    return loss_fn


@pytest.fixture
def mock_dp_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    return mesh


@pytest.mark.automodel
class TestGetMicrobatchIterator:
    def test_regular_batching(self, mock_tokenizer):
        """Test regular batching returns ProcessedMicrobatch objects."""
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (16, 128)),
                "sample_mask": torch.ones(16, dtype=torch.bool),
            }
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }
        mbs = 4
        mock_dp_mesh = MagicMock()

        processed_iterator, iterator_len = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
            tokenizer=mock_tokenizer,
            cp_size=1,
        )

        # Verify iterator length
        assert iterator_len == 4  # 16 / 4 = 4

        # Verify we get ProcessedMicrobatch objects
        batches = list(processed_iterator)
        assert len(batches) == 4
        for batch in batches:
            assert isinstance(batch, ProcessedMicrobatch)
            assert isinstance(batch.processed_inputs, ProcessedInputs)
            assert batch.original_batch_size == 4
            assert batch.original_seq_len == 128

    def test_dynamic_batching(self, mock_tokenizer):
        """Test dynamic batching."""
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (8, 128)),
                "sample_mask": torch.ones(8, dtype=torch.bool),
            }
        )

        # Mock the microbatch iterator methods
        batch1 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        batch2 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        batch3 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        mock_iterator = iter([batch1, batch2, batch3])
        data.make_microbatch_iterator_with_dynamic_shapes = MagicMock(
            return_value=mock_iterator
        )
        data.get_microbatch_iterator_dynamic_shapes_len = MagicMock(return_value=3)

        cfg = {
            "dynamic_batching": {"enabled": True},
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }
        mbs = 4
        mock_dp_mesh = MagicMock()

        processed_iterator, iterator_len = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
            tokenizer=mock_tokenizer,
            cp_size=1,
        )

        # Verify dynamic batching was used
        assert iterator_len == 3
        data.make_microbatch_iterator_with_dynamic_shapes.assert_called_once()
        data.get_microbatch_iterator_dynamic_shapes_len.assert_called_once()

        # Verify we get ProcessedMicrobatch objects
        batches = list(processed_iterator)
        assert len(batches) == 3
        for batch in batches:
            assert isinstance(batch, ProcessedMicrobatch)

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_sequence_packing(self, mock_all_reduce, mock_tokenizer):
        """Test sequence packing."""
        # Create test data
        batch1 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "input_lengths": torch.tensor([64, 80, 90, 100]),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        batch2 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "input_lengths": torch.tensor([70, 85, 95, 110]),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (8, 128)),
                "input_lengths": torch.randint(64, 128, (8,)),
                "sample_mask": torch.ones(8, dtype=torch.bool),
            }
        )

        # Mock the microbatch iterator methods
        mock_iterator = iter([batch1, batch2])
        data.make_microbatch_iterator_for_packable_sequences = MagicMock(
            return_value=mock_iterator
        )
        data.get_microbatch_iterator_for_packable_sequences_len = MagicMock(
            return_value=(2, 100)
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
            "sequence_packing": {"enabled": True, "train_mb_tokens": 512},
        }
        mbs = 4
        mock_dp_mesh = MagicMock()

        # Mock the all_reduce to simulate max_batch_ct = 2 across all ranks
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 2  # Simulate max batch count

        mock_all_reduce.side_effect = side_effect

        processed_iterator, iterator_len = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
            tokenizer=mock_tokenizer,
            cp_size=1,
        )

        # Verify sequence packing was used
        assert iterator_len == 2
        data.make_microbatch_iterator_for_packable_sequences.assert_called()
        data.get_microbatch_iterator_for_packable_sequences_len.assert_called_once()

        # Verify all_reduce was called to synchronize batch counts
        mock_all_reduce.assert_called_once()

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_sequence_packing_with_dummy_batches(self, mock_all_reduce, mock_tokenizer):
        """Test sequence packing with dummy batches when local batch count < max batch count."""
        # Create test data
        batch1 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "input_lengths": torch.tensor([64, 80, 90, 100]),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "input_lengths": torch.tensor([64, 80, 90, 100]),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )

        # Create a call counter for the iterator
        call_count = [0]

        def make_iterator():
            call_count[0] += 1
            return iter([batch1])

        # Mock the microbatch iterator methods
        data.make_microbatch_iterator_for_packable_sequences = MagicMock(
            side_effect=make_iterator
        )
        data.get_microbatch_iterator_for_packable_sequences_len = MagicMock(
            return_value=(1, 100)  # Local rank has only 1 batch
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
            "sequence_packing": {"enabled": True, "train_mb_tokens": 512},
        }
        mbs = 4
        mock_dp_mesh = MagicMock()

        # Mock the all_reduce to simulate max_batch_ct = 3 across all ranks
        # This means local rank needs 2 dummy batches
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 3  # Simulate max batch count > local count

        mock_all_reduce.side_effect = side_effect

        processed_iterator, iterator_len = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
            tokenizer=mock_tokenizer,
            cp_size=1,
        )

        # Verify local iterator_len is still 1 (not modified)
        assert iterator_len == 1

        # Verify make_microbatch_iterator_for_packable_sequences was called twice
        # (once for main iterator, once for dummy iterator)
        assert data.make_microbatch_iterator_for_packable_sequences.call_count == 2


@pytest.mark.automodel
class TestProcessMicrobatch:
    def test_regular_batching(self, mock_tokenizer):
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )

        cfg = {
            "dtensor_cfg": {"sequence_parallel": False},
        }
        enable_seq_packing = False
        cp_size = 1

        result = process_microbatch(
            mb=mb,
            tokenizer=mock_tokenizer,
            enable_seq_packing=enable_seq_packing,
            cfg=cfg,
            cp_size=cp_size,
        )

        # Verify outputs
        assert isinstance(result, ProcessedInputs)
        assert result.input_ids.shape == (4, 64)
        assert result.attention_mask is not None
        assert result.attention_mask.shape == (4, 64)
        assert result.position_ids is not None
        assert result.position_ids.shape == (4, 64)
        assert result.flash_attn_kwargs == {}
        assert result.vlm_kwargs == {}
        assert result.cp_buffers == []
        assert result.seq_index is None
        assert result.seq_len == 64

    @patch("nemo_rl.models.automodel.data.pack_sequences")
    @patch("nemo_rl.models.automodel.data.get_flash_attention_kwargs")
    def test_sequence_packing(
        self, mock_get_flash_attn, mock_pack_sequences, mock_tokenizer
    ):
        # Create test microbatch
        input_ids = torch.randint(0, 1000, (4, 64))
        input_lengths = torch.tensor([32, 48, 60, 64])
        mb = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )

        cfg = {
            "dtensor_cfg": {"sequence_parallel": False},
            "sequence_packing": {"train_mb_tokens": 256},
        }
        enable_seq_packing = True
        cp_size = 1

        # Mock pack_sequences to return packed inputs
        packed_input_ids = torch.randint(0, 1000, (1, 204))  # Sum of lengths
        packed_position_ids = torch.arange(204).unsqueeze(0)
        mock_pack_sequences.return_value = (packed_input_ids, packed_position_ids, None)

        # Mock flash attention kwargs
        mock_get_flash_attn.return_value = {
            "cu_seqlens": torch.tensor([0, 32, 80, 140, 204])
        }

        result = process_microbatch(
            mb=mb,
            tokenizer=mock_tokenizer,
            enable_seq_packing=enable_seq_packing,
            cfg=cfg,
            cp_size=cp_size,
        )

        # Verify pack_sequences was called
        mock_pack_sequences.assert_called_once()
        assert (
            mock_pack_sequences.call_args[1]["padding_value"]
            == mock_tokenizer.eos_token_id
        )

        # Verify outputs
        assert isinstance(result, ProcessedInputs)
        assert result.input_ids.shape == (1, 204)
        assert result.attention_mask is None
        assert result.position_ids is not None
        assert "cu_seqlens" in result.flash_attn_kwargs
        assert result.vlm_kwargs == {}

    def test_with_multimodal_inputs(self, mock_tokenizer):
        # Create test microbatch with multimodal data
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 64)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
                "pixel_values": torch.randn(2, 3, 224, 224),  # Simulated image data
            }
        )

        # Mock get_multimodal_dict
        mock_multimodal_dict = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mb.get_multimodal_dict = MagicMock(return_value=mock_multimodal_dict)

        cfg = {
            "dtensor_cfg": {"sequence_parallel": False},
        }
        enable_seq_packing = False
        cp_size = 1

        result = process_microbatch(
            mb=mb,
            tokenizer=mock_tokenizer,
            enable_seq_packing=enable_seq_packing,
            cfg=cfg,
            cp_size=cp_size,
        )

        # Verify multimodal kwargs were extracted
        assert isinstance(result, ProcessedInputs)
        assert "pixel_values" in result.vlm_kwargs
        # When multimodal inputs are present, position_ids should be None
        assert result.position_ids is None

    def test_with_context_parallel(self, mock_tokenizer):
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 128)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
            }
        )

        cfg = {
            "dtensor_cfg": {"sequence_parallel": False},
        }
        enable_seq_packing = False
        cp_size = 2  # Context parallel enabled

        result = process_microbatch(
            mb=mb,
            tokenizer=mock_tokenizer,
            enable_seq_packing=enable_seq_packing,
            cfg=cfg,
            cp_size=cp_size,
        )

        # Verify context parallel buffers were created
        assert isinstance(result, ProcessedInputs)
        assert len(result.cp_buffers) == 3  # input_ids, position_ids, seq_index
        assert result.seq_index is not None
        assert result.seq_index.shape == (1, 128)
        # Verify no multimodal inputs with CP
        assert result.vlm_kwargs == {}

    def test_context_parallel_with_multimodal_raises_error(self, mock_tokenizer):
        # Create test microbatch with multimodal data
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 64)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
                "pixel_values": torch.randn(2, 3, 224, 224),
            }
        )

        # Mock get_multimodal_dict to return non-empty dict
        mock_multimodal_dict = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mb.get_multimodal_dict = MagicMock(return_value=mock_multimodal_dict)

        cfg = {
            "dtensor_cfg": {"sequence_parallel": False},
        }
        enable_seq_packing = False
        cp_size = 2  # Context parallel enabled

        with pytest.raises(
            AssertionError, match="are not supported for context parallel"
        ):
            process_microbatch(
                mb=mb,
                tokenizer=mock_tokenizer,
                enable_seq_packing=enable_seq_packing,
                cfg=cfg,
                cp_size=cp_size,
            )

    def test_sequence_packing_with_multimodal_raises_error(self, mock_tokenizer):
        # Create test microbatch with multimodal data
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 64)),
                "input_lengths": torch.tensor([32, 64]),
                "sample_mask": torch.ones(2, dtype=torch.bool),
                "pixel_values": torch.randn(2, 3, 224, 224),
            }
        )

        # Mock get_multimodal_dict to return non-empty dict
        mock_multimodal_dict = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mb.get_multimodal_dict = MagicMock(return_value=mock_multimodal_dict)

        cfg = {
            "dtensor_cfg": {"sequence_parallel": False},
            "sequence_packing": {"train_mb_tokens": 128},
        }
        enable_seq_packing = True
        cp_size = 1

        with pytest.raises(
            AssertionError,
            match="multimodal kwargs are not supported for sequence packing",
        ):
            process_microbatch(
                mb=mb,
                tokenizer=mock_tokenizer,
                enable_seq_packing=enable_seq_packing,
                cfg=cfg,
                cp_size=cp_size,
            )

    def test_sequence_parallel_with_multimodal_raises_error(self, mock_tokenizer):
        # Create test microbatch with multimodal data
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 64)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
                "pixel_values": torch.randn(2, 3, 224, 224),
            }
        )

        # Mock get_multimodal_dict to return non-empty dict
        mock_multimodal_dict = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mb.get_multimodal_dict = MagicMock(return_value=mock_multimodal_dict)

        cfg = {
            "dtensor_cfg": {"sequence_parallel": True},
        }
        enable_seq_packing = False
        cp_size = 1

        with pytest.raises(
            AssertionError, match="Sequence parallel is not supported with multimodal"
        ):
            process_microbatch(
                mb=mb,
                tokenizer=mock_tokenizer,
                enable_seq_packing=enable_seq_packing,
                cfg=cfg,
                cp_size=cp_size,
            )


@pytest.mark.automodel
class TestProcessedInputsProperties:
    """Test ProcessedInputs dataclass properties."""

    def test_has_context_parallel_true(self):
        """Test has_context_parallel returns True when cp_buffers is non-empty."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
            cp_buffers=[torch.randn(2, 64), torch.randn(2, 64)],
        )
        assert processed_inputs.has_context_parallel is True

    def test_has_context_parallel_false(self):
        """Test has_context_parallel returns False when cp_buffers is empty."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
            cp_buffers=[],
        )
        assert processed_inputs.has_context_parallel is False

    def test_has_context_parallel_default(self):
        """Test has_context_parallel returns False by default."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
        )
        assert processed_inputs.has_context_parallel is False

    def test_has_flash_attention_true_with_dict(self):
        """Test has_flash_attention returns True when flash_attn_kwargs has data."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
            flash_attn_kwargs={"cu_seqlens": torch.tensor([0, 32, 64])},
        )
        assert processed_inputs.has_flash_attention is True

    def test_has_flash_attention_false_with_empty_dict(self):
        """Test has_flash_attention returns False when flash_attn_kwargs is empty."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
            flash_attn_kwargs={},
        )
        assert processed_inputs.has_flash_attention is False

    def test_has_flash_attention_default(self):
        """Test has_flash_attention returns False by default."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
        )
        assert processed_inputs.has_flash_attention is False

    def test_is_multimodal_true(self):
        """Test is_multimodal returns True when vlm_kwargs has data."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
            vlm_kwargs={"pixel_values": torch.randn(2, 3, 224, 224)},
        )
        assert processed_inputs.is_multimodal is True

    def test_is_multimodal_false_with_empty_dict(self):
        """Test is_multimodal returns False when vlm_kwargs is empty."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
            vlm_kwargs={},
        )
        assert processed_inputs.is_multimodal is False

    def test_is_multimodal_default(self):
        """Test is_multimodal returns False by default."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
        )
        assert processed_inputs.is_multimodal is False

    def test_combined_properties(self):
        """Test all properties together on a fully configured ProcessedInputs."""
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (2, 64)),
            seq_len=64,
            attention_mask=torch.ones(2, 64, dtype=torch.bool),
            position_ids=torch.arange(64).unsqueeze(0).expand(2, -1),
            flash_attn_kwargs={"cu_seqlens": torch.tensor([0, 32, 64])},
            vlm_kwargs={"pixel_values": torch.randn(2, 3, 224, 224)},
            cp_buffers=[torch.randn(2, 64)],
            seq_index=torch.arange(64).unsqueeze(0),
        )
        assert processed_inputs.has_context_parallel is True
        assert processed_inputs.has_flash_attention is True
        assert processed_inputs.is_multimodal is True


@pytest.mark.automodel
class TestProcessedMicrobatch:
    def test_processed_microbatch_creation(self, mock_tokenizer):
        """Test that ProcessedMicrobatch correctly stores all attributes."""
        # Create test data
        data_dict = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )

        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (4, 64)),
            seq_len=64,
            attention_mask=torch.ones(4, 64, dtype=torch.bool),
            position_ids=torch.arange(64).unsqueeze(0).expand(4, -1),
        )

        processed_mb = ProcessedMicrobatch(
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=4,
            original_seq_len=64,
        )

        # Verify all attributes are correctly stored
        assert processed_mb.data_dict is data_dict
        assert processed_mb.processed_inputs is processed_inputs
        assert processed_mb.original_batch_size == 4
        assert processed_mb.original_seq_len == 64

    def test_processed_microbatch_preserves_original_dimensions(self, mock_tokenizer):
        """Test that original dimensions are preserved even after processing changes shapes."""
        # Simulate sequence packing where batch dimension changes
        original_batch_size = 4
        original_seq_len = 64

        data_dict = BatchedDataDict(
            {
                "input_ids": torch.randint(
                    0, 1000, (original_batch_size, original_seq_len)
                ),
                "sample_mask": torch.ones(original_batch_size, dtype=torch.bool),
            }
        )

        # Simulate packed sequence (batch becomes 1, seq_len increases)
        packed_seq_len = 200
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (1, packed_seq_len)),
            seq_len=packed_seq_len,
        )

        processed_mb = ProcessedMicrobatch(
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=original_batch_size,
            original_seq_len=original_seq_len,
        )

        # Verify original dimensions are preserved
        assert processed_mb.original_batch_size == original_batch_size
        assert processed_mb.original_seq_len == original_seq_len
        # But processed inputs have different shape
        assert processed_mb.processed_inputs.input_ids.shape[0] == 1
        assert processed_mb.processed_inputs.seq_len == packed_seq_len


@pytest.mark.automodel
class TestMakeProcessedMicrobatchIterator:
    def test_basic_iteration(self, mock_tokenizer):
        """Test basic iteration over microbatches."""
        # Create test data
        batch1 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 32)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
            }
        )
        batch2 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 32)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
            }
        )

        raw_iterator = iter([batch1, batch2])
        cfg = {
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        # Collect all processed microbatches
        processed_mbs = list(processed_iterator)

        # Verify correct number of batches
        assert len(processed_mbs) == 2

        # Verify each is a ProcessedMicrobatch
        for processed_mb in processed_mbs:
            assert isinstance(processed_mb, ProcessedMicrobatch)
            assert isinstance(processed_mb.processed_inputs, ProcessedInputs)
            assert processed_mb.original_batch_size == 2
            assert processed_mb.original_seq_len == 32

    def test_preserves_data_dict_reference(self, mock_tokenizer):
        """Test that the original data dict is accessible."""
        input_ids = torch.randint(0, 1000, (4, 64))
        batch = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": torch.ones(4, dtype=torch.bool),
                "custom_field": torch.randn(4, 10),
            }
        )

        raw_iterator = iter([batch])
        cfg = {
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        processed_mb = next(processed_iterator)

        # Verify data_dict is accessible and contains original fields
        assert "input_ids" in processed_mb.data_dict
        assert "sample_mask" in processed_mb.data_dict
        assert "custom_field" in processed_mb.data_dict
        assert processed_mb.data_dict is batch

    def test_processed_inputs_are_correct(self, mock_tokenizer):
        """Test that processed inputs have correct attributes."""
        batch = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (3, 48)),
                "sample_mask": torch.ones(3, dtype=torch.bool),
            }
        )

        raw_iterator = iter([batch])
        cfg = {
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        processed_mb = next(processed_iterator)
        processed_inputs = processed_mb.processed_inputs

        # Verify processed inputs
        assert processed_inputs.input_ids.shape == (3, 48)
        assert processed_inputs.attention_mask is not None
        assert processed_inputs.attention_mask.shape == (3, 48)
        assert processed_inputs.position_ids is not None
        assert processed_inputs.position_ids.shape == (3, 48)
        assert processed_inputs.seq_len == 48
        assert processed_inputs.cp_buffers == []
        assert processed_inputs.vlm_kwargs == {}

    def test_with_context_parallel(self, mock_tokenizer):
        """Test iteration with context parallel enabled."""
        batch = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 64)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
            }
        )

        raw_iterator = iter([batch])
        cfg = {
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=2,  # Context parallel enabled
        )

        processed_mb = next(processed_iterator)

        # Verify context parallel buffers are created
        assert len(processed_mb.processed_inputs.cp_buffers) == 3
        assert processed_mb.processed_inputs.seq_index is not None

    def test_empty_iterator(self, mock_tokenizer):
        """Test that empty iterator yields nothing."""
        raw_iterator = iter([])
        cfg = {
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        processed_mbs = list(processed_iterator)
        assert len(processed_mbs) == 0

    def test_varying_batch_sizes(self, mock_tokenizer):
        """Test iteration with varying batch sizes."""
        batch1 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 32)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
            }
        )
        batch2 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 32)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        batch3 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (1, 32)),
                "sample_mask": torch.ones(1, dtype=torch.bool),
            }
        )

        raw_iterator = iter([batch1, batch2, batch3])
        cfg = {
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        processed_mbs = list(processed_iterator)

        # Verify each batch has correct original dimensions
        assert processed_mbs[0].original_batch_size == 2
        assert processed_mbs[1].original_batch_size == 4
        assert processed_mbs[2].original_batch_size == 1

        # All have same seq_len
        for processed_mb in processed_mbs:
            assert processed_mb.original_seq_len == 32

    @patch("nemo_rl.models.automodel.data.pack_sequences")
    @patch("nemo_rl.models.automodel.data.get_flash_attention_kwargs")
    def test_with_sequence_packing_enabled(
        self, mock_get_flash_attn, mock_pack_sequences, mock_tokenizer
    ):
        """Test iteration with sequence packing enabled via config."""
        input_ids = torch.randint(0, 1000, (4, 64))
        input_lengths = torch.tensor([32, 48, 60, 64])
        batch = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )

        raw_iterator = iter([batch])
        cfg = {
            "sequence_packing": {"enabled": True, "train_mb_tokens": 256},
            "dtensor_cfg": {"sequence_parallel": False},
        }

        # Mock pack_sequences to return packed inputs
        packed_input_ids = torch.randint(0, 1000, (1, 204))
        packed_position_ids = torch.arange(204).unsqueeze(0)
        mock_pack_sequences.return_value = (packed_input_ids, packed_position_ids, None)

        # Mock flash attention kwargs
        mock_get_flash_attn.return_value = {
            "cu_seqlens": torch.tensor([0, 32, 80, 140, 204])
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        processed_mbs = list(processed_iterator)

        # Verify sequence packing was applied
        assert len(processed_mbs) == 1
        processed_mb = processed_mbs[0]
        assert processed_mb.original_batch_size == 4  # Original batch size preserved
        assert processed_mb.original_seq_len == 64  # Original seq len preserved
        assert processed_mb.processed_inputs.seq_len == 204  # Packed seq len
        assert "cu_seqlens" in processed_mb.processed_inputs.flash_attn_kwargs

        # Verify pack_sequences was called
        mock_pack_sequences.assert_called_once()

    def test_config_without_sequence_packing_key(self, mock_tokenizer):
        """Test iteration when sequence_packing key is missing from config."""
        batch = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (2, 32)),
                "sample_mask": torch.ones(2, dtype=torch.bool),
            }
        )

        raw_iterator = iter([batch])
        cfg = {
            # No "sequence_packing" key
            "dtensor_cfg": {"sequence_parallel": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        processed_mbs = list(processed_iterator)

        # Should work with default (no sequence packing)
        assert len(processed_mbs) == 1
        assert processed_mbs[0].processed_inputs.flash_attn_kwargs == {}


@pytest.mark.automodel
class TestProcessGlobalBatch:
    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_basic_processing(self, mock_all_reduce, mock_loss_fn, mock_dp_mesh):
        # Create test data
        input_ids = torch.randint(0, 1000, (16, 64))
        sample_mask = torch.ones(16, dtype=torch.bool)
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": sample_mask,
            }
        )

        # Mock get_batch
        batch_data = BatchedDataDict(
            {
                "input_ids": input_ids[:4],
                "sample_mask": sample_mask[:4],
            }
        )
        data.get_batch = MagicMock(return_value=batch_data)

        # Mock all_reduce to simulate reduction across ranks
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 8  # Simulated global valid seqs (4 * 2 DP ranks)
            tensor[1] = 512  # Simulated global valid tokens (4 * 64 * 2 DP ranks)

        mock_all_reduce.side_effect = side_effect

        result = process_global_batch(
            data=data,
            loss_fn=mock_loss_fn,
            dp_group=mock_dp_mesh.get_group(),
            batch_idx=0,
            batch_size=4,
        )

        # Verify get_batch was called correctly
        data.get_batch.assert_called_once_with(batch_idx=0, batch_size=4)

        # Verify results
        assert "batch" in result
        assert result["batch"]["input_ids"].shape == (4, 64)
        assert result["global_valid_seqs"] == 8
        assert result["global_valid_toks"] == 512

        # Verify all_reduce was called
        mock_all_reduce.assert_called_once()

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_with_token_mask(self, mock_all_reduce, mock_loss_fn, mock_dp_mesh):
        # Create test data
        input_ids = torch.randint(0, 1000, (8, 64))
        sample_mask = torch.ones(8, dtype=torch.bool)
        token_mask = torch.ones(8, 64, dtype=torch.bool)
        # Mask out some tokens
        token_mask[:, :10] = 0  # Mask first 10 tokens

        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": sample_mask,
                "token_mask": token_mask,
            }
        )

        # Mock get_batch
        batch_data = BatchedDataDict(
            {
                "input_ids": input_ids[:4],
                "sample_mask": sample_mask[:4],
                "token_mask": token_mask[:4],
            }
        )
        data.get_batch = MagicMock(return_value=batch_data)

        # Mock all_reduce
        def side_effect(tensor, *args, **kwargs):
            # Local: 4 seqs, ~(4 * 54) tokens (excluding first 10 and last position)
            tensor[0] = 8  # Global valid seqs
            tensor[1] = 432  # Global valid tokens (approximately)

        mock_all_reduce.side_effect = side_effect

        result = process_global_batch(
            data=data,
            loss_fn=mock_loss_fn,
            dp_group=mock_dp_mesh.get_group(),
            batch_idx=0,
            batch_size=4,
        )

        # Verify batch has token_mask
        assert "token_mask" in result["batch"]
        assert result["batch"]["token_mask"].shape == (4, 64)

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_token_level_loss_requires_token_mask(self, mock_all_reduce, mock_dp_mesh):
        # Create loss function with token-level loss
        loss_fn = MagicMock()
        loss_fn.loss_type = LossType.TOKEN_LEVEL

        # Create test data WITHOUT token_mask
        input_ids = torch.randint(0, 1000, (8, 64))
        sample_mask = torch.ones(8, dtype=torch.bool)
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": sample_mask,
            }
        )

        # Mock get_batch
        batch_data = BatchedDataDict(
            {
                "input_ids": input_ids[:4],
                "sample_mask": sample_mask[:4],
            }
        )
        data.get_batch = MagicMock(return_value=batch_data)

        # Mock all_reduce
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 8
            tensor[1] = 512

        mock_all_reduce.side_effect = side_effect

        with pytest.raises(AssertionError, match="token_mask must be present"):
            process_global_batch(
                data=data,
                loss_fn=loss_fn,
                dp_group=mock_dp_mesh.get_group(),
                batch_idx=0,
                batch_size=4,
            )

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_missing_sample_mask_raises_error(
        self, mock_all_reduce, mock_loss_fn, mock_dp_mesh
    ):
        # Create test data WITHOUT sample_mask
        input_ids = torch.randint(0, 1000, (8, 64))
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
            }
        )

        # Mock get_batch to return data without sample_mask
        batch_data = BatchedDataDict(
            {
                "input_ids": input_ids[:4],
            }
        )
        data.get_batch = MagicMock(return_value=batch_data)

        with pytest.raises(AssertionError, match="sample_mask must be present"):
            process_global_batch(
                data=data,
                loss_fn=mock_loss_fn,
                dp_group=mock_dp_mesh.get_group(),
                batch_idx=0,
                batch_size=4,
            )

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_multiple_batch_processing(
        self, mock_all_reduce, mock_loss_fn, mock_dp_mesh
    ):
        # Create test data
        input_ids = torch.randint(0, 1000, (16, 64))
        sample_mask = torch.ones(16, dtype=torch.bool)
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": sample_mask,
            }
        )

        # Mock get_batch to return different batches
        def get_batch_side_effect(batch_idx, batch_size):
            start = batch_idx * batch_size
            end = start + batch_size
            return BatchedDataDict(
                {
                    "input_ids": input_ids[start:end],
                    "sample_mask": sample_mask[start:end],
                }
            )

        data.get_batch = MagicMock(side_effect=get_batch_side_effect)

        # Mock all_reduce
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 8  # Global valid seqs
            tensor[1] = 512  # Global valid tokens

        mock_all_reduce.side_effect = side_effect

        # Process first batch
        result1 = process_global_batch(
            data=data,
            loss_fn=mock_loss_fn,
            dp_group=mock_dp_mesh.get_group(),
            batch_idx=0,
            batch_size=4,
        )

        # Process second batch
        result2 = process_global_batch(
            data=data,
            loss_fn=mock_loss_fn,
            dp_group=mock_dp_mesh.get_group(),
            batch_idx=1,
            batch_size=4,
        )

        # Verify both batches were processed correctly
        assert result1["batch"]["input_ids"].shape == (4, 64)
        assert result2["batch"]["input_ids"].shape == (4, 64)

        # Verify get_batch was called with correct indices
        assert data.get_batch.call_count == 2
        data.get_batch.assert_any_call(batch_idx=0, batch_size=4)
        data.get_batch.assert_any_call(batch_idx=1, batch_size=4)

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_loss_fn_without_loss_type_attribute(self, mock_all_reduce, mock_dp_mesh):
        """Test that process_global_batch works when loss_fn has no loss_type attribute."""
        # Create test data
        input_ids = torch.randint(0, 1000, (8, 64))
        sample_mask = torch.ones(8, dtype=torch.bool)
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": sample_mask,
            }
        )

        # Mock get_batch
        batch_data = BatchedDataDict(
            {
                "input_ids": input_ids[:4],
                "sample_mask": sample_mask[:4],
            }
        )
        data.get_batch = MagicMock(return_value=batch_data)

        # Create loss_fn WITHOUT loss_type attribute
        loss_fn_without_type = MagicMock(spec=[])  # Empty spec means no attributes

        # Mock all_reduce
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 8
            tensor[1] = 512

        mock_all_reduce.side_effect = side_effect

        # Should not raise any error
        result = process_global_batch(
            data=data,
            loss_fn=loss_fn_without_type,
            dp_group=mock_dp_mesh.get_group(),
            batch_idx=0,
            batch_size=4,
        )

        # Verify results are still correct
        assert "batch" in result
        assert result["batch"]["input_ids"].shape == (4, 64)
        assert result["global_valid_seqs"] == 8
        assert result["global_valid_toks"] == 512

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_with_partial_sample_mask(
        self, mock_all_reduce, mock_loss_fn, mock_dp_mesh
    ):
        """Test processing with some samples masked out."""
        # Create test data with some samples masked
        input_ids = torch.randint(0, 1000, (8, 64))
        sample_mask = torch.tensor(
            [True, True, False, False, True, True, True, False], dtype=torch.bool
        )  # 5 valid samples
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": sample_mask,
            }
        )

        # Mock get_batch
        batch_data = BatchedDataDict(
            {
                "input_ids": input_ids[:4],
                "sample_mask": sample_mask[:4],  # [True, True, False, False] = 2 valid
            }
        )
        data.get_batch = MagicMock(return_value=batch_data)

        # Mock all_reduce - local has 2 valid seqs, assume another rank has 3
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 5  # Global valid seqs (2 + 3)
            tensor[1] = 320  # Global valid tokens

        mock_all_reduce.side_effect = side_effect

        result = process_global_batch(
            data=data,
            loss_fn=mock_loss_fn,
            dp_group=mock_dp_mesh.get_group(),
            batch_idx=0,
            batch_size=4,
        )

        # Verify results
        assert result["global_valid_seqs"] == 5
        assert result["global_valid_toks"] == 320


@pytest.mark.automodel
class TestIntegrationScenarios:
    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_full_pipeline_regular_batching(
        self, mock_all_reduce, mock_tokenizer, mock_loss_fn, mock_dp_mesh
    ):
        # Create test data
        input_ids = torch.randint(0, 1000, (16, 64))
        sample_mask = torch.ones(16, dtype=torch.bool)
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "sample_mask": sample_mask,
            }
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
            "dtensor_cfg": {"sequence_parallel": False},
        }
        mbs = 4
        cp_size = 1

        # Mock get_batch
        def get_batch_side_effect(batch_idx, batch_size):
            start = batch_idx * batch_size
            end = start + batch_size
            return BatchedDataDict(
                {
                    "input_ids": input_ids[start:end],
                    "sample_mask": sample_mask[start:end],
                }
            )

        data.get_batch = MagicMock(side_effect=get_batch_side_effect)

        # Mock all_reduce
        def all_reduce_side_effect(tensor, *args, **kwargs):
            tensor[0] = 8  # Global valid seqs
            tensor[1] = 512  # Global valid tokens

        mock_all_reduce.side_effect = all_reduce_side_effect

        # Step 1: Process global batch
        global_batch_result = process_global_batch(
            data=data,
            loss_fn=mock_loss_fn,
            dp_group=mock_dp_mesh.get_group(),
            batch_idx=0,
            batch_size=4,
        )

        batch = global_batch_result["batch"]

        # Step 2: Get processed microbatch iterator (now integrated with processing)
        processed_iterator, iterator_len = get_microbatch_iterator(
            data=batch,
            cfg=cfg,
            mbs=2,
            dp_mesh=mock_dp_mesh,
            tokenizer=mock_tokenizer,
            cp_size=cp_size,
        )

        # Step 3: Iterate through processed microbatches
        processed_mbs = list(processed_iterator)

        # Verify pipeline results
        assert len(processed_mbs) == iterator_len
        assert all(isinstance(mb, ProcessedMicrobatch) for mb in processed_mbs)
        assert all(mb.processed_inputs.input_ids.shape[0] == 2 for mb in processed_mbs)
        assert global_batch_result["global_valid_seqs"] == 8
        assert global_batch_result["global_valid_toks"] == 512
