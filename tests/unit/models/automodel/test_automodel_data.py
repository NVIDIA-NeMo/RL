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
    get_microbatch_iterator,
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
    def test_regular_batching(self):
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (16, 128)),
                "sample_mask": torch.ones(16, dtype=torch.bool),
            }
        )

        cfg = {"dynamic_batching": {"enabled": False}}
        mbs = 4
        enable_seq_packing = False
        mock_dp_mesh = MagicMock()

        mb_iterator, iterator_len, dummy_iterator = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            enable_seq_packing=enable_seq_packing,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
        )

        # Verify iterator length
        assert iterator_len == 4  # 16 / 4 = 4

        # Verify we can iterate through the data
        batches = list(mb_iterator)
        assert len(batches) == 4
        assert batches[0]["input_ids"].shape[0] == 4

        # Verify dummy iterator is empty
        assert list(dummy_iterator) == []

    def test_dynamic_batching(self):
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (8, 128)),
                "sample_mask": torch.ones(8, dtype=torch.bool),
            }
        )

        # Mock the microbatch iterator methods
        mock_iterator = iter([data, data, data])
        data.make_microbatch_iterator_with_dynamic_shapes = MagicMock(
            return_value=mock_iterator
        )
        data.get_microbatch_iterator_dynamic_shapes_len = MagicMock(return_value=3)

        cfg = {"dynamic_batching": {"enabled": True}}
        mbs = 4
        enable_seq_packing = False
        mock_dp_mesh = MagicMock()

        mb_iterator, iterator_len, dummy_iterator = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            enable_seq_packing=enable_seq_packing,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
        )

        # Verify dynamic batching was used
        assert iterator_len == 3
        data.make_microbatch_iterator_with_dynamic_shapes.assert_called_once()
        data.get_microbatch_iterator_dynamic_shapes_len.assert_called_once()

        # Verify dummy iterator is empty
        assert list(dummy_iterator) == []

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_sequence_packing(self, mock_all_reduce):
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (8, 128)),
                "input_lengths": torch.randint(64, 128, (8,)),
                "sample_mask": torch.ones(8, dtype=torch.bool),
            }
        )

        # Mock the microbatch iterator methods
        mock_iterator = iter([data, data])
        data.make_microbatch_iterator_for_packable_sequences = MagicMock(
            return_value=mock_iterator
        )
        data.get_microbatch_iterator_for_packable_sequences_len = MagicMock(
            return_value=(2, 100)
        )

        cfg = {"dynamic_batching": {"enabled": False}}
        mbs = 4
        enable_seq_packing = True
        mock_dp_mesh = MagicMock()

        # Mock the all_reduce to simulate max_batch_ct = 2 across all ranks
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 2  # Simulate max batch count

        mock_all_reduce.side_effect = side_effect

        mb_iterator, iterator_len, dummy_iterator = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            enable_seq_packing=enable_seq_packing,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
        )

        # Verify sequence packing was used
        assert iterator_len == 2
        data.make_microbatch_iterator_for_packable_sequences.assert_called()
        data.get_microbatch_iterator_for_packable_sequences_len.assert_called_once()

        # Verify all_reduce was called to synchronize batch counts
        mock_all_reduce.assert_called_once()

        # Verify dummy iterator is empty (when all ranks have same batch count)
        assert list(dummy_iterator) == []

    @patch("nemo_rl.models.automodel.data.torch.distributed.all_reduce")
    def test_sequence_packing_with_uneven_batch_counts(self, mock_all_reduce):
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (8, 128)),
                "input_lengths": torch.randint(64, 128, (8,)),
                "sample_mask": torch.ones(8, dtype=torch.bool),
            }
        )

        # Mock the microbatch iterator methods
        mock_iterator1 = iter([data, data])
        mock_iterator2 = iter([data, data])
        data.make_microbatch_iterator_for_packable_sequences = MagicMock(
            side_effect=[mock_iterator1, mock_iterator2]
        )
        data.get_microbatch_iterator_for_packable_sequences_len = MagicMock(
            return_value=(2, 100)
        )

        cfg = {"dynamic_batching": {"enabled": False}}
        mbs = 4
        enable_seq_packing = True
        mock_dp_mesh = MagicMock()

        # Mock all_reduce to simulate max_batch_ct = 4 (higher than local 2)
        def side_effect(tensor, *args, **kwargs):
            tensor[0] = 4  # Simulate max batch count from other ranks

        mock_all_reduce.side_effect = side_effect

        mb_iterator, iterator_len, dummy_iterator = get_microbatch_iterator(
            data=data,
            cfg=cfg,
            enable_seq_packing=enable_seq_packing,
            mbs=mbs,
            dp_mesh=mock_dp_mesh,
        )

        # Verify sequence packing was used
        assert iterator_len == 2

        # Verify dummy iterator has 2 batches (4 - 2 = 2)
        dummy_batches = list(dummy_iterator)
        assert len(dummy_batches) == 2


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
            batch_idx=0,
            batch_size=4,
            loss_fn=mock_loss_fn,
            dp_mesh=mock_dp_mesh,
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
            batch_idx=0,
            batch_size=4,
            loss_fn=mock_loss_fn,
            dp_mesh=mock_dp_mesh,
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
                batch_idx=0,
                batch_size=4,
                loss_fn=loss_fn,
                dp_mesh=mock_dp_mesh,
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
                batch_idx=0,
                batch_size=4,
                loss_fn=mock_loss_fn,
                dp_mesh=mock_dp_mesh,
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
            batch_idx=0,
            batch_size=4,
            loss_fn=mock_loss_fn,
            dp_mesh=mock_dp_mesh,
        )

        # Process second batch
        result2 = process_global_batch(
            data=data,
            batch_idx=1,
            batch_size=4,
            loss_fn=mock_loss_fn,
            dp_mesh=mock_dp_mesh,
        )

        # Verify both batches were processed correctly
        assert result1["batch"]["input_ids"].shape == (4, 64)
        assert result2["batch"]["input_ids"].shape == (4, 64)

        # Verify get_batch was called with correct indices
        assert data.get_batch.call_count == 2
        data.get_batch.assert_any_call(batch_idx=0, batch_size=4)
        data.get_batch.assert_any_call(batch_idx=1, batch_size=4)


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
            "dtensor_cfg": {"sequence_parallel": False},
        }
        mbs = 4
        enable_seq_packing = False
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
            batch_idx=0,
            batch_size=4,
            loss_fn=mock_loss_fn,
            dp_mesh=mock_dp_mesh,
        )

        batch = global_batch_result["batch"]

        # Step 2: Get microbatch iterator
        mb_iterator, iterator_len, _ = get_microbatch_iterator(
            data=batch,
            cfg=cfg,
            enable_seq_packing=enable_seq_packing,
            mbs=2,
            dp_mesh=mock_dp_mesh,
        )

        # Step 3: Process each microbatch
        processed_mbs = []
        for mb in mb_iterator:
            result = process_microbatch(
                mb=mb,
                tokenizer=mock_tokenizer,
                enable_seq_packing=enable_seq_packing,
                cfg=cfg,
                cp_size=cp_size,
            )
            processed_mbs.append(result)

        # Verify pipeline results
        assert len(processed_mbs) == iterator_len
        assert all(isinstance(mb, ProcessedInputs) for mb in processed_mbs)
        assert all(mb.input_ids.shape[0] == 2 for mb in processed_mbs)
        assert global_batch_result["global_valid_seqs"] == 8
        assert global_batch_result["global_valid_toks"] == 512
