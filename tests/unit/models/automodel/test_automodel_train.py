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

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    ProcessedInputs,
    ProcessedMicrobatch,
    check_sequence_dim,
    make_processed_microbatch_iterator,
)
from nemo_rl.models.automodel.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    ScorePostProcessor,
    TopkLogitsPostProcessor,
    apply_temperature_scaling,
    automodel_forward_backward,
    extract_logits,
    forward_with_post_processing_fn,
    model_forward,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.return_value = MagicMock(logits=torch.randn(4, 64, 32000))
    return model


@pytest.fixture
def mock_loss_fn():
    loss_fn = MagicMock()
    loss_fn.return_value = (torch.tensor(0.5), {"loss": 0.5})
    return loss_fn


@pytest.fixture
def mock_device_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    mesh.__getitem__ = MagicMock(return_value=mesh)
    return mesh


@pytest.fixture
def mock_cp_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    return mesh


@pytest.fixture
def mock_tp_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    return mesh


@pytest.fixture
def base_cfg():
    return {
        "dtensor_cfg": {"sequence_parallel": False},
        "sequence_packing": {"train_mb_tokens": 256},
        "generation": {"temperature": 1.0},
    }


@pytest.fixture
def processed_inputs_no_flash():
    return ProcessedInputs(
        input_ids=torch.randint(0, 1000, (4, 64)),
        seq_len=64,
        attention_mask=torch.ones(4, 64, dtype=torch.bool),
        position_ids=torch.arange(64).repeat(4, 1),
        flash_attn_kwargs={},
        vlm_kwargs={},
        cp_buffers=[],
        seq_index=None,
    )


@pytest.fixture
def processed_inputs_with_flash():
    @dataclass
    class MockFlashAttnKwargs:
        cu_seqlens_q: torch.Tensor

    flash_kwargs = MockFlashAttnKwargs(cu_seqlens_q=torch.tensor([0, 32, 64, 96, 128]))
    return ProcessedInputs(
        input_ids=torch.randint(0, 1000, (1, 128)),
        seq_len=128,
        attention_mask=None,
        position_ids=torch.arange(128).unsqueeze(0),
        flash_attn_kwargs=flash_kwargs,
        vlm_kwargs={},
        cp_buffers=[],
        seq_index=None,
    )


@pytest.fixture
def processed_inputs_multimodal():
    return ProcessedInputs(
        input_ids=torch.randint(0, 1000, (2, 64)),
        seq_len=64,
        attention_mask=torch.ones(2, 64, dtype=torch.bool),
        position_ids=None,
        flash_attn_kwargs={},
        vlm_kwargs={"pixel_values": torch.randn(2, 3, 224, 224)},
        cp_buffers=[],
        seq_index=None,
    )


# =====================
# Test model_forward
# =====================
@pytest.mark.automodel
class TestModelForward:
    def test_basic_forward(self, mock_model, processed_inputs_no_flash):
        result = model_forward(mock_model, processed_inputs_no_flash)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "input_ids" in call_kwargs
        assert "attention_mask" in call_kwargs
        assert "position_ids" in call_kwargs
        assert call_kwargs["use_cache"] is False

    def test_forward_with_flash_attention(
        self, mock_model, processed_inputs_with_flash
    ):
        result = model_forward(mock_model, processed_inputs_with_flash)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" in call_kwargs

    def test_forward_with_multimodal(self, mock_model, processed_inputs_multimodal):
        result = model_forward(mock_model, processed_inputs_multimodal)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "pixel_values" in call_kwargs
        # Flash attention should be removed for multimodal
        assert "flash_attn_kwargs" not in call_kwargs

    def test_forward_reward_model_removes_flash_attn(
        self, mock_model, processed_inputs_with_flash
    ):
        result = model_forward(
            mock_model, processed_inputs_with_flash, is_reward_model=True
        )

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        # Flash attention should be removed for reward models
        assert "flash_attn_kwargs" not in call_kwargs

    def test_forward_disallow_flash_attn_args(
        self, mock_model, processed_inputs_with_flash
    ):
        result = model_forward(
            mock_model, processed_inputs_with_flash, allow_flash_attn_args=False
        )

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" not in call_kwargs


# =====================
# Test extract_logits
# =====================
@pytest.mark.automodel
class TestExtractLogits:
    def test_tensor_output(self, mock_model):
        tensor_output = torch.randn(4, 64, 32000)
        result = extract_logits(mock_model, tensor_output)
        assert torch.equal(result, tensor_output)

    def test_output_with_logits_attribute(self, mock_model):
        mock_output = MagicMock()
        mock_output.logits = torch.randn(4, 64, 32000)
        result = extract_logits(mock_model, mock_output)
        assert torch.equal(result, mock_output.logits)

    def test_output_with_last_hidden_state(self, mock_model):
        mock_output = MagicMock(spec=["last_hidden_state"])
        mock_output.last_hidden_state = torch.randn(4, 64, 4096)
        mock_model.lm_head = MagicMock(return_value=torch.randn(4, 64, 32000))

        result = extract_logits(mock_model, mock_output)

        mock_model.lm_head.assert_called_once_with(mock_output.last_hidden_state)


# =====================
# Test apply_temperature_scaling
# =====================
@pytest.mark.automodel
class TestApplyTemperatureScaling:
    def test_temperature_scaling_applied(self):
        logits = torch.randn(4, 64, 32000)
        original_logits = logits.clone()
        cfg = {"generation": {"temperature": 2.0}}

        result = apply_temperature_scaling(logits, cfg)

        # Should be divided by temperature
        expected = original_logits / 2.0
        assert torch.allclose(result, expected)

    def test_no_scaling_without_generation_config(self):
        logits = torch.randn(4, 64, 32000)
        original_logits = logits.clone()
        cfg = {}

        result = apply_temperature_scaling(logits, cfg)

        assert torch.equal(result, original_logits)

    def test_no_scaling_with_none_generation(self):
        logits = torch.randn(4, 64, 32000)
        original_logits = logits.clone()
        cfg = {"generation": None}

        result = apply_temperature_scaling(logits, cfg)

        assert torch.equal(result, original_logits)


# =====================
# Test LossPostProcessor
# =====================
@pytest.mark.automodel
class TestLossPostProcessor:
    def test_basic_loss_computation(
        self,
        base_cfg,
        mock_loss_fn,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
        processed_inputs_no_flash,
    ):
        processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            }
        )
        global_valid_seqs = torch.tensor(8)
        global_valid_toks = torch.tensor(512)

        loss, metrics = processor(
            logits=logits,
            mb=mb,
            processed_inputs=processed_inputs_no_flash,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )

        # Verify loss function was called
        mock_loss_fn.assert_called_once()
        call_args = mock_loss_fn.call_args[0]
        assert torch.is_tensor(call_args[0])  # logits
        assert call_args[2] == global_valid_seqs  # global_valid_seqs
        assert call_args[3] == global_valid_toks  # global_valid_toks

    @patch("nemo_rl.models.automodel.train.SequencePackingLossWrapper")
    def test_loss_with_sequence_packing(
        self,
        mock_wrapper_class,
        base_cfg,
        mock_loss_fn,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
        processed_inputs_with_flash,
    ):
        # Setup mock wrapper
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.return_value = (torch.tensor(0.5), {"loss": 0.5})
        mock_wrapper_class.return_value = mock_wrapper_instance

        processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        batch_size = 1
        seq_len = 128
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            }
        )
        global_valid_seqs = torch.tensor(4)
        global_valid_toks = torch.tensor(128)

        loss, metrics = processor(
            logits=logits,
            mb=mb,
            processed_inputs=processed_inputs_with_flash,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )

        # Verify SequencePackingLossWrapper was created
        mock_wrapper_class.assert_called_once()
        # Verify the wrapper was called instead of raw loss_fn
        mock_wrapper_instance.assert_called_once()

    def test_loss_processor_initialization(
        self,
        base_cfg,
        mock_loss_fn,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=2,
            dp_size=4,
        )

        assert processor.loss_fn is mock_loss_fn
        assert processor.cfg is base_cfg
        assert processor.cp_size == 2
        assert processor.dp_size == 4


# =====================
# Test ScorePostProcessor
# =====================
@pytest.mark.automodel
class TestScorePostProcessor:
    def test_basic_scoring(self, base_cfg):
        processor = ScorePostProcessor(cfg=base_cfg)

        # Create mock logits with shape [batch_size, 1]
        logits = torch.randn(4, 1)

        result = processor(logits)

        assert result.shape == (4,)
        assert result.dtype == torch.float32

    def test_scoring_with_sequence_logits(self, base_cfg):
        processor = ScorePostProcessor(cfg=base_cfg)

        # Create mock logits with shape [batch_size, seq_len, 1]
        logits = torch.randn(4, 64, 1)

        result = processor(logits)

        assert result.shape == (4, 64)
        assert result.dtype == torch.float32


# =====================
# Test LogprobsPostProcessor
# =====================
@pytest.mark.automodel
class TestLogprobsPostProcessor:
    def test_basic_logprobs_computation(
        self, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        processor = LogprobsPostProcessor(
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_lengths = torch.full((batch_size,), seq_len)

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
        )

        result = processor(
            logits=logits,
            processed_inputs=processed_inputs,
            input_lengths=input_lengths,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            enable_seq_packing=False,
        )

        assert result.shape == (batch_size, seq_len)

    def test_logprobs_with_chunking(
        self, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        cfg_with_chunk = {**base_cfg, "logprob_chunk_size": 16}
        processor = LogprobsPostProcessor(
            cfg=cfg_with_chunk,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_lengths = torch.full((batch_size,), seq_len)

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
        )

        result = processor(
            logits=logits,
            processed_inputs=processed_inputs,
            input_lengths=input_lengths,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            enable_seq_packing=False,
        )

        assert result.shape == (batch_size, seq_len)


# =====================
# Test TopkLogitsPostProcessor
# =====================
@pytest.mark.automodel
class TestTopkLogitsPostProcessor:
    def test_basic_topk(self, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh):
        k = 10
        processor = TopkLogitsPostProcessor(
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            k=k,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.full((batch_size,), seq_len)

        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
        )

        vals, idx = processor(
            logits=logits,
            processed_inputs=processed_inputs,
            input_lengths=input_lengths,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            enable_seq_packing=False,
        )

        assert vals.shape == (batch_size, seq_len, k)
        assert idx.shape == (batch_size, seq_len, k)


# =====================
# Test ProcessedMicrobatch
# =====================
@pytest.mark.automodel
class TestProcessedMicrobatch:
    def test_processed_microbatch_creation(self, processed_inputs_no_flash):
        data_dict = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )

        pm = ProcessedMicrobatch(
            data_dict=data_dict,
            processed_inputs=processed_inputs_no_flash,
            original_batch_size=4,
            original_seq_len=64,
        )

        assert pm.original_batch_size == 4
        assert pm.original_seq_len == 64
        assert pm.data_dict is data_dict
        assert pm.processed_inputs is processed_inputs_no_flash


# =====================
# Test make_processed_microbatch_iterator
# =====================
@pytest.mark.automodel
class TestMakeProcessedMicrobatchIterator:
    def test_basic_iteration(self, mock_tokenizer):
        # Create test data
        data_dict1 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        data_dict2 = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "sample_mask": torch.ones(4, dtype=torch.bool),
            }
        )

        # Mock get_multimodal_dict to return empty dict
        data_dict1.get_multimodal_dict = MagicMock(return_value={})
        data_dict2.get_multimodal_dict = MagicMock(return_value={})

        raw_iterator = iter([data_dict1, data_dict2])

        cfg = {
            "dtensor_cfg": {"sequence_parallel": False},
            "sequence_packing": {"enabled": False},
        }

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            tokenizer=mock_tokenizer,
            cfg=cfg,
            cp_size=1,
        )

        results = list(processed_iterator)

        assert len(results) == 2
        assert all(isinstance(pm, ProcessedMicrobatch) for pm in results)
        assert all(pm.original_batch_size == 4 for pm in results)
        assert all(pm.original_seq_len == 64 for pm in results)


# =====================
# Test check_sequence_dim
# =====================
@pytest.mark.automodel
class TestCheckSequenceDim:
    def test_consistent_sequence_dim(self):
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "attention_mask": torch.ones(4, 64, dtype=torch.bool),
                "token_mask": torch.ones(4, 64, dtype=torch.bool),
                "sample_mask": torch.ones(4, dtype=torch.bool),  # 1D tensor
            }
        )

        seq_dim, seq_dim_size = check_sequence_dim(data)

        assert seq_dim == 1
        assert seq_dim_size == 64

    def test_inconsistent_sequence_dim_raises_error(self):
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "attention_mask": torch.ones(
                    4, 128, dtype=torch.bool
                ),  # Different seq len
            }
        )

        with pytest.raises(AssertionError, match="Dim 1 must be the sequence dim"):
            check_sequence_dim(data)

    def test_ignores_1d_tensors(self):
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "sample_mask": torch.ones(4, dtype=torch.bool),  # 1D tensor
                "labels": torch.randint(0, 2, (128,)),  # Different 1D tensor size
            }
        )

        seq_dim, seq_dim_size = check_sequence_dim(data)

        assert seq_dim == 1
        assert seq_dim_size == 64


# =====================
# Test ProcessedInputs properties
# =====================
@pytest.mark.automodel
class TestProcessedInputsProperties:
    def test_has_context_parallel_false(self, processed_inputs_no_flash):
        assert processed_inputs_no_flash.has_context_parallel is False

    def test_has_context_parallel_true(self):
        input_ids = torch.randint(0, 1000, (2, 128))
        position_ids = torch.arange(128).repeat(2, 1)
        seq_index = torch.arange(128).repeat(1, 1)

        processed = ProcessedInputs(
            input_ids=input_ids,
            seq_len=128,
            attention_mask=torch.ones(2, 128, dtype=torch.bool),
            position_ids=position_ids,
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[input_ids, position_ids, seq_index],
            seq_index=seq_index,
        )

        assert processed.has_context_parallel is True

    def test_has_flash_attention_false(self, processed_inputs_no_flash):
        assert processed_inputs_no_flash.has_flash_attention is False

    def test_has_flash_attention_true(self, processed_inputs_with_flash):
        assert processed_inputs_with_flash.has_flash_attention is True

    def test_is_multimodal_false(self, processed_inputs_no_flash):
        assert processed_inputs_no_flash.is_multimodal is False

    def test_is_multimodal_true(self, processed_inputs_multimodal):
        assert processed_inputs_multimodal.is_multimodal is True


# =====================
# Test forward_with_post_processing_fn
# =====================
@pytest.mark.automodel
class TestForwardWithPostProcessingFn:
    def test_forward_with_loss_post_processor(
        self,
        mock_model,
        mock_loss_fn,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        # Create processed inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
        )

        # Create data dict
        data_dict = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": torch.full((batch_size,), seq_len),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            }
        )

        # Create processed microbatch
        processed_mb = ProcessedMicrobatch(
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
        )

        # Create iterator
        data_iterator = iter([processed_mb])

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call forward_with_post_processing_fn
        result, metrics, returned_mb = forward_with_post_processing_fn(
            model=mock_model,
            cfg=base_cfg,
            post_processing_fn=loss_post_processor,
            data_iterator=data_iterator,
            global_valid_seqs=torch.tensor(batch_size),
            global_valid_toks=torch.tensor(batch_size * seq_len),
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify loss function was called
        mock_loss_fn.assert_called_once()

        # Verify returned microbatch is correct
        assert returned_mb is processed_mb

    def test_forward_with_score_post_processor(
        self,
        mock_model,
        base_cfg,
    ):
        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        # Setup mock model to return reward-like logits
        mock_model.return_value = MagicMock(logits=torch.randn(batch_size, 1))

        # Create processed inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
        )

        # Create data dict
        data_dict = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": torch.full((batch_size,), seq_len),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            }
        )

        # Create processed microbatch
        processed_mb = ProcessedMicrobatch(
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
        )

        # Create iterator
        data_iterator = iter([processed_mb])

        # Create score post-processor
        score_post_processor = ScorePostProcessor(cfg=base_cfg)

        # Call forward_with_post_processing_fn
        result, metrics, returned_mb = forward_with_post_processing_fn(
            model=mock_model,
            cfg=base_cfg,
            post_processing_fn=score_post_processor,
            data_iterator=data_iterator,
            is_reward_model=True,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify scores are in metrics
        assert "scores" in metrics

        # Verify result shape
        assert result.shape == (batch_size,)


# =====================
# Test automodel_forward_backward
# =====================
@pytest.mark.automodel
class TestAutomodelForwardBackward:
    def test_forward_backward_single_microbatch(
        self,
        mock_model,
        mock_loss_fn,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        # Create processed inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
        )

        # Create data dict
        data_dict = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": torch.full((batch_size,), seq_len),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            }
        )

        # Create processed microbatch
        processed_mb = ProcessedMicrobatch(
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
        )

        # Create iterator
        data_iterator = iter([processed_mb])

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call automodel_forward_backward in forward_only mode
        results = automodel_forward_backward(
            model=mock_model,
            cfg=base_cfg,
            data_iterator=data_iterator,
            post_processing_fn=loss_post_processor,
            forward_only=True,
            global_valid_seqs=torch.tensor(batch_size),
            global_valid_toks=torch.tensor(batch_size * seq_len),
        )

        # Verify results
        assert len(results) == 1
        result, metrics = results[0]

        # Verify loss function was called
        mock_loss_fn.assert_called_once()

    def test_forward_backward_multiple_microbatches(
        self,
        mock_model,
        mock_loss_fn,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        batch_size = 4
        seq_len = 64
        vocab_size = 32000
        num_microbatches = 3

        # Create multiple processed microbatches
        processed_mbs = []
        for _ in range(num_microbatches):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            processed_inputs = ProcessedInputs(
                input_ids=input_ids,
                seq_len=seq_len,
                attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
                position_ids=torch.arange(seq_len).repeat(batch_size, 1),
                flash_attn_kwargs={},
                vlm_kwargs={},
                cp_buffers=[],
                seq_index=None,
            )

            data_dict = BatchedDataDict(
                {
                    "input_ids": input_ids,
                    "input_lengths": torch.full((batch_size,), seq_len),
                    "sample_mask": torch.ones(batch_size, dtype=torch.bool),
                }
            )

            processed_mbs.append(
                ProcessedMicrobatch(
                    data_dict=data_dict,
                    processed_inputs=processed_inputs,
                    original_batch_size=batch_size,
                    original_seq_len=seq_len,
                )
            )

        # Create iterator
        data_iterator = iter(processed_mbs)

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call automodel_forward_backward in forward_only mode
        results = automodel_forward_backward(
            model=mock_model,
            cfg=base_cfg,
            data_iterator=data_iterator,
            post_processing_fn=loss_post_processor,
            forward_only=True,
            global_valid_seqs=torch.tensor(batch_size * num_microbatches),
            global_valid_toks=torch.tensor(batch_size * seq_len * num_microbatches),
        )

        # Verify results
        assert len(results) == num_microbatches

        # Verify model was called num_microbatches times
        assert mock_model.call_count == num_microbatches

        # Verify loss function was called num_microbatches times
        assert mock_loss_fn.call_count == num_microbatches
