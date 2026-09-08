# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossInputType
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
    PreparedModelForward,
    ScorePostProcessor,
    TopkLogitsPostProcessor,
    apply_temperature_scaling,
    automodel_forward_backward,
    extract_logits,
    forward_with_post_processing_fn,
    model_forward,
    prepare_model_forward,
)


def _prepare_cp1(
    model: Any,
    processed_inputs: ProcessedInputs,
    *,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
) -> PreparedModelForward:
    return prepare_model_forward(
        model,
        processed_inputs,
        device_mesh=None,
        cp_size=1,
        padding_token_id=0,
        is_reward_model=is_reward_model,
        allow_flash_attn_args=allow_flash_attn_args,
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
    loss_fn.input_type = LossInputType.LOGIT
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
        "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": None},
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
    )


# =====================
# Test model_forward
# =====================
@pytest.mark.automodel
class TestModelForward:
    def test_basic_forward(self, mock_model, processed_inputs_no_flash):
        prepared = _prepare_cp1(mock_model, processed_inputs_no_flash)
        model_forward(mock_model, prepared.model_batch)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "input_ids" in call_kwargs
        assert "attention_mask" in call_kwargs
        assert "position_ids" in call_kwargs
        assert call_kwargs["use_cache"] is False

    def test_forward_with_flash_attention(
        self, mock_model, processed_inputs_with_flash
    ):
        prepared = _prepare_cp1(mock_model, processed_inputs_with_flash)
        model_forward(mock_model, prepared.model_batch)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" in call_kwargs

    def test_forward_with_multimodal(self, mock_model, processed_inputs_multimodal):
        prepared = _prepare_cp1(mock_model, processed_inputs_multimodal)
        model_forward(mock_model, prepared.model_batch)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "pixel_values" in call_kwargs
        # Flash attention should be removed for multimodal
        assert "flash_attn_kwargs" not in call_kwargs

    def test_forward_filters_unsupported_multimodal_metadata(
        self, processed_inputs_multimodal
    ):
        class ExplicitMultimodalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pixel_values = None

            def forward(
                self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
                pixel_values=None,
            ):
                self.pixel_values = pixel_values
                return MagicMock(logits=torch.randn(2, 64, 1000))

        model = ExplicitMultimodalModel()
        processed_inputs_multimodal.vlm_kwargs.update(
            {
                "imgs_sizes": torch.tensor([[224, 224]]),
                "num_frames": torch.tensor([1]),
            }
        )

        prepared = _prepare_cp1(model, processed_inputs_multimodal)
        model_forward(model, prepared.model_batch)

        assert (
            model.pixel_values is processed_inputs_multimodal.vlm_kwargs["pixel_values"]
        )

    def test_forward_rejects_mixed_resolution_without_imgs_sizes_support(
        self, processed_inputs_multimodal
    ):
        class ExplicitMultimodalModel(torch.nn.Module):
            def forward(
                self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
                pixel_values=None,
            ):
                return MagicMock(logits=torch.randn(2, 64, 1000))

        model = ExplicitMultimodalModel()
        processed_inputs_multimodal.vlm_kwargs.update(
            {
                "imgs_sizes": torch.tensor([[224, 320], [256, 288]]),
                "num_frames": torch.ones(2, dtype=torch.long),
            }
        )

        # The mixed-resolution guard lives in ``filter_multimodal_kwargs_for_model``,
        # which ``prepare_model_forward`` invokes while building the model batch.
        with pytest.raises(ValueError, match="mixed-resolution"):
            _prepare_cp1(model, processed_inputs_multimodal)

    def test_forward_allows_uniform_resolution_without_imgs_sizes_support(
        self, processed_inputs_multimodal
    ):
        class ExplicitMultimodalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pixel_values = None

            def forward(
                self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
                pixel_values=None,
            ):
                self.pixel_values = pixel_values
                return MagicMock(logits=torch.randn(2, 64, 1000))

        model = ExplicitMultimodalModel()
        processed_inputs_multimodal.vlm_kwargs.update(
            {
                "imgs_sizes": torch.tensor([[224, 224], [224, 224]]),
                "num_frames": torch.ones(2, dtype=torch.long),
            }
        )

        # Uniform sizes (the shipped fixed-tile case): no raise, and imgs_sizes
        # is filtered out for a model that cannot consume it.
        prepared = _prepare_cp1(model, processed_inputs_multimodal)
        model_forward(model, prepared.model_batch)

        assert (
            model.pixel_values is processed_inputs_multimodal.vlm_kwargs["pixel_values"]
        )

    def test_forward_preserves_dynamic_resolution_omni_inputs(
        self, processed_inputs_multimodal
    ):
        class DynamicResolutionOmniLikeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.forward_kwargs = {}

            def forward(
                self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
                pixel_values=None,
                imgs_sizes=None,
            ):
                self.forward_kwargs = {
                    "pixel_values": pixel_values,
                    "imgs_sizes": imgs_sizes,
                }
                return MagicMock(logits=torch.randn(2, 64, 1000))

        model = DynamicResolutionOmniLikeModel()
        padded_images = torch.randn(2, 3, 256, 320)
        image_sizes = torch.tensor([[224, 320], [256, 288]])
        processed_inputs_multimodal.vlm_kwargs.update(
            {
                "pixel_values": padded_images,
                "imgs_sizes": image_sizes,
                "num_frames": torch.ones(2, dtype=torch.long),
            }
        )

        prepared = _prepare_cp1(model, processed_inputs_multimodal)
        model_forward(model, prepared.model_batch)

        assert model.forward_kwargs["pixel_values"] is padded_images
        assert model.forward_kwargs["imgs_sizes"] is image_sizes

    def test_forward_preserves_multimodal_metadata_for_kwargs_model(
        self, processed_inputs_multimodal
    ):
        class KwargsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.forward_kwargs = {}

            def forward(self, **kwargs):
                self.forward_kwargs = kwargs
                return MagicMock(logits=torch.randn(2, 64, 1000))

        model = KwargsModel()
        processed_inputs_multimodal.vlm_kwargs["num_frames"] = torch.tensor([1])

        prepared = _prepare_cp1(model, processed_inputs_multimodal)
        model_forward(model, prepared.model_batch)

        assert "num_frames" in model.forward_kwargs

    def test_forward_reward_model_removes_flash_attn(
        self, mock_model, processed_inputs_with_flash
    ):
        prepared = _prepare_cp1(
            mock_model, processed_inputs_with_flash, is_reward_model=True
        )
        model_forward(mock_model, prepared.model_batch)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        # Flash attention should be removed for reward models
        assert "flash_attn_kwargs" not in call_kwargs

    def test_forward_disallow_flash_attn_args(
        self, mock_model, processed_inputs_with_flash
    ):
        prepared = _prepare_cp1(
            mock_model,
            processed_inputs_with_flash,
            allow_flash_attn_args=False,
        )
        model_forward(mock_model, prepared.model_batch)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" not in call_kwargs

    def test_forward_injects_mm_token_type_ids_for_gemma4(
        self, mock_model, processed_inputs_no_flash
    ):
        # Gemma 4 needs mm_token_type_ids even for text-only inputs.
        mock_model.config.model_type = "gemma4"

        prepared = _prepare_cp1(mock_model, processed_inputs_no_flash)
        model_forward(mock_model, prepared.model_batch)

        call_kwargs = mock_model.call_args[1]
        assert "mm_token_type_ids" in call_kwargs
        assert torch.equal(
            call_kwargs["mm_token_type_ids"],
            torch.zeros_like(processed_inputs_no_flash.input_ids),
        )

    def test_forward_omits_mm_token_type_ids_for_non_gemma4(
        self, mock_model, processed_inputs_no_flash
    ):
        mock_model.config.model_type = "llama"

        prepared = _prepare_cp1(mock_model, processed_inputs_no_flash)
        model_forward(mock_model, prepared.model_batch)

        call_kwargs = mock_model.call_args[1]
        assert "mm_token_type_ids" not in call_kwargs


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
    """Tests for apply_temperature_scaling function."""

    def test_temperature_scaling_sampling_params_is_none(self):
        """Test that logits are unchanged when sampling_params is None."""
        logits = torch.randn(4, 64, 32000)
        original_logits = logits.clone()

        result = apply_temperature_scaling(logits, None)

        assert torch.equal(result, original_logits)

    def test_temperature_scaling_with_temperature_one(self):
        """Test that temperature=1.0 leaves logits unchanged."""
        logits = torch.randn(4, 64, 32000)
        original_logits = logits.clone()
        sampling_params = TrainingSamplingParams(temperature=1.0)

        result = apply_temperature_scaling(logits, sampling_params)

        assert torch.equal(result, original_logits)

    def test_temperature_scaling_with_temperature_two(self):
        """Test that logits are divided by the configured temperature=2.0."""
        logits = torch.randn(4, 64, 32000)
        original_logits = logits.clone()
        sampling_params = TrainingSamplingParams(temperature=2.0)

        result = apply_temperature_scaling(logits, sampling_params)

        # Should be divided by temperature
        expected = original_logits / 2.0
        assert torch.allclose(result, expected)


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
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        data_dict = BatchedDataDict(
            {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            }
        )
        global_valid_seqs = torch.tensor(8)
        global_valid_toks = torch.tensor(512)

        processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs_no_flash,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            cp_sharder=None,
        )

        # Verify loss function was called
        mock_loss_fn.assert_called_once()
        call_kwargs = mock_loss_fn.call_args[1]
        assert torch.is_tensor(call_kwargs["logits"])
        assert call_kwargs["global_valid_seqs"] == global_valid_seqs
        assert call_kwargs["global_valid_toks"] == global_valid_toks

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
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
            enable_seq_packing=True,
        )

        batch_size = 1
        seq_len = 128
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        data_dict = BatchedDataDict(
            {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            }
        )
        global_valid_seqs = torch.tensor(4)
        global_valid_toks = torch.tensor(128)

        processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs_with_flash,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            cp_sharder=None,
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
            cp_mesh=mock_cp_mesh,
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
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_lengths = torch.full((batch_size,), seq_len)
        data_dict = BatchedDataDict({"input_lengths": input_lengths})

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
        )

        result = processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            cp_sharder=None,
        )

        assert result.shape == (batch_size, seq_len)

    def test_logprobs_with_chunking(
        self, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        cfg_with_chunk = {**base_cfg, "logprob_chunk_size": 16}
        processor = LogprobsPostProcessor(
            cfg=cfg_with_chunk,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_lengths = torch.full((batch_size,), seq_len)
        data_dict = BatchedDataDict({"input_lengths": input_lengths})

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
        )

        result = processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            cp_sharder=None,
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
            tp_mesh=mock_tp_mesh,
            k=k,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.full((batch_size,), seq_len)
        data_dict = BatchedDataDict({"input_lengths": input_lengths})

        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
        )

        vals, idx = processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            cp_sharder=None,
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

    def test_ignores_packed_multimodal_wire_fields(self):
        """A packed multimodal field has patch/image count on dim 1, not
        seqlen. Without the skip every VLM step through the data plane trips
        this assert."""
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                # [B, max_patches, C, H, W] — dim 1 is 7 patches, not 64 tokens.
                "pixel_values": torch.randn(4, 7, 3, 2, 2),
                "image_grid_thw": torch.zeros(4, 2, 3, dtype=torch.long),
            }
        )

        seq_dim, seq_dim_size = check_sequence_dim(data)

        assert seq_dim == 1
        assert seq_dim_size == 64

    def test_per_token_multimodal_is_still_checked(self):
        """Only the *packed* registry is exempt. Per-token maps like
        ``mm_token_type_ids`` are ``[B, S]`` and a seqlen mismatch there is a
        real bug that must keep failing."""
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "mm_token_type_ids": torch.zeros(4, 32, dtype=torch.long),
            }
        )

        with pytest.raises(AssertionError, match="Dim 1 must be the sequence dim"):
            check_sequence_dim(data)


# =====================
# Test ProcessedInputs properties
# =====================
@pytest.mark.automodel
class TestProcessedInputsProperties:
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

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call forward_with_post_processing_fn
        _, _, returned_mb = forward_with_post_processing_fn(
            model=mock_model,
            prepared=_prepare_cp1(mock_model, processed_inputs),
            post_processing_fn=loss_post_processor,
            processed_mb=processed_mb,
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

        # Create score post-processor
        score_post_processor = ScorePostProcessor(cfg=base_cfg)

        # Call forward_with_post_processing_fn
        result, metrics, _ = forward_with_post_processing_fn(
            model=mock_model,
            prepared=_prepare_cp1(mock_model, processed_inputs, is_reward_model=True),
            post_processing_fn=score_post_processor,
            processed_mb=processed_mb,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify scores are in metrics
        assert "scores" in metrics

        # Verify result shape
        assert result.shape == (batch_size,)

    def test_forward_with_score_post_processor_rejects_cp(
        self,
        mock_model,
        base_cfg,
    ):
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 32000, (batch_size, seq_len))
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
        )
        processed_mb = ProcessedMicrobatch(
            data_dict=BatchedDataDict(
                {
                    "input_ids": input_ids,
                    "input_lengths": torch.full((batch_size,), seq_len),
                    "sample_mask": torch.ones(batch_size, dtype=torch.bool),
                }
            ),
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
        )
        prepared = PreparedModelForward(
            model_batch={"input_ids": input_ids},
            cp_size=2,
            cp_sharder=MagicMock(),
            model_context_factory=nullcontext,
        )

        with pytest.raises(
            NotImplementedError,
            match="ScorePostProcessor does not support context_parallel_size > 1",
        ):
            forward_with_post_processing_fn(
                model=mock_model,
                prepared=prepared,
                post_processing_fn=ScorePostProcessor(cfg=base_cfg),
                processed_mb=processed_mb,
            )

        mock_model.assert_not_called()


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

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call automodel_forward_backward in forward_only mode
        results = automodel_forward_backward(
            model=mock_model,
            data_iterator=iter([processed_mb]),
            post_processing_fn=loss_post_processor,
            device_mesh=mock_device_mesh,
            padding_token_id=0,
            autocast_context_factory=nullcontext,
            forward_only=True,
            global_valid_seqs=torch.tensor(batch_size),
            global_valid_toks=torch.tensor(batch_size * seq_len),
        )

        # Verify results
        assert len(results) == 1
        _result, _metrics = results[0]

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

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call automodel_forward_backward in forward_only mode
        results = automodel_forward_backward(
            model=mock_model,
            data_iterator=iter(processed_mbs),
            post_processing_fn=loss_post_processor,
            device_mesh=mock_device_mesh,
            padding_token_id=0,
            autocast_context_factory=nullcontext,
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

    @patch("nemo_rl.models.automodel.train.prepare_model_forward")
    def test_forward_backward_enters_prepared_and_autocast_contexts(
        self,
        mock_prepare_model_forward,
        mock_model,
        mock_loss_fn,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        """The loop enters the Automodel and worker precision contexts."""
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

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        context_calls = []

        class MockContext:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                context_calls.append(f"{self.name}_enter")
                return self

            def __exit__(self, *args):
                context_calls.append(f"{self.name}_exit")
                return False

        mock_prepare_model_forward.return_value = PreparedModelForward(
            model_batch={"input_ids": input_ids, "use_cache": False},
            cp_size=1,
            cp_sharder=None,
            model_context_factory=lambda: MockContext("model"),
        )

        results = automodel_forward_backward(
            model=mock_model,
            data_iterator=iter([processed_mb]),
            post_processing_fn=loss_post_processor,
            device_mesh=mock_device_mesh,
            padding_token_id=0,
            autocast_context_factory=lambda: MockContext("autocast"),
            forward_only=True,
            global_valid_seqs=torch.tensor(batch_size),
            global_valid_toks=torch.tensor(batch_size * seq_len),
        )

        assert context_calls == [
            "model_enter",
            "autocast_enter",
            "autocast_exit",
            "model_exit",
        ]
        assert len(results) == 1

    def test_forward_backward_with_on_microbatch_start(
        self,
        mock_model,
        mock_loss_fn,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        """Test automodel_forward_backward with on_microbatch_start callback."""
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

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Track callback calls
        callback_indices = []

        def on_microbatch_start(mb_idx):
            callback_indices.append(mb_idx)

        # Call automodel_forward_backward with on_microbatch_start
        results = automodel_forward_backward(
            model=mock_model,
            data_iterator=iter(processed_mbs),
            post_processing_fn=loss_post_processor,
            device_mesh=mock_device_mesh,
            padding_token_id=0,
            autocast_context_factory=nullcontext,
            forward_only=True,
            global_valid_seqs=torch.tensor(batch_size * num_microbatches),
            global_valid_toks=torch.tensor(batch_size * seq_len * num_microbatches),
            on_microbatch_start=on_microbatch_start,
        )

        # Verify callback was called for each microbatch
        assert callback_indices == [0, 1, 2]
        assert len(results) == num_microbatches

    def test_forward_backward_with_dummy_batches(
        self,
        mock_model,
        mock_loss_fn,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        """Test automodel_forward_backward with dummy batches (num_valid_microbatches)."""
        batch_size = 4
        seq_len = 64
        vocab_size = 32000
        total_microbatches = 3
        num_valid_microbatches = 2  # Only first 2 are valid

        # Create processed microbatches
        processed_mbs = []
        for _ in range(total_microbatches):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            processed_inputs = ProcessedInputs(
                input_ids=input_ids,
                seq_len=seq_len,
                attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
                position_ids=torch.arange(seq_len).repeat(batch_size, 1),
                flash_attn_kwargs={},
                vlm_kwargs={},
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

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call automodel_forward_backward with num_valid_microbatches
        results = automodel_forward_backward(
            model=mock_model,
            data_iterator=iter(processed_mbs),
            post_processing_fn=loss_post_processor,
            device_mesh=mock_device_mesh,
            padding_token_id=0,
            autocast_context_factory=nullcontext,
            forward_only=True,
            global_valid_seqs=torch.tensor(batch_size * num_valid_microbatches),
            global_valid_toks=torch.tensor(
                batch_size * seq_len * num_valid_microbatches
            ),
            num_valid_microbatches=num_valid_microbatches,
        )

        # Verify all microbatches processed
        assert len(results) == total_microbatches

        # Third batch (index 2) is dummy - result should be zeroed
        dummy_result, dummy_metrics = results[2]
        assert dummy_result.item() == 0.0  # Dummy batch loss is zeroed


# =====================
# Test forward_with_post_processing_fn (additional coverage)
# =====================
@pytest.mark.automodel
class TestForwardWithPostProcessingFnAdditional:
    def test_forward_with_logprobs_post_processor(
        self,
        mock_model,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        """Test forward_with_post_processing_fn with LogprobsPostProcessor."""
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

        # Create logprobs post-processor
        logprobs_post_processor = LogprobsPostProcessor(
            cfg=base_cfg,
        )

        # Call forward_with_post_processing_fn
        result, metrics, returned_mb = forward_with_post_processing_fn(
            model=mock_model,
            prepared=_prepare_cp1(mock_model, processed_inputs),
            post_processing_fn=logprobs_post_processor,
            processed_mb=processed_mb,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify logprobs are in metrics
        assert "logprobs" in metrics

        # Verify result shape
        assert result.shape == (batch_size, seq_len)

        # Verify returned microbatch is correct
        assert returned_mb is processed_mb

    def test_forward_with_topk_post_processor(
        self,
        mock_model,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        """Test forward_with_post_processing_fn with TopkLogitsPostProcessor."""
        batch_size = 4
        seq_len = 64
        vocab_size = 32000
        k = 10

        # Create processed inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
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

        # Create topk post-processor
        topk_post_processor = TopkLogitsPostProcessor(
            cfg=base_cfg,
            tp_mesh=mock_tp_mesh,
            k=k,
        )

        # Call forward_with_post_processing_fn
        result, metrics, _ = forward_with_post_processing_fn(
            model=mock_model,
            prepared=_prepare_cp1(mock_model, processed_inputs),
            post_processing_fn=topk_post_processor,
            processed_mb=processed_mb,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify topk values and indices are in metrics
        assert "topk_logits" in metrics
        assert "topk_indices" in metrics

        # Verify result is tuple of (vals, idx)
        vals, idx = result
        assert vals.shape == (batch_size, seq_len, k)
        assert idx.shape == (batch_size, seq_len, k)

    def test_forward_with_unknown_post_processor_raises_error(
        self,
        mock_model,
        base_cfg,
    ):
        """Test forward_with_post_processing_fn raises TypeError for unknown post-processor."""
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

        # Create unknown post-processor (not a valid type)
        class UnknownPostProcessor:
            pass

        unknown_post_processor = UnknownPostProcessor()

        # Call forward_with_post_processing_fn and expect TypeError
        with pytest.raises(TypeError, match="Unknown post-processing function type"):
            forward_with_post_processing_fn(
                model=mock_model,
                prepared=_prepare_cp1(mock_model, processed_inputs),
                post_processing_fn=unknown_post_processor,
                processed_mb=processed_mb,
            )

    def test_forward_with_processed_mb_directly(
        self,
        mock_model,
        mock_loss_fn,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        """Test forward_with_post_processing_fn with processed_mb provided directly."""
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

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Call forward_with_post_processing_fn with processed_mb directly (no iterator)
        _, _, returned_mb = forward_with_post_processing_fn(
            model=mock_model,
            prepared=_prepare_cp1(mock_model, processed_inputs),
            post_processing_fn=loss_post_processor,
            processed_mb=processed_mb,  # Directly provided
            global_valid_seqs=torch.tensor(batch_size),
            global_valid_toks=torch.tensor(batch_size * seq_len),
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify returned microbatch is correct
        assert returned_mb is processed_mb


# =====================
# Test LogprobsPostProcessor with sequence packing
# =====================
@pytest.mark.automodel
class TestLogprobsPostProcessorSeqPacking:
    def test_logprobs_with_sequence_packing(
        self, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        """Test LogprobsPostProcessor with sequence packing enabled."""
        processor = LogprobsPostProcessor(
            cfg=base_cfg,
            enable_seq_packing=True,
        )

        original_batch_size = 4
        original_seq_len = 32
        packed_seq_len = 128  # All 4 sequences packed
        vocab_size = 32000

        logits = torch.randn(1, packed_seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (1, packed_seq_len))
        input_lengths = torch.tensor([32, 32, 32, 32])
        data_dict = BatchedDataDict({"input_lengths": input_lengths})

        @dataclass
        class MockFlashAttnKwargs:
            cu_seqlens_q: torch.Tensor

        flash_kwargs = MockFlashAttnKwargs(
            cu_seqlens_q=torch.tensor([0, 32, 64, 96, 128])
        )

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=packed_seq_len,
            attention_mask=None,
            position_ids=torch.arange(packed_seq_len).unsqueeze(0),
            flash_attn_kwargs=flash_kwargs,
            vlm_kwargs={},
        )

        result = processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=original_batch_size,
            original_seq_len=original_seq_len,
            cp_sharder=None,
        )

        # Result should be unpacked to original shape
        assert result.shape == (original_batch_size, original_seq_len)

    def test_logprobs_masking_without_sequence_packing(
        self, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        """Test LogprobsPostProcessor applies mask when sequence packing is disabled."""
        processor = LogprobsPostProcessor(
            cfg=base_cfg,
        )

        batch_size = 4
        seq_len = 64
        vocab_size = 32000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_lengths = torch.tensor([32, 48, 64, 16])
        data_dict = BatchedDataDict({"input_lengths": input_lengths})

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
        )

        result = processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            cp_sharder=None,
        )

        # Verify result shape
        assert result.shape == (batch_size, seq_len)

        # Verify masking - positions beyond input_lengths should be zero
        for i, length in enumerate(input_lengths):
            # Positions beyond length should be zero
            assert torch.all(result[i, length:] == 0)


# =====================
# Test TopkLogitsPostProcessor with sequence packing
# =====================
@pytest.mark.automodel
class TestTopkLogitsPostProcessorSeqPacking:
    def test_topk_with_sequence_packing(
        self, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        """Test TopkLogitsPostProcessor with sequence packing enabled."""
        k = 10
        processor = TopkLogitsPostProcessor(
            cfg=base_cfg,
            tp_mesh=mock_tp_mesh,
            k=k,
            enable_seq_packing=True,
        )

        original_batch_size = 4
        original_seq_len = 32
        packed_seq_len = 128  # All 4 sequences packed
        vocab_size = 32000

        logits = torch.randn(1, packed_seq_len, vocab_size)
        input_lengths = torch.tensor([32, 32, 32, 32])
        data_dict = BatchedDataDict({"input_lengths": input_lengths})

        @dataclass
        class MockFlashAttnKwargs:
            cu_seqlens_q: torch.Tensor

        flash_kwargs = MockFlashAttnKwargs(
            cu_seqlens_q=torch.tensor([0, 32, 64, 96, 128])
        )

        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, vocab_size, (1, packed_seq_len)),
            seq_len=packed_seq_len,
            attention_mask=None,
            position_ids=torch.arange(packed_seq_len).unsqueeze(0),
            flash_attn_kwargs=flash_kwargs,
            vlm_kwargs={},
        )

        vals, idx = processor(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=original_batch_size,
            original_seq_len=original_seq_len,
            cp_sharder=None,
        )

        # Result should be unpacked to original shape
        assert vals.shape == (original_batch_size, original_seq_len, k)
        assert idx.shape == (original_batch_size, original_seq_len, k)


# =====================
# Test automodel_forward_backward with backward pass
# =====================
@pytest.mark.automodel
class TestAutomodelForwardBackwardWithGradients:
    def test_forward_backward_computes_gradients(
        self,
        base_cfg,
        mock_device_mesh,
        mock_cp_mesh,
        mock_tp_mesh,
    ):
        """Test automodel_forward_backward with forward_only=False computes gradients."""
        batch_size = 2
        seq_len = 16
        vocab_size = 100
        hidden_size = 32

        # Create a simple model with trainable parameters
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.proj = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids, **kwargs):
                x = self.embed(input_ids)
                logits = self.proj(x)
                return MagicMock(logits=logits)

        model = SimpleModel()
        model.train()

        # Create processed inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=seq_len,
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).repeat(batch_size, 1),
            flash_attn_kwargs={},
            vlm_kwargs={},
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

        # Create loss function that returns requires_grad tensor
        def loss_fn(logits, data, global_valid_seqs, global_valid_toks):
            loss = logits.mean()
            return loss, {"loss": loss.item()}

        loss_fn.input_type = LossInputType.LOGIT

        # Create loss post-processor
        loss_post_processor = LossPostProcessor(
            loss_fn=loss_fn,
            cfg=base_cfg,
            cp_mesh=mock_cp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # Verify no gradients initially
        assert model.proj.weight.grad is None

        # Call automodel_forward_backward with forward_only=False
        results = automodel_forward_backward(
            model=model,
            data_iterator=iter([processed_mb]),
            post_processing_fn=loss_post_processor,
            device_mesh=mock_device_mesh,
            padding_token_id=0,
            autocast_context_factory=nullcontext,
            forward_only=False,  # Enable backward pass
            global_valid_seqs=torch.tensor(batch_size),
            global_valid_toks=torch.tensor(batch_size * seq_len),
            dp_size=1,
            cp_size=1,
        )

        # Verify gradients were computed
        assert model.proj.weight.grad is not None
        assert len(results) == 1

    @pytest.mark.parametrize(
        ("input_type", "expected_scale"),
        [
            (LossInputType.LOGIT, 3.0),
            (LossInputType.LOGPROB, 3.0),
            (LossInputType.DISTILLATION, 3.0),
            (LossInputType.DISTILLATION_CROSS_TOKENIZER, 6.0),
        ],
    )
    @patch("nemo_rl.models.automodel.train.prepare_loss_input")
    @patch("nemo_rl.models.automodel.train.prepare_model_forward")
    def test_gradient_scale_follows_cp_fanout_contract(
        self,
        mock_prepare_model_forward: MagicMock,
        mock_prepare_loss_input: MagicMock,
        input_type: LossInputType,
        expected_scale: float,
        base_cfg: dict[str, Any],
        mock_device_mesh: MagicMock,
        mock_cp_mesh: MagicMock,
    ) -> None:
        """Backward applies DP and CP scaling exactly once for each CP contract."""

        def fake_prepare_loss_input(
            logits: torch.Tensor,
            data_dict: BatchedDataDict[Any],
            _loss_fn: Any,
            **_kwargs: Any,
        ) -> tuple[dict[str, torch.Tensor], BatchedDataDict[Any]]:
            return {"logits": logits}, data_dict

        mock_prepare_loss_input.side_effect = fake_prepare_loss_input

        class ScaleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> MagicMock:
                logits = self.weight * torch.ones_like(
                    input_ids, dtype=self.weight.dtype
                )
                return MagicMock(logits=logits)

        def run(*, cp_size: int, dp_size: int) -> torch.Tensor:
            model = ScaleModel()
            loss_fn = MagicMock(input_type=input_type)

            def compute_loss(
                *,
                logits: torch.Tensor,
                data: BatchedDataDict[Any],
                global_valid_seqs: torch.Tensor,
                global_valid_toks: torch.Tensor,
            ) -> tuple[torch.Tensor, dict[str, float]]:
                del data, global_valid_seqs, global_valid_toks
                loss = logits.mean()
                return loss, {"loss": loss.item()}

            loss_fn.side_effect = compute_loss
            loss_post_processor = LossPostProcessor(
                loss_fn=loss_fn,
                cfg=base_cfg,
                cp_mesh=mock_cp_mesh,
                cp_size=cp_size,
                dp_size=dp_size,
            )

            input_ids = torch.zeros((1, 4), dtype=torch.long)
            processed_inputs = ProcessedInputs(
                input_ids=input_ids,
                seq_len=4,
                attention_mask=torch.ones((1, 4), dtype=torch.bool),
                position_ids=torch.arange(4).unsqueeze(0),
                flash_attn_kwargs={},
                vlm_kwargs={},
            )
            processed_mb = ProcessedMicrobatch(
                data_dict=BatchedDataDict(
                    {
                        "input_ids": input_ids,
                        "input_lengths": torch.tensor([4]),
                        "sample_mask": torch.ones(1, dtype=torch.bool),
                    }
                ),
                processed_inputs=processed_inputs,
                original_batch_size=1,
                original_seq_len=4,
            )

            # Keep model preparation at CP1 so this test isolates the backward
            # scale from sharder collectives. The LossPostProcessor and training
            # loop still use the requested CP/DP sizes and execute real autograd.
            mock_prepare_model_forward.return_value = PreparedModelForward(
                model_batch={"input_ids": input_ids},
                cp_size=1,
                cp_sharder=None,
                model_context_factory=nullcontext,
            )

            automodel_forward_backward(
                model=model,
                data_iterator=iter([processed_mb]),
                post_processing_fn=loss_post_processor,
                device_mesh=mock_device_mesh,
                padding_token_id=0,
                autocast_context_factory=nullcontext,
                forward_only=False,
                global_valid_seqs=torch.tensor(1),
                global_valid_toks=torch.tensor(4),
                dp_size=dp_size,
                cp_size=cp_size,
            )

            assert model.weight.grad is not None
            return model.weight.grad.clone()

        baseline_grad = run(cp_size=1, dp_size=1)
        scaled_grad = run(cp_size=2, dp_size=3)

        torch.testing.assert_close(scaled_grad, baseline_grad * expected_scale)


# =====================
# Test aggregate_training_statistics
# =====================
@pytest.mark.automodel
class TestAggregateTrainingStatistics:
    """Tests for aggregate_training_statistics function."""

    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA A100")
    def test_basic_aggregation(self, mock_device_name, mock_get_rank, mock_all_reduce):
        """Test basic statistics aggregation."""
        from nemo_rl.models.automodel.train import (
            aggregate_training_statistics,
        )

        losses = [0.5, 0.4, 0.3]
        all_mb_metrics = [
            {"loss": 0.5, "num_valid_samples": 4},
            {"loss": 0.4, "num_valid_samples": 4},
            {"loss": 0.3, "num_valid_samples": 4},
        ]
        grad_norm = torch.tensor([1.5])
        mock_dp_group = MagicMock()
        dtype = torch.bfloat16

        metrics = aggregate_training_statistics(
            losses=losses,
            all_mb_metrics=all_mb_metrics,
            grad_norm=grad_norm,
            dp_group=mock_dp_group,
            dtype=dtype,
        )

        # Verify all_reduce was called
        mock_all_reduce.assert_called_once()

        # Verify metrics structure
        assert "global_loss" in metrics
        assert "grad_norm" in metrics
        assert "rank" in metrics
        assert "gpu_name" in metrics
        assert "model_dtype" in metrics
        assert "all_mb_metrics" in metrics

        # Verify values
        assert metrics["rank"] == 0
        assert metrics["gpu_name"] == "NVIDIA A100"
        assert metrics["model_dtype"] == torch.bfloat16
        assert torch.equal(metrics["grad_norm"], grad_norm)

        # Verify metrics aggregation
        assert metrics["all_mb_metrics"]["loss"] == [0.5, 0.4, 0.3]
        assert metrics["all_mb_metrics"]["num_valid_samples"] == [4, 4, 4]

    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_rank", return_value=2)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA H100")
    def test_with_none_grad_norm(
        self, mock_device_name, mock_get_rank, mock_all_reduce
    ):
        """Test aggregation with None grad_norm (eval mode)."""
        from nemo_rl.models.automodel.train import (
            aggregate_training_statistics,
        )

        losses = [0.5]
        all_mb_metrics = [{"loss": 0.5}]
        grad_norm = None
        mock_dp_group = MagicMock()
        dtype = torch.float16

        metrics = aggregate_training_statistics(
            losses=losses,
            all_mb_metrics=all_mb_metrics,
            grad_norm=grad_norm,
            dp_group=mock_dp_group,
            dtype=dtype,
        )

        assert metrics["grad_norm"] is None
        assert metrics["rank"] == 2
        assert metrics["model_dtype"] == torch.float16

    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA A100")
    def test_empty_metrics(self, mock_device_name, mock_get_rank, mock_all_reduce):
        """Test aggregation with empty metrics list."""
        from nemo_rl.models.automodel.train import (
            aggregate_training_statistics,
        )

        losses = []
        all_mb_metrics = []
        grad_norm = torch.tensor([0.0])
        mock_dp_group = MagicMock()
        dtype = torch.bfloat16

        metrics = aggregate_training_statistics(
            losses=losses,
            all_mb_metrics=all_mb_metrics,
            grad_norm=grad_norm,
            dp_group=mock_dp_group,
            dtype=dtype,
        )

        assert metrics["all_mb_metrics"] == {}
        assert metrics["global_loss"].numel() == 0

    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA A100")
    def test_global_loss_on_cpu(self, mock_device_name, mock_get_rank, mock_all_reduce):
        """Test that global_loss is moved to CPU."""
        from nemo_rl.models.automodel.train import (
            aggregate_training_statistics,
        )

        losses = [0.5, 0.4]
        all_mb_metrics = [{"loss": 0.5}, {"loss": 0.4}]
        grad_norm = torch.tensor([1.0])
        mock_dp_group = MagicMock()
        dtype = torch.bfloat16

        metrics = aggregate_training_statistics(
            losses=losses,
            all_mb_metrics=all_mb_metrics,
            grad_norm=grad_norm,
            dp_group=mock_dp_group,
            dtype=dtype,
        )

        # Verify global_loss is on CPU
        assert metrics["global_loss"].device == torch.device("cpu")
