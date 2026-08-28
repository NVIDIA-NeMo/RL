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
Unit tests for Megatron training utilities.

This module tests the training functions in nemo_rl.models.megatron.train,
focusing on:
- Model forward pass
- Forward with post-processing
- Loss/logprobs/topk post-processors
"""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossInputType

pytestmark = pytest.mark.mcore


def _run_direct_model_loss() -> tuple[torch.Tensor, dict[str, Any]]:
    from nemo_rl.algorithms.loss import NLLLossFn
    from nemo_rl.models.megatron import train as megatron_train

    processor = megatron_train.LossPostProcessor(
        loss_fn=NLLLossFn(),
        cfg={"sequence_packing": {"enabled": True}},
        num_microbatches=4,
    )
    data = MagicMock()
    data.__contains__.side_effect = lambda key: key == "sample_mask"
    data.__getitem__.side_effect = lambda key: torch.tensor([1.0])
    loss_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    with patch.object(
        megatron_train, "get_context_parallel_world_size", return_value=2
    ):
        wrapped = processor(
            data_dict=data,
            global_valid_toks=torch.tensor(6.0),
            prepacked_loss_mask=loss_mask,
        )

    loss, metrics = wrapped(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    return loss, metrics


class TestModelForward:
    """Tests for model_forward function."""

    def test_model_forward_basic(self):
        """Test basic model_forward without multimodal data."""
        from nemo_rl.models.megatron.train import model_forward

        # Setup mocks
        mock_model = MagicMock()
        mock_output = torch.randn(2, 10, 100)
        mock_model.return_value = mock_output

        mock_data_dict = MagicMock()
        mock_data_dict.get_multimodal_dict.return_value = {}

        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
        attention_mask = torch.ones(2, 3)

        result = model_forward(
            model=mock_model,
            data_dict=mock_data_dict,
            input_ids_cp_sharded=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        assert torch.equal(result, mock_output)
        mock_model.assert_called_once()

    def test_model_forward_with_straggler_timer(self):
        """Test model_forward uses straggler_timer context manager when provided."""
        from nemo_rl.models.megatron.train import model_forward

        mock_model = MagicMock()
        mock_output = torch.randn(1, 10, 100)
        mock_model.return_value = mock_output

        mock_data_dict = MagicMock()
        mock_data_dict.get_multimodal_dict.return_value = {}

        mock_timer = MagicMock()
        mock_ctx = MagicMock()
        mock_timer.return_value = mock_ctx

        result = model_forward(
            model=mock_model,
            data_dict=mock_data_dict,
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3),
            straggler_timer=mock_timer,
        )

        # Verify straggler_timer was called as a context manager
        mock_timer.assert_called_once()
        mock_ctx.__enter__.assert_called_once()
        mock_ctx.__exit__.assert_called_once()
        assert torch.equal(result, mock_output)

    def test_model_forward_with_packed_seq_params(self):
        """Test model_forward passes packed_seq_params to model."""
        from nemo_rl.models.megatron.train import model_forward

        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 10, 100)

        mock_data_dict = MagicMock()
        mock_data_dict.get_multimodal_dict.return_value = {}

        mock_packed_seq_params = MagicMock()

        model_forward(
            model=mock_model,
            data_dict=mock_data_dict,
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3),
            packed_seq_params=mock_packed_seq_params,
        )

        # Verify packed_seq_params was passed
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["packed_seq_params"] == mock_packed_seq_params

    def test_model_forward_with_defer_fp32_logits(self):
        """Test model_forward passes fp32_output when defer_fp32_logits is True."""
        from nemo_rl.models.megatron.train import model_forward

        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 10, 100)

        mock_data_dict = MagicMock()
        mock_data_dict.get_multimodal_dict.return_value = {}

        model_forward(
            model=mock_model,
            data_dict=mock_data_dict,
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3),
            defer_fp32_logits=True,
        )

        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["fp32_output"] is False

    def test_model_forward_clears_position_ids_for_multimodal(self):
        """Test model_forward sets position_ids to None for multimodal data."""
        from nemo_rl.models.megatron.train import model_forward

        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 10, 100)

        mock_data_dict = MagicMock()
        mock_data_dict.get_multimodal_dict.return_value = {
            "images": torch.randn(1, 3, 224, 224)
        }

        model_forward(
            model=mock_model,
            data_dict=mock_data_dict,
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3),
        )

        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["position_ids"] is None

    def test_model_forward_passes_direct_labels_and_loss_mask(self):
        from nemo_rl.models.megatron.train import model_forward

        model = MagicMock(return_value=torch.ones(1, 4))
        data = MagicMock()
        data.get_multimodal_dict.return_value = {}
        labels = torch.tensor([[2, 3, 4, -100]])
        loss_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        model_forward(
            model=model,
            data_dict=data,
            input_ids_cp_sharded=torch.tensor([[1, 2, 3, 4]]),
            position_ids=torch.tensor([[0, 1, 2, 3]]),
            attention_mask=None,
            labels_cp_sharded=labels,
            loss_mask_cp_sharded=loss_mask,
        )

        assert (
            model.call_args.kwargs["labels"] is labels,
            model.call_args.kwargs["loss_mask"] is loss_mask,
        ) == (True, True)

    def test_direct_labels_reject_fused_linear_logprobs(self):
        from nemo_rl.models.megatron.train import model_forward

        model = MagicMock(return_value=torch.ones(1, 4))
        data = MagicMock()
        data.get_multimodal_dict.return_value = {}
        labels = torch.tensor([[2, 3, 4, -100]])
        loss_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        with pytest.raises(
            ValueError,
            match="Direct packed SFT labels do not support fused linear logprobs",
        ):
            model_forward(
                model=model,
                data_dict=data,
                input_ids_cp_sharded=torch.tensor([[1, 2, 3, 4]]),
                position_ids=torch.tensor([[0, 1, 2, 3]]),
                attention_mask=None,
                labels_cp_sharded=labels,
                loss_mask_cp_sharded=loss_mask,
                use_fused_linear_logprobs=True,
            )

    def test_mtp_mask_remains_compatible_with_fused_linear_logprobs(self):
        from nemo_rl.models.megatron.train import model_forward

        model = MagicMock(return_value=torch.ones(1, 4))
        data = MagicMock()
        data.get_multimodal_dict.return_value = {}
        input_ids = torch.tensor([[1, 2, 3, 4]])
        mtp_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        model_forward(
            model=model,
            data_dict=data,
            input_ids_cp_sharded=input_ids,
            position_ids=torch.tensor([[0, 1, 2, 3]]),
            attention_mask=None,
            mtp_loss_mask=mtp_mask,
            use_fused_linear_logprobs=True,
        )

        assert model.call_args.kwargs["labels"] is input_ids
        assert model.call_args.kwargs["loss_mask"] is mtp_mask
        assert model.call_args.kwargs["return_logprobs_for_linear_ce_fusion"] is True


class TestApplyTemperatureScaling:
    """Tests for apply_temperature_scaling function."""

    def test_temperature_scaling_sampling_params_is_none(self):
        """Test that logits are unchanged when sampling_params is None."""
        from nemo_rl.models.megatron.train import apply_temperature_scaling

        logits = torch.ones(2, 10, 100) * 3.0
        sampling_params = None

        result = apply_temperature_scaling(logits, sampling_params)

        assert torch.allclose(result, torch.ones_like(result) * 3.0)

    def test_temperature_scaling_with_temperature_one(self):
        """Test that temperature=1.0 leaves logits unchanged."""
        from nemo_rl.models.megatron.train import apply_temperature_scaling

        logits = torch.randn(2, 10, 100)
        original = logits.clone()
        sampling_params = TrainingSamplingParams(temperature=1.0)

        result = apply_temperature_scaling(logits, sampling_params)

        assert torch.allclose(result, original)

    def test_temperature_scaling_with_temperature_two(self):
        """Test that logits are divided by the configured temperature=2.0."""
        from nemo_rl.models.megatron.train import apply_temperature_scaling

        logits = torch.ones(2, 10, 100) * 4.0
        sampling_params = TrainingSamplingParams(temperature=2.0)

        result = apply_temperature_scaling(logits, sampling_params)

        # 4.0 / 2.0 = 2.0
        assert torch.allclose(result, torch.ones_like(result) * 2.0)
        # Verify in-place: result is the same tensor
        assert result.data_ptr() == logits.data_ptr()


class TestForwardWithPostProcessingFn:
    """Tests for forward_with_post_processing_fn function."""

    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", return_value=1
    )
    @patch("nemo_rl.models.megatron.train.model_forward")
    def test_forward_with_loss_post_processor(
        self, mock_model_forward, mock_cp_size, mock_cp_grp, mock_tp_grp, mock_tp_rank
    ):
        """Test forward with LossPostProcessor."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            forward_with_post_processing_fn,
        )

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        mock_model_forward.return_value = torch.randn(2, 10, 100)

        # Create processed microbatch
        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
        )

        data_iterator = iter([processed_mb])
        mock_model = MagicMock()
        cfg = {"sequence_packing": {"enabled": False}}

        mock_loss_fn = MagicMock()
        post_processor = LossPostProcessor(loss_fn=mock_loss_fn, cfg=cfg)

        output, wrapped_fn = forward_with_post_processing_fn(
            data_iterator=data_iterator,
            model=mock_model,
            post_processing_fn=post_processor,
        )

        mock_model_forward.assert_called_once()

        # forward_with_post_processing_fn should return a callable
        assert callable(wrapped_fn)
        assert isinstance(output, torch.Tensor)

    @patch("nemo_rl.models.megatron.train.model_forward")
    def test_forward_with_logprobs_post_processor(self, mock_model_forward):
        """Test forward with LogprobsPostProcessor."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            LogprobsPostProcessor,
            forward_with_post_processing_fn,
        )

        mock_model_forward.return_value = torch.randn(2, 10, 100)

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
            original_seq_length=3,
        )

        data_iterator = iter([processed_mb])
        cfg = {"sequence_packing": {"enabled": False}}
        post_processor = LogprobsPostProcessor(cfg=cfg)

        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            forward_with_post_processing_fn(
                data_iterator=data_iterator,
                model=MagicMock(),
                post_processing_fn=post_processor,
            )

        mock_model_forward.assert_called_once()

    @patch("nemo_rl.models.megatron.train.model_forward")
    def test_forward_with_topk_post_processor(self, mock_model_forward):
        """Test forward with TopkLogitsPostProcessor."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            TopkLogitsPostProcessor,
            forward_with_post_processing_fn,
        )

        mock_model_forward.return_value = torch.randn(2, 10, 100)

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
            original_seq_length=3,
        )

        data_iterator = iter([processed_mb])
        cfg = {
            "sequence_packing": {"enabled": False},
            "megatron_cfg": {"context_parallel_size": 1},
        }
        post_processor = TopkLogitsPostProcessor(cfg=cfg, k=5)

        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            forward_with_post_processing_fn(
                data_iterator=data_iterator,
                model=MagicMock(),
                post_processing_fn=post_processor,
            )

        mock_model_forward.assert_called_once()

    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", return_value=1
    )
    @patch("nemo_rl.models.megatron.train.model_forward")
    @patch("nemo_rl.models.megatron.train.apply_temperature_scaling")
    def test_forward_applies_temperature_scaling_for_loss(
        self,
        mock_temp_scaling,
        mock_model_forward,
        mock_cp_size,
        mock_cp_grp,
        mock_tp_grp,
        mock_tp_rank,
    ):
        """Test that forward_with_post_processing_fn applies temperature scaling for LossPostProcessor."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            forward_with_post_processing_fn,
        )

        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        output_tensor = torch.randn(2, 10, 100)
        mock_model_forward.return_value = output_tensor

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
        )

        cfg = {
            "sequence_packing": {"enabled": False},
            "generation": {"temperature": 0.7, "top_p": 1.0, "top_k": None},
        }
        post_processor = LossPostProcessor(loss_fn=MagicMock(), cfg=cfg)
        sampling_params = TrainingSamplingParams(
            temperature=cfg["generation"]["temperature"]
        )

        forward_with_post_processing_fn(
            data_iterator=iter([processed_mb]),
            model=MagicMock(),
            post_processing_fn=post_processor,
            sampling_params=sampling_params,
        )

        # Verify apply_temperature_scaling was called with the output tensor and cfg
        mock_temp_scaling.assert_called_once_with(output_tensor, sampling_params)

    def test_forward_with_direct_labels_routes_model_loss_without_temperature_scaling(
        self,
    ):
        from nemo_rl.algorithms.loss import NLLLossFn
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        from nemo_rl.models.megatron import train as megatron_train
        from nemo_rl.models.megatron.data import ProcessedMicrobatch

        labels = torch.tensor([[2, 3, 4, 5]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        processed_mb = ProcessedMicrobatch(
            data_dict=BatchedDataDict({"sample_mask": torch.ones(1)}),
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=None,
            position_ids=torch.tensor([[0, 1, 2, 3]]),
            packed_seq_params=MagicMock(),
            cu_seqlens_padded=torch.tensor([0, 4]),
            labels_cp_sharded=labels,
            loss_mask_cp_sharded=loss_mask,
        )
        processor = megatron_train.LossPostProcessor(
            loss_fn=NLLLossFn(),
            cfg={"sequence_packing": {"enabled": True}},
        )

        with (
            patch.object(
                megatron_train,
                "model_forward",
                return_value=torch.ones(1, 4),
            ) as model_forward_mock,
            patch.object(
                megatron_train, "apply_temperature_scaling"
            ) as temperature_mock,
            patch.object(
                megatron_train, "get_context_parallel_world_size", return_value=1
            ),
        ):
            megatron_train.forward_with_post_processing_fn(
                data_iterator=iter([processed_mb]),
                model=MagicMock(),
                post_processing_fn=processor,
                global_valid_toks=torch.tensor(3.0),
                sampling_params=TrainingSamplingParams(temperature=0.5),
            )

        assert (
            model_forward_mock.call_args.kwargs["labels_cp_sharded"] is labels,
            model_forward_mock.call_args.kwargs["loss_mask_cp_sharded"] is loss_mask,
            temperature_mock.call_count,
        ) == (True, True, 0)

    @patch("nemo_rl.models.megatron.train.model_forward")
    @patch("nemo_rl.models.megatron.train.apply_temperature_scaling")
    def test_forward_applies_temperature_scaling_for_logprobs(
        self, mock_temp_scaling, mock_model_forward
    ):
        """Test that forward_with_post_processing_fn applies temperature scaling for LogprobsPostProcessor."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            LogprobsPostProcessor,
            forward_with_post_processing_fn,
        )

        output_tensor = torch.randn(2, 10, 100)
        mock_model_forward.return_value = output_tensor

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
            original_seq_length=3,
        )

        cfg = {
            "sequence_packing": {"enabled": False},
            "generation": {"temperature": 0.5, "top_p": 1.0, "top_k": None},
        }
        post_processor = LogprobsPostProcessor(cfg=cfg)
        sampling_params = TrainingSamplingParams(
            temperature=cfg["generation"]["temperature"]
        )

        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            forward_with_post_processing_fn(
                data_iterator=iter([processed_mb]),
                model=MagicMock(),
                post_processing_fn=post_processor,
                sampling_params=sampling_params,
            )

        mock_temp_scaling.assert_called_once_with(output_tensor, sampling_params)

    @patch("nemo_rl.models.megatron.train.model_forward")
    @patch("nemo_rl.models.megatron.train.apply_temperature_scaling")
    def test_forward_applies_temperature_scaling_for_topk(
        self, mock_temp_scaling, mock_model_forward
    ):
        """Test that forward_with_post_processing_fn applies temperature scaling for TopkLogitsPostProcessor."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            TopkLogitsPostProcessor,
            forward_with_post_processing_fn,
        )

        output_tensor = torch.randn(2, 10, 100)
        mock_model_forward.return_value = output_tensor

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
            original_seq_length=3,
        )

        cfg = {
            "sequence_packing": {"enabled": False},
            "megatron_cfg": {"context_parallel_size": 1},
            "generation": {"temperature": 1.5, "top_p": 1.0, "top_k": None},
        }
        post_processor = TopkLogitsPostProcessor(cfg=cfg, k=5)
        sampling_params = TrainingSamplingParams(
            temperature=cfg["generation"]["temperature"]
        )
        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            forward_with_post_processing_fn(
                data_iterator=iter([processed_mb]),
                model=MagicMock(),
                post_processing_fn=post_processor,
                sampling_params=sampling_params,
            )

        mock_temp_scaling.assert_called_once_with(output_tensor, sampling_params)

    @patch("nemo_rl.models.megatron.train.model_forward")
    @patch("nemo_rl.models.megatron.train.apply_temperature_scaling")
    def test_forward_does_not_apply_temperature_scaling_for_unknown_type(
        self, mock_temp_scaling, mock_model_forward
    ):
        """Test that temperature scaling is NOT applied for unknown post-processor types (before they raise)."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import forward_with_post_processing_fn

        mock_model_forward.return_value = torch.randn(2, 10, 100)

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
            position_ids=None,
            packed_seq_params=None,
            cu_seqlens_padded=None,
        )

        with pytest.raises(TypeError):
            forward_with_post_processing_fn(
                data_iterator=iter([processed_mb]),
                model=MagicMock(),
                post_processing_fn="not_a_processor",
            )

        mock_temp_scaling.assert_not_called()

    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", return_value=1
    )
    @patch("nemo_rl.models.megatron.train.model_forward")
    def test_forward_with_straggler_timer(
        self, mock_model_forward, mock_cp_size, mock_cp_grp, mock_tp_grp, mock_tp_rank
    ):
        """Test that straggler_timer is passed through to model_forward."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            forward_with_post_processing_fn,
        )

        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()
        mock_model_forward.return_value = torch.randn(2, 10, 100)

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
        )

        cfg = {"sequence_packing": {"enabled": False}}
        post_processor = LossPostProcessor(loss_fn=MagicMock(), cfg=cfg)
        mock_timer = MagicMock()

        forward_with_post_processing_fn(
            data_iterator=iter([processed_mb]),
            model=MagicMock(),
            post_processing_fn=post_processor,
            straggler_timer=mock_timer,
        )

        # Verify straggler_timer was passed to model_forward
        call_kwargs = mock_model_forward.call_args[1]
        assert call_kwargs["straggler_timer"] is mock_timer

    @patch("nemo_rl.models.megatron.train.model_forward")
    def test_forward_with_unknown_post_processor_raises(self, mock_model_forward):
        """Test that unknown post-processor type raises TypeError."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import forward_with_post_processing_fn

        mock_model_forward.return_value = torch.randn(2, 10, 100)

        processed_mb = ProcessedMicrobatch(
            data_dict=MagicMock(),
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
            position_ids=None,
            packed_seq_params=None,
            cu_seqlens_padded=None,
        )

        data_iterator = iter([processed_mb])
        unknown_processor = "not_a_processor"

        with pytest.raises(TypeError, match="Unknown post-processing function type"):
            forward_with_post_processing_fn(
                data_iterator=data_iterator,
                model=MagicMock(),
                post_processing_fn=unknown_processor,
            )

    @patch("nemo_rl.models.megatron.train.get_capture_context")
    @patch("nemo_rl.models.megatron.train.model_forward")
    def test_forward_without_draft_model_does_not_inject_student_logits(
        self, mock_model_forward, mock_get_capture_context
    ):
        """Without a draft model, the forward path should remain unchanged."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            LogprobsPostProcessor,
            forward_with_post_processing_fn,
        )

        mock_model_forward.return_value = torch.randn(2, 3, 5)
        mock_get_capture_context.return_value = (nullcontext(), None)

        data_dict = MagicMock()
        processed_mb = ProcessedMicrobatch(
            data_dict=data_dict,
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
            original_seq_length=3,
        )
        post_processor = LogprobsPostProcessor(
            cfg={"sequence_packing": {"enabled": False}}
        )

        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            output, wrapped_fn = forward_with_post_processing_fn(
                data_iterator=iter([processed_mb]),
                model=MagicMock(),
                post_processing_fn=post_processor,
                draft_model=None,
            )

        assert "student_logits" not in data_dict
        assert callable(wrapped_fn)
        assert torch.equal(output, mock_model_forward.return_value)

    @patch("megatron.core.transformer.multi_token_prediction.roll_tensor")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_capture_context")
    @patch("nemo_rl.models.megatron.train.model_forward")
    def test_forward_with_draft_model_rolls_input_embeds_before_draft_forward(
        self,
        mock_model_forward,
        mock_get_capture_context,
        mock_get_cp_group,
        mock_roll_tensor,
    ):
        """Draft forward should consume the one-token-shifted input embeddings."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch
        from nemo_rl.models.megatron.train import (
            LogprobsPostProcessor,
            forward_with_post_processing_fn,
        )

        output_tensor = torch.randn(2, 3, 5)
        student_logits = torch.randn(2, 3, 5)
        hidden_states = torch.randn(3, 1, 4)
        inputs_embeds = torch.randn(3, 1, 4)
        shifted_embeds = torch.randn(3, 1, 4)
        cp_group = MagicMock()

        mock_model_forward.return_value = output_tensor
        mock_get_cp_group.return_value = cp_group
        mock_roll_tensor.return_value = (shifted_embeds, None)
        mock_capture = MagicMock()
        mock_capture.get_captured_states.return_value = SimpleNamespace(
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
        )
        mock_get_capture_context.return_value = (nullcontext(), mock_capture)

        data_dict = {"input_ids": torch.tensor([[1, 2, 3]])}
        attention_mask = torch.ones(1, 3)
        processed_mb = ProcessedMicrobatch(
            data_dict=data_dict,
            input_ids=torch.tensor([[1, 2, 3]]),
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            attention_mask=attention_mask,
            position_ids=torch.tensor([[0, 1, 2]]),
            packed_seq_params=None,
            cu_seqlens_padded=None,
            original_seq_length=3,
        )
        post_processor = LogprobsPostProcessor(
            cfg={"sequence_packing": {"enabled": False}}
        )
        draft_model = MagicMock(return_value=student_logits)

        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            forward_with_post_processing_fn(
                data_iterator=iter([processed_mb]),
                model=MagicMock(),
                post_processing_fn=post_processor,
                draft_model=draft_model,
            )

        mock_roll_tensor.assert_called_once_with(
            inputs_embeds,
            shifts=-1,
            dims=0,
            cp_group=cp_group,
        )
        draft_model.assert_called_once_with(
            hidden_states=hidden_states,
            input_embeds=shifted_embeds,
            attention_mask=attention_mask,
        )
        assert data_dict["student_logits"] is student_logits


class TestMegatronForwardBackward:
    """Tests for megatron_forward_backward function."""

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_megatron_forward_backward_calls_forward_backward_func(self, mock_get_fb):
        """Test that megatron_forward_backward calls the forward_backward_func."""
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        mock_fb_func = MagicMock(return_value={"loss": torch.tensor(0.5)})
        mock_get_fb.return_value = mock_fb_func

        mock_model = MagicMock()
        mock_loss_fn = MagicMock()
        cfg = {"sequence_packing": {"enabled": False}}
        post_processor = LossPostProcessor(loss_fn=mock_loss_fn, cfg=cfg)

        megatron_forward_backward(
            model=mock_model,
            data_iterator=iter([]),
            num_microbatches=4,
            seq_length=128,
            mbs=2,
            post_processing_fn=post_processor,
        )

        mock_get_fb.assert_called_once()
        mock_fb_func.assert_called_once()

        # Verify key arguments
        call_kwargs = mock_fb_func.call_args[1]
        assert call_kwargs["num_microbatches"] == 4
        assert call_kwargs["seq_length"] == 128
        assert call_kwargs["micro_batch_size"] == 2

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_megatron_forward_backward_forward_only(self, mock_get_fb):
        """Test megatron_forward_backward with forward_only=True."""
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        mock_fb_func = MagicMock()
        mock_get_fb.return_value = mock_fb_func

        cfg = {"sequence_packing": {"enabled": False}}
        post_processor = LossPostProcessor(loss_fn=MagicMock(), cfg=cfg)

        megatron_forward_backward(
            model=MagicMock(),
            data_iterator=iter([]),
            num_microbatches=1,
            seq_length=64,
            mbs=1,
            post_processing_fn=post_processor,
            forward_only=True,
        )

        call_kwargs = mock_fb_func.call_args[1]
        assert call_kwargs["forward_only"] is True

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_forward_only_preserves_activation_offload_warmup(
        self, mock_get_fb: MagicMock
    ) -> None:
        """Forward-only RL stages must not consume activation-offload warmup."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        model_config = SimpleNamespace(fine_grained_activation_offloading=True)
        model = SimpleNamespace(config=model_config)
        empty_manager = SimpleNamespace(
            _is_warmup=True,
            _cached_chunks_forward=[],
        )

        def run_forward_only(**kwargs: Any) -> dict[str, torch.Tensor]:
            assert kwargs["forward_only"] is True
            assert model_config.fine_grained_activation_offloading is False
            PipelineOffloadManager.OFFLOAD_MGR = empty_manager
            return {"logprobs": torch.tensor(0.0)}

        mock_get_fb.return_value = run_forward_only
        post_processor = LossPostProcessor(
            loss_fn=MagicMock(), cfg={"sequence_packing": {"enabled": False}}
        )

        with patch.object(PipelineOffloadManager, "OFFLOAD_MGR", None):
            megatron_forward_backward(
                model=model,
                data_iterator=iter([]),
                num_microbatches=1,
                seq_length=64,
                mbs=1,
                post_processing_fn=post_processor,
                forward_only=True,
            )

            assert PipelineOffloadManager.OFFLOAD_MGR is empty_manager
            assert empty_manager._is_warmup is True
            assert empty_manager._cached_chunks_forward == []

        assert model_config.fine_grained_activation_offloading is True

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_forward_only_does_not_consume_warm_manager_chunks(
        self, mock_get_fb: MagicMock
    ) -> None:
        """Logprob stages must not advance chunks cached by a prior training step."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        class StatefulOffloadManager:
            def __init__(self) -> None:
                self.do_offload = True
                self.cached_forward_index = 0

            def disable_offload(self) -> None:
                self.do_offload = False

            def enable_offload(self) -> None:
                self.do_offload = True

            def consume_cached_chunk(self) -> None:
                if self.do_offload:
                    self.cached_forward_index += 1

            def reset(self) -> None:
                self.cached_forward_index = 0

        manager = StatefulOffloadManager()
        model_config = SimpleNamespace(fine_grained_activation_offloading=True)
        model = SimpleNamespace(config=model_config)
        observed_phases: list[tuple[bool, int]] = []

        def run_schedule(**kwargs: Any) -> dict[str, torch.Tensor]:
            manager.consume_cached_chunk()
            observed_phases.append(
                (kwargs["forward_only"], manager.cached_forward_index)
            )
            if not kwargs["forward_only"]:
                manager.reset()
            return {}

        mock_get_fb.return_value = run_schedule
        post_processor = LossPostProcessor(
            loss_fn=MagicMock(), cfg={"sequence_packing": {"enabled": False}}
        )

        with patch.object(PipelineOffloadManager, "OFFLOAD_MGR", manager):
            for forward_only in (True, False, True, False):
                megatron_forward_backward(
                    model=model,
                    data_iterator=iter([]),
                    num_microbatches=1,
                    seq_length=64,
                    mbs=1,
                    post_processing_fn=post_processor,
                    forward_only=forward_only,
                )

                assert manager.do_offload is True

        assert observed_phases == [(True, 0), (False, 1), (True, 0), (False, 1)]

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_forward_only_preserves_disabled_manager_state(
        self, mock_get_fb: MagicMock
    ) -> None:
        """Nested callers that disabled offload must remain disabled."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        manager = MagicMock()
        manager.do_offload = False
        model_config = SimpleNamespace(fine_grained_activation_offloading=True)

        def run_forward_only(**kwargs: Any) -> dict[str, torch.Tensor]:
            assert kwargs["forward_only"] is True
            assert manager.do_offload is False
            return {}

        mock_get_fb.return_value = run_forward_only
        post_processor = LossPostProcessor(
            loss_fn=MagicMock(), cfg={"sequence_packing": {"enabled": False}}
        )

        with patch.object(PipelineOffloadManager, "OFFLOAD_MGR", manager):
            megatron_forward_backward(
                model=SimpleNamespace(config=model_config),
                data_iterator=iter([]),
                num_microbatches=1,
                seq_length=64,
                mbs=1,
                post_processing_fn=post_processor,
                forward_only=True,
            )

        manager.disable_offload.assert_not_called()
        manager.enable_offload.assert_not_called()
        assert manager.do_offload is False

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_forward_only_restores_vpp_configs(self, mock_get_fb: MagicMock) -> None:
        """Shared and distinct VPP configs are suspended and restored atomically."""
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        shared_config = SimpleNamespace(fine_grained_activation_offloading=True)
        distinct_config = SimpleNamespace(fine_grained_activation_offloading=True)
        model = [MagicMock(), MagicMock(), MagicMock()]

        def run_forward_only(**kwargs: Any) -> dict[str, torch.Tensor]:
            assert shared_config.fine_grained_activation_offloading is False
            assert distinct_config.fine_grained_activation_offloading is False
            return {}

        mock_get_fb.return_value = run_forward_only
        post_processor = LossPostProcessor(
            loss_fn=MagicMock(), cfg={"sequence_packing": {"enabled": False}}
        )

        with patch(
            "nemo_rl.models.megatron.train.get_model_config",
            side_effect=[shared_config, shared_config, distinct_config],
        ):
            megatron_forward_backward(
                model=model,
                data_iterator=iter([]),
                num_microbatches=1,
                seq_length=64,
                mbs=1,
                post_processing_fn=post_processor,
                forward_only=True,
            )

        assert shared_config.fine_grained_activation_offloading is True
        assert distinct_config.fine_grained_activation_offloading is True

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_forward_only_config_discovery_is_atomic(
        self, mock_get_fb: MagicMock
    ) -> None:
        """A later VPP wrapper error must not leave earlier configs disabled."""
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        first_config = SimpleNamespace(fine_grained_activation_offloading=True)
        post_processor = LossPostProcessor(
            loss_fn=MagicMock(), cfg={"sequence_packing": {"enabled": False}}
        )

        with (
            patch(
                "nemo_rl.models.megatron.train.get_model_config",
                side_effect=[first_config, RuntimeError("invalid VPP wrapper")],
            ),
            pytest.raises(RuntimeError, match="invalid VPP wrapper"),
        ):
            megatron_forward_backward(
                model=[MagicMock(), MagicMock()],
                data_iterator=iter([]),
                num_microbatches=1,
                seq_length=64,
                mbs=1,
                post_processing_fn=post_processor,
                forward_only=True,
            )

        mock_get_fb.return_value.assert_not_called()
        assert first_config.fine_grained_activation_offloading is True

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_forward_only_restores_activation_offload_after_error(self, mock_get_fb):
        """A failed forward-only stage must restore activation offload for training."""
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        model_config = SimpleNamespace(fine_grained_activation_offloading=True)
        model = SimpleNamespace(config=model_config)

        def fail_forward_only(**kwargs):
            assert kwargs["forward_only"] is True
            assert model_config.fine_grained_activation_offloading is False
            raise RuntimeError("forward-only failure")

        mock_get_fb.return_value = fail_forward_only
        post_processor = LossPostProcessor(
            loss_fn=MagicMock(), cfg={"sequence_packing": {"enabled": False}}
        )

        with pytest.raises(RuntimeError, match="forward-only failure"):
            megatron_forward_backward(
                model=model,
                data_iterator=iter([]),
                num_microbatches=1,
                seq_length=64,
                mbs=1,
                post_processing_fn=post_processor,
                forward_only=True,
            )

        assert model_config.fine_grained_activation_offloading is True

    @patch("nemo_rl.models.megatron.train.get_forward_backward_func")
    def test_training_keeps_activation_offload_enabled(self, mock_get_fb):
        """Training must retain activation offload so MCore can warm up and run it."""
        from nemo_rl.models.megatron.train import (
            LossPostProcessor,
            megatron_forward_backward,
        )

        model_config = SimpleNamespace(fine_grained_activation_offloading=True)
        model = SimpleNamespace(config=model_config)

        def run_training(**kwargs):
            assert kwargs["forward_only"] is False
            assert model_config.fine_grained_activation_offloading is True
            return {"loss": torch.tensor(0.5)}

        mock_get_fb.return_value = run_training
        post_processor = LossPostProcessor(
            loss_fn=MagicMock(), cfg={"sequence_packing": {"enabled": False}}
        )

        megatron_forward_backward(
            model=model,
            data_iterator=iter([]),
            num_microbatches=1,
            seq_length=64,
            mbs=1,
            post_processing_fn=post_processor,
            forward_only=False,
        )

        assert model_config.fine_grained_activation_offloading is True


class TestLossPostProcessor:
    """Tests for LossPostProcessor class."""

    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", return_value=1
    )
    def test_loss_post_processor_no_packing(
        self, mock_cp_size, mock_cp_grp, mock_tp_grp, mock_tp_rank
    ):
        """Test LossPostProcessor without sequence packing."""
        from nemo_rl.models.megatron.train import LossPostProcessor

        mock_loss_fn = MagicMock(return_value=(torch.tensor(0.5), {"loss": 0.5}))
        mock_loss_fn.input_type = LossInputType.LOGIT
        cfg = {"sequence_packing": {"enabled": False}}

        processor = LossPostProcessor(loss_fn=mock_loss_fn, cfg=cfg, cp_normalize=False)

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        wrapped_fn = processor(
            data_dict=MagicMock(),
            packed_seq_params=None,
            global_valid_seqs=torch.tensor(10),
            global_valid_toks=torch.tensor(100),
        )

        # Call the wrapped function
        output_tensor = torch.randn(2, 10, 100)
        loss, metrics = wrapped_fn(output_tensor)

        assert torch.isclose(loss, torch.tensor(0.5))
        assert isinstance(metrics, dict)
        assert len(metrics) == 1 and metrics["loss"] == 0.5

    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", return_value=2
    )
    def test_loss_post_processor_with_cp_normalize(
        self, mock_cp_size, mock_cp_grp, mock_tp_grp, mock_tp_rank
    ):
        """Test LossPostProcessor with CP normalization and microbatch pre-scaling."""
        from nemo_rl.models.megatron.train import LossPostProcessor

        mock_loss_fn = MagicMock(return_value=(torch.tensor(1.0), {}))
        mock_loss_fn.input_type = LossInputType.LOGIT
        cfg = {"sequence_packing": {"enabled": False}}

        processor = LossPostProcessor(
            loss_fn=mock_loss_fn, cfg=cfg, num_microbatches=4, cp_normalize=True
        )

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        wrapped_fn = processor(data_dict=MagicMock())

        output_tensor = torch.randn(2, 10, 100)
        loss, _ = wrapped_fn(output_tensor)

        # Loss should be scaled by num_microbatches / (cp_size * cp_size) = 4 / (2 * 2) = 1.0
        assert torch.isclose(loss, torch.tensor(1.0))

    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", return_value=1
    )
    @patch("nemo_rl.models.megatron.train.SequencePackingLossWrapper")
    def test_loss_post_processor_with_packing(
        self, mock_wrapper, mock_cp_size, mock_cp_grp, mock_tp_grp, mock_tp_rank
    ):
        """Test LossPostProcessor with sequence packing."""
        from nemo_rl.models.megatron.train import LossPostProcessor

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        mock_loss_fn = MagicMock()
        cfg = {"sequence_packing": {"enabled": True}}

        mock_packed_seq_params = MagicMock()
        mock_packed_seq_params.cu_seqlens_q = torch.tensor([0, 5, 10])
        mock_packed_seq_params.cu_seqlens_q_padded = torch.tensor([0, 8, 16])

        processor = LossPostProcessor(loss_fn=mock_loss_fn, cfg=cfg, cp_normalize=False)

        processor(data_dict=MagicMock(), packed_seq_params=mock_packed_seq_params)

        # Verify SequencePackingLossWrapper was called
        mock_wrapper.assert_called_once()


def test_direct_model_loss_normalizes_target_aligned_tokens_and_schedule_scaling():
    loss, _ = _run_direct_model_loss()

    # (1+3)/6 masked mean * num_microbatches(4) / cp_size(2), then the default
    # cp_normalize division by cp_size(2) that every Megatron loss path applies.
    assert torch.isclose(loss, torch.tensor(2.0 / 3.0))


def test_direct_model_loss_defers_host_scalar_materialization():
    _, metrics = _run_direct_model_loss()

    assert isinstance(metrics["loss"], torch.Tensor)
    assert not metrics["loss"].requires_grad
    assert torch.isclose(metrics["loss"], torch.tensor(2.0 / 3.0))
    assert isinstance(metrics["num_valid_samples"], torch.Tensor)
    assert not metrics["num_valid_samples"].requires_grad
    assert torch.isclose(metrics["num_valid_samples"], torch.tensor(1.0))
    assert "num_unmasked_tokens" not in metrics


def test_direct_model_loss_rejects_non_nll_loss_semantics():
    from nemo_rl.models.megatron.train import LossPostProcessor

    processor = LossPostProcessor(
        loss_fn=MagicMock(),
        cfg={"sequence_packing": {"enabled": True}},
    )

    with pytest.raises(
        TypeError,
        match=r"direct Megatron-LM prepacked SFT requires.*NLLLossFn",
    ):
        processor(
            data_dict=MagicMock(),
            global_valid_toks=torch.tensor(1.0),
            prepacked_loss_mask=torch.ones(1, 4),
        )


def test_direct_model_loss_rejects_misaligned_target_mask():
    from nemo_rl.algorithms.loss import NLLLossFn
    from nemo_rl.models.megatron.train import LossPostProcessor

    processor = LossPostProcessor(
        loss_fn=NLLLossFn(),
        cfg={"sequence_packing": {"enabled": True}},
    )
    wrapped = processor(
        data_dict=MagicMock(),
        global_valid_toks=torch.tensor(1.0),
        prepacked_loss_mask=torch.ones(1, 3),
    )

    with pytest.raises(ValueError, match="loss and loss mask shapes must match"):
        wrapped(torch.ones(1, 4))


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {
                "packed_cu_seqlens": torch.tensor([[0, 4]]),
                "target_ids": torch.ones(1, 4),
            },
            True,
        ),
        ({"packed_cu_seqlens": torch.tensor([[0, 4]])}, False),
        ({"target_ids": torch.ones(1, 4)}, False),
    ],
)
def test_context_parallel_loss_reduction_is_selected_only_for_direct_packed_sft(
    data: dict[str, torch.Tensor], expected: bool
):
    from nemo_rl.models.megatron.train import (
        should_reduce_loss_across_context_parallel,
    )

    assert should_reduce_loss_across_context_parallel(data) is expected


def test_context_parallel_metric_cleanup_preserves_nonlocal_metrics():
    from nemo_rl.models.megatron.train import (
        strip_context_parallel_local_loss_metric,
    )

    result = strip_context_parallel_local_loss_metric(
        {"loss": [torch.tensor(1.0)], "lr": [1e-4]},
        enabled=True,
    )

    assert result == {"lr": [1e-4]}


class TestLogprobsPostProcessor:
    """Tests for LogprobsPostProcessor class."""

    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.from_parallel_logits_to_logprobs")
    def test_logprobs_post_processor_no_packing(
        self, mock_from_logits, mock_tp_rank, mock_tp_grp
    ):
        """Test LogprobsPostProcessor without sequence packing."""
        from nemo_rl.models.megatron.train import LogprobsPostProcessor

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()

        cfg = {"sequence_packing": {"enabled": False}}
        processor = LogprobsPostProcessor(cfg=cfg)

        mock_data_dict = MagicMock()
        mock_data_dict.__getitem__ = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
        )

        mock_logprobs = torch.randn(1, 7)  # One less than padded input length
        mock_from_logits.return_value = mock_logprobs

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]),
            cu_seqlens_padded=None,
            original_seq_length=5,
        )

        output_tensor = torch.randn(1, 8, 100)
        loss, result = wrapped_fn(output_tensor)

        # Loss should be 0
        assert loss.item() == 0.0
        # Result should have logprobs key
        assert "logprobs" in result
        # Logprobs should be prepended with a 0 and dense padding removed
        expected = torch.cat(
            [torch.zeros_like(mock_logprobs[:, :1]), mock_logprobs], dim=1
        )[:, :5]
        assert torch.equal(result["logprobs"], expected)

    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.from_parallel_logits_to_logprobs_packed_sequences"
    )
    def test_logprobs_post_processor_with_packing(
        self, mock_from_logits_packed, mock_cp_grp, mock_tp_rank, mock_tp_grp
    ):
        """Test LogprobsPostProcessor with sequence packing."""
        from nemo_rl.models.megatron.train import LogprobsPostProcessor

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        cfg = {"sequence_packing": {"enabled": True}}
        processor = LogprobsPostProcessor(cfg=cfg)

        mock_data_dict = MagicMock()
        mock_data_dict.__getitem__ = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 4, 5]])
        )

        mock_logprobs = torch.randn(1, 4)
        mock_from_logits_packed.return_value = mock_logprobs

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            cu_seqlens_padded=torch.tensor([0, 5]),
            original_seq_length=5,
        )

        output_tensor = torch.randn(1, 5, 100)
        loss, result = wrapped_fn(output_tensor)

        mock_from_logits_packed.assert_called_once()
        assert "logprobs" in result


class TestTopkLogitsPostProcessor:
    """Tests for TopkLogitsPostProcessor class."""

    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.distributed_vocab_topk")
    def test_topk_post_processor_no_packing(self, mock_topk, mock_tp_rank, mock_tp_grp):
        """Test TopkLogitsPostProcessor without sequence packing."""
        from nemo_rl.models.megatron.train import TopkLogitsPostProcessor

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()

        cfg = {
            "sequence_packing": {"enabled": False},
            "megatron_cfg": {"context_parallel_size": 1},
        }
        k = 5
        processor = TopkLogitsPostProcessor(cfg=cfg, k=k)

        mock_data_dict = MagicMock()
        mock_data_dict.__getitem__ = MagicMock(
            side_effect=lambda key: (
                torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
                if key == "input_ids"
                else torch.tensor([5])
            )
        )

        mock_topk_vals = torch.randn(1, 8, k)
        mock_topk_idx = torch.randint(0, 100, (1, 8, k))
        mock_topk.return_value = (mock_topk_vals, mock_topk_idx)

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            cu_seqlens_padded=None,
            original_seq_length=5,
        )

        output_tensor = torch.randn(1, 8, 100)
        loss, result = wrapped_fn(output_tensor)

        assert torch.equal(result["topk_logits"], mock_topk_vals[:, :5])
        assert torch.equal(result["topk_indices"], mock_topk_idx[:, :5])

    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.distributed_vocab_topk")
    def test_topk_post_processor_with_packing(
        self, mock_topk, mock_tp_rank, mock_tp_grp
    ):
        """Test TopkLogitsPostProcessor with sequence packing."""
        from nemo_rl.models.megatron.train import TopkLogitsPostProcessor

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()

        cfg = {
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {"context_parallel_size": 1},
        }
        k = 3
        processor = TopkLogitsPostProcessor(cfg=cfg, k=k)

        mock_data_dict = MagicMock()
        mock_data_dict.__getitem__ = MagicMock(
            side_effect=lambda key: (
                torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
                if key == "input_ids"
                else torch.tensor([5])
            )
        )

        mock_topk_vals = torch.randn(1, 8, k)
        mock_topk_idx = torch.randint(0, 100, (1, 8, k))
        mock_topk.return_value = (mock_topk_vals, mock_topk_idx)

        cu_seqlens_padded = torch.tensor([0, 5])

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            cu_seqlens_padded=cu_seqlens_padded,
            original_seq_length=8,
        )

        output_tensor = torch.randn(1, 8, 100)
        loss, result = wrapped_fn(output_tensor)

        assert "topk_logits" in result
        assert "topk_indices" in result
        # Output should be unpacked to batch shape
        assert result["topk_logits"].shape[0] == 1

    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.distributed_vocab_topk")
    def test_topk_cp_without_packing_raises(
        self, mock_topk, mock_tp_rank, mock_tp_grp, mock_cp_grp
    ):
        """Test that CP > 1 without packing raises RuntimeError."""
        from nemo_rl.models.megatron.train import TopkLogitsPostProcessor

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        cfg = {
            "sequence_packing": {"enabled": False},
            "megatron_cfg": {"context_parallel_size": 2},
        }
        processor = TopkLogitsPostProcessor(cfg=cfg, k=5)

        mock_data_dict = MagicMock()
        mock_data_dict.__getitem__ = MagicMock(
            side_effect=lambda key: (
                torch.tensor([[1, 2, 3]]) if key == "input_ids" else torch.tensor([3])
            )
        )

        mock_topk.return_value = (
            torch.randn(1, 3, 5),
            torch.randint(0, 100, (1, 3, 5)),
        )

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            cu_seqlens_padded=None,
            original_seq_length=3,
        )

        output_tensor = torch.randn(1, 3, 100)

        with pytest.raises(
            RuntimeError, match="Context Parallelism.*requires sequence packing"
        ):
            wrapped_fn(output_tensor)

    @patch("nemo_rl.models.megatron.train.allgather_cp_sharded_tensor")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.distributed_vocab_topk")
    def test_topk_cp_with_packing_single_sequence(
        self, mock_topk, mock_tp_rank, mock_tp_grp, mock_cp_grp, mock_allgather
    ):
        """Test TopkLogitsPostProcessor with CP > 1 and packing for a single sequence."""
        from nemo_rl.models.megatron.train import TopkLogitsPostProcessor

        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        cp_size = 2
        k = 3
        seq_len = 8  # Total packed length
        local_seq_len = seq_len // cp_size  # Each CP rank sees half

        cfg = {
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {"context_parallel_size": cp_size},
        }
        processor = TopkLogitsPostProcessor(cfg=cfg, k=k)

        mock_data_dict = MagicMock()
        mock_data_dict.__getitem__ = MagicMock(
            side_effect=lambda key: (
                torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
                if key == "input_ids"
                else torch.tensor([8])
            )
        )

        # distributed_vocab_topk returns local (CP-sharded) results
        mock_topk_vals = torch.randn(1, local_seq_len, k)
        mock_topk_idx = torch.randint(0, 100, (1, local_seq_len, k))
        mock_topk.return_value = (mock_topk_vals, mock_topk_idx)

        # allgather returns the full gathered tensor
        gathered_vals = torch.randn(1, seq_len, k)
        gathered_idx = torch.randint(0, 100, (1, seq_len, k))
        mock_allgather.side_effect = [gathered_vals, gathered_idx]

        cu_seqlens_padded = torch.tensor([0, seq_len])

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            cu_seqlens_padded=cu_seqlens_padded,
            original_seq_length=8,
        )

        output_tensor = torch.randn(1, local_seq_len, 100)
        loss, result = wrapped_fn(output_tensor)

        # Verify allgather was called for both vals and indices
        assert mock_allgather.call_count == 2
        assert "topk_logits" in result
        assert "topk_indices" in result
        # Output should be unpacked: (batch_size=1, unpacked_seqlen=8, k=3)
        assert result["topk_logits"].shape == (1, 8, k)
        assert result["topk_indices"].shape == (1, 8, k)

    @patch("nemo_rl.models.megatron.train.allgather_cp_sharded_tensor")
    @patch("nemo_rl.models.megatron.train.get_context_parallel_group")
    @patch("nemo_rl.models.megatron.train.get_tensor_model_parallel_group")
    @patch(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", return_value=0
    )
    @patch("nemo_rl.models.megatron.train.distributed_vocab_topk")
    def test_topk_cp_with_packing_multiple_sequences(
        self, mock_topk, mock_tp_rank, mock_tp_grp, mock_cp_grp, mock_allgather
    ):
        """Test TopkLogitsPostProcessor with CP > 1, packing, and multiple sequences in batch."""
        from nemo_rl.models.megatron.train import TopkLogitsPostProcessor

        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        cp_size = 2
        k = 3
        # Two sequences packed: seq1 has 4 tokens, seq2 has 6 tokens => total packed = 10
        seq1_len = 4
        seq2_len = 6
        total_packed_len = seq1_len + seq2_len
        local_packed_len = total_packed_len // cp_size
        unpacked_seqlen = 6  # Max seq length in batch (for output shape)

        cfg = {
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {"context_parallel_size": cp_size},
        }
        processor = TopkLogitsPostProcessor(cfg=cfg, k=k)

        mock_data_dict = MagicMock()
        mock_data_dict.__getitem__ = MagicMock(
            side_effect=lambda key: (
                torch.zeros(2, unpacked_seqlen, dtype=torch.long)
                if key == "input_ids"
                else torch.tensor([seq1_len, seq2_len])
            )
        )

        # distributed_vocab_topk returns local (CP-sharded) results
        mock_topk_vals = torch.randn(1, local_packed_len, k)
        mock_topk_idx = torch.randint(0, 100, (1, local_packed_len, k))
        mock_topk.return_value = (mock_topk_vals, mock_topk_idx)

        # allgather is called once per sequence (2 sequences x 2 tensors = 4 calls)
        def fake_allgather(local_tensor, group, seq_dim):
            # Simulate gathering: double the seq_dim since cp_size=2
            return local_tensor.repeat(1, cp_size, 1)

        mock_allgather.side_effect = fake_allgather

        cu_seqlens_padded = torch.tensor([0, seq1_len, total_packed_len])

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            cu_seqlens_padded=cu_seqlens_padded,
            original_seq_length=unpacked_seqlen,
        )

        output_tensor = torch.randn(1, local_packed_len, 100)
        loss, result = wrapped_fn(output_tensor)

        # allgather called 2x per sequence (vals + idx) x 2 sequences = 4 calls
        assert mock_allgather.call_count == 4
        assert "topk_logits" in result
        assert "topk_indices" in result
        # Output should be unpacked: (batch_size=2, unpacked_seqlen=6, k=3)
        assert result["topk_logits"].shape == (2, unpacked_seqlen, k)
        assert result["topk_indices"].shape == (2, unpacked_seqlen, k)


class TestAggregateTrainingStatistics:
    """Tests for aggregate_training_statistics function."""

    @patch("torch.distributed.all_reduce")
    def test_materializes_scalar_tensor_metrics_at_reporting_boundary(
        self, mock_all_reduce
    ):
        """Tensor metrics stay on device until one batched host transfer."""
        from nemo_rl.models.megatron.train import aggregate_training_statistics

        all_mb_metrics = [
            {"loss": torch.tensor(0.5), "num_valid_samples": torch.tensor(3.0)},
            {"loss": torch.tensor(0.3), "num_valid_samples": torch.tensor(5.0)},
        ]

        mb_metrics, global_loss = aggregate_training_statistics(
            all_mb_metrics=all_mb_metrics,
            losses=[torch.tensor(0.5), torch.tensor(0.3)],
            data_parallel_group=MagicMock(),
        )

        assert torch.equal(global_loss.cpu(), torch.tensor([0.5, 0.3]))
        assert mb_metrics["loss"] == pytest.approx([0.5, 0.3])
        assert mb_metrics["num_valid_samples"] == [3.0, 5.0]
        assert all(
            type(value) is float for values in mb_metrics.values() for value in values
        )

    @patch("torch.distributed.all_reduce")
    def test_aggregates_metrics_across_microbatches(self, mock_all_reduce):
        """Test that per-microbatch metrics are collected into lists by key."""
        from nemo_rl.models.megatron.train import aggregate_training_statistics

        all_mb_metrics = [
            {"loss": 0.5, "lr": 1e-4},
            {"loss": 0.3, "lr": 1e-4},
            {"loss": 0.2, "lr": 1e-4},
        ]

        mock_dp_group = MagicMock()

        mb_metrics, _ = aggregate_training_statistics(
            all_mb_metrics=all_mb_metrics,
            losses=[1.0],
            data_parallel_group=mock_dp_group,
        )

        assert mb_metrics["loss"] == [0.5, 0.3, 0.2]
        assert mb_metrics["lr"] == [1e-4, 1e-4, 1e-4]
        assert len(mb_metrics) == 2

    @patch("torch.distributed.all_reduce")
    def test_returns_plain_dict(self, mock_all_reduce):
        """Test that the returned mb_metrics is a plain dict, not defaultdict."""
        from nemo_rl.models.megatron.train import aggregate_training_statistics

        mb_metrics, _ = aggregate_training_statistics(
            all_mb_metrics=[{"loss": 0.5}],
            losses=[1.0],
            data_parallel_group=MagicMock(),
        )

        assert type(mb_metrics) is dict

    @patch("torch.distributed.all_reduce")
    def test_global_loss_tensor_from_losses(self, mock_all_reduce):
        """Test that losses list is converted to a CUDA tensor for all-reduce."""
        from nemo_rl.models.megatron.train import aggregate_training_statistics

        mock_dp_group = MagicMock()

        _, global_loss = aggregate_training_statistics(
            all_mb_metrics=[],
            losses=[0.5, 0.3, 0.2],
            data_parallel_group=mock_dp_group,
        )

        # Verify all_reduce was called with correct args
        mock_all_reduce.assert_called_once()
        call_args = mock_all_reduce.call_args
        assert call_args[1]["op"] == torch.distributed.ReduceOp.SUM
        assert call_args[1]["group"] is mock_dp_group

        # Verify tensor shape matches losses list
        reduced_tensor = call_args[0][0]
        assert reduced_tensor.shape == (3,)

    @patch("torch.distributed.all_reduce")
    def test_empty_metrics(self, mock_all_reduce):
        """Test with empty microbatch metrics list."""
        from nemo_rl.models.megatron.train import aggregate_training_statistics

        mb_metrics, global_loss = aggregate_training_statistics(
            all_mb_metrics=[],
            losses=[1.0],
            data_parallel_group=MagicMock(),
        )

        assert mb_metrics == {}
        mock_all_reduce.assert_called_once()

    @patch("torch.distributed.all_reduce")
    def test_handles_heterogeneous_metric_keys(self, mock_all_reduce):
        """Test that microbatches with different metric keys are handled correctly."""
        from nemo_rl.models.megatron.train import aggregate_training_statistics

        all_mb_metrics = [
            {"loss": 0.5, "lr": 1e-4},
            {"loss": 0.3, "global_valid_seqs": 8},
        ]

        mb_metrics, _ = aggregate_training_statistics(
            all_mb_metrics=all_mb_metrics,
            losses=[0.8],
            data_parallel_group=MagicMock(),
        )

        assert mb_metrics["loss"] == [0.5, 0.3]
        assert mb_metrics["lr"] == [1e-4]
        assert mb_metrics["global_valid_seqs"] == [8]

    @patch("torch.distributed.all_reduce")
    def test_no_grad_context(self, mock_all_reduce):
        """Test that all-reduce runs under torch.no_grad context."""
        from nemo_rl.models.megatron.train import aggregate_training_statistics

        grad_enabled_during_all_reduce = []

        def capture_grad_state(*args, **kwargs):
            grad_enabled_during_all_reduce.append(torch.is_grad_enabled())

        mock_all_reduce.side_effect = capture_grad_state

        aggregate_training_statistics(
            all_mb_metrics=[],
            losses=[1.0],
            data_parallel_group=MagicMock(),
        )

        assert grad_enabled_during_all_reduce == [False]
