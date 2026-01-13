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

from unittest.mock import MagicMock, patch

import pytest
import torch


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
        cfg = {}

        result = model_forward(
            model=mock_model,
            data_dict=mock_data_dict,
            cfg=cfg,
            input_ids_cp_sharded=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        assert torch.equal(result, mock_output)
        mock_model.assert_called_once()

    def test_model_forward_with_temperature_scaling(self):
        """Test model_forward applies temperature scaling."""
        from nemo_rl.models.megatron.train import model_forward

        mock_model = MagicMock()
        mock_output = torch.ones(2, 10, 100) * 2.0
        mock_model.return_value = mock_output

        mock_data_dict = MagicMock()
        mock_data_dict.get_multimodal_dict.return_value = {}

        cfg = {"generation": {"temperature": 2.0}}

        result = model_forward(
            model=mock_model,
            data_dict=mock_data_dict,
            cfg=cfg,
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3),
        )

        # Temperature scaling divides by 2.0, so 2.0 / 2.0 = 1.0
        assert torch.allclose(result, torch.ones_like(result))

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
            cfg={},
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
            cfg={},
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
            cfg={},
            input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3),
        )

        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["position_ids"] is None


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
            cfg=cfg,
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
        )

        data_iterator = iter([processed_mb])
        cfg = {"sequence_packing": {"enabled": False}}
        post_processor = LogprobsPostProcessor(cfg=cfg)

        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            output, wrapped_fn = forward_with_post_processing_fn(
                data_iterator=data_iterator,
                model=MagicMock(),
                cfg=cfg,
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
        )

        data_iterator = iter([processed_mb])
        cfg = {
            "sequence_packing": {"enabled": False},
            "megatron_cfg": {"context_parallel_size": 1},
        }
        post_processor = TopkLogitsPostProcessor(cfg=cfg, k=5)

        with patch.object(post_processor, "__call__", return_value=MagicMock()):
            output, wrapped_fn = forward_with_post_processing_fn(
                data_iterator=data_iterator,
                model=MagicMock(),
                cfg=cfg,
                post_processing_fn=post_processor,
            )

        mock_model_forward.assert_called_once()

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
                cfg={},
                post_processing_fn=unknown_processor,
            )


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

        result = megatron_forward_backward(
            model=mock_model,
            cfg=cfg,
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
            cfg=cfg,
            data_iterator=iter([]),
            num_microbatches=1,
            seq_length=64,
            mbs=1,
            post_processing_fn=post_processor,
            forward_only=True,
        )

        call_kwargs = mock_fb_func.call_args[1]
        assert call_kwargs["forward_only"] is True


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
        """Test LossPostProcessor with CP normalization."""
        from nemo_rl.models.megatron.train import LossPostProcessor

        mock_loss_fn = MagicMock(return_value=(torch.tensor(1.0), {}))
        cfg = {"sequence_packing": {"enabled": False}}

        processor = LossPostProcessor(loss_fn=mock_loss_fn, cfg=cfg, cp_normalize=True)

        # Set up mock return values for process groups
        mock_tp_grp.return_value = MagicMock()
        mock_cp_grp.return_value = MagicMock()

        wrapped_fn = processor(data_dict=MagicMock())

        output_tensor = torch.randn(2, 10, 100)
        loss, _ = wrapped_fn(output_tensor)

        # Loss should be divided by CP size (2)
        assert torch.isclose(loss, torch.tensor(0.5))

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
            return_value=torch.tensor([[1, 2, 3, 4, 5]])
        )

        mock_logprobs = torch.randn(1, 4)  # One less than input length
        mock_from_logits.return_value = mock_logprobs

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            cu_seqlens_padded=None,
        )

        output_tensor = torch.randn(1, 5, 100)
        loss, result = wrapped_fn(output_tensor)

        # Loss should be 0
        assert loss.item() == 0.0
        # Result should have logprobs key
        assert "logprobs" in result
        # Logprobs should be prepended with a 0
        assert result["logprobs"].shape[1] == 5

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
            side_effect=lambda key: torch.tensor([[1, 2, 3, 4, 5]])
            if key == "input_ids"
            else torch.tensor([5])
        )

        mock_topk_vals = torch.randn(1, 5, k)
        mock_topk_idx = torch.randint(0, 100, (1, 5, k))
        mock_topk.return_value = (mock_topk_vals, mock_topk_idx)

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            cu_seqlens_padded=None,
        )

        output_tensor = torch.randn(1, 5, 100)
        loss, result = wrapped_fn(output_tensor)

        assert "topk_logits" in result
        assert "topk_indices" in result
        assert result["topk_logits"].shape[-1] == k

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
            side_effect=lambda key: torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
            if key == "input_ids"
            else torch.tensor([5])
        )

        mock_topk_vals = torch.randn(1, 8, k)
        mock_topk_idx = torch.randint(0, 100, (1, 8, k))
        mock_topk.return_value = (mock_topk_vals, mock_topk_idx)

        cu_seqlens_padded = torch.tensor([0, 5])

        wrapped_fn = processor(
            data_dict=mock_data_dict,
            cu_seqlens_padded=cu_seqlens_padded,
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
            side_effect=lambda key: torch.tensor([[1, 2, 3]])
            if key == "input_ids"
            else torch.tensor([3])
        )

        mock_topk.return_value = (
            torch.randn(1, 3, 5),
            torch.randint(0, 100, (1, 3, 5)),
        )

        wrapped_fn = processor(data_dict=mock_data_dict, cu_seqlens_padded=None)

        output_tensor = torch.randn(1, 3, 100)

        with pytest.raises(
            RuntimeError, match="Context Parallelism.*requires sequence packing"
        ):
            wrapped_fn(output_tensor)
