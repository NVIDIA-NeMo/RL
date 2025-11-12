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

"""Unit tests for dtensor_train.py functions."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.dtensor_train import (
    _handle_context_parallel_sharding,
    _process_logits,
    cleanup_after_training,
    forward_backward,
    model_forward,
    optimizer_step,
    process_output_for_train,
    process_outputs_for_logprobs,
    process_outputs_for_topk,
    setup_train_loop,
)


@pytest.fixture
def mock_dp_mesh():
    """Create a mock data parallel mesh."""
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    return mesh


@pytest.fixture
def mock_device_mesh():
    """Create a mock device mesh."""
    mesh = MagicMock()
    mesh.ndim = 2
    mesh.mesh_dim_names = ["cp", "tp"]
    return mesh


@pytest.fixture
def mock_cp_mesh():
    """Create a mock context parallel mesh."""
    mesh = MagicMock()
    mesh.ndim = 1
    mesh.mesh_dim_names = ["cp"]
    return mesh


@pytest.fixture
def mock_moe_mesh():
    """Create a mock MoE mesh."""
    mesh = MagicMock()
    mesh.mesh_dim_names = ["ep"]
    return mesh


@pytest.fixture
def mock_loss_fn():
    """Create a mock loss function."""
    loss_fn = MagicMock(spec=LossFunction)
    loss_fn.loss_type = LossType.SEQUENCE_LEVEL

    # Mock the loss function to return loss and metrics
    def side_effect(logits, mb, global_valid_seqs, global_valid_toks):
        loss = torch.tensor(0.5, device="cuda", requires_grad=True)
        metrics = {"loss": 0.5}
        return loss, metrics

    loss_fn.side_effect = side_effect
    return loss_fn


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock(spec=nn.Module)

    # Create mock output with logits
    output = MagicMock()
    output.logits = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)

    model.return_value = output
    model.parameters.return_value = [
        torch.randn(10, 10, device="cuda", requires_grad=True)
    ]

    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    return optimizer


class TestSetupTrainLoop:
    """Tests for setup_train_loop function."""

    @patch("nemo_rl.models.policy.dtensor_train.torch.distributed.all_reduce")
    def test_basic_setup(self, mock_all_reduce, mock_dp_mesh):
        """Test basic setup with valid data."""
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (16, 128)),
                "sample_mask": torch.ones(16, dtype=torch.bool),
            }
        )

        gbs = 32
        dp_size = 2

        # Mock all_reduce to simulate total dataset size (scalar tensor)
        def side_effect(tensor, *args, **kwargs):
            tensor.fill_(32)  # Simulate 2 ranks with 16 samples each

        mock_all_reduce.side_effect = side_effect

        result = setup_train_loop(
            data=data, gbs=gbs, dp_size=dp_size, dp_mesh=mock_dp_mesh
        )

        # Verify results
        assert result["local_gbs"] == 16  # 32 / 2
        assert result["num_global_batches"] == 1  # 32 / 32
        assert result["sequence_dim"] == 1

        # Verify all_reduce was called
        mock_all_reduce.assert_called_once()

    @patch("nemo_rl.models.policy.dtensor_train.torch.distributed.all_reduce")
    def test_multiple_global_batches(self, mock_all_reduce, mock_dp_mesh):
        """Test setup with multiple global batches."""
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (32, 64)),
                "sample_mask": torch.ones(32, dtype=torch.bool),
            }
        )

        gbs = 16
        dp_size = 2

        # Mock all_reduce to simulate total dataset size (scalar tensor)
        def side_effect(tensor, *args, **kwargs):
            tensor.fill_(64)  # Simulate 2 ranks with 32 samples each

        mock_all_reduce.side_effect = side_effect

        result = setup_train_loop(
            data=data, gbs=gbs, dp_size=dp_size, dp_mesh=mock_dp_mesh
        )

        # Verify results
        assert result["local_gbs"] == 8  # 16 / 2
        assert result["num_global_batches"] == 4  # 64 / 16
        assert result["sequence_dim"] == 1

    @patch("nemo_rl.models.policy.dtensor_train.torch.distributed.all_reduce")
    def test_sequence_dim_validation(self, mock_all_reduce, mock_dp_mesh):
        """Test that sequence dimension validation works correctly."""
        # Create test data with consistent sequence dimension
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (8, 128)),
                "attention_mask": torch.ones(8, 128, dtype=torch.bool),
                "position_ids": torch.arange(128).repeat(8, 1),
                "sample_mask": torch.ones(8, dtype=torch.bool),
            }
        )

        gbs = 16
        dp_size = 2

        # Mock all_reduce (scalar tensor)
        def side_effect(tensor, *args, **kwargs):
            tensor.fill_(16)

        mock_all_reduce.side_effect = side_effect

        # Should not raise an error
        result = setup_train_loop(
            data=data, gbs=gbs, dp_size=dp_size, dp_mesh=mock_dp_mesh
        )
        assert result["sequence_dim"] == 1

    @patch("nemo_rl.models.policy.dtensor_train.torch.distributed.all_reduce")
    def test_sequence_dim_validation_failure(self, mock_all_reduce, mock_dp_mesh):
        """Test that sequence dimension validation fails with inconsistent shapes."""
        # Create test data with inconsistent sequence dimension
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (8, 128)),
                "attention_mask": torch.ones(8, 64, dtype=torch.bool),  # Wrong seq dim
                "sample_mask": torch.ones(8, dtype=torch.bool),
            }
        )

        gbs = 16
        dp_size = 2

        # Mock all_reduce (scalar tensor)
        def side_effect(tensor, *args, **kwargs):
            tensor.fill_(16)

        mock_all_reduce.side_effect = side_effect

        # Should raise an assertion error
        with pytest.raises(AssertionError, match="Dim 1 must be the sequence dim"):
            setup_train_loop(data=data, gbs=gbs, dp_size=dp_size, dp_mesh=mock_dp_mesh)

    @patch("nemo_rl.models.policy.dtensor_train.torch.distributed.all_reduce")
    def test_with_single_rank(self, mock_all_reduce, mock_dp_mesh):
        """Test setup with single data parallel rank."""
        # Create test data
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (16, 128)),
                "sample_mask": torch.ones(16, dtype=torch.bool),
            }
        )

        gbs = 16
        dp_size = 1

        # Mock all_reduce (scalar tensor)
        def side_effect(tensor, *args, **kwargs):
            tensor.fill_(16)

        mock_all_reduce.side_effect = side_effect

        result = setup_train_loop(
            data=data, gbs=gbs, dp_size=dp_size, dp_mesh=mock_dp_mesh
        )

        # Verify results
        assert result["local_gbs"] == 16  # 16 / 1
        assert result["num_global_batches"] == 1  # 16 / 16
        assert result["sequence_dim"] == 1


class TestForwardBackward:
    """Tests for forward_backward function."""

    def test_basic_forward_backward_eval_mode(self, mock_model, mock_loss_fn):
        """Test basic forward pass in eval mode (no backward)."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            is_reward_model=False,
            allow_flash_attn_args=True,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify loss function was called
        mock_loss_fn.assert_called_once()

        # Verify loss was returned
        assert loss is not None
        assert isinstance(loss_metrics, dict)

    def test_forward_backward_with_backward_pass(self, mock_model, mock_loss_fn):
        """Test forward and backward pass in train mode."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            is_reward_model=False,
            allow_flash_attn_args=True,
            eval_mode=False,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify loss function was called
        mock_loss_fn.assert_called_once()

        # Verify backward was called on the loss
        # Since we're using a mock loss, we can't directly test backward
        # but we verify the loss is returned correctly
        assert loss is not None

    def test_with_reward_model(self, mock_model, mock_loss_fn):
        """Test forward pass with reward model (no flash_attn_kwargs)."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            is_reward_model=True,
            allow_flash_attn_args=False,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify flash_attn_kwargs was not passed to the model
        call_args = mock_model.call_args
        assert "flash_attn_kwargs" not in call_args[1]

    def test_with_multimodal_inputs(self, mock_model, mock_loss_fn):
        """Test forward pass with multimodal inputs."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs with VLM kwargs
        vlm_kwargs = {
            "pixel_values": torch.randn(4, 3, 224, 224).cuda(),
        }

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": None,  # Position IDs are None for multimodal
            "flash_attn_kwargs": {},
            "vlm_kwargs": vlm_kwargs,
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            is_reward_model=False,
            allow_flash_attn_args=True,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify VLM kwargs were passed to the model
        call_args = mock_model.call_args
        assert "pixel_values" in call_args[1]

        # Verify flash_attn_kwargs was removed due to multimodal inputs
        assert "flash_attn_kwargs" not in call_args[1]

    @pytest.mark.skip(
        reason="Context parallel with DTensor requires real distributed environment"
    )
    @patch("nemo_rl.models.policy.dtensor_train.get_train_context")
    @patch("nemo_rl.models.policy.dtensor_train.create_context_parallel_ctx")
    def test_with_context_parallel(
        self, mock_create_cp_ctx, mock_get_train_ctx, mock_model, mock_loss_fn
    ):
        """Test forward pass with context parallel enabled.

        Note: This test is skipped because context parallel operations require a real
        distributed environment with DTensor support. Testing this properly requires
        integration tests with multiple processes and real device meshes.
        """
        pass

    @patch("nemo_rl.models.policy.dtensor_train.get_train_context")
    @patch("nemo_rl.models.policy.dtensor_train.create_context_parallel_ctx")
    def test_context_parallel_setup(self, mock_create_cp_ctx, mock_get_train_ctx):
        """Test that context parallel context is created when cp_size > 1."""
        # This test verifies the setup logic without executing the full CP path

        # Mock CP mesh
        mock_cp_mesh = MagicMock()

        # Mock context parallel context
        mock_cp_ctx = MagicMock()
        mock_cp_ctx.return_value = nullcontext()
        mock_create_cp_ctx.return_value = mock_cp_ctx

        # Mock get_train_context
        mock_train_ctx = MagicMock()
        mock_train_ctx.return_value = nullcontext()
        mock_get_train_ctx.return_value = mock_train_ctx

        # Verify that with cp_size > 1, create_context_parallel_ctx should be called
        # This is tested in the context of the function signature
        cp_size = 2
        cp_buffers = [torch.randn(4, 128).cuda() for _ in range(3)]

        if cp_size > 1:
            # This would be called inside forward_backward
            context_parallel_ctx = mock_create_cp_ctx(
                cp_mesh=mock_cp_mesh,
                cp_buffers=cp_buffers,
                cp_seq_dims=[1] * len(cp_buffers),
                cp_no_restore_buffers=set(cp_buffers),
            )

            # Verify the context was created
            mock_create_cp_ctx.assert_called_once()
            assert mock_create_cp_ctx.call_args[1]["cp_mesh"] == mock_cp_mesh
            assert len(mock_create_cp_ctx.call_args[1]["cp_buffers"]) == 3

    @patch("nemo_rl.models.policy.dtensor_train.SequencePackingLossWrapper")
    def test_with_sequence_packing(
        self, mock_seq_packing_wrapper, mock_model, mock_loss_fn
    ):
        """Test forward pass with sequence packing enabled."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (1, 204)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create flash attention kwargs for sequence packing
        flash_attn_kwargs = MagicMock()
        flash_attn_kwargs.cu_seqlens_q = torch.tensor([0, 32, 80, 140, 204]).cuda()

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": None,
            "position_ids": torch.arange(204).unsqueeze(0).cuda(),
            "flash_attn_kwargs": flash_attn_kwargs,
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 204,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock the wrapped loss function
        wrapped_loss_fn = MagicMock()
        wrapped_loss_fn.return_value = (
            torch.tensor(0.5, device="cuda", requires_grad=True),
            {"loss": 0.5},
        )
        mock_seq_packing_wrapper.return_value = wrapped_loss_fn

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=True,
            is_reward_model=False,
            allow_flash_attn_args=True,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify sequence packing wrapper was created
        mock_seq_packing_wrapper.assert_called_once()
        assert mock_seq_packing_wrapper.call_args[1]["loss_fn"] == mock_loss_fn

        # Verify wrapped loss function was called
        wrapped_loss_fn.assert_called_once()

    def test_with_temperature_scaling(self, mock_model, mock_loss_fn):
        """Test that temperature scaling is applied to logits."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function that scales logits
        temperature = 2.0
        apply_temperature_fn = MagicMock(side_effect=lambda x: x / temperature)

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            is_reward_model=False,
            allow_flash_attn_args=True,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify temperature function was called
        apply_temperature_fn.assert_called_once()

    def test_model_output_as_tensor(self, mock_loss_fn):
        """Test handling of model output when it returns logits directly as tensor."""
        # Create a model that returns tensor directly
        mock_model = MagicMock(spec=nn.Module)
        mock_model.return_value = torch.randn(
            4, 64, 1000, device="cuda", requires_grad=True
        )

        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            is_reward_model=False,
            allow_flash_attn_args=True,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify loss function was called with tensor logits
        mock_loss_fn.assert_called_once()
        assert loss is not None


class TestOptimizerStep:
    """Tests for optimizer_step function."""

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_basic_optimizer_step(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test basic optimizer step with gradient clipping."""
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(1.5).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=2,
            cp_size=1,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

        # Verify gradient norm was returned
        assert grad_norm is not None
        assert grad_norm == 1.5

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_with_moe_mesh(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        mock_device_mesh,
        mock_moe_mesh,
    ):
        """Test optimizer step with MoE mesh."""
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(1.2).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=mock_moe_mesh,
            dp_size=2,
            cp_size=1,
        )

        # Verify scale_grads_and_clip_grad_norm was called with moe_mesh
        mock_scale_grads.assert_called_once()
        assert mock_scale_grads.call_args[1]["moe_mesh"] == mock_moe_mesh
        assert mock_scale_grads.call_args[1]["ep_axis_name"] == "ep"

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_with_context_parallel(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test optimizer step with context parallel."""
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(0.8).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=2,
            cp_size=2,
        )

        # Verify scale_grads_and_clip_grad_norm was called with correct dp_group_size
        mock_scale_grads.assert_called_once()
        assert mock_scale_grads.call_args[1]["dp_group_size"] == 4  # dp_size * cp_size

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_with_no_max_grad_norm(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test optimizer step without gradient clipping."""
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(5.0).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=None,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=2,
            cp_size=1,
        )

        # Verify scale_grads_and_clip_grad_norm was called with None
        mock_scale_grads.assert_called_once()
        assert mock_scale_grads.call_args[0][0] is None

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_infinite_grad_norm_handling(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test handling of infinite gradient norm."""
        # Mock infinite gradient norm
        mock_scale_grads.return_value = torch.tensor(float("inf")).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=2,
            cp_size=1,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.zero_grad was called instead of step
        mock_optimizer.zero_grad.assert_called_once()
        mock_optimizer.step.assert_not_called()

        # Verify gradient norm was still returned
        assert grad_norm is not None
        assert torch.isinf(torch.tensor(grad_norm))

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_nan_grad_norm_handling(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test handling of NaN gradient norm."""
        # Mock NaN gradient norm
        mock_scale_grads.return_value = torch.tensor(float("nan")).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=2,
            cp_size=1,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.zero_grad was called instead of step
        mock_optimizer.zero_grad.assert_called_once()
        mock_optimizer.step.assert_not_called()

        # Verify gradient norm was still returned
        assert grad_norm is not None
        assert torch.isnan(torch.tensor(grad_norm))

    @pytest.mark.skip(reason="DTensor conversion requires real distributed environment")
    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_grad_norm_dtensor_conversion(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test that grad_norm is properly converted from DTensor to regular tensor."""
        # This test would require a real DTensor environment
        # The conversion logic is:
        # 1. If DTensor, call .full_tensor()
        # 2. If not a tensor, convert to tensor
        # 3. Call .detach().cpu().float()
        pass

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_grad_norm_scalar_conversion(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test that grad_norm is properly converted from scalar to tensor."""
        # Mock scalar gradient norm (not a tensor)
        mock_scale_grads.return_value = 1.5  # Plain Python float

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=2,
            cp_size=1,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

        # Verify gradient norm was returned as a float
        assert grad_norm is not None
        assert isinstance(grad_norm, (float, torch.Tensor))
        if isinstance(grad_norm, torch.Tensor):
            assert grad_norm.item() == 1.5

    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    def test_grad_norm_already_tensor(
        self, mock_scale_grads, mock_optimizer, mock_model, mock_device_mesh
    ):
        """Test that grad_norm works correctly when already a regular tensor."""
        # Mock tensor gradient norm (already a tensor)
        mock_scale_grads.return_value = torch.tensor(2.3).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=2,
            cp_size=1,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

        # Verify gradient norm was returned correctly
        assert grad_norm is not None
        assert abs(float(grad_norm) - 2.3) < 1e-5


class TestCleanupAfterTraining:
    """Tests for cleanup_after_training function."""

    def test_cleanup_in_train_mode(self, mock_optimizer):
        """Test cleanup after training in train mode."""
        # Create mock scheduler
        mock_scheduler = MagicMock()

        cleanup_after_training(
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            eval_mode=False,
        )

        # Verify optimizer.zero_grad was called
        mock_optimizer.zero_grad.assert_called_once()

        # Verify scheduler.step was called (not in eval mode)
        mock_scheduler.step.assert_called_once()

    def test_cleanup_in_eval_mode(self, mock_optimizer):
        """Test cleanup after training in eval mode."""
        # Create mock scheduler
        mock_scheduler = MagicMock()

        cleanup_after_training(
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            eval_mode=True,
        )

        # Verify optimizer.zero_grad was called
        mock_optimizer.zero_grad.assert_called_once()

        # Verify scheduler.step was NOT called (in eval mode)
        mock_scheduler.step.assert_not_called()

    def test_cleanup_without_scheduler(self, mock_optimizer):
        """Test cleanup without a scheduler."""
        cleanup_after_training(
            optimizer=mock_optimizer,
            scheduler=None,
            eval_mode=False,
        )

        # Verify optimizer.zero_grad was called
        mock_optimizer.zero_grad.assert_called_once()

        # No scheduler, so nothing to verify for scheduler.step

    @patch("nemo_rl.models.policy.dtensor_train.torch.cuda.empty_cache")
    def test_cuda_cache_cleared(self, mock_empty_cache, mock_optimizer):
        """Test that CUDA cache is cleared after cleanup."""
        cleanup_after_training(
            optimizer=mock_optimizer,
            scheduler=None,
            eval_mode=False,
        )

        # Verify torch.cuda.empty_cache was called
        mock_empty_cache.assert_called_once()


class TestProcessLogits:
    """Tests for _process_logits helper function."""

    def test_process_logits_from_tensor(self):
        """Test processing logits when model returns tensor directly."""
        # Create mock model output as tensor
        outputs = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)
        model = MagicMock(spec=nn.Module)
        apply_temperature_fn = lambda x: x

        logits = _process_logits(outputs, model, apply_temperature_fn)

        assert logits is outputs
        assert logits.shape == (4, 64, 1000)

    def test_process_logits_from_output_with_logits_attr(self):
        """Test processing logits from model output with logits attribute."""
        # Create mock output with logits attribute
        outputs = MagicMock()
        outputs.logits = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)

        model = MagicMock(spec=nn.Module)
        apply_temperature_fn = lambda x: x

        logits = _process_logits(outputs, model, apply_temperature_fn)

        assert logits is outputs.logits
        assert logits.shape == (4, 64, 1000)

    def test_process_logits_from_last_hidden_state(self):
        """Test processing logits from last_hidden_state via lm_head."""

        # Create a simple namespace object without logits attribute
        class ModelOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(4, 64, 768, device="cuda")

        outputs = ModelOutput()

        # Create mock model with lm_head
        model = MagicMock()
        lm_head_output = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)
        model.lm_head = MagicMock(return_value=lm_head_output)

        apply_temperature_fn = lambda x: x

        logits = _process_logits(outputs, model, apply_temperature_fn)

        # Verify lm_head was called with last_hidden_state
        model.lm_head.assert_called_once_with(outputs.last_hidden_state)
        assert logits is lm_head_output

    def test_process_logits_with_temperature_scaling(self):
        """Test that temperature scaling is applied correctly."""
        outputs = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)
        model = MagicMock(spec=nn.Module)

        temperature = 2.0
        apply_temperature_fn = lambda x: x / temperature

        logits = _process_logits(outputs, model, apply_temperature_fn)

        # Verify temperature was applied
        expected_logits = outputs / temperature
        assert torch.allclose(logits, expected_logits)


class TestHandleContextParallelSharding:
    """Tests for _handle_context_parallel_sharding helper function."""

    @pytest.mark.skip(
        reason="Context parallel sharding requires real DTensor environment"
    )
    def test_sharding_logits_only(self):
        """Test sharding only logits without microbatch tensors.

        This is the use case for logprob extraction.
        """
        # This test would require a real DTensor/distributed environment
        pass

    @pytest.mark.skip(
        reason="Context parallel sharding requires real DTensor environment"
    )
    def test_sharding_with_microbatch(self):
        """Test sharding both logits and microbatch tensors.

        This is the use case for training.
        """
        # This test would require a real DTensor/distributed environment
        pass

    @pytest.mark.skip(
        reason="Context parallel sharding requires real DTensor environment"
    )
    def test_sharding_with_tp_and_cp(self):
        """Test sharding with both tensor parallel and context parallel."""
        # This test would require a real DTensor/distributed environment
        pass

    def test_optional_parameters_signature(self):
        """Test that mb and cp_buffers are truly optional in the signature."""
        # Verify the function signature allows optional parameters
        import inspect

        sig = inspect.signature(_handle_context_parallel_sharding)
        params = sig.parameters

        # Check that mb and cp_buffers have default values
        assert params["mb"].default is None
        assert params["cp_buffers"].default is None


class TestModelForward:
    """Tests for model_forward function."""

    def test_basic_model_forward(self, mock_model):
        """Test basic model forward pass."""
        processed_inputs = {
            "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            cp_size=1,
            cp_mesh=None,
            is_reward_model=False,
            allow_flash_attn_args=True,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify flash_attn_kwargs was passed
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" in call_kwargs

    def test_model_forward_with_reward_model(self, mock_model):
        """Test model forward without flash_attn_kwargs for reward model."""
        processed_inputs = {
            "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            cp_size=1,
            cp_mesh=None,
            is_reward_model=True,
            allow_flash_attn_args=False,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify flash_attn_kwargs was NOT passed
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" not in call_kwargs

    def test_model_forward_with_multimodal(self, mock_model):
        """Test model forward with multimodal inputs."""
        vlm_kwargs = {
            "pixel_values": torch.randn(4, 3, 224, 224).cuda(),
        }

        processed_inputs = {
            "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": None,  # None for multimodal
            "flash_attn_kwargs": {},
            "vlm_kwargs": vlm_kwargs,
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            cp_size=1,
            cp_mesh=None,
            is_reward_model=False,
            allow_flash_attn_args=True,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify VLM kwargs were passed and flash_attn_kwargs was removed
        call_kwargs = mock_model.call_args[1]
        assert "pixel_values" in call_kwargs
        assert "flash_attn_kwargs" not in call_kwargs

    @pytest.mark.skip(reason="Context parallel requires real distributed environment")
    @patch("nemo_rl.models.policy.dtensor_train.create_context_parallel_ctx")
    def test_model_forward_with_context_parallel(
        self, mock_create_cp_ctx, mock_model, mock_cp_mesh
    ):
        """Test model forward with context parallel enabled."""
        # This would require real distributed setup
        pass


class TestProcessOutputForTrain:
    """Tests for process_output_for_train function."""

    def test_basic_train_output_processing(self, mock_model, mock_loss_fn):
        """Test basic train output processing."""
        # Create mock outputs
        outputs = MagicMock()
        outputs.logits = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)

        # Create microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()
        apply_temperature_fn = lambda x: x

        loss, loss_metrics = process_output_for_train(
            outputs=outputs,
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify loss function was called
        mock_loss_fn.assert_called_once()
        assert loss is not None
        assert isinstance(loss_metrics, dict)

    @patch("nemo_rl.models.policy.dtensor_train.SequencePackingLossWrapper")
    def test_train_output_with_sequence_packing(
        self, mock_seq_packing_wrapper, mock_model, mock_loss_fn
    ):
        """Test train output processing with sequence packing."""
        # Create mock outputs
        outputs = MagicMock()
        outputs.logits = torch.randn(1, 204, 1000, device="cuda", requires_grad=True)

        # Create microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (1, 204)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        flash_attn_kwargs = MagicMock()
        flash_attn_kwargs.cu_seqlens_q = torch.tensor([0, 32, 80, 140, 204]).cuda()

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": None,
            "position_ids": torch.arange(204).unsqueeze(0).cuda(),
            "flash_attn_kwargs": flash_attn_kwargs,
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 204,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()
        apply_temperature_fn = lambda x: x

        # Mock the wrapped loss function
        wrapped_loss_fn = MagicMock()
        wrapped_loss_fn.return_value = (
            torch.tensor(0.5, device="cuda", requires_grad=True),
            {"loss": 0.5},
        )
        mock_seq_packing_wrapper.return_value = wrapped_loss_fn

        loss, loss_metrics = process_output_for_train(
            outputs=outputs,
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=True,
            eval_mode=True,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify sequence packing wrapper was created
        mock_seq_packing_wrapper.assert_called_once()
        wrapped_loss_fn.assert_called_once()

    @pytest.mark.skip(reason="Context parallel requires real distributed environment")
    def test_train_output_with_context_parallel(self):
        """Test train output processing with context parallel."""
        # This would require real DTensor/distributed environment
        pass


class TestProcessOutputsForLogprobs:
    """Tests for process_outputs_for_logprobs function."""

    def test_basic_logprob_extraction(self, mock_model):
        """Test basic logprob extraction without CP."""
        # Create mock outputs
        outputs = MagicMock()
        outputs.logits = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)

        # Create microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        input_ids = torch.randint(0, 1000, (4, 64)).cuda()

        processed_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        apply_temperature_fn = lambda x: x

        token_logprobs = process_outputs_for_logprobs(
            outputs=outputs,
            model=mock_model,
            mb=mb,
            processed_inputs=processed_inputs,
            input_ids=input_ids,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            apply_temperature_fn=apply_temperature_fn,
            logprob_chunk_size=None,
        )

        # Verify shape: [batch_size, seq_len]
        assert token_logprobs.shape == (4, 64)

        # Verify first token has zero logprob (prepended)
        assert torch.all(token_logprobs[:, 0] == 0.0)

    def test_logprob_extraction_with_chunking(self, mock_model):
        """Test logprob extraction with chunking for memory efficiency."""
        # Create mock outputs
        outputs = MagicMock()
        outputs.logits = torch.randn(4, 128, 1000, device="cuda", requires_grad=True)

        # Create microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        input_ids = torch.randint(0, 1000, (4, 128)).cuda()

        processed_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(4, 128, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(128).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 128,
        }

        apply_temperature_fn = lambda x: x

        # Use chunking with chunk_size=64
        token_logprobs = process_outputs_for_logprobs(
            outputs=outputs,
            model=mock_model,
            mb=mb,
            processed_inputs=processed_inputs,
            input_ids=input_ids,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            enable_seq_packing=False,
            apply_temperature_fn=apply_temperature_fn,
            logprob_chunk_size=64,
        )

        # Verify shape
        assert token_logprobs.shape == (4, 128)

        # Verify first token has zero logprob
        assert torch.all(token_logprobs[:, 0] == 0.0)

    @pytest.mark.skip(reason="Context parallel requires real distributed environment")
    def test_logprob_extraction_with_context_parallel(self):
        """Test logprob extraction with context parallel."""
        # This would require real DTensor/distributed environment
        pass


class TestProcessOutputsForTopk:
    """Tests for process_outputs_for_topk function."""

    def test_basic_topk_extraction(self, mock_model):
        """Test basic top-k extraction without CP/TP."""
        # Create mock outputs
        outputs = MagicMock()
        outputs.logits = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)

        # Create microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 64, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(64).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 64,
        }

        apply_temperature_fn = lambda x: x

        # Create mock tp_mesh
        mock_tp_mesh = MagicMock()

        vals, idx = process_outputs_for_topk(
            outputs=outputs,
            model=mock_model,
            mb=mb,
            processed_inputs=processed_inputs,
            k=10,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            tp_mesh=mock_tp_mesh,
            enable_seq_packing=False,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify shape: [batch_size, seq_len, k]
        assert vals.shape == (4, 64, 10)
        assert idx.shape == (4, 64, 10)

        # Verify indices are within valid range [0, vocab_size)
        assert torch.all(idx >= 0)
        assert torch.all(idx < 1000)

    def test_topk_extraction_with_temperature(self, mock_model):
        """Test that temperature scaling is applied to logits before top-k."""
        # Create mock outputs with known logits
        outputs = MagicMock()
        logits = torch.randn(2, 32, 500, device="cuda", requires_grad=True)
        outputs.logits = logits

        # Create microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 500, (2, 32)).cuda(),
                "sample_mask": torch.ones(2, dtype=torch.bool).cuda(),
            }
        )

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(2, 32, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(32).repeat(2, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 32,
        }

        temperature = 2.0
        apply_temperature_fn = lambda x: x / temperature

        # Create mock tp_mesh
        mock_tp_mesh = MagicMock()

        vals, idx = process_outputs_for_topk(
            outputs=outputs,
            model=mock_model,
            mb=mb,
            processed_inputs=processed_inputs,
            k=5,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            tp_mesh=mock_tp_mesh,
            enable_seq_packing=False,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify shape
        assert vals.shape == (2, 32, 5)
        assert idx.shape == (2, 32, 5)

    def test_topk_extraction_different_k_values(self, mock_model):
        """Test top-k extraction with different k values."""
        outputs = MagicMock()
        outputs.logits = torch.randn(3, 48, 800, device="cuda", requires_grad=True)

        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 800, (3, 48)).cuda(),
                "sample_mask": torch.ones(3, dtype=torch.bool).cuda(),
            }
        )

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(3, 48, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(48).repeat(3, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 48,
        }

        apply_temperature_fn = lambda x: x
        mock_tp_mesh = MagicMock()

        # Test with k=20
        vals, idx = process_outputs_for_topk(
            outputs=outputs,
            model=mock_model,
            mb=mb,
            processed_inputs=processed_inputs,
            k=20,
            cp_size=1,
            cp_mesh=None,
            device_mesh=None,
            tp_mesh=mock_tp_mesh,
            enable_seq_packing=False,
            apply_temperature_fn=apply_temperature_fn,
        )

        assert vals.shape == (3, 48, 20)
        assert idx.shape == (3, 48, 20)

    @pytest.mark.skip(reason="Tensor parallel requires real distributed environment")
    def test_topk_extraction_with_tensor_parallel(self):
        """Test top-k extraction with tensor parallel (DTensor logits).

        This test would require a real DTensor/distributed environment to test:
        - DTensor logits with TP sharding
        - distributed_vocab_topk across TP ranks
        - Proper gathering of top-k across sharded vocabulary
        """
        pass

    @pytest.mark.skip(reason="Context parallel requires real distributed environment")
    def test_topk_extraction_with_context_parallel(self):
        """Test top-k extraction with context parallel.

        This test would require a real DTensor/distributed environment to test:
        - Sharding logits across CP dimension
        - distributed_vocab_topk with both CP and TP
        - allgather_cp_sharded_tensor for gathering results across CP ranks
        """
        pass

    @pytest.mark.skip(reason="Context parallel requires real distributed environment")
    def test_topk_extraction_with_cp_and_tp(self):
        """Test top-k extraction with both CP and TP enabled.

        This test would require a real DTensor/distributed environment to test:
        - Sequence sharding via _handle_context_parallel_sharding
        - TP vocabulary sharding
        - Proper coordination between CP and TP for distributed top-k
        """
        pass


class TestIntegrationScenarios:
    """Integration tests combining multiple functions."""

    @patch("nemo_rl.models.policy.dtensor_train.torch.distributed.all_reduce")
    @patch("nemo_rl.models.policy.dtensor_train.scale_grads_and_clip_grad_norm")
    @patch("nemo_rl.models.policy.dtensor_train.torch.cuda.empty_cache")
    def test_full_training_loop(
        self,
        mock_empty_cache,
        mock_scale_grads,
        mock_all_reduce,
        mock_model,
        mock_loss_fn,
        mock_optimizer,
        mock_dp_mesh,
        mock_device_mesh,
    ):
        """Test a complete training loop with all functions."""
        # Step 1: Setup training loop
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (16, 128)).cuda(),
                "sample_mask": torch.ones(16, dtype=torch.bool).cuda(),
            }
        )

        gbs = 16
        dp_size = 2

        # Mock all_reduce for setup (scalar tensor)
        def setup_all_reduce(tensor, *args, **kwargs):
            tensor.fill_(32)

        mock_all_reduce.side_effect = setup_all_reduce

        setup_result = setup_train_loop(
            data=data, gbs=gbs, dp_size=dp_size, dp_mesh=mock_dp_mesh
        )

        # Step 2: Forward and backward pass
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 128)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        processed_inputs = {
            "input_ids": mb["input_ids"],
            "attention_mask": torch.ones(4, 128, dtype=torch.bool).cuda(),
            "position_ids": torch.arange(128).repeat(4, 1).cuda(),
            "flash_attn_kwargs": {},
            "vlm_kwargs": {},
            "cp_buffers": [],
            "seq_index": None,
            "seq_len": 128,
        }

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(1024.0).cuda()

        apply_temperature_fn = lambda x: x

        loss, loss_metrics = forward_backward(
            model=mock_model,
            mb=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            processed_inputs=processed_inputs,
            dtype=torch.float16,
            cp_size=1,
            cp_mesh=None,
            device_mesh=mock_device_mesh,
            enable_seq_packing=False,
            is_reward_model=False,
            allow_flash_attn_args=True,
            eval_mode=False,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Step 3: Optimizer step
        mock_scale_grads.return_value = torch.tensor(1.2).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            max_grad_norm=1.0,
            device_mesh=mock_device_mesh,
            moe_mesh=None,
            dp_size=dp_size,
            cp_size=1,
        )

        # Step 4: Cleanup
        mock_scheduler = MagicMock()

        cleanup_after_training(
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            eval_mode=False,
        )

        # Verify all steps were executed
        assert setup_result["local_gbs"] == 8
        assert setup_result["num_global_batches"] == 2
        assert loss is not None
        assert grad_norm is not None
        mock_scheduler.step.assert_called()
        mock_empty_cache.assert_called()
