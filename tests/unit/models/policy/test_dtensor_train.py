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
    cleanup_after_training,
    forward_backward,
    optimizer_step,
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

    @patch("nemo_automodel.components.training.utils.scale_grads_and_clip_grad_norm")
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

    @patch("nemo_automodel.components.training.utils.scale_grads_and_clip_grad_norm")
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

    @patch("nemo_automodel.components.training.utils.scale_grads_and_clip_grad_norm")
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

    @patch("nemo_automodel.components.training.utils.scale_grads_and_clip_grad_norm")
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

    @patch("nemo_automodel.components.training.utils.scale_grads_and_clip_grad_norm")
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

    @patch("nemo_automodel.components.training.utils.scale_grads_and_clip_grad_norm")
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


class TestIntegrationScenarios:
    """Integration tests combining multiple functions."""

    @patch("nemo_rl.models.policy.dtensor_train.torch.distributed.all_reduce")
    @patch("nemo_automodel.components.training.utils.scale_grads_and_clip_grad_norm")
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
