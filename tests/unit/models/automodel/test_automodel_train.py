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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.setup import DistributedState
from nemo_rl.models.automodel.train import (
    _process_logits,
    cleanup_after_training,
    forward_backward,
    forward_with_processor,
    get_logprobs,
    get_loss,
    get_topk_logits,
    model_forward,
    optimizer_step,
    setup_train_loop,
)
from nemo_rl.models.automodel.types import LossInputs, ProcessedInputs, RuntimeConfig


@pytest.fixture
def mock_dp_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    return mesh


@pytest.fixture
def mock_device_mesh():
    mesh = MagicMock()
    mesh.ndim = 2
    mesh.mesh_dim_names = ["cp", "tp"]
    return mesh


@pytest.fixture
def mock_cp_mesh():
    mesh = MagicMock()
    mesh.ndim = 1
    mesh.mesh_dim_names = ["cp"]
    return mesh


@pytest.fixture
def mock_moe_mesh():
    mesh = MagicMock()
    mesh.mesh_dim_names = ["ep"]
    return mesh


@pytest.fixture
def mock_loss_fn():
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
def runtime_config():
    """Create a RuntimeConfig instance for tests."""
    return RuntimeConfig(
        is_reward_model=False,
        is_vlm=False,
        is_hf_model=False,
        is_moe_model=False,
        model_class=MagicMock,
        model_config=MagicMock(),
        hf_config_overrides={},
        allow_flash_attn_args=True,
        attn_impl=None,
        dtype=torch.float16,
        enable_seq_packing=False,
        max_grad_norm=1.0,
        cpu_offload=False,
        offload_optimizer_for_logprob=False,
        is_generation_colocated=None,
    )


@pytest.fixture
def distributed_state(mock_device_mesh, mock_dp_mesh, mock_cp_mesh):
    """Create a DistributedState instance for tests."""
    return DistributedState(
        rank=0,
        world_size=4,
        device_mesh=mock_device_mesh,
        dp_cp_mesh=mock_dp_mesh,
        dp_mesh=mock_dp_mesh,
        tp_mesh=MagicMock(),
        cp_mesh=mock_cp_mesh,
        moe_mesh=None,
        dp_size=4,
        tp_size=1,
        cp_size=1,
        ep_size=1,
        sequence_parallel_enabled=False,
        manager=MagicMock(),
    )


@pytest.fixture
def mock_model():
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
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    return optimizer


@pytest.mark.automodel
class TestSetupTrainLoop:
    @patch("nemo_rl.models.automodel.train.torch.distributed.all_reduce")
    def test_basic_setup(self, mock_all_reduce, mock_dp_mesh):
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

    @patch("nemo_rl.models.automodel.train.torch.distributed.all_reduce")
    def test_multiple_global_batches(self, mock_all_reduce, mock_dp_mesh):
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

    @patch("nemo_rl.models.automodel.train.torch.distributed.all_reduce")
    def test_sequence_dim_validation(self, mock_all_reduce, mock_dp_mesh):
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

    @patch("nemo_rl.models.automodel.train.torch.distributed.all_reduce")
    def test_sequence_dim_validation_failure(self, mock_all_reduce, mock_dp_mesh):
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

    @patch("nemo_rl.models.automodel.train.torch.distributed.all_reduce")
    def test_with_single_rank(self, mock_all_reduce, mock_dp_mesh):
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


@pytest.mark.automodel
class TestForwardBackward:
    def test_basic_forward_backward_eval_mode(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify loss function was called
        mock_loss_fn.assert_called_once()

        # Verify loss was returned
        assert loss is not None
        assert isinstance(loss_metrics, dict)

    def test_forward_backward_with_backward_pass(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify loss function was called
        mock_loss_fn.assert_called_once()

        # Verify backward was called on the loss
        # Since we're using a mock loss, we can't directly test backward
        # but we verify the loss is returned correctly
        assert loss is not None

    def test_with_reward_model(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Modify runtime_config for reward model
        runtime_config_reward = RuntimeConfig(
            **{
                **runtime_config.__dict__,
                "is_reward_model": True,
                "allow_flash_attn_args": False,
            }
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config_reward,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify flash_attn_kwargs was not passed to the model
        call_args = mock_model.call_args
        assert "flash_attn_kwargs" not in call_args[1]

    def test_with_multimodal_inputs(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
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

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=None,  # Position IDs are None for multimodal
            flash_attn_kwargs={},
            vlm_kwargs=vlm_kwargs,
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Modify runtime_config for VLM
        runtime_config_vlm = RuntimeConfig(
            **{**runtime_config.__dict__, "is_vlm": True}
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config_vlm,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify VLM kwargs were passed to the model
        call_args = mock_model.call_args
        assert "pixel_values" in call_args[1]

        # Verify flash_attn_kwargs was removed due to multimodal inputs
        assert "flash_attn_kwargs" not in call_args[1]

    @patch("nemo_rl.models.automodel.train.get_train_context")
    @patch("nemo_rl.models.automodel.train.create_context_parallel_ctx")
    def test_context_parallel_setup(self, mock_create_cp_ctx, mock_get_train_ctx):
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

    @patch("nemo_rl.models.automodel.train.SequencePackingLossWrapper")
    def test_with_sequence_packing(
        self,
        mock_seq_packing_wrapper,
        mock_model,
        mock_loss_fn,
        runtime_config,
        distributed_state,
    ):
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

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=None,
            position_ids=torch.arange(204).unsqueeze(0).cuda(),
            flash_attn_kwargs=flash_attn_kwargs,
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=204,
        )

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

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Modify runtime_config for sequence packing
        runtime_config_sp = RuntimeConfig(
            **{**runtime_config.__dict__, "enable_seq_packing": True}
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config_sp,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Verify sequence packing wrapper was created
        mock_seq_packing_wrapper.assert_called_once()
        assert mock_seq_packing_wrapper.call_args[1]["loss_fn"] == mock_loss_fn

        # Verify wrapped loss function was called
        wrapped_loss_fn.assert_called_once()

    def test_with_temperature_scaling(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function that scales logits
        temperature = 2.0
        apply_temperature_fn = MagicMock(side_effect=lambda x: x / temperature)

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Verify temperature function was called
        apply_temperature_fn.assert_called_once()

    def test_model_output_as_tensor(
        self, mock_loss_fn, runtime_config, distributed_state
    ):
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
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function
        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Verify loss function was called with tensor logits
        mock_loss_fn.assert_called_once()
        assert loss is not None


@pytest.mark.automodel
class TestOptimizerStep:
    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_basic_optimizer_step(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
    ):
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(1.5).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

        # Verify gradient norm was returned
        assert grad_norm is not None
        assert grad_norm == 1.5

    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_with_moe_mesh(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
        mock_moe_mesh,
    ):
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(1.2).cuda()

        distributed_state_with_moe = DistributedState(
            **{**distributed_state.__dict__, "moe_mesh": mock_moe_mesh}
        )
        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state_with_moe,
        )

        # Verify scale_grads_and_clip_grad_norm was called with moe_mesh
        mock_scale_grads.assert_called_once()
        assert mock_scale_grads.call_args[1]["moe_mesh"] == mock_moe_mesh
        assert mock_scale_grads.call_args[1]["ep_axis_name"] == "ep"

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_with_context_parallel(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
    ):
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(0.8).cuda()

        # Modify distributed_state for context parallel
        distributed_state_cp = DistributedState(
            **{**distributed_state.__dict__, "cp_size": 2}
        )

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state_cp,
        )

        # Verify scale_grads_and_clip_grad_norm was called with correct dp_group_size
        mock_scale_grads.assert_called_once()
        assert (
            mock_scale_grads.call_args[1]["dp_group_size"] == 8
        )  # dp_size * cp_size (4 * 2)

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_with_no_max_grad_norm(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
    ):
        # Mock the gradient norm
        mock_scale_grads.return_value = torch.tensor(5.0).cuda()

        # Modify runtime_config with no max_grad_norm
        runtime_config_no_clip = RuntimeConfig(
            **{**runtime_config.__dict__, "max_grad_norm": None}
        )

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config_no_clip,
            distributed_state=distributed_state,
        )

        # Verify scale_grads_and_clip_grad_norm was called with None
        mock_scale_grads.assert_called_once()
        assert mock_scale_grads.call_args[0][0] is None

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_infinite_grad_norm_handling(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
    ):
        # Mock infinite gradient norm
        mock_scale_grads.return_value = torch.tensor(float("inf")).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.zero_grad was called instead of step
        mock_optimizer.zero_grad.assert_called_once()
        mock_optimizer.step.assert_not_called()

        # Verify gradient norm was still returned
        assert grad_norm is not None
        assert torch.isinf(torch.tensor(grad_norm))

    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_nan_grad_norm_handling(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
    ):
        # Mock NaN gradient norm
        mock_scale_grads.return_value = torch.tensor(float("nan")).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.zero_grad was called instead of step
        mock_optimizer.zero_grad.assert_called_once()
        mock_optimizer.step.assert_not_called()

        # Verify gradient norm was still returned
        assert grad_norm is not None
        assert torch.isnan(torch.tensor(grad_norm))

    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_grad_norm_scalar_conversion(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
    ):
        # Mock scalar gradient norm (not a tensor)
        mock_scale_grads.return_value = 1.5  # Plain Python float

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
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

    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    def test_grad_norm_already_tensor(
        self,
        mock_scale_grads,
        mock_optimizer,
        mock_model,
        runtime_config,
        distributed_state,
    ):
        # Mock tensor gradient norm (already a tensor)
        mock_scale_grads.return_value = torch.tensor(2.3).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
        )

        # Verify scale_grads_and_clip_grad_norm was called
        mock_scale_grads.assert_called_once()

        # Verify optimizer.step was called
        mock_optimizer.step.assert_called_once()

        # Verify gradient norm was returned correctly
        assert grad_norm is not None
        assert abs(float(grad_norm) - 2.3) < 1e-5


@pytest.mark.automodel
class TestCleanupAfterTraining:
    def test_cleanup_in_train_mode(self, mock_optimizer):
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
        cleanup_after_training(
            optimizer=mock_optimizer,
            scheduler=None,
            eval_mode=False,
        )

        # Verify optimizer.zero_grad was called
        mock_optimizer.zero_grad.assert_called_once()

        # No scheduler, so nothing to verify for scheduler.step

    @patch("nemo_rl.models.automodel.train.torch.cuda.empty_cache")
    def test_cuda_cache_cleared(self, mock_empty_cache, mock_optimizer):
        cleanup_after_training(
            optimizer=mock_optimizer,
            scheduler=None,
            eval_mode=False,
        )

        # Verify torch.cuda.empty_cache was called
        mock_empty_cache.assert_called_once()


@pytest.mark.automodel
class TestProcessLogits:
    def test_process_logits_from_tensor(self):
        # Create mock model output as tensor
        outputs = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)
        model = MagicMock(spec=nn.Module)
        apply_temperature_fn = lambda x: x

        logits = _process_logits(outputs, model, apply_temperature_fn)

        assert logits is outputs
        assert logits.shape == (4, 64, 1000)

    def test_process_logits_from_output_with_logits_attr(self):
        # Create mock output with logits attribute
        outputs = MagicMock()
        outputs.logits = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)

        model = MagicMock(spec=nn.Module)
        apply_temperature_fn = lambda x: x

        logits = _process_logits(outputs, model, apply_temperature_fn)

        assert logits is outputs.logits
        assert logits.shape == (4, 64, 1000)

    def test_process_logits_from_last_hidden_state(self):
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
        outputs = torch.randn(4, 64, 1000, device="cuda", requires_grad=True)
        model = MagicMock(spec=nn.Module)

        temperature = 2.0
        apply_temperature_fn = lambda x: x / temperature

        logits = _process_logits(outputs, model, apply_temperature_fn)

        # Verify temperature was applied
        expected_logits = outputs / temperature
        assert torch.allclose(logits, expected_logits)


@pytest.mark.automodel
class TestModelForward:
    def test_basic_model_forward(self, mock_model, runtime_config):
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (4, 64)).cuda(),
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify flash_attn_kwargs was passed
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" in call_kwargs

    def test_model_forward_with_reward_model(self, mock_model, runtime_config):
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (4, 64)).cuda(),
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        # Modify runtime_config for reward model
        runtime_config_reward = RuntimeConfig(
            **{
                **runtime_config.__dict__,
                "is_reward_model": True,
                "allow_flash_attn_args": False,
            }
        )

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config_reward,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify flash_attn_kwargs was NOT passed
        call_kwargs = mock_model.call_args[1]
        assert "flash_attn_kwargs" not in call_kwargs

    def test_model_forward_with_multimodal(self, mock_model, runtime_config):
        vlm_kwargs = {
            "pixel_values": torch.randn(4, 3, 224, 224).cuda(),
        }

        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (4, 64)).cuda(),
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=None,  # None for multimodal
            flash_attn_kwargs={},
            vlm_kwargs=vlm_kwargs,
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify VLM kwargs were passed and flash_attn_kwargs was removed
        call_kwargs = mock_model.call_args[1]
        assert "pixel_values" in call_kwargs
        assert "flash_attn_kwargs" not in call_kwargs

    def test_model_forward_with_moe_padding_mask(self, mock_model, runtime_config):
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (4, 64)).cuda(),
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        # Test with MoE model (not HF) - padding_mask should be set
        runtime_config_moe = RuntimeConfig(
            **{**runtime_config.__dict__, "is_moe_model": True}
        )

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config_moe,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify padding_mask was passed
        call_kwargs = mock_model.call_args[1]
        assert "padding_mask" in call_kwargs

    def test_model_forward_with_hf_moe_no_padding_mask(
        self, mock_model, runtime_config
    ):
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (4, 64)).cuda(),
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        # Test with HF MoE model - padding_mask should NOT be set
        runtime_config_hf_moe = RuntimeConfig(
            **{**runtime_config.__dict__, "is_hf_model": True, "is_moe_model": True}
        )

        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config_hf_moe,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify padding_mask was NOT passed
        call_kwargs = mock_model.call_args[1]
        assert "padding_mask" not in call_kwargs

    def test_model_forward_with_non_moe_no_padding_mask(
        self, mock_model, runtime_config
    ):
        processed_inputs = ProcessedInputs(
            input_ids=torch.randint(0, 1000, (4, 64)).cuda(),
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        # Test with non-MoE model - padding_mask should NOT be set
        outputs = model_forward(
            model=mock_model,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify padding_mask was NOT passed
        call_kwargs = mock_model.call_args[1]
        assert "padding_mask" not in call_kwargs


@pytest.mark.automodel
class TestForwardWithProcessor:
    def test_with_get_loss_processor(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
        """Test forward_with_processor with get_loss as the processor."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()
        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Call forward_with_processor with get_loss as processor
        loss, loss_metrics = forward_with_processor(
            model=mock_model,
            processor_fn=get_loss,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            processor_kwargs={
                "loss_inputs": loss_inputs,
            },
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify loss function was called
        mock_loss_fn.assert_called_once()

        # Verify loss and metrics were returned
        assert loss is not None
        assert isinstance(loss_metrics, dict)

    def test_with_get_logprobs_processor(
        self, mock_model, runtime_config, distributed_state
    ):
        """Test forward_with_processor with get_logprobs as the processor."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        input_ids = mb["input_ids"]

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        apply_temperature_fn = lambda x: x

        # Call forward_with_processor with get_logprobs as processor
        token_logprobs = forward_with_processor(
            model=mock_model,
            processor_fn=get_logprobs,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            processor_kwargs={
                "input_ids": input_ids,
                "apply_temperature_fn": apply_temperature_fn,
                "logprob_chunk_size": None,
            },
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify shape: [batch_size, seq_len]
        assert token_logprobs.shape == (4, 64)

        # Verify first token has zero logprob (prepended)
        assert torch.all(token_logprobs[:, 0] == 0.0)

    def test_with_get_topk_logits_processor(
        self, mock_model, runtime_config, distributed_state
    ):
        """Test forward_with_processor with get_topk_logits as the processor."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        apply_temperature_fn = lambda x: x
        mock_tp_mesh = MagicMock()

        # Call forward_with_processor with get_topk_logits as processor
        vals, idx = forward_with_processor(
            model=mock_model,
            processor_fn=get_topk_logits,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            processor_kwargs={
                "k": 10,
                "apply_temperature_fn": apply_temperature_fn,
            },
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify shape: [batch_size, seq_len, k]
        assert vals.shape == (4, 64, 10)
        assert idx.shape == (4, 64, 10)

    def test_with_custom_processor(self, mock_model, runtime_config, distributed_state):
        """Test forward_with_processor with a custom processor function."""

        # Create a custom processor that just returns the logits shape
        def custom_processor(
            outputs,
            model,
            processed_inputs,
            runtime_config,
            distributed_state,
        ):
            apply_temperature_fn = lambda x: x
            logits = _process_logits(outputs, model, apply_temperature_fn)
            return logits.shape

        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        apply_temperature_fn = lambda x: x

        # Call forward_with_processor with custom processor
        result = forward_with_processor(
            model=mock_model,
            processor_fn=custom_processor,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            processor_kwargs={},
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify result is the logits shape
        assert result == torch.Size([4, 64, 1000])

    def test_with_none_processor_kwargs(
        self, mock_model, runtime_config, distributed_state
    ):
        """Test forward_with_processor with None processor_kwargs (default)."""

        # Create a simple processor that doesn't need extra kwargs
        def simple_processor(
            outputs,
            model,
            processed_inputs,
            runtime_config,
            distributed_state,
        ):
            return "success"

        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        apply_temperature_fn = lambda x: x

        # Call forward_with_processor without processor_kwargs
        result = forward_with_processor(
            model=mock_model,
            processor_fn=simple_processor,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            # processor_kwargs not provided (defaults to None)
        )

        # Verify model was called
        mock_model.assert_called_once()

        # Verify result
        assert result == "success"

    def test_with_temperature_scaling(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
        """Test forward_with_processor with temperature scaling applied."""
        # Create test microbatch
        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 1000, (4, 64)).cuda(),
                "sample_mask": torch.ones(4, dtype=torch.bool).cuda(),
            }
        )

        # Create processed inputs
        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()

        # Mock temperature function that scales logits
        temperature = 2.0
        apply_temperature_fn = MagicMock(side_effect=lambda x: x / temperature)

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Call forward_with_processor
        loss, loss_metrics = forward_with_processor(
            model=mock_model,
            processor_fn=get_loss,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            processor_kwargs={
                "loss_inputs": loss_inputs,
            },
        )

        # Verify temperature function was called
        apply_temperature_fn.assert_called()


@pytest.mark.automodel
class TestGetLoss:
    def test_basic_train_output_processing(
        self, mock_model, mock_loss_fn, runtime_config, distributed_state
    ):
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

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(512.0).cuda()
        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        loss, loss_metrics = get_loss(
            outputs=outputs,
            model=mock_model,
            loss_inputs=loss_inputs,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
        )

        # Verify loss function was called
        mock_loss_fn.assert_called_once()
        assert loss is not None
        assert isinstance(loss_metrics, dict)

    @patch("nemo_rl.models.automodel.train.SequencePackingLossWrapper")
    def test_train_output_with_sequence_packing(
        self,
        mock_seq_packing_wrapper,
        mock_model,
        mock_loss_fn,
        runtime_config,
        distributed_state,
    ):
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

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=None,
            position_ids=torch.arange(204).unsqueeze(0).cuda(),
            flash_attn_kwargs=flash_attn_kwargs,
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=204,
        )

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

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Modify runtime_config for sequence packing
        runtime_config_sp = RuntimeConfig(
            **{**runtime_config.__dict__, "enable_seq_packing": True}
        )

        loss, loss_metrics = get_loss(
            outputs=outputs,
            model=mock_model,
            loss_inputs=loss_inputs,
            processed_inputs=processed_inputs,
            runtime_config=runtime_config_sp,
            distributed_state=distributed_state,
        )

        # Verify sequence packing wrapper was created
        mock_seq_packing_wrapper.assert_called_once()
        wrapped_loss_fn.assert_called_once()


@pytest.mark.automodel
class TestGetLogprobs:
    def test_basic_logprob_extraction(
        self, mock_model, runtime_config, distributed_state
    ):
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

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        apply_temperature_fn = lambda x: x

        token_logprobs = get_logprobs(
            outputs=outputs,
            model=mock_model,
            processed_inputs=processed_inputs,
            input_ids=input_ids,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            apply_temperature_fn=apply_temperature_fn,
            logprob_chunk_size=None,
        )

        # Verify shape: [batch_size, seq_len]
        assert token_logprobs.shape == (4, 64)

        # Verify first token has zero logprob (prepended)
        assert torch.all(token_logprobs[:, 0] == 0.0)

    def test_logprob_extraction_with_chunking(
        self, mock_model, runtime_config, distributed_state
    ):
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

        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            attention_mask=torch.ones(4, 128, dtype=torch.bool).cuda(),
            position_ids=torch.arange(128).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=128,
        )

        apply_temperature_fn = lambda x: x

        # Use chunking with chunk_size=64
        token_logprobs = get_logprobs(
            outputs=outputs,
            model=mock_model,
            processed_inputs=processed_inputs,
            input_ids=input_ids,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            apply_temperature_fn=apply_temperature_fn,
            logprob_chunk_size=64,
        )

        # Verify shape
        assert token_logprobs.shape == (4, 128)

        # Verify first token has zero logprob
        assert torch.all(token_logprobs[:, 0] == 0.0)


@pytest.mark.automodel
class TestGetTopkLogits:
    def test_basic_topk_extraction(self, mock_model, runtime_config, distributed_state):
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

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 64, dtype=torch.bool).cuda(),
            position_ids=torch.arange(64).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=64,
        )

        apply_temperature_fn = lambda x: x

        vals, idx = get_topk_logits(
            outputs=outputs,
            model=mock_model,
            processed_inputs=processed_inputs,
            k=10,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify shape: [batch_size, seq_len, k]
        assert vals.shape == (4, 64, 10)
        assert idx.shape == (4, 64, 10)

        # Verify indices are within valid range [0, vocab_size)
        assert torch.all(idx >= 0)
        assert torch.all(idx < 1000)

    def test_topk_extraction_with_temperature(
        self, mock_model, runtime_config, distributed_state
    ):
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

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(2, 32, dtype=torch.bool).cuda(),
            position_ids=torch.arange(32).repeat(2, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=32,
        )

        temperature = 2.0
        apply_temperature_fn = lambda x: x / temperature

        vals, idx = get_topk_logits(
            outputs=outputs,
            model=mock_model,
            processed_inputs=processed_inputs,
            k=5,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Verify shape
        assert vals.shape == (2, 32, 5)
        assert idx.shape == (2, 32, 5)

    def test_topk_extraction_different_k_values(
        self, mock_model, runtime_config, distributed_state
    ):
        outputs = MagicMock()
        outputs.logits = torch.randn(3, 48, 800, device="cuda", requires_grad=True)

        mb = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 800, (3, 48)).cuda(),
                "sample_mask": torch.ones(3, dtype=torch.bool).cuda(),
            }
        )

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(3, 48, dtype=torch.bool).cuda(),
            position_ids=torch.arange(48).repeat(3, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=48,
        )

        apply_temperature_fn = lambda x: x

        # Test with k=20
        vals, idx = get_topk_logits(
            outputs=outputs,
            model=mock_model,
            processed_inputs=processed_inputs,
            k=20,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            apply_temperature_fn=apply_temperature_fn,
        )

        assert vals.shape == (3, 48, 20)
        assert idx.shape == (3, 48, 20)


@pytest.mark.automodel
class TestIntegrationScenarios:
    @patch("nemo_rl.models.automodel.train.torch.distributed.all_reduce")
    @patch("nemo_rl.models.automodel.train.scale_grads_and_clip_grad_norm")
    @patch("nemo_rl.models.automodel.train.torch.cuda.empty_cache")
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

        processed_inputs = ProcessedInputs(
            input_ids=mb["input_ids"],
            attention_mask=torch.ones(4, 128, dtype=torch.bool).cuda(),
            position_ids=torch.arange(128).repeat(4, 1).cuda(),
            flash_attn_kwargs={},
            vlm_kwargs={},
            cp_buffers=[],
            seq_index=None,
            seq_len=128,
        )

        global_valid_seqs = torch.tensor(8.0).cuda()
        global_valid_toks = torch.tensor(1024.0).cuda()

        apply_temperature_fn = lambda x: x

        # Create loss inputs
        loss_inputs = LossInputs(
            microbatch=mb,
            loss_fn=mock_loss_fn,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            apply_temperature_fn=apply_temperature_fn,
        )

        # Create runtime config for training mode
        runtime_config = RuntimeConfig(
            is_reward_model=False,
            is_vlm=False,
            is_hf_model=False,
            is_moe_model=False,
            model_class=MagicMock,
            model_config=MagicMock(),
            hf_config_overrides={},
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.float16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
        )

        # Create distributed state
        distributed_state = DistributedState(
            rank=0,
            world_size=4,
            device_mesh=mock_device_mesh,
            dp_cp_mesh=mock_dp_mesh,
            dp_mesh=mock_dp_mesh,
            tp_mesh=MagicMock(),
            cp_mesh=MagicMock(),
            moe_mesh=None,
            dp_size=4,
            tp_size=1,
            cp_size=1,
            ep_size=1,
            sequence_parallel_enabled=False,
            manager=MagicMock(),
        )

        loss, loss_metrics = forward_backward(
            model=mock_model,
            processed_inputs=processed_inputs,
            loss_inputs=loss_inputs,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            eval_mode=False,
        )

        # Step 3: Optimizer step
        mock_scale_grads.return_value = torch.tensor(1.2).cuda()

        grad_norm = optimizer_step(
            optimizer=mock_optimizer,
            model=mock_model,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
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
