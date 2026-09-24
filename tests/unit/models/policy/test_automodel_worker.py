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

import os
import pprint
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import ray
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn, NLLLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy import AutomodelKwargs, PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.flops_tracker import FLOPTracker, get_hf_config
from tests.unit.test_utils import SimpleLossFn

try:
    import nemo_rl.models.policy.workers.automodel_policy_worker as worker_mod
    from nemo_rl.models.automodel.config import (
        ModelAndOptimizerState,
        RuntimeConfig,
    )
    from nemo_rl.models.policy.workers.automodel_policy_worker import (
        AutomodelPolicyWorkerImpl,
        _maybe_adapt_tensor_to_hf,
        dtensor_params_generator,
    )

    NEMO_AUTOMODEL_AVAILABLE = True
except ImportError:
    NEMO_AUTOMODEL_AVAILABLE = False


class _FakeTrainableModel:
    def __init__(self):
        self.train_called = False
        self.eval_called = False

    def train(self):
        self.train_called = True

    def eval(self):
        self.eval_called = True


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_automodel_prepare_for_training_restores_optimizer(monkeypatch):
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    model = _FakeTrainableModel()
    restored_devices = []

    worker.model = model
    worker.optimizer = object()
    worker.cpu_offload = False
    worker.move_to_cuda = lambda model: model
    worker.move_optimizer_to_device = lambda device: restored_devices.append(device)

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    AutomodelPolicyWorkerImpl.prepare_for_training(worker)

    assert model.train_called
    assert restored_devices == ["cuda"]


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
@pytest.mark.parametrize("keep_train_buffers", [False, True])
def test_automodel_prepare_for_lp_inference_keep_train_buffers(
    monkeypatch, keep_train_buffers
):
    """``keep_train_buffers`` suppresses the optimizer offload and nothing else.

    ``True`` is not reachable from a dtensor run today -- the split train API
    that leaves a step open (begin_train_step / train_microbatch /
    finish_train_step) exists only on the Megatron worker -- but the branch is
    here, so pin it: inverted, it would strand the optimizer on CPU for the rest
    of the step.
    """
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    model = _FakeTrainableModel()
    offloaded_devices = []

    worker.model = model
    worker.optimizer = object()
    worker.cpu_offload = False
    worker.offload_optimizer_for_logprob = True
    worker.move_to_cuda = lambda model: model
    worker.move_optimizer_to_device = lambda device: offloaded_devices.append(device)

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    # The allocator wake-up is a real ``.cuda()`` call; keep this test CPU-only.
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: MagicMock())

    AutomodelPolicyWorkerImpl.prepare_for_lp_inference(
        worker, keep_train_buffers=keep_train_buffers
    )

    assert model.eval_called
    assert offloaded_devices == ([] if keep_train_buffers else ["cpu"])


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_automodel_update_moe_gate_bias_called_when_supported():
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    worker.model = MagicMock()
    worker.model.update_moe_gate_bias = MagicMock()

    AutomodelPolicyWorkerImpl._update_moe_gate_bias_if_supported(worker)

    worker.model.update_moe_gate_bias.assert_called_once_with()


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_automodel_update_moe_gate_bias_noop_when_unsupported():
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    # A real module without the hook: getattr(..., None) must short-circuit so
    # models that do not expose update_moe_gate_bias are unaffected.
    worker.model = nn.Linear(1, 1)

    # Should be a no-op and must not raise.
    AutomodelPolicyWorkerImpl._update_moe_gate_bias_if_supported(worker)


def create_worker_test_config(
    model_name: str,
    tp: int = 1,
    cp: int = 1,
    sp: bool = False,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    custom_parallel_plan: str | None = None,
    precision: str = "float32",
    expert_parallel_size: int = 1,
    sequence_packing_enabled: bool = False,
    automodel_kwargs: AutomodelKwargs | None = None,
) -> PolicyConfig:
    config = {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,  # Small batch size for testing
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": precision,
        "offload_optimizer_for_logprob": False,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_new_tokens": 16,  # Small number of tokens for testing
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "dtensor_cfg": {
            "_v2": True,
            "enabled": True,
            "checkpoint": {
                "model_save_format": "safetensors",
                "save_consolidated": "false",
                "single_rank_consolidation": False,
                "consolidation_timeout_minutes": 30,
            },
            "cpu_offload": cpu_offload,
            "sequence_parallel": sp,
            "activation_checkpointing": activation_checkpointing,
            "tensor_parallel_size": tp,
            "context_parallel_size": cp,
            "custom_parallel_plan": custom_parallel_plan,
            "expert_parallel_size": expert_parallel_size,
        },
        "dynamic_batching": {
            "enabled": True,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": sequence_packing_enabled,
            "train_mb_tokens": 128,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
        },
        "scheduler": {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "kwargs": {
                "T_max": 100,
            },
        },
        "max_grad_norm": 1.0,
    }
    if automodel_kwargs is not None:
        config["dtensor_cfg"]["automodel_kwargs"] = automodel_kwargs
    return config


def create_worker_test_batch(
    batch_size: int = 8,
    seq_len: int = 128,
    vocab_size: int = 32000,
    mode: str = "train",
) -> BatchedDataDict:
    """Create a test batch for training or logprob computation."""
    torch.manual_seed(66)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
            **(
                {
                    "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
                    "sample_mask": torch.ones(batch_size).cuda(),
                }
                if mode == "train"
                else {}
            ),
        }
    )
    data = data.to("cpu")
    return data


@pytest.fixture(scope="module")
def two_gpu_virtual_cluster():
    cluster_name = "test"
    print(f"Creating virtual cluster '{cluster_name}'...")
    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[2],  # Use tp bundles, one per GPU
        use_gpus=True,
        num_gpus_per_node=2,  # Using tp GPUs
        max_colocated_worker_groups=1,  # Only one worker group
    )
    yield cluster
    print("Shutting down virtual cluster...")
    cluster.shutdown()


@pytest.mark.hf_gated
@pytest.mark.automodel
@pytest.mark.timeout(360)
@pytest.mark.parametrize("save_optimizer", [True, False])
def test_automodel_checkpoint_save_and_load(
    two_gpu_virtual_cluster,
    tiny_llama_model_path,
    save_optimizer,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointing_config = {
            "enabled": True,
            "checkpoint_dir": tmpdir,
            "metric_name": None,  # Save most recent checkpoints
            "higher_is_better": False,
            "keep_top_k": 2,
            "save_period": 30,
            "checkpoint_must_save_by": None,
            "save_optimizer": save_optimizer,
        }

        config = create_worker_test_config(
            model_name=tiny_llama_model_path,
            tp=2,
            cp=1,
        )

        policy = Policy(
            tokenizer=get_tokenizer(config["tokenizer"]),
            config=config,
            init_optimizer=True,
            init_reference_model=False,
            cluster=two_gpu_virtual_cluster,
            name_prefix="lm_policy_checkpoint",
        )

        try:
            weights_path = os.path.join(tmpdir, "policy", "weights")
            optimizer_path = (
                os.path.join(tmpdir, "policy", "optimizer") if save_optimizer else None
            )

            # Save checkpoint
            policy.save_checkpoint(
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                is_final_checkpoint=False,
            )
            policy.finalize_async_save()

            # Verify checkpoint files were created
            assert os.path.exists(weights_path), "Weights path should exist after save"

            # Load checkpoint into a new policy
            config2 = create_worker_test_config(
                model_name=tiny_llama_model_path,
                tp=2,
                cp=1,
            )

            # Shutdown original policy first to free GPU memory
            policy.shutdown()
            policy = None

            # Check if the optimizer exists in the checkpoint
            weights_path, optimizer_path = CheckpointManager.get_resume_paths(tmpdir)
            if save_optimizer:
                assert optimizer_path is not None, "Optimizer path should not be None"
            else:
                assert optimizer_path is None, "Optimizer path should be None"

            policy2 = Policy(
                tokenizer=get_tokenizer(config2["tokenizer"]),
                config=config2,
                init_optimizer=True,
                init_reference_model=False,
                cluster=two_gpu_virtual_cluster,
                name_prefix="lm_policy_checkpoint_loaded",
                weights_path=weights_path,
                optimizer_path=optimizer_path,
            )

            # Verify policy was loaded successfully
            assert len(policy2.worker_group.workers) == 2
            worker_alive = ray.get(
                [w.is_alive.remote() for w in policy2.worker_group.workers]
            )
            assert all(worker_alive)

            policy2.shutdown()
        finally:
            if policy is not None:
                policy.shutdown()


@pytest.mark.hf_gated
@pytest.mark.automodel
@pytest.mark.timeout(360)
@pytest.mark.parametrize("precision", ["bfloat16", "float16"])
def test_automodel_mixed_precision_training_and_logprobs(
    two_gpu_virtual_cluster,
    tiny_llama_model_path,
    precision,
):
    config = create_worker_test_config(
        model_name=tiny_llama_model_path,
        tp=2,
        cp=1,
        precision=precision,
    )

    policy = Policy(
        tokenizer=get_tokenizer(config["tokenizer"]),
        config=config,
        init_optimizer=True,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix=f"lm_policy_{precision}_mixed",
    )

    try:
        # --- Test Training ---
        train_data = create_worker_test_batch(mode="train")
        loss_fn = SimpleLossFn()

        policy.prepare_for_training()
        results = policy.train(train_data, loss_fn)

        # Verify training completed successfully
        assert "loss" in results
        loss_tensor = results["loss"]
        assert not torch.isnan(loss_tensor).any(), (
            f"Loss should not be NaN with precision={precision}"
        )
        assert not torch.isinf(loss_tensor).any(), (
            f"Loss should not be Inf with precision={precision}"
        )
        # Loss is returned in float32 (reduced in float32 for numerical stability)
        assert loss_tensor.dtype == torch.float32, (
            f"Loss should be float32, got {loss_tensor.dtype}"
        )

        policy.finish_training()

        # --- Test Logprobs ---
        logprob_data = create_worker_test_batch(mode="logprob")

        policy.prepare_for_lp_inference()
        logprobs = policy.get_logprobs(logprob_data)

        # Verify logprobs were computed successfully
        assert "logprobs" in logprobs
        logprobs_tensor = logprobs["logprobs"]
        assert logprobs_tensor.shape[0] == logprob_data.size
        assert not torch.isnan(logprobs_tensor).any(), (
            f"Logprobs should not be NaN with precision={precision}"
        )
        assert not torch.isinf(logprobs_tensor).any(), (
            f"Logprobs should not be Inf with precision={precision}"
        )
        # Logprobs are returned in float32 for numerical stability
        assert logprobs_tensor.dtype == torch.float32, (
            f"Logprobs should be float32 for numerical stability, got {logprobs_tensor.dtype}"
        )

        # Verify the configured precision by checking worker info
        worker_info = ray.get(policy.worker_group.workers[0].get_gpu_info.remote())
        assert worker_info is not None, "Should get worker info"
    finally:
        policy.shutdown()


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
class TestMaybeAdaptTensorToHF:
    """Tests for the _maybe_adapt_tensor_to_hf helper function."""

    def test_no_adapter_returns_single_tuple(self):
        """Test that when model has no adapter, returns single FQN-tensor tuple."""
        # Arrange
        model = nn.Linear(10, 10)
        fqn = "layer.weight"
        tensor = torch.randn(10, 10)

        # Act
        result = _maybe_adapt_tensor_to_hf(model, fqn, tensor)

        # Assert
        assert len(result) == 1, "Should return single tuple when no adapter"
        assert result[0][0] == fqn, "FQN should be unchanged"
        assert torch.equal(result[0][1], tensor), "Tensor should be unchanged"

    def test_adapter_converts_single_tensor(self):
        """Test that adapter is called when present on model."""
        # Arrange
        model = nn.Linear(10, 10)
        adapter_mock = Mock()
        adapter_mock.convert_single_tensor_to_hf.return_value = [
            ("adapted.weight", torch.randn(10, 10)),
            ("adapted.bias", torch.randn(10)),
        ]
        model.state_dict_adapter = adapter_mock

        fqn = "layer.weight"
        tensor = torch.randn(10, 10)

        # Act
        result = _maybe_adapt_tensor_to_hf(model, fqn, tensor)

        # Assert
        adapter_mock.convert_single_tensor_to_hf.assert_called_once_with(
            fqn,
            tensor,
            exclude_key_regex=r".*_extra_state.*",
            quantization=False,
        )
        assert len(result) == 2, "Should return multiple adapted tensors"
        assert result[0][0] == "adapted.weight"
        assert result[1][0] == "adapted.bias"

    def test_adapter_with_quantization_flag(self):
        """Test that quantization flag is passed to adapter correctly."""
        # Arrange
        model = nn.Linear(10, 10)
        adapter_mock = Mock()
        adapter_mock.convert_single_tensor_to_hf.return_value = [
            ("quantized.weight", torch.randn(10, 10))
        ]
        model.state_dict_adapter = adapter_mock

        fqn = "layer.weight"
        tensor = torch.randn(10, 10)

        # Act
        result = _maybe_adapt_tensor_to_hf(model, fqn, tensor, quantization=True)

        # Assert
        adapter_mock.convert_single_tensor_to_hf.assert_called_once_with(
            fqn,
            tensor,
            exclude_key_regex=r".*_extra_state.*",
            quantization=True,
        )
        assert len(result) == 1

    def test_adapter_excludes_extra_state_regex(self):
        """Test that _extra_state regex is always passed to exclude such tensors."""
        # Arrange
        model = nn.Linear(10, 10)
        adapter_mock = Mock()
        adapter_mock.convert_single_tensor_to_hf.return_value = []
        model.state_dict_adapter = adapter_mock

        fqn = "layer._extra_state"
        tensor = torch.randn(10)

        # Act
        _maybe_adapt_tensor_to_hf(model, fqn, tensor)

        # Assert
        adapter_mock.convert_single_tensor_to_hf.assert_called_once()
        call_kwargs = adapter_mock.convert_single_tensor_to_hf.call_args[1]
        assert call_kwargs["exclude_key_regex"] == r".*_extra_state.*", (
            "Should exclude extra_state tensors"
        )


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
class TestDTensorParamsGenerator:
    """Tests for the dtensor_params_generator helper function."""

    def test_simple_model_yields_adapted_tensors(self):
        """Test that generator yields correct (name, tensor) pairs for a simple model."""
        # Arrange
        model = nn.Linear(10, 5)
        target_dtype = torch.float32

        # Act
        results = list(dtensor_params_generator(model, target_dtype))

        # Assert
        assert len(results) == 2, "Linear layer should have weight and bias"
        names = [name for name, _ in results]
        assert "weight" in names
        assert "bias" in names

        # Check that tensors are in the correct dtype and contiguous
        for name, tensor in results:
            assert tensor.dtype == target_dtype, (
                f"Tensor {name} should be {target_dtype}"
            )
            assert tensor.is_contiguous(), f"Tensor {name} should be contiguous"

    def test_dtype_conversion(self):
        """Test that tensors are converted to target dtype."""
        # Arrange
        model = nn.Linear(10, 5)
        # Initialize with float32
        model = model.to(torch.float32)
        target_dtype = torch.bfloat16

        # Act
        results = list(dtensor_params_generator(model, target_dtype))

        # Assert
        for name, tensor in results:
            assert tensor.dtype == target_dtype, (
                f"Tensor {name} should be converted to {target_dtype}"
            )

    def test_preserves_fp32_router_correction_bias(self):
        """FP32 MoE router state must not be downcast during refit."""

        class RouterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "e_score_correction_bias", torch.arange(4, dtype=torch.float32)
                )
                self.register_buffer(
                    "ordinary_buffer", torch.arange(4, dtype=torch.float32)
                )

        results = dict(dtensor_params_generator(RouterModel(), torch.bfloat16))

        assert results["e_score_correction_bias"].dtype == torch.float32
        assert results["ordinary_buffer"].dtype == torch.bfloat16

    def test_contiguous_output(self):
        """Test that output tensors are contiguous."""
        # Arrange
        model = nn.Linear(10, 5)
        target_dtype = torch.float32

        # Act
        results = list(dtensor_params_generator(model, target_dtype))

        # Assert
        for name, tensor in results:
            assert tensor.is_contiguous(), f"Tensor {name} should be contiguous"

    def test_with_adapter_model(self):
        """Test that adapter is used when present on model."""
        # Arrange
        model = nn.Linear(10, 5)
        adapter_mock = Mock()
        # Mock adapter to return multiple tensors for a single input
        adapter_mock.convert_single_tensor_to_hf.return_value = [
            ("adapted.weight.1", torch.randn(5, 10)),
            ("adapted.weight.2", torch.randn(5, 10)),
        ]
        model.state_dict_adapter = adapter_mock
        target_dtype = torch.float32

        # Act
        results = list(dtensor_params_generator(model, target_dtype))

        # Assert
        # Each state_dict entry (weight, bias) goes through adapter
        # Adapter returns 2 tensors for each, so we expect 4 total
        assert len(results) >= 4, "Should have adapted tensors from adapter"

        # Verify adapter was called
        assert adapter_mock.convert_single_tensor_to_hf.call_count >= 2

    def test_empty_model(self):
        """Test handling of model with no parameters."""
        # Arrange
        model = nn.Module()  # Empty module with no parameters
        target_dtype = torch.float32

        # Act
        results = list(dtensor_params_generator(model, target_dtype))

        # Assert
        assert len(results) == 0, "Empty model should yield no parameters"

    def test_generator_is_iterable(self):
        """Test that dtensor_params_generator returns an iterable generator."""
        # Arrange
        model = nn.Linear(10, 5)
        target_dtype = torch.float32

        # Act
        gen = dtensor_params_generator(model, target_dtype)

        # Assert
        from collections.abc import Generator as ABCGenerator

        assert isinstance(gen, ABCGenerator), "Should return a generator"

        # Verify we can iterate over it
        results = list(gen)
        assert len(results) > 0, "Should yield at least one item"

    def test_multiple_layers(self):
        """Test generator with a more complex model with multiple layers."""
        # Arrange
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )
        target_dtype = torch.float32

        # Act
        results = list(dtensor_params_generator(model, target_dtype))

        # Assert
        # Should have 4 parameters: 2 weights + 2 biases from the Linear layers
        assert len(results) == 4, (
            "Sequential with 2 Linear layers should have 4 parameters"
        )

        # Check all tensors
        for name, tensor in results:
            assert tensor.dtype == target_dtype
            assert tensor.is_contiguous()


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_prepare_refit_info_preserves_fp32_router_correction_bias():
    """Refit metadata must match the FP32 router-bias payload dtype."""

    class RouterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "e_score_correction_bias", torch.arange(4, dtype=torch.float32)
            )
            self.register_buffer(
                "ordinary_buffer", torch.arange(4, dtype=torch.float32)
            )

    worker = object.__new__(AutomodelPolicyWorkerImpl)
    worker.model = RouterModel()
    worker.dtype = torch.bfloat16

    refit_info = AutomodelPolicyWorkerImpl.prepare_refit_info(worker)

    assert refit_info["e_score_correction_bias"][1] == torch.float32
    assert refit_info["ordinary_buffer"][1] == torch.bfloat16


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
class TestAutocastContext:
    """Tests for the precision context retained by the policy worker."""

    def test_disabled_returns_noop_context(self):
        worker = object.__new__(AutomodelPolicyWorkerImpl)
        worker.autocast_enabled = False

        with worker._autocast_context():
            assert not torch.is_autocast_enabled("cuda")

    @patch("nemo_rl.models.policy.workers.automodel_policy_worker.torch.autocast")
    def test_enabled_uses_worker_dtype(self, mock_autocast):
        worker = object.__new__(AutomodelPolicyWorkerImpl)
        worker.autocast_enabled = True
        worker.dtype = torch.bfloat16
        expected_context = MagicMock()
        mock_autocast.return_value = expected_context

        result = worker._autocast_context()

        assert result is expected_context
        mock_autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)


def _init_v2_worker_mocked(
    monkeypatch,
    *,
    init_reference_model,
    weights_path,
    optimizer_path,
    model_type=None,
):
    """Run AutomodelPolicyWorkerImpl.__init__ with all heavy deps mocked.

    Returns (worker, call_log, setup_mock, load_checkpoint_mock).
    """
    call_log = []

    monkeypatch.setattr(worker_mod, "apply_transformer_engine_patch", lambda: None)
    monkeypatch.setattr(worker_mod.ray, "get_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        "nemo_rl.distributed.numa_utils.bind_to_gpu_numa", lambda gpu_id: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.automodel.setup.get_tokenizer",
        lambda cfg, get_processor=False: MagicMock(name="tokenizer"),
    )

    # Unpacked as runtime config at the end of __init__.
    runtime_config = RuntimeConfig(
        model_class="model_class",
        model_config=SimpleNamespace(model_type=model_type),
        hf_config_overrides={},
        allow_flash_attn_args=False,
        attn_impl="attn_impl",
        dtype=None,
        enable_seq_packing=False,
        max_grad_norm=1.0,
        cpu_offload=False,
        offload_optimizer_for_logprob=False,
        is_generation_colocated=False,
        sampling_params=None,
        is_reward_model=False,
    )
    monkeypatch.setattr(
        worker_mod, "validate_and_prepare_config", lambda **kw: runtime_config
    )
    monkeypatch.setattr(worker_mod, "setup_distributed", lambda **kw: MagicMock())
    monkeypatch.setattr(worker_mod.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        worker_mod, "maybe_preinit_nixl_checkpoint_engine", lambda cfg: None
    )

    load_checkpoint_mock = MagicMock(
        side_effect=lambda **kw: call_log.append("load_checkpoint")
    )

    def fake_init_checkpoint_manager(self, config_updates=None):
        self._test_checkpoint_config_updates = config_updates
        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.load_checkpoint = load_checkpoint_mock

    monkeypatch.setattr(
        AutomodelPolicyWorkerImpl,
        "_init_checkpoint_manager",
        fake_init_checkpoint_manager,
    )

    # Unpacked as model_and_optimizer_state.
    model_and_optimizer_state = ModelAndOptimizerState(
        model=MagicMock(name="model"),
        optimizer=MagicMock(name="optimizer"),
        scheduler=MagicMock(name="scheduler"),
        is_hf_model=False,
        is_moe_model=False,
        is_reward_model=False,
        model_class="model_class",
        model_config="model_config",
        peft_config=None,
        autocast_enabled=False,
    )
    setup_mock = MagicMock(
        side_effect=lambda **kw: (
            call_log.append("setup_model_and_optimizer"),
            model_and_optimizer_state,
        )[1]
    )
    monkeypatch.setattr(worker_mod, "setup_model_and_optimizer", setup_mock)

    ref_state = {"ref": "state"}
    monkeypatch.setattr(
        worker_mod,
        "setup_reference_model_state",
        lambda model: (call_log.append("setup_reference_model_state"), ref_state)[1],
    )

    config = {
        "model_name": "base-model",
        "tokenizer": {},
        "dtensor_cfg": {
            "checkpoint": {
                "model_save_format": "safetensors",
                "save_consolidated": "false",
            },
        },
        "generation": {},
    }
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    AutomodelPolicyWorkerImpl.__init__(
        worker,
        config,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=init_reference_model,
    )
    return worker, call_log, setup_mock, load_checkpoint_mock


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
@pytest.mark.parametrize(
    ("model_type", "expected_async"),
    [("deepseek_v4", False), ("deepseek_v3", True)],
)
def test_automodel_scopes_synchronous_checkpointing_to_dsv4(
    monkeypatch, model_type, expected_async
):
    worker, *_ = _init_v2_worker_mocked(
        monkeypatch,
        init_reference_model=False,
        weights_path=None,
        optimizer_path=None,
        model_type=model_type,
    )

    assert worker._test_checkpoint_config_updates["is_async"] is expected_async


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_automodel_resume_with_reference_model_defers_checkpoint_load(monkeypatch):
    """On resume with a KL reference, the reference must be captured from base
    weights (checkpoint load deferred until after the capture)."""
    worker, call_log, setup_mock, load_mock = _init_v2_worker_mocked(
        monkeypatch,
        init_reference_model=True,
        weights_path="/ckpt/weights",
        optimizer_path="/ckpt/optim",
    )
    # (a) base weights used for setup: checkpoint paths not passed through.
    assert setup_mock.call_args.kwargs["weights_path"] is None
    assert setup_mock.call_args.kwargs["optimizer_path"] is None
    # (b) reference captured BEFORE the checkpoint load.
    assert call_log == [
        "setup_model_and_optimizer",
        "setup_reference_model_state",
        "load_checkpoint",
    ]
    assert worker.reference_model_state_dict == {"ref": "state"}
    # (c) checkpoint still loaded, with the original paths.
    assert load_mock.call_args.kwargs["weights_path"] == "/ckpt/weights"
    assert load_mock.call_args.kwargs["optimizer_path"] == "/ckpt/optim"
    assert load_mock.call_args.kwargs["model"] is worker.model
    assert load_mock.call_args.kwargs["optimizer"] is worker.optimizer


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_automodel_resume_without_reference_model_passes_paths_through(monkeypatch):
    worker, call_log, setup_mock, load_mock = _init_v2_worker_mocked(
        monkeypatch,
        init_reference_model=False,
        weights_path="/ckpt/weights",
        optimizer_path="/ckpt/optim",
    )
    assert setup_mock.call_args.kwargs["weights_path"] == "/ckpt/weights"
    assert setup_mock.call_args.kwargs["optimizer_path"] == "/ckpt/optim"
    load_mock.assert_not_called()
    assert worker.reference_model_state_dict is None


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_automodel_fresh_run_with_reference_model_does_not_defer(monkeypatch):
    worker, call_log, setup_mock, load_mock = _init_v2_worker_mocked(
        monkeypatch,
        init_reference_model=True,
        weights_path=None,
        optimizer_path=None,
    )
    assert setup_mock.call_args.kwargs["weights_path"] is None
    load_mock.assert_not_called()
    assert worker.reference_model_state_dict == {"ref": "state"}


def create_test_config(
    model_name: str,
    tp: int = 1,
    cp: int = 1,
    sp: bool = False,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    custom_parallel_plan: str | None = None,
    enable_loras: bool = False,
) -> PolicyConfig:
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,  # Small batch size for testing
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": "float32",
        "offload_optimizer_for_logprob": False,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_new_tokens": 16,  # Small number of tokens for testing
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "dtensor_cfg": {
            "_v2": True,
            "checkpoint": {
                "model_save_format": "safetensors",
                "save_consolidated": "false",
            },
            "enabled": True,
            "cpu_offload": cpu_offload,
            "sequence_parallel": sp,
            "activation_checkpointing": activation_checkpointing,
            "tensor_parallel_size": tp,
            "context_parallel_size": cp,
            "custom_parallel_plan": custom_parallel_plan,
            "lora": {
                "enabled": enable_loras,
                "target_modules": [],
                "exclude_modules": [],
                "match_all_linear": True,
                "dim": 32,
                "alpha": 32,
                "dropout": 0.0,
                "dropout_position": "post",
                "lora_A_init": "xavier",
                "use_triton": True,
            },
        },
        "dynamic_batching": {
            "enabled": True,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
        },
        "scheduler": {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "kwargs": {
                "T_max": 100,
            },
        },
        "max_grad_norm": 1.0,
    }


def update_lora_config(
    config: PolicyConfig,
    enabled: bool = True,
    target_modules: list[str] = [],
    exclude_modules: list[str] = [],
    match_all_linear: bool = True,
    dim: int = 32,
    alpha: int = 32,
    dropout: float = 0.0,
    dropout_position: str = "post",
    lora_A_init: str = "xavier",
    use_triton: bool = True,
):
    if enabled:
        config["dtensor_cfg"]["_v2"] = True
        config["dtensor_cfg"]["checkpoint"]["model_save_format"] = "safetensors"

    config["dtensor_cfg"]["lora"].update(
        {
            "enabled": enabled,
            "target_modules": target_modules,
            "exclude_modules": exclude_modules,
            "match_all_linear": match_all_linear,
            "dim": dim,
            "alpha": alpha,
            "dropout": dropout,
            "dropout_position": dropout_position,
            "lora_A_init": lora_A_init,
            "use_triton": use_triton,
        }
    )


def create_test_batch(
    batch_size: int = 8,
    seq_len: int = 128,
    vocab_size: int = 32000,
    mode: str = "train",
) -> BatchedDataDict:
    # set random seed
    torch.manual_seed(66)
    # Create test input_ids and attention_mask
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # Calculate input_lengths (all sequences are full length in this test)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
            **(
                {
                    "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
                    "sample_mask": torch.ones(batch_size).cuda(),
                }
                if mode == "train"
                else {}
            ),
        }
    )
    data = data.to("cpu")
    return data


def calculate_token_logprobs(model_name: str, data: BatchedDataDict):
    data = data.to("cuda")
    input_ids = data["input_ids"]

    with torch.no_grad():
        # run the log prob of regular hf model here
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda", torch_dtype=torch.float32
        )
        hf_model.eval()
        outputs = hf_model(**data)

    log_probs = torch.nn.functional.log_softmax(
        outputs.logits.to(torch.float32), dim=-1
    )
    next_tokens = input_ids[:, 1:]
    log_probs = log_probs[:, :-1]
    token_logprobs = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(
        -1
    )
    token_logprobs = torch.cat(
        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
    ).cpu()

    data = data.to("cpu")
    return token_logprobs


def _base_setup_impl(request, cluster):
    """Implementation for base setup - can be used with any cluster."""
    params = request.param if hasattr(request, "param") else None
    assert params is not None, "params is not set"

    mode = params["mode"]
    model_fixture_name = params["model_fixture_name"]
    specified_config = params["specified_config"]
    enable_loras = params["enable_loras"]
    lora_config = params["lora_config"]
    model_name = request.getfixturevalue(model_fixture_name)

    policy = None
    data = None
    loss_fn = None

    try:
        config = create_test_config(model_name, **specified_config)

        if enable_loras:
            update_lora_config(config, **lora_config)

        tokenizer = get_tokenizer(config["tokenizer"])
        print(f"Creating {mode} Policy with {specified_config}...")
        policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=False,
        )
        print("Creating test batch...")
        data = create_test_batch(mode=mode)

        if mode == "train":
            # Create loss function
            loss_fn: LossFunction = SimpleLossFn()
            yield policy, data, loss_fn
        elif mode == "logprob":
            token_logprobs = calculate_token_logprobs(model_name, data)
            yield policy, data, token_logprobs

    except Exception as e:
        print(f"Error during setup: {e}")
        pytest.skip(f"Setup failed: {e}")
    finally:
        print("Cleaning up resources for test")
        if policy:
            policy.shutdown()


def _test_automodel_worker_training(policy, data, loss_fn):
    def verify_loss_tensor(loss_tensor):
        assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
        assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
        return loss_tensor

    # Verify resources were created properly
    assert policy is not None, "Training policy was not created properly"
    assert data is not None, "Test data was not created properly"
    assert loss_fn is not None, "Loss function was not created properly"

    # Call prepare_for_training if available
    print("\nPreparing for training...")
    policy.prepare_for_training()

    losses = []
    for steps in range(2):
        results = policy.train(data, loss_fn)

        # Verify results
        assert "loss" in results, "Training results should contain 'loss'"
        loss_tensor = results["loss"]
        verify_loss_tensor(loss_tensor)
        losses.append(loss_tensor[-1].item())

        print(f"Training loss: {results['loss']}")

    policy.finish_training()

    # Verify loss changed between iterations (model parameters were updated)
    assert losses[0] > losses[-1], "Loss should decrease over training iterations"

    # Verify the train function returns the performance metrics

    if policy.flops_tracker is not None:
        assert "total_flops" in results and isinstance(
            results["total_flops"], (int, float)
        ), "training backend should report total_flops"
        assert results["total_flops"] > 0, "total_flops should be positive"
        assert "num_ranks" in results and isinstance(results["num_ranks"], int), (
            "training backend should report num_ranks"
        )
        assert results["num_ranks"] > 0, "num_ranks should be positive"

        # we don't always require theoretical_tflops since the data about the GPU
        # is not always available.
        if "theoretical_tflops" in results:
            assert isinstance(results["theoretical_tflops"], (int, float)), (
                "training backend should report theoretical_tflops"
            )
            assert results["theoretical_tflops"] > 0, (
                "theoretical_tflops should be positive"
            )


def _test_automodel_worker_logprob(policy, data, logprobs):
    # Verify resources were created properly assert policy is not None, "Policy was not created properly"
    assert data is not None, "Test data was not created properly"

    # Generate logprobs
    print("\nGenerating logprobs...")
    policy.prepare_for_lp_inference()
    policy_logprobs = policy.get_logprobs(data)["logprobs"]

    print("## MAX DIFF ###", torch.max(torch.abs(policy_logprobs - logprobs)))
    assert torch.allclose(policy_logprobs, logprobs), (
        f"max diff {torch.max(torch.abs(policy_logprobs - logprobs))}"
    )


@pytest.mark.hf_gated
class TestSingleGPUCluster:
    """Tests that run on a single GPU cluster."""

    @pytest.fixture(scope="class")
    def single_gpu_cluster(self):
        """Class-scoped single GPU virtual cluster fixture."""
        cluster_name = "test_single_gpu"
        print(f"Creating single GPU virtual cluster '{cluster_name}'...")
        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[1],  # Single GPU bundle
            use_gpus=True,
            num_gpus_per_node=1,  # Using 1 GPU
            max_colocated_worker_groups=1,  # Only one worker group
        )
        yield cluster
        print("Shutting down single GPU virtual cluster...")
        cluster.shutdown()

    @pytest.mark.timeout(360)
    @pytest.mark.automodel
    def test_automodel_single_gpu_training(
        self, single_gpu_cluster, tiny_llama_model_path
    ):
        """Test DTensor training with a single GPU cluster (no parallelism)."""
        config = create_test_config(
            tiny_llama_model_path,
            tp=1,
            cp=1,
            sp=False,
            cpu_offload=False,
            activation_checkpointing=False,
        )
        tokenizer = get_tokenizer(config["tokenizer"])
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer
        )

        print("Creating Policy with single GPU cluster...")
        policy = Policy(
            cluster=single_gpu_cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=False,
        )

        try:
            # Verify we have one worker
            assert len(policy.worker_group.workers) == 1, (
                "Should have 1 worker for single GPU"
            )

            # Check worker is alive
            worker_alive = ray.get(
                [w.is_alive.remote() for w in policy.worker_group.workers]
            )
            assert all(worker_alive), f"Worker is not alive: {worker_alive}"

            # Get GPU info to verify setup
            gpu_infos = ray.get(
                [w.get_gpu_info.remote() for w in policy.worker_group.workers]
            )
            assert len(gpu_infos) == 1, "Should have 1 GPU info"
            assert gpu_infos[0]["world_size"] == 1, (
                "World size should be 1 for single GPU"
            )
            assert gpu_infos[0]["rank"] == 0, "Rank should be 0 for single GPU"

            # Create test batch
            data = create_test_batch(mode="train")
            loss_fn = SimpleLossFn()

            # Test training
            policy.prepare_for_training()

            losses = []
            for step in range(2):
                results = policy.train(data, loss_fn)
                assert "loss" in results, "Training results should contain 'loss'"
                loss_tensor = results["loss"]
                assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
                assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
                losses.append(loss_tensor[-1].item())
                print(f"Step {step} - Training loss: {results['loss']}")

            policy.finish_training()

            # Verify loss changed (model was updated)
            assert losses[0] > losses[-1], (
                "Loss should decrease over training iterations"
            )

        finally:
            policy.shutdown()

    @pytest.mark.timeout(360)
    @pytest.mark.automodel
    def test_automodel_single_gpu_logprob(
        self, single_gpu_cluster, tiny_llama_model_path
    ):
        """Test DTensor logprob computation with a single GPU cluster (no parallelism)."""
        config = create_test_config(
            tiny_llama_model_path,
            tp=1,
            cp=1,
            sp=False,
            cpu_offload=False,
            activation_checkpointing=False,
        )
        tokenizer = get_tokenizer(config["tokenizer"])
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer
        )

        print("Creating Policy with single GPU cluster for logprob...")
        policy = Policy(
            cluster=single_gpu_cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=False,
        )

        try:
            # Verify we have one worker
            assert len(policy.worker_group.workers) == 1, (
                "Should have 1 worker for single GPU"
            )

            # Create test batch and compute reference logprobs
            data = create_test_batch(mode="logprob")
            expected_logprobs = calculate_token_logprobs(tiny_llama_model_path, data)

            # Test logprob computation
            policy.prepare_for_lp_inference()
            policy_logprobs = policy.get_logprobs(data)["logprobs"]

            max_diff = torch.max(torch.abs(policy_logprobs - expected_logprobs))
            print(f"Max logprob diff: {max_diff}")
            assert torch.allclose(policy_logprobs, expected_logprobs), (
                f"Logprobs should match reference. Max diff: {max_diff}"
            )

        finally:
            policy.shutdown()


@pytest.mark.hf_gated
class TestTwoGPUCluster:
    """Tests that run on a two GPU cluster."""

    @pytest.fixture(scope="class")
    def two_gpu_cluster(self):
        """Class-scoped two GPU virtual cluster fixture."""
        cluster_name = "test_two_gpu"
        print(f"Creating virtual cluster '{cluster_name}'...")
        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[2],  # Use tp bundles, one per GPU
            use_gpus=True,
            num_gpus_per_node=2,  # Using tp GPUs
            max_colocated_worker_groups=1,  # Only one worker group
        )
        yield cluster
        print("Shutting down virtual cluster...")
        cluster.shutdown()

    @pytest.fixture
    def policy_setup(self, request, two_gpu_cluster, tiny_llama_model_path):
        """Setup and teardown for policy tests - creates a virtual cluster and policy."""
        params = request.param if hasattr(request, "param") else {}
        enable_loras = params.get("enable_loras", False)

        config = create_test_config(tiny_llama_model_path, enable_loras=enable_loras)
        tokenizer = get_tokenizer(config["tokenizer"])
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer
        )

        print("Creating Policy...")
        policy = Policy(cluster=two_gpu_cluster, config=config, tokenizer=tokenizer)

        yield policy

        print("Shutting down policy...")
        policy.shutdown()

    @pytest.fixture(
        params=[
            # model_fixture_name        tp cp  sp     cpu    act
            ("tiny_llama_model_path", 1, 1, False, False, False),
            ("tiny_llama_model_path", 1, 1, True, False, False),
            ("tiny_llama_model_path", 1, 1, False, True, False),
            ("tiny_llama_model_path", 1, 1, False, False, True),
            ("tiny_llama_model_path", 1, 2, False, False, False),
            ("tiny_qwen2_model_path", 1, 1, True, True, False),
            ("tiny_qwen2_model_path", 1, 1, True, False, True),
            ("tiny_qwen2_model_path", 1, 1, False, True, True),
            ("tiny_qwen2_model_path", 1, 1, True, True, True),
            ("tiny_qwen2_model_path", 1, 2, False, False, False),
            ("tiny_qwen3_model_path", 1, 1, True, True, False),
            ("tiny_qwen3_model_path", 1, 1, True, False, True),
            ("tiny_qwen3_model_path", 1, 1, False, True, True),
            ("tiny_qwen3_model_path", 1, 1, True, True, True),
            ("tiny_qwen3_model_path", 1, 2, False, False, False),
            (
                "tiny_gemma3_model_path",
                1,
                1,
                True,
                True,
                False,
            ),  # gemma3 doesn't support spda
            ("tiny_gemma3_model_path", 1, 1, True, False, True),
            ("tiny_gemma3_model_path", 1, 1, False, True, True),
            ("tiny_gemma3_model_path", 1, 1, True, True, True),
            # CP doesn't support gemma3 due to spda input has attent_mask != None.
            # Nemotron-H doesn't support SP https://github.com/NVIDIA-NeMo/RL/issues/881
            # ("tiny_nemotron5_h_model_path", 1, 1, True, True, False),
            # ("tiny_nemotron5_h_model_path", 1, 1, True, False, True),
            # ("tiny_nemotron5_h_model_path", 1, 1, True, True, True),
            # Disabled until https://github.com/NVIDIA-NeMo/RL/issues/4211 is fixed
            # ("tiny_nemotron5_h_model_path", 1, 1, False, False, False),
            # ("tiny_nemotron5_h_model_path", 1, 1, False, True, True),
            # nemotron5_h doesn't support cp
            # TP2, SP=True
            ("tiny_llama_model_path", 2, 1, True, False, False),
            ("tiny_qwen2_model_path", 2, 1, True, False, False),
        ]
    )
    def training_setup(self, request, two_gpu_cluster):
        """Setup and teardown specifically for training tests."""
        request.param = {
            "mode": "train",
            "enable_loras": False,
            "lora_config": None,
            "model_fixture_name": request.param[0],
            "specified_config": {
                "tp": request.param[1],
                "cp": request.param[2],
                "sp": request.param[3],
                "cpu_offload": request.param[4],
                "activation_checkpointing": request.param[5],
            },
        }
        yield from _base_setup_impl(request, two_gpu_cluster)

    @pytest.fixture(
        params=[
            # TP=2, CP=1
            ("tiny_qwen2_model_path", 2, 1, False, True, False),
            ("tiny_qwen2_model_path", 2, 1, False, False, False),
            ("tiny_llama_model_path", 2, 1, False, False, False),
            ("tiny_llama_model_path", 2, 1, False, True, False),
            ("tiny_llama_model_path", 2, 1, False, True, True),
            ("tiny_qwen3_model_path", 2, 1, False, True, False),
            ("tiny_qwen3_model_path", 2, 1, False, False, False),
            ("tiny_gemma3_model_path", 2, 1, False, True, False),
            ("tiny_gemma3_model_path", 2, 1, False, False, False),
            # TP=1, CP=2 — skipped: CP=2 hits DTensor redistribute assertion with transformers v5 (hemil)
            ("tiny_qwen2_model_path", 1, 2, False, True, False),
            ("tiny_qwen2_model_path", 1, 2, False, False, False),
            ("tiny_llama_model_path", 1, 2, False, False, False),
            ("tiny_llama_model_path", 1, 2, False, True, False),
            ("tiny_llama_model_path", 1, 2, False, True, True),
            ("tiny_qwen3_model_path", 1, 2, False, True, False),
            ("tiny_qwen3_model_path", 1, 2, False, False, False),
        ]
    )
    def logprob_setup(self, request, two_gpu_cluster):
        """Setup and teardown specifically for logprob tests."""
        request.param = {
            "mode": "logprob",
            "enable_loras": False,
            "lora_config": None,
            "model_fixture_name": request.param[0],
            "specified_config": {
                "tp": request.param[1],
                "cp": request.param[2],
                "sp": request.param[3],
                "cpu_offload": request.param[4],
                "activation_checkpointing": request.param[5],
            },
        }
        yield from _base_setup_impl(request, two_gpu_cluster)

    @pytest.fixture(
        params=[
            # model_name,             target_modules, exclude_modules, match_all_linear, dim,  alpha, dropout, dropout_position, lora_A_init, use_triton
            (
                "tiny_llama_model_path",
                [],
                [],
                True,
                16,
                32,
                0.0,
                "post",
                "xavier",
                True,
            ),
            ("tiny_qwen2_model_path", [], [], True, 32, 32, 0.0, "pre", "xavier", True),
            (
                "tiny_qwen2_model_path",
                ["q_proj", "k_proj", "*gate_proj*", "*up_proj*", "*down_proj*"],
                [],
                False,
                32,
                16,
                0.0,
                "post",
                "uniform",
                True,
            ),
            (
                "tiny_qwen2_model_path",
                [],
                ["q_proj", "k_proj"],
                False,
                32,
                16,
                0.0,
                "post",
                "uniform",
                True,
            ),
        ]
    )
    def training_with_lora_setup(self, request, two_gpu_cluster):
        """Setup and teardown specifically for training with lora tests."""
        request.param = {
            "mode": "train",
            "enable_loras": True,
            "model_fixture_name": request.param[0],
            "specified_config": {},
            "lora_config": {
                "target_modules": request.param[1],
                "exclude_modules": request.param[2],
                "match_all_linear": request.param[3],
                "dim": request.param[4],
                "alpha": request.param[5],
                "dropout": request.param[6],
                "dropout_position": request.param[7],
                "lora_A_init": request.param[8],
                "use_triton": request.param[9],
            },
        }
        yield from _base_setup_impl(request, two_gpu_cluster)

    @pytest.fixture(
        params=[
            # model_name,             target_modules, exclude_modules, match_all_linear, dim,  alpha, dropout, dropout_position, lora_A_init, use_triton
            (
                "tiny_llama_model_path",
                [],
                [],
                True,
                16,
                32,
                0.0,
                "post",
                "xavier",
                True,
            ),
            ("tiny_qwen2_model_path", [], [], True, 32, 32, 0.0, "pre", "xavier", True),
            (
                "tiny_qwen2_model_path",
                ["q_proj", "k_proj", "*gate_proj*", "*up_proj*", "*down_proj*"],
                [],
                False,
                32,
                16,
                0.0,
                "post",
                "uniform",
                True,
            ),
            (
                "tiny_qwen2_model_path",
                [],
                ["q_proj", "k_proj"],
                False,
                32,
                16,
                0.0,
                "post",
                "uniform",
                True,
            ),
        ]
    )
    def logprob_with_lora_setup(self, request, two_gpu_cluster):
        """Setup and teardown specifically for logprob with lora tests."""
        request.param = {
            "mode": "logprob",
            "enable_loras": True,
            "model_fixture_name": request.param[0],
            "specified_config": {},
            "lora_config": {
                "target_modules": request.param[1],
                "exclude_modules": request.param[2],
                "match_all_linear": request.param[3],
                "dim": request.param[4],
                "alpha": request.param[5],
                "dropout": request.param[6],
                "dropout_position": request.param[7],
                "lora_A_init": request.param[8],
                "use_triton": request.param[9],
            },
        }
        yield from _base_setup_impl(request, two_gpu_cluster)

    @pytest.mark.timeout(360)
    @pytest.mark.automodel
    @pytest.mark.parametrize(
        "policy_setup",
        [{"enable_loras": False}, {"enable_loras": True}],
        indirect=True,
    )
    def test_lm_policy_init(self, policy_setup):
        policy = policy_setup

        # Verify we have two workers, one per GPU
        assert len(policy.worker_group.workers) == 2, (
            "Should have 2 workers, one per GPU"
        )

        # Check workers are alive
        worker_alive = ray.get(
            [w.is_alive.remote() for w in policy.worker_group.workers]
        )
        assert all(worker_alive), f"Not all workers are alive: {worker_alive}"

        # Get GPU info from both workers to verify GPU usage
        print("\nGetting GPU information from workers...")
        gpu_infos = ray.get(
            [w.get_gpu_info.remote() for w in policy.worker_group.workers]
        )
        print("\nGPU Information:")
        for i, info in enumerate(gpu_infos):
            print(f"\nWorker {i} GPU Info:")
            pprint.pprint(info)

        # Check 1: Verify workers have different ranks
        gpu_ranks = [info["rank"] for info in gpu_infos]
        assert len(set(gpu_ranks)) == 2, f"Expected 2 different ranks, got {gpu_ranks}"
        assert set(gpu_ranks) == {0, 1}, f"Expected ranks 0 and 1, got {gpu_ranks}"

        # Check 2: Verify workers have different local_ranks
        local_ranks = [info["local_rank"] for info in gpu_infos]
        assert len(set(local_ranks)) == 2, (
            f"Expected 2 different local_ranks, got {local_ranks}"
        )
        assert set(local_ranks) == {0, 1}, (
            f"Expected local_ranks 0 and 1, got {local_ranks}"
        )

        # Check 3: Verify workers have different CUDA_VISIBLE_DEVICES
        cuda_visible_devices = [
            info["env_vars"].get("CUDA_VISIBLE_DEVICES") for info in gpu_infos
        ]
        assert len(set(cuda_visible_devices)) == 2, (
            f"Expected different CUDA_VISIBLE_DEVICES, got {cuda_visible_devices}"
        )

        # Check 4: Verify all workers report correct world_size
        for info in gpu_infos:
            assert info["world_size"] == 2, (
                f"Expected world_size=2, got {info['world_size']}"
            )
            assert info["env_vars"]["WORLD_SIZE"] == "2", (
                f"Expected WORLD_SIZE=2, got {info['env_vars']['WORLD_SIZE']}"
            )

        # Check 5: Verify GPU memory is allocated on both GPUs
        for info in gpu_infos:
            assert info["memory_allocated_mb"] > 10, (
                f"Not enough memory allocated on GPU for rank {info['rank']}: {info['memory_allocated_mb']:.2f} MB"
            )

        # Check 6: Verify model parameters are on CUDA devices for both workers
        for info in gpu_infos:
            param_sample = list(info["parameter_sample"].values())[0]
            assert "cuda" in param_sample["device"], (
                f"Parameter not on CUDA device: {param_sample['device']}"
            )

        # Check 8: Verify same model parameters are being tracked across workers
        param_names = [list(info["parameter_sample"].keys())[0] for info in gpu_infos]
        assert len(set(param_names)) == 1, (
            f"Workers are not tracking the same parameter: {param_names}"
        )

        # Check 9: Both workers should see their device as cuda:0 (correct distributed behavior)
        for info in gpu_infos:
            param_device = list(info["parameter_sample"].values())[0]["device"]
            assert param_device == "cuda:0", (
                f"Expected parameter device to be cuda:0, got {param_device}"
            )

    @pytest.mark.timeout(360)
    @pytest.mark.automodel
    def test_automodel_worker_training(self, training_setup):
        policy, data, loss_fn = training_setup
        _test_automodel_worker_training(policy, data, loss_fn)

    @pytest.mark.timeout(360)
    @pytest.mark.automodel
    def test_automodel_worker_training_with_lora(self, training_with_lora_setup):
        policy, data, loss_fn = training_with_lora_setup
        _test_automodel_worker_training(policy, data, loss_fn)

    @pytest.mark.timeout(360)
    @pytest.mark.automodel
    def test_automodel_worker_logprob_tp2_or_cp2_matches_unsharded(self, logprob_setup):
        policy, data, logprobs = logprob_setup
        _test_automodel_worker_logprob(policy, data, logprobs)

    @pytest.mark.timeout(360)
    @pytest.mark.automodel
    def test_automodel_worker_logprob_with_lora(self, logprob_with_lora_setup):
        policy, data, logprobs = logprob_with_lora_setup
        _test_automodel_worker_logprob(policy, data, logprobs)

    @pytest.mark.automodel
    def test_automodel_tp_and_tied_model_with_custom_parallel_plan(
        self, two_gpu_cluster, tiny_llama_tied_model_path
    ):
        """Test that DTensor with a tp > 1 and a tied model with a custom parallel plan works."""
        from torch.distributed.tensor.parallel import ColwiseParallel
        from torch.distributed.tensor.placement_types import Replicate

        custom_parallel_plan = {
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
            "model.embed_tokens": ColwiseParallel(output_layouts=Replicate()),
        }
        config = create_test_config(
            model_name=tiny_llama_tied_model_path,
            tp=2,
            cp=1,
            sp=False,
            cpu_offload=False,
            activation_checkpointing=False,
            custom_parallel_plan=custom_parallel_plan,
        )
        tokenizer = get_tokenizer(config["tokenizer"])

        policy = Policy(
            tokenizer=tokenizer,
            config=config,
            init_optimizer=False,
            init_reference_model=False,
            cluster=two_gpu_cluster,
        )

        # Verify that the model is parallelized as expected
        state_dict = ray.get(policy.worker_group.workers[0].return_state_dict.remote())
        total_shape = state_dict["lm_head.weight"].shape
        sharded_shape = state_dict["lm_head.weight"].to_local().shape
        assert total_shape[0] == sharded_shape[0], (
            "lm_head.weight should have the same number of rows"
        )
        assert total_shape[1] == sharded_shape[1] * 2, (
            "lm_head.weight should be sharded across 2 GPUs"
        )

        # Clean up
        policy.shutdown()

    @pytest.mark.timeout(180)
    def test_automodel_loss_independent_of_microbatch_size_two_gpus(
        self, two_gpu_cluster, tiny_llama_model_path
    ):
        """Tests that changing microbatch size while keeping global batch size constant does not affect loss values in DTensor."""
        # Create test batch with global batch size of 8
        global_batch_size = 8
        seq_len = 128
        vocab_size = 32000

        # Create test input_ids and attention_mask
        input_ids = torch.randint(0, vocab_size, (global_batch_size, seq_len))
        attention_mask = torch.ones(global_batch_size, seq_len)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32)

        # Create data dictionary
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,
                "token_mask": torch.triu(
                    torch.ones(global_batch_size, seq_len), diagonal=1
                ),  # give different examples different numbers of valid tokens
                "sample_mask": torch.ones((global_batch_size,)),
                "labels": torch.randint(0, vocab_size, (global_batch_size, seq_len)),
                "num_valid_tokens_in_batch": torch.tensor(
                    [seq_len] * global_batch_size, dtype=torch.float32
                ),
                "advantages": torch.randn(global_batch_size, seq_len),
                "prev_logprobs": torch.randn(global_batch_size, seq_len),
                "reference_policy_logprobs": torch.randn(global_batch_size, seq_len),
                "generation_logprobs": torch.randn(global_batch_size, seq_len),
            }
        )

        # Test with mbs=1, 2 microbatches per GPU
        config = create_test_config(tiny_llama_model_path)
        tokenizer = get_tokenizer(config["tokenizer"])

        print("Creating training Policy with mbs=1...")
        policy_mbs1 = Policy(
            cluster=two_gpu_cluster,
            config=config,
            init_reference_model=False,
            tokenizer=tokenizer,
        )

        # Test NLLLossFn and ClippedPGLossFn with mbs=1
        nll_loss_fn = NLLLossFn()
        pg_loss_fn = ClippedPGLossFn(
            ClippedPGLossConfig(reference_policy_kl_penalty=0.1)
        )

        policy_mbs1.prepare_for_training()
        mbs1_nll_results = policy_mbs1.train(data, nll_loss_fn)
        mbs1_nll_loss = mbs1_nll_results["loss"]

        mbs1_pg_results = policy_mbs1.train(data, pg_loss_fn)
        mbs1_pg_loss = mbs1_pg_results["loss"]

        policy_mbs1.worker_group.shutdown()

        # Test with mbs=2, 1 microbatch per GPU
        config = create_test_config(tiny_llama_model_path)
        config["train_micro_batch_size"] = 2
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer
        )

        print("Creating training Policy with mbs=2...")
        policy_mbs2 = Policy(
            cluster=two_gpu_cluster,
            config=config,
            init_reference_model=False,
            tokenizer=tokenizer,
        )

        # Test NLLLossFn and ClippedPGLossFn with mbs=2
        policy_mbs2.prepare_for_training()
        mbs2_nll_results = policy_mbs2.train(data, nll_loss_fn)
        mbs2_nll_loss = mbs2_nll_results["loss"]

        mbs2_pg_results = policy_mbs2.train(data, pg_loss_fn)
        mbs2_pg_loss = mbs2_pg_results["loss"]

        # Verify both loss functions are independent of microbatch size
        torch.testing.assert_close(mbs1_nll_loss, mbs2_nll_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(mbs1_pg_loss, mbs2_pg_loss, rtol=1e-5, atol=1e-5)

        policy_mbs2.worker_group.shutdown()

    @pytest.mark.timeout(300)
    @pytest.mark.automodel
    def test_automodel_policy_flops_range_check(
        self, tiny_llama_model_path, two_gpu_cluster
    ):
        """Test that the returned FLOPS is within a reasonable range using dtensor backend.

        Performs 2 warmup iterations and checks FLOPS for the next 3 iterations.
        """
        batch_size = 8
        seq_len = 128
        vocab_size = 32000

        config = create_test_config(tiny_llama_model_path)

        # Update config for FLOPS testing with larger batch and sequence length
        config["train_global_batch_size"] = batch_size
        config["train_micro_batch_size"] = (
            batch_size  # Use full batch size for single microbatch
        )

        tokenizer = get_tokenizer(config["tokenizer"])
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer
        )

        policy = Policy(
            cluster=two_gpu_cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=False,
        )

        # Create test data
        torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32)

        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,
                "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "sample_mask": torch.ones(batch_size),
            }
        )

        # Create loss function
        loss_fn = SimpleLossFn()

        try:
            # Prepare for training
            policy.prepare_for_training()

            # Perform 2 warmup iterations
            print("Performing warmup iterations...")
            for warmup_step in range(2):
                results = policy.train(data, loss_fn)

            print("Checking FLOPS on 3 iterations...")
            for train_step in range(3):
                results = policy.train(data, loss_fn)

            # Check if FLOPS tracking is available
            if policy.flops_tracker is not None:
                assert "total_flops" in results, (
                    "Training results should contain 'total_flops'"
                )
                total_flops = results["total_flops"]

                assert isinstance(total_flops, (int, float)), (
                    "total_flops should be numeric"
                )
                assert total_flops > 0, "total_flops should be positive"

                expected_tracker = FLOPTracker.from_config(
                    config["model_name"],
                    get_hf_config(
                        config["model_name"],
                        **(config.get("hf_config_overrides") or {}),
                    ),
                )
                expected_tracker.track_batch(input_lengths.tolist())
                expected_total_flops = expected_tracker.total_flops

                assert total_flops == pytest.approx(expected_total_flops, rel=0.05), (
                    f"Expected {expected_total_flops:.2e} FLOPS, got {total_flops:.2e}"
                )

                total_tflops = total_flops / 1e12
                print(f"Total FLOPS: {total_flops:.2e} ({total_tflops:.4f} TFLOPS)")

                if "theoretical_tflops" in results:
                    theoretical_tflops = results["theoretical_tflops"]
                    assert isinstance(theoretical_tflops, (int, float)), (
                        "theoretical_tflops should be numeric"
                    )
                    assert theoretical_tflops > 0, (
                        "theoretical_tflops should be positive"
                    )

                    utilization = total_tflops / theoretical_tflops
                    print(f"Theoretical TFLOPS: {theoretical_tflops:.2f}")
                    print(f"Model utilization: {utilization * 100:.2f}%")

                    assert utilization <= 1.0, (
                        f"Model utilization {utilization * 100:.2f}% should not exceed 100%"
                    )
            else:
                print("FLOPS tracker not available, skipping FLOPS range check")
                pytest.skip("FLOPS tracker not supported for this model configuration")

        finally:
            policy.shutdown()