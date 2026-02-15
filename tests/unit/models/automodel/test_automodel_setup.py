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

"""Unit tests for automodel setup utilities."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

pytest_plugins = []
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

import torch

from nemo_rl.models.automodel.config import DistributedContext
from nemo_rl.models.automodel.setup import (
    ModelAndOptimizerState,
    RuntimeConfig,
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_prepare_config,
)


@pytest.fixture
def mock_config():
    """Create a mock policy configuration for testing."""
    return {
        "model_name": "gpt2",
        "precision": "bfloat16",
        "max_grad_norm": 1.0,
        "offload_optimizer_for_logprob": False,
        "sequence_packing": {"enabled": False},
        "dtensor_cfg": {
            "cpu_offload": False,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_parallel_size": 1,
            "sequence_parallel": False,
            "activation_checkpointing": False,
        },
        "generation": None,
        "hf_config_overrides": {},
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {"lr": 1e-4},
        },
    }


@pytest.fixture
def mock_autoconfig():
    """Create a mock AutoConfig for testing."""
    config = MagicMock()
    config.architectures = ["GPT2LMHeadModel"]
    config.model_type = "gpt2"
    config.num_labels = 2
    config.torch_dtype = "float32"
    return config


@pytest.mark.automodel
class TestValidateAndPrepareConfig:
    """Test suite for validate_and_prepare_config function."""

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_basic_validation(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test basic configuration validation returns correct values."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        result = validate_and_prepare_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        # Verify result is a RuntimeConfig named tuple
        assert isinstance(result, RuntimeConfig)
        assert result.dtype == torch.bfloat16
        assert result.cpu_offload is False
        assert result.offload_optimizer_for_logprob is False
        assert result.max_grad_norm == 1.0
        assert result.enable_seq_packing is False
        assert result.model_class is not None
        assert result.model_config is not None
        assert isinstance(result.allow_flash_attn_args, bool)

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_precision_validation_invalid(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        """Test that invalid precision raises ValueError."""
        mock_config["precision"] = "invalid_precision"

        with pytest.raises(ValueError, match="Unknown precision"):
            validate_and_prepare_config(
                config=mock_config,
                processor=None,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_sequence_packing_with_vlm_raises_error(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        """Test that sequence packing with VLM raises ValueError."""
        mock_config["sequence_packing"]["enabled"] = True
        processor = MagicMock()

        with pytest.raises(
            ValueError, match="Sequence packing is not supported for VLM"
        ):
            validate_and_prepare_config(
                config=mock_config,
                processor=processor,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.NeMoAutoModelForSequenceClassification")
    def test_reward_model_bradley_terry(
        self,
        mock_rm_class,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test reward model configuration with Bradley-Terry type."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig

        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "bradley_terry",
        }

        result = validate_and_prepare_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        # Verify num_labels was set to 1 for bradley_terry reward model
        assert mock_autoconfig.num_labels == 1
        # Result should be valid RuntimeConfig
        assert isinstance(result, RuntimeConfig)
        assert result.is_reward_model is True

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_context_parallel_with_sequence_packing_raises_error(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        """Test that CP with sequence packing raises ValueError."""
        mock_config["sequence_packing"]["enabled"] = True
        mock_config["dtensor_cfg"]["context_parallel_size"] = 2

        with pytest.raises(
            ValueError, match="Context parallel is not supported for sequence packing"
        ):
            validate_and_prepare_config(
                config=mock_config,
                processor=None,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_sequence_parallel_with_tp_size_one_prints_warning(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
        capsys,
    ):
        """Test that sequence parallel with tp = 1 prints a warning."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        mock_config["dtensor_cfg"]["sequence_parallel"] = True
        mock_config["dtensor_cfg"]["tensor_parallel_size"] = 1

        # Should not raise an error, just print a warning
        result = validate_and_prepare_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        # Verify result is valid
        assert isinstance(result, RuntimeConfig)

        # Check warning was printed
        captured = capsys.readouterr()
        assert (
            "sequence_parallel=True, but tp_size=1 which has no effect" in captured.out
        )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_attention_implementation_selection(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test attention implementation is selected correctly."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        # Test FA2 for sequence packing with cp=1
        mock_config["sequence_packing"]["enabled"] = True
        mock_config["dtensor_cfg"]["context_parallel_size"] = 1
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.attn_impl == "flash_attention_2"

        # Test SDPA for cp > 1
        mock_config["sequence_packing"]["enabled"] = False
        mock_config["dtensor_cfg"]["context_parallel_size"] = 2
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.attn_impl == "sdpa"

        # Test None for cp=1 without sequence packing
        mock_config["dtensor_cfg"]["context_parallel_size"] = 1
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.attn_impl is None

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_precision_types(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test all supported precision types."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        # Test float32
        mock_config["precision"] = "float32"
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.dtype == torch.float32

        # Test float16
        mock_config["precision"] = "float16"
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.dtype == torch.float16

        # Test bfloat16
        mock_config["precision"] = "bfloat16"
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.dtype == torch.bfloat16

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch.dict(os.environ, {}, clear=True)
    def test_generation_colocated(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test generation colocated configuration."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        # Test with generation colocated enabled
        mock_config["generation"] = {"colocated": {"enabled": True}}
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.is_generation_colocated is True
        # NCCL_CUMEM_ENABLE should not be set when colocated
        assert "NCCL_CUMEM_ENABLE" not in os.environ

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch.dict(os.environ, {}, clear=True)
    def test_generation_not_colocated(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test generation not colocated sets NCCL environment variable."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        # Test with generation colocated disabled
        mock_config["generation"] = {"colocated": {"enabled": False}}
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.is_generation_colocated is False
        # NCCL_CUMEM_ENABLE should be set when not colocated
        assert os.environ.get("NCCL_CUMEM_ENABLE") == "1"

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_allow_flash_attn_args_nemotron_nas(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        """Test flash attention args disabled for Nemotron NAS."""
        mock_autoconfig = MagicMock()
        mock_autoconfig.architectures = ["DeciLMForCausalLM"]
        mock_autoconfig.model_type = "nemotron-nas"
        mock_autoconfig.torch_dtype = "float32"
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.allow_flash_attn_args is False

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_sequence_packing_with_reward_model_raises_error(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test that sequence packing with reward model raises NotImplementedError."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_config["sequence_packing"]["enabled"] = True
        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "bradley_terry",
        }

        with pytest.raises(
            NotImplementedError,
            match="Sequence packing is not supported for reward models",
        ):
            validate_and_prepare_config(
                config=mock_config,
                processor=None,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_unknown_reward_model_type_raises_error(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test that unknown reward model type raises ValueError."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "unknown_type",
        }

        with pytest.raises(ValueError, match="Unknown reward model type: unknown_type"):
            validate_and_prepare_config(
                config=mock_config,
                processor=None,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.NeMoAutoModelForSequenceClassification")
    def test_reward_model_bradley_terry_num_labels_already_one(
        self,
        mock_rm_class,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        capsys,
    ):
        """Test reward model with num_labels already set to 1 does not print warning."""
        mock_autoconfig = MagicMock()
        mock_autoconfig.architectures = ["GPT2LMHeadModel"]
        mock_autoconfig.model_type = "gpt2"
        mock_autoconfig.num_labels = 1  # Already 1
        mock_autoconfig.torch_dtype = "float32"
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig

        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "bradley_terry",
        }

        result = validate_and_prepare_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        # Should not print the warning about num_labels
        captured = capsys.readouterr()
        assert "model_config.num_labels is not 1" not in captured.out
        assert result.is_reward_model is True

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_sequence_packing_enabled_prints_info(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
        capsys,
    ):
        """Test that sequence packing enabled prints info messages."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        mock_config["sequence_packing"]["enabled"] = True
        mock_config["dtensor_cfg"]["context_parallel_size"] = 1

        result = validate_and_prepare_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        captured = capsys.readouterr()
        assert "[Rank 0] Sequence packing is enabled for model gpt2" in captured.out
        assert "[Rank 0] Using FlashAttention2 for sequence packing" in captured.out
        assert result.enable_seq_packing is True

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_hf_config_overrides_none_becomes_empty_dict(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test that None hf_config_overrides becomes empty dict."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        mock_config["hf_config_overrides"] = None

        result = validate_and_prepare_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        assert result.hf_config_overrides == {}

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    def test_missing_hf_config_overrides_becomes_empty_dict(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        """Test that missing hf_config_overrides becomes empty dict."""
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        del mock_config["hf_config_overrides"]

        result = validate_and_prepare_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        assert result.hf_config_overrides == {}


@pytest.mark.automodel
class TestSetupReferenceModelState:
    """Test suite for setup_reference_model_state function."""

    @patch("nemo_rl.models.automodel.setup.get_cpu_state_dict")
    def test_setup_reference_model_state_calls_get_cpu_state_dict(
        self, mock_get_cpu_state_dict
    ):
        """Test that setup_reference_model_state calls get_cpu_state_dict correctly."""
        mock_model = MagicMock()
        mock_state_dict = {
            "weight1": torch.tensor([1.0]),
            "weight2": torch.tensor([2.0]),
        }
        mock_model.state_dict.return_value = mock_state_dict
        mock_get_cpu_state_dict.return_value = {"weight1": torch.tensor([1.0])}

        result = setup_reference_model_state(mock_model)

        mock_model.state_dict.assert_called_once()
        mock_get_cpu_state_dict.assert_called_once()
        # Verify pin_memory=True was passed
        call_kwargs = mock_get_cpu_state_dict.call_args[1]
        assert call_kwargs["pin_memory"] is True
        assert result == {"weight1": torch.tensor([1.0])}

    @patch("nemo_rl.models.automodel.setup.get_cpu_state_dict")
    def test_setup_reference_model_state_returns_dict(self, mock_get_cpu_state_dict):
        """Test that setup_reference_model_state returns a dictionary."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        expected_result = {"param": torch.zeros(10)}
        mock_get_cpu_state_dict.return_value = expected_result

        result = setup_reference_model_state(mock_model)

        assert result == expected_result


@pytest.mark.automodel
class TestSetupDistributed:
    """Test suite for setup_distributed function."""

    @pytest.fixture
    def mock_runtime_config(self):
        """Create a mock RuntimeConfig for testing."""
        return RuntimeConfig(
            model_class=Mock,
            model_config=MagicMock(),
            hf_config_overrides={},
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=False,
        )

    @pytest.fixture
    def mock_device_mesh(self):
        """Create a mock device mesh with subscriptable dimension sizes."""
        mock_mesh = MagicMock()
        # Configure dimension subscript access
        dp_dim = MagicMock()
        dp_dim.size.return_value = 4
        tp_dim = MagicMock()
        tp_dim.size.return_value = 1
        cp_dim = MagicMock()
        cp_dim.size.return_value = 1

        mock_mesh.__getitem__ = lambda self, key: {
            "dp": dp_dim,
            "tp": tp_dim,
            "cp": cp_dim,
        }[key]
        return mock_mesh

    @patch("nemo_rl.models.automodel.setup.MoEParallelizerConfig")
    @patch("nemo_rl.models.automodel.setup.create_device_mesh")
    @patch("nemo_rl.models.automodel.setup.FSDP2Config")
    @patch("nemo_rl.models.automodel.setup.torch.distributed")
    def test_setup_distributed_basic(
        self,
        mock_torch_dist,
        mock_fsdp2_config,
        mock_create_mesh,
        mock_moe_config,
        mock_config,
        mock_runtime_config,
        mock_device_mesh,
    ):
        """Test basic distributed setup without CPU offload."""
        mock_torch_dist.get_world_size.return_value = 8
        mock_fsdp2_config_instance = MagicMock()
        mock_fsdp2_config.return_value = mock_fsdp2_config_instance
        mock_moe_config_instance = MagicMock()
        mock_moe_config.return_value = mock_moe_config_instance
        mock_moe_mesh = MagicMock()
        mock_create_mesh.return_value = (mock_device_mesh, mock_moe_mesh)

        result = setup_distributed(mock_config, mock_runtime_config)

        mock_torch_dist.init_process_group.assert_called_once_with(backend="nccl")
        assert isinstance(result, DistributedContext)
        assert result.device_mesh == mock_device_mesh
        assert result.moe_mesh == mock_moe_mesh
        assert result.fsdp2_config == mock_fsdp2_config_instance
        assert result.moe_config == mock_moe_config_instance

    @patch("nemo_rl.models.automodel.setup.MoEParallelizerConfig")
    @patch("nemo_rl.models.automodel.setup.create_device_mesh")
    @patch("nemo_rl.models.automodel.setup.FSDP2Config")
    @patch("nemo_rl.models.automodel.setup.torch.distributed")
    def test_setup_distributed_with_cpu_offload(
        self,
        mock_torch_dist,
        mock_fsdp2_config,
        mock_create_mesh,
        mock_moe_config,
        mock_config,
        mock_device_mesh,
    ):
        """Test distributed setup with CPU offload."""
        mock_torch_dist.get_world_size.return_value = 4
        mock_fsdp2_config.return_value = MagicMock()
        mock_moe_config.return_value = MagicMock()
        mock_create_mesh.return_value = (mock_device_mesh, None)

        runtime_config = RuntimeConfig(
            model_class=Mock,
            model_config=MagicMock(),
            hf_config_overrides={},
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=True,  # CPU offload enabled
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=False,
        )

        result = setup_distributed(mock_config, runtime_config)

        mock_torch_dist.init_process_group.assert_called_once_with(
            backend="cuda:nccl,cpu:gloo"
        )
        assert isinstance(result, DistributedContext)

    @patch("nemo_rl.models.automodel.setup.MoEParallelizerConfig")
    @patch("nemo_rl.models.automodel.setup.create_device_mesh")
    @patch("nemo_rl.models.automodel.setup.FSDP2Config")
    @patch("nemo_rl.models.automodel.setup.torch.distributed")
    def test_setup_distributed_world_size_one_cpu_offload_raises(
        self,
        mock_torch_dist,
        mock_fsdp2_config,
        mock_create_mesh,
        mock_moe_config,
        mock_config,
    ):
        """Test that world_size=1 with cpu_offload raises NotImplementedError."""
        mock_torch_dist.get_world_size.return_value = 1
        mock_fsdp2_config.return_value = MagicMock()
        mock_moe_config.return_value = MagicMock()

        runtime_config = RuntimeConfig(
            model_class=Mock,
            model_config=MagicMock(),
            hf_config_overrides={},
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=True,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=False,
        )

        with pytest.raises(
            NotImplementedError, match="CPUOffload doesn't work on single GPU"
        ):
            setup_distributed(mock_config, runtime_config)

    @patch("nemo_rl.models.automodel.setup.MoEParallelizerConfig")
    @patch("nemo_rl.models.automodel.setup.create_device_mesh")
    @patch("nemo_rl.models.automodel.setup.FSDP2Config")
    @patch("nemo_rl.models.automodel.setup.torch.distributed")
    def test_setup_distributed_passes_correct_params(
        self,
        mock_torch_dist,
        mock_fsdp2_config,
        mock_create_mesh,
        mock_moe_config,
        mock_config,
        mock_runtime_config,
        mock_device_mesh,
    ):
        """Test that FSDP2Config and create_device_mesh are called with correct parameters."""
        mock_torch_dist.get_world_size.return_value = 4
        mock_fsdp2_config.return_value = MagicMock()
        mock_moe_config.return_value = MagicMock()
        mock_create_mesh.return_value = (mock_device_mesh, None)

        setup_distributed(mock_config, mock_runtime_config)

        # Verify FSDP2Config was constructed with correct kwargs
        fsdp2_call_kwargs = mock_fsdp2_config.call_args[1]
        assert fsdp2_call_kwargs["sequence_parallel"] is False
        assert fsdp2_call_kwargs["activation_checkpointing"] is False
        assert fsdp2_call_kwargs["backend"] == "nccl"

        # Verify create_device_mesh was called with correct size params
        mesh_call_kwargs = mock_create_mesh.call_args[1]
        assert mesh_call_kwargs["tp_size"] == 1
        assert mesh_call_kwargs["pp_size"] == 1
        assert mesh_call_kwargs["cp_size"] == 1
        assert mesh_call_kwargs["ep_size"] == 1
        assert mesh_call_kwargs["world_size"] == 4


@pytest.mark.automodel
class TestSetupModelAndOptimizer:
    """Test suite for setup_model_and_optimizer function."""

    @pytest.fixture
    def mock_runtime_config(self, mock_autoconfig):
        """Create a mock RuntimeConfig for testing."""
        return RuntimeConfig(
            model_class=MagicMock(),
            model_config=mock_autoconfig,
            hf_config_overrides={},
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=False,
        )

    @pytest.fixture
    def mock_distributed_context(self):
        """Create a mock DistributedContext for testing."""
        mock_fsdp2_config = MagicMock()
        mock_fsdp2_config.sequence_parallel = False
        return DistributedContext(
            device_mesh=MagicMock(),
            moe_mesh=MagicMock(),
            fsdp2_config=mock_fsdp2_config,
            moe_config=MagicMock(),
            dp_size=1,
            tp_size=1,
            cp_size=1,
        )

    @pytest.fixture
    def mock_checkpoint_manager(self):
        """Create a mock checkpoint manager for testing."""
        return MagicMock()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        return tokenizer

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_and_optimizer_basic(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test basic model and optimizer setup."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        # Setup mock model
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.zeros(10)}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = None
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
            is_vlm=False,
            init_optimizer=True,
        )

        assert isinstance(result, ModelAndOptimizerState)
        # Verify from_pretrained was called with distributed kwargs
        mock_runtime_config.model_class.from_pretrained.assert_called_once()
        call_kwargs = mock_runtime_config.model_class.from_pretrained.call_args[1]
        assert call_kwargs["device_mesh"] == mock_distributed_context.device_mesh
        assert (
            call_kwargs["distributed_config"] == mock_distributed_context.fsdp2_config
        )
        # Verify config= is NOT passed (avoids duplicate arg for custom models)
        assert "config" not in call_kwargs

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_passes_hf_config_overrides_as_flat_kwargs(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that hf_config_overrides are passed as flat kwargs to from_pretrained."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0

        # Set hf_config_overrides on runtime_config
        overrides = {
            "rope_scaling": {"type": "linear", "factor": 2.0},
            "max_position_embeddings": 4096,
        }
        runtime_config = RuntimeConfig(
            model_class=MagicMock(),
            model_config=mock_runtime_config.model_config,
            hf_config_overrides=overrides,
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=False,
        )
        runtime_config.model_class.from_pretrained.return_value = mock_model
        runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        call_kwargs = runtime_config.model_class.from_pretrained.call_args[1]
        # hf_config_overrides should be passed as flat kwargs
        assert call_kwargs["rope_scaling"] == {"type": "linear", "factor": 2.0}
        assert call_kwargs["max_position_embeddings"] == 4096
        # config= should NOT be passed
        assert "config" not in call_kwargs

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_reward_model_passes_num_labels(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_autoconfig,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that reward model passes num_labels=1 to from_pretrained."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0

        # Configure as reward model with num_labels already set to 1
        # (validate_and_prepare_config sets this)
        mock_autoconfig.num_labels = 1
        runtime_config = RuntimeConfig(
            model_class=MagicMock(),
            model_config=mock_autoconfig,
            hf_config_overrides={},
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=True,
        )
        runtime_config.model_class.from_pretrained.return_value = mock_model
        runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        call_kwargs = runtime_config.model_class.from_pretrained.call_args[1]
        assert call_kwargs["num_labels"] == 1

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_non_reward_model_does_not_pass_num_labels(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that non-reward model does not pass num_labels to from_pretrained."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        call_kwargs = mock_runtime_config.model_class.from_pretrained.call_args[1]
        assert "num_labels" not in call_kwargs

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_reward_model_with_hf_config_overrides(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_autoconfig,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that reward model correctly combines hf_config_overrides with num_labels."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0

        mock_autoconfig.num_labels = 1
        overrides = {"max_position_embeddings": 4096}
        runtime_config = RuntimeConfig(
            model_class=MagicMock(),
            model_config=mock_autoconfig,
            hf_config_overrides=overrides,
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=True,
        )
        runtime_config.model_class.from_pretrained.return_value = mock_model
        runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        call_kwargs = runtime_config.model_class.from_pretrained.call_args[1]
        # Both overrides and num_labels should be present
        assert call_kwargs["num_labels"] == 1
        assert call_kwargs["max_position_embeddings"] == 4096
        assert "config" not in call_kwargs

    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_and_optimizer_no_optimizer(
        self,
        mock_get_class,
        mock_get_rank,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup without optimizer initialization."""
        mock_get_rank.return_value = 0

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
            init_optimizer=False,
        )

        assert result.optimizer is None
        assert result.scheduler is None

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_weights_path(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with checkpoint loading."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
            weights_path="/path/to/weights",
            optimizer_path="/path/to/optimizer",
        )

        mock_checkpoint_manager.load_checkpoint.assert_called_once()
        call_kwargs = mock_checkpoint_manager.load_checkpoint.call_args[1]
        assert call_kwargs["weights_path"] == "/path/to/weights"
        assert call_kwargs["optimizer_path"] == "/path/to/optimizer"

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_no_weights_path_prints_message(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
        capsys,
    ):
        """Test that no weights path prints info message."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
            weights_path=None,
        )

        captured = capsys.readouterr()
        assert "No weights path provided" in captured.out

    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_dict_scheduler(
        self,
        mock_get_class,
        mock_get_rank,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with scheduler as dict config."""
        mock_get_rank.return_value = 0

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        def get_class_side_effect(name):
            if "optim" in name.lower():
                return MagicMock(return_value=mock_optimizer)
            return MagicMock(return_value=mock_scheduler)

        mock_get_class.side_effect = get_class_side_effect

        mock_config["scheduler"] = {
            "name": "torch.optim.lr_scheduler.StepLR",
            "kwargs": {"step_size": 10},
        }

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert result.scheduler is not None

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.SequentialLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_list_scheduler(
        self,
        mock_get_class,
        mock_get_rank,
        mock_sequential_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with scheduler as list config (SequentialLR)."""
        mock_get_rank.return_value = 0
        mock_sequential_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        def get_class_side_effect(name):
            if "optim.Adam" in name or "optim.SGD" in name:
                return MagicMock(return_value=mock_optimizer)
            return MagicMock(return_value=mock_scheduler)

        mock_get_class.side_effect = get_class_side_effect

        mock_config["scheduler"] = [
            {
                "name": "torch.optim.lr_scheduler.LinearLR",
                "kwargs": {"start_factor": 0.1},
            },
            {"name": "torch.optim.lr_scheduler.StepLR", "kwargs": {"step_size": 10}},
            {"milestones": [5]},
        ]

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert result.scheduler is not None
        mock_sequential_lr.assert_called_once()

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_sets_pad_token_id(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that pad_token_id is set from tokenizer when None."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = None  # Initially None
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]
        mock_tokenizer.pad_token_id = 42

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert mock_model.config.pad_token_id == 42

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_moe_model(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup detects MoE model correctly."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        # Include "expert" in state dict keys to trigger MoE detection
        mock_model.state_dict.return_value = {
            "layer.expert.weight": torch.zeros(10),
            "layer.weight": torch.zeros(10),
        }
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert result.is_moe_model is True

    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_cp_raises_for_vlm(
        self,
        mock_get_class,
        mock_get_rank,
        mock_config,
        mock_runtime_config,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that context parallel with VLM raises AssertionError."""
        mock_get_rank.return_value = 0
        mock_fsdp2_config = MagicMock()
        mock_fsdp2_config.sequence_parallel = False
        distributed_context = DistributedContext(
            device_mesh=MagicMock(),
            moe_mesh=MagicMock(),
            fsdp2_config=mock_fsdp2_config,
            moe_config=MagicMock(),
            dp_size=1,
            tp_size=1,
            cp_size=2,  # CP enabled
        )

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        with pytest.raises(
            AssertionError, match="Context parallel is yet not supported for VLM models"
        ):
            setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=mock_runtime_config,
                distributed_context=distributed_context,
                checkpoint_manager=mock_checkpoint_manager,
                is_vlm=True,
            )

    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_cp_and_sp_raises_error(
        self,
        mock_get_class,
        mock_get_rank,
        mock_config,
        mock_runtime_config,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that CP with sequence parallel raises AssertionError."""
        mock_get_rank.return_value = 0
        mock_fsdp2_config = MagicMock()
        mock_fsdp2_config.sequence_parallel = True  # SP enabled
        distributed_context = DistributedContext(
            device_mesh=MagicMock(),
            moe_mesh=MagicMock(),
            fsdp2_config=mock_fsdp2_config,
            moe_config=MagicMock(),
            dp_size=1,
            tp_size=2,  # TP enabled
            cp_size=2,  # CP enabled
        )

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        with pytest.raises(
            AssertionError,
            match="context parallel can't be used together with sequence parallel",
        ):
            setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=mock_runtime_config,
                distributed_context=distributed_context,
                checkpoint_manager=mock_checkpoint_manager,
            )

    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_cp_raises_for_gemma3(
        self,
        mock_get_class,
        mock_get_rank,
        mock_config,
        mock_runtime_config,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test that context parallel with Gemma3 raises AssertionError."""
        mock_get_rank.return_value = 0
        mock_fsdp2_config = MagicMock()
        mock_fsdp2_config.sequence_parallel = False
        distributed_context = DistributedContext(
            device_mesh=MagicMock(),
            moe_mesh=MagicMock(),
            fsdp2_config=mock_fsdp2_config,
            moe_config=MagicMock(),
            dp_size=1,
            tp_size=1,
            cp_size=2,  # CP enabled
        )

        # Set model_type to gemma3 to trigger validation
        mock_runtime_config.model_config.model_type = "gemma3"
        mock_runtime_config.model_config.architectures = ["Gemma3ForCausalLM"]

        with pytest.raises(
            AssertionError,
            match="Context parallel is not supported for Gemma3ForCausalLM",
        ):
            setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=mock_runtime_config,
                distributed_context=distributed_context,
                checkpoint_manager=mock_checkpoint_manager,
            )

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    @patch("nemo_rl.models.automodel.setup._resolve_target")
    def test_setup_model_with_backend_automodel_kwargs(
        self,
        mock_resolve_target,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with custom backend in automodel_kwargs."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_backend_class = MagicMock()
        mock_resolve_target.return_value = mock_backend_class

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        mock_config["dtensor_cfg"]["automodel_kwargs"] = {
            "backend": {
                "_target_": "some.backend.Class",
                "param1": "value1",
            }
        }

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        mock_resolve_target.assert_called_once_with("some.backend.Class")
        mock_backend_class.assert_called_once_with(param1="value1")

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    @patch("nemo_rl.models.automodel.setup.PeftConfig")
    def test_setup_model_with_lora(
        self,
        mock_peft_config,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with LoRA enabled."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_peft_config_instance = MagicMock()
        mock_peft_config_instance.lora_A_init = "kaiming"
        mock_peft_config.from_dict.return_value = mock_peft_config_instance

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        mock_config["dtensor_cfg"]["lora_cfg"] = {
            "enabled": True,
            "use_triton": False,
            "rank": 8,
        }

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        mock_peft_config.from_dict.assert_called_once()
        # Verify peft_config was passed to from_pretrained
        call_kwargs = mock_runtime_config.model_class.from_pretrained.call_args[1]
        assert call_kwargs["peft_config"] == mock_peft_config_instance
        assert result.peft_config == mock_peft_config_instance

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    @patch("nemo_rl.models.automodel.setup.cuda", create=True)
    def test_setup_model_with_activation_checkpointing(
        self,
        mock_cuda,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with activation checkpointing enabled."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        mock_config["dtensor_cfg"]["activation_checkpointing"] = True

        with patch(
            "nemo_rl.models.automodel.setup.torch.backends.cuda"
        ) as mock_torch_cuda:
            setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=mock_runtime_config,
                distributed_context=mock_distributed_context,
                checkpoint_manager=mock_checkpoint_manager,
            )

            mock_torch_cuda.enable_cudnn_sdp.assert_called_with(False)

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_tied_word_embeddings(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with tied word embeddings."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        mock_embed_weight = torch.nn.Parameter(torch.zeros(100, 768))
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_model.config.tie_word_embeddings = True
        mock_model.lm_head = MagicMock()

        # Setup named_parameters to return embed_tokens
        mock_model.named_parameters.return_value = [
            ("model.embed_tokens.weight", mock_embed_weight),
            ("lm_head.weight", torch.nn.Parameter(torch.zeros(100, 768))),
        ]

        mock_runtime_config.model_class.from_pretrained.return_value = mock_model
        mock_runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Verify lm_head.weight was set to embed_tokens weight
        assert mock_model.lm_head.weight is mock_embed_weight

    @patch("nemo_rl.models.automodel.setup.torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.get_class")
    def test_setup_model_with_cpu_offload(
        self,
        mock_get_class,
        mock_get_rank,
        mock_lambda_lr,
        mock_config,
        mock_autoconfig,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with CPU offload."""
        mock_get_rank.return_value = 0
        mock_lambda_lr.return_value = MagicMock()

        runtime_config = RuntimeConfig(
            model_class=MagicMock(),
            model_config=mock_autoconfig,
            hf_config_overrides={},
            allow_flash_attn_args=True,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=True,  # CPU offload enabled
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            is_reward_model=False,
        )

        mock_buffer = MagicMock()
        mock_buffer.data = MagicMock()
        mock_buffer.data.to.return_value = mock_buffer.data

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_model.buffers.return_value = [mock_buffer]
        runtime_config.model_class.from_pretrained.return_value = mock_model
        runtime_config.model_config.architectures = ["GPT2LMHeadModel"]

        mock_optimizer = MagicMock()
        mock_get_class.return_value = MagicMock(return_value=mock_optimizer)

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Verify buffers were moved to CPU
        mock_buffer.data.to.assert_called_with("cpu")
        # Verify model was moved to CPU
        mock_model.to.assert_called_with("cpu")
