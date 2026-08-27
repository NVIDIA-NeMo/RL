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
from unittest.mock import MagicMock, Mock, create_autospec, patch

import pytest

pytest_plugins = []
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

import torch
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager
from nemo_rl.models.automodel.config import DistributedContext
from nemo_rl.models.automodel.setup import (
    ModelAndOptimizerState,
    RuntimeConfig,
    _maybe_set_force_hf,
    get_tokenizer,
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_prepare_config,
)


@pytest.mark.automodel
def test_token_classification_backport_still_required():
    with pytest.raises(ImportError):
        from nemo_automodel import NeMoAutoModelForTokenClassification  # noqa: F401


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
        "generation": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "colocated": {"enabled": True},
        },
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
        mock_config["generation"]["colocated"]["enabled"] = True
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

        mock_config["generation"]["backend"] = "vllm"
        mock_config["generation"]["colocated"]["enabled"] = False
        result = validate_and_prepare_config(mock_config, None, 0)
        assert result.is_generation_colocated is False
        # NCCL_CUMEM_ENABLE should be set when not colocated
        assert os.environ.get("NCCL_CUMEM_ENABLE") == "1"

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch.dict(os.environ, {}, clear=True)
    def test_generation_sglang_not_colocated(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock
        mock_config["generation"]["backend"] = "sglang"
        mock_config["generation"]["colocated"]["enabled"] = False

        result = validate_and_prepare_config(mock_config, None, 0)

        assert result.is_generation_colocated is False
        assert os.environ.get("NCCL_CUMEM_ENABLE") == "0"

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch.dict(os.environ, {}, clear=True)
    def test_no_generation_leaves_nccl_cumem_unset(
        self,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock
        del mock_config["generation"]

        result = validate_and_prepare_config(mock_config, None, 0)

        assert result.is_generation_colocated is None
        assert "NCCL_CUMEM_ENABLE" not in os.environ

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
            sampling_params=None,
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
            sampling_params=None,
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
            sampling_params=None,
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
        mock_config["dtensor_cfg"]["dp_replicate_size"] = 2

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
        assert mesh_call_kwargs["dp_replicate_size"] == 2
        assert mesh_call_kwargs["world_size"] == 4

    @patch("nemo_rl.models.automodel.setup.MoEParallelizerConfig")
    @patch("nemo_rl.models.automodel.setup.create_device_mesh")
    @patch("nemo_rl.models.automodel.setup.FSDP2Config")
    @patch("nemo_rl.models.automodel.setup.torch.distributed")
    def test_setup_distributed_dp_replicate_size_requires_divisible_dp(
        self,
        mock_torch_dist,
        mock_fsdp2_config,
        mock_create_mesh,
        mock_moe_config,
        mock_config,
        mock_runtime_config,
    ):
        """dp_replicate_size must divide the inferred data-parallel size."""
        mock_torch_dist.get_world_size.return_value = 6
        mock_fsdp2_config.return_value = MagicMock()
        mock_moe_config.return_value = MagicMock()
        mock_config["dtensor_cfg"]["dp_replicate_size"] = 4

        with pytest.raises(ValueError, match="dp_replicate_size"):
            setup_distributed(mock_config, mock_runtime_config)


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
            sampling_params=None,
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

    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    def test_restore_from_without_lora_enabled_raises(
        self,
        mock_get_rank,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """restore_from with LoRA disabled must fail loudly, not silently no-op."""
        mock_get_rank.return_value = 0
        mock_config["dtensor_cfg"]["lora_cfg"] = {
            "enabled": False,
            "restore_from": "/donor/step_5/policy/weights",
        }

        with pytest.raises(ValueError, match="lora_cfg.restore_from is set"):
            setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=mock_runtime_config,
                distributed_context=mock_distributed_context,
                checkpoint_manager=mock_checkpoint_manager,
            )

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
            sampling_params=None,
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
            sampling_params=None,
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
            sampling_params=None,
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

    @pytest.mark.hf_gated
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    def test_setup_model_with_tied_word_embeddings(
        self,
        mock_get_rank,
        mock_config,
        mock_runtime_config,
        mock_distributed_context,
        mock_checkpoint_manager,
        mock_tokenizer,
    ):
        """Test model setup with tied word embeddings."""
        from transformers import AutoModelForCausalLM

        # Mock the rank to be 0
        mock_get_rank.return_value = 0

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        mock_runtime_config.model_class.from_pretrained.return_value = model

        setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=mock_runtime_config,
            distributed_context=mock_distributed_context,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Verify lm_head.weight was set to embed_tokens weight
        assert (
            model.lm_head.weight.data_ptr()
            == model.get_input_embeddings().weight.data_ptr()
        )

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
            sampling_params=None,
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


@pytest.mark.automodel
class TestGetTokenizer:
    """Test suite for get_tokenizer function."""

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_basic_tokenizer_loading(self, mock_nemo_auto_tokenizer):
        """Test basic tokenizer loading uses NeMoAutoTokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        result = get_tokenizer({"name": "gpt2"})

        mock_nemo_auto_tokenizer.from_pretrained.assert_called_once_with(
            "gpt2", trust_remote_code=True
        )
        assert result is mock_tokenizer

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_forwards_tokenizer_kwargs(self, mock_nemo_auto_tokenizer):
        """Test tokenizer_kwargs are forwarded to NeMoAutoTokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer({"name": "gpt2", "tokenizer_kwargs": {"model_max_length": 123}})

        mock_nemo_auto_tokenizer.from_pretrained.assert_called_once_with(
            "gpt2", trust_remote_code=True, model_max_length=123
        )

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_sets_pad_token_from_eos(self, mock_nemo_auto_tokenizer):
        """Test that pad_token is set to eos_token when None."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer({"name": "gpt2"})

        assert mock_tokenizer.pad_token == "<eos>"

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_does_not_override_existing_pad_token(self, mock_nemo_auto_tokenizer):
        """Test that existing pad_token is not overridden."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer({"name": "gpt2"})

        assert mock_tokenizer.pad_token == "<pad>"

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_passthrough_chat_template(self, mock_nemo_auto_tokenizer, capsys):
        """Test that chat_template=None sets passthrough template."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer({"name": "gpt2", "chat_template": None})

        captured = capsys.readouterr()
        assert "Using passthrough chat template" in captured.out
        assert (
            mock_tokenizer.chat_template
            == "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_default_chat_template(self, mock_nemo_auto_tokenizer, capsys):
        """Test that chat_template='default' keeps tokenizer's default."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.chat_template = "original_template"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer({"name": "gpt2", "chat_template": "default"})

        captured = capsys.readouterr()
        assert "Using tokenizer's default chat template" in captured.out
        assert mock_tokenizer.chat_template == "original_template"

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_custom_chat_template(self, mock_nemo_auto_tokenizer, capsys):
        """Test that a custom chat template string is set."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        custom_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        get_tokenizer({"name": "gpt2", "chat_template": custom_template})

        captured = capsys.readouterr()
        assert "Using custom chat template" in captured.out
        assert mock_tokenizer.chat_template == custom_template

    @patch("builtins.open", create=True)
    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_jinja_file_chat_template(
        self, mock_nemo_auto_tokenizer, mock_open, capsys
    ):
        """Test that .jinja file template is loaded from file."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.read.return_value = "template_from_file"

        get_tokenizer({"name": "gpt2", "chat_template": "/path/to/template.jinja"})

        captured = capsys.readouterr()
        assert "Loading chat template from file" in captured.out
        mock_open.assert_called_once_with("/path/to/template.jinja", "r")
        assert mock_tokenizer.chat_template == "template_from_file"

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_no_chat_template_key(self, mock_nemo_auto_tokenizer, capsys):
        """Test that missing chat_template key uses tokenizer's default."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer({"name": "gpt2"})

        captured = capsys.readouterr()
        assert "No chat template provided, using tokenizer's default" in captured.out

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_chat_template_kwargs(self, mock_nemo_auto_tokenizer):
        """Test that chat_template_kwargs are applied via functools.partial."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        original_apply = mock_tokenizer.apply_chat_template
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer(
            {
                "name": "gpt2",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )

        # apply_chat_template should be wrapped with partial
        assert mock_tokenizer.apply_chat_template is not original_apply

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_chat_template_kwargs_none_is_ignored(self, mock_nemo_auto_tokenizer):
        """Test that chat_template_kwargs=None does not wrap apply_chat_template."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        original_apply = mock_tokenizer.apply_chat_template
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        get_tokenizer(
            {
                "name": "gpt2",
                "chat_template_kwargs": None,
            }
        )

        assert mock_tokenizer.apply_chat_template is original_apply

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_chat_template_kwargs_invalid_type_raises(self, mock_nemo_auto_tokenizer):
        """Test that non-dict chat_template_kwargs raises assertion."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with pytest.raises(AssertionError, match="chat_template_kwargs should be"):
            get_tokenizer(
                {
                    "name": "gpt2",
                    "chat_template_kwargs": "not_a_dict",
                }
            )

    @patch("nemo_rl.models.automodel.setup.AutoProcessor")
    def test_get_processor(self, mock_auto_processor):
        """Test that get_processor=True returns an AutoProcessor."""
        mock_processor = MagicMock()
        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.pad_token = "<pad>"
        mock_inner_tokenizer.eos_token = "<eos>"
        mock_inner_tokenizer.bos_token = "<bos>"
        mock_inner_tokenizer.pad_token_id = 0
        mock_inner_tokenizer.eos_token_id = 1
        mock_inner_tokenizer.bos_token_id = 2
        mock_inner_tokenizer.name_or_path = "test-model"
        mock_processor.tokenizer = mock_inner_tokenizer
        mock_auto_processor.from_pretrained.return_value = mock_processor

        result = get_tokenizer({"name": "test-vlm"}, get_processor=True)

        mock_auto_processor.from_pretrained.assert_called_once_with(
            "test-vlm", trust_remote_code=True, use_fast=True
        )
        assert result is mock_processor
        assert mock_processor.pad_token == "<pad>"
        assert mock_processor.eos_token == "<eos>"
        assert mock_processor.bos_token == "<bos>"
        assert mock_processor.pad_token_id == 0
        assert mock_processor.eos_token_id == 1
        assert mock_processor.bos_token_id == 2
        assert mock_processor.name_or_path == "test-model"

    @patch("nemo_rl.models.automodel.setup.AutoProcessor")
    def test_get_processor_forwards_tokenizer_kwargs(self, mock_auto_processor):
        """Test tokenizer_kwargs are forwarded through AutoProcessor."""
        mock_processor = MagicMock()
        mock_processor.tokenizer.pad_token = "<pad>"
        mock_auto_processor.from_pretrained.return_value = mock_processor
        config = {
            "name": "test-vlm",
            "tokenizer_kwargs": {"model_max_length": 123, "use_fast": False},
        }

        get_tokenizer(config, get_processor=True)

        mock_auto_processor.from_pretrained.assert_called_once_with(
            "test-vlm",
            trust_remote_code=True,
            use_fast=False,
            model_max_length=123,
        )
        assert config["tokenizer_kwargs"] == {
            "model_max_length": 123,
            "use_fast": False,
        }

    @patch("nemo_rl.models.automodel.setup.AutoProcessor")
    def test_get_processor_sets_pad_from_eos(self, mock_auto_processor):
        """Test that processor path also sets pad_token from eos when None."""
        mock_processor = MagicMock()
        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.pad_token = None
        mock_inner_tokenizer.eos_token = "<eos>"
        mock_inner_tokenizer.bos_token = "<bos>"
        mock_inner_tokenizer.pad_token_id = None
        mock_inner_tokenizer.eos_token_id = 1
        mock_inner_tokenizer.bos_token_id = 2
        mock_inner_tokenizer.name_or_path = "test-vlm"
        mock_processor.tokenizer = mock_inner_tokenizer
        mock_auto_processor.from_pretrained.return_value = mock_processor

        result = get_tokenizer({"name": "test-vlm"}, get_processor=True)

        assert mock_inner_tokenizer.pad_token == "<eos>"
        assert result is mock_processor

    @patch("nemo_rl.models.automodel.setup.NeMoAutoTokenizer")
    def test_does_not_use_hf_auto_tokenizer(self, mock_nemo_auto_tokenizer):
        """Test that NeMoAutoTokenizer is used, not HF AutoTokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_nemo_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with patch("nemo_rl.models.automodel.setup.AutoTokenizer") as mock_hf:
            get_tokenizer({"name": "gpt2"})
            mock_hf.from_pretrained.assert_not_called()

        mock_nemo_auto_tokenizer.from_pretrained.assert_called_once()


@pytest.mark.automodel
class TestMaybeSetForceHf:
    """Tests for _maybe_set_force_hf adapter compatibility check."""

    def _make_config(self, arch):
        """Create a mock model config with the given architecture."""
        config = Mock()
        config.architectures = [arch]
        return config

    def test_force_hf_true_skips_check(self):
        """When force_hf=True, no check is needed."""
        kwargs = {"force_hf": True}
        config = self._make_config("Qwen2ForCausalLM")
        _maybe_set_force_hf(kwargs, config)
        assert kwargs["force_hf"] is True

    def test_unknown_arch_skips_check(self):
        """When arch is not in the registry, no adapter is involved."""
        kwargs = {}
        config = self._make_config("SomeUnknownModelForCausalLM")
        _maybe_set_force_hf(kwargs, config)
        assert "force_hf" not in kwargs

    def test_no_architectures_skips_check(self):
        """When model config has no architectures, skip check."""
        kwargs = {}
        config = Mock()
        config.architectures = None
        _maybe_set_force_hf(kwargs, config)
        assert "force_hf" not in kwargs

    def test_qwen2_auto_sets_force_hf(self):
        """Qwen2's CombinedProjectionStateDictAdapter lacks convert_single_tensor_to_hf,
        so force_hf should be auto-set when not explicitly configured."""
        from nemo_automodel._transformers.registry import ModelRegistry

        if "Qwen2ForCausalLM" not in ModelRegistry.model_arch_name_to_cls:
            pytest.skip("Qwen2ForCausalLM not in registry")

        kwargs = {}
        config = self._make_config("Qwen2ForCausalLM")
        _maybe_set_force_hf(kwargs, config)
        assert kwargs.get("force_hf") is True

    def test_llama_auto_sets_force_hf(self):
        """Llama also uses CombinedProjectionStateDictAdapter."""
        from nemo_automodel._transformers.registry import ModelRegistry

        if "LlamaForCausalLM" not in ModelRegistry.model_arch_name_to_cls:
            pytest.skip("LlamaForCausalLM not in registry")

        kwargs = {}
        config = self._make_config("LlamaForCausalLM")
        _maybe_set_force_hf(kwargs, config)
        assert kwargs.get("force_hf") is True

    def test_qwen2_explicit_false_raises(self):
        """When force_hf is explicitly False and adapter is incompatible, raise."""
        from nemo_automodel._transformers.registry import ModelRegistry

        if "Qwen2ForCausalLM" not in ModelRegistry.model_arch_name_to_cls:
            pytest.skip("Qwen2ForCausalLM not in registry")

        kwargs = {"force_hf": False}
        config = self._make_config("Qwen2ForCausalLM")
        with pytest.raises(RuntimeError, match="force_hf=False"):
            _maybe_set_force_hf(kwargs, config)

    def test_compatible_adapter_no_change(self):
        """Models with adapters that implement convert_single_tensor_to_hf should
        not have force_hf auto-set."""
        from nemo_automodel._transformers.registry import ModelRegistry

        # Find a model whose adapter has convert_single_tensor_to_hf
        # (e.g. Qwen3Moe, NemotronH, DeepseekV3)
        compatible_archs = [
            "Qwen3MoeForCausalLM",
            "NemotronHForCausalLM",
            "DeepseekV3ForCausalLM",
        ]
        arch = None
        for a in compatible_archs:
            if a in ModelRegistry.model_arch_name_to_cls:
                arch = a
                break
        if arch is None:
            pytest.skip("No compatible model arch found in registry")

        kwargs = {}
        config = self._make_config(arch)
        _maybe_set_force_hf(kwargs, config)
        assert "force_hf" not in kwargs


@pytest.mark.automodel
def test_automodel_dtype_restore_workaround_still_needed(monkeypatch):
    """Tripwire for the temporary fp32 master-weight workaround in setup.py.

    ``_disable_automodel_checkpoint_dtype_restore`` no-ops Automodel's
    ``_restore_loaded_model_dtype`` because (pre PR #2419) it downgrades an explicitly-fp32
    load back to the bf16 checkpoint dtype, breaking optimizer master weights. This test
    reproduces that downgrade against the *live* pinned function (no model/checkpoint load).

    It PASSES while the bug is present. It FAILS — telling us to delete the workaround in
    ``nemo_rl/models/automodel/setup.py`` (and this test) — once Automodel either removes the
    function or ships PR #2419 (honors the explicit fp32 via ``promote_types`` so the weight
    stays fp32).
    """
    import inspect
    import types

    import nemo_automodel.components.checkpoint.utils as ckpt_utils
    from nemo_automodel._transformers import model_init

    restore = getattr(model_init, "_restore_loaded_model_dtype", None)
    # An earlier test in this process may have triggered the setup.py workaround
    # (_disable_automodel_checkpoint_dtype_restore), which globally and irreversibly
    # replaces this symbol with a no-op. Recover the genuine upstream function it
    # stashed so this tripwire exercises Automodel's real behavior, not our no-op.
    restore = getattr(restore, "_nrl_original", restore)
    if restore is None:
        pytest.fail(
            "Automodel removed _restore_loaded_model_dtype - remove the fp32 master-weight "
            "workaround _disable_automodel_checkpoint_dtype_restore() in "
            "nemo_rl/models/automodel/setup.py."
        )

    model = torch.nn.Linear(4, 4).float()
    assert model.weight.dtype == torch.float32

    # Pretend the checkpoint stored the weight in bf16.
    monkeypatch.setattr(
        ckpt_utils,
        "_get_checkpoint_tensor_dtypes",
        lambda *args, **kwargs: {"weight": torch.bfloat16},
    )
    # Reproduce NeMo-RL's explicit-fp32 load. PR #2419 honors the request ONLY via the
    # new `requested_dtype` parameter (it ignores hf_config.torch_dtype / load_kwargs in
    # this function): with requested_dtype=fp32 it promotes the bf16 checkpoint tensor up
    # to fp32 and leaves the weight unchanged. The current pin predates #2419 and its
    # signature has no such parameter, so pass it only when the signature accepts it:
    #   pre-#2419  -> requested_dtype absent -> weight downgraded to bf16 (assert holds, workaround needed)
    #   post-#2419 -> requested_dtype=fp32   -> weight stays fp32      (assert fails, fires the removal tripwire)
    hf_config = types.SimpleNamespace(torch_dtype=torch.float32)
    restore_kwargs = {}
    if "requested_dtype" in inspect.signature(restore).parameters:
        restore_kwargs["requested_dtype"] = torch.float32
    restore(
        model,
        "dummy",
        hf_config,
        None,
        {"torch_dtype": "torch.float32"},
        **restore_kwargs,
    )

    assert model.weight.dtype == torch.bfloat16, (
        "Automodel no longer downgrades an explicit-fp32 load (likely PR #2419 landed); the "
        "_disable_automodel_checkpoint_dtype_restore() workaround in setup.py is obsolete - "
        "remove it and this test."
    )


class _TinyLoraModel(torch.nn.Module):
    """Tiny module with one LoRA-adapted linear for warm-start tests."""

    def __init__(self):
        from nemo_automodel.components._peft.lora import LinearLoRA

        super().__init__()
        self.layer = LinearLoRA(torch.nn.Linear(4, 8), dim=2, alpha=4)


def _write_adapter_checkpoint(
    adapter_dir,
    *,
    dim=2,
    alpha=4,
    base_model_name_or_path="tiny-model",
    keys=("layer.lora_A.weight", "layer.lora_B.weight"),
    peft_type="LORA",
    fill=1.0,
):
    """Write a minimal HF-PEFT-style adapter checkpoint (config + safetensors).

    Tensors are filled with a non-zero value so a test cannot confuse
    "loaded the donor" with "left the zero-init adapters alone".
    """
    import json as _json

    from safetensors.torch import save_file

    os.makedirs(adapter_dir, exist_ok=True)
    tensors = {}
    for key in keys:
        shape = (2, 4) if "lora_A" in key else (8, 2)
        tensors[key] = torch.full(shape, fill)
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        _json.dump(
            {
                "peft_type": peft_type,
                "r": dim,
                "lora_alpha": alpha,
                "base_model_name_or_path": base_model_name_or_path,
                "target_modules": ["layer"],
            },
            f,
        )


def _lora_cfg(dim=2, alpha=4):
    return {
        "enabled": True,
        "target_modules": [],
        "exclude_modules": [],
        "match_all_linear": True,
        "dim": dim,
        "alpha": alpha,
        "dropout": 0.0,
        "dropout_position": "post",
        "lora_A_init": "xavier",
    }


@pytest.mark.automodel
class TestResolveLoraAdapterDir:
    def test_direct_adapter_dir(self, tmp_path):
        from nemo_rl.models.automodel.setup import _resolve_lora_adapter_dir

        _write_adapter_checkpoint(tmp_path / "adapter")
        assert _resolve_lora_adapter_dir(str(tmp_path / "adapter")) == str(
            tmp_path / "adapter"
        )

    def test_weights_dir_with_model_subdir(self, tmp_path):
        from nemo_rl.models.automodel.setup import _resolve_lora_adapter_dir

        weights_dir = tmp_path / "step_5" / "policy" / "weights"
        _write_adapter_checkpoint(weights_dir / "model")
        assert _resolve_lora_adapter_dir(str(weights_dir)) == str(weights_dir / "model")

    def test_missing_adapter_file_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _resolve_lora_adapter_dir

        with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
            _resolve_lora_adapter_dir(str(tmp_path))


@pytest.mark.automodel
class TestValidateLoraAdapterConfig:
    def test_matching_config_passes(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_config

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(adapter_dir)
        _validate_lora_adapter_config(
            str(adapter_dir), _lora_cfg(), "tiny-model"
        )  # should not raise

    def test_peft_type_mismatch_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_config

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(adapter_dir, peft_type="IA3")
        with pytest.raises(ValueError, match="peft_type"):
            _validate_lora_adapter_config(str(adapter_dir), _lora_cfg(), "tiny-model")

    def test_rank_mismatch_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_config

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(adapter_dir, dim=8)
        with pytest.raises(ValueError, match="r=8"):
            _validate_lora_adapter_config(str(adapter_dir), _lora_cfg(), "tiny-model")

    def test_alpha_mismatch_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_config

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(adapter_dir, alpha=16)
        with pytest.raises(ValueError, match="lora_alpha"):
            _validate_lora_adapter_config(str(adapter_dir), _lora_cfg(), "tiny-model")

    def test_base_model_mismatch_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_config

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(adapter_dir, base_model_name_or_path="other-model")
        with pytest.raises(ValueError, match="other-model"):
            _validate_lora_adapter_config(str(adapter_dir), _lora_cfg(), "tiny-model")

    def test_unknown_base_model_does_not_raise(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_config

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(adapter_dir, base_model_name_or_path="N/A")
        _validate_lora_adapter_config(str(adapter_dir), _lora_cfg(), "tiny-model")

    def test_missing_config_file_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_config

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            _validate_lora_adapter_config(str(adapter_dir), _lora_cfg(), "tiny-model")


@pytest.mark.automodel
class TestValidateLoraAdapterKeys:
    def test_prefixed_keys_match(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_keys

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(
            adapter_dir,
            keys=(
                "base_model.model.layer.lora_A.weight",
                "base_model.model.layer.lora_B.weight",
            ),
        )
        _validate_lora_adapter_keys(str(adapter_dir), _TinyLoraModel())

    def test_missing_key_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_keys

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(
            adapter_dir, keys=("base_model.model.layer.lora_A.weight",)
        )
        with pytest.raises(ValueError, match="Missing from donor"):
            _validate_lora_adapter_keys(str(adapter_dir), _TinyLoraModel())

    def test_unexpected_key_raises(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_keys

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(
            adapter_dir,
            keys=(
                "layer.lora_A.weight",
                "layer.lora_B.weight",
                "layer2.lora_A.weight",
            ),
        )
        with pytest.raises(ValueError, match="unexpected in donor"):
            _validate_lora_adapter_keys(str(adapter_dir), _TinyLoraModel())

    def test_state_dict_adapter_model_still_validated(self, tmp_path):
        """A custom state_dict_adapter must not disable the coverage check.

        Regression test: the escape hatch used to be gated on
        ``state_dict_adapter``, which 26/33 Automodel architectures set
        (including plain LlamaForCausalLM) -- so the check silently passed
        while the non-strict PEFT load left donor-uncovered adapters at fresh
        init. Only expert-parallel models may skip.
        """
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_keys

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(
            adapter_dir, keys=("base_model.model.layer.lora_A.weight",)
        )
        model = _TinyLoraModel()
        model.state_dict_adapter = object()  # custom key space, but not EP
        with pytest.raises(ValueError, match="Missing from donor"):
            _validate_lora_adapter_keys(str(adapter_dir), model)

    def test_expert_parallel_model_skips_key_check(self, tmp_path):
        from nemo_rl.models.automodel.setup import _validate_lora_adapter_keys

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(adapter_dir, keys=("anything.at.all",))
        model = _TinyLoraModel()
        # Expert-parallel models give their LoRA params custom state_dict
        # names (e.g. gate_up_linear.weight0) that don't appear in
        # named_parameters(), so the coverage check cannot run for them.
        model.layer.ep_size = 2
        _validate_lora_adapter_keys(str(adapter_dir), model)  # should not raise


@pytest.mark.automodel
class TestLoadInitialLoraAdapter:
    def test_loads_through_checkpointer(self, tmp_path):
        from nemo_rl.models.automodel.setup import _load_initial_lora_adapter

        weights_dir = tmp_path / "step_5" / "policy" / "weights"
        adapter_dir = weights_dir / "model"
        _write_adapter_checkpoint(
            adapter_dir,
            keys=(
                "base_model.model.layer.lora_A.weight",
                "base_model.model.layer.lora_B.weight",
            ),
        )
        model = _TinyLoraModel()
        # autospec binds the mocks to the real signatures, so a wrong kwarg
        # name in setup.py fails the test instead of silently recording a call.
        manager = create_autospec(AutomodelCheckpointManager, instance=True)
        manager.checkpointer = create_autospec(Checkpointer, instance=True)
        _load_initial_lora_adapter(
            model=model,
            checkpoint_manager=manager,
            restore_from=str(weights_dir),
            lora_cfg=_lora_cfg(),
            model_name="tiny-model",
        )
        manager.update_checkpointer_config.assert_called_once()
        config_updates = manager.update_checkpointer_config.call_args.kwargs[
            "config_updates"
        ]
        assert config_updates["is_peft"] is True
        manager.checkpointer.load_model.assert_called_once_with(
            model=model, model_path=str(adapter_dir)
        )

    def test_invalid_donor_fails_before_load(self, tmp_path):
        from nemo_rl.models.automodel.setup import _load_initial_lora_adapter

        adapter_dir = tmp_path / "adapter"
        # dim=8 donor vs dim=2 run -> validation must fail before any load.
        _write_adapter_checkpoint(adapter_dir, dim=8)
        manager = MagicMock()
        with pytest.raises(ValueError, match="r=8"):
            _load_initial_lora_adapter(
                model=_TinyLoraModel(),
                checkpoint_manager=manager,
                restore_from=str(adapter_dir),
                lora_cfg=_lora_cfg(),
                model_name="tiny-model",
            )
        manager.checkpointer.load_model.assert_not_called()

    def test_bare_adapter_dir_loaded_via_model_path(self, tmp_path):
        """A bare adapter dir must still take the checkpointer's PEFT branch.

        Automodel selects its PEFT safetensors read by a substring test on the
        path ("/model" in path), so the bare layout is staged under a
        temporary "model" path component before loading.
        """
        from nemo_rl.models.automodel.setup import _load_initial_lora_adapter

        adapter_dir = tmp_path / "adapter"
        _write_adapter_checkpoint(
            adapter_dir,
            keys=(
                "base_model.model.layer.lora_A.weight",
                "base_model.model.layer.lora_B.weight",
            ),
        )
        model = _TinyLoraModel()
        manager = create_autospec(AutomodelCheckpointManager, instance=True)
        manager.checkpointer = create_autospec(Checkpointer, instance=True)

        seen = {}

        def fake_load_model(*, model, model_path, **kwargs):
            # Inspected mid-call: the staging symlink is cleaned up after load.
            seen["model_path"] = model_path
            seen["resolves"] = os.path.isfile(
                os.path.join(model_path, "adapter_model.safetensors")
            )

        manager.checkpointer.load_model.side_effect = fake_load_model
        _load_initial_lora_adapter(
            model=model,
            checkpoint_manager=manager,
            restore_from=str(adapter_dir),
            lora_cfg=_lora_cfg(),
            model_name="tiny-model",
        )
        assert os.path.basename(seen["model_path"]) == "model"
        # The staged path still resolved to the donor's adapter file.
        assert seen["resolves"]
        # The staging directory was cleaned up after the load.
        assert not os.path.exists(os.path.dirname(seen["model_path"]))


@pytest.fixture
def _init_gloo_pg():
    """Single-process gloo PG so the real Automodel Checkpointer can run on CPU."""
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29517")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    yield


class _TwoLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4), torch.nn.Linear(4, 1)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.automodel
class TestLoadInitialLoraAdapterEndToEnd:
    """Warm start must actually put the donor's weights into the model."""

    def test_donor_adapter_weights_land_in_model(self, _init_gloo_pg, tmp_path):
        from nemo_automodel.components._peft.lora import (
            PeftConfig,
            apply_lora_to_linear_modules,
        )

        from nemo_rl.models.automodel.setup import _load_initial_lora_adapter

        peft_config = PeftConfig(
            target_modules=[],
            match_all_linear=True,
            dim=2,
            alpha=4,
            dropout=0.0,
            dropout_position="post",
            lora_A_init="xavier",
            use_triton=False,
        )

        # Donor: distinctive non-zero adapter weights, so "loaded the donor"
        # cannot be confused with "left the zero-init adapters alone".
        donor = _TwoLinearModel()
        apply_lora_to_linear_modules(donor, peft_config)
        for name, param in donor.named_parameters():
            if "lora_" in name:
                torch.nn.init.normal_(param, mean=3.0, std=1.0)
        donor_lora = {
            k: v.clone() for k, v in donor.state_dict().items() if "lora_" in k
        }
        assert donor_lora

        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cpu", (1,), mesh_dim_names=("dp",)
        )
        manager = AutomodelCheckpointManager(dp_mesh=mesh, tp_mesh=mesh)
        manager.init_checkpointer(
            config_updates={"model_save_format": "safetensors", "is_peft": True}
        )
        weights_path = str(tmp_path / "step_5" / "policy" / "weights")
        manager.save_checkpoint(
            model=donor,
            weights_path=weights_path,
            checkpointing_cfg={
                "enabled": True,
                "model_save_format": "safetensors",
                "is_peft": True,
            },
            lora_enabled=True,
            peft_config=peft_config,
        )

        # Fresh run: adapters start at zero, like a cold LoRA init.
        model = _TwoLinearModel()
        apply_lora_to_linear_modules(model, peft_config)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data.zero_()

        _load_initial_lora_adapter(
            model=model,
            checkpoint_manager=manager,
            restore_from=weights_path,
            lora_cfg=_lora_cfg(),
            model_name="tiny-model",
        )

        loaded = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        assert set(loaded) == set(donor_lora)
        for key, expected in donor_lora.items():
            assert torch.allclose(loaded[key], expected), f"{key} was not warm-started"
            assert not torch.allclose(loaded[key], torch.zeros_like(loaded[key]))

    def test_donor_covering_fewer_modules_raises(self, _init_gloo_pg, tmp_path):
        """A donor targeting fewer modules than the run must fail closed.

        Regression test for the state_dict_adapter escape hatch: the PEFT load
        is unconditionally non-strict, so without key validation this donor
        would load partially (one layer warm-started, the other silently left
        at fresh init) and still print a success message.
        """
        from nemo_automodel.components._peft.lora import (
            PeftConfig,
            apply_lora_to_linear_modules,
        )

        from nemo_rl.models.automodel.setup import _load_initial_lora_adapter

        donor_peft_config = PeftConfig(
            target_modules=["*layers.0*"],
            match_all_linear=False,
            dim=2,
            alpha=4,
            dropout=0.0,
            dropout_position="post",
            lora_A_init="xavier",
            use_triton=False,
        )
        donor = _TwoLinearModel()
        apply_lora_to_linear_modules(donor, donor_peft_config)
        donor_lora_names = [n for n, _ in donor.named_parameters() if "lora_" in n]
        # Sanity: the donor adapter covers layers.0 only, not layers.1.
        assert donor_lora_names
        assert all("layers.0" in n for n in donor_lora_names)

        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cpu", (1,), mesh_dim_names=("dp",)
        )
        manager = AutomodelCheckpointManager(dp_mesh=mesh, tp_mesh=mesh)
        manager.init_checkpointer(
            config_updates={"model_save_format": "safetensors", "is_peft": True}
        )
        weights_path = str(tmp_path / "step_5" / "policy" / "weights")
        manager.save_checkpoint(
            model=donor,
            weights_path=weights_path,
            checkpointing_cfg={
                "enabled": True,
                "model_save_format": "safetensors",
                "is_peft": True,
            },
            lora_enabled=True,
            peft_config=donor_peft_config,
        )

        run_peft_config = PeftConfig(
            target_modules=[],
            match_all_linear=True,
            dim=2,
            alpha=4,
            dropout=0.0,
            dropout_position="post",
            lora_A_init="xavier",
            use_triton=False,
        )
        model = _TwoLinearModel()
        apply_lora_to_linear_modules(model, run_peft_config)

        with pytest.raises(ValueError, match="Missing from donor"):
            _load_initial_lora_adapter(
                model=model,
                checkpoint_manager=manager,
                restore_from=weights_path,
                lora_cfg=_lora_cfg(),
                model_name="tiny-model",
            )
