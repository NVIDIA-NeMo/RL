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

from nemo_rl.models.automodel.setup import (
    ModelAndOptimizerState,
    RuntimeConfig,
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
            "context_parallel_size": 1,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "data_parallel_size": None,
            "sequence_parallel": False,
            "use_hf_tp_plan": False,
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


@pytest.mark.automodel
class TestModelAndOptimizerState:
    """Test suite for ModelAndOptimizerState dataclass."""

    def test_dataclass_initialization(self):
        """Test ModelAndOptimizerState can be initialized correctly."""
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        model_config = MagicMock()

        state = ModelAndOptimizerState(
            model=model,
            model_state_dict_keys=["key1", "key2"],
            optimizer=optimizer,
            scheduler=scheduler,
            is_hf_model=True,
            is_moe_model=False,
            is_reward_model=False,
            model_class=type(model),
            model_config=model_config,
            peft_config=None,
            autocast_enabled=True,
        )

        assert state.model == model
        assert state.optimizer == optimizer
        assert state.scheduler == scheduler
        assert state.is_hf_model is True
        assert state.is_moe_model is False
        assert state.is_reward_model is False
        assert state.model_state_dict_keys == ["key1", "key2"]
        assert state.peft_config is None
        assert state.autocast_enabled is True

    def test_dataclass_with_optional_fields(self):
        """Test ModelAndOptimizerState with optional fields set to None."""
        model = MagicMock()
        model_config = MagicMock()

        state = ModelAndOptimizerState(
            model=model,
            model_state_dict_keys=[],
            optimizer=None,
            scheduler=None,
            is_hf_model=False,
            is_moe_model=True,
            is_reward_model=False,
            model_class=type(model),
            model_config=model_config,
            peft_config=None,
            autocast_enabled=False,
        )

        assert state.optimizer is None
        assert state.scheduler is None
        assert state.is_moe_model is True
        assert state.autocast_enabled is False
