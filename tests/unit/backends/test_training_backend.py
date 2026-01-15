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
"""Unit tests for training backend protocol and implementations.

Tests cover:
- TrainingBackend protocol definition
- DTensorBackend implementation
- MegatronBackend implementation
- Backend registry and factory functions
- Custom backend registration
"""

import pytest
from typing import Any, Optional
from pathlib import Path
from unittest.mock import MagicMock, patch

from nemo_rl.backends.training.base import (
    TrainingBackend,
    TrainingBackendConfig,
    get_training_backend,
    register_training_backend,
    list_training_backends,
    _TRAINING_BACKEND_REGISTRY,
)
from nemo_rl.backends.training.dtensor import DTensorBackend
from nemo_rl.backends.training.megatron import MegatronBackend


class TestTrainingBackendConfig:
    """Tests for TrainingBackendConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingBackendConfig()
        assert config.backend_type == "dtensor"
        assert config.model_name == ""
        assert config.precision == "bfloat16"
        assert config.train_global_batch_size == 32
        assert config.train_micro_batch_size == 4
        assert config.max_grad_norm == 1.0
        assert config.backend_kwargs == {}

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = TrainingBackendConfig(
            backend_type="megatron",
            model_name="meta-llama/Llama-2-7b-hf",
            precision="float16",
            train_global_batch_size=64,
            train_micro_batch_size=8,
            max_grad_norm=0.5,
            backend_kwargs={"custom_key": "custom_value"},
        )
        assert config.backend_type == "megatron"
        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.precision == "float16"
        assert config.train_global_batch_size == 64
        assert config.train_micro_batch_size == 8
        assert config.max_grad_norm == 0.5
        assert config.backend_kwargs == {"custom_key": "custom_value"}


class TestTrainingBackendProtocol:
    """Tests for TrainingBackend protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that TrainingBackend is a runtime checkable protocol."""
        from typing import runtime_checkable, Protocol

        assert hasattr(TrainingBackend, "__protocol_attrs__") or isinstance(
            TrainingBackend, type
        )

    def test_dtensor_backend_implements_protocol(self):
        """Test that DTensorBackend implements TrainingBackend protocol."""
        backend = DTensorBackend()
        assert isinstance(backend, TrainingBackend)

    def test_megatron_backend_implements_protocol(self):
        """Test that MegatronBackend implements TrainingBackend protocol."""
        backend = MegatronBackend()
        assert isinstance(backend, TrainingBackend)


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_training_backends(self):
        """Test listing registered backends."""
        backends = list_training_backends()
        assert "dtensor" in backends
        assert "megatron" in backends

    def test_get_training_backend_dtensor(self):
        """Test getting DTensor backend by name."""
        backend = get_training_backend("dtensor")
        assert isinstance(backend, DTensorBackend)
        assert backend.backend_type == "dtensor"

    def test_get_training_backend_megatron(self):
        """Test getting Megatron backend by name."""
        backend = get_training_backend("megatron")
        assert isinstance(backend, MegatronBackend)
        assert backend.backend_type == "megatron"

    def test_get_training_backend_unknown(self):
        """Test error for unknown backend name."""
        with pytest.raises(ValueError) as excinfo:
            get_training_backend("unknown_backend")
        assert "Unknown training backend 'unknown_backend'" in str(excinfo.value)
        assert "dtensor" in str(excinfo.value)
        assert "megatron" in str(excinfo.value)

    def test_register_custom_backend(self):
        """Test registering a custom backend."""
        # Create a custom backend class
        class CustomTestBackend:
            def setup(self, config: TrainingBackendConfig) -> None:
                self._config = config
                self._is_initialized = True

            def train_step(self, batch, loss_fn, **kwargs) -> dict:
                return {"loss": 0.0}

            def get_logprobs(self, batch, **kwargs):
                return batch

            def save_checkpoint(self, path, **kwargs) -> None:
                pass

            def load_checkpoint(self, path, **kwargs) -> None:
                pass

            def prepare_for_training(self) -> None:
                pass

            def prepare_for_inference(self) -> None:
                pass

            def shutdown(self) -> None:
                pass

            @property
            def is_initialized(self) -> bool:
                return getattr(self, "_is_initialized", False)

            @property
            def backend_type(self) -> str:
                return "custom_test"

        # Register the custom backend
        register_training_backend("custom_test")(CustomTestBackend)

        # Verify it can be retrieved
        backend = get_training_backend("custom_test")
        assert isinstance(backend, CustomTestBackend)
        assert backend.backend_type == "custom_test"

        # Clean up
        del _TRAINING_BACKEND_REGISTRY["custom_test"]

    def test_register_duplicate_backend_raises(self):
        """Test that registering duplicate backend name raises error."""
        class DuplicateBackend:
            pass

        # This should raise because 'dtensor' is already registered
        with pytest.raises(ValueError) as excinfo:
            register_training_backend("dtensor")(DuplicateBackend)
        assert "already registered" in str(excinfo.value)


class TestDTensorBackend:
    """Tests for DTensorBackend implementation."""

    def test_initialization(self):
        """Test DTensorBackend initialization."""
        backend = DTensorBackend()
        assert not backend.is_initialized
        assert backend.backend_type == "dtensor"
        assert backend.config is None

    def test_setup_with_valid_config(self):
        """Test setup with valid DTensor configuration."""
        backend = DTensorBackend()
        config = TrainingBackendConfig(
            backend_type="dtensor",
            model_name="meta-llama/Llama-2-7b-hf",
            precision="bfloat16",
            train_global_batch_size=32,
            train_micro_batch_size=4,
        )
        backend.setup(config)

        assert backend.is_initialized
        assert backend.config == config
        assert backend.policy_config is not None
        assert backend.policy_config["model_name"] == "meta-llama/Llama-2-7b-hf"
        assert backend.policy_config["dtensor_cfg"]["enabled"] is True
        assert backend.policy_config["megatron_cfg"]["enabled"] is False

    def test_setup_with_wrong_backend_type(self):
        """Test that setup fails with wrong backend type."""
        backend = DTensorBackend()
        config = TrainingBackendConfig(backend_type="megatron")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)
        assert "Expected 'dtensor'" in str(excinfo.value)

    def test_setup_with_backend_kwargs(self):
        """Test setup with backend-specific kwargs."""
        backend = DTensorBackend()
        config = TrainingBackendConfig(
            backend_type="dtensor",
            model_name="test-model",
            backend_kwargs={
                "dtensor_cfg": {
                    "tensor_parallel_size": 2,
                    "context_parallel_size": 1,
                    "sequence_parallel": True,
                    "cpu_offload": False,
                }
            },
        )
        backend.setup(config)

        assert backend.policy_config["dtensor_cfg"]["tensor_parallel_size"] == 2
        assert backend.policy_config["dtensor_cfg"]["sequence_parallel"] is True

    def test_train_step_requires_initialization(self):
        """Test that train_step requires initialization."""
        backend = DTensorBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.train_step(MagicMock(), MagicMock())
        assert "not initialized" in str(excinfo.value)

    def test_get_logprobs_requires_initialization(self):
        """Test that get_logprobs requires initialization."""
        backend = DTensorBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.get_logprobs(MagicMock())
        assert "not initialized" in str(excinfo.value)

    def test_save_checkpoint_requires_initialization(self):
        """Test that save_checkpoint requires initialization."""
        backend = DTensorBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.save_checkpoint("/path/to/checkpoint")
        assert "not initialized" in str(excinfo.value)

    def test_shutdown(self):
        """Test shutdown clears initialization state."""
        backend = DTensorBackend()
        config = TrainingBackendConfig(backend_type="dtensor", model_name="test")
        backend.setup(config)

        assert backend.is_initialized
        backend.shutdown()
        assert not backend.is_initialized


class TestMegatronBackend:
    """Tests for MegatronBackend implementation."""

    def test_initialization(self):
        """Test MegatronBackend initialization."""
        backend = MegatronBackend()
        assert not backend.is_initialized
        assert backend.backend_type == "megatron"
        assert backend.config is None

    def test_setup_with_valid_config(self):
        """Test setup with valid Megatron configuration."""
        backend = MegatronBackend()
        config = TrainingBackendConfig(
            backend_type="megatron",
            model_name="meta-llama/Llama-2-7b-hf",
            precision="bfloat16",
            train_global_batch_size=32,
            train_micro_batch_size=4,
        )
        backend.setup(config)

        assert backend.is_initialized
        assert backend.config == config
        assert backend.policy_config is not None
        assert backend.policy_config["model_name"] == "meta-llama/Llama-2-7b-hf"
        assert backend.policy_config["megatron_cfg"]["enabled"] is True
        assert backend.policy_config["dtensor_cfg"]["enabled"] is False
        assert backend.policy_config["megatron_cfg"]["bf16"] is True

    def test_setup_with_wrong_backend_type(self):
        """Test that setup fails with wrong backend type."""
        backend = MegatronBackend()
        config = TrainingBackendConfig(backend_type="dtensor")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)
        assert "Expected 'megatron'" in str(excinfo.value)

    def test_setup_with_backend_kwargs(self):
        """Test setup with Megatron-specific kwargs."""
        backend = MegatronBackend()
        config = TrainingBackendConfig(
            backend_type="megatron",
            model_name="test-model",
            precision="float16",
            backend_kwargs={
                "megatron_cfg": {
                    "tensor_model_parallel_size": 4,
                    "pipeline_model_parallel_size": 2,
                    "sequence_parallel": True,
                }
            },
        )
        backend.setup(config)

        assert backend.policy_config["megatron_cfg"]["tensor_model_parallel_size"] == 4
        assert backend.policy_config["megatron_cfg"]["pipeline_model_parallel_size"] == 2
        assert backend.policy_config["megatron_cfg"]["sequence_parallel"] is True
        assert backend.policy_config["megatron_cfg"]["fp16"] is True
        assert backend.policy_config["megatron_cfg"]["bf16"] is False

    def test_train_step_requires_initialization(self):
        """Test that train_step requires initialization."""
        backend = MegatronBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.train_step(MagicMock(), MagicMock())
        assert "not initialized" in str(excinfo.value)

    def test_get_logprobs_requires_initialization(self):
        """Test that get_logprobs requires initialization."""
        backend = MegatronBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.get_logprobs(MagicMock())
        assert "not initialized" in str(excinfo.value)

    def test_shutdown(self):
        """Test shutdown clears initialization state."""
        backend = MegatronBackend()
        config = TrainingBackendConfig(backend_type="megatron", model_name="test")
        backend.setup(config)

        assert backend.is_initialized
        backend.shutdown()
        assert not backend.is_initialized


class TestBackendSelection:
    """Tests for backend selection via string parameter."""

    @pytest.mark.parametrize(
        "backend_name,expected_class",
        [
            ("dtensor", DTensorBackend),
            ("megatron", MegatronBackend),
        ],
    )
    def test_backend_selection_by_name(self, backend_name, expected_class):
        """Test that backends can be selected by string name."""
        backend = get_training_backend(backend_name)
        assert isinstance(backend, expected_class)
        assert backend.backend_type == backend_name

    def test_backend_kwargs_passed_to_constructor(self):
        """Test that kwargs are passed to backend constructor."""
        mock_worker_group = MagicMock()
        mock_sharding = MagicMock()

        backend = get_training_backend(
            "dtensor",
            worker_group=mock_worker_group,
            sharding_annotations=mock_sharding,
        )

        assert backend._worker_group is mock_worker_group
        assert backend._sharding_annotations is mock_sharding


class TestBackendIntegration:
    """Integration tests for backend usage patterns."""

    def test_backend_workflow_dtensor(self):
        """Test complete workflow with DTensor backend."""
        # Get backend by name
        backend = get_training_backend("dtensor")
        assert backend.backend_type == "dtensor"

        # Setup backend
        config = TrainingBackendConfig(
            backend_type="dtensor",
            model_name="test-model",
            precision="bfloat16",
            train_global_batch_size=16,
            train_micro_batch_size=2,
        )
        backend.setup(config)
        assert backend.is_initialized

        # Verify configuration
        assert backend.config.model_name == "test-model"
        assert backend.policy_config["train_global_batch_size"] == 16

        # Shutdown
        backend.shutdown()
        assert not backend.is_initialized

    def test_backend_workflow_megatron(self):
        """Test complete workflow with Megatron backend."""
        # Get backend by name
        backend = get_training_backend("megatron")
        assert backend.backend_type == "megatron"

        # Setup backend
        config = TrainingBackendConfig(
            backend_type="megatron",
            model_name="test-model",
            precision="float16",
            train_global_batch_size=32,
            train_micro_batch_size=4,
            backend_kwargs={
                "megatron_cfg": {
                    "tensor_model_parallel_size": 2,
                }
            },
        )
        backend.setup(config)
        assert backend.is_initialized

        # Verify configuration
        assert backend.config.model_name == "test-model"
        assert backend.policy_config["megatron_cfg"]["tensor_model_parallel_size"] == 2

        # Shutdown
        backend.shutdown()
        assert not backend.is_initialized


class TestErrorMessages:
    """Tests for clear error messages."""

    def test_unknown_backend_error_message(self):
        """Test that unknown backend error message is helpful."""
        with pytest.raises(ValueError) as excinfo:
            get_training_backend("nonexistent")

        error_msg = str(excinfo.value)
        assert "Unknown training backend 'nonexistent'" in error_msg
        assert "Available backends:" in error_msg
        assert "@register_training_backend" in error_msg

    def test_wrong_backend_type_error_dtensor(self):
        """Test error message for wrong backend type in DTensor."""
        backend = DTensorBackend()
        config = TrainingBackendConfig(backend_type="megatron")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)

        error_msg = str(excinfo.value)
        assert "backend_type='megatron'" in error_msg
        assert "Expected 'dtensor'" in error_msg

    def test_wrong_backend_type_error_megatron(self):
        """Test error message for wrong backend type in Megatron."""
        backend = MegatronBackend()
        config = TrainingBackendConfig(backend_type="dtensor")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)

        error_msg = str(excinfo.value)
        assert "backend_type='dtensor'" in error_msg
        assert "Expected 'megatron'" in error_msg

    def test_not_initialized_error_message(self):
        """Test error message when methods called before setup."""
        backend = DTensorBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.train_step(MagicMock(), MagicMock())

        error_msg = str(excinfo.value)
        assert "not initialized" in error_msg
        assert "Call setup() first" in error_msg
