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
"""Unit tests for the unified backend factory.

Tests cover:
- BackendFactory class methods
- Custom backend registration
- Backend selection by string name
- Error handling for unknown backends
- VERIFY criterion: Register custom backend and retrieve by name
"""

import pytest
from typing import Any, Optional
from unittest.mock import MagicMock

from nemo_rl.backends.factory import BackendFactory
from nemo_rl.backends.training.base import (
    TrainingBackend,
    TrainingBackendConfig,
    _TRAINING_BACKEND_REGISTRY,
)
from nemo_rl.backends.generation.base import (
    GenerationBackend,
    GenerationBackendConfig,
    _GENERATION_BACKEND_REGISTRY,
)
from nemo_rl.backends.training.dtensor import DTensorBackend
from nemo_rl.backends.training.megatron import MegatronBackend
from nemo_rl.backends.generation.vllm import VLLMBackend
from nemo_rl.backends.generation.megatron import MegatronInferenceBackend


class TestBackendFactoryTraining:
    """Tests for BackendFactory training backend methods."""

    def test_get_training_backend_dtensor(self):
        """Test getting DTensor backend through factory."""
        backend = BackendFactory.get_training_backend("dtensor")
        assert isinstance(backend, DTensorBackend)
        assert backend.backend_type == "dtensor"

    def test_get_training_backend_megatron(self):
        """Test getting Megatron backend through factory."""
        backend = BackendFactory.get_training_backend("megatron")
        assert isinstance(backend, MegatronBackend)
        assert backend.backend_type == "megatron"

    def test_list_training_backends(self):
        """Test listing training backends through factory."""
        backends = BackendFactory.list_training_backends()
        assert "dtensor" in backends
        assert "megatron" in backends
        assert isinstance(backends, list)

    def test_get_training_backend_unknown_raises(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            BackendFactory.get_training_backend("unknown")
        assert "Unknown training backend 'unknown'" in str(excinfo.value)
        assert "Available backends:" in str(excinfo.value)


class TestBackendFactoryGeneration:
    """Tests for BackendFactory generation backend methods."""

    def test_get_generation_backend_vllm(self):
        """Test getting vLLM backend through factory."""
        backend = BackendFactory.get_generation_backend("vllm")
        assert isinstance(backend, VLLMBackend)
        assert backend.backend_type == "vllm"

    def test_get_generation_backend_megatron(self):
        """Test getting Megatron inference backend through factory."""
        backend = BackendFactory.get_generation_backend("megatron")
        assert isinstance(backend, MegatronInferenceBackend)
        assert backend.backend_type == "megatron"

    def test_list_generation_backends(self):
        """Test listing generation backends through factory."""
        backends = BackendFactory.list_generation_backends()
        assert "vllm" in backends
        assert "megatron" in backends
        assert isinstance(backends, list)

    def test_get_generation_backend_unknown_raises(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            BackendFactory.get_generation_backend("unknown")
        assert "Unknown generation backend 'unknown'" in str(excinfo.value)
        assert "Available backends:" in str(excinfo.value)


class TestBackendFactoryUnified:
    """Tests for BackendFactory unified methods."""

    def test_list_all_backends(self):
        """Test listing all backends through factory."""
        all_backends = BackendFactory.list_all_backends()

        assert "training" in all_backends
        assert "generation" in all_backends

        assert "dtensor" in all_backends["training"]
        assert "megatron" in all_backends["training"]
        assert "vllm" in all_backends["generation"]
        assert "megatron" in all_backends["generation"]

    def test_is_backend_registered_training(self):
        """Test checking if training backend is registered."""
        assert BackendFactory.is_backend_registered("training", "dtensor")
        assert BackendFactory.is_backend_registered("training", "megatron")
        assert not BackendFactory.is_backend_registered("training", "unknown")

    def test_is_backend_registered_generation(self):
        """Test checking if generation backend is registered."""
        assert BackendFactory.is_backend_registered("generation", "vllm")
        assert BackendFactory.is_backend_registered("generation", "megatron")
        assert not BackendFactory.is_backend_registered("generation", "unknown")

    def test_is_backend_registered_invalid_type(self):
        """Test that invalid backend type raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            BackendFactory.is_backend_registered("invalid", "dtensor")
        assert "Unknown backend_type 'invalid'" in str(excinfo.value)
        assert "'training' or 'generation'" in str(excinfo.value)


class TestCustomBackendRegistration:
    """Tests for custom backend registration.

    These tests verify the VERIFY criterion:
    'Register a custom backend with @register_training_backend('custom'),
    verify it can be retrieved by name'
    """

    def test_register_custom_training_backend(self):
        """VERIFY: Register custom training backend and retrieve by name."""
        # Create a custom training backend
        class CustomTestTrainingBackend:
            """Custom training backend for testing."""

            def __init__(self):
                self._is_initialized = False
                self._config = None

            def setup(self, config: TrainingBackendConfig) -> None:
                self._config = config
                self._is_initialized = True

            def train_step(self, batch, loss_fn, **kwargs) -> dict:
                return {"loss": 0.0, "grad_norm": 0.0}

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
                self._is_initialized = False

            @property
            def is_initialized(self) -> bool:
                return self._is_initialized

            @property
            def backend_type(self) -> str:
                return "custom_verify_training"

        # Register the custom backend using decorator
        BackendFactory.register_training_backend("custom_verify_training")(
            CustomTestTrainingBackend
        )

        # Verify it can be retrieved by name
        backend = BackendFactory.get_training_backend("custom_verify_training")

        # Verify correct type and properties
        assert isinstance(backend, CustomTestTrainingBackend)
        assert backend.backend_type == "custom_verify_training"
        assert not backend.is_initialized

        # Test full workflow
        config = TrainingBackendConfig(
            backend_type="custom_verify_training",
            model_name="test-model",
        )
        backend.setup(config)
        assert backend.is_initialized

        backend.shutdown()
        assert not backend.is_initialized

        # Clean up
        del _TRAINING_BACKEND_REGISTRY["custom_verify_training"]

    def test_register_custom_generation_backend(self):
        """VERIFY: Register custom generation backend and retrieve by name."""
        # Create a custom generation backend
        class CustomTestGenerationBackend:
            """Custom generation backend for testing."""

            def __init__(self):
                self._is_initialized = False
                self._config = None

            def setup(self, config: GenerationBackendConfig) -> None:
                self._config = config
                self._is_initialized = True

            def generate(self, prompts, **kwargs):
                return prompts

            def update_weights(self, state_dict) -> None:
                pass

            def prepare_for_generation(self) -> None:
                pass

            def finish_generation(self) -> None:
                pass

            def shutdown(self) -> None:
                self._is_initialized = False

            @property
            def is_initialized(self) -> bool:
                return self._is_initialized

            @property
            def backend_type(self) -> str:
                return "custom_verify_generation"

        # Register the custom backend using decorator
        BackendFactory.register_generation_backend("custom_verify_generation")(
            CustomTestGenerationBackend
        )

        # Verify it can be retrieved by name
        backend = BackendFactory.get_generation_backend("custom_verify_generation")

        # Verify correct type and properties
        assert isinstance(backend, CustomTestGenerationBackend)
        assert backend.backend_type == "custom_verify_generation"
        assert not backend.is_initialized

        # Test full workflow
        config = GenerationBackendConfig(
            backend_type="custom_verify_generation",
            model_name="test-model",
        )
        backend.setup(config)
        assert backend.is_initialized

        backend.shutdown()
        assert not backend.is_initialized

        # Clean up
        del _GENERATION_BACKEND_REGISTRY["custom_verify_generation"]

    def test_decorator_syntax_training(self):
        """Test decorator syntax for training backend registration."""

        @BackendFactory.register_training_backend("decorator_test_training")
        class DecoratorTestTrainingBackend:
            def setup(self, config): pass
            def train_step(self, batch, loss_fn, **kwargs): return {}
            def get_logprobs(self, batch, **kwargs): return batch
            def save_checkpoint(self, path, **kwargs): pass
            def load_checkpoint(self, path, **kwargs): pass
            def prepare_for_training(self): pass
            def prepare_for_inference(self): pass
            def shutdown(self): pass
            @property
            def is_initialized(self): return False
            @property
            def backend_type(self): return "decorator_test_training"

        # Verify registration
        assert BackendFactory.is_backend_registered("training", "decorator_test_training")

        backend = BackendFactory.get_training_backend("decorator_test_training")
        assert backend.backend_type == "decorator_test_training"

        # Clean up
        del _TRAINING_BACKEND_REGISTRY["decorator_test_training"]

    def test_decorator_syntax_generation(self):
        """Test decorator syntax for generation backend registration."""

        @BackendFactory.register_generation_backend("decorator_test_generation")
        class DecoratorTestGenerationBackend:
            def setup(self, config): pass
            def generate(self, prompts, **kwargs): return prompts
            def update_weights(self, state_dict): pass
            def prepare_for_generation(self): pass
            def finish_generation(self): pass
            def shutdown(self): pass
            @property
            def is_initialized(self): return False
            @property
            def backend_type(self): return "decorator_test_generation"

        # Verify registration
        assert BackendFactory.is_backend_registered("generation", "decorator_test_generation")

        backend = BackendFactory.get_generation_backend("decorator_test_generation")
        assert backend.backend_type == "decorator_test_generation"

        # Clean up
        del _GENERATION_BACKEND_REGISTRY["decorator_test_generation"]


class TestBackendFactoryErrorMessages:
    """Tests for clear error messages from BackendFactory."""

    def test_unknown_training_backend_error(self):
        """Test helpful error message for unknown training backend."""
        with pytest.raises(ValueError) as excinfo:
            BackendFactory.get_training_backend("nonexistent")

        error_msg = str(excinfo.value)
        assert "Unknown training backend 'nonexistent'" in error_msg
        assert "dtensor" in error_msg
        assert "megatron" in error_msg
        assert "@register_training_backend" in error_msg

    def test_unknown_generation_backend_error(self):
        """Test helpful error message for unknown generation backend."""
        with pytest.raises(ValueError) as excinfo:
            BackendFactory.get_generation_backend("nonexistent")

        error_msg = str(excinfo.value)
        assert "Unknown generation backend 'nonexistent'" in error_msg
        assert "vllm" in error_msg
        assert "megatron" in error_msg
        assert "@register_generation_backend" in error_msg

    def test_duplicate_registration_error_training(self):
        """Test error when registering duplicate training backend name."""
        class DuplicateBackend:
            pass

        with pytest.raises(ValueError) as excinfo:
            BackendFactory.register_training_backend("dtensor")(DuplicateBackend)
        assert "already registered" in str(excinfo.value)

    def test_duplicate_registration_error_generation(self):
        """Test error when registering duplicate generation backend name."""
        class DuplicateBackend:
            pass

        with pytest.raises(ValueError) as excinfo:
            BackendFactory.register_generation_backend("vllm")(DuplicateBackend)
        assert "already registered" in str(excinfo.value)


class TestBackendFactoryWorkflow:
    """Integration tests for complete backend workflows."""

    def test_training_backend_workflow(self):
        """Test complete training backend workflow through factory."""
        # Get backend
        backend = BackendFactory.get_training_backend("dtensor")
        assert backend.backend_type == "dtensor"

        # Setup
        config = TrainingBackendConfig(
            backend_type="dtensor",
            model_name="test-model",
            train_global_batch_size=16,
        )
        backend.setup(config)
        assert backend.is_initialized

        # Verify config
        assert backend.config.model_name == "test-model"
        assert backend.config.train_global_batch_size == 16

        # Shutdown
        backend.shutdown()
        assert not backend.is_initialized

    def test_generation_backend_workflow(self):
        """Test complete generation backend workflow through factory."""
        # Get backend
        backend = BackendFactory.get_generation_backend("vllm")
        assert backend.backend_type == "vllm"

        # Setup
        config = GenerationBackendConfig(
            backend_type="vllm",
            model_name="test-model",
            max_new_tokens=256,
            temperature=0.8,
        )
        backend.setup(config)
        assert backend.is_initialized

        # Verify config
        assert backend.config.model_name == "test-model"
        assert backend.config.max_new_tokens == 256
        assert backend.config.temperature == 0.8

        # Shutdown
        backend.shutdown()
        assert not backend.is_initialized

    def test_backend_factory_kwargs_passed(self):
        """Test that kwargs are passed to backend constructor."""
        mock_worker_group = MagicMock()

        backend = BackendFactory.get_training_backend(
            "dtensor",
            worker_group=mock_worker_group,
        )

        assert backend._worker_group is mock_worker_group
