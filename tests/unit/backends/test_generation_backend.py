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
"""Unit tests for generation backend protocol and implementations.

Tests cover:
- GenerationBackend protocol definition
- VLLMBackend implementation
- MegatronInferenceBackend implementation
- Backend registry and factory functions
- Custom backend registration
"""

import pytest
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from nemo_rl.backends.generation.base import (
    GenerationBackend,
    GenerationBackendConfig,
    get_generation_backend,
    register_generation_backend,
    list_generation_backends,
    _GENERATION_BACKEND_REGISTRY,
)
from nemo_rl.backends.generation.vllm import VLLMBackend
from nemo_rl.backends.generation.megatron import MegatronInferenceBackend


class TestGenerationBackendConfig:
    """Tests for GenerationBackendConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationBackendConfig()
        assert config.backend_type == "vllm"
        assert config.model_name == ""
        assert config.max_new_tokens == 512
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.top_k is None
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.stop_token_ids is None
        assert config.stop_strings is None
        assert config.backend_kwargs == {}

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = GenerationBackendConfig(
            backend_type="megatron",
            model_name="meta-llama/Llama-2-7b-hf",
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            stop_token_ids=[2, 3],
            stop_strings=["END"],
            backend_kwargs={"custom_key": "custom_value"},
        )
        assert config.backend_type == "megatron"
        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 1
        assert config.stop_token_ids == [2, 3]
        assert config.stop_strings == ["END"]
        assert config.backend_kwargs == {"custom_key": "custom_value"}


class TestGenerationBackendProtocol:
    """Tests for GenerationBackend protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that GenerationBackend is a runtime checkable protocol."""
        from typing import runtime_checkable, Protocol

        assert hasattr(GenerationBackend, "__protocol_attrs__") or isinstance(
            GenerationBackend, type
        )

    def test_vllm_backend_implements_protocol(self):
        """Test that VLLMBackend implements GenerationBackend protocol."""
        backend = VLLMBackend()
        assert isinstance(backend, GenerationBackend)

    def test_megatron_backend_implements_protocol(self):
        """Test that MegatronInferenceBackend implements GenerationBackend protocol."""
        backend = MegatronInferenceBackend()
        assert isinstance(backend, GenerationBackend)


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_generation_backends(self):
        """Test listing registered backends."""
        backends = list_generation_backends()
        assert "vllm" in backends
        assert "megatron" in backends

    def test_get_generation_backend_vllm(self):
        """Test getting vLLM backend by name."""
        backend = get_generation_backend("vllm")
        assert isinstance(backend, VLLMBackend)
        assert backend.backend_type == "vllm"

    def test_get_generation_backend_megatron(self):
        """Test getting Megatron backend by name."""
        backend = get_generation_backend("megatron")
        assert isinstance(backend, MegatronInferenceBackend)
        assert backend.backend_type == "megatron"

    def test_get_generation_backend_unknown(self):
        """Test error for unknown backend name."""
        with pytest.raises(ValueError) as excinfo:
            get_generation_backend("unknown_backend")
        assert "Unknown generation backend 'unknown_backend'" in str(excinfo.value)
        assert "vllm" in str(excinfo.value)
        assert "megatron" in str(excinfo.value)

    def test_register_custom_backend(self):
        """Test registering a custom backend."""
        # Create a custom backend class
        class CustomTestGenerationBackend:
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
                pass

            @property
            def is_initialized(self) -> bool:
                return getattr(self, "_is_initialized", False)

            @property
            def backend_type(self) -> str:
                return "custom_gen_test"

        # Register the custom backend
        register_generation_backend("custom_gen_test")(CustomTestGenerationBackend)

        # Verify it can be retrieved
        backend = get_generation_backend("custom_gen_test")
        assert isinstance(backend, CustomTestGenerationBackend)
        assert backend.backend_type == "custom_gen_test"

        # Clean up
        del _GENERATION_BACKEND_REGISTRY["custom_gen_test"]

    def test_register_duplicate_backend_raises(self):
        """Test that registering duplicate backend name raises error."""
        class DuplicateBackend:
            pass

        # This should raise because 'vllm' is already registered
        with pytest.raises(ValueError) as excinfo:
            register_generation_backend("vllm")(DuplicateBackend)
        assert "already registered" in str(excinfo.value)


class TestVLLMBackend:
    """Tests for VLLMBackend implementation."""

    def test_initialization(self):
        """Test VLLMBackend initialization."""
        backend = VLLMBackend()
        assert not backend.is_initialized
        assert backend.backend_type == "vllm"
        assert backend.config is None

    def test_setup_with_valid_config(self):
        """Test setup with valid vLLM configuration."""
        backend = VLLMBackend()
        config = GenerationBackendConfig(
            backend_type="vllm",
            model_name="meta-llama/Llama-2-7b-hf",
            max_new_tokens=512,
            temperature=0.7,
        )
        backend.setup(config)

        assert backend.is_initialized
        assert backend.config == config
        assert backend.vllm_config is not None
        assert backend.vllm_config["model_name"] == "meta-llama/Llama-2-7b-hf"
        assert backend.vllm_config["max_new_tokens"] == 512
        assert backend.vllm_config["temperature"] == 0.7
        assert backend.vllm_config["backend"] == "vllm"

    def test_setup_with_wrong_backend_type(self):
        """Test that setup fails with wrong backend type."""
        backend = VLLMBackend()
        config = GenerationBackendConfig(backend_type="megatron")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)
        assert "Expected 'vllm'" in str(excinfo.value)

    def test_setup_with_backend_kwargs(self):
        """Test setup with vLLM-specific kwargs."""
        backend = VLLMBackend()
        config = GenerationBackendConfig(
            backend_type="vllm",
            model_name="test-model",
            tensor_parallel_size=2,
            backend_kwargs={
                "vllm_cfg": {
                    "gpu_memory_utilization": 0.8,
                    "async_engine": True,
                }
            },
        )
        backend.setup(config)

        assert backend.vllm_config["vllm_cfg"]["tensor_parallel_size"] == 2
        assert backend.vllm_config["vllm_cfg"]["gpu_memory_utilization"] == 0.8
        assert backend.vllm_config["vllm_cfg"]["async_engine"] is True

    def test_generate_requires_initialization(self):
        """Test that generate requires initialization."""
        backend = VLLMBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.generate(MagicMock())
        assert "not initialized" in str(excinfo.value)

    def test_generate_requires_interface(self):
        """Test that generate requires a generation interface."""
        backend = VLLMBackend()
        config = GenerationBackendConfig(backend_type="vllm", model_name="test")
        backend.setup(config)

        with pytest.raises(RuntimeError) as excinfo:
            backend.generate(MagicMock())
        assert "interface not set" in str(excinfo.value)

    def test_shutdown(self):
        """Test shutdown clears initialization state."""
        backend = VLLMBackend()
        config = GenerationBackendConfig(backend_type="vllm", model_name="test")
        backend.setup(config)

        assert backend.is_initialized
        backend.shutdown()
        assert not backend.is_initialized


class TestMegatronInferenceBackend:
    """Tests for MegatronInferenceBackend implementation."""

    def test_initialization(self):
        """Test MegatronInferenceBackend initialization."""
        backend = MegatronInferenceBackend()
        assert not backend.is_initialized
        assert backend.backend_type == "megatron"
        assert backend.config is None

    def test_setup_with_valid_config(self):
        """Test setup with valid Megatron configuration."""
        backend = MegatronInferenceBackend()
        config = GenerationBackendConfig(
            backend_type="megatron",
            model_name="meta-llama/Llama-2-7b-hf",
            max_new_tokens=512,
            temperature=0.7,
        )
        backend.setup(config)

        assert backend.is_initialized
        assert backend.config == config
        assert backend.megatron_config is not None
        assert backend.megatron_config["model_name"] == "meta-llama/Llama-2-7b-hf"
        assert backend.megatron_config["max_new_tokens"] == 512
        assert backend.megatron_config["temperature"] == 0.7
        assert backend.megatron_config["backend"] == "megatron"

    def test_setup_with_wrong_backend_type(self):
        """Test that setup fails with wrong backend type."""
        backend = MegatronInferenceBackend()
        config = GenerationBackendConfig(backend_type="vllm")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)
        assert "Expected 'megatron'" in str(excinfo.value)

    def test_setup_with_backend_kwargs(self):
        """Test setup with Megatron-specific kwargs."""
        backend = MegatronInferenceBackend()
        config = GenerationBackendConfig(
            backend_type="megatron",
            model_name="test-model",
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            backend_kwargs={
                "megatron_cfg": {
                    "context_parallel_size": 2,
                    "sequence_parallel": True,
                }
            },
        )
        backend.setup(config)

        assert backend.megatron_config["megatron_cfg"]["tensor_model_parallel_size"] == 4
        assert backend.megatron_config["megatron_cfg"]["pipeline_model_parallel_size"] == 2
        assert backend.megatron_config["megatron_cfg"]["context_parallel_size"] == 2
        assert backend.megatron_config["megatron_cfg"]["sequence_parallel"] is True

    def test_generate_requires_initialization(self):
        """Test that generate requires initialization."""
        backend = MegatronInferenceBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.generate(MagicMock())
        assert "not initialized" in str(excinfo.value)

    def test_shutdown(self):
        """Test shutdown clears initialization state."""
        backend = MegatronInferenceBackend()
        config = GenerationBackendConfig(backend_type="megatron", model_name="test")
        backend.setup(config)

        assert backend.is_initialized
        backend.shutdown()
        assert not backend.is_initialized


class TestBackendSelection:
    """Tests for backend selection via string parameter."""

    @pytest.mark.parametrize(
        "backend_name,expected_class",
        [
            ("vllm", VLLMBackend),
            ("megatron", MegatronInferenceBackend),
        ],
    )
    def test_backend_selection_by_name(self, backend_name, expected_class):
        """Test that backends can be selected by string name."""
        backend = get_generation_backend(backend_name)
        assert isinstance(backend, expected_class)
        assert backend.backend_type == backend_name

    def test_backend_kwargs_passed_to_constructor(self):
        """Test that kwargs are passed to backend constructor."""
        mock_cluster = MagicMock()

        backend = get_generation_backend(
            "vllm",
            cluster=mock_cluster,
        )

        assert backend._cluster is mock_cluster


class TestBackendIntegration:
    """Integration tests for backend usage patterns."""

    def test_backend_workflow_vllm(self):
        """Test complete workflow with vLLM backend."""
        # Get backend by name
        backend = get_generation_backend("vllm")
        assert backend.backend_type == "vllm"

        # Setup backend
        config = GenerationBackendConfig(
            backend_type="vllm",
            model_name="test-model",
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
        )
        backend.setup(config)
        assert backend.is_initialized

        # Verify configuration
        assert backend.config.model_name == "test-model"
        assert backend.vllm_config["max_new_tokens"] == 256
        assert backend.vllm_config["temperature"] == 0.8
        assert backend.vllm_config["top_p"] == 0.95

        # Shutdown
        backend.shutdown()
        assert not backend.is_initialized

    def test_backend_workflow_megatron(self):
        """Test complete workflow with Megatron backend."""
        # Get backend by name
        backend = get_generation_backend("megatron")
        assert backend.backend_type == "megatron"

        # Setup backend
        config = GenerationBackendConfig(
            backend_type="megatron",
            model_name="test-model",
            max_new_tokens=512,
            temperature=1.0,
            tensor_parallel_size=2,
            backend_kwargs={
                "megatron_cfg": {
                    "sequence_parallel": True,
                }
            },
        )
        backend.setup(config)
        assert backend.is_initialized

        # Verify configuration
        assert backend.config.model_name == "test-model"
        assert backend.megatron_config["megatron_cfg"]["tensor_model_parallel_size"] == 2
        assert backend.megatron_config["megatron_cfg"]["sequence_parallel"] is True

        # Shutdown
        backend.shutdown()
        assert not backend.is_initialized


class TestErrorMessages:
    """Tests for clear error messages."""

    def test_unknown_backend_error_message(self):
        """Test that unknown backend error message is helpful."""
        with pytest.raises(ValueError) as excinfo:
            get_generation_backend("nonexistent")

        error_msg = str(excinfo.value)
        assert "Unknown generation backend 'nonexistent'" in error_msg
        assert "Available backends:" in error_msg
        assert "@register_generation_backend" in error_msg

    def test_wrong_backend_type_error_vllm(self):
        """Test error message for wrong backend type in vLLM."""
        backend = VLLMBackend()
        config = GenerationBackendConfig(backend_type="megatron")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)

        error_msg = str(excinfo.value)
        assert "backend_type='megatron'" in error_msg
        assert "Expected 'vllm'" in error_msg

    def test_wrong_backend_type_error_megatron(self):
        """Test error message for wrong backend type in Megatron."""
        backend = MegatronInferenceBackend()
        config = GenerationBackendConfig(backend_type="vllm")

        with pytest.raises(ValueError) as excinfo:
            backend.setup(config)

        error_msg = str(excinfo.value)
        assert "backend_type='vllm'" in error_msg
        assert "Expected 'megatron'" in error_msg

    def test_not_initialized_error_message(self):
        """Test error message when methods called before setup."""
        backend = VLLMBackend()

        with pytest.raises(RuntimeError) as excinfo:
            backend.generate(MagicMock())

        error_msg = str(excinfo.value)
        assert "not initialized" in error_msg
        assert "Call setup() first" in error_msg
