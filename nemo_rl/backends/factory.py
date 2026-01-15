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
"""Unified backend factory for NeMo RL.

This module provides a unified BackendFactory class that handles both training
and generation backend creation, along with custom backend registration.

Example:
    >>> from nemo_rl.backends import BackendFactory
    >>>
    >>> # Get built-in backends
    >>> training_backend = BackendFactory.get_training_backend('dtensor')
    >>> generation_backend = BackendFactory.get_generation_backend('vllm')
    >>>
    >>> # Register custom backend
    >>> @BackendFactory.register_training_backend('custom')
    ... class CustomTrainingBackend:
    ...     ...
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from nemo_rl.backends.generation.base import (
    GenerationBackend,
    GenerationBackendConfig,
    _GENERATION_BACKEND_REGISTRY,
    get_generation_backend as _get_generation_backend,
    list_generation_backends as _list_generation_backends,
    register_generation_backend as _register_generation_backend,
)
from nemo_rl.backends.training.base import (
    TrainingBackend,
    TrainingBackendConfig,
    _TRAINING_BACKEND_REGISTRY,
    get_training_backend as _get_training_backend,
    list_training_backends as _list_training_backends,
    register_training_backend as _register_training_backend,
)

T = TypeVar("T")


class BackendFactory:
    """Unified factory for creating training and generation backends.

    This class provides a centralized API for backend creation and registration.
    It supports both built-in backends and custom user-defined backends.

    Built-in Training Backends:
        - 'dtensor': DTensor-based distributed training
        - 'megatron': NVIDIA Megatron-LM based training

    Built-in Generation Backends:
        - 'vllm': vLLM-based high-throughput inference
        - 'megatron': Megatron-LM based inference

    Example:
        >>> # Get backends by name
        >>> training = BackendFactory.get_training_backend('dtensor')
        >>> generation = BackendFactory.get_generation_backend('vllm')
        >>>
        >>> # List available backends
        >>> print(BackendFactory.list_training_backends())
        ['dtensor', 'megatron']
        >>> print(BackendFactory.list_generation_backends())
        ['vllm', 'megatron']
        >>>
        >>> # Register custom backend
        >>> @BackendFactory.register_training_backend('my_custom')
        ... class MyCustomBackend:
        ...     def setup(self, config): ...
        ...     # implement other required methods
    """

    # Training backend methods
    @staticmethod
    def get_training_backend(name: str, **kwargs: Any) -> TrainingBackend:
        """Get a training backend instance by name.

        Args:
            name: Name of the backend ('dtensor', 'megatron', or custom).
            **kwargs: Additional arguments passed to backend constructor.

        Returns:
            An instance of the requested training backend.

        Raises:
            ValueError: If the backend name is not registered.

        Example:
            >>> backend = BackendFactory.get_training_backend('dtensor')
            >>> backend.setup(TrainingBackendConfig(model_name='my-model'))
        """
        return _get_training_backend(name, **kwargs)

    @staticmethod
    def list_training_backends() -> list[str]:
        """List all registered training backend names.

        Returns:
            List of registered training backend names.

        Example:
            >>> BackendFactory.list_training_backends()
            ['dtensor', 'megatron']
        """
        return _list_training_backends()

    @staticmethod
    def register_training_backend(name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a custom training backend.

        Args:
            name: The name to register the backend under.

        Returns:
            Decorator function that registers the class.

        Raises:
            ValueError: If the name is already registered.

        Example:
            >>> @BackendFactory.register_training_backend('custom')
            ... class CustomTrainingBackend:
            ...     def setup(self, config): ...
            ...     def train_step(self, batch, loss_fn): ...
            ...     # ... implement other required methods
        """
        return _register_training_backend(name)

    # Generation backend methods
    @staticmethod
    def get_generation_backend(name: str, **kwargs: Any) -> GenerationBackend:
        """Get a generation backend instance by name.

        Args:
            name: Name of the backend ('vllm', 'megatron', or custom).
            **kwargs: Additional arguments passed to backend constructor.

        Returns:
            An instance of the requested generation backend.

        Raises:
            ValueError: If the backend name is not registered.

        Example:
            >>> backend = BackendFactory.get_generation_backend('vllm')
            >>> backend.setup(GenerationBackendConfig(model_name='my-model'))
        """
        return _get_generation_backend(name, **kwargs)

    @staticmethod
    def list_generation_backends() -> list[str]:
        """List all registered generation backend names.

        Returns:
            List of registered generation backend names.

        Example:
            >>> BackendFactory.list_generation_backends()
            ['vllm', 'megatron']
        """
        return _list_generation_backends()

    @staticmethod
    def register_generation_backend(name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a custom generation backend.

        Args:
            name: The name to register the backend under.

        Returns:
            Decorator function that registers the class.

        Raises:
            ValueError: If the name is already registered.

        Example:
            >>> @BackendFactory.register_generation_backend('custom')
            ... class CustomGenerationBackend:
            ...     def setup(self, config): ...
            ...     def generate(self, prompts): ...
            ...     # ... implement other required methods
        """
        return _register_generation_backend(name)

    # Unified methods
    @staticmethod
    def list_all_backends() -> dict[str, list[str]]:
        """List all registered backends organized by type.

        Returns:
            Dictionary with 'training' and 'generation' keys, each containing
            a list of registered backend names.

        Example:
            >>> BackendFactory.list_all_backends()
            {'training': ['dtensor', 'megatron'], 'generation': ['vllm', 'megatron']}
        """
        return {
            "training": _list_training_backends(),
            "generation": _list_generation_backends(),
        }

    @staticmethod
    def is_backend_registered(backend_type: str, name: str) -> bool:
        """Check if a backend is registered.

        Args:
            backend_type: Either 'training' or 'generation'.
            name: Name of the backend to check.

        Returns:
            True if the backend is registered, False otherwise.

        Raises:
            ValueError: If backend_type is not 'training' or 'generation'.

        Example:
            >>> BackendFactory.is_backend_registered('training', 'dtensor')
            True
            >>> BackendFactory.is_backend_registered('training', 'unknown')
            False
        """
        if backend_type == "training":
            return name in _TRAINING_BACKEND_REGISTRY
        elif backend_type == "generation":
            return name in _GENERATION_BACKEND_REGISTRY
        else:
            raise ValueError(
                f"Unknown backend_type '{backend_type}'. "
                "Must be 'training' or 'generation'."
            )


# Convenience functions at module level
get_training_backend = BackendFactory.get_training_backend
get_generation_backend = BackendFactory.get_generation_backend
register_training_backend = BackendFactory.register_training_backend
register_generation_backend = BackendFactory.register_generation_backend
list_training_backends = BackendFactory.list_training_backends
list_generation_backends = BackendFactory.list_generation_backends
