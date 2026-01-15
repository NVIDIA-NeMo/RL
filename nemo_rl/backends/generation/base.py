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
"""Generation backend protocol and registry.

This module defines the GenerationBackend protocol that all generation backends
must implement, along with a registry for backend selection by name.

Example:
    >>> from nemo_rl.backends.generation import GenerationBackend, get_generation_backend
    >>>
    >>> # Get a backend by name
    >>> backend = get_generation_backend('vllm')
    >>>
    >>> # Setup the backend
    >>> backend.setup(config)
    >>>
    >>> # Generate text
    >>> outputs = backend.generate(prompts)
    >>>
    >>> # Update weights from training
    >>> backend.update_weights(state_dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import ray

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.interfaces import (
        GenerationDatumSpec,
        GenerationOutputSpec,
    )


@dataclass
class GenerationBackendConfig:
    """Configuration for generation backends.

    This provides a unified configuration interface for all generation backends,
    allowing backend selection via a single string parameter.

    Attributes:
        backend_type: The type of backend to use ('vllm' or 'megatron').
        model_name: Name or path of the model to load.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling parameter.
        top_k: Top-k sampling parameter (None disables).
        tensor_parallel_size: Tensor parallelism size.
        pipeline_parallel_size: Pipeline parallelism size.
        backend_kwargs: Additional backend-specific configuration.
    """

    backend_type: str = "vllm"
    model_name: str = ""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    stop_token_ids: Optional[list[int]] = None
    stop_strings: Optional[list[str]] = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GenerationBackend(Protocol):
    """Protocol defining the generation backend interface.

    All generation backends must implement this interface to be compatible
    with the NeMo RL generation framework.

    Methods:
        setup: Initialize the backend with configuration.
        generate: Generate text from prompts.
        update_weights: Update model weights from training.
        prepare_for_generation: Prepare for generation mode.
        finish_generation: Clean up after generation.
        shutdown: Clean up resources.
    """

    def setup(self, config: GenerationBackendConfig) -> None:
        """Initialize the backend with the given configuration.

        Args:
            config: Generation backend configuration.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If setup fails.
        """
        ...

    def generate(
        self,
        prompts: "BatchedDataDict[GenerationDatumSpec]",
        greedy: bool = False,
        sampling_params: Optional[dict[str, Any]] = None,
    ) -> "BatchedDataDict[GenerationOutputSpec]":
        """Generate text from input prompts.

        Args:
            prompts: BatchedDataDict containing input_ids and input_lengths.
            greedy: If True, use greedy decoding instead of sampling.
            sampling_params: Optional override for sampling parameters.

        Returns:
            BatchedDataDict containing:
                - output_ids: Generated token sequences
                - generation_lengths: Length of generated responses
                - unpadded_sequence_lengths: Total sequence lengths
                - logprobs: Log probabilities of generated tokens
        """
        ...

    def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights from a state dictionary.

        This is typically called after training to sync weights from
        the training policy to the generation backend.

        Args:
            state_dict: Dictionary containing model state.
        """
        ...

    def prepare_for_generation(self) -> None:
        """Prepare the backend for generation mode.

        This should be called before generate() to ensure the model
        is in the correct state.
        """
        ...

    def finish_generation(self) -> None:
        """Clean up after generation.

        This should be called after generation is complete.
        """
        ...

    def shutdown(self) -> None:
        """Clean up resources and shut down the backend.

        This should be called when the backend is no longer needed.
        """
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the backend has been initialized.

        Returns:
            True if setup() has been called successfully.
        """
        ...

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier.

        Returns:
            String identifier for this backend type.
        """
        ...


# Backend registry
_GENERATION_BACKEND_REGISTRY: dict[str, type[GenerationBackend]] = {}

T = TypeVar("T", bound=GenerationBackend)


def register_generation_backend(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a generation backend implementation.

    Args:
        name: The name to register the backend under.

    Returns:
        Decorator function that registers the class.

    Example:
        >>> @register_generation_backend('custom')
        ... class CustomBackend:
        ...     def setup(self, config): ...
        ...     # ... implement other methods
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _GENERATION_BACKEND_REGISTRY:
            raise ValueError(
                f"Generation backend '{name}' is already registered. "
                f"Registered backends: {list(_GENERATION_BACKEND_REGISTRY.keys())}"
            )
        _GENERATION_BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_generation_backend(name: str, **kwargs: Any) -> GenerationBackend:
    """Get a generation backend instance by name.

    Args:
        name: Name of the backend ('vllm', 'megatron', or custom registered name).
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An instance of the requested generation backend.

    Raises:
        ValueError: If the backend name is not registered.

    Example:
        >>> backend = get_generation_backend('vllm')
        >>> backend.setup(config)
    """
    if name not in _GENERATION_BACKEND_REGISTRY:
        available = list(_GENERATION_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown generation backend '{name}'. "
            f"Available backends: {available}. "
            f"Use @register_generation_backend('{name}') to register a custom backend."
        )

    backend_cls = _GENERATION_BACKEND_REGISTRY[name]
    return backend_cls(**kwargs)


def list_generation_backends() -> list[str]:
    """List all registered generation backend names.

    Returns:
        List of registered backend names.
    """
    return list(_GENERATION_BACKEND_REGISTRY.keys())
