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
"""Training backend protocol and registry.

This module defines the TrainingBackend protocol that all training backends
must implement, along with a registry for backend selection by name.

Example:
    >>> from nemo_rl.backends.training import TrainingBackend, get_training_backend
    >>>
    >>> # Get a backend by name
    >>> backend = get_training_backend('dtensor')
    >>>
    >>> # Setup the backend
    >>> backend.setup(config)
    >>>
    >>> # Train a step
    >>> metrics = backend.train_step(batch, loss_fn)
    >>>
    >>> # Get log probabilities
    >>> logprobs = backend.get_logprobs(batch)
    >>>
    >>> # Save checkpoint
    >>> backend.save_checkpoint('/path/to/checkpoint')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    import torch

    from nemo_rl.algorithms.interfaces import LossFunction
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@dataclass
class TrainingBackendConfig:
    """Configuration for training backends.

    This provides a unified configuration interface for all training backends,
    allowing backend selection via a single string parameter.

    Attributes:
        backend_type: The type of backend to use ('dtensor' or 'megatron').
        model_name: Name or path of the model to load.
        precision: Model precision ('float32', 'bfloat16', 'float16').
        train_global_batch_size: Global batch size for training.
        train_micro_batch_size: Micro batch size for gradient accumulation.
        max_grad_norm: Maximum gradient norm for clipping.
        backend_kwargs: Additional backend-specific configuration.
    """

    backend_type: str = "dtensor"
    model_name: str = ""
    precision: str = "bfloat16"
    train_global_batch_size: int = 32
    train_micro_batch_size: int = 4
    max_grad_norm: float = 1.0
    backend_kwargs: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrainingBackend(Protocol):
    """Protocol defining the training backend interface.

    All training backends must implement this interface to be compatible
    with the NeMo RL training framework.

    Methods:
        setup: Initialize the backend with configuration.
        train_step: Execute a single training step.
        get_logprobs: Compute log probabilities for input data.
        save_checkpoint: Save model state to disk.
        load_checkpoint: Load model state from disk.
        prepare_for_training: Prepare the backend for training mode.
        prepare_for_inference: Prepare the backend for inference mode.
        shutdown: Clean up resources.
    """

    def setup(self, config: TrainingBackendConfig) -> None:
        """Initialize the backend with the given configuration.

        Args:
            config: Training backend configuration.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If setup fails.
        """
        ...

    def train_step(
        self,
        batch: "BatchedDataDict[Any]",
        loss_fn: "LossFunction",
        eval_mode: bool = False,
        global_batch_size: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute a single training step.

        Args:
            batch: Batched input data.
            loss_fn: Loss function to compute gradients.
            eval_mode: If True, don't update weights (evaluation only).
            global_batch_size: Override global batch size.
            micro_batch_size: Override micro batch size.

        Returns:
            Dictionary containing training metrics:
                - 'loss': Scalar loss value
                - 'grad_norm': Gradient norm (if not eval_mode)
                - Additional algorithm-specific metrics
        """
        ...

    def get_logprobs(
        self,
        batch: "BatchedDataDict[Any]",
        micro_batch_size: Optional[int] = None,
    ) -> "BatchedDataDict[Any]":
        """Compute log probabilities for the input batch.

        Args:
            batch: Batched input data containing 'input_ids' and 'input_lengths'.
            micro_batch_size: Override micro batch size for inference.

        Returns:
            BatchedDataDict containing 'logprobs' tensor of shape [batch_size, seq_len].
        """
        ...

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer_path: Optional[str | Path] = None,
        tokenizer_path: Optional[str | Path] = None,
    ) -> None:
        """Save model checkpoint to disk.

        Args:
            path: Path to save model weights.
            optimizer_path: Optional path to save optimizer state.
            tokenizer_path: Optional path to save tokenizer.

        Raises:
            IOError: If saving fails.
        """
        ...

    def load_checkpoint(
        self,
        path: str | Path,
        optimizer_path: Optional[str | Path] = None,
    ) -> None:
        """Load model checkpoint from disk.

        Args:
            path: Path to load model weights from.
            optimizer_path: Optional path to load optimizer state from.

        Raises:
            IOError: If loading fails.
            ValueError: If checkpoint format is incompatible.
        """
        ...

    def prepare_for_training(self) -> None:
        """Prepare the backend for training mode.

        This should be called before train_step() to ensure the model
        is in the correct state (e.g., on correct device, train mode).
        """
        ...

    def prepare_for_inference(self) -> None:
        """Prepare the backend for inference mode.

        This should be called before get_logprobs() to ensure the model
        is in the correct state (e.g., on correct device, eval mode).
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
_TRAINING_BACKEND_REGISTRY: dict[str, type[TrainingBackend]] = {}

T = TypeVar("T", bound=TrainingBackend)


def register_training_backend(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a training backend implementation.

    Args:
        name: The name to register the backend under.

    Returns:
        Decorator function that registers the class.

    Example:
        >>> @register_training_backend('custom')
        ... class CustomBackend:
        ...     def setup(self, config): ...
        ...     # ... implement other methods
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _TRAINING_BACKEND_REGISTRY:
            raise ValueError(
                f"Training backend '{name}' is already registered. "
                f"Registered backends: {list(_TRAINING_BACKEND_REGISTRY.keys())}"
            )
        _TRAINING_BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_training_backend(name: str, **kwargs: Any) -> TrainingBackend:
    """Get a training backend instance by name.

    Args:
        name: Name of the backend ('dtensor', 'megatron', or custom registered name).
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An instance of the requested training backend.

    Raises:
        ValueError: If the backend name is not registered.

    Example:
        >>> backend = get_training_backend('dtensor')
        >>> backend.setup(config)
    """
    if name not in _TRAINING_BACKEND_REGISTRY:
        available = list(_TRAINING_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown training backend '{name}'. "
            f"Available backends: {available}. "
            f"Use @register_training_backend('{name}') to register a custom backend."
        )

    backend_cls = _TRAINING_BACKEND_REGISTRY[name]
    return backend_cls(**kwargs)


def list_training_backends() -> list[str]:
    """List all registered training backend names.

    Returns:
        List of registered backend names.
    """
    return list(_TRAINING_BACKEND_REGISTRY.keys())
