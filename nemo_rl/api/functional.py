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
"""Functional interface helpers for NeMo RL.

This module provides helper functions for working with trainers
in a functional style.

Example:
    >>> from nemo_rl.api import create_trainer, list_algorithms
    >>> 
    >>> # List available algorithms
    >>> print(list_algorithms())  # ['grpo', 'sft', 'dpo']
    >>> 
    >>> # Create trainer dynamically
    >>> trainer = create_trainer("grpo", model="Qwen/Qwen2.5-1.5B")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from nemo_rl.trainers.base import BaseTrainer


# Registry of available algorithms
_ALGORITHM_REGISTRY: Dict[str, Type["BaseTrainer"]] = {}


def _ensure_registry_populated() -> None:
    """Ensure the algorithm registry is populated."""
    global _ALGORITHM_REGISTRY

    if _ALGORITHM_REGISTRY:
        return

    # Lazy import to avoid circular dependencies
    try:
        from nemo_rl.algorithms.grpo import GRPOTrainer

        _ALGORITHM_REGISTRY["grpo"] = GRPOTrainer
    except ImportError:
        pass

    try:
        from nemo_rl.algorithms.sft import SFTTrainer

        _ALGORITHM_REGISTRY["sft"] = SFTTrainer
    except ImportError:
        pass

    try:
        from nemo_rl.algorithms.dpo import DPOTrainer

        _ALGORITHM_REGISTRY["dpo"] = DPOTrainer
    except ImportError:
        pass


def list_algorithms() -> List[str]:
    """List available training algorithms.

    Returns:
        List of algorithm names that can be used with create_trainer().

    Example:
        >>> list_algorithms()
        ['grpo', 'sft', 'dpo']
    """
    _ensure_registry_populated()
    return sorted(_ALGORITHM_REGISTRY.keys())


def get_algorithm(name: str) -> Type["BaseTrainer"]:
    """Get the trainer class for an algorithm.

    Args:
        name: Algorithm name (case-insensitive).

    Returns:
        Trainer class for the algorithm.

    Raises:
        ValueError: If algorithm is not found.

    Example:
        >>> GRPOTrainer = get_algorithm("grpo")
        >>> trainer = GRPOTrainer(config)
    """
    _ensure_registry_populated()
    name = name.lower()

    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown algorithm: {name}. "
            f"Available algorithms: {available}"
        )

    return _ALGORITHM_REGISTRY[name]


def create_trainer(
    algorithm: str,
    model: str,
    **kwargs: Any,
) -> "BaseTrainer":
    """Create a trainer for the specified algorithm.

    This is a convenience function that creates a trainer with
    sensible defaults. For more control, use the trainer classes directly.

    Args:
        algorithm: Algorithm name ('grpo', 'sft', 'dpo').
        model: Model identifier (HuggingFace name or local path).
        **kwargs: Additional configuration options.

    Returns:
        Configured trainer instance.

    Raises:
        ValueError: If algorithm is unknown.

    Example:
        >>> trainer = create_trainer(
        ...     "grpo",
        ...     model="Qwen/Qwen2.5-1.5B",
        ...     learning_rate=1e-6,
        ...     batch_size=32,
        ... )
    """
    trainer_class = get_algorithm(algorithm)

    # Build config from kwargs
    config = _build_config_for_trainer(
        algorithm=algorithm,
        model=model,
        **kwargs,
    )

    return trainer_class(config)


def _build_config_for_trainer(
    algorithm: str,
    model: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build configuration dictionary for a trainer.

    Args:
        algorithm: Algorithm name.
        model: Model identifier.
        **kwargs: Configuration overrides.

    Returns:
        Configuration dictionary.
    """
    # Extract common parameters
    learning_rate = kwargs.pop("learning_rate", 1e-6)
    batch_size = kwargs.pop("batch_size", 32)
    max_steps = kwargs.pop("max_steps", 1000)
    max_epochs = kwargs.pop("max_epochs", 1)

    # Base policy config
    config: Dict[str, Any] = {
        "policy": {
            "model_name": model,
            "learning_rate": learning_rate,
        },
    }

    # Algorithm-specific config
    if algorithm == "grpo":
        config["grpo"] = {
            "num_prompts_per_step": batch_size,
            "num_generations_per_prompt": kwargs.pop("num_generations_per_prompt", 16),
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            "normalize_rewards": kwargs.pop("normalize_rewards", True),
            "use_leave_one_out_baseline": kwargs.pop("use_leave_one_out_baseline", True),
        }
    elif algorithm == "sft":
        config["sft"] = {
            "batch_size": batch_size,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
        }
    elif algorithm == "dpo":
        config["dpo"] = {
            "batch_size": batch_size,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            "beta": kwargs.pop("beta", 0.1),
        }

    # Add any remaining kwargs to the algorithm config
    algo_key = algorithm
    if algo_key in config:
        config[algo_key].update(kwargs)

    return config


def register_algorithm(name: str, trainer_class: Type["BaseTrainer"]) -> None:
    """Register a custom algorithm.

    This allows users to add custom trainers to the registry
    for use with create_trainer() and list_algorithms().

    Args:
        name: Algorithm name (will be lowercased).
        trainer_class: Trainer class to register.

    Example:
        >>> from nemo_rl.trainers.base import BaseTrainer
        >>> 
        >>> class MyCustomTrainer(BaseTrainer):
        ...     def _train_step(self, batch):
        ...         return {"loss": 0.0}
        ...     def _compute_loss(self, batch, outputs):
        ...         return 0.0
        >>> 
        >>> register_algorithm("custom", MyCustomTrainer)
        >>> trainer = create_trainer("custom", model="...")
    """
    _ensure_registry_populated()
    _ALGORITHM_REGISTRY[name.lower()] = trainer_class
