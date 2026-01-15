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
"""Single-function training API for NeMo RL.

This module provides the `train()` function - the simplest way to train
reinforcement learning models with NeMo RL.

Example:
    >>> import nemo_rl
    >>> 
    >>> # Minimal 5-line training
    >>> result = nemo_rl.train(
    ...     model="Qwen/Qwen2.5-1.5B",
    ...     dataset="nvidia/OpenMathInstruct-2",
    ...     reward_fn=lambda p, r: 1.0 if "correct" in r else 0.0,
    ... )
    >>> 
    >>> # Access trained model
    >>> trainer = result.trainer
    >>> print(f"Final loss: {result.metrics['loss']}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    from nemo_rl.trainers.base import BaseTrainer, TrainingResult
    from nemo_rl.trainers.callbacks import Callback
    from nemo_rl.data.module import DataModule

logger = logging.getLogger(__name__)

# Type aliases
RewardFnType = Callable[[str, str], Union[float, Dict[str, float]]]
DataType = Union[str, Sequence[Mapping[str, Any]], "DataModule", Any]


@dataclass
class TrainResult:
    """Result of a training run.

    Attributes:
        trainer: The trainer instance (for further operations like inference).
        metrics: Dictionary of final training metrics.
        checkpoint_path: Path to the best checkpoint saved.
        total_steps: Total number of training steps completed.
    """

    trainer: "BaseTrainer"
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    total_steps: int = 0

    def __repr__(self) -> str:
        return (
            f"TrainResult(steps={self.total_steps}, "
            f"loss={self.metrics.get('loss', 'N/A'):.4f})"
        )


def train(
    model: str,
    dataset: Optional[DataType] = None,
    reward_fn: Optional[RewardFnType] = None,
    algorithm: str = "grpo",
    max_steps: int = 1000,
    max_epochs: int = 1,
    learning_rate: float = 1e-6,
    batch_size: int = 32,
    num_generations_per_prompt: int = 16,
    output_dir: Optional[str] = None,
    callbacks: Optional[Sequence["Callback"]] = None,
    **kwargs: Any,
) -> TrainResult:
    """Train a language model with reinforcement learning.

    This is the simplest API for training RL models. It creates a trainer
    with sensible defaults and runs training.

    Args:
        model: Model identifier. Can be:
            - HuggingFace Hub model name (e.g., "Qwen/Qwen2.5-1.5B")
            - Local path to model directory
        dataset: Training data. Can be:
            - HuggingFace dataset name (e.g., "nvidia/OpenMathInstruct-2")
            - List of dictionaries with "prompt" keys
            - DataModule instance
            - Local path to data file
        reward_fn: Reward function with signature `(prompt, response) -> float`.
            Required for GRPO. Not required for SFT.
        algorithm: Training algorithm. One of: "grpo", "sft", "dpo".
            Default: "grpo"
        max_steps: Maximum number of training steps. Default: 1000
        max_epochs: Maximum number of training epochs. Default: 1
        learning_rate: Learning rate. Default: 1e-6
        batch_size: Training batch size. Default: 32
        num_generations_per_prompt: Number of generations per prompt (GRPO only).
            Default: 16
        output_dir: Directory to save checkpoints and logs.
            Default: "./outputs/{model_name}"
        callbacks: List of callback instances for custom hooks.
        **kwargs: Additional algorithm-specific parameters.

    Returns:
        TrainResult with trainer, metrics, and checkpoint path.

    Raises:
        ValueError: If required parameters are missing for the algorithm.

    Example:
        >>> # Minimal GRPO training
        >>> result = nemo_rl.train(
        ...     model="Qwen/Qwen2.5-1.5B",
        ...     dataset="nvidia/OpenMathInstruct-2",
        ...     reward_fn=my_reward_fn,
        ... )
        >>> 
        >>> # SFT (no reward function needed)
        >>> result = nemo_rl.train(
        ...     model="Qwen/Qwen2.5-1.5B",
        ...     dataset="nvidia/OpenMathInstruct-2",
        ...     algorithm="sft",
        ... )
        >>> 
        >>> # Custom parameters
        >>> result = nemo_rl.train(
        ...     model="Qwen/Qwen2.5-1.5B",
        ...     dataset="my_data.json",
        ...     reward_fn=my_reward_fn,
        ...     max_steps=5000,
        ...     learning_rate=5e-7,
        ...     num_generations_per_prompt=8,
        ... )
    """
    # Validate inputs
    algorithm = algorithm.lower()
    _validate_inputs(algorithm, model, dataset, reward_fn)

    # Create trainer with appropriate config
    trainer = _create_trainer_for_algorithm(
        algorithm=algorithm,
        model=model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_steps=max_steps,
        max_epochs=max_epochs,
        num_generations_per_prompt=num_generations_per_prompt,
        output_dir=output_dir,
        **kwargs,
    )

    # Setup reward function for GRPO
    if algorithm == "grpo" and reward_fn is not None:
        _setup_reward_function(trainer, reward_fn)

    # Run training
    logger.info(f"Starting {algorithm.upper()} training with model: {model}")

    training_result = trainer.fit(
        dataset=dataset,
        max_steps=max_steps,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )

    # Build result
    return TrainResult(
        trainer=trainer,
        metrics=training_result.metrics,
        checkpoint_path=training_result.best_checkpoint_path,
        total_steps=training_result.total_steps,
    )


def _validate_inputs(
    algorithm: str,
    model: str,
    dataset: Optional[DataType],
    reward_fn: Optional[RewardFnType],
) -> None:
    """Validate training inputs.

    Args:
        algorithm: Training algorithm name.
        model: Model identifier.
        dataset: Training data.
        reward_fn: Reward function.

    Raises:
        ValueError: If inputs are invalid.
    """
    valid_algorithms = {"grpo", "sft", "dpo"}

    if algorithm not in valid_algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Valid options: {', '.join(sorted(valid_algorithms))}"
        )

    if not model:
        raise ValueError(
            "Model is required. Provide a HuggingFace model name "
            "(e.g., 'Qwen/Qwen2.5-1.5B') or local path."
        )

    if dataset is None:
        raise ValueError(
            "Dataset is required. Provide a HuggingFace dataset name "
            "(e.g., 'nvidia/OpenMathInstruct-2'), list of dicts, or DataModule."
        )

    if algorithm == "grpo" and reward_fn is None:
        raise ValueError(
            "reward_fn is required for GRPO training. "
            "Provide a function with signature (prompt: str, response: str) -> float"
        )


def _create_trainer_for_algorithm(
    algorithm: str,
    model: str,
    learning_rate: float,
    batch_size: int,
    max_steps: int,
    max_epochs: int,
    num_generations_per_prompt: int,
    output_dir: Optional[str],
    **kwargs: Any,
) -> "BaseTrainer":
    """Create a trainer for the specified algorithm.

    Args:
        algorithm: Algorithm name.
        model: Model identifier.
        learning_rate: Learning rate.
        batch_size: Batch size.
        max_steps: Max training steps.
        max_epochs: Max epochs.
        num_generations_per_prompt: Generations per prompt (GRPO).
        output_dir: Output directory.
        **kwargs: Additional parameters.

    Returns:
        Configured trainer instance.
    """
    # Determine output directory
    if output_dir is None:
        model_name = model.split("/")[-1] if "/" in model else model
        output_dir = f"./outputs/{model_name}"

    # Build config based on algorithm
    if algorithm == "grpo":
        config = _build_grpo_config(
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            max_epochs=max_epochs,
            num_generations_per_prompt=num_generations_per_prompt,
            output_dir=output_dir,
            **kwargs,
        )
        from nemo_rl.algorithms.grpo import GRPOTrainer

        return GRPOTrainer(config)

    elif algorithm == "sft":
        config = _build_sft_config(
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            max_epochs=max_epochs,
            output_dir=output_dir,
            **kwargs,
        )
        from nemo_rl.algorithms.sft import SFTTrainer

        return SFTTrainer(config)

    elif algorithm == "dpo":
        config = _build_dpo_config(
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            max_epochs=max_epochs,
            output_dir=output_dir,
            **kwargs,
        )
        from nemo_rl.algorithms.dpo import DPOTrainer

        return DPOTrainer(config)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _build_grpo_config(
    model: str,
    learning_rate: float,
    batch_size: int,
    max_steps: int,
    max_epochs: int,
    num_generations_per_prompt: int,
    output_dir: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build GRPO configuration.

    Args:
        model: Model identifier.
        learning_rate: Learning rate.
        batch_size: Batch size.
        max_steps: Max steps.
        max_epochs: Max epochs.
        num_generations_per_prompt: Generations per prompt.
        output_dir: Output directory.
        **kwargs: Additional parameters.

    Returns:
        Configuration dictionary.
    """
    return {
        "policy": {
            "model_name": model,
            "learning_rate": learning_rate,
        },
        "grpo": {
            "num_prompts_per_step": batch_size,
            "num_generations_per_prompt": num_generations_per_prompt,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            "normalize_rewards": kwargs.get("normalize_rewards", True),
            "use_leave_one_out_baseline": kwargs.get(
                "use_leave_one_out_baseline", True
            ),
            **{k: v for k, v in kwargs.items() if k.startswith("grpo_")},
        },
        "checkpointing": {
            "checkpoint_dir": f"{output_dir}/checkpoints",
            "enabled": kwargs.get("save_checkpoints", True),
            "save_period": kwargs.get("save_every", 100),
        },
        "logger": {
            "log_level": kwargs.get("log_level", "INFO"),
            "tensorboard_enabled": kwargs.get("tensorboard", True),
            "tensorboard_dir": f"{output_dir}/logs",
        },
    }


def _build_sft_config(
    model: str,
    learning_rate: float,
    batch_size: int,
    max_steps: int,
    max_epochs: int,
    output_dir: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build SFT configuration.

    Args:
        model: Model identifier.
        learning_rate: Learning rate.
        batch_size: Batch size.
        max_steps: Max steps.
        max_epochs: Max epochs.
        output_dir: Output directory.
        **kwargs: Additional parameters.

    Returns:
        Configuration dictionary.
    """
    return {
        "policy": {
            "model_name": model,
            "learning_rate": learning_rate,
        },
        "sft": {
            "batch_size": batch_size,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            **{k: v for k, v in kwargs.items() if k.startswith("sft_")},
        },
        "checkpointing": {
            "checkpoint_dir": f"{output_dir}/checkpoints",
            "enabled": kwargs.get("save_checkpoints", True),
            "save_period": kwargs.get("save_every", 100),
        },
        "logger": {
            "log_level": kwargs.get("log_level", "INFO"),
            "tensorboard_enabled": kwargs.get("tensorboard", True),
            "tensorboard_dir": f"{output_dir}/logs",
        },
    }


def _build_dpo_config(
    model: str,
    learning_rate: float,
    batch_size: int,
    max_steps: int,
    max_epochs: int,
    output_dir: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build DPO configuration.

    Args:
        model: Model identifier.
        learning_rate: Learning rate.
        batch_size: Batch size.
        max_steps: Max steps.
        max_epochs: Max epochs.
        output_dir: Output directory.
        **kwargs: Additional parameters.

    Returns:
        Configuration dictionary.
    """
    return {
        "policy": {
            "model_name": model,
            "learning_rate": learning_rate,
        },
        "dpo": {
            "batch_size": batch_size,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            "beta": kwargs.get("beta", 0.1),
            **{k: v for k, v in kwargs.items() if k.startswith("dpo_")},
        },
        "checkpointing": {
            "checkpoint_dir": f"{output_dir}/checkpoints",
            "enabled": kwargs.get("save_checkpoints", True),
            "save_period": kwargs.get("save_every", 100),
        },
        "logger": {
            "log_level": kwargs.get("log_level", "INFO"),
            "tensorboard_enabled": kwargs.get("tensorboard", True),
            "tensorboard_dir": f"{output_dir}/logs",
        },
    }


def _setup_reward_function(
    trainer: "BaseTrainer",
    reward_fn: RewardFnType,
) -> None:
    """Set up the reward function for the trainer.

    Args:
        trainer: The trainer instance.
        reward_fn: The reward function.
    """
    from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

    # Wrap the reward function
    reward_wrapper = FunctionalRewardWrapper(reward_fn, name="train_reward")

    # Attach to trainer (implementation-specific)
    # This will be used by the rollout engine
    if hasattr(trainer, "_reward_wrapper"):
        trainer._reward_wrapper = reward_wrapper
    elif hasattr(trainer, "_rollout_engine") and trainer._rollout_engine is not None:
        trainer._rollout_engine._environment = reward_wrapper

    logger.debug(f"Reward function configured: {reward_wrapper}")
