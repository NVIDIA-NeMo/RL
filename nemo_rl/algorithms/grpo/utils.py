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
"""GRPO utility functions.

This module contains helper functions for GRPO training, including
backend detection and policy management.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo.config import MasterConfig


def should_use_async_rollouts(master_config: "MasterConfig") -> bool:
    """Determine if async rollouts should be used.

    Async rollouts are enabled when:
    - Environment supports async operations
    - Async GRPO is enabled

    Args:
        master_config: Full training configuration.

    Returns:
        True if async rollouts should be used.
    """
    grpo_config = master_config.get("grpo", {})
    async_config = grpo_config.get("async_grpo", {})

    return async_config.get("enabled", False)


def should_use_nemo_gym(master_config: "MasterConfig") -> bool:
    """Determine if NeMo Gym environment should be used.

    Checks if the environment configuration indicates a NeMo Gym environment.

    Args:
        master_config: Full training configuration.

    Returns:
        True if NeMo Gym should be used.
    """
    env_config = master_config.get("env", {})
    env_type = env_config.get("type", "")

    return "nemo_gym" in env_type.lower()


def get_generation_backend(master_config: "MasterConfig") -> str:
    """Get the generation backend type from config.

    Args:
        master_config: Full training configuration.

    Returns:
        Backend type string ('vllm', 'megatron', etc.)
    """
    policy_config = master_config.get("policy", {})
    generation_config = policy_config.get("generation", {})

    return generation_config.get("backend", "vllm")


def is_colocation_enabled(master_config: "MasterConfig") -> bool:
    """Check if policy-generation colocation is enabled.

    Args:
        master_config: Full training configuration.

    Returns:
        True if colocation is enabled.
    """
    policy_config = master_config.get("policy", {})
    generation_config = policy_config.get("generation", {})
    colocation = generation_config.get("colocated", {})

    return colocation.get("enabled", False)


def compute_effective_batch_size(
    num_prompts_per_step: int,
    num_generations_per_prompt: int,
) -> int:
    """Compute the effective batch size for training.

    Args:
        num_prompts_per_step: Number of prompts per step.
        num_generations_per_prompt: Generations per prompt.

    Returns:
        Total number of samples per step.
    """
    return num_prompts_per_step * num_generations_per_prompt


def log_training_metrics(
    metrics: dict[str, Any],
    step: int,
    logger: Any,
) -> None:
    """Log training metrics to the logger.

    Args:
        metrics: Dictionary of metrics to log.
        step: Current training step.
        logger: Logger instance.
    """
    if logger is None:
        return

    logger.log_metrics(metrics, step=step)
