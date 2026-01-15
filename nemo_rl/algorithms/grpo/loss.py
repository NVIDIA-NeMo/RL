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
"""GRPO loss functions and advantage computation.

This module contains the loss functions and advantage normalization
utilities for GRPO training.
"""

from typing import Any

import torch

from nemo_rl.algorithms.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def normalize_advantages_with_epsilon(
    batch: BatchedDataDict,
    use_leave_one_out_baseline: bool,
    num_generations_per_prompt: int,
    should_normalize: bool,
    std_eps: float = 1e-8,
) -> BatchedDataDict:
    """Normalize advantages within each prompt group.

    When normalizing, computes mean and std per prompt group
    and applies z-score normalization.

    Args:
        batch: BatchedDataDict with rewards.
        use_leave_one_out_baseline: If True, use leave-one-out for baseline.
        num_generations_per_prompt: Number of generations per prompt.
        should_normalize: Whether to normalize advantages.
        std_eps: Small epsilon for numerical stability.

    Returns:
        BatchedDataDict with normalized advantages added.
    """
    from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt

    batch_size = batch["rewards"].shape[0]
    num_prompts = batch_size // num_generations_per_prompt
    shaped_rewards = batch["rewards"].view(num_prompts, num_generations_per_prompt)

    # Calculate baseline and std
    baseline, std = calculate_baseline_and_std_per_prompt(
        shaped_rewards,
        use_leave_one_out=use_leave_one_out_baseline,
    )

    # Compute advantages
    advantages = shaped_rewards - baseline.unsqueeze(-1)

    if should_normalize:
        advantages = advantages / (std.unsqueeze(-1) + std_eps)

    batch["advantages"] = advantages.view(-1)
    return batch


def scale_rewards(
    rewards: torch.Tensor,
    source_min: float = 0.0,
    source_max: float = 1.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> torch.Tensor:
    """Scale rewards from source range to target range.

    Clamps rewards to [source_min, source_max] and linearly maps
    to [target_min, target_max].

    Args:
        rewards: Reward tensor to scale.
        source_min: Minimum of source range.
        source_max: Maximum of source range.
        target_min: Minimum of target range.
        target_max: Maximum of target range.

    Returns:
        Scaled reward tensor.
    """
    # Clamp to source range
    clamped = torch.clamp(rewards, source_min, source_max)

    # Linear mapping
    source_range = source_max - source_min
    target_range = target_max - target_min

    if source_range == 0:
        return torch.full_like(rewards, (target_min + target_max) / 2)

    scaled = (clamped - source_min) / source_range * target_range + target_min
    return scaled


def create_loss_function(config: ClippedPGLossConfig) -> ClippedPGLossFn:
    """Create the GRPO loss function.

    Args:
        config: Loss function configuration.

    Returns:
        Configured ClippedPGLossFn instance.
    """
    return ClippedPGLossFn(config)


def compute_grpo_loss(
    batch: BatchedDataDict,
    loss_fn: ClippedPGLossFn,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the GRPO training loss.

    Args:
        batch: BatchedDataDict with all required fields.
        loss_fn: Loss function to use.

    Returns:
        Tuple of (loss tensor, metrics dict).
    """
    loss, metrics = loss_fn(batch)
    return loss, metrics
