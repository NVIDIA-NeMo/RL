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
"""GRPO data transforms and batch processing.

This module contains functions for data processing, batch construction,
and dynamic sampling for GRPO training.
"""

from typing import Any, Callable, Optional

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def filter_overlong_sequences(
    batch: BatchedDataDict,
    max_sequence_length: int,
) -> BatchedDataDict:
    """Filter out sequences that exceed the maximum length.

    Args:
        batch: BatchedDataDict with sequences.
        max_sequence_length: Maximum allowed sequence length.

    Returns:
        Filtered BatchedDataDict.
    """
    if "unpadded_sequence_lengths" not in batch:
        return batch

    lengths = batch["unpadded_sequence_lengths"]
    mask = lengths <= max_sequence_length

    if mask.all():
        return batch

    # Filter all tensors
    filtered = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == len(mask):
            filtered[key] = value[mask]
        else:
            filtered[key] = value

    return BatchedDataDict(filtered)


def check_batch_has_variance(
    rewards: torch.Tensor,
    num_generations_per_prompt: int,
    std_threshold: float = 1e-6,
) -> tuple[bool, torch.Tensor]:
    """Check if rewards have sufficient variance per prompt.

    For GRPO to learn effectively, we need variance in rewards
    within each prompt group.

    Args:
        rewards: Reward tensor.
        num_generations_per_prompt: Number of generations per prompt.
        std_threshold: Minimum standard deviation threshold.

    Returns:
        Tuple of (has_variance bool, per-prompt std tensor).
    """
    batch_size = rewards.shape[0]
    num_prompts = batch_size // num_generations_per_prompt

    shaped_rewards = rewards.view(num_prompts, num_generations_per_prompt)
    stds = shaped_rewards.std(dim=-1)

    has_variance = (stds > std_threshold).all().item()
    return has_variance, stds


def dynamic_sample_batch(
    batch_generator: Callable[[], BatchedDataDict],
    num_prompts_needed: int,
    num_generations_per_prompt: int,
    max_gen_batches: int = 10,
    std_threshold: float = 1e-6,
) -> BatchedDataDict:
    """Dynamically sample prompts until we have enough with variance.

    Discards prompts whose rewards have zero standard deviation.

    Args:
        batch_generator: Function that generates batches.
        num_prompts_needed: Number of prompts needed.
        num_generations_per_prompt: Generations per prompt.
        max_gen_batches: Maximum batches to generate before error.
        std_threshold: Minimum std for a prompt to be valid.

    Returns:
        BatchedDataDict with valid prompts.

    Raises:
        RuntimeError: If max_gen_batches exceeded.
    """
    valid_samples = []
    total_generated = 0

    for _ in range(max_gen_batches):
        batch = batch_generator()
        total_generated += 1

        if "rewards" not in batch:
            valid_samples.append(batch)
            continue

        # Check variance per prompt
        batch_size = batch["rewards"].shape[0]
        num_prompts = batch_size // num_generations_per_prompt
        shaped_rewards = batch["rewards"].view(num_prompts, num_generations_per_prompt)
        stds = shaped_rewards.std(dim=-1)

        # Find prompts with variance
        valid_mask = stds > std_threshold

        if valid_mask.any():
            # Extract valid prompt samples
            for i in range(num_prompts):
                if valid_mask[i]:
                    start_idx = i * num_generations_per_prompt
                    end_idx = start_idx + num_generations_per_prompt

                    sample = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            if value.shape[0] == batch_size:
                                sample[key] = value[start_idx:end_idx]
                            else:
                                sample[key] = value
                        else:
                            sample[key] = value

                    valid_samples.append(sample)

        # Check if we have enough
        if len(valid_samples) >= num_prompts_needed:
            break

    if len(valid_samples) < num_prompts_needed:
        raise RuntimeError(
            f"Could not generate {num_prompts_needed} valid prompts "
            f"after {total_generated} batches. Got {len(valid_samples)} valid prompts."
        )

    # Combine valid samples
    combined = {}
    for key in valid_samples[0].keys():
        values = [s[key] for s in valid_samples[:num_prompts_needed]]
        if isinstance(values[0], torch.Tensor):
            combined[key] = torch.cat(values, dim=0)
        else:
            combined[key] = values[0]

    return BatchedDataDict(combined)


def prepare_batch_for_training(
    batch: BatchedDataDict,
    num_generations_per_prompt: int,
    use_leave_one_out_baseline: bool = False,
    normalize_rewards: bool = True,
) -> BatchedDataDict:
    """Prepare a batch for GRPO training.

    Applies all necessary transformations:
    - Computes advantages
    - Normalizes if requested
    - Adds any derived fields

    Args:
        batch: Raw batch from rollout.
        num_generations_per_prompt: Number of generations per prompt.
        use_leave_one_out_baseline: Use leave-one-out baseline.
        normalize_rewards: Whether to normalize advantages.

    Returns:
        Batch ready for loss computation.
    """
    from nemo_rl.algorithms.grpo.loss import normalize_advantages_with_epsilon

    return normalize_advantages_with_epsilon(
        batch,
        use_leave_one_out_baseline=use_leave_one_out_baseline,
        num_generations_per_prompt=num_generations_per_prompt,
        should_normalize=normalize_rewards,
    )
