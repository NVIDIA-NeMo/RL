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
"""GRPO configuration classes.

This module contains all configuration TypedDicts for the GRPO algorithm.
"""

from typing import Any, NotRequired, TypedDict

from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.data import DataConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.logger import LoggerConfig


class RewardScalingConfig(TypedDict):
    """Configure linear reward scaling with clamping.

    When `enabled` is True, each reward is clamped to the source interval
    [source_min, source_max] and linearly mapped to the target interval
    [target_min, target_max].

    Defaults:
        source_min=0.0, source_max=1.0, target_min=0.0, target_max=1.0
    """

    enabled: bool
    source_min: NotRequired[float]
    source_max: NotRequired[float]
    target_min: NotRequired[float]
    target_max: NotRequired[float]


class AsyncGRPOConfig(TypedDict):
    """Configuration for asynchronous GRPO training."""

    enabled: bool
    # Maximum trajectory age in training steps for samples drawn from the
    # async replay buffer.
    max_trajectory_age_steps: int
    # Does the weight synchronization as soon as the training is done
    in_flight_weight_updates: NotRequired[bool]
    # Recomputes the KV cache after the in-flight weight updates.
    recompute_kv_cache_after_weight_updates: NotRequired[bool]


class GRPOConfig(TypedDict):
    """Main GRPO training configuration."""

    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_epochs: int
    max_num_steps: int
    max_rollout_turns: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    max_val_samples: int
    seed: int
    async_grpo: NotRequired[AsyncGRPOConfig]
    overlong_filtering: NotRequired[bool]
    use_dynamic_sampling: bool
    dynamic_sampling_max_gen_batches: NotRequired[int]
    batch_multiplier: NotRequired[float]
    reward_shaping: RewardShapingConfig
    reward_scaling: RewardScalingConfig


class GRPOSaveState(TypedDict):
    """State saved during checkpointing."""

    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int
    val_reward: NotRequired[float]


def default_grpo_save_state() -> GRPOSaveState:
    """Create default GRPO save state."""
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_steps": 0,
        "total_valid_tokens": 0,
        "val_reward": -99999999.0,
    }


class GRPOLoggerConfig(LoggerConfig):
    """GRPO-specific logger configuration."""

    num_val_samples_to_print: int


class MasterConfig(TypedDict):
    """Complete GRPO training configuration."""

    policy: PolicyConfig
    loss_fn: dict[str, Any]  # ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
