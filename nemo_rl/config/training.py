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
"""Training configuration for RL algorithms.

This module provides configuration classes for training algorithms
including GRPO, SFT, DPO, and related settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator

from nemo_rl.config.base import BaseConfig
from nemo_rl.config.cluster import ClusterConfig
from nemo_rl.config.generation import VLLMConfig
from nemo_rl.config.policy import PolicyConfig


class LoggerConfig(BaseConfig):
    """Configuration for logging during training.

    Attributes:
        wandb_enabled: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (username or team).
        wandb_run_name: W&B run name.
        tensorboard_enabled: Enable TensorBoard logging.
        tensorboard_dir: Directory for TensorBoard logs.
        log_interval: Log metrics every N steps.
        log_level: Logging level ('debug', 'info', 'warning', 'error').
        num_samples_to_print: Number of samples to print to stdout.
        num_val_samples_to_print: Number of validation samples to print.
    """

    wandb_enabled: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    tensorboard_enabled: bool = True
    tensorboard_dir: str = "logs/tensorboard"
    log_interval: Annotated[int, Field(gt=0)] = 10
    log_level: str = "info"
    num_samples_to_print: Annotated[int, Field(ge=0)] = 3
    num_val_samples_to_print: Annotated[int, Field(ge=0)] = 5


class CheckpointingConfig(BaseConfig):
    """Configuration for checkpoint management.

    Attributes:
        enabled: Whether checkpointing is enabled.
        checkpoint_dir: Directory for saving checkpoints.
        save_period: Save checkpoint every N steps.
        keep_top_k: Number of best checkpoints to keep.
        metric_name: Metric for determining best checkpoint.
        higher_is_better: Whether higher metric values are better.
        model_save_format: Format for saving ('safetensors', 'torch_save').
        save_consolidated: Save consolidated checkpoint for HF compatibility.
        model_cache_dir: Directory for model cache.
        model_repo_id: HuggingFace repository ID for saving.
        is_peft: Whether model uses PEFT/LoRA.
        checkpoint_must_save_by: Time deadline for checkpoint saving.
    """

    enabled: bool = True
    checkpoint_dir: str = "checkpoints"
    save_period: Annotated[int, Field(gt=0)] = 100
    keep_top_k: Annotated[int, Field(gt=0)] | None = 5
    metric_name: str | None = "val:reward"
    higher_is_better: bool = True
    model_save_format: str = "safetensors"
    save_consolidated: bool = False
    model_cache_dir: str = ""
    model_repo_id: str = ""
    is_peft: bool = False
    peft_config: Any = None
    checkpoint_must_save_by: str | None = None


class DataConfig(BaseConfig):
    """Configuration for data loading.

    Attributes:
        train_path: Path to training data.
        val_path: Path to validation data.
        num_workers: Number of data loading workers.
        prefetch_factor: Number of batches to prefetch.
        shuffle: Whether to shuffle training data.
        seed: Random seed for shuffling.
    """

    train_path: str | None = None
    val_path: str | None = None
    num_workers: Annotated[int, Field(ge=0)] = 4
    prefetch_factor: Annotated[int, Field(gt=0)] = 2
    shuffle: bool = True
    seed: int = 42


class RewardShapingConfig(BaseConfig):
    """Configuration for reward shaping.

    Attributes:
        enabled: Whether reward shaping is enabled.
        kl_penalty_coeff: Coefficient for KL penalty.
        entropy_bonus_coeff: Coefficient for entropy bonus.
    """

    enabled: bool = False
    kl_penalty_coeff: Annotated[float, Field(ge=0.0)] = 0.0
    entropy_bonus_coeff: Annotated[float, Field(ge=0.0)] = 0.0


class RewardScalingConfig(BaseConfig):
    """Configuration for reward scaling.

    Linear scaling that maps rewards from source range to target range.

    Attributes:
        enabled: Whether reward scaling is enabled.
        source_min: Source range minimum.
        source_max: Source range maximum.
        target_min: Target range minimum.
        target_max: Target range maximum.
    """

    enabled: bool = False
    source_min: float = 0.0
    source_max: float = 1.0
    target_min: float = 0.0
    target_max: float = 1.0


class ClippedPGLossConfig(BaseConfig):
    """Configuration for clipped policy gradient loss.

    Attributes:
        clip_ratio: PPO clip ratio.
        entropy_coeff: Entropy loss coefficient.
        value_loss_coeff: Value loss coefficient (if using value function).
        kl_coeff: KL divergence coefficient.
    """

    clip_ratio: Annotated[float, Field(gt=0.0)] = 0.2
    entropy_coeff: Annotated[float, Field(ge=0.0)] = 0.0
    value_loss_coeff: Annotated[float, Field(ge=0.0)] = 0.5
    kl_coeff: Annotated[float, Field(ge=0.0)] = 0.0


class AsyncGRPOConfig(BaseConfig):
    """Configuration for async GRPO training.

    Attributes:
        enabled: Whether async GRPO is enabled.
        max_trajectory_age_steps: Maximum age of trajectories in replay buffer.
        in_flight_weight_updates: Sync weights without waiting for generations.
        recompute_kv_cache_after_weight_updates: Recompute KV cache after update.
    """

    enabled: bool = False
    max_trajectory_age_steps: Annotated[int, Field(gt=0)] = 10
    in_flight_weight_updates: bool = False
    recompute_kv_cache_after_weight_updates: bool = False


class GRPOConfig(BaseConfig):
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    GRPO is a reinforcement learning algorithm for language model post-training.

    Attributes:
        policy: Policy model configuration.
        cluster: Cluster resource configuration.
        vllm: vLLM generation configuration.
        logger: Logging configuration.
        checkpointing: Checkpoint configuration.
        loss_fn: Loss function configuration.
        data: Data configuration.
        num_prompts_per_step: Number of prompts per training step.
        num_generations_per_prompt: Generations per prompt for RL.
        max_num_epochs: Maximum training epochs.
        max_num_steps: Maximum training steps.
        max_rollout_turns: Maximum turns for multi-turn rollout.
        normalize_rewards: Whether to normalize rewards.
        use_leave_one_out_baseline: Use leave-one-out baseline.
        use_dynamic_sampling: Enable dynamic sampling.
        dynamic_sampling_max_gen_batches: Max batches for dynamic sampling.
        batch_multiplier: Multiplier for dynamic sampling batch size.
        overlong_filtering: Filter out overlong sequences.
        val_period: Validation every N steps.
        val_batch_size: Validation batch size.
        val_at_start: Run validation at start.
        max_val_samples: Maximum validation samples.
        seed: Random seed.
        reward_shaping: Reward shaping configuration.
        reward_scaling: Reward scaling configuration.
        async_grpo: Async GRPO configuration.
        env: Environment configuration (task-specific).

    Example:
        >>> config = GRPOConfig(
        ...     policy=PolicyConfig(model_name="Qwen/Qwen2.5-1.5B"),
        ...     cluster=ClusterConfig(num_nodes=1, gpus_per_node=8),
        ...     num_prompts_per_step=32,
        ...     num_generations_per_prompt=16,
        ... )
    """

    # Core components
    policy: PolicyConfig
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    loss_fn: ClippedPGLossConfig = Field(default_factory=ClippedPGLossConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    # Training hyperparameters
    num_prompts_per_step: Annotated[int, Field(gt=0)] = 32
    num_generations_per_prompt: Annotated[int, Field(gt=0)] = 16
    max_num_epochs: Annotated[int, Field(gt=0)] = 1
    max_num_steps: Annotated[int, Field(gt=0)] = 10000
    max_rollout_turns: Annotated[int, Field(gt=0)] = 1

    # Reward settings
    normalize_rewards: bool = True
    use_leave_one_out_baseline: bool = True
    reward_shaping: RewardShapingConfig = Field(default_factory=RewardShapingConfig)
    reward_scaling: RewardScalingConfig = Field(default_factory=RewardScalingConfig)

    # Dynamic sampling
    use_dynamic_sampling: bool = False
    dynamic_sampling_max_gen_batches: Annotated[int, Field(gt=0)] = 10
    batch_multiplier: Annotated[float, Field(gt=0.0)] = 2.0
    overlong_filtering: bool = True

    # Validation settings
    val_period: Annotated[int, Field(gt=0)] = 100
    val_batch_size: Annotated[int, Field(gt=0)] = 32
    val_at_start: bool = False
    max_val_samples: Annotated[int, Field(gt=0)] = 1000

    # Misc
    seed: int = 42
    async_grpo: AsyncGRPOConfig = Field(default_factory=AsyncGRPOConfig)
    env: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> "GRPOConfig":
        """Validate that batch sizes are consistent."""
        total_generations = self.num_prompts_per_step * self.num_generations_per_prompt
        if total_generations > 10000:
            # Warning: very large batch size
            pass
        return self

    @classmethod
    def minimal(
        cls,
        model_name: str,
        num_prompts_per_step: int = 32,
        num_generations_per_prompt: int = 16,
        **kwargs,
    ) -> "GRPOConfig":
        """Create a minimal GRPO configuration with sensible defaults.

        This is the recommended way to create a GRPO configuration for
        quick experimentation. Only the model name is required.

        Args:
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen2.5-1.5B").
            num_prompts_per_step: Number of prompts per training step.
            num_generations_per_prompt: Generations per prompt.
            **kwargs: Additional config overrides.

        Returns:
            GRPOConfig with sensible defaults for the model.

        Example:
            >>> config = GRPOConfig.minimal("Qwen/Qwen2.5-1.5B")
            >>> trainer.fit(config)
        """
        # Merge kwargs with defaults
        config_dict = {
            "policy": {"model_name": model_name},
            "cluster": ClusterConfig.auto_detect().model_dump(),
            "num_prompts_per_step": num_prompts_per_step,
            "num_generations_per_prompt": num_generations_per_prompt,
            **kwargs,
        }

        # Handle nested policy config if provided
        if "policy" in kwargs and isinstance(kwargs["policy"], dict):
            config_dict["policy"] = {"model_name": model_name, **kwargs["policy"]}

        return cls.model_validate(config_dict)


class SFTConfig(BaseConfig):
    """Configuration for Supervised Fine-Tuning (SFT).

    Attributes:
        policy: Policy model configuration.
        cluster: Cluster resource configuration.
        logger: Logging configuration.
        checkpointing: Checkpoint configuration.
        data: Data configuration.
        max_num_epochs: Maximum training epochs.
        max_num_steps: Maximum training steps.
        val_period: Validation every N steps.
        val_batch_size: Validation batch size.
        val_at_start: Run validation at start.
        seed: Random seed.
    """

    policy: PolicyConfig
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    max_num_epochs: Annotated[int, Field(gt=0)] = 1
    max_num_steps: Annotated[int, Field(gt=0)] = 10000
    val_period: Annotated[int, Field(gt=0)] = 100
    val_batch_size: Annotated[int, Field(gt=0)] = 32
    val_at_start: bool = False

    seed: int = 42

    @classmethod
    def minimal(
        cls,
        model_name: str,
        train_path: str | None = None,
        **kwargs,
    ) -> "SFTConfig":
        """Create a minimal SFT configuration with sensible defaults.

        Args:
            model_name: HuggingFace model name or path.
            train_path: Path to training data.
            **kwargs: Additional config overrides.

        Returns:
            SFTConfig with sensible defaults.

        Example:
            >>> config = SFTConfig.minimal("Qwen/Qwen2.5-1.5B")
        """
        config_dict = {
            "policy": {"model_name": model_name},
            "cluster": ClusterConfig.auto_detect().model_dump(),
            **kwargs,
        }

        if train_path:
            config_dict.setdefault("data", {})["train_path"] = train_path

        if "policy" in kwargs and isinstance(kwargs["policy"], dict):
            config_dict["policy"] = {"model_name": model_name, **kwargs["policy"]}

        return cls.model_validate(config_dict)


class DPOLossConfig(BaseConfig):
    """Configuration for DPO loss function.

    Attributes:
        beta: DPO beta parameter (inverse temperature).
        label_smoothing: Label smoothing coefficient.
        reference_free: Whether to use reference-free DPO.
    """

    beta: Annotated[float, Field(gt=0.0)] = 0.1
    label_smoothing: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    reference_free: bool = False


class DPOConfig(BaseConfig):
    """Configuration for Direct Preference Optimization (DPO).

    Attributes:
        policy: Policy model configuration.
        cluster: Cluster resource configuration.
        logger: Logging configuration.
        checkpointing: Checkpoint configuration.
        data: Data configuration.
        loss_fn: DPO loss function configuration.
        max_num_epochs: Maximum training epochs.
        max_num_steps: Maximum training steps.
        val_period: Validation every N steps.
        val_batch_size: Validation batch size.
        val_at_start: Run validation at start.
        seed: Random seed.
    """

    policy: PolicyConfig
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    loss_fn: DPOLossConfig = Field(default_factory=DPOLossConfig)

    max_num_epochs: Annotated[int, Field(gt=0)] = 1
    max_num_steps: Annotated[int, Field(gt=0)] = 10000
    val_period: Annotated[int, Field(gt=0)] = 100
    val_batch_size: Annotated[int, Field(gt=0)] = 32
    val_at_start: bool = False

    seed: int = 42

    @classmethod
    def minimal(
        cls,
        model_name: str,
        train_path: str | None = None,
        beta: float = 0.1,
        **kwargs,
    ) -> "DPOConfig":
        """Create a minimal DPO configuration with sensible defaults.

        Args:
            model_name: HuggingFace model name or path.
            train_path: Path to preference data.
            beta: DPO beta parameter (inverse temperature).
            **kwargs: Additional config overrides.

        Returns:
            DPOConfig with sensible defaults.

        Example:
            >>> config = DPOConfig.minimal("Qwen/Qwen2.5-1.5B", beta=0.1)
        """
        config_dict = {
            "policy": {"model_name": model_name},
            "cluster": ClusterConfig.auto_detect().model_dump(),
            "loss_fn": {"beta": beta},
            **kwargs,
        }

        if train_path:
            config_dict.setdefault("data", {})["train_path"] = train_path

        if "policy" in kwargs and isinstance(kwargs["policy"], dict):
            config_dict["policy"] = {"model_name": model_name, **kwargs["policy"]}

        return cls.model_validate(config_dict)
