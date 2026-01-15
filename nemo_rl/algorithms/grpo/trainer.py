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
"""GRPO Trainer implementation.

This module provides the GRPOTrainer class that extends BaseTrainer
for Group Relative Policy Optimization training.

Example:
    >>> from nemo_rl.algorithms.grpo import GRPOTrainer
    >>> 
    >>> trainer = GRPOTrainer(config)
    >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from nemo_rl.trainers.base import BaseTrainer, TrainingResult

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo.config import GRPOConfig, MasterConfig
    from nemo_rl.algorithms.rollout import RolloutEngine
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

logger = logging.getLogger(__name__)


class GRPOTrainer(BaseTrainer):
    """Trainer for Group Relative Policy Optimization (GRPO).

    GRPOTrainer extends BaseTrainer to implement the GRPO algorithm,
    which trains language models using relative preferences within
    groups of generations from the same prompt.

    Key features:
    - Multiple generations per prompt for relative comparison
    - Advantage normalization within prompt groups
    - Leave-one-out baseline computation
    - Support for both sync and async training modes

    Attributes:
        rollout_engine: Engine for generation and reward collection.
        num_generations_per_prompt: Number of responses per prompt.
        normalize_rewards: Whether to normalize advantages.
        use_leave_one_out_baseline: Use leave-one-out for baseline.

    Example:
        >>> # From pretrained model
        >>> trainer = GRPOTrainer.from_pretrained(
        ...     "Qwen/Qwen2.5-1.5B",
        ...     num_prompts_per_step=32,
        ...     num_generations_per_prompt=16,
        ... )
        >>> trainer.fit(
        ...     dataset="nvidia/OpenMathInstruct-2",
        ...     reward_fn=my_reward_function,
        ... )
        >>> 
        >>> # Or from config directly
        >>> trainer = GRPOTrainer(config)
        >>> trainer.fit(
        ...     dataset="nvidia/OpenMathInstruct-2",
        ...     callbacks=[CheckpointCallback(every_n_steps=100)],
        ... )
    """

    @classmethod
    def _build_config_from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build GRPO-specific configuration from a pretrained model.

        Args:
            model_name_or_path: Model identifier or path.
            **kwargs: Configuration overrides. GRPO-specific options:
                - num_prompts_per_step: Prompts per training step (default: 32)
                - num_generations_per_prompt: Generations per prompt (default: 16)
                - normalize_rewards: Whether to normalize rewards (default: True)
                - use_leave_one_out_baseline: LOO baseline (default: True)
                - max_steps: Maximum training steps (default: 1000)
                - max_epochs: Maximum epochs (default: 1)

        Returns:
            GRPO configuration dictionary.
        """
        # Get base config
        config = super()._build_config_from_pretrained(model_name_or_path, **kwargs)

        # Extract GRPO-specific parameters
        num_prompts_per_step = kwargs.pop("num_prompts_per_step", 32)
        num_generations_per_prompt = kwargs.pop("num_generations_per_prompt", 16)
        max_steps = kwargs.pop("max_steps", 1000)
        max_epochs = kwargs.pop("max_epochs", 1)

        # Add GRPO config
        config["grpo"] = {
            "num_prompts_per_step": num_prompts_per_step,
            "num_generations_per_prompt": num_generations_per_prompt,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            "normalize_rewards": kwargs.pop("normalize_rewards", True),
            "use_leave_one_out_baseline": kwargs.pop("use_leave_one_out_baseline", True),
            "use_dynamic_sampling": kwargs.pop("use_dynamic_sampling", False),
        }

        return config

    def __init__(self, config: "MasterConfig"):
        """Initialize the GRPO trainer.

        Args:
            config: Full GRPO configuration (MasterConfig).
        """
        super().__init__(config)

        # Extract GRPO-specific config
        self._grpo_config: "GRPOConfig" = config.get("grpo", {})

        # GRPO parameters
        self.num_prompts_per_step = self._grpo_config.get("num_prompts_per_step", 32)
        self.num_generations_per_prompt = self._grpo_config.get(
            "num_generations_per_prompt", 16
        )
        self.normalize_rewards = self._grpo_config.get("normalize_rewards", True)
        self.use_leave_one_out_baseline = self._grpo_config.get(
            "use_leave_one_out_baseline", True
        )
        self.use_dynamic_sampling = self._grpo_config.get("use_dynamic_sampling", False)

        # Components initialized in setup
        self._rollout_engine: Optional["RolloutEngine"] = None
        self._loss_fn = None
        self._policy = None
        self._generation = None

    def _train_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single GRPO training step.

        The GRPO training step:
        1. Generate responses for prompts
        2. Collect rewards from environment
        3. Compute advantages within prompt groups
        4. Compute policy gradient loss
        5. Update model weights

        Args:
            batch: Batch of prompts from dataloader.

        Returns:
            Dictionary with loss, reward, and other metrics.
        """
        from nemo_rl.algorithms.grpo.data import prepare_batch_for_training

        # Generate responses and collect rewards
        if self._rollout_engine is not None:
            rollout_result = self._rollout_engine.rollout(batch)
            batch = rollout_result.responses
            batch["rewards"] = rollout_result.rewards

        # Prepare batch (compute advantages)
        batch = prepare_batch_for_training(
            batch,
            num_generations_per_prompt=self.num_generations_per_prompt,
            use_leave_one_out_baseline=self.use_leave_one_out_baseline,
            normalize_rewards=self.normalize_rewards,
        )

        # Compute loss
        if self._loss_fn is not None:
            loss, loss_metrics = self._loss_fn(batch)
        else:
            loss = batch.get("loss", 0.0)
            loss_metrics = {}

        # Collect metrics
        metrics = {
            "loss": loss.item() if hasattr(loss, "item") else loss,
            "reward_mean": batch["rewards"].mean().item()
            if "rewards" in batch
            else 0.0,
            "reward_std": batch["rewards"].std().item() if "rewards" in batch else 0.0,
            **loss_metrics,
        }

        return metrics

    def _compute_loss(self, batch: Any, outputs: Any) -> Any:
        """Compute the GRPO policy gradient loss.

        Args:
            batch: Input batch with advantages.
            outputs: Model outputs (logprobs).

        Returns:
            Loss tensor.
        """
        from nemo_rl.algorithms.grpo.loss import compute_grpo_loss

        if self._loss_fn is not None:
            loss, _ = compute_grpo_loss(batch, self._loss_fn)
            return loss

        return outputs.get("loss", 0.0)

    def _validate_step(self, batch: Any) -> dict[str, Any]:
        """Perform a validation step.

        Generates responses and computes rewards without training.

        Args:
            batch: Batch of validation prompts.

        Returns:
            Dictionary with validation metrics.
        """
        if self._rollout_engine is not None:
            rollout_result = self._rollout_engine.rollout(
                batch, collect_rewards=True
            )
            rewards = rollout_result.rewards

            return {
                "val_reward_mean": rewards.mean().item() if rewards is not None else 0.0,
                "val_reward_std": rewards.std().item() if rewards is not None else 0.0,
            }

        return {"val_reward_mean": 0.0, "val_reward_std": 0.0}

    def _setup_model(self) -> None:
        """Set up the policy model and generation backend."""
        # Note: Full implementation would initialize Policy, Generation, etc.
        # This is a skeleton showing the architecture direction
        logger.info("Setting up GRPO model components")

    def _setup_optimizer(self) -> None:
        """Set up the optimizer and learning rate scheduler."""
        # Note: Full implementation would create optimizer from config
        logger.info("Setting up GRPO optimizer")

    def _on_train_begin(self) -> None:
        """Called before training starts."""
        super()._on_train_begin()

        if self._logger:
            self._logger.info(
                f"GRPO training starting: "
                f"{self.num_prompts_per_step} prompts x "
                f"{self.num_generations_per_prompt} generations"
            )

    def _on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Called after each epoch.

        Args:
            epoch: Current epoch number.
            metrics: Epoch metrics.
        """
        super()._on_epoch_end(epoch, metrics)

        if self._logger:
            self._logger.info(
                f"Epoch {epoch + 1} complete - "
                f"Loss: {metrics.get('loss', 0):.4f}, "
                f"Reward: {metrics.get('reward_mean', 0):.4f}"
            )

    def _get_max_epochs(self) -> int:
        """Get max epochs from GRPO config."""
        return self._grpo_config.get("max_num_epochs", 1)

    def _get_max_steps(self) -> int:
        """Get max steps from GRPO config."""
        return self._grpo_config.get("max_num_steps", 10000)

    def _get_val_period(self) -> int:
        """Get validation period from GRPO config."""
        return self._grpo_config.get("val_period", 100)

    @property
    def effective_batch_size(self) -> int:
        """Get the effective batch size (prompts * generations)."""
        return self.num_prompts_per_step * self.num_generations_per_prompt
