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
"""DPO Trainer implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from nemo_rl.trainers.base import BaseTrainer

if TYPE_CHECKING:
    from nemo_rl.algorithms.dpo.config import DPOConfig, MasterConfig

logger = logging.getLogger(__name__)


class DPOTrainer(BaseTrainer):
    """Trainer for Direct Preference Optimization (DPO).

    Extends BaseTrainer for preference-based fine-tuning using DPO loss.
    Trains a policy to prefer chosen responses over rejected responses
    using a reference policy for KL regularization.

    Example:
        >>> # From pretrained model
        >>> trainer = DPOTrainer.from_pretrained(
        ...     "Qwen/Qwen2.5-1.5B",
        ...     beta=0.1,
        ...     learning_rate=5e-7,
        ... )
        >>> trainer.fit(dataset="Anthropic/hh-rlhf")
        >>> 
        >>> # Or from config
        >>> trainer = DPOTrainer(config)
        >>> trainer.fit(dataset="Anthropic/hh-rlhf")
    """

    @classmethod
    def _build_config_from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build DPO-specific configuration from a pretrained model.

        Args:
            model_name_or_path: Model identifier or path.
            **kwargs: Configuration overrides. DPO-specific options:
                - beta: KL penalty coefficient (default: 0.1)
                - batch_size: Training batch size (default: 8)
                - max_steps: Maximum training steps (default: 1000)
                - max_epochs: Maximum epochs (default: 1)
                - reference_free: Skip reference model (default: False)

        Returns:
            DPO configuration dictionary.
        """
        # Get base config
        config = super()._build_config_from_pretrained(model_name_or_path, **kwargs)

        # Extract DPO-specific parameters
        batch_size = kwargs.pop("batch_size", 8)
        max_steps = kwargs.pop("max_steps", 1000)
        max_epochs = kwargs.pop("max_epochs", 1)
        beta = kwargs.pop("beta", 0.1)

        # Add DPO config
        config["dpo"] = {
            "batch_size": batch_size,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            "beta": beta,
            "val_period": kwargs.pop("val_period", 100),
            "val_batches": kwargs.pop("val_batches", -1),
            "val_at_start": kwargs.pop("val_at_start", False),
            "reference_free": kwargs.pop("reference_free", False),
        }

        return config

    def __init__(self, config: "MasterConfig"):
        """Initialize DPOTrainer with configuration."""
        super().__init__(config)
        self._dpo_config: "DPOConfig" = config.get("dpo", {})
        self.val_period = self._dpo_config.get("val_period", 100)
        self.val_batches = self._dpo_config.get("val_batches", -1)
        self.val_at_start = self._dpo_config.get("val_at_start", False)
        self._loss_fn = None
        self._policy = None
        self._tokenizer = None

    def _prepare_batch(self, batch: Any, is_val: bool = False) -> Any:
        """Prepare preference batch with reference policy logprobs."""
        if self._policy is not None and "reference_policy_logprobs" not in batch:
            from nemo_rl.algorithms.dpo.data import add_ref_logprobs_to_batch
            return add_ref_logprobs_to_batch(batch, self._policy, self.config, is_val)
        return batch

    def _train_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single DPO training step."""
        batch = self._prepare_batch(batch)
        if self._policy is not None and self._loss_fn is not None:
            results = self._policy.train(batch, self._loss_fn)
            loss = results.get("loss", 0.0)
            grad_norm = results.get("grad_norm", 0.0)
            metrics = {
                "loss": loss.item() if hasattr(loss, "item") else loss,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            }
            if "all_mb_metrics" in results:
                for k, v in results["all_mb_metrics"].items():
                    if k not in metrics:
                        metrics[k] = sum(v) if isinstance(v, list) else v
            return metrics
        return {"loss": 0.0}

    def _compute_loss(self, batch: Any, outputs: Any) -> Any:
        """Compute the DPO loss."""
        from nemo_rl.algorithms.dpo.loss import create_dpo_loss_function
        if self._loss_fn is None:
            self._loss_fn = create_dpo_loss_function(self._dpo_config)
        return outputs.get("loss", 0.0)

    def _validate_step(self, batch: Any) -> dict[str, Any]:
        """Perform a validation step."""
        batch = self._prepare_batch(batch, is_val=True)
        if self._policy is not None and self._loss_fn is not None:
            results = self._policy.train(
                batch, self._loss_fn, eval_mode=True, gbs=batch.size,
                mbs=self._dpo_config.get("val_micro_batch_size", 1) * 2
            )
            loss = results.get("loss", 0.0)
            metrics = {"val_loss": loss.item() if hasattr(loss, "item") else loss}
            if "all_mb_metrics" in results:
                for k, v in results["all_mb_metrics"].items():
                    val_key = f"val_{k}" if not k.startswith("val_") else k
                    metrics[val_key] = sum(v) if isinstance(v, list) else v
            return metrics
        return {"val_loss": 0.0}

    def _setup_model(self) -> None:
        logger.info("Setting up DPO model components")

    def _setup_optimizer(self) -> None:
        logger.info("Setting up DPO optimizer")

    def _on_train_begin(self) -> None:
        super()._on_train_begin()
        if self._logger:
            self._logger.info("DPO training starting")

    def _on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        super()._on_epoch_end(epoch, metrics)
        if self._logger:
            self._logger.info(f"Epoch {epoch + 1} - Loss: {metrics.get('loss', 0):.4f}")

    def _get_max_epochs(self) -> int:
        return self._dpo_config.get("max_num_epochs", 1)

    def _get_max_steps(self) -> int:
        return self._dpo_config.get("max_num_steps", 10000)

    def _get_val_period(self) -> int:
        return self._dpo_config.get("val_period", 100)
