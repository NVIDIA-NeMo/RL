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
"""SFT Trainer implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from nemo_rl.trainers.base import BaseTrainer

if TYPE_CHECKING:
    from nemo_rl.algorithms.sft.config import MasterConfig, SFTConfig

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """Trainer for Supervised Fine-Tuning (SFT).

    Extends BaseTrainer for supervised fine-tuning using NLL loss.

    Example:
        >>> # From pretrained model
        >>> trainer = SFTTrainer.from_pretrained(
        ...     "Qwen/Qwen2.5-1.5B",
        ...     batch_size=16,
        ...     learning_rate=2e-5,
        ... )
        >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")
        >>> 
        >>> # Or from config
        >>> trainer = SFTTrainer(config)
        >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")
    """

    @classmethod
    def _build_config_from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build SFT-specific configuration from a pretrained model.

        Args:
            model_name_or_path: Model identifier or path.
            **kwargs: Configuration overrides. SFT-specific options:
                - batch_size: Training batch size (default: 32)
                - max_steps: Maximum training steps (default: 1000)
                - max_epochs: Maximum epochs (default: 1)
                - val_period: Validation frequency (default: 100)

        Returns:
            SFT configuration dictionary.
        """
        # Get base config
        config = super()._build_config_from_pretrained(model_name_or_path, **kwargs)

        # Extract SFT-specific parameters
        batch_size = kwargs.pop("batch_size", 32)
        max_steps = kwargs.pop("max_steps", 1000)
        max_epochs = kwargs.pop("max_epochs", 1)
        val_period = kwargs.pop("val_period", 100)

        # Add SFT config
        config["sft"] = {
            "batch_size": batch_size,
            "max_num_steps": max_steps,
            "max_num_epochs": max_epochs,
            "val_period": val_period,
            "val_batches": kwargs.pop("val_batches", -1),
            "val_at_start": kwargs.pop("val_at_start", False),
        }

        return config

    def __init__(self, config: "MasterConfig"):
        super().__init__(config)
        self._sft_config: "SFTConfig" = config.get("sft", {})
        self.val_period = self._sft_config.get("val_period", 100)
        self.val_batches = self._sft_config.get("val_batches", -1)
        self.val_at_start = self._sft_config.get("val_at_start", False)
        self._loss_fn = None
        self._policy = None
        self._tokenizer = None

    def _prepare_batch(self, batch: Any) -> Any:
        """Prepare batch with loss masks if needed."""
        if "input_ids" not in batch and "message_log" in batch:
            from nemo_rl.algorithms.sft.data import prepare_batch_for_sft
            return prepare_batch_for_sft(
                batch, self._tokenizer,
                make_sequence_length_divisible_by=self.config.get("policy", {}).get("make_sequence_length_divisible_by", 1),
            )
        return batch

    def _train_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single SFT training step."""
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
        """Compute the SFT NLL loss."""
        from nemo_rl.algorithms.sft.loss import create_sft_loss_function
        if self._loss_fn is None:
            self._loss_fn = create_sft_loss_function()
        return outputs.get("loss", 0.0)

    def _validate_step(self, batch: Any) -> dict[str, Any]:
        """Perform a validation step."""
        batch = self._prepare_batch(batch)
        if self._policy is not None and self._loss_fn is not None:
            results = self._policy.train(
                batch, self._loss_fn, eval_mode=True, gbs=batch.size,
                mbs=self._sft_config.get("val_micro_batch_size", 1)
            )
            loss = results.get("loss", 0.0)
            return {"val_loss": loss.item() if hasattr(loss, "item") else loss}
        return {"val_loss": 0.0}

    def _setup_model(self) -> None:
        logger.info("Setting up SFT model components")

    def _setup_optimizer(self) -> None:
        logger.info("Setting up SFT optimizer")

    def _on_train_begin(self) -> None:
        super()._on_train_begin()
        if self._logger:
            self._logger.info("SFT training starting")

    def _on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        super()._on_epoch_end(epoch, metrics)
        if self._logger:
            self._logger.info(f"Epoch {epoch + 1} - Loss: {metrics.get('loss', 0):.4f}")

    def _get_max_epochs(self) -> int:
        return self._sft_config.get("max_num_epochs", 1)

    def _get_max_steps(self) -> int:
        return self._sft_config.get("max_num_steps", 10000)

    def _get_val_period(self) -> int:
        return self._sft_config.get("val_period", 100)
