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
"""Unified validation runner for all NeMo RL algorithms.

This module provides the ValidationRunner class that handles validation
in a consistent way across GRPO, SFT, and DPO algorithms.

Example:
    >>> from nemo_rl.trainers import ValidationRunner
    >>> 
    >>> runner = ValidationRunner(
    ...     metrics=["loss", "perplexity"],
    ...     frequency=100,
    ...     mode="steps"
    ... )
    >>> 
    >>> if runner.should_validate(step=100, epoch=0):
    ...     results = runner.run(trainer, dataloader)
    ...     print(results)  # {"loss": 0.5, "perplexity": 1.65}
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Protocol

if TYPE_CHECKING:
    from nemo_rl.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation.

    Attributes:
        frequency: How often to validate (in steps or epochs).
        mode: Validation mode - "steps" or "epochs".
        metrics: List of metric names to compute.
        max_batches: Maximum batches to evaluate (-1 for all).
        log_prefix: Prefix for logged metric names.
    """

    frequency: int = 100
    mode: str = "steps"
    metrics: list[str] = field(default_factory=lambda: ["loss"])
    max_batches: int = -1
    log_prefix: str = "val"


@dataclass
class ValidationResult:
    """Result of a validation run.

    Attributes:
        metrics: Dictionary of computed metrics.
        num_samples: Total samples evaluated.
        num_batches: Total batches processed.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    num_batches: int = 0

    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"ValidationResult({metrics_str}, samples={self.num_samples})"


class MetricComputer(Protocol):
    """Protocol for custom metric computation."""

    def __call__(
        self, batch: Any, outputs: dict[str, Any]
    ) -> dict[str, float]: ...


class ValidationRunner:
    """Unified validation runner for all algorithms.

    Provides consistent validation logic with configurable frequency,
    metrics, and logging format. Can be used with GRPO, SFT, and DPO.

    Example:
        >>> runner = ValidationRunner(
        ...     metrics=["loss", "perplexity"],
        ...     frequency=100,
        ...     mode="steps"
        ... )
        >>> 
        >>> # Check if validation is due
        >>> if runner.should_validate(step=100, epoch=0):
        ...     results = runner.run(trainer, val_dataloader)
        ...     print(f"Validation loss: {results.metrics['loss']:.4f}")

    Attributes:
        config: Validation configuration.
        custom_metrics: Custom metric computation functions.
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        frequency: int = 100,
        mode: str = "steps",
        max_batches: int = -1,
        log_prefix: str = "val",
    ):
        """Initialize ValidationRunner.

        Args:
            metrics: List of metric names to track. Default: ["loss"].
            frequency: Validation frequency in steps or epochs.
            mode: "steps" or "epochs" for frequency interpretation.
            max_batches: Max batches per validation (-1 for all).
            log_prefix: Prefix for metric names in logs.
        """
        self.config = ValidationConfig(
            frequency=frequency,
            mode=mode,
            metrics=metrics or ["loss"],
            max_batches=max_batches,
            log_prefix=log_prefix,
        )
        self._custom_metrics: dict[str, MetricComputer] = {}
        self._last_val_step: int = -1
        self._last_val_epoch: int = -1

    def register_metric(self, name: str, compute_fn: MetricComputer) -> None:
        """Register a custom metric computation function.

        Args:
            name: Metric name (used in results dict).
            compute_fn: Function(batch, outputs) -> dict[str, float].
        """
        self._custom_metrics[name] = compute_fn

    def should_validate(self, step: int, epoch: int) -> bool:
        """Check if validation should run at this point.

        Args:
            step: Current global step.
            epoch: Current epoch number.

        Returns:
            True if validation should run, False otherwise.
        """
        if self.config.frequency <= 0:
            return False

        if self.config.mode == "steps":
            if step > 0 and step % self.config.frequency == 0:
                if step != self._last_val_step:
                    self._last_val_step = step
                    return True
        elif self.config.mode == "epochs":
            if epoch >= 0 and epoch % self.config.frequency == 0:
                if epoch != self._last_val_epoch:
                    self._last_val_epoch = epoch
                    return True

        return False

    def run(
        self,
        trainer: "BaseTrainer",
        dataloader: Iterator,
        step_fn: Optional[Callable[[Any], dict[str, Any]]] = None,
    ) -> ValidationResult:
        """Run validation and return metrics.

        Args:
            trainer: BaseTrainer instance with _validate_step method.
            dataloader: Validation data iterator.
            step_fn: Optional custom step function. If None, uses
                trainer._validate_step.

        Returns:
            ValidationResult with aggregated metrics.
        """
        validate_fn = step_fn or trainer._validate_step

        accumulated: dict[str, float] = {m: 0.0 for m in self.config.metrics}
        num_samples = 0
        num_batches = 0

        logger.info(f"Starting validation (max_batches={self.config.max_batches})")

        for batch in dataloader:
            if self.config.max_batches > 0 and num_batches >= self.config.max_batches:
                break

            # Run validation step
            outputs = validate_fn(batch)

            # Accumulate standard metrics
            for metric in self.config.metrics:
                if metric in outputs:
                    value = outputs[metric]
                    accumulated[metric] += value if isinstance(value, (int, float)) else value.item()

            # Run custom metrics
            for name, compute_fn in self._custom_metrics.items():
                custom_results = compute_fn(batch, outputs)
                for k, v in custom_results.items():
                    key = f"{name}_{k}" if k != name else k
                    if key not in accumulated:
                        accumulated[key] = 0.0
                    accumulated[key] += v

            batch_size = getattr(batch, "size", 1)
            num_samples += batch_size
            num_batches += 1

        # Average metrics
        metrics = {}
        if num_batches > 0:
            for key, value in accumulated.items():
                metrics[f"{self.config.log_prefix}_{key}"] = value / num_batches

            # Compute perplexity if loss is available
            if "loss" in self.config.metrics and f"{self.config.log_prefix}_loss" in metrics:
                loss = metrics[f"{self.config.log_prefix}_loss"]
                metrics[f"{self.config.log_prefix}_perplexity"] = math.exp(min(loss, 20.0))

        result = ValidationResult(
            metrics=metrics,
            num_samples=num_samples,
            num_batches=num_batches,
        )

        self._log_results(result)
        return result

    def _log_results(self, result: ValidationResult) -> None:
        """Log validation results in consistent format.

        Args:
            result: ValidationResult to log.
        """
        if not result.metrics:
            logger.warning("Validation completed with no metrics")
            return

        # Format metrics for logging
        parts = []
        for key, value in sorted(result.metrics.items()):
            parts.append(f"{key}={value:.4f}")

        logger.info(
            f"Validation: {', '.join(parts)} "
            f"(samples={result.num_samples}, batches={result.num_batches})"
        )

    def reset(self) -> None:
        """Reset validation state (for new training run)."""
        self._last_val_step = -1
        self._last_val_epoch = -1


def create_validation_runner(
    config: dict[str, Any] | None = None,
    frequency: int = 100,
    mode: str = "steps",
    metrics: list[str] | None = None,
) -> ValidationRunner:
    """Factory function to create a ValidationRunner.

    Args:
        config: Configuration dict with validation settings.
        frequency: Default frequency if not in config.
        mode: Default mode if not in config.
        metrics: Default metrics if not in config.

    Returns:
        Configured ValidationRunner instance.
    """
    if config is not None:
        frequency = config.get("val_period", config.get("frequency", frequency))
        mode = config.get("val_mode", mode)
        metrics = config.get("val_metrics", metrics)
        max_batches = config.get("val_batches", -1)
    else:
        max_batches = -1

    return ValidationRunner(
        metrics=metrics,
        frequency=frequency,
        mode=mode,
        max_batches=max_batches,
    )
