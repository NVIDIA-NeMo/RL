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
"""Callback system for trainer extensibility.

This module provides a callback system that allows users to hook into
the training lifecycle at various points. Callbacks can be used for:
- Checkpointing
- Logging
- Early stopping
- Custom metrics
- Any custom behavior

Example:
    >>> from nemo_rl.trainers.callbacks import Callback, EarlyStoppingCallback
    >>> 
    >>> class MyCallback(Callback):
    ...     def on_epoch_end(self, trainer, epoch, logs):
    ...         print(f"Epoch {epoch} finished with loss {logs.get('loss')}")
    >>> 
    >>> trainer.fit(
    ...     dataset="my-dataset",
    ...     callbacks=[
    ...         MyCallback(),
    ...         EarlyStoppingCallback(monitor='loss', patience=3),
    ...     ]
    ... )
"""

from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for all callbacks.
    
    Callbacks provide hooks into the training lifecycle. Subclass this
    to implement custom behavior at various training stages.
    
    All hook methods receive the trainer instance and relevant context.
    Override the methods you need; all methods have no-op defaults.
    
    Lifecycle Order:
        1. on_train_begin
        2. For each epoch:
            a. on_epoch_begin
            b. For each step:
                i. on_step_begin
                ii. on_step_end
            c. on_epoch_end
        3. on_train_end
    
    Example:
        >>> class MetricsCallback(Callback):
        ...     def __init__(self):
        ...         self.losses = []
        ...     
        ...     def on_step_end(self, trainer, step, logs):
        ...         self.losses.append(logs.get('loss'))
    """
    
    def on_train_begin(self, trainer: "BaseTrainer") -> None:
        """Called at the start of training.
        
        Args:
            trainer: The trainer instance.
        """
        pass
    
    def on_train_end(self, trainer: "BaseTrainer") -> None:
        """Called at the end of training.
        
        Args:
            trainer: The trainer instance.
        """
        pass
    
    def on_epoch_begin(self, trainer: "BaseTrainer", epoch: int) -> None:
        """Called at the start of each epoch.
        
        Args:
            trainer: The trainer instance.
            epoch: Current epoch number (0-indexed).
        """
        pass
    
    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, logs: dict[str, Any]
    ) -> None:
        """Called at the end of each epoch.
        
        Args:
            trainer: The trainer instance.
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing epoch metrics (loss, etc.).
        """
        pass
    
    def on_step_begin(self, trainer: "BaseTrainer", step: int) -> None:
        """Called at the start of each training step.
        
        Args:
            trainer: The trainer instance.
            step: Current global step number.
        """
        pass
    
    def on_step_end(
        self, trainer: "BaseTrainer", step: int, logs: dict[str, Any]
    ) -> None:
        """Called at the end of each training step.
        
        Args:
            trainer: The trainer instance.
            step: Current global step number.
            logs: Dictionary containing step metrics.
        """
        pass


class CallbackList:
    """Container for managing multiple callbacks.
    
    Automatically invokes all callback hooks in order.
    
    Example:
        >>> callbacks = CallbackList([
        ...     LoggingCallback(log_every=10),
        ...     CheckpointCallback(every_n_steps=100),
        ... ])
        >>> callbacks.on_train_begin(trainer)
    """
    
    def __init__(self, callbacks: list[Callback] | None = None):
        """Initialize the callback list.
        
        Args:
            callbacks: List of callback instances.
        """
        self.callbacks: list[Callback] = list(callbacks) if callbacks else []
    
    def append(self, callback: Callback) -> None:
        """Add a callback to the list.
        
        Args:
            callback: Callback to add.
        """
        self.callbacks.append(callback)
    
    def extend(self, callbacks: list[Callback]) -> None:
        """Add multiple callbacks to the list.
        
        Args:
            callbacks: List of callbacks to add.
        """
        self.callbacks.extend(callbacks)
    
    def on_train_begin(self, trainer: "BaseTrainer") -> None:
        """Call on_train_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: "BaseTrainer") -> None:
        """Call on_train_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: "BaseTrainer", epoch: int) -> None:
        """Call on_epoch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, logs: dict[str, Any]
    ) -> None:
        """Call on_epoch_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)
    
    def on_step_begin(self, trainer: "BaseTrainer", step: int) -> None:
        """Call on_step_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_step_begin(trainer, step)
    
    def on_step_end(
        self, trainer: "BaseTrainer", step: int, logs: dict[str, Any]
    ) -> None:
        """Call on_step_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_step_end(trainer, step, logs)
    
    def __len__(self) -> int:
        return len(self.callbacks)
    
    def __iter__(self):
        return iter(self.callbacks)


# =========================================================================
# Built-in Callbacks
# =========================================================================


class CheckpointCallback(Callback):
    """Callback for automatic checkpointing.
    
    Saves model checkpoints at regular intervals based on steps or epochs.
    
    Args:
        every_n_steps: Save checkpoint every N steps. If 0, disabled.
        every_n_epochs: Save checkpoint every N epochs. If 0, disabled.
        checkpoint_dir: Directory to save checkpoints. If None, uses
            trainer's checkpoint directory.
        save_best_only: If True, only save when metric improves.
        monitor: Metric to monitor for save_best_only (default: 'loss').
        mode: 'min' or 'max' - whether to minimize or maximize the metric.
    
    Example:
        >>> callback = CheckpointCallback(
        ...     every_n_steps=100,
        ...     save_best_only=True,
        ...     monitor='val_loss',
        ... )
        >>> trainer.fit(dataset, callbacks=[callback])
    """
    
    def __init__(
        self,
        every_n_steps: int = 0,
        every_n_epochs: int = 1,
        checkpoint_dir: str | Path | None = None,
        save_best_only: bool = False,
        monitor: str = "loss",
        mode: str = "min",
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        # Tracking state
        self.best_value: float | None = None
        self._last_epoch_saved = -1
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.best_value is None:
            return True
        
        if self.mode == "min":
            return current < self.best_value
        else:
            return current > self.best_value
    
    def _save_checkpoint(self, trainer: "BaseTrainer", step: int) -> None:
        """Save a checkpoint."""
        if trainer.checkpoint_manager is None:
            logger.warning("Checkpoint manager not available, skipping save")
            return
        
        # Get the directory
        checkpoint_dir = self.checkpoint_dir or trainer.checkpoint_manager.checkpoint_dir
        
        logger.info(f"Saving checkpoint at step {step} to {checkpoint_dir}")
        trainer._save_checkpoint()
    
    def on_step_end(
        self, trainer: "BaseTrainer", step: int, logs: dict[str, Any]
    ) -> None:
        """Save checkpoint if step interval reached."""
        if self.every_n_steps <= 0:
            return
        
        if step > 0 and step % self.every_n_steps == 0:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is not None and self._is_improvement(current):
                    self.best_value = current
                    self._save_checkpoint(trainer, step)
            else:
                self._save_checkpoint(trainer, step)
    
    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, logs: dict[str, Any]
    ) -> None:
        """Save checkpoint if epoch interval reached."""
        if self.every_n_epochs <= 0:
            return
        
        if (epoch + 1) % self.every_n_epochs == 0 and epoch > self._last_epoch_saved:
            self._last_epoch_saved = epoch
            
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is not None and self._is_improvement(current):
                    self.best_value = current
                    self._save_checkpoint(trainer, trainer.global_step)
            else:
                self._save_checkpoint(trainer, trainer.global_step)


class LoggingCallback(Callback):
    """Callback for metrics logging.
    
    Logs training metrics at regular intervals to console and/or
    external logging services (WandB, TensorBoard).
    
    Args:
        log_every: Log every N steps.
        log_to_console: Whether to print metrics to console.
        log_to_wandb: Whether to log to WandB (if enabled in trainer).
        log_to_tensorboard: Whether to log to TensorBoard (if enabled).
        metrics: List of metric names to log. If None, logs all.
    
    Example:
        >>> callback = LoggingCallback(
        ...     log_every=10,
        ...     metrics=['loss', 'learning_rate', 'reward'],
        ... )
        >>> trainer.fit(dataset, callbacks=[callback])
    """
    
    def __init__(
        self,
        log_every: int = 10,
        log_to_console: bool = True,
        log_to_wandb: bool = True,
        log_to_tensorboard: bool = True,
        metrics: list[str] | None = None,
    ):
        super().__init__()
        self.log_every = log_every
        self.log_to_console = log_to_console
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard
        self.metrics = metrics
    
    def _filter_metrics(self, logs: dict[str, Any]) -> dict[str, Any]:
        """Filter metrics to log based on self.metrics."""
        if self.metrics is None:
            return logs
        return {k: v for k, v in logs.items() if k in self.metrics}
    
    def _format_metrics(self, logs: dict[str, Any]) -> str:
        """Format metrics for console output."""
        parts = []
        for key, value in sorted(logs.items()):
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
        return " | ".join(parts)
    
    def on_train_begin(self, trainer: "BaseTrainer") -> None:
        """Log training start."""
        if self.log_to_console:
            logger.info("Training started")
    
    def on_train_end(self, trainer: "BaseTrainer") -> None:
        """Log training end."""
        if self.log_to_console:
            logger.info(
                f"Training completed - "
                f"Total steps: {trainer.state.total_steps}"
            )
    
    def on_epoch_begin(self, trainer: "BaseTrainer", epoch: int) -> None:
        """Log epoch start."""
        if self.log_to_console:
            logger.info(f"Epoch {epoch + 1} started")
    
    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, logs: dict[str, Any]
    ) -> None:
        """Log epoch end with metrics."""
        filtered = self._filter_metrics(logs)
        
        if self.log_to_console and filtered:
            logger.info(
                f"Epoch {epoch + 1} ended - {self._format_metrics(filtered)}"
            )
    
    def on_step_end(
        self, trainer: "BaseTrainer", step: int, logs: dict[str, Any]
    ) -> None:
        """Log step metrics."""
        if step % self.log_every != 0:
            return
        
        filtered = self._filter_metrics(logs)
        
        if self.log_to_console and filtered:
            logger.info(f"Step {step} - {self._format_metrics(filtered)}")
        
        # Log to trainer's logger if available
        if trainer.logger is not None:
            trainer.logger.log_metrics(filtered, step=step)


class EarlyStoppingCallback(Callback):
    """Callback for early stopping based on a monitored metric.
    
    Stops training when the monitored metric stops improving for a
    specified number of epochs (patience).
    
    Args:
        monitor: Metric name to monitor.
        patience: Number of epochs with no improvement to wait.
        mode: 'min' or 'max' - whether to minimize or maximize the metric.
        min_delta: Minimum change to qualify as an improvement.
        restore_best_weights: Whether to restore best weights when stopping.
    
    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor='val_loss',
        ...     patience=3,
        ...     mode='min',
        ... )
        >>> trainer.fit(dataset, callbacks=[callback])
    """
    
    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        # Tracking state
        self.best_value: float | None = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.best_value is None:
            return True
        
        if self.mode == "min":
            return current < (self.best_value - self.min_delta)
        else:
            return current > (self.best_value + self.min_delta)
    
    def on_train_begin(self, trainer: "BaseTrainer") -> None:
        """Reset state at training start."""
        self.best_value = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, logs: dict[str, Any]
    ) -> None:
        """Check for improvement and trigger early stopping."""
        current = logs.get(self.monitor)
        
        if current is None:
            logger.warning(
                f"EarlyStoppingCallback: metric '{self.monitor}' not found "
                f"in logs. Available metrics: {list(logs.keys())}"
            )
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.state.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best {self.monitor}: {self.best_value:.4f} at epoch "
                    f"{self.best_epoch + 1}"
                )


class ProgressCallback(Callback):
    """Callback for progress bar display.
    
    Displays a progress bar during training using tqdm (if available).
    Falls back to simple console output if tqdm is not installed.
    
    Args:
        show_epoch_progress: Show progress within each epoch.
        show_metrics: List of metrics to display in progress bar.
    
    Example:
        >>> callback = ProgressCallback(show_metrics=['loss', 'reward'])
        >>> trainer.fit(dataset, callbacks=[callback])
    """
    
    def __init__(
        self,
        show_epoch_progress: bool = True,
        show_metrics: list[str] | None = None,
    ):
        super().__init__()
        self.show_epoch_progress = show_epoch_progress
        self.show_metrics = show_metrics or ["loss"]
        self._pbar = None
        self._total_steps: int = 0
    
    def on_train_begin(self, trainer: "BaseTrainer") -> None:
        """Initialize progress tracking."""
        try:
            from tqdm import tqdm
            self._total_steps = trainer._get_max_steps()
            self._pbar = tqdm(total=self._total_steps, desc="Training")
        except ImportError:
            self._pbar = None
    
    def on_step_end(
        self, trainer: "BaseTrainer", step: int, logs: dict[str, Any]
    ) -> None:
        """Update progress bar."""
        if self._pbar is not None:
            self._pbar.update(1)
            
            # Update metrics display
            postfix = {}
            for metric in self.show_metrics:
                if metric in logs:
                    value = logs[metric]
                    if isinstance(value, float):
                        postfix[metric] = f"{value:.4f}"
                    else:
                        postfix[metric] = str(value)
            if postfix:
                self._pbar.set_postfix(postfix)
    
    def on_train_end(self, trainer: "BaseTrainer") -> None:
        """Close progress bar."""
        if self._pbar is not None:
            self._pbar.close()


class LambdaCallback(Callback):
    """Callback for quick custom hooks via lambda functions.
    
    Allows defining callback behavior inline without subclassing.
    
    Args:
        on_train_begin: Function to call at training start.
        on_train_end: Function to call at training end.
        on_epoch_begin: Function to call at epoch start.
        on_epoch_end: Function to call at epoch end.
        on_step_begin: Function to call at step start.
        on_step_end: Function to call at step end.
    
    Example:
        >>> callback = LambdaCallback(
        ...     on_epoch_end=lambda t, e, l: print(f"Epoch {e} loss: {l['loss']}")
        ... )
        >>> trainer.fit(dataset, callbacks=[callback])
    """
    
    def __init__(
        self,
        on_train_begin: callable = None,
        on_train_end: callable = None,
        on_epoch_begin: callable = None,
        on_epoch_end: callable = None,
        on_step_begin: callable = None,
        on_step_end: callable = None,
    ):
        super().__init__()
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_step_begin = on_step_begin
        self._on_step_end = on_step_end
    
    def on_train_begin(self, trainer: "BaseTrainer") -> None:
        if self._on_train_begin:
            self._on_train_begin(trainer)
    
    def on_train_end(self, trainer: "BaseTrainer") -> None:
        if self._on_train_end:
            self._on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: "BaseTrainer", epoch: int) -> None:
        if self._on_epoch_begin:
            self._on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, logs: dict[str, Any]
    ) -> None:
        if self._on_epoch_end:
            self._on_epoch_end(trainer, epoch, logs)
    
    def on_step_begin(self, trainer: "BaseTrainer", step: int) -> None:
        if self._on_step_begin:
            self._on_step_begin(trainer, step)
    
    def on_step_end(
        self, trainer: "BaseTrainer", step: int, logs: dict[str, Any]
    ) -> None:
        if self._on_step_end:
            self._on_step_end(trainer, step, logs)
