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
"""Base trainer class for all NeMo RL training algorithms.

This module provides the BaseTrainer class that handles common setup logic
shared across all training algorithms. Algorithm-specific trainers should
inherit from BaseTrainer and implement the abstract methods.

Features:
- Logger initialization
- Checkpointing setup
- Data loading
- Cluster/resource management
- Common lifecycle methods

Example:
    >>> class MyTrainer(BaseTrainer):
    ...     def _train_step(self, batch):
    ...         # Algorithm-specific training logic
    ...         return {"loss": loss}
    ...
    ...     def _compute_loss(self, batch, outputs):
    ...         return my_loss_function(batch, outputs)
    ...
    >>> trainer = MyTrainer(config)
    >>> trainer.fit(train_data, val_data)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, TypeVar

import numpy as np

if TYPE_CHECKING:
    from nemo_rl.config.base import BaseConfig
    from nemo_rl.config.cluster import ClusterConfig
    from nemo_rl.config.training import (
        CheckpointingConfig,
        LoggerConfig,
    )
    from nemo_rl.data.module import DataModule
    from nemo_rl.infra.checkpointing import CheckpointManager
    from nemo_rl.infra.logging import LoggerFacade
    from nemo_rl.infra.resources import ResourceManager
    from nemo_rl.trainers.callbacks import Callback

T = TypeVar("T")


@dataclass
class TrainingResult:
    """Result of a training run.

    Attributes:
        metrics: Dictionary of final metrics.
        best_checkpoint_path: Path to the best checkpoint.
        total_steps: Total training steps completed.
        final_loss: Final loss value.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    best_checkpoint_path: str | None = None
    total_steps: int = 0
    final_loss: float | None = None


@dataclass
class ValidationResult:
    """Result of a validation run.

    Attributes:
        metrics: Dictionary of validation metrics.
        loss: Validation loss.
        samples: Number of samples evaluated.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    loss: float | None = None
    samples: int = 0


class TrainerState:
    """Tracks the state of training.

    Attributes:
        epoch: Current epoch number.
        global_step: Current global step.
        total_steps: Total steps completed.
        best_metric: Best metric value seen.
        should_stop: Whether training should stop.
    """

    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.total_steps: int = 0
        self.best_metric: float | None = None
        self.should_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "total_steps": self.total_steps,
            "best_metric": self.best_metric,
        }

    def load_dict(self, state: dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)
        self.total_steps = state.get("total_steps", 0)
        self.best_metric = state.get("best_metric")


class BaseTrainer(ABC):
    """Base class for all NeMo RL trainers.

    BaseTrainer handles common setup logic shared across training algorithms:
    - Logger initialization
    - Checkpointing setup
    - Data loading
    - Cluster/resource management

    Subclasses must implement:
    - _train_step(): Perform one training step
    - _compute_loss(): Compute loss for a batch

    Subclasses may override:
    - _validate_step(): Validation logic
    - _setup_model(): Model initialization
    - _setup_optimizer(): Optimizer initialization
    - _on_train_begin(): Hook called before training
    - _on_train_end(): Hook called after training
    - _on_epoch_begin(): Hook called before each epoch
    - _on_epoch_end(): Hook called after each epoch

    Example:
        >>> class GRPOTrainer(BaseTrainer):
        ...     def _train_step(self, batch):
        ...         # Generate responses
        ...         # Compute rewards
        ...         # Update policy
        ...         return {"loss": loss, "reward": reward}

    Attributes:
        config: Training configuration.
        state: Current training state.
        logger: Logger instance.
        checkpoint_manager: Checkpoint manager.
        resource_manager: Resource manager.
    """

    def __init__(self, config: "BaseConfig"):
        """Initialize the trainer.

        Args:
            config: Training configuration object.
        """
        self.config = config
        self.state = TrainerState()
        self._setup_complete = False

        # Components initialized in setup()
        self._logger: "LoggerFacade | None" = None
        self._checkpoint_manager: "CheckpointManager | None" = None
        self._resource_manager: "ResourceManager | None" = None
        self._datamodule: "DataModule | None" = None
        
        # Callback system - initialized as None, set up in fit()
        self._callbacks = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> "BaseTrainer":
        """Create a trainer from a pretrained model.

        This method provides a HuggingFace-style interface for creating trainers.
        It loads model configurations from HuggingFace Hub or local paths and
        creates a fully configured trainer.

        Args:
            model_name_or_path: Either:
                - A HuggingFace Hub model identifier (e.g., 'Qwen/Qwen2.5-1.5B')
                - A local path to a model directory
            **kwargs: Optional configuration overrides. Common options:
                - learning_rate: Learning rate for training
                - batch_size: Training batch size
                - max_steps: Maximum training steps
                - max_epochs: Maximum training epochs
                - output_dir: Directory for checkpoints and logs
                - backend: Training backend ('dtensor' or 'megatron')

        Returns:
            Configured trainer instance ready for training.

        Raises:
            ValueError: If model cannot be loaded.
            FileNotFoundError: If local path doesn't exist.

        Example:
            >>> # From HuggingFace Hub
            >>> trainer = GRPOTrainer.from_pretrained(
            ...     "Qwen/Qwen2.5-1.5B",
            ...     learning_rate=1e-6,
            ...     num_prompts_per_step=32,
            ... )
            >>> 
            >>> # From local path
            >>> trainer = SFTTrainer.from_pretrained(
            ...     "/path/to/model",
            ...     batch_size=16,
            ... )
            >>> 
            >>> # Then train
            >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")
        """
        config = cls._build_config_from_pretrained(model_name_or_path, **kwargs)
        return cls(config)

    @classmethod
    def _build_config_from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build configuration from a pretrained model.

        Subclasses should override this to provide algorithm-specific configs.

        Args:
            model_name_or_path: Model identifier or path.
            **kwargs: Configuration overrides.

        Returns:
            Configuration dictionary.
        """
        import os

        # Determine model source
        is_local = os.path.exists(model_name_or_path)

        # Load model config from HuggingFace or local
        model_config = cls._load_model_config(model_name_or_path, is_local)

        # Build trainer config
        config: dict[str, Any] = {
            "policy": {
                "model_name": model_name_or_path,
                "learning_rate": kwargs.pop("learning_rate", 1e-6),
                **model_config,
            },
        }

        # Add output directory
        output_dir = kwargs.pop("output_dir", None)
        if output_dir is None:
            model_short = model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path
            output_dir = f"./outputs/{model_short}"

        config["checkpointing"] = {
            "checkpoint_dir": f"{output_dir}/checkpoints",
            "enabled": kwargs.pop("save_checkpoints", True),
            "save_period": kwargs.pop("save_every", 100),
        }

        config["logger"] = {
            "log_level": kwargs.pop("log_level", "INFO"),
            "tensorboard_enabled": kwargs.pop("tensorboard", True),
            "tensorboard_dir": f"{output_dir}/logs",
        }

        # Add any remaining kwargs to root config
        config.update(kwargs)

        return config

    @classmethod
    def _load_model_config(
        cls,
        model_name_or_path: str,
        is_local: bool,
    ) -> dict[str, Any]:
        """Load model configuration from HuggingFace or local.

        Args:
            model_name_or_path: Model identifier or path.
            is_local: Whether path is local.

        Returns:
            Model configuration dictionary.
        """
        try:
            if is_local:
                return cls._load_local_model_config(model_name_or_path)
            else:
                return cls._load_hub_model_config(model_name_or_path)
        except Exception as e:
            import logging
            logging.warning(f"Could not load model config: {e}. Using defaults.")
            return {}

    @classmethod
    def _load_local_model_config(cls, path: str) -> dict[str, Any]:
        """Load model config from local path.

        Args:
            path: Local path to model directory.

        Returns:
            Model configuration dictionary.
        """
        import json
        import os

        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                hf_config = json.load(f)

            return cls._extract_config_from_hf(hf_config)

        return {}

    @classmethod
    def _load_hub_model_config(cls, model_id: str) -> dict[str, Any]:
        """Load model config from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            Model configuration dictionary.
        """
        try:
            from huggingface_hub import hf_hub_download

            # Download config.json
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
            )

            import json
            with open(config_path) as f:
                hf_config = json.load(f)

            return cls._extract_config_from_hf(hf_config)
        except ImportError:
            import logging
            logging.debug("huggingface_hub not available, using empty config")
            return {}
        except Exception:
            return {}

    @classmethod
    def _extract_config_from_hf(cls, hf_config: dict[str, Any]) -> dict[str, Any]:
        """Extract relevant config from HuggingFace model config.

        Args:
            hf_config: Raw HuggingFace config.json content.

        Returns:
            Extracted configuration for training.
        """
        # Extract useful parameters for training
        config: dict[str, Any] = {}

        # Model architecture info (useful for defaults)
        if "hidden_size" in hf_config:
            config["hidden_size"] = hf_config["hidden_size"]

        if "num_hidden_layers" in hf_config:
            config["num_layers"] = hf_config["num_hidden_layers"]

        if "vocab_size" in hf_config:
            config["vocab_size"] = hf_config["vocab_size"]

        if "max_position_embeddings" in hf_config:
            config["max_seq_length"] = hf_config["max_position_embeddings"]

        return config

    # =========================================================================
    # Public API
    # =========================================================================

    def setup(self) -> None:
        """Initialize all trainer components.

        This method should be called before fit() to set up:
        - Logger
        - Checkpointing
        - Cluster resources
        - Data module

        Can also be called automatically by fit() if not done explicitly.
        """
        if self._setup_complete:
            return

        self._setup_seed()
        self._setup_logger()
        self._setup_checkpointing()
        self._setup_cluster()
        self._setup_model()
        self._setup_optimizer()

        self._setup_complete = True

    def fit(
        self,
        datamodule: "DataModule | None" = None,
        dataset: str | Any | None = None,
        train_data: Any = None,
        val_data: Any = None,
        max_epochs: int | None = None,
        max_steps: int | None = None,
        callbacks: Sequence["Callback"] | None = None,
    ) -> TrainingResult:
        """Run the training loop.

        Args:
            datamodule: DataModule instance for providing data.
            dataset: Dataset name (HuggingFace Hub) or data object. This is the
                preferred way to provide data. Strings are automatically loaded
                from HuggingFace with auto-column mapping.
            train_data: Training data (alternative to datamodule).
            val_data: Validation data (alternative to datamodule).
            max_epochs: Override maximum number of epochs.
            max_steps: Override maximum number of steps.
            callbacks: List of callback instances for custom hooks. Callbacks
                are invoked at various points in the training lifecycle.

        Returns:
            TrainingResult with metrics and checkpoint info.

        Raises:
            ValueError: If no data source is provided.

        Example:
            >>> # Load HuggingFace dataset with zero configuration
            >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")
            >>> 
            >>> # With callbacks
            >>> from nemo_rl.trainers.callbacks import (
            ...     CheckpointCallback, EarlyStoppingCallback
            ... )
            >>> trainer.fit(
            ...     dataset="my-dataset",
            ...     callbacks=[
            ...         CheckpointCallback(every_n_steps=100),
            ...         EarlyStoppingCallback(monitor='loss', patience=3),
            ...     ]
            ... )
            >>> 
            >>> # Or with explicit datamodule
            >>> trainer.fit(datamodule=MyDataModule())
        """
        from nemo_rl.trainers.callbacks import CallbackList
        
        # Ensure setup is complete
        if not self._setup_complete:
            self.setup()

        # Setup callbacks
        self._callbacks = CallbackList(list(callbacks) if callbacks else None)

        # Setup data - prioritize explicit datamodule, then dataset, then train_data
        if datamodule is not None:
            self._datamodule = datamodule
        elif dataset is not None:
            # Dataset can be a string (HF name) or any other data format
            self._datamodule = self._create_datamodule(dataset, val_data)
        elif train_data is not None:
            self._datamodule = self._create_datamodule(train_data, val_data)
        elif self._datamodule is None:
            raise ValueError(
                "Must provide one of: datamodule, dataset, or train_data. "
                "For HuggingFace datasets, use dataset='dataset-name'."
            )

        self._datamodule.setup("fit")

        # Get max epochs/steps from config if not overridden
        max_epochs = max_epochs or self._get_max_epochs()
        max_steps = max_steps or self._get_max_steps()

        # Call training hooks (includes callback invocation)
        self._on_train_begin()

        try:
            result = self._training_loop(max_epochs, max_steps)
        finally:
            self._on_train_end()

        return result

    def validate(
        self,
        datamodule: "DataModule | None" = None,
        val_data: Any = None,
    ) -> ValidationResult:
        """Run validation.

        Args:
            datamodule: DataModule instance for validation data.
            val_data: Validation data (alternative to datamodule).

        Returns:
            ValidationResult with metrics.
        """
        if not self._setup_complete:
            self.setup()

        if datamodule is not None:
            self._datamodule = datamodule
            self._datamodule.setup("validate")
        elif val_data is not None:
            self._datamodule = self._create_datamodule(None, val_data)
            self._datamodule.setup("validate")

        if self._datamodule is None:
            raise ValueError("Must provide datamodule or val_data")

        return self._validation_loop()

    def cleanup(self) -> None:
        """Clean up resources after training.

        Should be called when training is complete to release resources.
        """
        if self._datamodule is not None:
            self._datamodule.teardown()

        if self._resource_manager is not None:
            self._resource_manager.release_all()

        self._setup_complete = False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def logger(self) -> "LoggerFacade | None":
        """Logger instance."""
        return self._logger

    @property
    def checkpoint_manager(self) -> "CheckpointManager | None":
        """Checkpoint manager."""
        return self._checkpoint_manager

    @property
    def resource_manager(self) -> "ResourceManager | None":
        """Resource manager."""
        return self._resource_manager

    @property
    def current_epoch(self) -> int:
        """Current epoch number."""
        return self.state.epoch

    @property
    def global_step(self) -> int:
        """Current global step."""
        return self.state.global_step

    # =========================================================================
    # Abstract Methods (Must Override)
    # =========================================================================

    @abstractmethod
    def _train_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            batch: Batch of training data.

        Returns:
            Dictionary with at least 'loss' key.
        """
        pass

    @abstractmethod
    def _compute_loss(self, batch: Any, outputs: Any) -> Any:
        """Compute the training loss.

        Args:
            batch: Input batch.
            outputs: Model outputs.

        Returns:
            Loss value (scalar tensor).
        """
        pass

    # =========================================================================
    # Optional Override Methods
    # =========================================================================

    def _validate_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single validation step.

        Override this to customize validation. Default calls _train_step
        in eval mode.

        Args:
            batch: Batch of validation data.

        Returns:
            Dictionary with validation metrics.
        """
        # Default: same as train step but in eval mode
        return self._train_step(batch)

    def _setup_model(self) -> None:
        """Set up the model for training.

        Override to initialize model-specific components.
        """
        pass

    def _setup_optimizer(self) -> None:
        """Set up the optimizer and scheduler.

        Override to initialize optimizer-specific components.
        """
        pass

    def _on_train_begin(self) -> None:
        """Hook called before training starts.

        Override to add custom logic before training loop.
        Callbacks are invoked automatically.
        """
        if self._logger:
            self._logger.info("Training started")
        
        # Invoke callbacks
        if self._callbacks is not None:
            self._callbacks.on_train_begin(self)

    def _on_train_end(self) -> None:
        """Hook called after training completes.

        Override to add custom logic after training loop.
        Callbacks are invoked automatically.
        """
        # Invoke callbacks
        if self._callbacks is not None:
            self._callbacks.on_train_end(self)
        
        if self._logger:
            self._logger.info("Training completed")

    def _on_epoch_begin(self, epoch: int) -> None:
        """Hook called before each epoch.

        Args:
            epoch: Current epoch number.
        """
        # Invoke callbacks
        if self._callbacks is not None:
            self._callbacks.on_epoch_begin(self, epoch)

    def _on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Hook called after each epoch.

        Args:
            epoch: Current epoch number.
            metrics: Metrics for this epoch.
        """
        # Invoke callbacks
        if self._callbacks is not None:
            self._callbacks.on_epoch_end(self, epoch, metrics)

    def _on_step_begin(self, step: int) -> None:
        """Hook called before each training step.

        Args:
            step: Current step number.
        """
        # Invoke callbacks
        if self._callbacks is not None:
            self._callbacks.on_step_begin(self, step)

    def _on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Hook called after each training step.

        Args:
            step: Current step number.
            metrics: Metrics for this step.
        """
        # Invoke callbacks
        if self._callbacks is not None:
            self._callbacks.on_step_end(self, step, metrics)

    # =========================================================================
    # Internal Setup Methods
    # =========================================================================

    def _setup_seed(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self._get_seed()
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _setup_logger(self) -> None:
        """Initialize the logger."""
        from nemo_rl.infra.logging import LoggerFacade, configure_logging

        logger_config = self._get_logger_config()
        if logger_config is not None:
            configure_logging(level=logger_config.log_level)

        self._logger = LoggerFacade(
            name=self.__class__.__name__,
            wandb_enabled=logger_config.wandb_enabled if logger_config else False,
            tensorboard_enabled=logger_config.tensorboard_enabled
            if logger_config
            else True,
            log_dir=logger_config.tensorboard_dir if logger_config else "logs",
        )

        if self._logger:
            self._logger.info(f"Initialized {self.__class__.__name__}")

    def _setup_checkpointing(self) -> None:
        """Initialize the checkpoint manager."""
        from nemo_rl.infra.checkpointing import CheckpointManager

        ckpt_config = self._get_checkpointing_config()
        if ckpt_config is not None and ckpt_config.enabled:
            self._checkpoint_manager = CheckpointManager(
                checkpoint_dir=ckpt_config.checkpoint_dir,
                save_period=ckpt_config.save_period,
                keep_top_k=ckpt_config.keep_top_k,
            )

            # Load existing state if resuming
            last_checkpoint = self._checkpoint_manager.get_latest_checkpoint()
            if last_checkpoint is not None:
                self._load_checkpoint(last_checkpoint)

    def _setup_cluster(self) -> None:
        """Initialize cluster resources."""
        from nemo_rl.infra.resources import ResourceManager

        cluster_config = self._get_cluster_config()
        if cluster_config is not None:
            self._resource_manager = ResourceManager(
                num_nodes=cluster_config.num_nodes,
                gpus_per_node=cluster_config.gpus_per_node,
            )
        else:
            self._resource_manager = ResourceManager.auto_detect()

    # =========================================================================
    # Internal Training Loop
    # =========================================================================

    def _training_loop(
        self, max_epochs: int, max_steps: int
    ) -> TrainingResult:
        """Main training loop.

        Args:
            max_epochs: Maximum number of epochs.
            max_steps: Maximum number of steps.

        Returns:
            TrainingResult with final metrics.
        """
        train_dataloader = self._datamodule.train_dataloader()
        final_metrics: dict[str, Any] = {}
        final_loss = None

        for epoch in range(max_epochs):
            if self.state.should_stop:
                break

            self.state.epoch = epoch
            self._on_epoch_begin(epoch)

            epoch_metrics = self._train_epoch(train_dataloader, max_steps)
            final_metrics.update(epoch_metrics)

            if "loss" in epoch_metrics:
                final_loss = epoch_metrics["loss"]

            # Validation at end of epoch
            val_result = self._maybe_validate()
            if val_result:
                final_metrics.update(
                    {f"val_{k}": v for k, v in val_result.metrics.items()}
                )

            self._on_epoch_end(epoch, epoch_metrics)

            # Check early stopping
            if self.state.global_step >= max_steps:
                break

        return TrainingResult(
            metrics=final_metrics,
            best_checkpoint_path=self._checkpoint_manager.best_checkpoint_path
            if self._checkpoint_manager
            else None,
            total_steps=self.state.total_steps,
            final_loss=final_loss,
        )

    def _train_epoch(
        self, dataloader: Iterator, max_steps: int
    ) -> dict[str, Any]:
        """Train for one epoch.

        Args:
            dataloader: Training data iterator.
            max_steps: Maximum steps (stops early if reached).

        Returns:
            Dictionary of epoch metrics.
        """
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_metrics: dict[str, Any] = {}

        for batch in dataloader:
            if self.state.global_step >= max_steps:
                break

            self._on_step_begin(self.state.global_step)

            # Training step
            step_metrics = self._train_step(batch)
            loss = step_metrics.get("loss", 0.0)
            epoch_loss += loss if isinstance(loss, (int, float)) else loss.item()
            epoch_steps += 1

            # Update state
            self.state.global_step += 1
            self.state.total_steps += 1

            # Logging
            if self._logger and self.state.global_step % self._get_log_interval() == 0:
                self._logger.log_metrics(step_metrics, step=self.state.global_step)

            # Checkpointing
            if self._checkpoint_manager and self._should_checkpoint():
                self._save_checkpoint()

            self._on_step_end(self.state.global_step, step_metrics)

            # Update epoch metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value if isinstance(value, (int, float)) else 0.0

        # Average epoch metrics
        if epoch_steps > 0:
            epoch_metrics["loss"] = epoch_loss / epoch_steps
            for key in epoch_metrics:
                if key != "loss":
                    epoch_metrics[key] /= epoch_steps

        return epoch_metrics

    def _validation_loop(self) -> ValidationResult:
        """Run validation loop.

        Returns:
            ValidationResult with metrics.
        """
        val_dataloader = self._datamodule.val_dataloader()
        if val_dataloader is None:
            return ValidationResult()

        val_loss = 0.0
        val_samples = 0
        val_metrics: dict[str, Any] = {}

        for batch in val_dataloader:
            step_metrics = self._validate_step(batch)
            loss = step_metrics.get("loss", 0.0)
            val_loss += loss if isinstance(loss, (int, float)) else loss.item()
            val_samples += 1

            for key, value in step_metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = 0.0
                val_metrics[key] += value if isinstance(value, (int, float)) else 0.0

        # Average metrics
        if val_samples > 0:
            val_loss /= val_samples
            for key in val_metrics:
                val_metrics[key] /= val_samples

        return ValidationResult(
            metrics=val_metrics,
            loss=val_loss,
            samples=val_samples,
        )

    def _maybe_validate(self) -> ValidationResult | None:
        """Run validation if due.

        Returns:
            ValidationResult or None if not due.
        """
        val_period = self._get_val_period()
        if val_period > 0 and self.state.global_step % val_period == 0:
            return self._validation_loop()
        return None

    # =========================================================================
    # Checkpoint Methods
    # =========================================================================

    def _should_checkpoint(self) -> bool:
        """Check if should save checkpoint."""
        if self._checkpoint_manager is None:
            return False
        save_period = self._get_checkpoint_period()
        return save_period > 0 and self.state.global_step % save_period == 0

    def _save_checkpoint(self) -> None:
        """Save a checkpoint."""
        if self._checkpoint_manager is None:
            return

        state_dict = {
            "trainer_state": self.state.to_dict(),
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else {},
        }

        self._checkpoint_manager.save(
            state_dict=state_dict,
            step=self.state.global_step,
        )

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        if self._checkpoint_manager is None:
            return

        state_dict = self._checkpoint_manager.load(checkpoint_path)
        if state_dict and "trainer_state" in state_dict:
            self.state.load_dict(state_dict["trainer_state"])

    # =========================================================================
    # Config Accessors (Override if config structure differs)
    # =========================================================================

    def _get_seed(self) -> int:
        """Get random seed from config."""
        return getattr(self.config, "seed", 42)

    def _get_max_epochs(self) -> int:
        """Get max epochs from config."""
        return getattr(self.config, "max_num_epochs", 1)

    def _get_max_steps(self) -> int:
        """Get max steps from config."""
        return getattr(self.config, "max_num_steps", 10000)

    def _get_log_interval(self) -> int:
        """Get logging interval from config."""
        logger_config = self._get_logger_config()
        return logger_config.log_interval if logger_config else 10

    def _get_val_period(self) -> int:
        """Get validation period from config."""
        return getattr(self.config, "val_period", 100)

    def _get_checkpoint_period(self) -> int:
        """Get checkpoint save period from config."""
        ckpt_config = self._get_checkpointing_config()
        return ckpt_config.save_period if ckpt_config else 100

    def _get_logger_config(self) -> "LoggerConfig | None":
        """Get logger config from main config."""
        return getattr(self.config, "logger", None)

    def _get_cluster_config(self) -> "ClusterConfig | None":
        """Get cluster config from main config."""
        return getattr(self.config, "cluster", None)

    def _get_checkpointing_config(self) -> "CheckpointingConfig | None":
        """Get checkpointing config from main config."""
        return getattr(self.config, "checkpointing", None)

    def _create_datamodule(
        self, train_data: Any, val_data: Any = None
    ) -> "DataModule":
        """Create a DataModule from raw data.

        Args:
            train_data: Training data.
            val_data: Validation data.

        Returns:
            DataModule instance.
        """
        from nemo_rl.data.module import create_datamodule

        return create_datamodule(train_data=train_data, val_data=val_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(step={self.state.global_step})"
