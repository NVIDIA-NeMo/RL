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
"""Unified logging interface for NeMo RL.

This module provides a consistent logging facade that all components should use
instead of direct print statements or varied loggers. It supports:
- Structured logging with consistent format
- Configurable log levels and outputs (file, console, WandB)
- Metric logging to various backends (TensorBoard, WandB, MLflow)
- GPU monitoring integration

Example usage:
    >>> from nemo_rl.infra.logging import get_logger, LoggerFacade
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started")
    >>> logger.debug("Batch size: %d", 32)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from nemo_rl.config.training import LoggerConfig as TrainingLoggerConfig

# Global registry for logger instances
_loggers: dict[str, logging.Logger] = {}
_default_level: int = logging.INFO
_log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_date_format: str = "%Y-%m-%d %H:%M:%S"
_file_handler: logging.FileHandler | None = None


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_logging_level(self) -> int:
        """Convert to Python logging level."""
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[self]


def configure_logging(
    level: str | LogLevel = LogLevel.INFO,
    log_file: str | Path | None = None,
    log_format: str | None = None,
    date_format: str | None = None,
    use_rich: bool = True,
) -> None:
    """Configure global logging settings.

    This should be called once at application startup to configure logging
    for all NeMo RL components.

    Args:
        level: Logging level (debug, info, warning, error, critical).
        log_file: Optional path to log file.
        log_format: Optional custom log format string.
        date_format: Optional custom date format string.
        use_rich: Whether to use rich console formatting.
    """
    global _default_level, _log_format, _date_format, _file_handler

    # Set logging level
    if isinstance(level, str):
        level = LogLevel(level.lower())
    _default_level = level.to_logging_level()

    # Set format strings
    if log_format:
        _log_format = log_format
    if date_format:
        _date_format = date_format

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_default_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                show_time=True,
                show_path=True,
                markup=True,
            )
            console_handler.setLevel(_default_level)
            root_logger.addHandler(console_handler)
        except ImportError:
            # Fallback to standard handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(_default_level)
            formatter = logging.Formatter(_log_format, _date_format)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_default_level)
        formatter = logging.Formatter(_log_format, _date_format)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.FileHandler(str(log_file))
        _file_handler.setLevel(_default_level)
        formatter = logging.Formatter(_log_format, _date_format)
        _file_handler.setFormatter(formatter)
        root_logger.addHandler(_file_handler)

    # Update all existing loggers
    for logger in _loggers.values():
        logger.setLevel(_default_level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Returns a logger with the unified NeMo RL configuration.
    All components should use this function to get loggers.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing batch %d", batch_idx)
    """
    if name is None:
        return logging.getLogger()

    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(_default_level)
        _loggers[name] = logger

    return _loggers[name]


class LoggerFacade:
    """Unified logger facade for NeMo RL components.

    This class provides a consistent interface for logging across all
    NeMo RL components, combining:
    - Standard Python logging
    - Metric logging (TensorBoard, WandB, etc.)
    - Structured logging

    Attributes:
        name: Logger name.
        logger: Underlying Python logger.

    Example:
        >>> facade = LoggerFacade("my_module")
        >>> facade.info("Training started")
        >>> facade.log_metrics({"loss": 0.5, "reward": 1.2}, step=100)
    """

    def __init__(
        self,
        name: str,
        config: "TrainingLoggerConfig | None" = None,
    ):
        """Initialize the logger facade.

        Args:
            name: Logger name.
            config: Optional logger configuration.
        """
        self.name = name
        self.logger = get_logger(name)
        self._config = config
        self._metrics_logger: Any | None = None

        # Initialize metrics logger if config provided
        if config:
            self._init_metrics_logger(config)

    def _init_metrics_logger(self, config: "TrainingLoggerConfig") -> None:
        """Initialize metrics logger from config."""
        try:
            from nemo_rl.utils.logger import Logger as MetricsLogger
            from nemo_rl.utils.logger import LoggerConfig

            # Convert our config to the legacy config format
            legacy_config: LoggerConfig = {
                "log_dir": config.tensorboard_dir,
                "wandb_enabled": config.wandb_enabled,
                "swanlab_enabled": False,
                "tensorboard_enabled": config.tensorboard_enabled,
                "mlflow_enabled": False,
                "wandb": {
                    "project": config.wandb_project or "",
                    "name": config.wandb_run_name or "",
                },
                "monitor_gpus": False,
                "gpu_monitoring": {
                    "collection_interval": 1.0,
                    "flush_interval": 60.0,
                },
            }
            self._metrics_logger = MetricsLogger(legacy_config)
        except ImportError:
            self.logger.warning("Could not initialize metrics logger")

    # =========================================================================
    # Standard Logging Methods
    # =========================================================================

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)

    # =========================================================================
    # Structured Logging
    # =========================================================================

    def log_structured(
        self,
        event: str,
        level: str = "info",
        **fields: Any,
    ) -> None:
        """Log a structured event with fields.

        Args:
            event: Event name/description.
            level: Log level.
            **fields: Additional structured fields.
        """
        # Format as JSON-like structure for easy parsing
        field_str = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        message = f"[{event}] {field_str}" if field_str else f"[{event}]"

        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message)

    # =========================================================================
    # Metric Logging
    # =========================================================================

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to all configured backends.

        Args:
            metrics: Dictionary of metric names and values.
            step: Training step number.
            prefix: Optional prefix for metric names.
        """
        # Log to metrics backends
        if self._metrics_logger:
            self._metrics_logger.log_metrics(metrics, step=step, prefix=prefix)

        # Also log summary to standard logger at debug level
        summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items())
        self.debug(f"Step {step}: {summary}")

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameter names and values.
        """
        if self._metrics_logger:
            self._metrics_logger.log_hyperparams(params)

        # Log summary
        self.info(f"Hyperparameters: {dict(params)}")

    # =========================================================================
    # Context and State
    # =========================================================================

    def set_level(self, level: str | LogLevel) -> None:
        """Set the logging level.

        Args:
            level: New logging level.
        """
        if isinstance(level, str):
            level = LogLevel(level.lower())
        self.logger.setLevel(level.to_logging_level())

    def close(self) -> None:
        """Close the logger and release resources."""
        if self._metrics_logger and hasattr(self._metrics_logger, "finish"):
            self._metrics_logger.finish()


def create_logger_from_config(
    config: "TrainingLoggerConfig",
    name: str = "nemo_rl",
) -> LoggerFacade:
    """Create a LoggerFacade from a configuration object.

    Args:
        config: Logger configuration.
        name: Logger name.

    Returns:
        Configured LoggerFacade instance.
    """
    # Configure global logging
    configure_logging(
        level=config.log_level,
        use_rich=True,
    )

    return LoggerFacade(name, config)


# ============================================================================
# Module-level convenience functions
# ============================================================================

# Default facade for simple use cases
_default_facade: LoggerFacade | None = None


def log_info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message using the default logger."""
    get_logger("nemo_rl").info(msg, *args, **kwargs)


def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message using the default logger."""
    get_logger("nemo_rl").debug(msg, *args, **kwargs)


def log_warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message using the default logger."""
    get_logger("nemo_rl").warning(msg, *args, **kwargs)


def log_error(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message using the default logger."""
    get_logger("nemo_rl").error(msg, *args, **kwargs)


__all__ = [
    "LogLevel",
    "LoggerFacade",
    "configure_logging",
    "get_logger",
    "create_logger_from_config",
    "log_info",
    "log_debug",
    "log_warning",
    "log_error",
]
