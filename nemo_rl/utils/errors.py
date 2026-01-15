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
"""Comprehensive error classes and utilities for NeMo RL.

This module provides enhanced error handling with:
- Specific field names and values in error messages
- Suggested fixes for common mistakes
- Fuzzy matching for typos
- Links to relevant documentation
- Context about what operation was attempted

Example:
    >>> from nemo_rl.utils.errors import ConfigError
    >>> raise ConfigError(
    ...     "Invalid backend type",
    ...     field="backend",
    ...     value="tensor",
    ...     valid_options=["dtensor", "megatron"],
    ...     context="PolicyConfig initialization"
    ... )
"""

from __future__ import annotations

import difflib
from typing import Any, Sequence


# Documentation base URL
DOCS_BASE_URL = "https://nvidia.github.io/nemo-rl"


def get_doc_link(topic: str) -> str:
    """Get a documentation link for a given topic.

    Args:
        topic: Documentation topic (e.g., 'config', 'backends', 'training').

    Returns:
        URL to the documentation page.
    """
    topic_map = {
        "config": "/api/config",
        "backends": "/api/backends",
        "training": "/api/trainers",
        "grpo": "/api/trainers#grpo",
        "sft": "/api/trainers#sft",
        "dpo": "/api/trainers#dpo",
        "policy": "/api/config#policy",
        "cluster": "/api/config#cluster",
        "environment": "/api/environments",
        "quickstart": "/quickstart",
        "migration": "/migration",
    }
    path = topic_map.get(topic.lower(), "")
    return f"{DOCS_BASE_URL}{path}"


def fuzzy_match(value: str, options: Sequence[str], cutoff: float = 0.6) -> str | None:
    """Find the closest match for a value from a list of options.

    Uses sequence matching to find similar strings, useful for
    detecting typos and suggesting corrections.

    Args:
        value: The value to match.
        options: List of valid options.
        cutoff: Minimum similarity ratio (0-1). Default 0.6.

    Returns:
        The closest match if found, or None if no close match exists.

    Example:
        >>> fuzzy_match("tensor", ["dtensor", "megatron"])
        'dtensor'
        >>> fuzzy_match("xyz", ["dtensor", "megatron"])
        None
    """
    if not value or not options:
        return None

    # Convert to lowercase for comparison
    value_lower = value.lower()
    options_lower = [opt.lower() for opt in options]

    matches = difflib.get_close_matches(value_lower, options_lower, n=1, cutoff=cutoff)
    if matches:
        # Return the original case option
        idx = options_lower.index(matches[0])
        return options[idx]
    return None


def format_options(options: Sequence[Any], max_display: int = 5) -> str:
    """Format a list of options for display in error messages.

    Args:
        options: List of valid options.
        max_display: Maximum number of options to display.

    Returns:
        Formatted string showing the options.
    """
    if not options:
        return "(no valid options)"

    str_options = [repr(opt) for opt in options]
    if len(str_options) <= max_display:
        return ", ".join(str_options)
    else:
        shown = ", ".join(str_options[:max_display])
        remaining = len(str_options) - max_display
        return f"{shown}, ... and {remaining} more"


class NeMoRLError(Exception):
    """Base exception class for all NeMo RL errors.

    Provides structured error messages with context information.
    """

    def __init__(
        self,
        message: str,
        *,
        context: str | None = None,
        suggestion: str | None = None,
        doc_topic: str | None = None,
    ):
        """Initialize the error.

        Args:
            message: Main error message.
            context: Optional context about what operation failed.
            suggestion: Optional suggestion for fixing the error.
            doc_topic: Optional documentation topic for reference.
        """
        self.message = message
        self.context = context
        self.suggestion = suggestion
        self.doc_topic = doc_topic

        # Build the full error message
        parts = [message]
        if context:
            parts.append(f"\nContext: {context}")
        if suggestion:
            parts.append(f"\nSuggestion: {suggestion}")
        if doc_topic:
            parts.append(f"\nDocumentation: {get_doc_link(doc_topic)}")

        super().__init__("\n".join(parts))


class ConfigError(NeMoRLError):
    """Error raised for configuration-related issues.

    Provides detailed information about which field is invalid,
    what value was provided, and what values are valid.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        expected_type: type | str | None = None,
        valid_options: Sequence[Any] | None = None,
        valid_range: tuple[Any, Any] | None = None,
        context: str | None = None,
        suggestion: str | None = None,
        doc_topic: str | None = None,
    ):
        """Initialize the configuration error.

        Args:
            message: Main error message.
            field: Name of the invalid field.
            value: The invalid value provided.
            expected_type: Expected type of the field.
            valid_options: List of valid values (for enum-like fields).
            valid_range: Valid range as (min, max) tuple.
            context: Context about what operation failed.
            suggestion: Suggestion for fixing the error.
            doc_topic: Documentation topic for reference.
        """
        self.field = field
        self.value = value
        self.expected_type = expected_type
        self.valid_options = valid_options
        self.valid_range = valid_range

        # Build detailed message
        parts = [f"Configuration Error: {message}"]

        if field:
            parts.append(f"\n  Field: {field}")

        if value is not None:
            parts.append(f"\n  Provided: {value!r} (type: {type(value).__name__})")

        if expected_type:
            type_str = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
            parts.append(f"\n  Expected type: {type_str}")

        if valid_options:
            parts.append(f"\n  Valid options: {format_options(valid_options)}")

            # Try fuzzy matching for suggestions
            if value is not None and isinstance(value, str):
                match = fuzzy_match(value, [str(o) for o in valid_options])
                if match and not suggestion:
                    suggestion = f"Did you mean '{match}'?"

        if valid_range:
            min_val, max_val = valid_range
            parts.append(f"\n  Valid range: [{min_val}, {max_val}]")

        # Generate default suggestion if not provided
        if not suggestion:
            suggestion = self._generate_default_suggestion()

        # Set default doc topic for config errors
        if not doc_topic:
            doc_topic = "config"

        full_message = "".join(parts)
        super().__init__(
            full_message,
            context=context,
            suggestion=suggestion,
            doc_topic=doc_topic,
        )

    def _generate_default_suggestion(self) -> str | None:
        """Generate a default suggestion based on error context."""
        if self.valid_options:
            return f"Choose one of: {format_options(self.valid_options, max_display=10)}"
        if self.valid_range:
            min_val, max_val = self.valid_range
            return f"Use a value between {min_val} and {max_val}"
        if self.expected_type:
            type_str = self.expected_type.__name__ if isinstance(self.expected_type, type) else str(self.expected_type)
            return f"Provide a valid {type_str} value"
        return None


class BackendError(NeMoRLError):
    """Error raised for backend-related issues.

    Used when there are problems with training or generation backends.
    """

    VALID_TRAINING_BACKENDS = ["dtensor", "megatron"]
    VALID_GENERATION_BACKENDS = ["vllm", "megatron"]

    def __init__(
        self,
        message: str,
        *,
        backend_type: str | None = None,
        backend_name: str | None = None,
        valid_backends: Sequence[str] | None = None,
        context: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize the backend error.

        Args:
            message: Main error message.
            backend_type: Type of backend ('training' or 'generation').
            backend_name: Name of the backend that caused the error.
            valid_backends: List of valid backend names.
            context: Context about what operation failed.
            suggestion: Suggestion for fixing the error.
        """
        self.backend_type = backend_type
        self.backend_name = backend_name

        # Set valid backends based on type if not provided
        if valid_backends is None and backend_type:
            if backend_type == "training":
                valid_backends = self.VALID_TRAINING_BACKENDS
            elif backend_type == "generation":
                valid_backends = self.VALID_GENERATION_BACKENDS

        self.valid_backends = valid_backends

        parts = [f"Backend Error: {message}"]

        if backend_type:
            parts.append(f"\n  Backend type: {backend_type}")

        if backend_name:
            parts.append(f"\n  Backend name: {backend_name!r}")

        if valid_backends:
            parts.append(f"\n  Valid backends: {format_options(valid_backends)}")

            # Fuzzy match for suggestions
            if backend_name and not suggestion:
                match = fuzzy_match(backend_name, valid_backends)
                if match:
                    suggestion = f"Did you mean '{match}'?"

        full_message = "".join(parts)
        super().__init__(
            full_message,
            context=context,
            suggestion=suggestion or f"Use one of: {', '.join(valid_backends or [])}",
            doc_topic="backends",
        )


class TrainingError(NeMoRLError):
    """Error raised during training operations.

    Used for runtime errors during model training.
    """

    def __init__(
        self,
        message: str,
        *,
        step: int | None = None,
        epoch: int | None = None,
        batch_info: dict[str, Any] | None = None,
        context: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize the training error.

        Args:
            message: Main error message.
            step: Training step where error occurred.
            epoch: Training epoch where error occurred.
            batch_info: Information about the batch being processed.
            context: Context about what operation failed.
            suggestion: Suggestion for fixing the error.
        """
        self.step = step
        self.epoch = epoch
        self.batch_info = batch_info

        parts = [f"Training Error: {message}"]

        if step is not None:
            parts.append(f"\n  Step: {step}")

        if epoch is not None:
            parts.append(f"\n  Epoch: {epoch}")

        if batch_info:
            parts.append(f"\n  Batch info: {batch_info}")

        full_message = "".join(parts)
        super().__init__(
            full_message,
            context=context,
            suggestion=suggestion,
            doc_topic="training",
        )


class DataError(NeMoRLError):
    """Error raised for data-related issues.

    Used when there are problems with data loading or processing.
    """

    def __init__(
        self,
        message: str,
        *,
        data_path: str | None = None,
        row_index: int | None = None,
        column: str | None = None,
        expected_format: str | None = None,
        context: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize the data error.

        Args:
            message: Main error message.
            data_path: Path to the data file.
            row_index: Row index where error occurred.
            column: Column name where error occurred.
            expected_format: Expected data format.
            context: Context about what operation failed.
            suggestion: Suggestion for fixing the error.
        """
        self.data_path = data_path
        self.row_index = row_index
        self.column = column
        self.expected_format = expected_format

        parts = [f"Data Error: {message}"]

        if data_path:
            parts.append(f"\n  File: {data_path}")

        if row_index is not None:
            parts.append(f"\n  Row: {row_index}")

        if column:
            parts.append(f"\n  Column: {column}")

        if expected_format:
            parts.append(f"\n  Expected format: {expected_format}")

        full_message = "".join(parts)
        super().__init__(
            full_message,
            context=context,
            suggestion=suggestion,
        )


class EnvironmentError(NeMoRLError):
    """Error raised for environment/reward function issues.

    Used when there are problems with reward computation or environments.
    """

    def __init__(
        self,
        message: str,
        *,
        environment_type: str | None = None,
        prompt: str | None = None,
        response: str | None = None,
        context: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize the environment error.

        Args:
            message: Main error message.
            environment_type: Type of environment that raised the error.
            prompt: The prompt being scored (truncated if long).
            response: The response being scored (truncated if long).
            context: Context about what operation failed.
            suggestion: Suggestion for fixing the error.
        """
        self.environment_type = environment_type
        self.prompt = prompt
        self.response = response

        parts = [f"Environment Error: {message}"]

        if environment_type:
            parts.append(f"\n  Environment: {environment_type}")

        if prompt:
            truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            parts.append(f"\n  Prompt: {truncated_prompt!r}")

        if response:
            truncated_response = response[:100] + "..." if len(response) > 100 else response
            parts.append(f"\n  Response: {truncated_response!r}")

        full_message = "".join(parts)
        super().__init__(
            full_message,
            context=context,
            suggestion=suggestion,
            doc_topic="environment",
        )


class CheckpointError(NeMoRLError):
    """Error raised for checkpoint-related issues.

    Used when there are problems saving or loading checkpoints.
    """

    def __init__(
        self,
        message: str,
        *,
        checkpoint_path: str | None = None,
        step: int | None = None,
        context: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize the checkpoint error.

        Args:
            message: Main error message.
            checkpoint_path: Path to the checkpoint.
            step: Training step for the checkpoint.
            context: Context about what operation failed.
            suggestion: Suggestion for fixing the error.
        """
        self.checkpoint_path = checkpoint_path
        self.step = step

        parts = [f"Checkpoint Error: {message}"]

        if checkpoint_path:
            parts.append(f"\n  Path: {checkpoint_path}")

        if step is not None:
            parts.append(f"\n  Step: {step}")

        full_message = "".join(parts)
        super().__init__(
            full_message,
            context=context,
            suggestion=suggestion,
        )


# Common error scenarios with pre-built suggestions
COMMON_ERROR_SUGGESTIONS = {
    "out_of_memory": (
        "GPU out of memory. Try:\n"
        "  1. Reduce batch size (train_global_batch_size, train_micro_batch_size)\n"
        "  2. Enable activation checkpointing (activation_checkpointing=True)\n"
        "  3. Use a smaller model\n"
        "  4. Reduce sequence length (max_total_sequence_length)"
    ),
    "invalid_model": (
        "Model not found or invalid. Check:\n"
        "  1. Model name is correct (e.g., 'Qwen/Qwen2.5-1.5B')\n"
        "  2. HuggingFace credentials are set for gated models\n"
        "  3. Model path exists for local models"
    ),
    "cuda_not_available": (
        "CUDA not available. Ensure:\n"
        "  1. NVIDIA drivers are installed\n"
        "  2. PyTorch is installed with CUDA support\n"
        "  3. CUDA_VISIBLE_DEVICES is set correctly"
    ),
    "data_format": (
        "Invalid data format. Ensure:\n"
        "  1. Data is in JSONL, JSON, or Parquet format\n"
        "  2. Each row has required fields (prompt, response, etc.)\n"
        "  3. Text fields are strings, not null"
    ),
    "parallelism": (
        "Parallelism configuration error. Check:\n"
        "  1. tensor_parallel_size divides number of GPUs evenly\n"
        "  2. pipeline_parallel_size * tensor_parallel_size <= total GPUs\n"
        "  3. Parallel sizes are powers of 2"
    ),
}


def get_common_suggestion(error_type: str) -> str | None:
    """Get a pre-built suggestion for common error scenarios.

    Args:
        error_type: Type of error ('out_of_memory', 'invalid_model', etc.).

    Returns:
        Suggestion string or None if not found.
    """
    return COMMON_ERROR_SUGGESTIONS.get(error_type)


__all__ = [
    "NeMoRLError",
    "ConfigError",
    "BackendError",
    "TrainingError",
    "DataError",
    "EnvironmentError",
    "CheckpointError",
    "fuzzy_match",
    "format_options",
    "get_doc_link",
    "get_common_suggestion",
    "COMMON_ERROR_SUGGESTIONS",
]
