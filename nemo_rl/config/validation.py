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
"""Custom validators for NeMo RL configuration.

This module provides custom validation functions and decorators for
configuration validation beyond what Pydantic provides by default.

Features:
- Comprehensive error messages with field names and values
- Suggested fixes for common mistakes
- Context-aware validation helpers
- Fuzzy matching for typos in enum-like fields

Example:
    >>> from nemo_rl.config.validation import validate_one_of, validate_positive
    >>> validate_one_of("tensor", "backend", ["dtensor", "megatron"])
    # Raises: ConfigValidationError with suggestion "Did you mean 'dtensor'?"
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeVar

from pydantic import ValidationError

# Re-export ConfigValidationError for convenience
from nemo_rl.config.base import ConfigValidationError

T = TypeVar("T")


def _try_fuzzy_match(value: str, options: Sequence[str]) -> str | None:
    """Try to find a fuzzy match for a value in options.

    Args:
        value: The value to match.
        options: List of valid options.

    Returns:
        The closest match or None.
    """
    try:
        from nemo_rl.utils.errors import fuzzy_match
        return fuzzy_match(value, options)
    except ImportError:
        return None


def validate_positive(value: int | float, field_name: str) -> int | float:
    """Validate that a value is positive (> 0).

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.

    Returns:
        The value if valid.

    Raises:
        ConfigValidationError: If the value is not positive.
    """
    if value <= 0:
        raise ConfigValidationError(
            message=f"{field_name} must be positive",
            field=field_name,
            value=value,
            suggestion=f"Use a value greater than 0 for {field_name}",
        )
    return value


def validate_non_negative(value: int | float, field_name: str) -> int | float:
    """Validate that a value is non-negative (>= 0).

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.

    Returns:
        The value if valid.

    Raises:
        ConfigValidationError: If the value is negative.
    """
    if value < 0:
        raise ConfigValidationError(
            message=f"{field_name} must be non-negative",
            field=field_name,
            value=value,
            suggestion=f"Use a value >= 0 for {field_name}",
        )
    return value


def validate_range(
    value: int | float,
    field_name: str,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> int | float:
    """Validate that a value is within a specified range.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value (inclusive). None means no minimum.
        max_value: Maximum allowed value (inclusive). None means no maximum.

    Returns:
        The value if valid.

    Raises:
        ConfigValidationError: If the value is outside the range.
    """
    if min_value is not None and value < min_value:
        raise ConfigValidationError(
            message=f"{field_name} must be >= {min_value}",
            field=field_name,
            value=value,
            suggestion=f"Use a value >= {min_value} for {field_name}",
        )
    if max_value is not None and value > max_value:
        raise ConfigValidationError(
            message=f"{field_name} must be <= {max_value}",
            field=field_name,
            value=value,
            suggestion=f"Use a value <= {max_value} for {field_name}",
        )
    return value


def validate_one_of(value: T, field_name: str, allowed_values: list[T]) -> T:
    """Validate that a value is one of the allowed values.

    Includes fuzzy matching to suggest corrections for typos.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        allowed_values: List of allowed values.

    Returns:
        The value if valid.

    Raises:
        ConfigValidationError: If the value is not in the allowed list.

    Example:
        >>> validate_one_of("tensor", "backend", ["dtensor", "megatron"])
        # Raises with suggestion: "Did you mean 'dtensor'?"
    """
    if value not in allowed_values:
        suggestion = f"Choose one of: {', '.join(repr(v) for v in allowed_values)}"

        # Try fuzzy matching if value is a string
        if isinstance(value, str):
            str_options = [str(v) for v in allowed_values]
            match = _try_fuzzy_match(value, str_options)
            if match:
                suggestion = f"Did you mean '{match}'? Valid options: {', '.join(repr(v) for v in allowed_values)}"

        raise ConfigValidationError(
            message=f"Invalid value for '{field_name}'",
            field=field_name,
            value=value,
            suggestion=suggestion,
        )
    return value


def validate_probability(value: float, field_name: str) -> float:
    """Validate that a value is a valid probability (0 <= value <= 1).

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.

    Returns:
        The value if valid.

    Raises:
        ConfigValidationError: If the value is not a valid probability.
    """
    return validate_range(value, field_name, 0.0, 1.0)


def validate_power_of_two(value: int, field_name: str) -> int:
    """Validate that a value is a power of two.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.

    Returns:
        The value if valid.

    Raises:
        ConfigValidationError: If the value is not a power of two.
    """
    if value <= 0 or (value & (value - 1)) != 0:
        raise ConfigValidationError(
            message=f"{field_name} must be a power of two",
            field=field_name,
            value=value,
            suggestion="Use a value like 1, 2, 4, 8, 16, 32, 64, etc.",
        )
    return value


def validate_divisible_by(value: int, field_name: str, divisor: int) -> int:
    """Validate that a value is divisible by a given divisor.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        divisor: The divisor to check.

    Returns:
        The value if valid.

    Raises:
        ConfigValidationError: If the value is not divisible by the divisor.
    """
    if value % divisor != 0:
        raise ConfigValidationError(
            message=f"{field_name} must be divisible by {divisor}",
            field=field_name,
            value=value,
            suggestion=f"Use a value divisible by {divisor} (e.g., {(value // divisor) * divisor} or {((value // divisor) + 1) * divisor})",
        )
    return value


def validate_config(config: Any) -> bool:
    """Validate a configuration object.

    This function can be used to validate any Pydantic-based configuration
    object and provides detailed error messages with suggestions.

    Args:
        config: The configuration object to validate.

    Returns:
        True if the configuration is valid.

    Raises:
        ConfigValidationError: If the configuration is invalid.

    Example:
        >>> from nemo_rl.config import GRPOConfig
        >>> config = GRPOConfig(policy={"model_name": "test"})
        >>> validate_config(config)  # Returns True if valid
    """
    try:
        # Try to revalidate the model to catch any issues
        if hasattr(config, "model_validate"):
            config.model_validate(config.model_dump())
        return True
    except ValidationError as e:
        errors = e.errors()
        if errors:
            first_error = errors[0]
            field = ".".join(str(loc) for loc in first_error["loc"])
            message = first_error["msg"]
            value = first_error.get("input")
            error_type = first_error.get("type", "")

            # Generate context-aware suggestion
            suggestion = _generate_validation_suggestion(error_type, field, value)

            raise ConfigValidationError(
                message=f"Configuration validation failed: {message}",
                field=field,
                value=value,
                suggestion=suggestion,
            ) from e
        raise ConfigValidationError(
            message=f"Configuration validation failed: {e}",
        ) from e


def _generate_validation_suggestion(error_type: str, field: str, value: Any) -> str | None:
    """Generate a helpful suggestion based on error context.

    Args:
        error_type: Type of Pydantic validation error.
        field: Name of the field that failed validation.
        value: The invalid value that was provided.

    Returns:
        A helpful suggestion string or None.
    """
    # Known field-specific suggestions
    known_fields = {
        "backend": (["dtensor", "megatron"], "training backend"),
        "precision": (["float32", "float16", "bfloat16"], "training precision"),
        "log_level": (["debug", "info", "warning", "error"], "logging level"),
    }

    # Check if field matches known fields (including nested paths)
    for known_field, (options, description) in known_fields.items():
        if field == known_field or field.endswith(f".{known_field}"):
            if isinstance(value, str):
                match = _try_fuzzy_match(value, options)
                if match:
                    return f"Did you mean '{match}'? Valid {description}s: {', '.join(repr(o) for o in options)}"
            return f"Valid {description}s: {', '.join(repr(o) for o in options)}"

    # Generic suggestions based on error type
    type_suggestions = {
        "greater_than": "Value must be greater than the minimum. Try a larger value.",
        "greater_than_equal": "Value must be greater than or equal to the minimum.",
        "less_than": "Value must be less than the maximum. Try a smaller value.",
        "less_than_equal": "Value must be less than or equal to the maximum.",
        "missing": "This field is required. Please provide a value.",
        "string_type": "Value must be a string (text in quotes).",
        "int_type": "Value must be an integer (whole number).",
        "float_type": "Value must be a number.",
        "bool_type": "Value must be true or false.",
        "list_type": "Value must be a list: [item1, item2, ...]",
        "dict_type": "Value must be a dictionary: {\"key\": value, ...}",
    }

    return type_suggestions.get(error_type)


def validate_model_name(value: str, field_name: str = "model_name") -> str:
    """Validate a model name is non-empty and reasonably formatted.

    Args:
        value: The model name to validate.
        field_name: Name of the field for error messages.

    Returns:
        The validated model name.

    Raises:
        ConfigValidationError: If the model name is invalid.
    """
    if not value or not value.strip():
        raise ConfigValidationError(
            message=f"Invalid model name: '{field_name}' cannot be empty",
            field=field_name,
            value=value,
            suggestion="Provide a valid HuggingFace model name (e.g., 'Qwen/Qwen2.5-1.5B') or local path",
        )
    return value.strip()


def validate_path_exists(value: str, field_name: str, must_exist: bool = False) -> str:
    """Validate a file or directory path.

    Args:
        value: The path to validate.
        field_name: Name of the field for error messages.
        must_exist: If True, validate that the path exists.

    Returns:
        The validated path.

    Raises:
        ConfigValidationError: If the path is invalid.
    """
    from pathlib import Path

    if not value or not value.strip():
        raise ConfigValidationError(
            message=f"Invalid path: '{field_name}' cannot be empty",
            field=field_name,
            value=value,
            suggestion="Provide a valid file or directory path",
        )

    path = Path(value.strip())
    if must_exist and not path.exists():
        raise ConfigValidationError(
            message=f"Path does not exist: {value}",
            field=field_name,
            value=value,
            suggestion=f"Check that the path '{value}' exists and is accessible",
        )

    return value.strip()


def validate_batch_size_consistency(
    global_batch: int,
    micro_batch: int,
    num_gpus: int,
    field_prefix: str = "",
) -> None:
    """Validate that batch sizes are consistent with GPU count.

    Args:
        global_batch: Global batch size.
        micro_batch: Micro batch size per GPU.
        num_gpus: Number of GPUs.
        field_prefix: Prefix for field names in error messages.

    Raises:
        ConfigValidationError: If batch sizes are inconsistent.
    """
    if global_batch < micro_batch:
        raise ConfigValidationError(
            message="Global batch size cannot be smaller than micro batch size",
            field=f"{field_prefix}train_global_batch_size" if field_prefix else "train_global_batch_size",
            value=global_batch,
            suggestion=f"Either increase global_batch_size to at least {micro_batch} or decrease micro_batch_size",
        )

    if global_batch % (micro_batch * num_gpus) != 0:
        ideal_global = micro_batch * num_gpus
        suggestion_multiples = [ideal_global * i for i in range(1, 5)]
        raise ConfigValidationError(
            message=f"Global batch size ({global_batch}) must be divisible by micro_batch_size ({micro_batch}) × num_gpus ({num_gpus}) = {micro_batch * num_gpus}",
            field=f"{field_prefix}train_global_batch_size" if field_prefix else "train_global_batch_size",
            value=global_batch,
            suggestion=f"Try one of: {', '.join(str(s) for s in suggestion_multiples)}",
        )


def validate_parallelism(
    tensor_parallel: int,
    pipeline_parallel: int,
    num_gpus: int,
    field_prefix: str = "",
) -> None:
    """Validate parallelism configuration is consistent with GPU count.

    Args:
        tensor_parallel: Tensor parallel size.
        pipeline_parallel: Pipeline parallel size.
        num_gpus: Total number of GPUs.
        field_prefix: Prefix for field names in error messages.

    Raises:
        ConfigValidationError: If parallelism configuration is invalid.
    """
    total_parallel = tensor_parallel * pipeline_parallel

    if total_parallel > num_gpus:
        raise ConfigValidationError(
            message=f"Parallelism exceeds available GPUs: tensor_parallel ({tensor_parallel}) × pipeline_parallel ({pipeline_parallel}) = {total_parallel} > {num_gpus} GPUs",
            field=f"{field_prefix}tensor_parallel_size" if field_prefix else "tensor_parallel_size",
            value=tensor_parallel,
            suggestion=f"Reduce tensor_parallel_size or pipeline_parallel_size so their product ≤ {num_gpus}",
        )

    if num_gpus % total_parallel != 0:
        raise ConfigValidationError(
            message=f"Number of GPUs ({num_gpus}) must be divisible by total parallelism ({total_parallel})",
            field=f"{field_prefix}tensor_parallel_size" if field_prefix else "tensor_parallel_size",
            value=tensor_parallel,
            suggestion="Adjust parallelism settings so total_parallel divides num_gpus evenly",
        )


__all__ = [
    "ConfigValidationError",
    "validate_config",
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_one_of",
    "validate_probability",
    "validate_power_of_two",
    "validate_divisible_by",
    "validate_model_name",
    "validate_path_exists",
    "validate_batch_size_consistency",
    "validate_parallelism",
]
