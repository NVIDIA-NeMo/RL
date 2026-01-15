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
"""Base configuration class using Pydantic for runtime validation.

This module provides the foundation for all NeMo RL configurations with:
- Type-safe configuration with runtime validation
- Support for YAML, JSON, and Python dict loading
- Clear error messages for invalid configurations
- Immutable configurations after creation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

T = TypeVar("T", bound="BaseConfig")


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors.

    Provides detailed error messages with field names, invalid values,
    and suggestions for fixing the issue.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        suggestion: str | None = None,
    ):
        self.field = field
        self.value = value
        self.suggestion = suggestion

        # Build detailed error message
        error_parts = [message]
        if field:
            error_parts.append(f"Field: {field}")
        if value is not None:
            error_parts.append(f"Value: {value!r}")
        if suggestion:
            error_parts.append(f"Suggestion: {suggestion}")

        super().__init__("\n".join(error_parts))


class BaseConfig(BaseModel):
    """Base configuration class with validation and serialization support.

    All NeMo RL configuration classes should inherit from this base class
    to get consistent validation, serialization, and error handling.

    Features:
        - Pydantic-based runtime validation
        - Type annotations with validation
        - Clear error messages at load time
        - Support for YAML, JSON, and dict loading
        - Immutable after creation (frozen)

    Example:
        >>> class MyConfig(BaseConfig):
        ...     learning_rate: float = 1e-4
        ...     batch_size: int = 32
        ...
        >>> config = MyConfig(learning_rate=1e-3)
        >>> config.batch_size
        32
    """

    model_config = ConfigDict(
        # Make configurations immutable after creation
        frozen=True,
        # Validate field values on assignment
        validate_assignment=True,
        # Allow extra fields to be ignored (for forward compatibility)
        extra="forbid",
        # Use enum values instead of names
        use_enum_values=True,
        # Validate default values
        validate_default=True,
        # Allow population by field name
        populate_by_name=True,
    )

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated configuration instance.

        Raises:
            ConfigValidationError: If the configuration is invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Failed to parse YAML file: {path}",
                suggestion=f"Check YAML syntax: {e}",
            ) from e

    @classmethod
    def from_json(cls: type[T], path: str | Path) -> T:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Validated configuration instance.

        Raises:
            ConfigValidationError: If the configuration is invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ConfigValidationError(
                f"Failed to parse JSON file: {path}",
                suggestion=f"Check JSON syntax: {e}",
            ) from e

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create configuration from a Python dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            Validated configuration instance.

        Raises:
            ConfigValidationError: If the configuration is invalid.
        """
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            # Convert Pydantic validation error to our custom error
            errors = e.errors()
            if errors:
                first_error = errors[0]
                field = ".".join(str(loc) for loc in first_error["loc"])
                message = first_error["msg"]
                value = first_error.get("input")
                error_type = first_error.get("type", "")

                # Generate helpful suggestions based on error type and field
                suggestion = _generate_suggestion(first_error, field)

                # Try fuzzy matching for enum-like errors
                if error_type in ("enum", "literal_error") and isinstance(value, str):
                    valid_options = _get_valid_options_for_field(field)
                    if valid_options:
                        try:
                            from nemo_rl.utils.errors import fuzzy_match
                            match = fuzzy_match(value, valid_options)
                            if match:
                                suggestion = f"Did you mean '{match}'? Valid options: {', '.join(repr(o) for o in valid_options)}"
                        except ImportError:
                            suggestion = f"Valid options: {', '.join(repr(o) for o in valid_options)}"

                # Build a more detailed error message
                error_details = [f"Configuration validation failed: {message}"]
                error_details.append(f"Field: {field}")
                if value is not None:
                    error_details.append(f"Value: {value!r}")
                if suggestion:
                    error_details.append(f"Suggestion: {suggestion}")

                raise ConfigValidationError(
                    message="\n".join(error_details),
                    field=field,
                    value=value,
                    suggestion=suggestion,
                ) from e
            raise ConfigValidationError(
                message=f"Configuration validation failed: {e}",
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a Python dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return self.model_dump()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML configuration file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the JSON configuration file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _generate_suggestion(error: dict[str, Any], field: str | None = None) -> str | None:
    """Generate a helpful suggestion based on the validation error type.

    Args:
        error: Pydantic validation error dictionary.
        field: Optional field name for context-aware suggestions.

    Returns:
        Suggestion string or None if no suggestion is available.
    """
    error_type = error.get("type", "")
    ctx = error.get("ctx", {})
    input_value = error.get("input")

    # Known field-specific suggestions
    field_suggestions = {
        "backend": {
            "valid_options": ["dtensor", "megatron"],
            "doc_topic": "backends",
        },
        "precision": {
            "valid_options": ["float32", "float16", "bfloat16"],
            "doc_topic": "config",
        },
        "log_level": {
            "valid_options": ["debug", "info", "warning", "error"],
        },
        "optimizer.name": {
            "valid_options": ["adamw", "adam", "sgd", "adafactor"],
        },
        "scheduler.name": {
            "valid_options": ["cosine", "linear", "constant", "constant_with_warmup"],
        },
    }

    # Check for fuzzy matching on enum-like errors
    if error_type in ("enum", "literal_error") and input_value is not None:
        expected = ctx.get("expected", "")
        # Try to extract valid options from expected string
        valid_options = None

        # Check field-specific options first
        if field and field in field_suggestions:
            valid_options = field_suggestions[field].get("valid_options")

        # Try fuzzy matching if we have valid options and a string input
        if valid_options and isinstance(input_value, str):
            try:
                from nemo_rl.utils.errors import fuzzy_match
                match = fuzzy_match(input_value, valid_options)
                if match:
                    return f"Did you mean '{match}'? Valid options: {', '.join(repr(o) for o in valid_options)}"
            except ImportError:
                pass

        if valid_options:
            return f"Value must be one of: {', '.join(repr(o) for o in valid_options)}"
        return f"Value must be {expected}"

    if error_type == "greater_than":
        gt_value = ctx.get("gt", "the minimum")
        return f"Value must be greater than {gt_value}. Try a value like {float(gt_value) * 1.1 if isinstance(gt_value, (int, float)) else gt_value}"
    elif error_type == "greater_than_equal":
        ge_value = ctx.get("ge", "the minimum")
        return f"Value must be greater than or equal to {ge_value}"
    elif error_type == "less_than":
        lt_value = ctx.get("lt", "the maximum")
        return f"Value must be less than {lt_value}. Try a value like {float(lt_value) * 0.9 if isinstance(lt_value, (int, float)) else lt_value}"
    elif error_type == "less_than_equal":
        le_value = ctx.get("le", "the maximum")
        return f"Value must be less than or equal to {le_value}"
    elif error_type == "missing":
        # Provide specific guidance for required fields
        if field:
            if "model_name" in field:
                return "This field is required. Provide a HuggingFace model name (e.g., 'Qwen/Qwen2.5-1.5B')"
            elif "train_path" in field or "data" in field:
                return "This field is required. Provide a path to your training data file"
        return "This field is required and must be provided"
    elif error_type == "string_type":
        if isinstance(input_value, (int, float)):
            return f"Value must be a string, not a number. Try wrapping in quotes: \"{input_value}\""
        return "Value must be a string (text in quotes)"
    elif error_type == "int_type":
        if isinstance(input_value, float):
            return f"Value must be an integer, not a float. Try: {int(input_value)}"
        elif isinstance(input_value, str):
            try:
                return f"Value must be an integer. Try: {int(input_value)}"
            except ValueError:
                pass
        return "Value must be a whole number (integer)"
    elif error_type == "float_type":
        if isinstance(input_value, str):
            try:
                return f"Value must be a number. Try: {float(input_value)}"
            except ValueError:
                pass
        return "Value must be a number (float)"
    elif error_type == "bool_type":
        if isinstance(input_value, str):
            if input_value.lower() in ("true", "yes", "1"):
                return "Value must be a boolean. Use 'true' (without quotes) or True"
            elif input_value.lower() in ("false", "no", "0"):
                return "Value must be a boolean. Use 'false' (without quotes) or False"
        return "Value must be a boolean (true/false)"
    elif error_type == "list_type":
        return "Value must be a list. Use square brackets: [item1, item2, ...]"
    elif error_type == "dict_type":
        return "Value must be a dictionary/object. Use curly braces: {\"key\": value, ...}"
    elif "url" in error_type:
        return "Value must be a valid URL (e.g., 'https://example.com')"
    elif error_type == "value_error":
        # Custom value errors often have messages in ctx
        custom_msg = ctx.get("message", ctx.get("error", ""))
        if custom_msg:
            return str(custom_msg)
        return "Value does not meet the validation requirements"
    elif error_type == "extra_forbidden":
        return f"Unknown field '{field}'. Check spelling or see documentation for valid fields"

    return None


def _get_valid_options_for_field(field: str) -> list[str] | None:
    """Get valid options for known enum-like fields.

    Args:
        field: Field name to look up.

    Returns:
        List of valid options or None.
    """
    field_options = {
        "backend": ["dtensor", "megatron"],
        "precision": ["float32", "float16", "bfloat16"],
        "log_level": ["debug", "info", "warning", "error"],
        "model_save_format": ["safetensors", "torch_save"],
        "lr_decay_style": ["cosine", "linear", "constant"],
        "dropout_position": ["pre", "post"],
        "data_parallel_sharding_strategy": ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
    }

    # Check direct match
    if field in field_options:
        return field_options[field]

    # Check if field ends with any known suffix
    for known_field, options in field_options.items():
        if field.endswith(known_field) or field.endswith(f".{known_field}"):
            return options

    return None
