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
"""Deprecation warning utilities for NeMo RL backward compatibility.

This module provides utilities for emitting deprecation warnings
when using old API patterns. Warnings include migration suggestions
and links to documentation.

Example:
    >>> from nemo_rl.compat.deprecation import deprecated_import
    >>> deprecated_import("GRPO", "GRPOTrainer", "nemo_rl.algorithms.grpo")
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

# Documentation URLs
MIGRATION_GUIDE_URL = "https://nvidia.github.io/nemo-rl/migration"
API_DOCS_URL = "https://nvidia.github.io/nemo-rl/api"

F = TypeVar("F", bound=Callable[..., Any])


def deprecated_import(
    old_name: str,
    new_name: str,
    new_module: str,
    removal_version: str = "2.0",
) -> None:
    """Emit a deprecation warning for an old import.

    Args:
        old_name: The old import name being used.
        new_name: The new name to use instead.
        new_module: The new module path.
        removal_version: Version when the old import will be removed.
    """
    warnings.warn(
        f"'{old_name}' is deprecated and will be removed in version {removal_version}. "
        f"Use '{new_name}' from '{new_module}' instead.\n"
        f"Migration guide: {MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=3,
    )


def deprecated_function(
    new_function: str,
    new_module: str,
    removal_version: str = "2.0",
    extra_message: str = "",
) -> Callable[[F], F]:
    """Decorator to mark a function as deprecated.

    Args:
        new_function: The new function name to use instead.
        new_module: The new module containing the replacement.
        removal_version: Version when the old function will be removed.
        extra_message: Additional message to include in the warning.

    Returns:
        Decorator function.

    Example:
        >>> @deprecated_function("train", "nemo_rl.api")
        ... def grpo_train(*args, **kwargs):
        ...     return new_train(*args, **kwargs)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg = (
                f"'{func.__name__}' is deprecated and will be removed in version {removal_version}. "
                f"Use '{new_function}' from '{new_module}' instead."
            )
            if extra_message:
                msg += f" {extra_message}"
            msg += f"\nMigration guide: {MIGRATION_GUIDE_URL}"

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def deprecated_class(
    new_class: str,
    new_module: str,
    removal_version: str = "2.0",
) -> Callable[[type], type]:
    """Decorator to mark a class as deprecated.

    The warning is emitted when the class is instantiated.

    Args:
        new_class: The new class name to use instead.
        new_module: The new module containing the replacement.
        removal_version: Version when the old class will be removed.

    Returns:
        Decorator that wraps the class.

    Example:
        >>> @deprecated_class("GRPOTrainer", "nemo_rl.algorithms.grpo")
        ... class GRPO:
        ...     pass
    """
    def decorator(cls: type) -> type:
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                f"'{cls.__name__}' is deprecated and will be removed in version {removal_version}. "
                f"Use '{new_class}' from '{new_module}' instead.\n"
                f"Migration guide: {MIGRATION_GUIDE_URL}",
                DeprecationWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init  # type: ignore
        return cls
    return decorator


def deprecated_method(
    new_method: str,
    removal_version: str = "2.0",
    extra_message: str = "",
) -> Callable[[F], F]:
    """Decorator to mark a method as deprecated.

    Args:
        new_method: The new method name to use instead.
        removal_version: Version when the old method will be removed.
        extra_message: Additional message to include in the warning.

    Returns:
        Decorator function.

    Example:
        >>> @deprecated_method("fit")
        ... def train(self, *args, **kwargs):
        ...     return self.fit(*args, **kwargs)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            msg = (
                f"'{func.__name__}' is deprecated and will be removed in version {removal_version}. "
                f"Use '{new_method}' instead."
            )
            if extra_message:
                msg += f" {extra_message}"
            msg += f"\nMigration guide: {MIGRATION_GUIDE_URL}"

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(self, *args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def deprecated_parameter(
    param_name: str,
    new_param: str | None = None,
    removal_version: str = "2.0",
) -> None:
    """Emit a deprecation warning for a parameter.

    Args:
        param_name: The deprecated parameter name.
        new_param: The new parameter name (if renamed), or None if removed.
        removal_version: Version when the parameter will be removed.
    """
    if new_param:
        msg = (
            f"Parameter '{param_name}' is deprecated and will be removed in version {removal_version}. "
            f"Use '{new_param}' instead."
        )
    else:
        msg = (
            f"Parameter '{param_name}' is deprecated and will be removed in version {removal_version}."
        )

    warnings.warn(msg, DeprecationWarning, stacklevel=3)


class DeprecatedAlias:
    """Create a deprecated alias for a class or module attribute.

    This class allows creating lazy aliases that emit deprecation warnings
    when accessed.

    Example:
        >>> GRPO = DeprecatedAlias(GRPOTrainer, "GRPO", "GRPOTrainer", "nemo_rl.algorithms.grpo")
        >>> grpo = GRPO(config)  # Emits deprecation warning
    """

    def __init__(
        self,
        target: Any,
        old_name: str,
        new_name: str,
        new_module: str,
        removal_version: str = "2.0",
    ):
        """Initialize the deprecated alias.

        Args:
            target: The actual object being aliased.
            old_name: The deprecated name.
            new_name: The new name to use.
            new_module: The module containing the new name.
            removal_version: Version when the alias will be removed.
        """
        self._target = target
        self._old_name = old_name
        self._new_name = new_name
        self._new_module = new_module
        self._removal_version = removal_version

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the target, emitting a deprecation warning."""
        warnings.warn(
            f"'{self._old_name}' is deprecated and will be removed in version {self._removal_version}. "
            f"Use '{self._new_name}' from '{self._new_module}' instead.\n"
            f"Migration guide: {MIGRATION_GUIDE_URL}",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Get attribute from target, emitting a deprecation warning on first access."""
        if name.startswith("_"):
            raise AttributeError(name)
        warnings.warn(
            f"'{self._old_name}' is deprecated and will be removed in version {self._removal_version}. "
            f"Use '{self._new_name}' from '{self._new_module}' instead.\n"
            f"Migration guide: {MIGRATION_GUIDE_URL}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self._target, name)


__all__ = [
    "deprecated_import",
    "deprecated_function",
    "deprecated_class",
    "deprecated_method",
    "deprecated_parameter",
    "DeprecatedAlias",
    "MIGRATION_GUIDE_URL",
    "API_DOCS_URL",
]
