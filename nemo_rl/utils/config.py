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

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Optional, Union, cast, get_type_hints

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import TypeAdapter


def resolve_path(base_path: Path, path: str) -> Path:
    """Resolve a path relative to the base path."""
    if path.startswith("/"):
        return Path(path)
    return base_path / path


def merge_with_override(
    base_config: DictConfig, override_config: DictConfig
) -> DictConfig:
    """Merge configs with support for _override_ marker to completely override sections."""
    for key in list(override_config.keys()):
        if isinstance(override_config[key], DictConfig):
            if override_config[key].get("_override_", False):
                # remove the _override_ marker
                override_config[key].pop("_override_")
                # remove the key from base_config so it won't be merged
                if key in base_config:
                    base_config.pop(key)

    merged_config = cast(DictConfig, OmegaConf.merge(base_config, override_config))
    return merged_config


def load_config_with_inheritance(
    config_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> DictConfig:
    """Load a config file with inheritance support.

    Args:
        config_path: Path to the config file
        base_dir: Base directory for resolving relative paths. If None, uses config_path's directory

    Returns:
        Merged config dictionary
    """
    config_path = Path(config_path)
    if base_dir is None:
        base_dir = config_path.parent
    base_dir = Path(base_dir)

    config = OmegaConf.load(config_path)
    assert isinstance(config, DictConfig), (
        "Config must be a Dictionary Config (List Config not supported)"
    )

    # Handle inheritance
    if "defaults" in config:
        defaults = config.pop("defaults")
        if isinstance(defaults, (str, Path)):
            defaults = [defaults]
        elif isinstance(defaults, ListConfig):
            defaults = [str(d) for d in defaults]

        # Load and merge all parent configs
        base_config = OmegaConf.create({})
        for default in defaults:
            parent_path = resolve_path(base_dir, str(default))
            # Use parent's directory as base_dir for resolving its own defaults
            parent_config = load_config_with_inheritance(
                parent_path, parent_path.parent
            )
            base_config = cast(
                DictConfig, merge_with_override(base_config, parent_config)
            )

        # Merge with current config
        config = cast(DictConfig, merge_with_override(base_config, config))

    return config


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load a config file with inheritance support and convert it to an OmegaConf object.

    The config inheritance system supports:

    1. Single inheritance:
        ```yaml
        # child.yaml
        defaults: parent.yaml
        common:
          value: 43
        ```

    2. Multiple inheritance:
        ```yaml
        # child.yaml
        defaults:
          - parent1.yaml
          - parent2.yaml
        common:
          value: 44
        ```

    3. Nested inheritance:
        ```yaml
        # parent.yaml
        defaults: grandparent.yaml
        common:
          value: 43

        # child.yaml
        defaults: parent.yaml
        common:
          value: 44
        ```

    4. Variable interpolation:
        ```yaml
        # parent.yaml
        base_value: 42
        derived:
          value: ${base_value}

        # child.yaml
        defaults: parent.yaml
        base_value: 43  # This will update both base_value and derived.value
        ```

    The system handles:
    - Relative and absolute paths
    - Multiple inheritance
    - Nested inheritance
    - Variable interpolation

    The inheritance is resolved depth-first, with later configs overriding earlier ones.
    This means in multiple inheritance, the last config in the list takes precedence.

    Args:
        config_path: Path to the config file

    Returns:
        Merged config dictionary
    """
    return load_config_with_inheritance(config_path)


class OverridesError(Exception):
    """Custom exception for Hydra override parsing errors."""

    pass


def parse_hydra_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    """Parse and apply Hydra overrides to an OmegaConf config.

    Args:
        cfg: OmegaConf config to apply overrides to
        overrides: List of Hydra override strings

    Returns:
        Updated config with overrides applied

    Raises:
        OverridesError: If there's an error parsing or applying overrides
    """
    try:
        OmegaConf.set_struct(cfg, True)
        parser = OverridesParser.create()
        parsed = parser.parse_overrides(overrides=overrides)
        ConfigLoaderImpl._apply_overrides_to_config(overrides=parsed, cfg=cfg)
        return cfg
    except Exception as e:
        raise OverridesError(f"Failed to parse Hydra overrides: {str(e)}") from e


def _merge_extras_back(
    validated: dict[str, Any], user: dict[str, Any]
) -> dict[str, Any]:
    """Merge extra keys from *user* (not in the schema) back into *validated*.

    Pydantic with ``extra='ignore'`` drops unknown keys during validation.
    This function restores them so that user forks, deprecated YAML fields,
    and any keys not yet covered by a schema survive the round-trip.

    Recurses into nested dicts so that extra keys at every level are preserved.
    """
    result = validated.copy()
    for k, v in user.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_extras_back(result[k], v)
        elif k not in result:
            result[k] = v  # carry forward unknown key as-is
    return result


def validate_config(user_config: dict[str, Any], schema: type) -> dict[str, Any]:
    """Validate *user_config* against a pydantic dataclass *schema* and fill defaults.

    Implements the "left join" pattern proposed in #2102:

    1. **Validate** — ``TypeAdapter(schema).validate_python(user_config)`` checks
       types of known fields and raises ``pydantic.ValidationError`` on mismatch.
       Pydantic also coerces compatible types (e.g. ``"42"`` → ``42``).
    2. **Fill defaults** — ``dataclasses.asdict()`` on the validated instance
       produces a dict where every field with a default has a value.
    3. **Preserve extras** — ``_merge_extras_back()`` restores any keys from
       *user_config* that were not in the schema (dropped by ``extra='ignore'``).

    Priority order: **CLI overrides > YAML values > dataclass defaults**.
    Unknown keys are always preserved for backward compatibility.

    Args:
        user_config: Plain dict from ``OmegaConf.to_container(resolve=True)``.
        schema: A **pydantic** dataclass class (decorated with
            ``@pydantic.dataclasses.dataclass(config=ConfigDict(extra='ignore'))``).

    Returns:
        A new dict — known fields validated and defaulted, extra keys preserved.

    Raises:
        pydantic.ValidationError: If a known field has the wrong type.
    """
    ta = TypeAdapter(schema)
    validated = ta.validate_python(user_config)
    validated_dict = dataclasses.asdict(validated)
    return _merge_extras_back(validated_dict, user_config)


def apply_config_defaults(config: dict[str, Any], defaults_cls: type) -> dict[str, Any]:
    """Deprecated — use :func:`validate_config` instead.

    Kept for backward compatibility.  Delegates to :func:`validate_config`
    when *defaults_cls* is a pydantic dataclass, otherwise falls back to the
    legacy recursive fill.
    """
    warnings.warn(
        "apply_config_defaults() is deprecated, use validate_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # If it's a pydantic dataclass (has __pydantic_config__), use the new path.
    if hasattr(defaults_cls, "__pydantic_config__"):
        return validate_config(config, defaults_cls)

    # Legacy fallback for plain stdlib dataclasses (should not happen in new code).
    if not dataclasses.is_dataclass(defaults_cls):
        return config

    hints = get_type_hints(defaults_cls)

    for f in dataclasses.fields(defaults_cls):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            continue

        if f.name not in config:
            if dataclasses.is_dataclass(default):
                config[f.name] = {}
                apply_config_defaults(config[f.name], type(default))
            else:
                config[f.name] = default
        elif isinstance(config[f.name], dict):
            nested_type = hints.get(f.name)
            if (
                nested_type is not None
                and isinstance(nested_type, type)
                and dataclasses.is_dataclass(nested_type)
            ):
                apply_config_defaults(config[f.name], nested_type)

    return config


def register_omegaconf_resolvers() -> None:
    """Register shared OmegaConf resolvers used in configs."""
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    if not OmegaConf.has_resolver("max"):
        OmegaConf.register_new_resolver("max", lambda a, b: max(a, b))
