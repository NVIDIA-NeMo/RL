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
from pathlib import Path
from typing import Any, Optional, Union, cast, get_type_hints

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig, OmegaConf


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


def apply_config_defaults(config: dict[str, Any], defaults_cls: type) -> dict[str, Any]:
    """Recursively fill missing config keys from a dataclass's default values.

    Only keys that are absent from ``config`` are filled.  Existing keys
    (including ``None`` and ``False``) are never overwritten.  This ensures
    that YAML and CLI values always take precedence over dataclass defaults.

    Args:
        config: The loaded config dict (already resolved).
        defaults_cls: A dataclass class whose fields define defaults.

    Returns:
        The same config dict, mutated in place, with missing keys filled.
    """
    if not dataclasses.is_dataclass(defaults_cls):
        return config

    hints = get_type_hints(defaults_cls)

    for f in dataclasses.fields(defaults_cls):
        # Determine the concrete default for this field.
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            # No default — field is required and must come from YAML.
            continue

        if f.name not in config:
            if dataclasses.is_dataclass(default):
                # Nested defaults: create an empty dict and fill recursively.
                config[f.name] = {}
                apply_config_defaults(config[f.name], type(default))
            else:
                config[f.name] = default
        elif isinstance(config[f.name], dict):
            # Recurse into nested dataclass defaults.
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
