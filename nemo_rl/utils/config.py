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

from pathlib import Path
from typing import Optional, Union, cast

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig, OmegaConf


def resolve_path(base_path: Path, path: str) -> Path:
    """Resolve a path relative to the base path."""
    if path.startswith("/"):
        return Path(path)
    return base_path / path


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
            parent_path = resolve_path(base_dir, default)
            parent_config = load_config_with_inheritance(parent_path, base_dir)
            base_config = cast(DictConfig, OmegaConf.merge(base_config, parent_config))

        # Merge with current config
        config = cast(DictConfig, OmegaConf.merge(base_config, config))

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
