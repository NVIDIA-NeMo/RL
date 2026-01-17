#!/usr/bin/env -S uv run --script -q
# /// script
# dependencies = [
#   "omegaconf"
# ]
# ///
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
"""Utilities for working with YAML configs in this repo.

Subcommands:
  - expand: Resolve a config with OmegaConf interpolation and inheritance.
  - minimize: Remove keys in the config that are equal to what it inherits from
    its `defaults` chain. By default, the base is inferred from the config's
    `defaults` key (which must be a string, not a list). Optionally, pass
    --base to override and rebase the config to a different parent.
  - minimize-check: Same args as `minimize` but only checks if minimization
    would change the file; exits non-zero if changes are needed.

The `expand` and `minimize` commands support printing to stdout or in-place editing of the config file.

Example:
  # Expand a config with a root level "defaults" key to see the full config; print to stdout
  tools/config_cli.py expand examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2.yaml

  # Expand a config with a root level "defaults" key to see the full config; edit the config in place
  tools/config_cli.py expand examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2.yaml --in-place

  # Minimize a config (base inferred from its defaults key); print to stdout
  tools/config_cli.py minimize examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2.yaml

  # Minimize a config in place
  tools/config_cli.py minimize examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2.yaml --in-place

  # Minimize with explicit base (rebase to a different parent)
  tools/config_cli.py minimize examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2.yaml --base examples/configs/dpo.yaml --in-place

  # Minimize all configs:
  for recipe in examples/configs/recipes/{llm,vlm}/*.yaml; do
    if ! tools/config_cli.py minimize-check $recipe 2>/dev/null; then
      tools/config_cli.py minimize $recipe --in-place
    fi
  done

  # Compare two configs
  tools/config_cli.py compare examples/configs/grpo_math_1B.yaml examples/configs/grpo_math_8B.yaml
"""

import argparse
import sys
from pathlib import Path

# ============================================================================
# VENDORED SECTION: Minimal self-contained config loader (no nemo_rl dependency)
#
# Original source: `nemo_rl/utils/config.py`
#   - Functions adapted: `resolve_path`, `load_config_with_inheritance`, `load_config`
#   - Purpose: avoid importing from nemo_rl so this script is standalone
#   - If upstream changes, consider updating this vendored block accordingly
# ============================================================================
from typing import Any, Iterable, Optional, Union, cast

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
            parent_path = resolve_path(base_dir, str(default))
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


# ============================================================================
# END VENDORED SECTION
# ============================================================================


def _dict_like(obj: Any) -> bool:
    return isinstance(obj, dict)


def _list_like(obj: Any) -> bool:
    return isinstance(obj, list)


REMOVE = object()


def _prune_equal(a: Any, b: Any) -> Any:
    """Return a copy of `a` with entries equal to `b` removed.

    - If both are dicts: recursively prune and drop keys whose subtree is empty
      after pruning or equal.
    - If both are lists of same length: recursively prune by index and drop list
      if becomes entirely empty or equal.
    - Else: if equal, return a sentinel indicating removal; otherwise return `a`.
    """
    if _dict_like(a) and _dict_like(b):
        out: dict[str, Any] = {}
        a_dict: dict[str, Any] = a  # type: ignore[assignment]
        b_dict: dict[str, Any] = b  # type: ignore[assignment]
        for key, a_val in a_dict.items():
            if key in b_dict:
                pruned = _prune_equal(a_val, b_dict[key])
                if pruned is REMOVE:
                    # equal, skip
                    continue
                # keep if subtree has content
                if pruned != {} and pruned != []:
                    out[key] = pruned
            else:
                out[key] = a_val
        return out

    if _list_like(a) and _list_like(b) and len(a) == len(b):
        # Only remove if entire list equals base; avoid partial list pruning
        # to prevent semantic changes in ordered config sections.
        if a == b:
            return REMOVE
        return a

    # Base types
    if a == b:
        return REMOVE
    return a


def _ensure_defaults_relative(
    child_path: Path, base_path: Path, child_cfg: dict[str, Any]
) -> None:
    """Ensure `defaults:` points to the base, with a path relative to the base config file.

    The path we store must be a string such that, when the resulting minimized
    config sits at `child_path`, the `defaults` string references the base
    config location. The instruction asks that the defaults path in the resulting
    config is relative to the base config; we interpret this as "express `base`
    relative to the directory of the base file", then make that path relative
    to the child config so that hydra resolution works from the child file.
    """
    # Compute a relative reference from child dir to base file
    import os

    rel_from_child_to_base = os.path.relpath(
        str(base_path), start=str(child_path.parent)
    )

    existing = child_cfg.get("defaults")
    if existing is None:
        child_cfg["defaults"] = str(rel_from_child_to_base)
        return
    # Normalize various forms: string, single list element, list
    if isinstance(existing, str):
        existing_list: list[Any] = [existing]
    else:
        existing_list = list(existing) if isinstance(existing, Iterable) else [existing]
    # Put our base at the first position if not present
    if str(rel_from_child_to_base) not in [str(x) for x in existing_list]:
        existing_list.insert(0, str(rel_from_child_to_base))
    # If it's a single element list, collapse to string for this repo's style
    if len(existing_list) == 1:
        child_cfg["defaults"] = existing_list[0]
    else:
        child_cfg["defaults"] = existing_list


def expand(args: argparse.Namespace) -> int:
    # Merge defaults/inheritance using repo loader; preserve ${...}
    cfg = load_config(str(Path(args.config).resolve()))
    # Preserve ${...} by not resolving
    text = OmegaConf.to_yaml(cfg)
    if args.in_place:
        Path(args.config).write_text(text)
    else:
        print(text + ("\n" if not text.endswith("\n") else ""), end="")
    return 0


def _infer_base_from_defaults(child_path: Path, child_cfg_raw: DictConfig) -> Path:
    """Infer the base config path from the child's defaults key.

    Args:
        child_path: Resolved path to the child config
        child_cfg_raw: Raw loaded child config

    Returns:
        Resolved path to the base config

    Raises:
        ValueError: If defaults is missing, not a string, or is a list
    """
    defaults = child_cfg_raw.get("defaults")
    if defaults is None:
        raise ValueError(
            f"Config {child_path} has no 'defaults' key. "
            "Either add a 'defaults' key or use --base to specify the base config."
        )
    if not isinstance(defaults, str):
        raise ValueError(
            f"Config {child_path} has 'defaults' as a list, but only string defaults are "
            "supported for minimize. Please simplify to a single defaults entry or use --base."
        )
    return (child_path.parent / defaults).resolve()


def minimize(args: argparse.Namespace) -> int:
    child_path = Path(args.config).resolve()

    child_cfg_raw = OmegaConf.load(child_path)
    if not isinstance(child_cfg_raw, DictConfig):
        raise TypeError(
            f"Config at {child_path} must be a mapping (DictConfig), got {type(child_cfg_raw)}"
        )

    # Determine base: from --base arg or infer from defaults
    if args.base:
        base_path = Path(args.base).resolve()
        base_inferred = False
        # Load raw base for comparison
        base_cfg_raw = OmegaConf.load(base_path)
        if not isinstance(base_cfg_raw, DictConfig):
            raise TypeError(
                f"Config at {base_path} must be a mapping (DictConfig), got {type(base_cfg_raw)}"
            )
        base_resolved = OmegaConf.to_container(base_cfg_raw)
    else:
        base_path = _infer_base_from_defaults(child_path, child_cfg_raw)
        base_inferred = True
        # Load EXPANDED base (full inheritance chain) for proper comparison
        base_resolved = OmegaConf.to_container(load_config(str(base_path)))

    # Get child's explicit values (without defaults key for comparison)
    child_resolved = OmegaConf.to_container(child_cfg_raw)

    if not isinstance(child_resolved, dict) or not isinstance(base_resolved, dict):
        raise TypeError("Both child and base configs must be mappings after resolution")

    # Remove defaults from child before pruning (we'll handle it separately)
    child_defaults = child_resolved.pop("defaults", None)

    pruned = _prune_equal(child_resolved, base_resolved)

    # Ensure mapping output
    if pruned is None or not isinstance(pruned, dict):
        pruned = {} if pruned is None else {"value": pruned}

    if base_inferred:
        # Keep the existing defaults as-is
        if child_defaults is not None:
            pruned = {"defaults": child_defaults, **pruned}
    else:
        # Explicit base: update defaults to point to the new base
        _ensure_defaults_relative(child_path, base_path, pruned)
        if "defaults" in pruned:
            pruned = {"defaults": pruned["defaults"], **pruned}

    # Emit
    text = OmegaConf.to_yaml(OmegaConf.create(pruned))
    if args.in_place:
        Path(args.config).write_text(text)
    else:
        print(text + ("\n" if not text.endswith("\n") else ""), end="")
    return 0


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            key = f"{prefix}[{i}]"
            out.update(_flatten(v, key))
    else:
        out[prefix] = d
    return out


def compare(args: argparse.Namespace) -> int:
    left_path = Path(args.left).resolve()
    right_path = Path(args.right).resolve()

    # Expand via repo loader, then convert to plain dict/list so _flatten works
    left = OmegaConf.to_container(load_config(str(left_path)))  # type: ignore[assignment]
    right = OmegaConf.to_container(load_config(str(right_path)))  # type: ignore[assignment]

    lf = _flatten(left)
    rf = _flatten(right)

    left_keys = set(lf.keys())
    right_keys = set(rf.keys())

    added = sorted(right_keys - left_keys)
    removed = sorted(left_keys - right_keys)
    common = sorted(left_keys & right_keys)

    changed: list[str] = []
    for k in common:
        if lf[k] != rf[k]:
            changed.append(k)

    if not added and not removed and not changed:
        print("Configs are identical after expansion")
        return 0

    # Print concise report with explicit left/right context
    print("Comparing configs after expansion:")
    print(f"  Left : {left_path}")
    print(f"  Right: {right_path}")

    if added:
        print("\nAdded in Right (missing in Left):")
        for k in added:
            print(f"  {k} = {rf[k]}")

    if removed:
        print("\nRemoved in Right (only in Left):")
        for k in removed:
            print(f"  {k} = {lf[k]}")

    if changed:
        print("\nChanged (Left -> Right):")
        for k in changed:
            print(f"  {k}: {lf[k]} -> {rf[k]}")
    return 0


def minimize_check(args: argparse.Namespace) -> int:
    """Check if minimizing would change the file. Exit non-zero if so.

    Args (same as `minimize`):
      config: Child config path
      base: Optional base config path (inferred from defaults if not provided)
    """
    child_path = Path(args.config).resolve()

    # Compute minimized text (same as minimize())
    child_cfg_raw = OmegaConf.load(child_path)
    if not isinstance(child_cfg_raw, DictConfig):
        print(
            f"[minimize-check] Config must be a mapping: {child_path}",
            file=sys.stderr,
        )
        return 2

    # Determine base: from --base arg or infer from defaults
    if args.base:
        base_path = Path(args.base).resolve()
        base_inferred = False
        base_cfg_raw = OmegaConf.load(base_path)
        if not isinstance(base_cfg_raw, DictConfig):
            print(
                f"[minimize-check] Base config must be a mapping: {base_path}",
                file=sys.stderr,
            )
            return 2
        base_resolved = OmegaConf.to_container(base_cfg_raw)
    else:
        try:
            base_path = _infer_base_from_defaults(child_path, child_cfg_raw)
        except ValueError as e:
            print(f"[minimize-check] {e}", file=sys.stderr)
            return 2
        base_inferred = True
        # Load EXPANDED base (full inheritance chain) for proper comparison
        base_resolved = OmegaConf.to_container(load_config(str(base_path)))

    child_resolved = OmegaConf.to_container(child_cfg_raw)
    if not isinstance(child_resolved, dict) or not isinstance(base_resolved, dict):
        print(
            f"[minimize-check] Both child and base must resolve to mappings: {child_path} vs {base_path}",
            file=sys.stderr,
        )
        return 2

    # Remove defaults from child before pruning (we'll handle it separately)
    child_defaults = child_resolved.pop("defaults", None)

    pruned = _prune_equal(child_resolved, base_resolved)
    if pruned is None or not isinstance(pruned, dict):
        pruned = {} if pruned is None else {"value": pruned}

    if base_inferred:
        # Keep the existing defaults as-is
        if child_defaults is not None:
            pruned = {"defaults": child_defaults, **pruned}
    else:
        # Explicit base: update defaults to point to the new base
        _ensure_defaults_relative(child_path, base_path, pruned)
        if "defaults" in pruned:
            pruned = {"defaults": pruned["defaults"], **pruned}

    minimized_text = OmegaConf.to_yaml(OmegaConf.create(pruned))

    # Normalize current file via OmegaConf to reduce noise from formatting differences
    try:
        current_norm_text = OmegaConf.to_yaml(OmegaConf.load(child_path))
    except Exception:
        current_norm_text = child_path.read_text()

    if current_norm_text != minimized_text:
        suggested_cmd = f"tools/config_cli.py minimize {child_path} --in-place"
        if args.base:
            suggested_cmd = f"tools/config_cli.py minimize {child_path} --base {base_path} --in-place"
        print(
            f"[minimize-check] {child_path} is not minimized.\n"
            f"  Suggested fix: {suggested_cmd}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config tools (expand, minimize)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_expand = sub.add_parser("expand", help="Resolve a config with OmegaConf")
    p_expand.add_argument("config", help="Path to config YAML")
    p_expand.add_argument(
        "--in-place",
        action="store_true",
        dest="in_place",
        help="Edit file in place instead of printing",
    )
    p_expand.set_defaults(func=expand)

    p_min = sub.add_parser(
        "minimize",
        help="Remove keys equal to inherited values from defaults chain",
    )
    p_min.add_argument("config", help="Config file to minimize")
    p_min.add_argument(
        "--base",
        help="Base config path (if not provided, inferred from config's defaults key)",
    )
    p_min.add_argument(
        "--in-place",
        action="store_true",
        dest="in_place",
        help="Edit file in place instead of printing",
    )
    p_min.set_defaults(func=minimize)

    p_cmp = sub.add_parser(
        "compare", help="Compare two configs after expanding their defaults"
    )
    p_cmp.add_argument("left", help="Left config path")
    p_cmp.add_argument("right", help="Right config path")
    p_cmp.set_defaults(func=compare)

    p_minchk = sub.add_parser(
        "minimize-check",
        help="Exit non-zero if minimizing would change the file",
    )
    p_minchk.add_argument("config", help="Config file to check")
    p_minchk.add_argument(
        "--base",
        help="Base config path (if not provided, inferred from config's defaults key)",
    )
    p_minchk.set_defaults(func=minimize_check)

    args = parser.parse_args()
    ret = args.func(args)
    if isinstance(ret, int):
        sys.exit(ret)
