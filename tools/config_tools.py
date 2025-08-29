#!/usr/bin/env -S uv run -q
"""Utilities for working with YAML configs in this repo.

Subcommands:
  - expand: Resolve a config with OmegaConf interpolation and inheritance.
  - minimize: Given a config and a base config, remove keys in the config that
    are equal to the base, and ensure a defaults entry pointing to the base
    exists. The defaults path in the resulting config is written relative to
    the base config file.

Both commands support printing to stdout or in-place editing of the config file.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import load_config


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
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))


def minimize(args: argparse.Namespace) -> int:
    child_path = Path(args.config).resolve()
    base_path = Path(args.base).resolve()

    child_cfg_raw = OmegaConf.load(child_path)
    if not isinstance(child_cfg_raw, DictConfig):
        raise TypeError(
            f"Config at {child_path} must be a mapping (DictConfig), got {type(child_cfg_raw)}"
        )
    base_cfg_raw = OmegaConf.load(base_path)
    if not isinstance(base_cfg_raw, DictConfig):
        raise TypeError(
            f"Config at {base_path} must be a mapping (DictConfig), got {type(base_cfg_raw)}"
        )

    # Resolve both before comparison
    child_resolved = OmegaConf.to_container(child_cfg_raw)
    base_resolved = OmegaConf.to_container(base_cfg_raw)

    if not isinstance(child_resolved, dict) or not isinstance(base_resolved, dict):
        raise TypeError("Both child and base configs must be mappings after resolution")

    pruned = _prune_equal(child_resolved, base_resolved)

    # Ensure mapping output
    if pruned is None or not isinstance(pruned, dict):
        pruned = {} if pruned is None else {"value": pruned}

    # Ensure defaults reference base (relative path from child)
    _ensure_defaults_relative(child_path, base_path, pruned)

    # Ensure `defaults` appears first in the top-level mapping
    if "defaults" in pruned:
        pruned = {"defaults": pruned["defaults"], **pruned}

    # Emit
    text = OmegaConf.to_yaml(OmegaConf.create(pruned))
    if args.in_place:
        Path(args.config).write_text(text)
    else:
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))


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
        help="Remove keys equal to base and ensure defaults reference base",
    )
    p_min.add_argument("config", help="Child config path")
    p_min.add_argument("base", help="Base config path")
    p_min.add_argument(
        "--in-place",
        action="store_true",
        dest="in_place",
        help="Edit file in place instead of printing",
    )
    p_min.set_defaults(func=minimize)

    args = parser.parse_args()
    args.func(args)
