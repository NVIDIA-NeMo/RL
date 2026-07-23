"""Shared NeMo-RL environment setup utilities.

Consolidates env var setup, OmegaConf resolver registration, config path
resolution, and nested dict helpers that were previously duplicated across
train.py, sft_train.py, backend/local.py, and rl/nemo_gym.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def setup_nemo_env() -> None:
    """Set standard NeMo-RL environment variables. Idempotent."""
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    os.environ.setdefault("NVTE_FLASH_ATTN", "0")
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")


def register_omegaconf_resolvers() -> None:
    """Register custom OmegaConf resolvers used by NeMo-RL configs. Idempotent."""
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)


def resolve_config_path(config: str | Path) -> Path:
    """Resolve a config file path, searching multiple locations.

    Search order:
        1. As-is (absolute or relative to cwd)
        2. Relative to nousnet package root (src/../)
        3. Relative to /workspace/nousnet/ (inside NGC container)

    Args:
        config: Path string or Path object.

    Returns:
        Resolved Path (may not exist if not found anywhere).
    """
    config_path = Path(config)
    if config_path.is_absolute() or config_path.exists():
        return config_path

    # Search relative to nousnet package root
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    candidate = pkg_root / config_path
    if candidate.exists():
        return candidate

    # Search relative to /workspace/nousnet/ (container mount)
    workspace = Path("/workspace/nousnet") / config_path
    if workspace.exists():
        return workspace

    return config_path


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set a nested config value using dotted key notation.

    Args:
        cfg: Config dict to modify in-place.
        dotted_key: Key like "policy.optimizer.kwargs.lr".
        value: Value to set.

    Example:
        >>> cfg = {}
        >>> set_nested(cfg, "a.b.c", 42)
        >>> cfg
        {'a': {'b': {'c': 42}}}
    """
    parts = dotted_key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value
