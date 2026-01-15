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
"""Backward-compatible configuration utilities.

This module provides utilities for loading and converting old-style
OmegaConf/YAML configurations to the new Pydantic-based configs.

Old API (deprecated):
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load("config.yaml")
    >>> grpo = GRPO.from_config(cfg)

New API (recommended):
    >>> from nemo_rl.config import GRPOConfig
    >>> config = GRPOConfig.from_yaml("config.yaml")
    >>> trainer = GRPOTrainer(config)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from nemo_rl.compat.deprecation import MIGRATION_GUIDE_URL

T = TypeVar("T")


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and return as dict.

    This is a compatibility function for loading old-style YAML configs.
    It emits a deprecation warning suggesting the new approach.

    Args:
        path: Path to the YAML config file.

    Returns:
        Configuration dictionary.

    Example:
        >>> cfg = load_yaml_config("configs/grpo_math_1B.yaml")
        >>> # Convert to new config
        >>> from nemo_rl.config import GRPOConfig
        >>> config = GRPOConfig.from_dict(cfg)
    """
    warnings.warn(
        "load_yaml_config() is deprecated. Use Config.from_yaml() instead.\n"
        "Example: GRPOConfig.from_yaml('config.yaml')\n"
        f"Migration guide: {MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=2,
    )

    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def convert_omegaconf_to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert an OmegaConf object to a plain dictionary.

    Args:
        cfg: OmegaConf configuration object.

    Returns:
        Plain dictionary with resolved values.
    """
    if hasattr(cfg, "to_container"):
        return cfg.to_container(resolve=True)
    elif hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    elif isinstance(cfg, dict):
        return dict(cfg)
    else:
        return {}


def convert_old_config_to_new(
    old_config: Dict[str, Any],
    algorithm: str = "grpo",
) -> Dict[str, Any]:
    """Convert old-style config dict to new config format.

    This function maps old configuration keys to new ones and
    handles structural differences between the formats.

    Args:
        old_config: Old-style configuration dictionary.
        algorithm: Algorithm type ('grpo', 'sft', 'dpo').

    Returns:
        New-style configuration dictionary.

    Example:
        >>> old_cfg = OmegaConf.load("old_config.yaml")
        >>> new_cfg_dict = convert_old_config_to_new(old_cfg, "grpo")
        >>> from nemo_rl.config import GRPOConfig
        >>> config = GRPOConfig.from_dict(new_cfg_dict)
    """
    warnings.warn(
        "convert_old_config_to_new() is deprecated. "
        "Use the new config classes directly.\n"
        f"Migration guide: {MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=2,
    )

    new_config: Dict[str, Any] = {}

    # Map old policy config to new
    old_policy = old_config.get("policy", {})
    new_config["policy"] = _convert_policy_config(old_policy)

    # Map algorithm-specific config
    if algorithm == "grpo":
        old_grpo = old_config.get("grpo", {})
        new_config.update(_convert_grpo_config(old_grpo))
    elif algorithm == "sft":
        old_sft = old_config.get("sft", {})
        new_config.update(_convert_sft_config(old_sft))
    elif algorithm == "dpo":
        old_dpo = old_config.get("dpo", {})
        new_config.update(_convert_dpo_config(old_dpo))

    # Map cluster config
    old_cluster = old_config.get("cluster", {})
    if old_cluster:
        new_config["cluster"] = _convert_cluster_config(old_cluster)

    # Map checkpointing config
    old_checkpoint = old_config.get("checkpointing", {})
    if old_checkpoint:
        new_config["checkpointing"] = _convert_checkpoint_config(old_checkpoint)

    # Map logger config
    old_logger = old_config.get("logger", {})
    if old_logger:
        new_config["logger"] = old_logger

    return new_config


def _convert_policy_config(old_policy: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old policy config to new format."""
    new_policy: Dict[str, Any] = {}

    # Direct mappings
    direct_fields = [
        "model_name",
        "precision",
        "train_global_batch_size",
        "train_micro_batch_size",
        "max_total_sequence_length",
        "max_grad_norm",
    ]
    for field in direct_fields:
        if field in old_policy:
            new_policy[field] = old_policy[field]

    # Map backend
    if "backend" in old_policy:
        backend = old_policy["backend"]
        if isinstance(backend, str):
            new_policy["backend"] = backend.lower()

    # Map optimizer
    old_optimizer = old_policy.get("optimizer", {})
    if old_optimizer:
        new_optimizer = {"name": "adamw", "kwargs": {}}
        if "name" in old_optimizer:
            # Old format might have full class path
            name = old_optimizer["name"]
            if "AdamW" in name:
                new_optimizer["name"] = "adamw"
            elif "SGD" in name:
                new_optimizer["name"] = "sgd"
            elif "Adam" in name:
                new_optimizer["name"] = "adam"

        # Map kwargs (especially lr)
        old_kwargs = old_optimizer.get("kwargs", {})
        if old_kwargs:
            new_optimizer["kwargs"] = old_kwargs

        new_policy["optimizer"] = new_optimizer

    # Map DTensor config
    old_dtensor = old_policy.get("dtensor_cfg", {})
    if old_dtensor:
        new_policy["dtensor_cfg"] = old_dtensor

    # Map Megatron config
    old_megatron = old_policy.get("megatron_cfg", {})
    if old_megatron:
        new_policy["megatron_cfg"] = old_megatron

    return new_policy


def _convert_grpo_config(old_grpo: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old GRPO config to new format."""
    new_config: Dict[str, Any] = {}

    # Direct mappings
    grpo_fields = [
        "num_prompts_per_step",
        "num_generations_per_prompt",
        "max_num_epochs",
        "max_num_steps",
        "normalize_rewards",
        "use_leave_one_out_baseline",
        "val_period",
        "val_batch_size",
        "seed",
    ]
    for field in grpo_fields:
        if field in old_grpo:
            new_config[field] = old_grpo[field]

    # Map loss function config
    old_loss = old_grpo.get("loss_fn", {})
    if old_loss:
        new_config["loss_fn"] = old_loss

    return new_config


def _convert_sft_config(old_sft: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old SFT config to new format."""
    new_config: Dict[str, Any] = {}

    sft_fields = [
        "max_num_epochs",
        "max_num_steps",
        "val_period",
        "val_batch_size",
        "seed",
    ]
    for field in sft_fields:
        if field in old_sft:
            new_config[field] = old_sft[field]

    return new_config


def _convert_dpo_config(old_dpo: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old DPO config to new format."""
    new_config: Dict[str, Any] = {}

    dpo_fields = [
        "max_num_epochs",
        "max_num_steps",
        "val_period",
        "val_batch_size",
        "seed",
    ]
    for field in dpo_fields:
        if field in old_dpo:
            new_config[field] = old_dpo[field]

    # Map DPO-specific loss params
    if "beta" in old_dpo:
        new_config.setdefault("loss_fn", {})["beta"] = old_dpo["beta"]
    if "label_smoothing" in old_dpo:
        new_config.setdefault("loss_fn", {})["label_smoothing"] = old_dpo["label_smoothing"]

    return new_config


def _convert_cluster_config(old_cluster: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old cluster config to new format."""
    new_cluster: Dict[str, Any] = {}

    cluster_fields = [
        "num_nodes",
        "gpus_per_node",
        "master_addr",
        "master_port",
    ]
    for field in cluster_fields:
        if field in old_cluster:
            new_cluster[field] = old_cluster[field]

    return new_cluster


def _convert_checkpoint_config(old_checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old checkpoint config to new format."""
    new_checkpoint: Dict[str, Any] = {}

    checkpoint_fields = [
        "enabled",
        "checkpoint_dir",
        "save_period",
        "keep_top_k",
        "metric_name",
        "higher_is_better",
        "model_save_format",
    ]
    for field in checkpoint_fields:
        if field in old_checkpoint:
            new_checkpoint[field] = old_checkpoint[field]

    return new_checkpoint


class OmegaConfAdapter:
    """Adapter class that mimics OmegaConf interface.

    This class provides a compatibility layer for code that expects
    OmegaConf-style access (cfg.policy.model_name) while using
    the new Pydantic configs internally.

    Example:
        >>> cfg = OmegaConfAdapter(grpo_config.model_dump())
        >>> model_name = cfg.policy.model_name  # Works like OmegaConf
    """

    def __init__(self, data: Dict[str, Any]):
        """Initialize the adapter.

        Args:
            data: Configuration dictionary.
        """
        warnings.warn(
            "OmegaConfAdapter is deprecated. Use Pydantic config objects directly.\n"
            f"Migration guide: {MIGRATION_GUIDE_URL}",
            DeprecationWarning,
            stacklevel=2,
        )
        self._data = data

    def __getattr__(self, name: str) -> Any:
        """Get attribute with OmegaConf-style access."""
        if name.startswith("_"):
            raise AttributeError(name)

        value = self._data.get(name)
        if isinstance(value, dict):
            return OmegaConfAdapter.__new__(OmegaConfAdapter)._init_nested(value)
        return value

    def _init_nested(self, data: Dict[str, Any]) -> "OmegaConfAdapter":
        """Initialize a nested adapter without deprecation warning."""
        self._data = data
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with OmegaConf-style access."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def to_container(self, resolve: bool = True) -> Dict[str, Any]:
        """Return the underlying dictionary (OmegaConf compatibility)."""
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying dictionary."""
        return self._data


__all__ = [
    "load_yaml_config",
    "convert_omegaconf_to_dict",
    "convert_old_config_to_new",
    "OmegaConfAdapter",
]
