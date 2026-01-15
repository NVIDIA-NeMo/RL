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
"""Backward-compatible algorithm imports.

This module provides the old-style algorithm class names for backward
compatibility. All classes emit deprecation warnings when used.

Old API (deprecated):
    >>> from nemo_rl.compat.algorithms import GRPO
    >>> grpo = GRPO.from_config(cfg)

New API (recommended):
    >>> from nemo_rl.algorithms.grpo import GRPOTrainer
    >>> trainer = GRPOTrainer.from_pretrained("model")
"""

from __future__ import annotations

import warnings
from typing import Any, TypeVar

from nemo_rl.compat.deprecation import (
    MIGRATION_GUIDE_URL,
    deprecated_method,
)

T = TypeVar("T")


def _emit_deprecation_warning(old_name: str, new_name: str, new_module: str) -> None:
    """Emit a deprecation warning for an old algorithm name."""
    warnings.warn(
        f"'{old_name}' is deprecated and will be removed in version 2.0. "
        f"Use '{new_name}' from '{new_module}' instead.\n"
        f"Migration guide: {MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=3,
    )


class GRPO:
    """Deprecated: Use GRPOTrainer from nemo_rl.algorithms.grpo instead.

    This class provides backward compatibility for old-style GRPO usage.
    It wraps the new GRPOTrainer class and emits deprecation warnings.

    Old usage:
        >>> grpo = GRPO.from_config(cfg)
        >>> grpo.fit()

    New usage:
        >>> from nemo_rl.algorithms.grpo import GRPOTrainer
        >>> trainer = GRPOTrainer.from_pretrained("model")
        >>> trainer.fit(dataset=data, reward_fn=my_reward)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize GRPO with deprecation warning."""
        _emit_deprecation_warning("GRPO", "GRPOTrainer", "nemo_rl.algorithms.grpo")

        # Import here to avoid circular imports
        from nemo_rl.algorithms.grpo import GRPOTrainer
        self._trainer = GRPOTrainer(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg: Any, **kwargs: Any) -> "GRPO":
        """Create GRPO from OmegaConf config (deprecated).

        Args:
            cfg: OmegaConf configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            GRPO instance wrapping the new trainer.

        Note:
            This method is deprecated. Use GRPOTrainer.from_pretrained() instead.
        """
        _emit_deprecation_warning(
            "GRPO.from_config()",
            "GRPOTrainer.from_pretrained()",
            "nemo_rl.algorithms.grpo"
        )

        instance = cls.__new__(cls)

        # Import here to avoid circular imports
        from nemo_rl.algorithms.grpo import GRPOTrainer, GRPOConfig
        from nemo_rl.config import PolicyConfig

        # Convert OmegaConf to dict if needed
        if hasattr(cfg, "to_container"):
            config_dict = cfg.to_container(resolve=True)
        elif hasattr(cfg, "to_dict"):
            config_dict = cfg.to_dict()
        else:
            config_dict = dict(cfg) if cfg else {}

        # Extract model name from policy config
        policy_cfg = config_dict.get("policy", {})
        model_name = policy_cfg.get("model_name", "")

        # Create trainer with config
        try:
            instance._trainer = GRPOTrainer.from_pretrained(model_name, **kwargs)
        except Exception:
            # Fallback: create with minimal config
            instance._trainer = GRPOTrainer(model=model_name)

        return instance

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Train the model (deprecated).

        Note:
            This method wraps the new trainer's fit() method.
        """
        return self._trainer.fit(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying trainer."""
        return getattr(self._trainer, name)


class SFT:
    """Deprecated: Use SFTTrainer from nemo_rl.algorithms.sft instead.

    This class provides backward compatibility for old-style SFT usage.
    It wraps the new SFTTrainer class and emits deprecation warnings.

    Old usage:
        >>> sft = SFT.from_config(cfg)
        >>> sft.fit(dataset_path="data.jsonl")

    New usage:
        >>> from nemo_rl.algorithms.sft import SFTTrainer
        >>> trainer = SFTTrainer.from_pretrained("model")
        >>> trainer.fit(dataset="data.jsonl")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize SFT with deprecation warning."""
        _emit_deprecation_warning("SFT", "SFTTrainer", "nemo_rl.algorithms.sft")

        from nemo_rl.algorithms.sft import SFTTrainer
        self._trainer = SFTTrainer(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg: Any, **kwargs: Any) -> "SFT":
        """Create SFT from OmegaConf config (deprecated).

        Args:
            cfg: OmegaConf configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            SFT instance wrapping the new trainer.
        """
        _emit_deprecation_warning(
            "SFT.from_config()",
            "SFTTrainer.from_pretrained()",
            "nemo_rl.algorithms.sft"
        )

        instance = cls.__new__(cls)

        from nemo_rl.algorithms.sft import SFTTrainer

        # Convert OmegaConf to dict if needed
        if hasattr(cfg, "to_container"):
            config_dict = cfg.to_container(resolve=True)
        elif hasattr(cfg, "to_dict"):
            config_dict = cfg.to_dict()
        else:
            config_dict = dict(cfg) if cfg else {}

        policy_cfg = config_dict.get("policy", {})
        model_name = policy_cfg.get("model_name", "")

        try:
            instance._trainer = SFTTrainer.from_pretrained(model_name, **kwargs)
        except Exception:
            instance._trainer = SFTTrainer(model=model_name)

        return instance

    def fit(self, *args: Any, dataset_path: str | None = None, **kwargs: Any) -> Any:
        """Train the model (deprecated).

        Args:
            dataset_path: Path to training data (old parameter name).
            **kwargs: Additional arguments passed to fit().
        """
        if dataset_path and "dataset" not in kwargs:
            kwargs["dataset"] = dataset_path
        return self._trainer.fit(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying trainer."""
        return getattr(self._trainer, name)


class DPO:
    """Deprecated: Use DPOTrainer from nemo_rl.algorithms.dpo instead.

    This class provides backward compatibility for old-style DPO usage.
    It wraps the new DPOTrainer class and emits deprecation warnings.

    Old usage:
        >>> dpo = DPO.from_config(cfg)
        >>> dpo.fit(dataset_path="preferences.jsonl")

    New usage:
        >>> from nemo_rl.algorithms.dpo import DPOTrainer
        >>> trainer = DPOTrainer.from_pretrained("model", beta=0.1)
        >>> trainer.fit(dataset="preferences.jsonl")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize DPO with deprecation warning."""
        _emit_deprecation_warning("DPO", "DPOTrainer", "nemo_rl.algorithms.dpo")

        from nemo_rl.algorithms.dpo import DPOTrainer
        self._trainer = DPOTrainer(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg: Any, **kwargs: Any) -> "DPO":
        """Create DPO from OmegaConf config (deprecated).

        Args:
            cfg: OmegaConf configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            DPO instance wrapping the new trainer.
        """
        _emit_deprecation_warning(
            "DPO.from_config()",
            "DPOTrainer.from_pretrained()",
            "nemo_rl.algorithms.dpo"
        )

        instance = cls.__new__(cls)

        from nemo_rl.algorithms.dpo import DPOTrainer

        # Convert OmegaConf to dict if needed
        if hasattr(cfg, "to_container"):
            config_dict = cfg.to_container(resolve=True)
        elif hasattr(cfg, "to_dict"):
            config_dict = cfg.to_dict()
        else:
            config_dict = dict(cfg) if cfg else {}

        policy_cfg = config_dict.get("policy", {})
        model_name = policy_cfg.get("model_name", "")

        # Extract DPO-specific params
        dpo_cfg = config_dict.get("dpo", {})
        beta = dpo_cfg.get("beta", 0.1)

        try:
            instance._trainer = DPOTrainer.from_pretrained(model_name, beta=beta, **kwargs)
        except Exception:
            instance._trainer = DPOTrainer(model=model_name)

        return instance

    def fit(self, *args: Any, dataset_path: str | None = None, **kwargs: Any) -> Any:
        """Train the model (deprecated).

        Args:
            dataset_path: Path to preference data (old parameter name).
            **kwargs: Additional arguments passed to fit().
        """
        if dataset_path and "dataset" not in kwargs:
            kwargs["dataset"] = dataset_path
        return self._trainer.fit(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying trainer."""
        return getattr(self._trainer, name)


# Functions for procedural-style training (deprecated)
def grpo_train(
    model: str,
    dataset: str,
    reward_fn: Any,
    **kwargs: Any,
) -> Any:
    """Deprecated: Use nemo_rl.train() instead.

    This function provides backward compatibility for procedural-style
    GRPO training.

    Args:
        model: Model name or path.
        dataset: Dataset name or path.
        reward_fn: Reward function.
        **kwargs: Additional training arguments.

    Returns:
        Training result.
    """
    warnings.warn(
        "grpo_train() is deprecated. Use nemo_rl.train() instead.\n"
        f"Migration guide: {MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=2,
    )

    import nemo_rl
    return nemo_rl.train(
        model=model,
        dataset=dataset,
        reward_fn=reward_fn,
        algorithm="grpo",
        **kwargs,
    )


def sft_train(
    model: str,
    dataset: str,
    **kwargs: Any,
) -> Any:
    """Deprecated: Use nemo_rl.train() instead.

    This function provides backward compatibility for procedural-style
    SFT training.

    Args:
        model: Model name or path.
        dataset: Dataset name or path.
        **kwargs: Additional training arguments.

    Returns:
        Training result.
    """
    warnings.warn(
        "sft_train() is deprecated. Use nemo_rl.train() instead.\n"
        f"Migration guide: {MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=2,
    )

    import nemo_rl
    return nemo_rl.train(
        model=model,
        dataset=dataset,
        algorithm="sft",
        **kwargs,
    )


def dpo_train(
    model: str,
    dataset: str,
    beta: float = 0.1,
    **kwargs: Any,
) -> Any:
    """Deprecated: Use nemo_rl.train() instead.

    This function provides backward compatibility for procedural-style
    DPO training.

    Args:
        model: Model name or path.
        dataset: Dataset name or path.
        beta: DPO beta parameter.
        **kwargs: Additional training arguments.

    Returns:
        Training result.
    """
    warnings.warn(
        "dpo_train() is deprecated. Use nemo_rl.train() instead.\n"
        f"Migration guide: {MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=2,
    )

    import nemo_rl
    return nemo_rl.train(
        model=model,
        dataset=dataset,
        algorithm="dpo",
        beta=beta,
        **kwargs,
    )


__all__ = [
    # Deprecated classes
    "GRPO",
    "SFT",
    "DPO",
    # Deprecated functions
    "grpo_train",
    "sft_train",
    "dpo_train",
]
