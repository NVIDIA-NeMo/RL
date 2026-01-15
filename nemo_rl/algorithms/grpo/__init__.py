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
"""GRPO Algorithm Package.

This package provides the Group Relative Policy Optimization (GRPO) algorithm
split into focused modules:

- config.py: Configuration classes (GRPOConfig, MasterConfig, etc.)
- loss.py: Loss functions and advantage computation
- data.py: Data transforms and batch processing
- trainer.py: GRPOTrainer extending BaseTrainer
- utils.py: Utility functions

For backward compatibility, all public symbols from the original grpo.py
are re-exported from this package.

Example (new API):
    >>> from nemo_rl.algorithms.grpo import GRPOTrainer, GRPOConfig
    >>> 
    >>> trainer = GRPOTrainer(config)
    >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")

Example (backward compatible):
    >>> from nemo_rl.algorithms.grpo import (
    ...     MasterConfig, grpo_train, setup
    ... )
    >>> 
    >>> # Legacy API still works
    >>> grpo_train(...)
"""

# New modular exports
from nemo_rl.algorithms.grpo.config import (
    AsyncGRPOConfig,
    GRPOConfig,
    GRPOLoggerConfig,
    GRPOSaveState,
    MasterConfig,
    RewardScalingConfig,
    default_grpo_save_state,
)
from nemo_rl.algorithms.grpo.data import (
    check_batch_has_variance,
    dynamic_sample_batch,
    filter_overlong_sequences,
    prepare_batch_for_training,
)
from nemo_rl.algorithms.grpo.loss import (
    compute_grpo_loss,
    create_loss_function,
    normalize_advantages_with_epsilon as normalize_advantages_in_batch,
    scale_rewards,
)
from nemo_rl.algorithms.grpo.trainer import GRPOTrainer
from nemo_rl.algorithms.grpo.utils import (
    compute_effective_batch_size,
    get_generation_backend,
    is_colocation_enabled,
    log_training_metrics,
    should_use_async_rollouts,
    should_use_nemo_gym,
)

# Backward compatibility: re-export from original grpo.py
# This ensures existing code continues to work
from nemo_rl.algorithms.grpo_legacy import (
    _default_grpo_save_state,
    async_grpo_train,
    dynamic_sampling,
    grpo_train,
    normalize_advantages_with_epsilon,  # Use legacy version for backward compat
    setup,
    validate,
)

# Legacy names that are now in config.py
# (already exported above, just noting for documentation)
# MasterConfig, GRPOConfig, GRPOSaveState, GRPOLoggerConfig

__all__ = [
    # New modular API
    "GRPOTrainer",
    # Config classes
    "GRPOConfig",
    "MasterConfig",
    "AsyncGRPOConfig",
    "RewardScalingConfig",
    "GRPOSaveState",
    "GRPOLoggerConfig",
    "default_grpo_save_state",
    # Loss functions
    "normalize_advantages_with_epsilon",
    "normalize_advantages_in_batch",  # New batch-based API
    "scale_rewards",
    "compute_grpo_loss",
    "create_loss_function",
    # Data functions
    "filter_overlong_sequences",
    "check_batch_has_variance",
    "dynamic_sample_batch",
    "prepare_batch_for_training",
    # Utility functions
    "should_use_async_rollouts",
    "should_use_nemo_gym",
    "get_generation_backend",
    "is_colocation_enabled",
    "compute_effective_batch_size",
    "log_training_metrics",
    # Legacy API (backward compatibility)
    "setup",
    "grpo_train",
    "async_grpo_train",
    "validate",
    "_default_grpo_save_state",
    "dynamic_sampling",
]
