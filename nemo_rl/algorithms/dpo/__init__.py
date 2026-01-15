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
"""DPO Algorithm Package.

This package provides the Direct Preference Optimization (DPO) algorithm
split into focused modules:

- config.py: Configuration classes (DPOConfig, MasterConfig, etc.)
- loss.py: DPO loss function with reference model
- data.py: Preference data transforms
- trainer.py: DPOTrainer extending BaseTrainer

For backward compatibility, all public symbols from the original dpo.py
are re-exported from this package.

Example (new API):
    >>> from nemo_rl.algorithms.dpo import DPOTrainer, DPOConfig
    >>> 
    >>> trainer = DPOTrainer(config)
    >>> trainer.fit(dataset="Anthropic/hh-rlhf")

Example (backward compatible):
    >>> from nemo_rl.algorithms.dpo import (
    ...     MasterConfig, dpo_train, setup
    ... )
    >>> 
    >>> # Legacy API still works
    >>> dpo_train(...)
"""

# New modular exports
from nemo_rl.algorithms.dpo.config import (
    DPOConfig,
    DPOSaveState,
    DPOValMetrics,
    MasterConfig,
    default_dpo_save_state,
)
from nemo_rl.algorithms.dpo.data import (
    add_ref_logprobs_to_batch,
    prepare_preference_batch,
)
from nemo_rl.algorithms.dpo.loss import (
    DPOLoss,
    create_dpo_loss_function,
)
from nemo_rl.algorithms.dpo.trainer import DPOTrainer

# Backward compatibility: re-export from original dpo.py
from nemo_rl.algorithms.dpo_legacy import (
    _default_dpo_save_state,
    add_ref_logprobs_to_data,
    dpo_train,
    setup,
    validate,
    validate_one_dataset,
)

__all__ = [
    # New modular API
    "DPOTrainer",
    # Config classes
    "DPOConfig",
    "MasterConfig",
    "DPOSaveState",
    "DPOValMetrics",
    "default_dpo_save_state",
    # Loss functions
    "DPOLoss",
    "create_dpo_loss_function",
    # Data functions
    "prepare_preference_batch",
    "add_ref_logprobs_to_batch",
    # Legacy API (backward compatibility)
    "setup",
    "dpo_train",
    "validate",
    "validate_one_dataset",
    "_default_dpo_save_state",
    "add_ref_logprobs_to_data",
]
