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
"""Backward Compatibility Module for NeMo RL.

This module provides backward-compatible wrappers for the old NeMo RL API.
All exports emit deprecation warnings when used, encouraging migration
to the new API.

The old API will continue to work until version 2.0, at which point
these compatibility wrappers will be removed.

Deprecation Timeline:
    - Version 0.9 (current): Old API fully supported, new API available
    - Version 1.0 (+3 months): Old API deprecated with warnings (this module)
    - Version 1.1 (+6 months): Old API removed from main codebase
    - Version 2.0 (+12 months): This compatibility module removed

Old API Example (deprecated):
    >>> from nemo_rl.compat import GRPO
    >>> from omegaconf import OmegaConf
    >>> 
    >>> cfg = OmegaConf.load("configs/grpo_math_1B.yaml")
    >>> grpo = GRPO.from_config(cfg)
    >>> grpo.fit()

New API Example (recommended):
    >>> import nemo_rl
    >>> 
    >>> result = nemo_rl.train(
    ...     model="Qwen/Qwen2.5-1.5B",
    ...     dataset="nvidia/OpenMathInstruct-2",
    ...     reward_fn=my_reward,
    ...     max_steps=1000,
    ... )

Or with trainer classes:
    >>> from nemo_rl import GRPOTrainer
    >>> 
    >>> trainer = GRPOTrainer.from_pretrained("Qwen/Qwen2.5-1.5B")
    >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2", reward_fn=my_reward)

Migration Guide:
    For detailed migration instructions, see:
    https://nvidia.github.io/nemo-rl/migration
"""

# Algorithm classes (deprecated)
from nemo_rl.compat.algorithms import (
    DPO,
    GRPO,
    SFT,
    dpo_train,
    grpo_train,
    sft_train,
)

# Config utilities (deprecated)
from nemo_rl.compat.config import (
    OmegaConfAdapter,
    convert_old_config_to_new,
    convert_omegaconf_to_dict,
    load_yaml_config,
)

# Deprecation utilities (for internal use)
from nemo_rl.compat.deprecation import (
    API_DOCS_URL,
    MIGRATION_GUIDE_URL,
    DeprecatedAlias,
    deprecated_class,
    deprecated_function,
    deprecated_import,
    deprecated_method,
    deprecated_parameter,
)

__all__ = [
    # Deprecated algorithm classes (use new Trainer classes instead)
    "GRPO",  # Use GRPOTrainer from nemo_rl.algorithms.grpo
    "SFT",   # Use SFTTrainer from nemo_rl.algorithms.sft
    "DPO",   # Use DPOTrainer from nemo_rl.algorithms.dpo
    # Deprecated training functions (use nemo_rl.train() instead)
    "grpo_train",
    "sft_train",
    "dpo_train",
    # Deprecated config utilities (use Config.from_yaml() instead)
    "load_yaml_config",
    "convert_omegaconf_to_dict",
    "convert_old_config_to_new",
    "OmegaConfAdapter",
    # Deprecation utilities (for library developers)
    "deprecated_import",
    "deprecated_function",
    "deprecated_class",
    "deprecated_method",
    "deprecated_parameter",
    "DeprecatedAlias",
    "MIGRATION_GUIDE_URL",
    "API_DOCS_URL",
]

# Package version where these will be removed
REMOVAL_VERSION = "2.0"
