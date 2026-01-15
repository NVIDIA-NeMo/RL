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
"""NeMo RL Configuration Module.

This module provides a unified configuration system using Pydantic for runtime
validation. All configurations are type-annotated and validated at load time,
providing clear error messages for invalid configurations.

Features:
- Type-safe configuration with runtime validation
- Sensible defaults - minimal config requires only model_name
- Multiple format support: YAML, JSON, Python dict
- Factory methods for common configurations
- Automatic resource detection with ClusterConfig.auto_detect()

Example usage:
    >>> # Minimal config - just model name required!
    >>> from nemo_rl.config import GRPOConfig
    >>> config = GRPOConfig.minimal("Qwen/Qwen2.5-1.5B")

    >>> # Full control with custom settings
    >>> from nemo_rl.config import GRPOConfig, PolicyConfig, ClusterConfig
    >>> config = GRPOConfig(
    ...     policy=PolicyConfig(model_name="Qwen/Qwen2.5-1.5B"),
    ...     cluster=ClusterConfig.auto_detect(),
    ... )

    >>> # Use templates for common scenarios
    >>> from nemo_rl.config import load_template
    >>> config = load_template("grpo_1b", model_name="Qwen/Qwen2.5-1.5B")

    >>> # Load from YAML/JSON/dict
    >>> config = GRPOConfig.from_yaml("config.yaml")
    >>> config = GRPOConfig.from_json("config.json")
    >>> config = GRPOConfig.from_dict({"policy": {"model_name": "..."}})
"""

from nemo_rl.config.base import BaseConfig
from nemo_rl.config.cluster import ClusterConfig
from nemo_rl.config.generation import GenerationConfig, VLLMConfig
from nemo_rl.config.policy import (
    DTensorConfig,
    DynamicBatchingConfig,
    LoRAConfig,
    MegatronConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
    SequencePackingConfig,
    TokenizerConfig,
)
from nemo_rl.config.training import (
    CheckpointingConfig,
    DPOConfig,
    GRPOConfig,
    LoggerConfig,
    SFTConfig,
)
from nemo_rl.config.validation import (
    ConfigValidationError,
    validate_config,
)
from nemo_rl.config.defaults import (
    get_default_optimizer,
    get_default_policy_config,
    get_default_scheduler,
    get_dpo_config,
    get_grpo_config_for_1b_model,
    get_grpo_config_for_8b_model,
    get_sft_config,
    list_templates,
    load_template,
)

__all__ = [
    # Base
    "BaseConfig",
    # Cluster
    "ClusterConfig",
    # Generation
    "GenerationConfig",
    "VLLMConfig",
    # Policy
    "PolicyConfig",
    "DTensorConfig",
    "MegatronConfig",
    "LoRAConfig",
    "TokenizerConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "DynamicBatchingConfig",
    "SequencePackingConfig",
    # Training
    "GRPOConfig",
    "SFTConfig",
    "DPOConfig",
    "LoggerConfig",
    "CheckpointingConfig",
    # Validation
    "ConfigValidationError",
    "validate_config",
    # Defaults and Templates
    "get_default_optimizer",
    "get_default_scheduler",
    "get_default_policy_config",
    "get_grpo_config_for_1b_model",
    "get_grpo_config_for_8b_model",
    "get_sft_config",
    "get_dpo_config",
    "load_template",
    "list_templates",
]
