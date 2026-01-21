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

"""Configuration classes for automodel-based training in NeMo RL."""

from typing import Any, NamedTuple, Optional

import torch
from nemo_automodel.components._peft.lora import PeftConfig


class RuntimeConfig(NamedTuple):
    """Runtime configuration for model training and inference.

    This contains all validated runtime settings needed for model initialization,
    parallelization, and training.
    """

    # Model loading configuration
    model_class: type
    model_config: Any  # AutoConfig
    hf_config_overrides: dict[str, Any]

    # Attention configuration
    allow_flash_attn_args: bool
    attn_impl: Optional[str]

    # Training/inference settings
    dtype: torch.dtype
    enable_seq_packing: bool
    max_grad_norm: float

    # Memory management
    cpu_offload: bool
    offload_optimizer_for_logprob: bool

    # Generation configuration
    is_generation_colocated: Optional[bool]

    # Reward model flag
    is_reward_model: bool


class ModelAndOptimizerState(NamedTuple):
    """Container for model and optimizer state.

    This named tuple holds all model-related state including the model itself,
    optimizer, scheduler, and metadata about the model type and configuration.
    """

    model: torch.nn.Module
    model_state_dict_keys: list[str]
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[Any]
    is_hf_model: bool
    is_moe_model: bool
    is_reward_model: bool
    model_class: type
    model_config: Any
    peft_config: Optional[PeftConfig]
    autocast_enabled: bool
