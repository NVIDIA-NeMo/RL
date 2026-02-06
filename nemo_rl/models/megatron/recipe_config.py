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

"""
Recipe-based configuration for NeMo-RL Megatron integration.

This module provides a clean integration with Megatron-Bridge recipes,
allowing NeMo-RL to use pre-configured training recipes as a base and
layer RL-specific settings on top.

Example usage:
    from nemo_rl.models.megatron.recipe_config import create_config_from_recipe
    
    megatron_cfg = create_config_from_recipe(
        hf_model_name="meta-llama/Llama-3.1-8B-Instruct",
        policy_config=config,
        pretrained_path="/path/to/checkpoint",
        weights_path=None,
    )

Internal flag for testing:
    # To use pure recipe settings with minimal RL overrides (for testing):
    megatron_cfg = create_config_from_recipe(
        ...,
        _apply_full_overrides=False,  # Internal flag - keeps recipe's optimizer/scheduler
    )
"""

import warnings
from typing import Any, Callable, Optional

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)

from nemo_rl.models.policy import PolicyConfig


# =============================================================================
# RECIPE DISCOVERY
# =============================================================================

def _import_llama_recipes():
    """Import Llama recipes from Megatron-Bridge."""
    try:
        from megatron.bridge.recipes.llama.llama3 import (
            llama31_8b_pretrain_config,
            llama31_70b_pretrain_config,
            llama31_405b_pretrain_config,
            llama3_8b_pretrain_config,
            llama3_70b_pretrain_config,
            llama32_1b_pretrain_config,
            llama32_3b_pretrain_config,
        )
        return {
            "llama-3.2-1b": llama32_1b_pretrain_config,
            "llama-3.2-3b": llama32_3b_pretrain_config,
            "llama-3-8b": llama3_8b_pretrain_config,
            "llama-3.1-8b": llama31_8b_pretrain_config,
            "meta-llama-3-8b": llama3_8b_pretrain_config,
            "meta-llama-3.1-8b": llama31_8b_pretrain_config,
            "llama-3-70b": llama3_70b_pretrain_config,
            "llama-3.1-70b": llama31_70b_pretrain_config,
            "llama-3.1-405b": llama31_405b_pretrain_config,
        }
    except ImportError:
        return {}


def _import_qwen_recipes():
    """Import Qwen recipes from Megatron-Bridge."""
    try:
        from megatron.bridge.recipes.qwen.qwen3 import (
            qwen3_600m_pretrain_config,
            qwen3_1p7b_pretrain_config,
            qwen3_4b_pretrain_config,
            qwen3_8b_pretrain_config,
        )
        return {
            "qwen3-0.6b": qwen3_600m_pretrain_config,
            "qwen3-1.7b": qwen3_1p7b_pretrain_config,
            "qwen3-4b": qwen3_4b_pretrain_config,
            "qwen3-8b": qwen3_8b_pretrain_config,
        }
    except ImportError:
        return {}


def get_recipe_function(hf_model_name: str) -> Optional[Callable[..., ConfigContainer]]:
    """
    Get the appropriate Megatron-Bridge recipe function for a model.
    
    Args:
        hf_model_name: HuggingFace model name or path
        
    Returns:
        Recipe function or None if no matching recipe found
    """
    model_lower = hf_model_name.lower().replace("/", "-").replace("_", "-")
    
    # Load recipes lazily
    all_recipes = {}
    all_recipes.update(_import_llama_recipes())
    all_recipes.update(_import_qwen_recipes())
    
    # Try match
    for pattern, recipe_fn in all_recipes.items():
        if pattern in model_lower:
            return recipe_fn
    
    return None


def get_available_recipes() -> list[str]:
    """Return a list of available recipe patterns."""
    all_recipes = {}
    all_recipes.update(_import_llama_recipes())
    all_recipes.update(_import_qwen_recipes())
    return list(all_recipes.keys())


