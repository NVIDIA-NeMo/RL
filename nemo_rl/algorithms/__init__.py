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
"""NeMo RL Algorithms Module.

This module provides RL training algorithms including:
- GRPO (Group Relative Policy Optimization)
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- Rollout orchestration

Example:
    >>> from nemo_rl.algorithms import RolloutEngine, SamplingParams
    >>> 
    >>> engine = RolloutEngine(backend, environment)
    >>> result = engine.rollout(prompts, SamplingParams(temperature=0.7))
    
    >>> # Or use GRPO trainer
    >>> from nemo_rl.algorithms.grpo import GRPOTrainer
    >>> trainer = GRPOTrainer(config)
    >>> trainer.fit(dataset)
"""

from nemo_rl.algorithms.rollout import (
    RolloutEngine,
    RolloutResult,
    SamplingParams,
    create_rollout_engine,
)

# GRPO exports (convenient access)
from nemo_rl.algorithms.grpo import GRPOTrainer, GRPOConfig

# SFT exports (convenient access)
from nemo_rl.algorithms.sft import SFTTrainer, SFTConfig

# DPO exports (convenient access)
from nemo_rl.algorithms.dpo import DPOTrainer, DPOConfig

__all__ = [
    # Rollout
    "RolloutEngine",
    "RolloutResult",
    "SamplingParams",
    "create_rollout_engine",
    # GRPO
    "GRPOTrainer",
    "GRPOConfig",
    # SFT
    "SFTTrainer",
    "SFTConfig",
    # DPO
    "DPOTrainer",
    "DPOConfig",
]
