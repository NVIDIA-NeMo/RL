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
"""Environment modules for NeMo RL.

This module provides environment interfaces and implementations for
computing rewards in reinforcement learning.

Key Components:
    - Environment: Easy-to-extend base class (only override score() for simple cases)
    - EnvironmentInterface: Low-level interface for all environments
    - FunctionalRewardWrapper: Adapter for simple callable reward functions
    - Various environment implementations (math, code, etc.)

Quick Start - Choose the simplest approach for your needs:

1. **Simple callable** (easiest):
    >>> reward_fn = lambda p, r: 1.0 if "correct" in r else 0.0
    >>> trainer.fit(dataset="data", reward_fn=reward_fn)

2. **Environment subclass** (when you need state or complex logic):
    >>> from nemo_rl.environments import Environment
    >>>
    >>> class MyReward(Environment):
    ...     def score(self, prompt: str, response: str) -> float:
    ...         return 1.0 if "correct" in response else 0.0
    >>>
    >>> env = MyReward()
    >>> trainer.fit(dataset="data", environment=env)

3. **FunctionalRewardWrapper** (when you have a function but need env interface):
    >>> from nemo_rl.environments import FunctionalRewardWrapper
    >>>
    >>> def my_reward(prompt: str, response: str) -> float:
    ...     return 1.0 if "correct" in response else 0.0
    >>>
    >>> env = FunctionalRewardWrapper(my_reward)
"""

from nemo_rl.environments.base import (
    Environment,
    EnvironmentConfig,
    SimpleEnvironment,
    StatefulEnvironment,
)
from nemo_rl.environments.functional_reward import (
    FunctionalRewardConfig,
    FunctionalRewardWrapper,
    RewardFunction,
    batched_reward,
    create_reward_wrapper,
    reward_function,
    validate_reward_callable,
    wrap_reward_callable,
)
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

__all__ = [
    # Base Environment class (easy to extend)
    "Environment",
    "EnvironmentConfig",
    "SimpleEnvironment",
    "StatefulEnvironment",
    # Interfaces
    "EnvironmentInterface",
    "EnvironmentReturn",
    # Functional rewards
    "FunctionalRewardWrapper",
    "FunctionalRewardConfig",
    "RewardFunction",
    "create_reward_wrapper",
    "reward_function",
    "batched_reward",
    # Callable helpers
    "wrap_reward_callable",
    "validate_reward_callable",
]
