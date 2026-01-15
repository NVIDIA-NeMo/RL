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
"""NeMo RL High-Level API.

This module provides the simplest possible interface for training RL models.
It enables 5-line training scripts:

Example:
    >>> import nemo_rl
    >>> 
    >>> result = nemo_rl.train(
    ...     model="Qwen/Qwen2.5-1.5B",
    ...     dataset="nvidia/OpenMathInstruct-2",
    ...     reward_fn=my_reward_function,
    ... )

For more control, use the individual trainer classes:
    >>> from nemo_rl import GRPOTrainer
    >>> trainer = GRPOTrainer.from_pretrained("Qwen/Qwen2.5-1.5B")
    >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")
"""

from nemo_rl.api.train import (
    TrainResult,
    train,
)
from nemo_rl.api.functional import (
    create_trainer,
    get_algorithm,
    list_algorithms,
)

__all__ = [
    # Main training function
    "train",
    "TrainResult",
    # Functional helpers
    "create_trainer",
    "get_algorithm",
    "list_algorithms",
]
