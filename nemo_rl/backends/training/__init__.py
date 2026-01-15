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
"""Training backend abstractions.

This module provides the TrainingBackend protocol and implementations for
different training backends (DTensor, Megatron).

Example:
    >>> from nemo_rl.backends.training import get_training_backend
    >>> backend = get_training_backend('dtensor')
    >>> backend.setup(config)
    >>> metrics = backend.train_step(batch, loss_fn)
"""

from nemo_rl.backends.training.base import (
    TrainingBackend,
    TrainingBackendConfig,
    get_training_backend,
    list_training_backends,
    register_training_backend,
)
from nemo_rl.backends.training.dtensor import DTensorBackend
from nemo_rl.backends.training.megatron import MegatronBackend

__all__ = [
    "TrainingBackend",
    "TrainingBackendConfig",
    "DTensorBackend",
    "MegatronBackend",
    "get_training_backend",
    "list_training_backends",
    "register_training_backend",
]
