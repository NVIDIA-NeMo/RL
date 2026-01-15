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
"""Backend abstractions for NeMo RL.

This module provides clean abstractions for training and generation backends,
allowing backend selection via simple string parameters.

Example:
    >>> from nemo_rl.backends import get_training_backend, get_generation_backend
    >>> training_backend = get_training_backend('dtensor')
    >>> generation_backend = get_generation_backend('vllm')
"""

from nemo_rl.backends.factory import BackendFactory
from nemo_rl.backends.generation import (
    GenerationBackend,
    GenerationBackendConfig,
    MegatronInferenceBackend,
    VLLMBackend,
    get_generation_backend,
    list_generation_backends,
    register_generation_backend,
)
from nemo_rl.backends.training import (
    DTensorBackend,
    MegatronBackend,
    TrainingBackend,
    TrainingBackendConfig,
    get_training_backend,
    list_training_backends,
    register_training_backend,
)

__all__ = [
    # Factory
    "BackendFactory",
    # Training backends
    "TrainingBackend",
    "TrainingBackendConfig",
    "DTensorBackend",
    "MegatronBackend",
    "get_training_backend",
    "register_training_backend",
    "list_training_backends",
    # Generation backends
    "GenerationBackend",
    "GenerationBackendConfig",
    "VLLMBackend",
    "MegatronInferenceBackend",
    "get_generation_backend",
    "register_generation_backend",
    "list_generation_backends",
]
