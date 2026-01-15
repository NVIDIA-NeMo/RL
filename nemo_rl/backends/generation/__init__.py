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
"""Generation backend abstractions.

This module provides the GenerationBackend protocol and implementations for
different generation backends (vLLM, Megatron).

Example:
    >>> from nemo_rl.backends.generation import get_generation_backend
    >>> backend = get_generation_backend('vllm')
    >>> backend.setup(config)
    >>> outputs = backend.generate(prompts)
"""

from nemo_rl.backends.generation.base import (
    GenerationBackend,
    GenerationBackendConfig,
    get_generation_backend,
    list_generation_backends,
    register_generation_backend,
)
from nemo_rl.backends.generation.megatron import MegatronInferenceBackend
from nemo_rl.backends.generation.vllm import VLLMBackend

__all__ = [
    "GenerationBackend",
    "GenerationBackendConfig",
    "VLLMBackend",
    "MegatronInferenceBackend",
    "get_generation_backend",
    "register_generation_backend",
    "list_generation_backends",
]
