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

from nemo_rl.models.generation.vllm.config import VllmConfig

__all__ = [
    "VllmConfig",
    "VllmGeneration",
    "VllmGenerationWorker",
    "VllmAsyncGenerationWorker",
]


def __getattr__(name):
    # Qwen 3.5 smoke tests run several Ray actors that only need VllmConfig.
    # Avoid importing worker modules until a caller explicitly asks for them.
    if name == "VllmGeneration":
        from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

        return VllmGeneration
    if name == "VllmGenerationWorker":
        from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorker

        return VllmGenerationWorker
    if name == "VllmAsyncGenerationWorker":
        from nemo_rl.models.generation.vllm.vllm_worker_async import (
            VllmAsyncGenerationWorker,
        )

        return VllmAsyncGenerationWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
