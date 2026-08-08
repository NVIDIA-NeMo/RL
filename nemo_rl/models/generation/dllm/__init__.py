# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Generation for masked diffusion language models (dLLMs).

Masked diffusion models decode by iteratively unmasking a fixed-width canvas
rather than by appending tokens. SGLang can serve LLaDA2.0 and SDAR through
``--dllm-algorithm``; LLaDA-8B is served by no engine. Generation here runs on the
training weights, in the same spirit as the ``megatron_generation`` backend.
"""

from nemo_rl.models.generation.dllm.denoise import (
    block_denoise,
    build_canvas,
    get_num_transfer_tokens,
    unpack_generations,
)
from nemo_rl.models.generation.dllm.dllm_generation import DllmGeneration

__all__ = [
    "DllmGeneration",
    "block_denoise",
    "build_canvas",
    "get_num_transfer_tokens",
    "unpack_generations",
]
