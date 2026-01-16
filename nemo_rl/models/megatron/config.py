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

from typing import Any, Callable, NamedTuple, Optional, TypedDict

import torch
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer import MegatronModule


class MegatronGenerationConfig(TypedDict):
    # Total GPU memory (in GB) allocated for KV cache buffers
    buffer_size_gb: int
    # Fraction of buffer reserved for guaranteed active requests
    buffer_guaranteed_fraction: float
    # Number of CUDA graphs to pre-compile for different batch sizes
    num_cuda_graphs: int
    # Size of each KV cache block in tokens (affects memory granularity)
    block_size_tokens: int
    # Enable CUDA graphs for prefill/context processing
    use_cuda_graphs_for_non_decode_steps: bool
    # Split long prefills into chunks for better memory management
    enable_chunked_prefill: bool
    # Unified memory usage level (0=disabled, higher values enable more aggressive paging)
    unified_memory_level: int
    # Maximum number of tokens to use in a single step. Analogous to vllm's max_num_batched_tokens.
    # Can cause OOM if set too high so should be tuned with buffer_size_gb if OOMing. If set too
    # low, then will only do 512 tokens at a time, which can be slow.
    max_tokens: int


## returned from validate_and_set_config
class RuntimeConfig(NamedTuple):
    """Runtime configuration for model training and inference.

    This contains all validated runtime settings needed for model initialization,
    parallelization, and training.
    """

    megatron_cfg: ConfigContainer
    model_cfg: Any
    dtype: torch.dtype
    optimizer_cpu_offload: bool
    offload_optimizer_for_logprob: bool
    is_generation_colocated: Optional[bool]
    final_padded_vocab_size: int


## returned from setup_model_and_optimizer
class ModelAndOptimizerState(NamedTuple):
    """Container for model and optimizer state.

    This named tuple holds all model-related state including the model itself,
    optimizer, scheduler, and metadata about the model type and configuration.
    """

    state: GlobalState
    model: MegatronModule
    optimizer: MegatronOptimizer
    scheduler: OptimizerParamScheduler
    checkpointing_context: dict[str, Any]
    param_sync_func: Optional[Callable]
