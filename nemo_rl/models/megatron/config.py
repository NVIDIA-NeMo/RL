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

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import torch
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer import MegatronModule

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams


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
    offload_optimizer_for_refit: bool
    is_generation_colocated: Optional[bool]
    sampling_params: Optional[TrainingSamplingParams]
    final_padded_vocab_size: int


@dataclass
class ColocatedReshardPlan:
    """Setup-time plan for building the dedicated colocated inference model.

    Produced by `setup_model_and_optimizer` when the inference and training layouts differ.
    Consumed by the first `prepare_for_generation` call.
    """

    # Pre-wrap provider snapshot; build_inference_model mutates it to the inference layout.
    initial_model_provider: ModelProviderMixin
    # The resolved megatron_cfg the inference model runs with.
    inference_megatron_cfg: dict[str, Any]


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
    draft_model: Optional[MegatronModule] = None
    colocated_reshard_plan: Optional[ColocatedReshardPlan] = None
