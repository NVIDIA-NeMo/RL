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

from typing import Any, NotRequired, Optional, TypedDict, Union

from nemo_rl.models.generation.interfaces import GenerationConfig


class DTensorConfig(TypedDict):
    enabled: bool
    cpu_offload: bool
    sequence_parallel: bool
    activation_checkpointing: bool
    tensor_parallel_size: int
    context_parallel_size: int
    custom_parallel_plan: str


class TokenizerConfig(TypedDict):
    name: str
    chat_template: str


class PytorchOptimizerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


class SinglePytorchSchedulerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


SchedulerMilestones = dict[str, list[int]]


class DynamicBatchingConfig(TypedDict):
    # dynamic_batching improves performance by ensuring logprob and training microbatches
    # have a sufficent number of tokens to maximize GPU utilization. Specifically, variable length
    # responses are sorted by sequence length and bucketed into microbatches with a total
    # amount of tokens is approximately close to 'train_mb_tokens' and 'logprob_mb_tokens' for the
    # training and logprob stages respectively.
    enabled: bool
    train_mb_tokens: int
    logprob_mb_tokens: int
    sequence_length_round: int


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    learning_rate: float
    logprob_batch_size: int
    generation: Optional[GenerationConfig]
    generation_batch_size: NotRequired[
        int
    ]  # used in static batched (framework) generation
    precision: str
    dtensor_cfg: DTensorConfig
    dynamic_batching: DynamicBatchingConfig
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    max_grad_norm: Optional[Union[float, int]]
    fsdp_offload_enabled: bool
    activation_checkpointing_enabled: bool
    optimizer: NotRequired[PytorchOptimizerConfig] = None
    scheduler: NotRequired[list[SinglePytorchSchedulerConfig] | SchedulerMilestones] = (
        None
    )
