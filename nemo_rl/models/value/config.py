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

from typing import Any, NotRequired, TypedDict

from nemo_rl.models.policy import (
    DTensorConfig,
    DTensorConfigDisabled,
    DynamicBatchingConfig,
    DynamicBatchingConfigDisabled,
    PytorchOptimizerConfig,
    RewardModelConfig,
    SchedulerMilestones,
    SequencePackingConfig,
    SequencePackingConfigDisabled,
    SinglePytorchMilestonesConfig,
    SinglePytorchSchedulerConfig,
    TokenizerConfig,
)


class ValueConfig(TypedDict):
    """Configuration for Value models in PPO.

    Value models use a subset of PolicyConfig fields, excluding generation-specific
    and reference policy settings.
    """

    model_name: str
    tokenizer: TokenizerConfig

    # Training batch sizes
    train_global_batch_size: int
    train_micro_batch_size: int
    logprob_batch_size: NotRequired[int]  # Used for value inference batch size

    # Precision
    precision: str

    # Reward model config (value models use regression head)
    reward_model_cfg: RewardModelConfig

    # Backend configuration - only DTensor is supported for value models
    dtensor_cfg: DTensorConfig | DTensorConfigDisabled

    # HuggingFace config overrides
    hf_config_overrides: NotRequired[dict[str, Any]]

    # Batching strategies
    dynamic_batching: DynamicBatchingConfig | DynamicBatchingConfigDisabled
    sequence_packing: NotRequired[SequencePackingConfig | SequencePackingConfigDisabled]

    # Sequence length settings
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int

    # Gradient clipping
    max_grad_norm: NotRequired[float | int | None]

    # Checkpoint loading
    dequantize_base_checkpoint: NotRequired[bool]

    # Optimizer and scheduler
    optimizer: NotRequired[PytorchOptimizerConfig | None]
    scheduler: NotRequired[
        list[SinglePytorchSchedulerConfig | SinglePytorchMilestonesConfig]
        | SchedulerMilestones
        | None
    ]
