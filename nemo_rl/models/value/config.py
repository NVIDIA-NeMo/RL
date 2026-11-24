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

from pydantic import BaseModel, Field, PositiveInt

from nemo_rl.models.policy import (
    AutomodelConfig,
    AutomodelConfigDisabled,
    DynamicBatchingConfig,
    DynamicBatchingConfigDisabled,
    MegatronConfig,
    MegatronConfigDisabled,
    PytorchOptimizerConfig,
    RewardModelConfig,
    SchedulerMilestones,
    SequencePackingConfig,
    SequencePackingConfigDisabled,
    SinglePytorchMilestonesConfig,
    SinglePytorchSchedulerConfig,
    TokenizerConfig,
)


class PrivilegedValueConfig(BaseModel, extra="forbid"):
    """``value.privileged_value``: prefix the critic input with a reference block.

    The block is built per prompt from its ``extra_env_info`` by the function at
    ``prefix_builder_fqn`` and is never shown to the policy. Keep
    ``max_total_tokens`` and ``prefix_builder_kwargs`` identical to the
    critic-pretrain run a critic is warm-started from: they change which content
    is truncated, a distribution shift the warm start was not trained on.
    """

    # Build privileged critic inputs. False leaves the critic blind.
    enabled: bool
    # Token budget for the whole block.
    max_total_tokens: PositiveInt
    # Dotted path to build_privilege_prefix(extra_env_info, tokenizer, cfg) ->
    # (block, stats); see nemo_rl.algorithms.ppo_utils.privileged_value.
    prefix_builder_fqn: str
    # Options the builder reads from cfg, e.g. the SWE builder's per-field caps.
    prefix_builder_kwargs: dict[str, Any] = Field(default_factory=dict)


class ValueResourcesConfig(BaseModel, extra="forbid"):
    num_nodes: int | None = Field(
        default=None,
        description="Number of full nodes reserved for a dedicated value model.",
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

    # Backend configuration - Automodel or Megatron
    automodel_cfg: NotRequired[AutomodelConfig | AutomodelConfigDisabled]
    megatron_cfg: NotRequired[MegatronConfig | MegatronConfigDisabled]
    dedicated: NotRequired[bool]
    resources: NotRequired[ValueResourcesConfig]

    # HuggingFace config overrides
    hf_config_overrides: NotRequired[dict[str, Any]]

    # Batching strategies
    dynamic_batching: DynamicBatchingConfig | DynamicBatchingConfigDisabled
    sequence_packing: NotRequired[SequencePackingConfig | SequencePackingConfigDisabled]

    # Sequence length settings
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int

    # Privileged value: prefix the critic's input with a reference block the
    # policy never sees. max_total_sequence_length and the packing token budgets
    # must then cover policy length + the block budget.
    privileged_value: NotRequired[PrivilegedValueConfig]

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
