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


class SwePrivilegedCriticCaps(BaseModel, extra="forbid"):
    """Per-field token ceilings within the reference block's total budget.

    They stop one pathological field from consuming the budget (one corpus
    instance carries a 323k-token golden patch, and pass_to_pass reaches 125k)
    and deliberately sum to more than the default total, so the total stays the
    binding constraint.
    """

    golden_patch: PositiveInt = 24576
    test_patch: PositiveInt = 16384
    pass_to_pass: PositiveInt = 4096
    fail_to_pass: PositiveInt = 4096


class SwePrivilegedCriticConfig(BaseModel, extra="forbid"):
    """``value.swe_privileged_critic``: prefix the critic input with a reference block.

    The block carries the accepted fix and the grading tests of the rollout's SWE
    instance (see ``nemo_rl.algorithms.swe_privileged_critic``). Keep
    ``max_total_tokens`` and ``caps`` identical to the critic-pretrain run a critic
    is warm-started from: they change which content is truncated, a distribution
    shift the warm start was not trained on.
    """

    # Build privileged critic inputs. False leaves the critic blind.
    enabled: bool = False
    # Token budget for the whole block, spent in priority order fail_to_pass >
    # golden_patch > test_patch > pass_to_pass. Measured over 400 instances (real
    # tokenizer) the untruncated block is median 5467 / p90 17683 / p99 77911
    # tokens, so 32768 truncates ~4.8% of them; privilege/frac_truncated reports
    # the share a run actually cuts.
    max_total_tokens: PositiveInt = 32768
    caps: SwePrivilegedCriticCaps = Field(default_factory=SwePrivilegedCriticCaps)


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

    # HuggingFace config overrides
    hf_config_overrides: NotRequired[dict[str, Any]]

    # Batching strategies
    dynamic_batching: DynamicBatchingConfig | DynamicBatchingConfigDisabled
    sequence_packing: NotRequired[SequencePackingConfig | SequencePackingConfigDisabled]

    # Sequence length settings
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int

    # SWE privileged critic: prefix the critic's input with a reference block
    # (accepted fix + grading tests). max_total_sequence_length and the packing
    # token budgets must then cover policy length + the block budget.
    swe_privileged_critic: NotRequired[SwePrivilegedCriticConfig]

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
