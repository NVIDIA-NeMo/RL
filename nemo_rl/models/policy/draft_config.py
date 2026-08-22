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

import math
from typing import Annotated, ClassVar, Literal, Self, TypeAlias

from pydantic import BaseModel, Field, model_validator


class DraftOptimizerConfig(BaseModel, extra="forbid"):
    """Optional optimizer schedule for draft-model parameters."""

    lr: Annotated[float, Field(gt=0)]
    min_lr: Annotated[float, Field(ge=0)] | None = None
    weight_decay: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_lr_range(self) -> Self:
        """Require the draft minimum learning rate to fit its schedule."""
        if self.min_lr is not None and self.min_lr > self.lr:
            raise ValueError("draft optimizer min_lr must not exceed lr")
        return self


class AlwaysDraftUpdateScheduleConfig(BaseModel, extra="forbid"):
    mode: Literal["always"] = "always"


class FixedDraftUpdateScheduleConfig(BaseModel, extra="forbid"):
    mode: Literal["fixed"] = "fixed"
    action: Literal["sparse_update", "refit_only"]
    fixed_interval: Annotated[int, Field(gt=0)]


class AdaptiveDraftUpdateScheduleConfig(BaseModel, extra="forbid"):
    mode: Literal["adaptive"] = "adaptive"
    action: Literal["sparse_update"] = "sparse_update"
    min_interval: Annotated[int, Field(gt=0)] = 10
    max_interval: Annotated[int, Field(gt=0)] = 100
    ewma_alpha: float = 0.1
    degradation_threshold: float = 0.02
    recovery_threshold: float = 0.01
    min_observations: Annotated[int, Field(gt=0)] = 20
    max_burst_updates: Annotated[int, Field(gt=0)] = 10

    @model_validator(mode="after")
    def validate_adaptive_schedule(self) -> Self:
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be at least min_interval")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        thresholds = (self.recovery_threshold, self.degradation_threshold)
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("adaptive thresholds must be finite")
        if not 0.0 <= self.recovery_threshold < self.degradation_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= recovery < degradation <= 1")
        return self


DraftUpdateScheduleConfig: TypeAlias = Annotated[
    AlwaysDraftUpdateScheduleConfig
    | FixedDraftUpdateScheduleConfig
    | AdaptiveDraftUpdateScheduleConfig,
    Field(discriminator="mode"),
]


class Eagle3DraftConfig(BaseModel, extra="allow"):
    """Configuration for EAGLE-3 draft-model co-training with the policy."""

    supports_context_parallel: ClassVar[bool] = False
    supports_sequence_packing: ClassVar[bool] = False
    supports_target_sequence_parallel: ClassVar[bool] = False

    speculator_type: Literal["eagle3"] = "eagle3"
    enabled: bool = False
    model_name: str | None = None
    loss_weight: float = 0.1
    num_layers: int | None = None
    aux_layer_indices: list[int] | None = None
    optimizer: DraftOptimizerConfig | None = None


class DFlashDraftConfig(BaseModel, extra="forbid"):
    """Configuration for body-only DFlash co-training with a live target."""

    supports_context_parallel: ClassVar[bool] = True
    supports_sequence_packing: ClassVar[bool] = True
    supports_target_sequence_parallel: ClassVar[bool] = True

    speculator_type: Literal["dflash"] = "dflash"
    enabled: bool = False
    model_name: str | None = None
    model_revision: str | None = None
    aux_layer_indices: None = Field(default=None, exclude=True, repr=False)
    loss_weight: Annotated[float, Field(gt=0)] = 0.1
    gamma: Annotated[int, Field(gt=0)]
    anchors_per_sample: Annotated[int, Field(gt=0)]
    mask_token_id: Annotated[int, Field(ge=0)]
    target_hidden_state_layer_ids: Annotated[list[int], Field(min_length=1)]
    num_layers: Annotated[int, Field(gt=0)] = 5
    seed: int = 0
    vocab_tile_size: Annotated[int, Field(gt=0)] = 256
    position_decay: Annotated[float, Field(gt=0, le=1)] = 1.0
    max_cp_boundary_exclusion_fraction: Annotated[
        float,
        Field(ge=0, le=1),
    ] = 0.25
    optimizer: DraftOptimizerConfig | None = None
    update_schedule: DraftUpdateScheduleConfig = Field(
        default_factory=AlwaysDraftUpdateScheduleConfig
    )
    update_probe_enabled: bool = False

    @model_validator(mode="after")
    def validate_target_taps(self) -> Self:
        """Reject ambiguous and out-of-range layer taps before model creation."""
        if any(layer_id < 0 for layer_id in self.target_hidden_state_layer_ids):
            raise ValueError("target hidden-state layer IDs must be non-negative")
        if len(set(self.target_hidden_state_layer_ids)) != len(
            self.target_hidden_state_layer_ids
        ):
            raise ValueError("target hidden-state layer IDs must be unique")
        return self


class DSparkDraftConfig(BaseModel, extra="forbid"):
    """Configuration for DSpark co-training with target-owned embeddings/head."""

    supports_context_parallel: ClassVar[bool] = True
    supports_sequence_packing: ClassVar[bool] = True
    supports_target_sequence_parallel: ClassVar[bool] = True

    speculator_type: Literal["dspark"] = "dspark"
    enabled: bool = False
    model_name: str | None = None
    model_revision: str | None = None
    loss_weight: Annotated[float, Field(gt=0)] = 0.1
    aux_layer_indices: None = Field(default=None, exclude=True, repr=False)
    block_size: Annotated[int, Field(gt=1)]
    anchors_per_sample: Annotated[int, Field(gt=0)]
    mask_token_id: Annotated[int, Field(ge=0)]
    target_hidden_state_layer_ids: Annotated[list[int], Field(min_length=1)]
    num_layers: Annotated[int, Field(gt=0)] = 5
    draft_vocab_size: Annotated[int, Field(gt=0)] | None = None
    markov_rank: Annotated[int, Field(gt=0)] = 256
    markov_head_type: Literal["vanilla"] = "vanilla"
    confidence_enabled: bool = True
    confidence_with_markov: bool = True
    ce_loss_weight: Annotated[float, Field(ge=0)] = 0.1
    tv_loss_weight: Annotated[float, Field(ge=0)] = 0.9
    confidence_loss_weight: Annotated[float, Field(ge=0)] = 1.0
    loss_decay_gamma: Annotated[float, Field(gt=0)] = 4.0
    seed: int = 0
    vocab_tile_size: Annotated[int, Field(gt=0)] = 256
    max_cp_boundary_exclusion_fraction: Annotated[
        float,
        Field(ge=0, le=1),
    ] = 0.25
    optimizer: DraftOptimizerConfig | None = None
    update_schedule: DraftUpdateScheduleConfig = Field(
        default_factory=AlwaysDraftUpdateScheduleConfig
    )
    update_probe_enabled: bool = False

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        """Reject invalid taps and confidence dependencies before model creation."""
        if any(layer_id < 0 for layer_id in self.target_hidden_state_layer_ids):
            raise ValueError("target hidden-state layer IDs must be non-negative")
        if len(set(self.target_hidden_state_layer_ids)) != len(
            self.target_hidden_state_layer_ids
        ):
            raise ValueError("target hidden-state layer IDs must be unique")
        if self.confidence_with_markov and not self.confidence_enabled:
            raise ValueError("confidence_with_markov requires confidence_enabled")
        if not any(
            weight > 0
            for weight in (
                self.ce_loss_weight,
                self.tv_loss_weight,
                self.confidence_loss_weight,
            )
        ):
            raise ValueError("at least one DSpark loss weight must be positive")
        return self

    def resolve_draft_vocab_size(self, *, target_vocab_size: int) -> int:
        """Require the online draft to share the live target vocabulary."""
        if target_vocab_size <= 0:
            raise ValueError("live target vocabulary size must be positive")
        if (
            self.draft_vocab_size is not None
            and self.draft_vocab_size != target_vocab_size
        ):
            raise ValueError(
                "DSpark draft vocabulary must match the live target vocabulary"
            )
        return target_vocab_size


DraftConfig: TypeAlias = Eagle3DraftConfig | DFlashDraftConfig | DSparkDraftConfig


def draft_refit_enabled(config: DraftConfig | None) -> bool:
    """Return whether generation must accept refitted draft weights."""
    return config is not None and config.enabled
