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

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

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


class Eagle3DraftConfig(BaseModel, extra="allow"):
    """Configuration for EAGLE-3 draft-model co-training with the policy."""

    speculator_type: Literal["eagle3"] = "eagle3"
    enabled: bool = False
    model_name: str | None = None
    loss_weight: float = 0.1
    num_layers: int | None = None
    aux_layer_indices: list[int] | None = None
    optimizer: DraftOptimizerConfig | None = None


def coerce_draft_config(
    config: "Eagle3DraftConfig | Mapping[str, Any] | None",
) -> Eagle3DraftConfig | None:
    """Accept either a validated model or a raw mapping at API boundaries.

    ``MasterConfig`` validation normally produces the model, but ``PolicyConfig``
    is a TypedDict, so callers that assemble one by hand still pass a plain dict.
    """
    if config is None or isinstance(config, Eagle3DraftConfig):
        return config
    return Eagle3DraftConfig.model_validate(config)


def draft_refit_enabled(
    config: "Eagle3DraftConfig | Mapping[str, Any] | None",
) -> bool:
    """Return whether generation must accept refitted draft weights."""
    coerced = coerce_draft_config(config)
    return coerced is not None and coerced.enabled
