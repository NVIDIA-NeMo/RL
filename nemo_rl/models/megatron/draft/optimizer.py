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

from dataclasses import dataclass
from typing import Protocol

import torch
from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)
from megatron.core.optimizer import ParamKey, ParamPredicate
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

from nemo_rl.models.policy.draft_config import DraftOptimizerConfig


class DraftOptimizerConfigOwner(Protocol):
    """Draft config fields required to opt in to optimizer overrides."""

    enabled: bool
    optimizer: DraftOptimizerConfig | None


def _is_draft_parameter(parameter: torch.nn.Parameter) -> bool:
    return getattr(parameter, "grad_norm_group", None) == "draft"


@dataclass
class DraftOptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Add scheduler overrides only for parameters owned by a draft model."""

    draft_optimizer: DraftOptimizerConfig

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        """Combine standard Megatron groups with the opt-in draft schedule."""
        overrides = super().build_config_overrides(context) or {}
        minimum_lr = self.draft_optimizer.min_lr
        if minimum_lr is None:
            minimum_lr = context.optimizer_config.min_lr
        if minimum_lr is not None and minimum_lr > self.draft_optimizer.lr:
            raise ValueError(
                "draft optimizer lr must be at least the inherited optimizer min_lr"
            )

        draft_override = ParamGroupOverride(max_lr=self.draft_optimizer.lr)
        if minimum_lr is not None:
            draft_override["min_lr"] = minimum_lr
        if self.draft_optimizer.weight_decay is not None:
            # Megatron's standard overrides still set wd_mult=0.0 for norm/bias
            # and 1-D draft params, and the scheduler multiplies:
            # weight_decay = get_wd(group) * wd_mult. So this override changes
            # decay only for weight matrices; draft norms/biases stay at 0,
            # matching the main model's convention.
            draft_override["start_wd"] = self.draft_optimizer.weight_decay
            draft_override["end_wd"] = self.draft_optimizer.weight_decay

        overrides[
            ParamKey(
                predicate=ParamPredicate(
                    name="draft_parameter",
                    fn=_is_draft_parameter,
                )
            )
        ] = draft_override
        return overrides


def build_draft_optimizer_override_provider(
    draft_config: DraftOptimizerConfigOwner | None,
) -> DraftOptimizerConfigOverrideProvider | None:
    """Build an override provider only when a draft schedule is configured."""
    if (
        draft_config is None
        or not draft_config.enabled
        or draft_config.optimizer is None
    ):
        return None
    return DraftOptimizerConfigOverrideProvider(draft_config.optimizer)
