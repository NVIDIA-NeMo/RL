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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

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

    draft_optimizer: DraftOptimizerConfig | None

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        """Combine standard Megatron groups with the opt-in draft schedule."""
        overrides = super().build_config_overrides(context) or {}
        draft_override = ParamGroupOverride()
        if self.draft_optimizer is not None:
            minimum_lr = self.draft_optimizer.min_lr
            if minimum_lr is None:
                minimum_lr = context.optimizer_config.min_lr
            if minimum_lr is not None and minimum_lr > self.draft_optimizer.lr:
                raise ValueError(
                    "draft optimizer lr must be at least the inherited optimizer min_lr"
                )

            draft_override["max_lr"] = self.draft_optimizer.lr
            if minimum_lr is not None:
                draft_override["min_lr"] = minimum_lr
            if self.draft_optimizer.weight_decay is not None:
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
    """Build a draft group selector whenever online draft training is enabled."""
    if draft_config is None or not draft_config.enabled:
        return None
    return DraftOptimizerConfigOverrideProvider(draft_config.optimizer)


def _optimizer_param_group_owners(optimizer: Any) -> tuple[Any, ...]:
    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if chained_optimizers is not None:
        return tuple(
            owner
            for chained_optimizer in chained_optimizers
            for owner in _optimizer_param_group_owners(chained_optimizer)
        )
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    return () if base_optimizer is None else (base_optimizer,)


@contextmanager
def suspend_draft_optimizer_groups(optimizer: Any) -> Iterator[None]:
    """Temporarily remove homogeneous draft-only optimizer parameter groups."""
    planned: list[tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for current in _optimizer_param_group_owners(optimizer):
        original = list(current.param_groups)
        kept: list[dict[str, Any]] = []
        for group in original:
            parameters = group.get("params", ())
            has_draft = any(_is_draft_parameter(parameter) for parameter in parameters)
            has_policy = any(
                not _is_draft_parameter(parameter) for parameter in parameters
            )
            if has_draft and has_policy:
                raise RuntimeError(
                    "optimizer parameter group mixes policy and draft parameters"
                )
            if not has_draft:
                kept.append(group)
        planned.append((current, original, kept))

    try:
        for current, _, kept in planned:
            current.param_groups[:] = kept
        yield
    finally:
        for current, original, _ in reversed(planned):
            current.param_groups[:] = original
