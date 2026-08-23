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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    combine_param_group_overrides,
)
from pydantic import ValidationError

from nemo_rl.models.megatron.draft.optimizer import (
    DraftOptimizerConfigOverrideProvider,
    build_draft_optimizer_override_provider,
)
from nemo_rl.models.policy.draft_config import (
    DraftOptimizerConfig,
    Eagle3DraftConfig,
)


pytestmark = pytest.mark.mcore


def _combined_overrides_for_parameter(
    provider: DraftOptimizerConfigOverrideProvider,
    parameter: torch.nn.Parameter,
    name: str,
    *,
    policy_lr: float = 2.0e-3,
    policy_min_lr: float = 2.0e-4,
) -> dict[str, float]:
    overrides = provider.build_config_overrides(
        OptimizerConfigOverrideProviderContext(
            scheduler_config=MagicMock(),
            optimizer_config=OptimizerConfig(
                lr=policy_lr,
                min_lr=policy_min_lr,
                weight_decay=0.1,
            ),
            model=MagicMock(),
        )
    )
    assert overrides is not None
    return combine_param_group_overrides(
        [
            override
            for key, override in overrides.items()
            if key.matches(parameter, name)
        ]
    )


def _linear_scheduler(optimizer: torch.optim.Optimizer) -> OptimizerParamScheduler:
    return OptimizerParamScheduler(
        SimpleNamespace(param_groups=optimizer.param_groups),
        init_lr=0.0,
        max_lr=2.0e-3,
        min_lr=2.0e-4,
        lr_warmup_steps=2,
        lr_decay_steps=6,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=6,
        wd_incr_style="constant",
    )


@pytest.mark.parametrize(
    "config",
    [
        {"lr": 0.0},
        {"lr": -1.0},
        {"lr": 1.0e-5, "min_lr": -1.0e-6},
        {"lr": 1.0e-5, "min_lr": 2.0e-5},
        {"lr": 1.0e-5, "weight_decay": -0.1},
        {"lr": 1.0e-5, "learning_rate": 2.0e-5},
    ],
)
def test_draft_optimizer_config_rejects_invalid_values(
    config: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        DraftOptimizerConfig(**config)


def test_draft_optimizer_override_is_opt_in() -> None:
    configured_but_disabled = Eagle3DraftConfig(
        enabled=False,
        optimizer={"lr": 1.0e-5},
    )

    assert build_draft_optimizer_override_provider(None) is None
    assert (
        build_draft_optimizer_override_provider(Eagle3DraftConfig(enabled=True)) is None
    )
    assert build_draft_optimizer_override_provider(configured_but_disabled) is None


def test_draft_optimizer_provider_only_overrides_tagged_parameters() -> None:
    draft_parameter = torch.nn.Parameter(torch.ones(2, 2))
    draft_parameter.grad_norm_group = "draft"
    policy_parameter = torch.nn.Parameter(torch.ones(2, 2))
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, min_lr=1.0e-4, weight_decay=0.02)
    )

    assert _combined_overrides_for_parameter(
        provider, draft_parameter, "draft.weight"
    ) == {
        "max_lr": 1.0e-3,
        "min_lr": 1.0e-4,
        "start_wd": 0.02,
        "end_wd": 0.02,
    }
    assert (
        _combined_overrides_for_parameter(provider, policy_parameter, "policy.weight")
        == {}
    )


def test_draft_weight_decay_override_keeps_norm_and_bias_at_zero_decay() -> None:
    """Megatron's standard wd_mult=0.0 for 1-D params survives the draft override:
    the scheduler multiplies get_wd(group) * wd_mult, so draft norms/biases keep
    zero decay even when draft.optimizer.weight_decay is set."""
    from megatron.core.optimizer import get_standard_config_overrides

    bias_like = torch.nn.Parameter(torch.ones(4))
    bias_like.grad_norm_group = "draft"
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, weight_decay=0.02)
    )
    context = OptimizerConfigOverrideProviderContext(
        scheduler_config=MagicMock(),
        optimizer_config=OptimizerConfig(lr=2.0e-3, min_lr=2.0e-4, weight_decay=0.1),
        model=MagicMock(),
    )
    draft_overrides = provider.build_config_overrides(context)
    assert draft_overrides is not None
    standard_overrides = get_standard_config_overrides(context.optimizer_config)

    merged = {**standard_overrides, **draft_overrides}
    combined = combine_param_group_overrides(
        [
            override
            for key, override in merged.items()
            if key.matches(bias_like, "draft.norm.bias")
        ]
    )

    assert combined["start_wd"] == 0.02
    assert combined["wd_mult"] == 0.0
    assert combined["start_wd"] * combined["wd_mult"] == 0.0


def test_draft_optimizer_provider_rejects_incompatible_inherited_min_lr() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad_norm_group = "draft"
    provider = DraftOptimizerConfigOverrideProvider(DraftOptimizerConfig(lr=1.0e-5))

    with pytest.raises(ValueError, match="draft optimizer lr"):
        _combined_overrides_for_parameter(
            provider,
            parameter,
            "draft.weight",
            policy_lr=2.0e-5,
            policy_min_lr=2.0e-5,
        )


def test_real_scheduler_applies_distinct_draft_lr_trajectory() -> None:
    policy_parameter = torch.nn.Parameter(torch.ones(1))
    draft_parameter = torch.nn.Parameter(torch.ones(1))
    draft_parameter.grad_norm_group = "draft"
    provider = DraftOptimizerConfigOverrideProvider(
        DraftOptimizerConfig(lr=1.0e-3, min_lr=1.0e-4)
    )
    draft_overrides = _combined_overrides_for_parameter(
        provider, draft_parameter, "draft.weight"
    )
    optimizer = torch.optim.SGD(
        [
            {"params": [policy_parameter]},
            {"params": [draft_parameter], **draft_overrides},
        ],
        lr=2.0e-3,
    )
    scheduler = _linear_scheduler(optimizer)

    trajectories: list[tuple[float, float]] = []
    for step in (0, 2, 4, 6):
        scheduler.step(step - scheduler.num_steps)
        trajectories.append(
            (optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
        )

    assert trajectories == [
        (0.0, 0.0),
        (2.0e-3, 1.0e-3),
        (1.1e-3, 5.5e-4),
        (2.0e-4, 1.0e-4),
    ]


def test_draft_lr_override_produces_finite_stable_toy_updates() -> None:
    def run(*, draft_lr: float | None) -> tuple[list[float], float, float]:
        policy_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        draft_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        draft_group: dict[str, object] = {"params": [draft_parameter]}
        if draft_lr is not None:
            draft_group.update({"max_lr": draft_lr, "min_lr": draft_lr / 10})
        optimizer = torch.optim.SGD(
            [{"params": [policy_parameter]}, draft_group],
            lr=2.0e-3,
        )
        scheduler = _linear_scheduler(optimizer)
        losses: list[float] = []
        for _ in range(6):
            optimizer.zero_grad()
            loss = policy_parameter.square().sum() + draft_parameter.square().sum()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step(1)
        return losses, policy_parameter.item(), draft_parameter.item()

    baseline_losses, baseline_policy, baseline_draft = run(draft_lr=None)
    override_losses, override_policy, override_draft = run(draft_lr=1.0e-3)

    assert torch.isfinite(torch.tensor(baseline_losses + override_losses)).all()
    assert baseline_losses[-1] < baseline_losses[0]
    assert override_losses[-1] < override_losses[0]
    assert override_policy == baseline_policy
    assert override_draft > baseline_draft
