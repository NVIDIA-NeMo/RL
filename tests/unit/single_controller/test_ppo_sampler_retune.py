# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO warmup scheduling must preserve the selected sampler's admission policy."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    CustomSamplerConfig,
    InOrderSampler,
    ReadyFirstSamplerConfig,
    SamplerConfig,
    WeightFifoSamplerConfig,
    WindowedSamplerConfig,
    create_sampler,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor


@pytest.mark.parametrize(
    "sampler_config",
    [
        ReadyFirstSamplerConfig(max_staleness_versions=2),
        WeightFifoSamplerConfig(max_staleness_versions=2),
        WindowedSamplerConfig(max_staleness_versions=2),
    ],
    ids=["ready_first", "weight_fifo", "windowed"],
)
@pytest.mark.parametrize("trainer_version,start", [(0, 0), (0, 4), (4, 4)])
def test_ppo_retune_preserves_non_in_order_sampler(
    sampler_config: SamplerConfig, trainer_version: int, start: int
) -> None:
    controller = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    controller._is_ppo = True
    controller._trainer_version = trainer_version
    controller._algo_cfg = SimpleNamespace(policy_training_start_step=start)
    controller._async_cfg = SimpleNamespace(sampler=sampler_config)
    sampler = create_sampler(MagicMock(), sampler_config)
    controller._sampler = sampler
    capacity = sampler.required_buffer_capacity(groups_per_step=8)
    cursor = sampler.dispatch_index

    controller._retune_lookahead_versions()

    assert sampler.max_staleness_versions == 2
    assert sampler.required_buffer_capacity(groups_per_step=8) == capacity
    assert sampler.dispatch_index == cursor


def test_custom_in_order_sampler_keeps_its_warmup_schedule() -> None:
    class CustomInOrderSampler(InOrderSampler):
        pass

    controller = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    controller._is_ppo = True
    controller._trainer_version = 2
    controller._algo_cfg = SimpleNamespace(policy_training_start_step=4)
    # The runtime instance owns defaults even when a custom config forwards none.
    controller._async_cfg = SimpleNamespace(
        sampler=CustomSamplerConfig(target="custom:CustomInOrderSampler")
    )
    sampler = CustomInOrderSampler(
        MagicMock(), max_lookahead_versions=1, warmup_lookahead_versions=5
    )
    sampler.set_gate_window = MagicMock(wraps=sampler.set_gate_window)
    controller._sampler = sampler

    controller._retune_lookahead_versions()

    sampler.set_gate_window.assert_called_once_with(3)
    assert sampler.required_buffer_capacity(groups_per_step=8) == 48
