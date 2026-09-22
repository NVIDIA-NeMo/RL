# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract test against MCore's actual legacy loss callback scaling."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.dynamic_context_parallel import cp_loss_multiplier

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize("base_cp", [1, 2])
@pytest.mark.parametrize("active_cp", [1, 2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 3])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("replicated_cp_loss", [False, True])
def test_mcore_legacy_loss_scaling(
    base_cp, active_cp, num_microbatches, per_token, replicated_cp_loss
):
    # MCore is optional in the standard CPU test environment.
    from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss

    original_loss = torch.tensor(2.0, requires_grad=True)
    multiplier = cp_loss_multiplier(
        active_cp_size=active_cp,
        schedule_cp_size=base_cp,
        num_microbatches=num_microbatches,
        replicated_cp_loss=replicated_cp_loss,
    )
    metrics = []
    loss, _ = forward_step_calc_loss(
        model=None,
        output_tensor=original_loss,
        loss_func=lambda value: (value * multiplier, {"loss": value.detach().item()}),
        config=SimpleNamespace(timers=None, calculate_per_token_loss=per_token),
        vp_stage=None,
        collect_non_loss_data=False,
        num_microbatches=num_microbatches,
        forward_data_store=metrics,
        cp_group_size=base_cp,
        is_last_stage=True,
    )
    loss.backward()
    expected_grad = 1.0 / active_cp if replicated_cp_loss else 1.0
    assert original_loss.grad.item() == pytest.approx(expected_grad)
    assert metrics == [{"loss": 2.0}]
