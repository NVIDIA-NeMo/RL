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
"""Backend-independent score collection stays lazy and preserves positions."""

import pytest
import torch
from test_fast_selection import confidence_batch, make_schedule
from just_grpo.diffusion.denoising_schedule import aggregate_diffusion_logprobs


@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_aggregation_is_lazy_and_preserves_original_positions(microbatch_size):
    data = confidence_batch()
    schedule = make_schedule(data)
    produced = 0
    scored = 0
    empty_views = 0

    def trajectories():
        nonlocal produced
        for view in schedule.return_schedule(microbatch_size):
            assert produced == scored
            produced += 1
            yield view

    def score(view):
        nonlocal scored, empty_views
        assert not torch.is_grad_enabled()
        assert produced == scored + 1
        scored += 1
        empty_views += int(not view["token_mask"].any())
        return view["target_ids"].float() / 10

    result = aggregate_diffusion_logprobs(
        trajectories(),
        score,
        original_shape=data["input_ids"].shape,
        device=torch.device("cpu"),
    )
    assert produced == scored
    if microbatch_size == 1:
        assert empty_views > 0  # Do not skip distributed scoring collectives.
    torch.testing.assert_close(
        result, data["input_ids"].float() / 10 * data["token_mask"]
    )
    assert not result.requires_grad


def test_aggregator_follows_schedule_level_order(monkeypatch):
    data = confidence_batch()
    schedule = make_schedule(data)
    order = [2, 0, 3, 1]
    produced, consumed = [], []

    def levels():
        for step in order:
            assert produced == consumed
            produced.append(step)
            view = schedule.generate_single_trajectory(step)
            view["test_level"] = torch.full((schedule.num_samples,), step)
            yield view

    def score(view):
        consumed.append(int(view["test_level"][0]))
        return view["target_ids"].float()

    result = aggregate_diffusion_logprobs(
        levels(),
        score,
        original_shape=data["input_ids"].shape,
        device=torch.device("cpu"),
    )
    assert produced == consumed == order
    torch.testing.assert_close(result, data["input_ids"].float() * data["token_mask"])
