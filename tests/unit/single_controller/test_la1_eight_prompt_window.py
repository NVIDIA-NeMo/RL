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

"""Exercise the production 8-prompt LA1 window with adversarial completion order."""

import asyncio

import pytest

from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSampler
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    validate_sampler_buffer_capacity,
)
from tests.unit.single_controller.test_sampler_interface import FakeBuffer


@pytest.mark.parametrize("capacity", [8, 15, 16])
def test_eight_prompt_la1_capacity_guard(capacity: int) -> None:
    config = AsyncRLConfig(max_buffered_rollouts=capacity)
    if capacity < 16:
        with pytest.raises(ValueError, match="required capacity \\(16\\)"):
            validate_sampler_buffer_capacity(
                config, required_capacity=16, sampler_name="in_order"
            )
    else:
        validate_sampler_buffer_capacity(
            config, required_capacity=16, sampler_name="in_order"
        )


def test_eight_prompt_window_waits_for_current_batch_then_consumes_exactly_once() -> (
    None
):
    async def exercise() -> None:
        buffer = FakeBuffer()
        sampler = InOrderSampler(buffer, max_lookahead_versions=1)
        assert sampler.required_buffer_capacity(groups_per_step=8) == 16
        version = 0
        for batch in (0, 1):
            assert await sampler.admit(trainer_version_fn=lambda: version) == batch
            for prompt in range(8):
                buffer.add(f"b{batch}p{prompt}", weight=0, target_step=batch)

        # Future batch is complete; the last prompt of the current batch is not.
        buffer.ready_list[7] = False
        meta, count = await sampler.select(
            current_train_weight=0, min_prompt_groups=8, max_prompt_groups=8
        )
        assert meta is None and count == 0
        assert len(buffer.ready_list) == 16
        admission = asyncio.create_task(
            sampler.admit(trainer_version_fn=lambda: version)
        )
        await asyncio.sleep(0.02)
        assert not admission.done()
        buffer.ready_list[7] = True
        consumed = []
        for batch in (0, 1):
            meta, count = await sampler.select(
                current_train_weight=batch, min_prompt_groups=8, max_prompt_groups=8
            )
            assert count == 8
            assert meta.sample_ids == [f"b{batch}p{prompt}_g0" for prompt in range(8)]
            consumed.extend(meta.sample_ids)
            assert await sampler.evict(current_train_weight=batch) == 0
            if batch == 0:
                version = 1
                assert await asyncio.wait_for(admission, timeout=1) == 2
        assert len(set(consumed)) == len(consumed) == 16
        assert not buffer.ready_list

    asyncio.run(exercise())
