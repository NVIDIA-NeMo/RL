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
"""The train pump must close a step whose batch lost prompts to rollout faults.

Job 5982927 wedged after step 7 when two prompts hit 500s from one sick env
server: ``InOrderSampler`` only offers groups stamped for the current step, so
the 126 survivors could never reach the 128 the pump waited for. These tests
drive the real sampler with the pump's selection loop to pin that down.
"""

import asyncio
from collections import Counter
from typing import Optional

from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSampler

NUM_PROMPTS_PER_STEP = 128
MIN_GROUPS_FOR_STREAMING_TRAIN = 32


class _FakeMeta:
    """Stands in for KVBatchMeta, which only needs to concat here."""

    def concat(self, *others: "_FakeMeta") -> "_FakeMeta":
        return self


class _FakeBuffer:
    """Enough of TQReplayBuffer for the sampler's index bookkeeping."""

    def __init__(self) -> None:
        self.meta_list: list[_FakeMeta] = []
        self.start_weight_list: list[int] = []
        self.target_step_list: list[Optional[int]] = []
        self.ready_list: list[bool] = []

    def commit(self, target_step: int, count: int) -> None:
        for _ in range(count):
            self.meta_list.append(_FakeMeta())
            self.start_weight_list.append(target_step)
            self.target_step_list.append(target_step)
            self.ready_list.append(True)

    async def remove(self, idxs: list[int], remove_in_dp: bool) -> int:
        for i in sorted(idxs, reverse=True):
            del self.meta_list[i]
            del self.start_weight_list[i]
            del self.target_step_list[i]
            del self.ready_list[i]
        return len(idxs)


def _make_sampler() -> tuple[InOrderSampler, _FakeBuffer]:
    buffer = _FakeBuffer()
    return InOrderSampler(buffer, max_lookahead_versions=1), buffer  # type: ignore[arg-type]


async def _drain_step(
    sampler: InOrderSampler,
    *,
    trainer_version: int,
    shortfall: Counter[int],
    max_iterations: int = 64,
) -> tuple[int, bool]:
    """Run the pump's selection loop for one step.

    Returns the groups it collected and whether the step closed, rather than
    spinning forever the way the real pump does when the batch cannot fill.
    """
    groups_dispatched = 0
    for _ in range(max_iterations):
        groups_wanted = NUM_PROMPTS_PER_STEP - shortfall[trainer_version]
        if groups_dispatched >= groups_wanted:
            return groups_dispatched, True
        max_prompt_groups = groups_wanted - groups_dispatched
        min_prompt_groups = min(MIN_GROUPS_FOR_STREAMING_TRAIN, max_prompt_groups)
        train_meta, num_groups = await sampler.select(
            current_train_weight=trainer_version,
            min_prompt_groups=min_prompt_groups,
            max_prompt_groups=max_prompt_groups,
        )
        if train_meta is None:
            continue
        groups_dispatched += num_groups
    return groups_dispatched, False


def test_full_batch_closes() -> None:
    sampler, buffer = _make_sampler()
    buffer.commit(target_step=0, count=NUM_PROMPTS_PER_STEP)

    collected, closed = asyncio.run(
        _drain_step(sampler, trainer_version=0, shortfall=Counter())
    )

    assert closed
    assert collected == NUM_PROMPTS_PER_STEP


def test_short_batch_wedges_without_the_shortfall() -> None:
    """The 5982927 failure: two dropped prompts and the step never closes."""
    sampler, buffer = _make_sampler()
    buffer.commit(target_step=0, count=NUM_PROMPTS_PER_STEP - 2)

    collected, closed = asyncio.run(
        _drain_step(sampler, trainer_version=0, shortfall=Counter())
    )

    assert not closed
    assert collected == NUM_PROMPTS_PER_STEP - 2


def test_short_batch_closes_once_the_shortfall_is_recorded() -> None:
    sampler, buffer = _make_sampler()
    buffer.commit(target_step=0, count=NUM_PROMPTS_PER_STEP - 2)

    collected, closed = asyncio.run(
        _drain_step(sampler, trainer_version=0, shortfall=Counter({0: 2}))
    )

    assert closed
    assert collected == NUM_PROMPTS_PER_STEP - 2


def test_shortfall_recorded_after_the_step_opens_still_closes() -> None:
    """A re-dispatch can still be in backoff when the trainer starts waiting.

    The pump re-reads the shortfall every pass for exactly this case, so a drop
    landing mid-step has to release it rather than leave it spinning.
    """
    sampler, buffer = _make_sampler()
    buffer.commit(target_step=0, count=NUM_PROMPTS_PER_STEP - 2)
    shortfall: Counter[int] = Counter()

    _, closed = asyncio.run(
        _drain_step(
            sampler, trainer_version=0, shortfall=shortfall, max_iterations=4
        )
    )
    assert not closed

    shortfall[0] = 2
    _, closed = asyncio.run(
        _drain_step(sampler, trainer_version=0, shortfall=shortfall)
    )

    assert closed


def test_shortfall_does_not_leak_into_the_next_step() -> None:
    sampler, buffer = _make_sampler()
    buffer.commit(target_step=0, count=NUM_PROMPTS_PER_STEP - 2)
    buffer.commit(target_step=1, count=NUM_PROMPTS_PER_STEP)
    shortfall: Counter[int] = Counter({0: 2})

    _, closed = asyncio.run(
        _drain_step(sampler, trainer_version=0, shortfall=shortfall)
    )
    assert closed
    shortfall.pop(0, None)

    collected, closed = asyncio.run(
        _drain_step(sampler, trainer_version=1, shortfall=shortfall)
    )

    assert closed
    assert collected == NUM_PROMPTS_PER_STEP
