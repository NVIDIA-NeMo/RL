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

import pytest
from pydantic import TypeAdapter

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSampler,
    InOrderSamplerConfig,
    SamplerConfig,
    WindowedSampler,
    WindowedSamplerConfig,
    required_buffer_capacity_for_config,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    validate_sampler_buffer_capacity,
)

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

    def commit_unstamped(self, weight: int, count: int) -> None:
        """Commit the way a non-gated sampler does: no target_step stamp."""
        for _ in range(count):
            self.meta_list.append(_FakeMeta())
            self.start_weight_list.append(weight)
            self.target_step_list.append(None)
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


def _make_windowed(staleness: int = 1) -> tuple[WindowedSampler, _FakeBuffer]:
    buffer = _FakeBuffer()
    return (
        WindowedSampler(buffer, max_staleness_versions=staleness),  # type: ignore[arg-type]
        buffer,
    )


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
        _drain_step(sampler, trainer_version=0, shortfall=shortfall, max_iterations=4)
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


# ── windowed selection ──────────────────────────────────────────────────────


def test_windowed_does_not_stamp_a_target_step() -> None:
    """No stamp means no batch identity, hence nothing for a drop to shorten."""
    sampler, _ = _make_windowed()

    target_step = asyncio.run(sampler.admit(trainer_version_fn=lambda: 0))

    assert target_step is None


def test_windowed_closes_a_step_that_lost_prompts() -> None:
    """The wedge is specific to in_order: windowed backfills from the window.

    Two prompts are lost from what would have been this step's batch and no
    shortfall is ever recorded, because ``target_step`` is None. Groups from a
    later dispatch fill the gap, so the step still closes at the full count.
    """
    sampler, buffer = _make_windowed(staleness=1)
    buffer.commit_unstamped(weight=0, count=NUM_PROMPTS_PER_STEP - 2)
    buffer.commit_unstamped(weight=0, count=2)

    collected, closed = asyncio.run(
        _drain_step(sampler, trainer_version=0, shortfall=Counter())
    )

    assert closed
    assert collected == NUM_PROMPTS_PER_STEP


def test_windowed_selects_across_the_staleness_window() -> None:
    sampler, buffer = _make_windowed(staleness=1)
    buffer.commit_unstamped(weight=0, count=64)
    buffer.commit_unstamped(weight=1, count=64)

    collected, closed = asyncio.run(
        _drain_step(sampler, trainer_version=1, shortfall=Counter())
    )

    assert closed
    assert collected == NUM_PROMPTS_PER_STEP


def test_windowed_ignores_groups_older_than_the_window() -> None:
    sampler, buffer = _make_windowed(staleness=1)
    buffer.commit_unstamped(weight=0, count=NUM_PROMPTS_PER_STEP)

    _, num_groups = asyncio.run(
        sampler.select(
            current_train_weight=2,
            min_prompt_groups=1,
            max_prompt_groups=NUM_PROMPTS_PER_STEP,
        )
    )

    assert num_groups == 0


# ── capacity validation ─────────────────────────────────────────────────────


def test_windowed_declares_a_capacity_floor() -> None:
    """Without a floor the validation skips windowed and a too-small buffer
    presents as a silent hang instead of a config error."""
    required = required_buffer_capacity_for_config(
        WindowedSamplerConfig(max_staleness_versions=4), NUM_PROMPTS_PER_STEP
    )

    assert required == NUM_PROMPTS_PER_STEP


def test_windowed_capacity_validation_rejects_a_too_small_buffer() -> None:
    cfg = WindowedSamplerConfig(max_staleness_versions=4)
    required = required_buffer_capacity_for_config(cfg, NUM_PROMPTS_PER_STEP)

    with pytest.raises(ValueError, match="max_buffered_rollouts"):
        validate_sampler_buffer_capacity(
            AsyncRLConfig(max_buffered_rollouts=64),
            required_capacity=required,
            sampler_name="windowed",
        )


def test_windowed_capacity_validation_accepts_the_sweep_setting() -> None:
    cfg = WindowedSamplerConfig(max_staleness_versions=4)

    validate_sampler_buffer_capacity(
        AsyncRLConfig(max_buffered_rollouts=640),
        required_capacity=required_buffer_capacity_for_config(
            cfg, NUM_PROMPTS_PER_STEP
        ),
        sampler_name="windowed",
    )


def test_gated_capacity_floor_is_unchanged() -> None:
    """The windowed floor must not weaken the gated samplers' requirement."""
    required = required_buffer_capacity_for_config(
        InOrderSamplerConfig(max_lookahead_versions=1), NUM_PROMPTS_PER_STEP
    )

    assert required == NUM_PROMPTS_PER_STEP * 2


# ── sampler selection from config ───────────────────────────────────────────


def test_windowed_staleness_override_takes_effect() -> None:
    cfg = TypeAdapter(SamplerConfig).validate_python(
        {"name": "windowed", "max_staleness_versions": 4}
    )

    assert isinstance(cfg, WindowedSamplerConfig)
    assert cfg.max_staleness_versions == 4


def test_windowed_silently_ignores_the_lookahead_key() -> None:
    """Guards a sweep-invalidating footgun.

    Both sampler configs are ``extra="allow"``, and windowed spells its slack
    ``max_staleness_versions``. Handing it the gated samplers'
    ``max_lookahead_versions`` is accepted without complaint and leaves the
    staleness at its default, so every arm of a sweep would run identically.
    The launcher emits the key matching the chosen sampler for this reason.
    """
    cfg = TypeAdapter(SamplerConfig).validate_python(
        {"name": "windowed", "max_lookahead_versions": 6}
    )

    assert isinstance(cfg, WindowedSamplerConfig)
    assert cfg.max_staleness_versions == 1
