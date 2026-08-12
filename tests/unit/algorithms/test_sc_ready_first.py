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
"""``ready_first`` must stream selection and discard nothing.

Companion to test_sc_short_batch.py, which pins ``in_order``'s wedge and the
``_batch_shortfall`` release that covers it. The staleness sweep runs
``ready_first`` with ``evict_stale_samples=false`` instead, on two claims:

  - **streaming selection** — a step closes on the first ``num_prompts_per_step``
    groups that are ready, never on a specific cohort, so a lost prompt is
    backfilled rather than waited for;
  - **zero eviction** — nothing that finished generating is dropped, so a late
    straggler stays selectable and is preferred.

Every test here contrasts ``ready_first`` against ``in_order`` or ``windowed``
on the same scenario, or against its own opposite setting. That is deliberate.
An assertion that only said "ready_first closes the step" would still pass if
the sampler quietly stamped cohorts or inherited the weight-window ``evict``,
and those are precisely the two ways this class is easy to get wrong: it must
override ``evict`` (``BaseSampler``'s default drops exactly the stragglers the
policy exists to keep) and must *not* override ``_stamp`` (a stamp reintroduces
cohort gating). Each test below is checked against a deliberate mutation of one
of those, so a sampler that regressed either way fails a named test rather than
passing quietly.
"""

import asyncio
from collections import Counter
from typing import Optional

from pydantic import TypeAdapter

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSampler,
    ReadyFirstSampler,
    ReadyFirstSamplerConfig,
    SamplerConfig,
    WindowedSampler,
    required_buffer_capacity_for_config,
)

NUM_PROMPTS_PER_STEP = 128
MIN_GROUPS_FOR_STREAMING_TRAIN = 32


class _FakeMeta:
    """Stands in for KVBatchMeta: concat, plus the lengths selection totals."""

    def __init__(self, sequence_lengths: Optional[list[int]] = None) -> None:
        self.sequence_lengths = (
            [1] if sequence_lengths is None else list(sequence_lengths)
        )

    def concat(self, *others: "_FakeMeta") -> "_FakeMeta":
        return self


class _FakeBuffer:
    """Enough of TQReplayBuffer for the sampler's index bookkeeping."""

    def __init__(self) -> None:
        self.meta_list: list[_FakeMeta] = []
        self.start_weight_list: list[int] = []
        self.target_step_list: list[Optional[int]] = []
        self.ready_list: list[bool] = []
        self.remove_calls: list[tuple[list[int], bool]] = []

    def commit(self, *, weight: int, target_step: Optional[int], count: int) -> None:
        for _ in range(count):
            self.meta_list.append(_FakeMeta())
            self.start_weight_list.append(weight)
            self.target_step_list.append(target_step)
            self.ready_list.append(True)

    async def remove(self, idxs: list[int], remove_in_dp: bool) -> int:
        self.remove_calls.append((sorted(idxs), remove_in_dp))
        for i in sorted(idxs, reverse=True):
            del self.meta_list[i]
            del self.start_weight_list[i]
            del self.target_step_list[i]
            del self.ready_list[i]
        return len(idxs)


def _ready_first(staleness: int, *, evict: bool = False) -> ReadyFirstSampler:
    return ReadyFirstSampler(
        _FakeBuffer(),  # type: ignore[arg-type]
        max_staleness_versions=staleness,
        evict_stale_samples=evict,
    )


def _buffer_of(sampler: object) -> _FakeBuffer:
    return sampler._buffer  # type: ignore[attr-defined]


async def _dispatch_cohort(
    sampler: object,
    *,
    trainer_version: int,
    committed: int,
) -> Optional[int]:
    """Admit one cohort and commit the prompts that survived their rollouts.

    Goes through the real ``admit`` rather than hand-stamping, so each sampler
    labels the cohort the way it actually would: ``in_order`` with its dispatch
    index, ``ready_first`` with nothing. ``committed`` below
    ``NUM_PROMPTS_PER_STEP`` is how a dropped prompt shows up.
    """
    target_step = await sampler.admit(  # type: ignore[attr-defined]
        trainer_version_fn=lambda: trainer_version
    )
    _buffer_of(sampler).commit(
        weight=trainer_version, target_step=target_step, count=committed
    )
    return target_step


async def _drain_step(
    sampler: object,
    *,
    trainer_version: int,
    shortfall: Counter[int],
    max_iterations: int = 64,
) -> tuple[int, bool]:
    """Run the train pump's selection loop for one step.

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
        train_meta, num_groups = await sampler.select(  # type: ignore[attr-defined]
            current_train_weight=trainer_version,
            min_prompt_groups=min_prompt_groups,
            max_prompt_groups=max_prompt_groups,
        )
        if train_meta is None:
            continue
        groups_dispatched += num_groups
    return groups_dispatched, False


# ── streaming selection ─────────────────────────────────────────────────────


def test_streaming_backfills_a_short_cohort_where_in_order_wedges() -> None:
    """The 5982927 wedge, put to both samplers with no shortfall recorded.

    Two prompts are lost from the first cohort and the lookahead cohort has
    already committed. ``in_order`` can only offer groups stamped for step 0, so
    it stalls at 126; ``ready_first`` has no stamp to match on and closes.
    """

    async def run(sampler: object) -> tuple[int, bool]:
        await _dispatch_cohort(
            sampler, trainer_version=0, committed=NUM_PROMPTS_PER_STEP - 2
        )
        await _dispatch_cohort(sampler, trainer_version=0, committed=2)
        return await _drain_step(sampler, trainer_version=0, shortfall=Counter())

    rf_collected, rf_closed = asyncio.run(run(_ready_first(staleness=1)))
    io_collected, io_closed = asyncio.run(
        run(InOrderSampler(_FakeBuffer(), max_lookahead_versions=1))  # type: ignore[arg-type]
    )

    assert (rf_collected, rf_closed) == (NUM_PROMPTS_PER_STEP, True)
    assert (io_collected, io_closed) == (NUM_PROMPTS_PER_STEP - 2, False)


def test_admit_stamps_nothing_so_no_shortfall_is_ever_recorded() -> None:
    """Why the pump's shortfall accounting is inert here, and may stay so.

    ``_rollout_with_retries`` credits ``_batch_shortfall[target_step]`` only
    when the sampler stamped a target step, and the train pump subtracts that
    credit from the groups it waits for. ``ready_first`` never stamps, so the
    credit is never taken — which is safe only because the step does not need
    it: any ready group substitutes.
    """
    sampler = _ready_first(staleness=4)

    target_step = asyncio.run(_dispatch_cohort(sampler, trainer_version=0, committed=0))
    assert target_step is None

    stamped = InOrderSampler(_FakeBuffer(), max_lookahead_versions=4)  # type: ignore[arg-type]
    assert asyncio.run(_dispatch_cohort(stamped, trainer_version=0, committed=0)) == 0


def test_selection_is_oldest_first_across_mixed_weight_versions() -> None:
    """Oldest-first is the starvation guarantee.

    A straggler cannot be passed over by fresher work. Freshest-first ordering
    under zero eviction would strand it forever, so the ordering is load-bearing
    rather than incidental.
    """
    sampler = _ready_first(staleness=6)
    buffer = _buffer_of(sampler)
    buffer.commit(weight=0, target_step=None, count=1)
    buffer.commit(weight=6, target_step=None, count=NUM_PROMPTS_PER_STEP)

    asyncio.run(
        sampler.select(current_train_weight=6, min_prompt_groups=1, max_prompt_groups=1)
    )

    # The single weight-0 group was index 0 and is the one that got taken.
    assert buffer.remove_calls == [([0], False)]
    assert buffer.start_weight_list == [6] * NUM_PROMPTS_PER_STEP


# ── zero eviction ───────────────────────────────────────────────────────────


def test_zero_eviction_keeps_a_straggler_that_windowed_discards() -> None:
    """Same buffer, same window, opposite disposition.

    The groups ``ready_first`` trains on late are exactly the groups
    ``windowed`` throws away — 21.4 / 6.6 / 3.0% of them at staleness 2 / 4 / 6
    in the previous sweep.
    """
    rf = _ready_first(staleness=1)
    _buffer_of(rf).commit(weight=0, target_step=None, count=1)

    wd = WindowedSampler(_FakeBuffer(), max_staleness_versions=1)  # type: ignore[arg-type]
    _buffer_of(wd).commit(weight=0, target_step=None, count=1)

    assert asyncio.run(rf.evict(current_train_weight=5)) == 0
    assert asyncio.run(wd.evict(current_train_weight=5)) == 1

    _, rf_selected = asyncio.run(
        rf.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1)
    )
    _, wd_selected = asyncio.run(
        wd.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1)
    )

    assert rf_selected == 1
    assert wd_selected == 0


def test_evict_never_touches_the_buffer_at_any_staleness() -> None:
    """The wedge in review comment 2 needs an eviction to exist at all.

    Uncredited evictions starve the admission gate, so the run hangs rather
    than errors. Under the shipped default there is no eviction to be
    uncredited: assert the buffer is untouched, not merely that the count is
    zero, so a policy that removed rows and under-reported would still fail.
    """
    sampler = _ready_first(staleness=1)
    buffer = _buffer_of(sampler)
    buffer.commit(weight=0, target_step=None, count=NUM_PROMPTS_PER_STEP)

    for trainer_version in (0, 1, 5, 50, 5000):
        assert asyncio.run(sampler.evict(current_train_weight=trainer_version)) == 0

    assert buffer.remove_calls == []
    assert len(buffer.start_weight_list) == NUM_PROMPTS_PER_STEP


def test_eviction_would_strand_a_step_that_zero_eviction_closes() -> None:
    """Shows the previous test is not vacuous, and prices the flag.

    Identical buffer and trainer version; only ``evict_stale_samples`` differs.
    With eviction on, the pump's evict pass destroys the very groups the step
    needed and it can never close — which is why the flag stays off and the
    recipe says so explicitly rather than inheriting the default.
    """

    async def run(evict: bool) -> tuple[int, bool]:
        sampler = _ready_first(staleness=1, evict=evict)
        _buffer_of(sampler).commit(
            weight=0, target_step=None, count=NUM_PROMPTS_PER_STEP
        )
        # The pump evicts before it selects, on every pass.
        await sampler.evict(current_train_weight=5)
        return await _drain_step(sampler, trainer_version=5, shortfall=Counter())

    assert asyncio.run(run(evict=False)) == (NUM_PROMPTS_PER_STEP, True)
    assert asyncio.run(run(evict=True)) == (0, False)


# ── config: the knobs the sweep sets ────────────────────────────────────────


def test_staleness_override_takes_effect_and_the_wrong_key_does_not() -> None:
    """Guards a sweep-invalidating footgun.

    ``ready_first`` spells its slack ``max_staleness_versions``, and the config
    is ``extra="allow"``. Handing it the gated samplers' ``max_lookahead_versions``
    is accepted without complaint and leaves the staleness at 1, so every arm of
    a four-lag sweep would run identically. The launcher emits the key matching
    the chosen sampler for this reason.
    """
    right = TypeAdapter(SamplerConfig).validate_python(
        {"name": "ready_first", "max_staleness_versions": 6}
    )
    wrong = TypeAdapter(SamplerConfig).validate_python(
        {"name": "ready_first", "max_lookahead_versions": 6}
    )

    assert isinstance(right, ReadyFirstSamplerConfig)
    assert right.max_staleness_versions == 6
    assert isinstance(wrong, ReadyFirstSamplerConfig)
    assert wrong.max_staleness_versions == 1


def test_eviction_defaults_off_and_survives_being_set_explicitly() -> None:
    implied = TypeAdapter(SamplerConfig).validate_python({"name": "ready_first"})
    explicit = TypeAdapter(SamplerConfig).validate_python(
        {"name": "ready_first", "evict_stale_samples": False}
    )

    assert isinstance(implied, ReadyFirstSamplerConfig)
    assert isinstance(explicit, ReadyFirstSamplerConfig)
    assert implied.evict_stale_samples is False
    assert explicit.evict_stale_samples is False


def test_capacity_floor_matches_the_launcher_buffer_arithmetic() -> None:
    """The launcher sizes both the buffer and the in-flight cap as
    ``num_prompts_per_step * (staleness + 1)``. If the sampler's floor ever
    exceeded that, every arm would be rejected at startup by
    ``validate_sampler_buffer_capacity``."""
    floors = {
        staleness: required_buffer_capacity_for_config(
            ReadyFirstSamplerConfig(max_staleness_versions=staleness),
            NUM_PROMPTS_PER_STEP,
        )
        for staleness in (1, 2, 4, 6)
    }

    assert floors == {1: 256, 2: 384, 4: 640, 6: 896}
