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

"""Contract tests for the split PromptGroupSampler policies + factory.

CPU-only; mirrors the FakeBuffer surface used by test_staleness_sampler.py.
Covers what the split buys over the monolithic sampler: per-policy admission,
InOrderSampler's target_step-keyed evict (evict/select agree by construction),
and FQN loading of an out-of-repo sampler.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSampler,
    InOrderSamplerConfig,
    PromptGroupSampler,
    ReadyFirstSampler,
    ReadyFirstSamplerConfig,
    SamplerConfig,
    WeightFifoSampler,
    WeightFifoSamplerConfig,
    WindowedSampler,
    WindowedSamplerConfig,
    create_sampler,
    required_buffer_capacity_for_config,
)
from nemo_rl.data_plane import KVBatchMeta


class FakeBuffer:
    """Minimal TQReplayBuffer surface the samplers read/mutate."""

    def __init__(self, partition_id: str = "rollout_data") -> None:
        self._partition_id = partition_id
        self.meta_list: list[KVBatchMeta | None] = []
        self.start_weight_list: list[int] = []
        self.end_weight_list: list[int] = []
        self.target_step_list: list[int | None] = []
        self.ready_list: list[bool] = []
        self.remove_calls: list[tuple[list[int], bool]] = []

    def add(
        self,
        group_id: str,
        weight: int,
        *,
        ready: bool = True,
        target_step: int | None = None,
    ) -> None:
        meta = KVBatchMeta(
            partition_id=self._partition_id,
            task_name=None,
            sample_ids=[f"{group_id}_g0"],
            tags=[{"weight_version": weight, "group_id": group_id}],
        )
        self.meta_list.append(meta if ready else None)
        self.start_weight_list.append(weight)
        self.end_weight_list.append(weight)
        self.target_step_list.append(target_step)
        self.ready_list.append(ready)

    async def remove(self, idxs: list[int], remove_in_dp: bool) -> int:
        self.remove_calls.append((list(idxs), remove_in_dp))
        for i in sorted(idxs, reverse=True):
            del self.meta_list[i]
            del self.start_weight_list[i]
            del self.end_weight_list[i]
            del self.target_step_list[i]
            del self.ready_list[i]
        return len(idxs)


def _run(coro):
    return asyncio.run(coro)


class TestBuiltinsImplementInterface:
    @pytest.mark.parametrize(
        "sampler",
        [
            WindowedSampler(FakeBuffer(), max_staleness_versions=1),
            ReadyFirstSampler(FakeBuffer(), max_staleness_versions=1),
            WeightFifoSampler(FakeBuffer(), max_staleness_versions=1),
            InOrderSampler(FakeBuffer(), max_lookahead_versions=1),
        ],
    )
    def test_isinstance_protocol(self, sampler):
        assert isinstance(sampler, PromptGroupSampler)


class TestAdmission:
    def test_windowed_never_gates_and_never_stamps(self):
        s = WindowedSampler(FakeBuffer(), max_staleness_versions=2)
        # trainer stuck at 0, but over-sampled admission returns immediately.
        assert _run(s.admit(trainer_version_fn=lambda: 0)) is None
        assert _run(s.admit(trainer_version_fn=lambda: 0)) is None

    def test_in_order_stamps_monotonic_dispatch_index(self):
        s = InOrderSampler(FakeBuffer(), max_lookahead_versions=5)
        assert _run(s.admit(trainer_version_fn=lambda: 10)) == 0
        assert _run(s.admit(trainer_version_fn=lambda: 10)) == 1

    def test_weight_fifo_gates_on_lookahead_and_does_not_stamp(self):
        # dispatch_index starts at -1; window 0 => admits exactly one batch
        # ahead of the trainer, then blocks. Assert the second admit would block
        # by giving it a trainer_version that keeps the gate closed.
        s = WeightFifoSampler(FakeBuffer(), max_staleness_versions=0)
        assert _run(s.admit(trainer_version_fn=lambda: 0)) is None  # -1 -> 0
        # Now dispatch_index=0, trainer=0, window=0 -> 0 >= 0 blocks forever.
        with pytest.raises(asyncio.TimeoutError):
            _run(asyncio.wait_for(s.admit(trainer_version_fn=lambda: 0), timeout=0.05))

    def test_ready_first_opens_one_more_batch_after_trainer_advances(self):
        trainer_version = 0
        s = ReadyFirstSampler(FakeBuffer(), max_staleness_versions=1)

        # eta=1 admits the live batch and one lookahead batch without stamping.
        assert _run(s.admit(trainer_version_fn=lambda: trainer_version)) is None
        assert _run(s.admit(trainer_version_fn=lambda: trainer_version)) is None
        with pytest.raises(asyncio.TimeoutError):
            _run(
                asyncio.wait_for(
                    s.admit(trainer_version_fn=lambda: trainer_version), timeout=0.05
                )
            )

        trainer_version = 1
        assert _run(s.admit(trainer_version_fn=lambda: trainer_version)) is None
        with pytest.raises(asyncio.TimeoutError):
            _run(
                asyncio.wait_for(
                    s.admit(trainer_version_fn=lambda: trainer_version), timeout=0.05
                )
            )


class TestInOrderEvictMatchesSelect:
    """The bug the split fixes: monolithic evict keyed on weight could drop a
    slot whose target_step was still upcoming. InOrderSampler keys evict on
    target_step, so it never drops a slot select would later match."""

    def test_future_target_not_evicted_even_if_weight_out_of_window(self):
        buf = FakeBuffer()
        # weight far below the window, but target_step is still upcoming.
        buf.add("g", weight=0, ready=True, target_step=2)
        s = InOrderSampler(buf, max_lookahead_versions=1)
        removed = _run(s.evict(current_train_weight=2))
        assert removed == 0  # target_step 2 == current, not past -> kept
        assert len(buf.target_step_list) == 1

    def test_past_target_ready_slot_is_evicted(self):
        buf = FakeBuffer()
        buf.add("g", weight=0, ready=True, target_step=1)
        s = InOrderSampler(buf, max_lookahead_versions=1)
        removed = _run(s.evict(current_train_weight=3))  # target 1 < 3 -> stale
        assert removed == 1

    def test_unready_slot_is_never_evicted(self):
        buf = FakeBuffer()
        buf.add("g", weight=0, ready=False, target_step=1)
        s = InOrderSampler(buf, max_lookahead_versions=1)
        # past target, but unready -> skipped to avoid the commit race.
        assert _run(s.evict(current_train_weight=5)) == 0


class TestFactory:
    def test_windowed_config_builds_windowed(self):
        s = create_sampler(
            FakeBuffer(), WindowedSamplerConfig(max_staleness_versions=3)
        )
        assert isinstance(s, WindowedSampler)
        assert s.max_staleness_versions == 3

    def test_in_order_config_builds_in_order(self):
        s = create_sampler(FakeBuffer(), InOrderSamplerConfig(max_lookahead_versions=2))
        assert isinstance(s, InOrderSampler)
        assert s.max_lookahead_versions == 2

    def test_weight_fifo_config_builds_weight_fifo(self):
        s = create_sampler(
            FakeBuffer(), WeightFifoSamplerConfig(max_staleness_versions=4)
        )
        assert isinstance(s, WeightFifoSampler)
        assert s.max_staleness_versions == 4

    def test_ready_first_config_builds_ready_first_sampler(self):
        s = create_sampler(
            FakeBuffer(),
            ReadyFirstSamplerConfig(max_staleness_versions=3),
        )
        assert isinstance(s, ReadyFirstSampler)
        assert s.max_staleness_versions == 3


class TestReadyFirstConfig:
    def test_discriminated_union_parses_ready_first(self):
        cfg = TypeAdapter(SamplerConfig).validate_python(
            {
                "name": "ready_first",
                "max_staleness_versions": 2,
            }
        )

        assert isinstance(cfg, ReadyFirstSamplerConfig)
        assert cfg.max_staleness_versions == 2

    def test_negative_staleness_is_rejected(self):
        with pytest.raises(ValidationError):
            ReadyFirstSamplerConfig(max_staleness_versions=-1)

    def test_required_capacity_covers_live_and_lookahead_batches(self):
        cfg = ReadyFirstSamplerConfig(max_staleness_versions=2)
        assert required_buffer_capacity_for_config(cfg, groups_per_step=4) == 12
        sampler = create_sampler(FakeBuffer(), cfg)
        assert sampler.required_buffer_capacity(groups_per_step=4) == 12


class TestCustomFqnSampler:
    def test_custom_target_loads_out_of_repo_sampler(self):
        # A user sampler defined anywhere importable; here, this test module.
        from nemo_rl.algorithms.async_utils.staleness_sampler import (
            CustomSamplerConfig,
        )

        s = create_sampler(
            FakeBuffer(),
            CustomSamplerConfig(
                target=f"{__name__}:EchoSampler", max_lookahead_versions=1
            ),
        )
        assert isinstance(s, EchoSampler)
        assert isinstance(s, PromptGroupSampler)
        assert s.max_lookahead_versions == 1


class TestWindowedSelect:
    def test_selects_ready_groups_in_window(self):
        buf = FakeBuffer()
        buf.add("a", weight=3)
        buf.add("b", weight=5)  # current
        buf.add("c", weight=1)  # below window (5-2=3)
        s = WindowedSampler(buf, max_staleness_versions=2)
        meta, n = _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=8)
        )
        assert n == 2  # a(3) and b(5); c(1) excluded
        assert len(buf.start_weight_list) == 1  # only c remains

    def test_below_min_returns_none(self):
        buf = FakeBuffer()
        buf.add("a", weight=5)
        s = WindowedSampler(buf, max_staleness_versions=2)
        assert _run(
            s.select(current_train_weight=5, min_prompt_groups=2, max_prompt_groups=8)
        ) == (None, 0)

    def test_unready_excluded(self):
        buf = FakeBuffer()
        buf.add("a", weight=5, ready=False)
        s = WindowedSampler(buf, max_staleness_versions=2)
        assert _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=8)
        ) == (None, 0)

    def test_freshest_first_orders_by_lag(self):
        buf = FakeBuffer()
        buf.add("old", weight=1)
        buf.add("new", weight=5)
        s = WindowedSampler(buf, max_staleness_versions=10, sample_freshest_first=True)
        meta, n = _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1)
        )
        # freshest (weight 5) picked first -> "old" (weight 1) remains.
        assert n == 1
        assert buf.start_weight_list == [1]


class TestWeightFifoSelect:
    def test_drains_oldest_in_window_weight_first(self):
        buf = FakeBuffer()
        buf.add("old1", weight=3)
        buf.add("new", weight=5)
        buf.add("old2", weight=3)
        s = WeightFifoSampler(buf, max_staleness_versions=5)
        meta, n = _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=8)
        )
        assert n == 2  # both weight-3 groups; weight-5 waits its turn
        assert buf.start_weight_list == [5]

    def test_waits_for_partial_oldest_batch(self):
        buf = FakeBuffer()
        buf.add("old", weight=3)
        s = WeightFifoSampler(buf, max_staleness_versions=5)
        # oldest weight has only 1 group but min is 2 -> wait (None), don't skip
        # ahead to a newer weight.
        assert _run(
            s.select(current_train_weight=5, min_prompt_groups=2, max_prompt_groups=8)
        ) == (None, 0)

    def test_empty_window_returns_none(self):
        buf = FakeBuffer()
        buf.add("future", weight=9)
        s = WeightFifoSampler(buf, max_staleness_versions=2)
        assert _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=8)
        ) == (None, 0)


class TestReadyFirstSelect:
    def test_mixes_ready_weight_versions_in_buffer_order(self):
        buf = FakeBuffer()
        buf.add("old", weight=1)
        buf.add("current", weight=3)
        buf.add("middle", weight=2)
        buf.add("future", weight=4)
        s = ReadyFirstSampler(buf, max_staleness_versions=1)

        meta, n = _run(
            s.select(current_train_weight=3, min_prompt_groups=3, max_prompt_groups=3)
        )

        assert n == 3
        assert meta is not None
        assert meta.sample_ids == ["old_g0", "current_g0", "middle_g0"]
        assert buf.start_weight_list == [4]

    def test_no_eviction_keeps_late_straggler_selectable(self):
        buf = FakeBuffer()
        buf.add("late", weight=0)
        s = ReadyFirstSampler(buf, max_staleness_versions=1)

        assert _run(s.evict(current_train_weight=5)) == 0
        meta, n = _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1)
        )

        assert n == 1
        assert meta is not None
        assert meta.sample_ids == ["late_g0"]
        assert buf.remove_calls == [([0], False)]


class TestInOrderSelect:
    def test_matches_target_step_ignoring_weight_window(self):
        buf = FakeBuffer()
        # weight far outside any window, but target_step == trainer version.
        buf.add("g", weight=100, target_step=5)
        s = InOrderSampler(buf, max_lookahead_versions=1)
        meta, n = _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=8)
        )
        assert n == 1

    def test_non_matching_target_not_selected(self):
        buf = FakeBuffer()
        buf.add("g", weight=5, target_step=6)
        s = InOrderSampler(buf, max_lookahead_versions=1)
        assert _run(
            s.select(current_train_weight=5, min_prompt_groups=1, max_prompt_groups=8)
        ) == (None, 0)


class TestDefaultEvictSkipsUnready:
    def test_windowed_evict_drops_ready_below_window(self):
        buf = FakeBuffer()
        buf.add("stale", weight=0, ready=True)
        buf.add("fresh", weight=5, ready=True)
        s = WindowedSampler(buf, max_staleness_versions=1)
        removed = _run(s.evict(current_train_weight=5))  # min_valid = 4
        assert removed == 1
        assert buf.start_weight_list == [5]

    def test_windowed_evict_skips_unready_stale(self):
        buf = FakeBuffer()
        buf.add("stale_unready", weight=0, ready=False)
        s = WindowedSampler(buf, max_staleness_versions=1)
        assert _run(s.evict(current_train_weight=5)) == 0


class TestDispatchCursorRestore:
    """Checkpoint resume calls set_dispatch_index(current_step), restoring the
    fresh-start invariant _dispatch_index == trainer_version - 1. Without it,
    a restored InOrderSampler would stamp target_steps starting at 0 and every
    dispatched batch would be instantly evicted (target < trainer_version)."""

    def test_resumed_in_order_stamps_from_trainer_version(self):
        s = InOrderSampler(FakeBuffer(), max_lookahead_versions=1)
        s.set_dispatch_index(7)
        assert _run(s.admit(trainer_version_fn=lambda: 7)) == 7
        assert _run(s.admit(trainer_version_fn=lambda: 8)) == 8

    def test_resumed_gate_admits_window_then_blocks(self):
        s = WeightFifoSampler(FakeBuffer(), max_staleness_versions=0)
        s.set_dispatch_index(7)
        # Resumed at step 7, window 0: one batch admitted, then the gate
        # closes exactly as it would on a fresh run at step 0.
        assert _run(s.admit(trainer_version_fn=lambda: 7)) is None
        with pytest.raises(asyncio.TimeoutError):
            _run(asyncio.wait_for(s.admit(trainer_version_fn=lambda: 7), timeout=0.05))

    def test_fresh_start_is_a_noop_seed(self):
        s = InOrderSampler(FakeBuffer(), max_lookahead_versions=1)
        s.set_dispatch_index(0)
        assert _run(s.admit(trainer_version_fn=lambda: 0)) == 0

    def test_negative_resume_step_rejected(self):
        with pytest.raises(ValueError, match="resume_from_step"):
            WindowedSampler(FakeBuffer(), max_staleness_versions=1).set_dispatch_index(
                -1
            )

    def test_custom_fqn_sampler_supports_seeding(self):
        from nemo_rl.algorithms.async_utils.staleness_sampler import (
            CustomSamplerConfig,
        )

        s = create_sampler(
            FakeBuffer(),
            CustomSamplerConfig(
                target=f"{__name__}:EchoSampler", max_lookahead_versions=1
            ),
        )
        s.set_dispatch_index(6)
        assert _run(s.admit(trainer_version_fn=lambda: 6)) == 6


class TestInflightAbortPolicy:
    def test_windowed_aborts_only_below_weight_window(self):
        sampler = WindowedSampler(
            FakeBuffer(),
            max_staleness_versions=2,
        )

        assert sampler.should_abort_inflight(
            start_weight_version=2,
            current_train_weight=5,
        )
        assert not sampler.should_abort_inflight(
            start_weight_version=3,
            current_train_weight=5,
        )

    @pytest.mark.parametrize(
        "sampler",
        [
            WeightFifoSampler(FakeBuffer(), max_staleness_versions=1),
            InOrderSampler(FakeBuffer(), max_lookahead_versions=1),
        ],
    )
    def test_gated_samplers_never_abort_inflight(self, sampler):
        assert not sampler.should_abort_inflight(
            start_weight_version=0,
            current_train_weight=5,
        )


class EchoSampler(InOrderSampler):
    """Stand-in for a user-defined sampler loaded by FQN."""
