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

"""Unit tests for StalenessSampler (pure filter over TQReplayBuffer state)."""

from __future__ import annotations

import asyncio

import pytest

from nemo_rl.algorithms.async_utils.staleness_sampler import StalenessSampler
from nemo_rl.data_plane import KVBatchMeta


class FakeBuffer:
    """Minimal TQReplayBuffer surface used by StalenessSampler tests."""

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
        group_size: int = 1,
        ready: bool = True,
        end_weight: int | None = None,
        target_step: int | None = None,
    ) -> KVBatchMeta:
        sample_ids = [f"{group_id}_g{i}" for i in range(group_size)]
        meta = KVBatchMeta(
            partition_id=self._partition_id,
            task_name=None,
            sample_ids=sample_ids,
            tags=[{"weight_version": weight, "group_id": group_id}] * group_size,
        )
        self.meta_list.append(meta if ready else None)
        self.start_weight_list.append(weight)
        self.end_weight_list.append(weight if end_weight is None else end_weight)
        self.target_step_list.append(target_step)
        self.ready_list.append(ready)
        return meta

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


class TestStalenessSamplerSelect:
    def test_select_returns_none_when_insufficient(self):
        buf = FakeBuffer()
        buf.add("g0", weight=5)
        sampler = StalenessSampler(buf, max_staleness_versions=2)

        result = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )
        assert result == (None, 0)

    def test_select_filters_by_staleness_window(self):
        buf = FakeBuffer()
        # Weights 3, 4, 5, 2, 6 against trainer=5, max_staleness=2:
        # lags = 2, 1, 0, 3 (stale), -1 (future)
        for i, w in enumerate([3, 4, 5, 2, 6]):
            buf.add(f"g{i}", weight=w)
        sampler = StalenessSampler(
            buf, max_staleness_versions=2, sample_freshest_first=True
        )

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )

        assert selected is not None
        # Freshest first → g2 (lag 0), g1 (lag 1)
        assert selected.sample_ids == ["g2_g0", "g1_g0"]
        assert num_groups == 2

    def test_select_fifo_orders_by_insertion(self):
        buf = FakeBuffer()
        for w in [3, 4, 5]:
            buf.add(f"v{w}", weight=w)
        sampler = StalenessSampler(
            buf, max_staleness_versions=5, sample_freshest_first=False
        )

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=6, min_prompt_groups=2, max_prompt_groups=2
            )
        )
        assert selected is not None
        assert selected.sample_ids == ["v3_g0", "v4_g0"]
        assert num_groups == 2

    def test_select_skips_future_weight(self):
        buf = FakeBuffer()
        buf.add("now", weight=5)
        buf.add("future", weight=7)
        sampler = StalenessSampler(buf, max_staleness_versions=10)

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1
            )
        )

        assert selected is not None
        assert selected.sample_ids == ["now_g0"]
        assert num_groups == 1

    def test_select_concats_groups(self):
        buf = FakeBuffer()
        buf.add("g0", weight=5, group_size=2)
        buf.add("g1", weight=5, group_size=2)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )

        assert selected is not None
        assert selected.sample_ids == [
            "g0_g0",
            "g0_g1",
            "g1_g0",
            "g1_g1",
        ]
        # Two groups concatenated, each of size 2 → 4 sample_ids total.
        assert num_groups == 2

    def test_select_strict_on_policy_requires_exact_version(self):
        buf = FakeBuffer()
        for i, w in enumerate([4, 5, 5, 6]):
            buf.add(f"g{i}", weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        # 3 eligible (need weight=5), only have 2
        result = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=3, max_prompt_groups=3
            )
        )
        assert result == (None, 0)

        # Buffer still intact: select with min=3 returned None without dropping anything.
        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )
        assert selected is not None
        assert selected.sample_ids == ["g1_g0", "g2_g0"]
        assert num_groups == 2

    def test_select_drops_returned_entries_from_buffer(self):
        buf = FakeBuffer()
        for i, w in enumerate([5, 5, 5]):
            buf.add(f"g{i}", weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        first_meta, first_num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1
            )
        )
        assert first_meta is not None
        assert first_meta.sample_ids == ["g0_g0"]
        assert first_num_groups == 1
        assert buf.start_weight_list == [5, 5]
        # remove_in_dp=False; DP rows kept for trainer.
        assert buf.remove_calls[-1][1] is False

        second_meta, second_num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1
            )
        )
        assert second_meta is not None
        assert second_meta.sample_ids == ["g1_g0"]
        assert second_num_groups == 1

    def test_select_rejects_zero_min_prompt_groups(self):
        buf = FakeBuffer()
        sampler = StalenessSampler(buf, max_staleness_versions=0)
        with pytest.raises(ValueError):
            _run(
                sampler.select(
                    current_train_weight=0, min_prompt_groups=0, max_prompt_groups=0
                )
            )

    def test_select_rejects_max_less_than_min(self):
        buf = FakeBuffer()
        for i in range(3):
            buf.add(f"g{i}", weight=5)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        with pytest.raises(ValueError):
            _run(
                sampler.select(
                    current_train_weight=5, min_prompt_groups=2, max_prompt_groups=1
                )
            )

    def test_select_caps_at_max_prompt_groups(self):
        buf = FakeBuffer()
        for i in range(5):
            buf.add(f"g{i}", weight=5)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=3
            )
        )

        assert selected is not None
        # FIFO order; capped at max=3 even though 5 are eligible.
        assert selected.sample_ids == ["g0_g0", "g1_g0", "g2_g0"]
        assert num_groups == 3
        # The remaining two stay in the buffer.
        assert buf.start_weight_list == [5, 5]

    def test_select_takes_all_available_when_between_min_and_max(self):
        buf = FakeBuffer()
        for i in range(3):
            buf.add(f"g{i}", weight=5)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=8
            )
        )

        assert selected is not None
        assert selected.sample_ids == ["g0_g0", "g1_g0", "g2_g0"]
        assert num_groups == 3
        assert buf.start_weight_list == []


class TestStalenessSamplerEvict:
    def test_evict_removes_stale_groups(self):
        buf = FakeBuffer()
        # trainer=5, max_staleness=1 → lag >1 means stale (weights 0, 1, 2 stale; 4, 5 fresh)
        for i, w in enumerate([0, 1, 4, 5, 2]):
            buf.add(f"g{i}", weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        dropped = _run(sampler.evict(current_train_weight=5))

        assert dropped == 3
        assert buf.start_weight_list == [4, 5]
        # Survivors' sample_ids
        assert [m.sample_ids[0] for m in buf.meta_list] == ["g2_g0", "g3_g0"]

    def test_evict_returns_zero_when_nothing_stale(self):
        buf = FakeBuffer()
        for w in [4, 5]:
            buf.add(f"v{w}", weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        assert _run(sampler.evict(current_train_weight=5)) == 0
        assert buf.remove_calls == []

    def test_evict_keeps_future_groups(self):
        buf = FakeBuffer()
        buf.add("future", weight=7)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        assert _run(sampler.evict(current_train_weight=5)) == 0
        assert buf.start_weight_list == [7]

    def test_evict_drops_whole_group(self):
        buf = FakeBuffer()
        buf.add("stale", weight=1, group_size=4)
        buf.add("fresh", weight=5, group_size=4)
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        dropped = _run(sampler.evict(current_train_weight=5))

        assert dropped == 1
        assert buf.remove_calls == [([0], True)]
        assert buf.start_weight_list == [5]
        assert [m.sample_ids[0] for m in buf.meta_list] == ["fresh_g0"]


class TestStalenessSamplerInit:
    def test_rejects_negative_max_staleness(self):
        buf = FakeBuffer()
        with pytest.raises(ValueError):
            StalenessSampler(buf, max_staleness_versions=-1)

    def test_rejects_strict_weight_fifo_with_freshest_first(self):
        buf = FakeBuffer()
        with pytest.raises(ValueError):
            StalenessSampler(
                buf,
                max_staleness_versions=0,
                sample_freshest_first=True,
                strict_weight_fifo=True,
            )


class TestStalenessSamplerReady:
    def test_default_mode_skips_unready_slots(self):
        buf = FakeBuffer()
        buf.add("g0", weight=5, ready=False)
        buf.add("g1", weight=5, ready=True)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=1
            )
        )

        assert selected is not None
        assert selected.sample_ids == ["g1_g0"]
        assert num_groups == 1

    def test_default_mode_waits_when_too_few_ready(self):
        buf = FakeBuffer()
        buf.add("g0", weight=5, ready=False)
        buf.add("g1", weight=5, ready=True)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        result = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )
        assert result == (None, 0)


class TestStalenessSamplerStrictWeightFifo:
    def test_consumes_oldest_batch_first(self):
        buf = FakeBuffer()
        # Two complete batches: v=4 then v=5; strict_weight_fifo must take v=4 first.
        for i, w in enumerate((4, 4, 5, 5)):
            buf.add(f"v{w}_{i}", weight=w)
        sampler = StalenessSampler(
            buf, max_staleness_versions=1, strict_weight_fifo=True
        )

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )

        assert selected is not None
        # Insertion-order FIFO inside the oldest batch.
        assert selected.sample_ids == ["v4_0_g0", "v4_1_g0"]
        assert num_groups == 2
        assert buf.start_weight_list == [5, 5]

    def test_waits_when_oldest_batch_partially_ready(self):
        buf = FakeBuffer()
        # Oldest batch v=4 has 1 ready + 1 unready; v=5 batch is fully ready.
        # strict_weight_fifo must NOT skip ahead to v=5.
        buf.add("v4_a", weight=4, ready=True)
        buf.add("v4_b", weight=4, ready=False)
        buf.add("v5_a", weight=5, ready=True)
        buf.add("v5_b", weight=5, ready=True)
        sampler = StalenessSampler(
            buf, max_staleness_versions=1, strict_weight_fifo=True
        )

        result = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )
        assert result == (None, 0)
        # Buffer untouched: nothing removed.
        assert buf.start_weight_list == [4, 4, 5, 5]
        assert buf.ready_list == [True, False, True, True]

    def test_returns_none_when_oldest_batch_not_filled(self):
        buf = FakeBuffer()
        buf.add("v4_a", weight=4, ready=True)
        # Only 1 ready in oldest batch; need 2.
        sampler = StalenessSampler(
            buf, max_staleness_versions=1, strict_weight_fifo=True
        )

        result = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )
        assert result == (None, 0)

    def test_ignores_future_versions_when_picking_target(self):
        buf = FakeBuffer()
        # Trainer at 5, staleness 1: window is [4, 5]; v=7 (future) must not
        # become the oldest target.
        buf.add("v7", weight=7, ready=True)
        buf.add("v5_a", weight=5, ready=True)
        buf.add("v5_b", weight=5, ready=True)
        sampler = StalenessSampler(
            buf, max_staleness_versions=1, strict_weight_fifo=True
        )

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=2, max_prompt_groups=2
            )
        )

        assert selected is not None
        assert selected.sample_ids == ["v5_a_g0", "v5_b_g0"]
        assert num_groups == 2
        assert buf.start_weight_list == [7]


class TestStalenessSamplerForceInOrder:
    def test_selects_only_groups_whose_target_step_matches_current(self):
        buf = FakeBuffer()
        # Two matching-target groups and one future-target group.
        buf.add("g_match_a", weight=5, target_step=5)
        buf.add("g_match_b", weight=5, target_step=5)
        buf.add("g_future", weight=5, target_step=7)
        sampler = StalenessSampler(buf, max_staleness_versions=0, force_in_order=True)

        selected, num_groups = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=4
            )
        )

        assert selected is not None
        assert selected.sample_ids == ["g_match_a_g0", "g_match_b_g0"]
        assert num_groups == 2
        # The future-target group survives; only the two matches were removed.
        assert buf.target_step_list == [7]
        assert buf.start_weight_list == [5]

    def test_ignores_stale_target_step(self):
        buf = FakeBuffer()
        # Trainer at 5; a slot with target_step=4 must NOT satisfy force_in_order.
        buf.add("g_stale", weight=4, target_step=4)
        sampler = StalenessSampler(
            buf,
            # Wide staleness window: force_in_order must ignore it entirely.
            max_staleness_versions=10,
            force_in_order=True,
        )

        result = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=4
            )
        )

        assert result == (None, 0)
        # Buffer untouched.
        assert buf.target_step_list == [4]

    def test_ignores_unready_group_with_matching_target(self):
        buf = FakeBuffer()
        buf.add("g_unready", weight=5, target_step=5, ready=False)
        sampler = StalenessSampler(buf, max_staleness_versions=0, force_in_order=True)

        result = _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=4
            )
        )

        assert result == (None, 0)
        # Buffer untouched.
        assert buf.ready_list == [False]
