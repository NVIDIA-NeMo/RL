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
    """Minimal TQReplayBuffer surface used by StalenessSampler."""

    def __init__(self, partition_id: str = "rollout_data") -> None:
        self._partition_id = partition_id
        self.meta_list: list[KVBatchMeta] = []
        self.weight_list: list[int] = []
        self.remove_calls: list[list[str]] = []

    def add_group(
        self,
        group_id: str,
        group_size: int,
        weight: int,
    ) -> KVBatchMeta:
        sample_ids = [f"{group_id}_s{j}" for j in range(group_size)]
        meta = KVBatchMeta(
            partition_id=self._partition_id,
            task_name=None,
            sample_ids=sample_ids,
            tags=[{"weight_version": weight, "group_id": group_id}] * group_size,
        )
        self.meta_list.append(meta)
        self.weight_list.append(weight)
        return meta

    async def remove(self, sample_ids: list[str]) -> int:
        self.remove_calls.append(list(sample_ids))
        ids = set(sample_ids)
        keep_meta: list[KVBatchMeta] = []
        keep_w: list[int] = []
        n = 0
        for meta, w in zip(self.meta_list, self.weight_list):
            if set(meta.sample_ids) <= ids:
                n += 1
            else:
                keep_meta.append(meta)
                keep_w.append(w)
        self.meta_list = keep_meta
        self.weight_list = keep_w
        return n


def _run(coro):
    return asyncio.run(coro)


class TestStalenessSamplerSelect:
    def test_select_returns_none_when_insufficient(self):
        buf = FakeBuffer()
        buf.add_group("g0", group_size=1, weight=5)
        sampler = StalenessSampler(buf, max_staleness_versions=2)

        assert (
            sampler.select(current_train_weight=5, min_prompt_groups=2) is None
        )

    def test_select_returns_none_on_empty_buffer(self):
        buf = FakeBuffer()
        sampler = StalenessSampler(buf, max_staleness_versions=2)

        assert (
            sampler.select(current_train_weight=5, min_prompt_groups=1) is None
        )

    def test_select_filters_by_staleness_window(self):
        buf = FakeBuffer()
        # Weights 3, 4, 5, 2, 6 against trainer=5, max_staleness=2:
        # lags = 2, 1, 0, 3 (stale), -1 (future)
        for i, w in enumerate([3, 4, 5, 2, 6]):
            buf.add_group(f"g{i}", group_size=1, weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=2)

        selected = sampler.select(current_train_weight=5, min_prompt_groups=2)

        assert selected is not None
        # Freshest first → g2 (lag 0), g1 (lag 1)
        assert selected.sample_ids == ["g2_s0", "g1_s0"]

    def test_select_freshest_first_orders_by_lag(self):
        buf = FakeBuffer()
        for i, w in enumerate([3, 4, 5]):
            buf.add_group(f"v{w}", group_size=1, weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=5)

        selected = sampler.select(current_train_weight=6, min_prompt_groups=2)
        assert selected is not None
        assert selected.sample_ids == ["v5_s0", "v4_s0"]

    def test_select_fifo_orders_by_insertion(self):
        buf = FakeBuffer()
        for i, w in enumerate([3, 4, 5]):
            buf.add_group(f"v{w}", group_size=1, weight=w)
        sampler = StalenessSampler(
            buf, max_staleness_versions=5, sample_freshest_first=False
        )

        selected = sampler.select(current_train_weight=6, min_prompt_groups=2)
        assert selected is not None
        assert selected.sample_ids == ["v3_s0", "v4_s0"]

    def test_select_skips_future_weight(self):
        buf = FakeBuffer()
        buf.add_group("now", group_size=1, weight=5)
        buf.add_group("future", group_size=1, weight=7)
        sampler = StalenessSampler(buf, max_staleness_versions=10)

        selected = sampler.select(current_train_weight=5, min_prompt_groups=1)

        assert selected is not None
        assert selected.sample_ids == ["now_s0"]

    def test_select_concats_groups(self):
        buf = FakeBuffer()
        buf.add_group("g0", group_size=2, weight=5)
        buf.add_group("g1", group_size=2, weight=5)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        selected = sampler.select(current_train_weight=5, min_prompt_groups=2)

        assert selected is not None
        assert selected.sample_ids == [
            "g0_s0",
            "g0_s1",
            "g1_s0",
            "g1_s1",
        ]

    def test_select_strict_on_policy_requires_exact_version(self):
        buf = FakeBuffer()
        for i, w in enumerate([4, 5, 5, 6]):
            buf.add_group(f"g{i}", group_size=1, weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        # 3 eligible (need weight=5), only have 2
        assert (
            sampler.select(current_train_weight=5, min_prompt_groups=3) is None
        )

        selected = sampler.select(current_train_weight=5, min_prompt_groups=2)
        assert selected is not None
        assert selected.sample_ids == ["g1_s0", "g2_s0"]

    def test_select_rejects_zero_min_prompt_groups(self):
        buf = FakeBuffer()
        sampler = StalenessSampler(buf, max_staleness_versions=0)
        with pytest.raises(ValueError):
            sampler.select(current_train_weight=0, min_prompt_groups=0)


class TestStalenessSamplerEvict:
    def test_evict_removes_stale_groups(self):
        buf = FakeBuffer()
        # trainer=5, max_staleness=1 → lag >1 means stale (weights 0, 1, 2 stale; 4, 5 fresh)
        for i, w in enumerate([0, 1, 4, 5, 2]):
            buf.add_group(f"g{i}", group_size=1, weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        dropped = _run(sampler.evict(current_train_weight=5))

        assert dropped == 3
        assert buf.weight_list == [4, 5]
        # Survivors' sample_ids
        assert [m.sample_ids[0] for m in buf.meta_list] == ["g2_s0", "g3_s0"]

    def test_evict_returns_zero_when_nothing_stale(self):
        buf = FakeBuffer()
        for w in [4, 5]:
            buf.add_group(f"v{w}", group_size=1, weight=w)
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        assert _run(sampler.evict(current_train_weight=5)) == 0
        assert buf.remove_calls == []

    def test_evict_keeps_future_groups(self):
        buf = FakeBuffer()
        buf.add_group("future", group_size=1, weight=7)
        sampler = StalenessSampler(buf, max_staleness_versions=0)

        assert _run(sampler.evict(current_train_weight=5)) == 0
        assert buf.weight_list == [7]

    def test_evict_drops_whole_group(self):
        buf = FakeBuffer()
        buf.add_group("stale", group_size=4, weight=1)
        buf.add_group("fresh", group_size=4, weight=5)
        sampler = StalenessSampler(buf, max_staleness_versions=1)

        dropped = _run(sampler.evict(current_train_weight=5))

        assert dropped == 1
        assert len(buf.remove_calls) == 1
        assert set(buf.remove_calls[0]) == {f"stale_s{i}" for i in range(4)}
        assert buf.weight_list == [5]


class TestStalenessSamplerInit:
    def test_rejects_negative_max_staleness(self):
        buf = FakeBuffer()
        with pytest.raises(ValueError):
            StalenessSampler(buf, max_staleness_versions=-1)
