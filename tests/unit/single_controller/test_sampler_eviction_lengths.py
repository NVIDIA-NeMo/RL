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

"""Sequence-length instrumentation on the sampler evict/select paths.

CPU-only; mirrors the FakeBuffer surface used by test_sampler_interface.py.
Covers the aggregation itself (including the absent/malformed metadata paths
that must not raise inside the train pump) and the evicted-vs-selected
comparison the eviction line reports.
"""

from __future__ import annotations

import asyncio

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSampler,
    WindowedSampler,
    summarize_group_lengths,
)
from nemo_rl.data_plane import KVBatchMeta

_PARTITION = "rollout_data"


def _meta(group_id: str, sequence_lengths: list[int] | None) -> KVBatchMeta:
    n = len(sequence_lengths) if sequence_lengths is not None else 1
    return KVBatchMeta(
        partition_id=_PARTITION,
        task_name="train",
        sample_ids=[f"{group_id}_g{i}" for i in range(n)],
        fields=["input_ids"],
        sequence_lengths=sequence_lengths,
    )


class FakeBuffer:
    """Minimal TQReplayBuffer surface the samplers read/mutate."""

    def __init__(self) -> None:
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
        sequence_lengths: list[int] | None = None,
        ready: bool = True,
        target_step: int | None = None,
    ) -> None:
        # An unready slot holds None until commit fills it, matching the buffer.
        self.meta_list.append(_meta(group_id, sequence_lengths) if ready else None)
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


class TestSummarizeGroupLengths:
    def test_totals_over_populated_groups(self):
        stats = summarize_group_lengths(
            [_meta("a", [10, 20]), _meta("b", [30, 40, 50])]
        )
        assert stats.groups == 2
        assert stats.measured_groups == 2
        assert stats.samples == 5
        assert stats.tokens == 150
        assert stats.mean_tokens_per_group == 75.0

    def test_empty_input_is_all_zeros(self):
        stats = summarize_group_lengths([])
        assert (stats.groups, stats.measured_groups, stats.tokens) == (0, 0, 0)
        assert stats.mean_tokens_per_group == 0.0

    def test_missing_metadata_counted_but_not_measured(self):
        # None slot (reserved, uncommitted), explicit None lengths, and an empty
        # list are all "no usable metadata" and must not raise or skew the mean.
        stats = summarize_group_lengths(
            [None, _meta("a", None), _meta("b", []), _meta("c", [100])]
        )
        assert stats.groups == 4
        assert stats.measured_groups == 1
        assert stats.samples == 1
        assert stats.tokens == 100
        assert stats.mean_tokens_per_group == 100.0

    def test_malformed_lengths_are_skipped_without_raising(self):
        bad = _meta("bad", [1])
        bad.sequence_lengths = ["not-a-length"]  # type: ignore[list-item]
        stats = summarize_group_lengths([bad, _meta("ok", [40, 60])])
        assert stats.groups == 2
        assert stats.measured_groups == 1
        assert stats.tokens == 100

    def test_mean_ignores_unmeasured_groups(self):
        stats = summarize_group_lengths([None, None, _meta("a", [10, 10])])
        assert stats.mean_tokens_per_group == 20.0


class TestSelectionTotals:
    def test_select_accumulates_measured_groups_only(self):
        buf = FakeBuffer()
        buf.add("a", weight=1, sequence_lengths=[100, 100])
        buf.add("b", weight=1, sequence_lengths=None)
        sampler = WindowedSampler(buf, max_staleness_versions=1)
        meta, num_groups = _run(
            sampler.select(
                current_train_weight=1, min_prompt_groups=1, max_prompt_groups=8
            )
        )
        assert meta is not None
        assert num_groups == 2
        # Only the group carrying lengths contributes to the mean.
        assert sampler._selected_groups == 1
        assert sampler.selected_mean_tokens_per_group == 200.0

    def test_mean_is_zero_before_any_selection(self):
        sampler = WindowedSampler(FakeBuffer(), max_staleness_versions=1)
        assert sampler.selected_mean_tokens_per_group == 0.0


class TestEvictionReporting:
    def test_eviction_line_reports_lengths_and_ratio(self, capsys):
        buf = FakeBuffer()
        # Short groups get selected at weight 5 ...
        buf.add("short_a", weight=5, sequence_lengths=[100, 100])
        buf.add("short_b", weight=5, sequence_lengths=[100, 100])
        sampler = WindowedSampler(buf, max_staleness_versions=0)
        _run(
            sampler.select(
                current_train_weight=5, min_prompt_groups=1, max_prompt_groups=8
            )
        )
        # ... while a slow, long group committed against an older weight ages
        # out of the window and is evicted instead.
        buf.add("long", weight=4, sequence_lengths=[1000, 1000])
        capsys.readouterr()

        assert _run(sampler.evict(current_train_weight=5)) == 1

        line = capsys.readouterr().out
        assert "eviction lengths (train_weight=5)" in line
        assert "evicted_groups=1" in line
        assert "measured_groups=1" in line
        assert "evicted_tokens=2000" in line
        assert "evicted_mean_tokens_per_group=2000.0" in line
        assert "selected_mean_tokens_per_group=200.0" in line
        assert "evicted_over_selected=10.00x" in line

    def test_ratio_is_na_before_any_selection(self, capsys):
        buf = FakeBuffer()
        buf.add("long", weight=0, sequence_lengths=[500])
        sampler = WindowedSampler(buf, max_staleness_versions=0)
        capsys.readouterr()
        assert _run(sampler.evict(current_train_weight=3)) == 1
        assert "evicted_over_selected=n/a" in capsys.readouterr().out

    def test_evict_without_metadata_does_not_raise(self, capsys):
        # A ready slot whose meta never carried lengths still evicts cleanly;
        # an exception here would take down the train pump.
        buf = FakeBuffer()
        buf.add("no_lengths", weight=0, sequence_lengths=None)
        sampler = WindowedSampler(buf, max_staleness_versions=0)
        capsys.readouterr()
        assert _run(sampler.evict(current_train_weight=3)) == 1
        line = capsys.readouterr().out
        assert "evicted_groups=1" in line
        assert "measured_groups=0" in line
        assert "evicted_tokens=0" in line
        assert "evicted_over_selected=n/a" in line

    def test_no_eviction_prints_nothing(self, capsys):
        buf = FakeBuffer()
        buf.add("fresh", weight=5, sequence_lengths=[100])
        sampler = WindowedSampler(buf, max_staleness_versions=1)
        capsys.readouterr()
        assert _run(sampler.evict(current_train_weight=5)) == 0
        assert capsys.readouterr().out == ""

    def test_in_order_evict_reports_through_the_same_path(self, capsys):
        buf = FakeBuffer()
        buf.add("past", weight=0, sequence_lengths=[700], target_step=1)
        sampler = InOrderSampler(buf, max_lookahead_versions=1)
        capsys.readouterr()
        assert _run(sampler.evict(current_train_weight=3)) == 1
        line = capsys.readouterr().out
        assert "evicted_groups=1" in line
        assert "evicted_tokens=700" in line

    def test_in_order_keeps_future_target_and_stays_silent(self, capsys):
        buf = FakeBuffer()
        buf.add("future", weight=0, sequence_lengths=[700], target_step=2)
        sampler = InOrderSampler(buf, max_lookahead_versions=1)
        capsys.readouterr()
        assert _run(sampler.evict(current_train_weight=2)) == 0
        assert capsys.readouterr().out == ""
