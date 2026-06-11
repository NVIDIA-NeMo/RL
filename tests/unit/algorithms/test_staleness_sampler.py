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

"""Unit tests for StalenessSampler (focusing on select_one_group)."""

from __future__ import annotations

from typing import Any

from nemo_rl.algorithms.staleness_sampler import StalenessSampler
from nemo_rl.data_plane import KVBatchMeta


def _meta_with_groups(
    groups: list[dict[str, Any]],
) -> KVBatchMeta:
    """Build a KVBatchMeta from a list of group specs.

    Each group dict has keys: group_id, weight_version, committed (default True),
    expected_num_samples, num_samples (default = expected_num_samples).
    """
    sample_ids: list[str] = []
    tags: list[dict[str, Any]] = []
    for g in groups:
        gid = g["group_id"]
        expected = g["expected_num_samples"]
        n = g.get("num_samples", expected)
        for i in range(n):
            sample_ids.append(f"{gid}_g{i}")
            tags.append(
                {
                    "group_id": gid,
                    "weight_version": g["weight_version"],
                    "committed": g.get("committed", True),
                    "expected_num_samples": expected,
                }
            )
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids,
        tags=tags,
    )


def test_select_one_group_returns_none_on_empty_meta():
    sampler = StalenessSampler(max_staleness_versions=2)
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[],
        tags=[],
    )
    assert (
        sampler.select_one_group(meta, trainer_version=5, generations_per_prompt=1)
        is None
    )


def test_select_one_group_returns_only_complete_group():
    sampler = StalenessSampler(max_staleness_versions=2)
    meta = _meta_with_groups(
        [
            {"group_id": "g0", "weight_version": 5, "expected_num_samples": 1},
        ]
    )
    assert sampler.select_one_group(
        meta, trainer_version=5, generations_per_prompt=1
    ) == [0]


def test_select_one_group_picks_lowest_lag_first():
    sampler = StalenessSampler(max_staleness_versions=3)
    meta = _meta_with_groups(
        [
            {"group_id": "g0", "weight_version": 3, "expected_num_samples": 1},
            {"group_id": "g1", "weight_version": 5, "expected_num_samples": 1},
            {"group_id": "g2", "weight_version": 4, "expected_num_samples": 1},
        ]
    )
    # trainer=5: g0 lag=2, g1 lag=0, g2 lag=1 → picks g1 (index 1)
    assert sampler.select_one_group(
        meta, trainer_version=5, generations_per_prompt=1
    ) == [1]


def test_select_one_group_tiebreak_leftmost_wins():
    sampler = StalenessSampler(max_staleness_versions=2)
    meta = _meta_with_groups(
        [
            {"group_id": "g0", "weight_version": 4, "expected_num_samples": 2},
            {"group_id": "g1", "weight_version": 4, "expected_num_samples": 2},
        ]
    )
    # Both lag=1. Tiebreak: leftmost indices[0]. g0 occupies [0,1], g1 [2,3]
    assert sampler.select_one_group(
        meta, trainer_version=5, generations_per_prompt=2
    ) == [0, 1]


def test_select_one_group_skips_incomplete_and_uncommitted():
    sampler = StalenessSampler(max_staleness_versions=2)
    meta = _meta_with_groups(
        [
            # Incomplete group: expected 2, only 1 sample
            {
                "group_id": "g0",
                "weight_version": 5,
                "expected_num_samples": 2,
                "num_samples": 1,
            },
            # Uncommitted group
            {
                "group_id": "g1",
                "weight_version": 5,
                "expected_num_samples": 1,
                "committed": False,
            },
            # Eligible group
            {"group_id": "g2", "weight_version": 5, "expected_num_samples": 1},
        ]
    )
    # g0 occupies idx 0; g1 idx 1; g2 idx 2 → picks g2
    assert sampler.select_one_group(
        meta, trainer_version=5, generations_per_prompt=1
    ) == [2]


def test_select_one_group_rejects_future_version():
    sampler = StalenessSampler(max_staleness_versions=5)
    meta = _meta_with_groups(
        [
            {"group_id": "g0", "weight_version": 6, "expected_num_samples": 1},
            {"group_id": "g1", "weight_version": 4, "expected_num_samples": 1},
        ]
    )
    # trainer=5: g0 has weight_version > trainer_version, rejected; g1 lag=1
    assert sampler.select_one_group(
        meta, trainer_version=5, generations_per_prompt=1
    ) == [1]


def test_select_one_group_strict_on_policy():
    sampler = StalenessSampler(max_staleness_versions=0)
    meta = _meta_with_groups(
        [
            {"group_id": "g0", "weight_version": 4, "expected_num_samples": 1},
            {"group_id": "g1", "weight_version": 5, "expected_num_samples": 1},
        ]
    )
    # strict: only weight_version==trainer_version eligible
    assert sampler.select_one_group(
        meta, trainer_version=5, generations_per_prompt=1
    ) == [1]
    # All stale → None
    meta_stale = _meta_with_groups(
        [
            {"group_id": "g0", "weight_version": 4, "expected_num_samples": 1},
        ]
    )
    assert (
        sampler.select_one_group(
            meta_stale, trainer_version=5, generations_per_prompt=1
        )
        is None
    )
