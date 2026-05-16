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
"""Plan §4.4 — KVBatchMeta dataclass invariants and pickle survival.

Key risk caught here: ``KVBatchMeta`` must survive ``cloudpickle`` round
trips (R-H1) — Ray uses cloudpickle for actor dispatch; if the meta
breaks in transit, every TQ-mediated dispatch raises mid-step.
"""

from __future__ import annotations

import pickle

import pytest

from nemo_rl.data_plane import KVBatchMeta


def test_size_matches_keys():
    """T1-meta-len — ``size`` is the source of truth derived from
    ``keys``; the two cannot drift."""
    meta = KVBatchMeta(
        partition_id="p",
        task_name="t",
        keys=["a", "b", "c"],
        sequence_lengths=[1, 2, 3],
    )
    assert meta.size == 3
    assert meta.size == len(meta.keys)


def test_default_fields_and_extra_info_optional():
    """``fields`` and ``sequence_lengths`` default to None;
    ``extra_info`` defaults to an empty dict."""
    meta = KVBatchMeta(partition_id="p", task_name="t", keys=[])
    assert meta.fields is None
    assert meta.sequence_lengths is None
    assert meta.extra_info == {}


def test_pickle_roundtrip_structural_equality():
    """T1-meta-cloudpickle-roundtrip — Ray actor dispatch uses
    cloudpickle. Use stdlib pickle as a strict subset; if pickle works,
    cloudpickle does too."""
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        keys=["k0", "k1", "k2"],
        fields=["input_ids", "advantages"],
        sequence_lengths=[10, 20, 30],
        extra_info={"step": 5},
    )
    rt = pickle.loads(pickle.dumps(meta))
    assert rt.partition_id == meta.partition_id
    assert rt.task_name == meta.task_name
    assert rt.keys == meta.keys
    assert rt.fields == meta.fields
    assert rt.sequence_lengths == meta.sequence_lengths
    assert rt.extra_info == meta.extra_info
    assert rt.size == meta.size


def test_keys_with_duplicates_allowed_or_warned():
    """KVBatchMeta does not enforce key uniqueness — that's the
    adapter's job (R-H2-style: dup keys at put time should fail).

    This test pins the current behavior: meta accepts any list; dupe
    detection is downstream.
    """
    meta = KVBatchMeta(partition_id="p", task_name="t", keys=["a", "a"])
    assert meta.size == 2  # no dedup at meta level


def test_empty_meta_is_valid():
    """T1-shard-empty-input — an empty meta is a valid value (e.g. a DP
    rank with no work after sharding)."""
    meta = KVBatchMeta(partition_id="p", task_name="t", keys=[])
    assert meta.size == 0
    # Cloud-pickle survives empty too.
    rt = pickle.loads(pickle.dumps(meta))
    assert rt.size == 0


def test_partition_id_is_required():
    """``partition_id`` is positional and required — plan R-M3."""
    with pytest.raises(TypeError):
        KVBatchMeta(task_name="t", keys=[])  # type: ignore[call-arg]


def test_extra_info_default_is_unique_per_instance():
    """Mutable default trap — two metas should not share the same
    ``extra_info`` dict object."""
    a = KVBatchMeta(partition_id="p", task_name="t", keys=[])
    b = KVBatchMeta(partition_id="p", task_name="t", keys=[])
    a.extra_info["x"] = 1
    assert "x" not in b.extra_info


def test_tags_align_with_keys():
    """``tags`` must be exactly one dict per key, or ``None``."""
    KVBatchMeta(
        partition_id="p", task_name="t", keys=["a", "b"], tags=[{"x": 1}, {"x": 2}]
    )
    with pytest.raises(ValueError, match=r"align 1:1"):
        KVBatchMeta(
            partition_id="p", task_name="t", keys=["a", "b"], tags=[{"x": 1}]
        )


def test_tags_travel_with_subset_slice_concat():
    """Per-key tags must follow keys through ``subset`` / ``slice`` /
    ``concat`` so consumers can filter on tags without fetching data."""
    m = KVBatchMeta(
        partition_id="p",
        task_name="t",
        keys=["a", "b", "c", "d"],
        sequence_lengths=[1, 2, 3, 4],
        tags=[{"std": 0.1}, {"std": 0.0}, {"std": 0.3}, {"std": 0.0}],
    )

    survivors = m.subset([0, 2])
    assert survivors.keys == ["a", "c"]
    assert survivors.tags == [{"std": 0.1}, {"std": 0.3}]
    assert survivors.sequence_lengths == [1, 3]

    front = m.slice(0, 2)
    assert front.tags == [{"std": 0.1}, {"std": 0.0}]

    joined = front.concat(m.slice(2, 4))
    assert joined.keys == m.keys
    assert joined.tags == m.tags


def test_tags_none_when_either_side_missing_in_concat():
    """``concat`` drops tags if either side has none — symmetric with
    the ``sequence_lengths`` behavior."""
    with_tags = KVBatchMeta(
        partition_id="p", task_name="t", keys=["a"], tags=[{"x": 1}]
    )
    without = KVBatchMeta(partition_id="p", task_name="t", keys=["b"])
    assert with_tags.concat(without).tags is None
