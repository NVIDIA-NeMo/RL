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
"""Unit tests for the dp_dispatch decorator's polymorphic dispatch."""

from __future__ import annotations

from typing import Any

import pytest

from nemo_rl.data_plane import KVBatchMeta, dp_dispatch


class _FakeAxes:
    def __init__(self, dp_size: int = 2):
        self._dp = dp_size

    def get_axis_size(self, name: str) -> int:
        return self._dp if name == "data_parallel" else 1


class _FakeWorkerGroup:
    def __init__(self):
        self.calls: list[dict] = []

    def run_all_workers_sharded_data(self, method_name: str, **kwargs):
        self.calls.append({"method_name": method_name, **kwargs})
        return f"futures-for-{method_name}"

    def get_all_worker_results(self, futures):
        # Pretend two DP ranks each returned a tag carrying their shard size.
        return [{"shard_size": 2}, {"shard_size": 2}]


class _FakePolicy:
    def __init__(self, dp_size: int = 2):
        self.sharding_annotations = _FakeAxes(dp_size)
        self.worker_group = _FakeWorkerGroup()
        self.legacy_calls: list[Any] = []

    @dp_dispatch(
        sharder=lambda meta, dp: [
            KVBatchMeta(
                partition_id=meta.partition_id,
                task_name=meta.task_name,
                keys=meta.keys[r::dp],
                fields=meta.fields,
                sequence_lengths=(
                    meta.sequence_lengths[r::dp]
                    if meta.sequence_lengths is not None
                    else None
                ),
            )
            for r in range(dp)
        ],
        sharded_axes=["data_parallel"],
        replicate_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
        worker_method="train_presharded",
        aggregate=lambda results: {"total_shards": sum(r["shard_size"] for r in results)},
    )
    def train(self, data, *, loss_fn=None):
        # Legacy in-memory path — only reached for non-meta inputs.
        self.legacy_calls.append((data, loss_fn))
        return {"legacy": True, "data": data}


def test_legacy_passthrough_for_non_meta():
    policy = _FakePolicy()
    out = policy.train({"some": "data"}, loss_fn="loss")
    assert out == {"legacy": True, "data": {"some": "data"}}
    assert policy.legacy_calls == [({"some": "data"}, "loss")]
    assert policy.worker_group.calls == []


def test_meta_input_routes_to_worker_method():
    policy = _FakePolicy(dp_size=2)
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        keys=["a", "b", "c", "d"],
        fields=["x"],
        sequence_lengths=[10, 20, 30, 40],
    )
    out = policy.train(meta, loss_fn="loss")

    # Aggregator was applied.
    assert out == {"total_shards": 4}
    # Legacy body was NOT called.
    assert policy.legacy_calls == []

    # Dispatch happened with the right method + axis annotations.
    assert len(policy.worker_group.calls) == 1
    call = policy.worker_group.calls[0]
    assert call["method_name"] == "train_presharded"
    assert call["in_sharded_axes"] == ["data_parallel"]
    assert call["replicate_on_axes"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    # Per-rank shards: 2 metas, each with 2 keys (4 keys / dp_size=2).
    shards = call["meta"]
    assert len(shards) == 2
    assert all(isinstance(s, KVBatchMeta) for s in shards)
    assert sum(s.size for s in shards) == 4
    # Loss-fn travelled via common_kwargs, not in worker meta.
    assert call["common_kwargs"] == {"loss_fn": "loss"}


def test_dispatch_introspection_attribute():
    policy = _FakePolicy()
    assert hasattr(policy.train, "__dp_dispatch__")
    info = policy.train.__dp_dispatch__
    assert info["worker_method"] == "train_presharded"
    assert info["sharded_axes"] == ("data_parallel",)


def test_pre_sharded_meta_list_skips_sharder():
    policy = _FakePolicy(dp_size=2)
    pre_shards = [
        KVBatchMeta(
            partition_id="train",
            task_name="train",
            keys=[f"r0_s{i}" for i in range(3)],
            fields=["x"],
            sequence_lengths=[10, 20, 30],
            extra_info={"micro_batch_indices": [[[0, 1], [1, 3]]]},
        ),
        KVBatchMeta(
            partition_id="train",
            task_name="train",
            keys=[f"r1_s{i}" for i in range(3)],
            fields=["x"],
            sequence_lengths=[15, 25, 35],
            extra_info={"micro_batch_indices": [[[0, 2], [2, 3]]]},
        ),
    ]
    out = policy.train(pre_shards, loss_fn="loss")

    assert policy.legacy_calls == []
    assert len(policy.worker_group.calls) == 1
    call = policy.worker_group.calls[0]
    # Pre-sharded list was forwarded verbatim — sharder NOT invoked, so the
    # extra_info packing metadata each rank needs is preserved.
    assert call["meta"] is pre_shards
    assert call["meta"][0].extra_info == {"micro_batch_indices": [[[0, 1], [1, 3]]]}
    assert out == {"total_shards": 4}


def test_pre_sharded_meta_list_size_mismatch_raises():
    policy = _FakePolicy(dp_size=2)
    too_few = [
        KVBatchMeta(partition_id="train", task_name="train", keys=["a"], fields=["x"]),
    ]
    with pytest.raises(ValueError, match="DP world size"):
        policy.train(too_few, loss_fn="loss")
