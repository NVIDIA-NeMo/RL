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
"""Stage 4 unit tests — sharding helper + minimal codec."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import (
    KVBatchMeta,
    materialize,
    shard_keys_by_seqlen,
)


def test_shard_partitions_keys_disjointly():
    meta = KVBatchMeta(
        partition_id="p",
        task_name="train",
        keys=[f"k{i}" for i in range(8)],
        sequence_lengths=[10, 90, 20, 80, 30, 70, 40, 60],
    )
    shards = shard_keys_by_seqlen(meta, dp_world_size=4)

    assert len(shards) == 4
    flat = sorted(k for s in shards for k in s.keys)
    assert flat == sorted(meta.keys)


def test_shard_balances_total_seqlen():
    """Sort+stride should keep per-rank token counts within ~max_seqlen."""
    seqlens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    meta = KVBatchMeta(
        partition_id="p",
        task_name="train",
        keys=[f"k{i}" for i in range(len(seqlens))],
        sequence_lengths=seqlens,
    )
    shards = shard_keys_by_seqlen(meta, dp_world_size=3)
    totals = [sum(s.sequence_lengths) for s in shards]
    assert max(totals) - min(totals) <= max(seqlens)


def test_shard_requires_seqlens():
    meta = KVBatchMeta(
        partition_id="p", task_name="train", keys=["a", "b"], sequence_lengths=None
    )
    with pytest.raises(ValueError):
        shard_keys_by_seqlen(meta, dp_world_size=2)


def test_shard_rejects_zero_world_size():
    meta = KVBatchMeta(
        partition_id="p",
        task_name="train",
        keys=["a"],
        sequence_lengths=[1],
    )
    with pytest.raises(ValueError):
        shard_keys_by_seqlen(meta, dp_world_size=0)


def test_materialize_padded_passthrough():
    td = TensorDict(
        {
            "input_ids": torch.arange(8).reshape(4, 2),
            "advantages": torch.zeros(4),
        },
        batch_size=[4],
    )
    bd = materialize(td, layout="padded")
    assert torch.equal(bd["input_ids"], torch.arange(8).reshape(4, 2))
    assert torch.equal(bd["advantages"], torch.zeros(4))


def test_materialize_jagged_unsupported():
    td = TensorDict({"x": torch.arange(4)}, batch_size=[4])
    with pytest.raises(NotImplementedError):
        materialize(td, layout="jagged")
